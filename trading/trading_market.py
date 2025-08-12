# Market and trade client that hides an asyncio loop behind a synchronous API.
# This refactor preserves logic/behavior while consolidating repeated structures.

from __future__ import annotations

import asyncio
import contextlib
import json
import threading
import time
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Literal, Tuple, Callable

import aiohttp
import websockets
from websockets.legacy.client import WebSocketClientProtocol
import numpy as np

# Project-local imports
from .trading_config import TradingConfig
from .asset_utils import COUNTRY_MAP, EXCHANGE_MAP, ASSET_TYPE_MAP

TradeMode = Literal["local", "paper", "real"]


# ------------------------------- Async token bucket ------------------------------- #
class _TokenBucket:
    def __init__(self, capacity: int, rps: float):
        self.capacity = float(capacity)
        self.tokens = float(capacity)
        self.rps = float(rps)
        self._last = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self, cost: float = 1.0) -> None:
        while True:
            async with self._lock:
                now = time.time()
                dt = now - self._last
                if dt > 0:
                    self.tokens = min(self.capacity, self.tokens + dt * self.rps)
                    self._last = now
                if self.tokens >= cost:
                    self.tokens -= cost
                    return
            await asyncio.sleep(0.01)


@dataclass
class _Channel:
    name: str
    url_attr: str              # config attribute name for WS URL
    ws_attr: str               # instance attribute for WebSocket
    task_attr: str             # instance attribute for loop task
    parser: Callable[..., Any] # parser function
    kind: Literal["market", "trades"]


class TradingMarket:
    """
    Present a synchronous-style API while performing all network I/O on a
    dedicated asyncio event loop running in a daemon thread.

    Required fields for :class:`TradingVecEnv`:
      - ``trade_mode``: ``"local"``, ``"paper"``, or ``"real"``
      - ``init_symbols``: initial subscription list
      - ``country_id``, ``exchange_id``, ``asset_type``: integer identifiers used
        in the ``asset_id`` observation feature

    Methods expected by :class:`TradingVecEnv`:
      - ``get_market_snapshot(symbols)`` → ``{sym: {"o","h","l","c","v","t"}}``
      - ``get_account_snapshot(symbols)`` → per-symbol account state
      - ``submit_orders(symbols, sides, qtys, trade_mode)`` → list of order
        results (local mode requires ``{"filled_avg_price", "action"}``)
      - ``step_account(order_results)`` → ``(reward, truncated, terminated)``
      - ``close()`` to release all resources
    """

    # -------------------------------- ctor / lifecycle -------------------------------- #
    def __init__(self, *, trading_config: TradingConfig, symbols: Optional[List[str]] = None):
        self.cfg = trading_config
        self.trade_mode: TradeMode = trading_config.trade_mode
        self.init_symbols: List[str] = list(dict.fromkeys(symbols or []))
        self.freeze_subscriptions = (self.trade_mode != "local")
        self.logger = logging.getLogger(__name__)

        # IDs used in observation asset_id
        self.country_id = COUNTRY_MAP.get(self.cfg.country_code, 1)
        self.exchange_id = EXCHANGE_MAP.get(self.country_id, {}).get(self.cfg.exchange_code, 1)
        self.asset_type = ASSET_TYPE_MAP.get(self.cfg.asset_type, 1)

        # Shared caches protected by ``_cache_lock``
        self._cache_lock = threading.Lock()
        self._market_cache: Dict[str, Dict[str, Any]] = {}   # symbol → latest bar
        self._account_cache: Dict[str, Any] = {}             # account fields & NAV
        self._orders_cache: Dict[str, Dict[str, Any]] = {}   # order_id → payload
        self._subscribed: set[str] = set()

        # Rate limiting
        self._rl_rest = _TokenBucket(self.cfg.rest_burst, self.cfg.rest_rps)
        self._rl_ws = _TokenBucket(self.cfg.ws_pull_burst, self.cfg.ws_pull_rps)

        # Async resources (only accessed from the loop thread)
        self._session: Optional[aiohttp.ClientSession] = None
        self._market_ws: Optional[WebSocketClientProtocol] = None
        self._trades_ws: Optional[WebSocketClientProtocol] = None
        self._market_task: Optional[asyncio.Task] = None
        self._trades_task: Optional[asyncio.Task] = None
        self._stop_evt: Optional[asyncio.Event] = None

        # Private event loop in a daemon thread
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, name="AlpacaLoop", daemon=True)
        self._thread.start()

        # Start streaming and perform an initial snapshot backfill
        self.connect_market()                 # WebSocket connect and subscribe
        try:
            # Populate cache immediately; works even when markets are closed
            self._backfill_from_snapshot(self.init_symbols)
        except Exception:
            pass
        # Order/account updates stream only for paper or real trading
        if self.trade_mode in ("paper", "real"):
            self.connect_trades()

    # -------------------------------- internals -------------------------------- #
    def _channel_defs(self) -> List[_Channel]:
        return [
            _Channel("market", "market_ws_url", "_market_ws", "_market_task", self._parse_market_message, "market"),
            _Channel("trades", "trades_ws_url", "_trades_ws", "_trades_task", self._parse_trade_message, "trades"),
        ]

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_forever()
        finally:
            pending = asyncio.all_tasks(self._loop)
            for t in pending:
                t.cancel()
            if pending:
                self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            self._loop.run_until_complete(self._loop.shutdown_asyncgens())
            self._loop.close()

    def _submit(self, coro, *, timeout: Optional[float] = None):
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result(timeout=timeout)

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession(headers={
                "APCA-API-KEY-ID": self.cfg.api_key,
                "APCA-API-SECRET-KEY": self.cfg.secret_key,
                "Content-Type": "application/json",
            })
        return self._session

    # ------------------------------- Endpoints ------------------------------- #
    @property
    def rest_base(self) -> str:
        """Trading REST base (accounts, orders)."""
        return self.cfg.paper_rest_base if self.trade_mode == "paper" else self.cfg.live_rest_base

    @property
    def data_rest_base(self) -> str:
        """
        Data REST base (bars). If config provides one, use it.
        Otherwise derive a sane default from broker + asset_type.
        """
        if getattr(self.cfg, "data_rest_base", None):
            return self.cfg.data_rest_base.rstrip("/")

        b = (self.cfg.broker or "").lower()
        a = (self.cfg.asset_type or "").lower()

        if b == "alpaca":
            # Crypto uses v1beta3; stocks use v2
            return ("https://data.alpaca.markets/v1beta3/crypto/us"
                    if a.startswith("crypto") else
                    "https://data.alpaca.markets/v2/stocks")
        if b == "binance":
            return "https://api.binance.com/api"
        # Fallback to trading REST base if nothing else
        return (self.cfg.paper_rest_base if self.trade_mode == "paper" else self.cfg.live_rest_base).rstrip("/")
    
    @property
    def bars_timeframe(self) -> str:
        """Timeframe for backfill bars (e.g., '1Min')."""
        return getattr(self.cfg, "bars_timeframe", "1Min")

    # ------------------------------- Lifecycle (sync wrappers) ------------------------------- #
    def connect_market(self) -> None:
        self._submit(self._connect_channel_async(self._channel_defs()[0]))

    def connect_trades(self) -> None:
        self._submit(self._connect_channel_async(self._channel_defs()[1]))

    def close_market(self) -> None:
        self._submit(self._close_channel_async(self._channel_defs()[0]))

    def close_trades(self) -> None:
        self._submit(self._close_channel_async(self._channel_defs()[1]))

    def close(self) -> None:
        self._submit(self._close_all_async())
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def exit(self):
        """Gracefully shut down all async tasks and the loop."""
        try:
            if self._market_task and not self._market_task.done():
                self._market_task.cancel()
            if self._trades_task and not self._trades_task.done():
                self._trades_task.cancel()

            if self._market_ws and not self._market_ws.closed:
                self._loop.call_soon_threadsafe(lambda: asyncio.create_task(self._market_ws.close()))
            if self._trades_ws and not self._trades_ws.closed:
                self._loop.call_soon_threadsafe(lambda: asyncio.create_task(self._trades_ws.close()))

            if self._session and not self._session.closed:
                self._loop.call_soon_threadsafe(lambda: asyncio.create_task(self._session.close()))

            if self._loop.is_running():
                self._loop.call_soon_threadsafe(self._loop.stop)

            if self._thread.is_alive():
                self._thread.join(timeout=5)
        except Exception as e:
            print(f"[AlpacaMarket.exit] Error during cleanup: {e}")

    def __del__(self):
        self.exit()

    # ------------------------------- Generic channel plumbing ------------------------------- #
    async def _connect_channel_async(self, ch: _Channel) -> None:
        await self._ensure_session()
        self._stop_evt = self._stop_evt or asyncio.Event()
        ws = await websockets.connect(getattr(self.cfg, ch.url_attr))
        await self._send_ws(ws, {"action": "auth", "key": self.cfg.api_key, "secret": self.cfg.secret_key})
        _ = await self._recv_ws(ws, self.cfg.recv_timeout_sec)
        setattr(self, ch.ws_attr, ws)
        # market channel subscribes initial symbols
        if ch.kind == "market" and self.init_symbols:
            await self._subscribe_async(self.init_symbols)
            
        # trades channel may require an explicit subscribe for order/account streams (broker-specific)
        if ch.kind == "trades":
            broker = getattr(self.cfg, "broker", "").lower()
            # Alpaca compatibility: try modern and legacy payloads
            if broker == "alpaca":
                # v2-style subscribe (no-op if unsupported)
                await self._send_ws(ws, {"action": "subscribe", "orders": ["*"], "account": ["*"]})
                # legacy listen API (paper stream)
                await self._send_ws(ws, {"action": "listen", "data": {"streams": ["trade_updates", "account_updates"]}})
            
        task = asyncio.create_task(self._channel_loop(ch))
        setattr(self, ch.task_attr, task)

    async def _close_channel_async(self, ch: _Channel) -> None:
        task = getattr(self, ch.task_attr)
        ws = getattr(self, ch.ws_attr)
        if task and not task.done():
            task.cancel()
            with contextlib.suppress(Exception):
                await task
        if ws:
            with contextlib.suppress(Exception):
                await ws.close()
        setattr(self, ch.task_attr, None)
        setattr(self, ch.ws_attr, None)

    async def _close_all_async(self) -> None:
        if self._stop_evt:
            self._stop_evt.set()
        # Close trades then market (mirrors original ordering)
        await self._close_channel_async(self._channel_defs()[1])
        await self._close_channel_async(self._channel_defs()[0])
        if self._session:
            with contextlib.suppress(Exception):
                await self._session.close()
        self._session = None

    async def _channel_loop(self, ch: _Channel) -> None:
        ws: Optional[WebSocketClientProtocol] = getattr(self, ch.ws_attr)
        if not ws:
            return
        while self._stop_evt and not self._stop_evt.is_set():
            await self._rl_ws.acquire()
            raw = await self._recv_ws(ws, self.cfg.recv_timeout_sec)
            if not raw:
                continue
            try:
                msg = json.loads(raw) if isinstance(raw, (bytes, bytearray, str)) else raw
                if ch.kind == "market":
                    pairs = ch.parser(msg)  # list[(symbol, payload)]
                    if not pairs:
                        continue
                    with self._cache_lock:
                        for sym, payload in pairs:
                            self._market_cache[sym] = payload
                else:
                    kind, payload = ch.parser(msg)
                    with self._cache_lock:
                        if kind == "order" and payload:
                            oid = str(payload.get("id", f"order-{int(time.time()*1e6)}"))
                            self._orders_cache[oid] = payload
                        elif kind == "account" and payload:
                            self._account_cache.update(payload)
            except Exception:
                continue

    # ------------------------------- WS helpers ------------------------------- #
    async def _send_ws(self, ws: WebSocketClientProtocol, obj: Dict[str, Any]) -> None:
        try:
            await ws.send(json.dumps(obj))
        except Exception:
            pass

    async def _recv_ws(self, ws: WebSocketClientProtocol, timeout: float):
        try:
            return await asyncio.wait_for(ws.recv(), timeout=timeout)
        except Exception:
            return None

    # ------------------------------- subscribe (sync wrappers) ------------------------------- #
    def subscribe(self, symbols: List[str]) -> None:
        if getattr(self, "freeze_subscriptions", False):
            for s in symbols:
                try:
                    self.logger.info(f"[freeze] subscribe ignored: {s}")
                except Exception:
                    pass
            return False
        self._submit(self._subscribe_async(symbols))

    def unsubscribe(self, symbols: List[str]) -> None:
        if getattr(self, "freeze_subscriptions", False):
            for s in symbols:
                try:
                    self.logger.info(f"[freeze] unsubscribe ignored: {s}")
                except Exception:
                    pass
            return False
        self._submit(self._unsubscribe_async(symbols))

    async def _subscribe_async(self, symbols: List[str]) -> None:
        add = [s for s in symbols if s not in self._subscribed]
        if not add:
            return
        self._subscribed.update(add)
        if self._market_ws:
            await self._send_ws(self._market_ws, {"action": "subscribe", "bars": add})

    async def _unsubscribe_async(self, symbols: List[str]) -> None:
        rm = [s for s in symbols if s in self._subscribed]
        if not rm:
            return
        for s in rm:
            self._subscribed.discard(s)
        if self._market_ws:
            await self._send_ws(self._market_ws, {"action": "unsubscribe", "bars": rm})

    # ------------------------------- REST helpers ------------------------------- #
    async def _rest_get_json(self, base: str, path: str, *, timeout: float = 5.0) -> Tuple[int, Any]:
        await self._rl_rest.acquire()
        sess = await self._ensure_session()
        url = f"{base.rstrip('/')}/{path.lstrip('/')}"
        try:
            async with sess.get(url, timeout=timeout) as r:
                data = await (r.json() if r.status // 100 == 2 else asyncio.sleep(0))
                return r.status, (data if r.status // 100 == 2 else None)
        except Exception:
            return 0, None

    async def _rest_post_json(self, base: str, path: str, payload: dict, *, timeout: float = 5.0) -> Tuple[int, Any]:
        await self._rl_rest.acquire()
        sess = await self._ensure_session()
        url = f"{base.rstrip('/')}/{path.lstrip('/')}"
        try:
            async with sess.post(url, data=json.dumps(payload), timeout=timeout) as r:
                data = await (r.json() if r.status // 100 == 2 else asyncio.sleep(0))
                return r.status, (data if r.status // 100 == 2 else None)
        except Exception:
            return 0, None

    async def _rest_delete(self, base: str, path: str, *, timeout: float = 5.0) -> int:
        await self._rl_rest.acquire()
        sess = await self._ensure_session()
        url = f"{base.rstrip('/')}/{path.lstrip('/')}"
        try:
            async with sess.delete(url, timeout=timeout) as r:
                return r.status
        except Exception:
            return 0

    # ------------------------------- REST backfill helpers ------------------------------- #
    def _backfill_from_snapshot(self, symbols: List[str]) -> int:
        """Fetch latest bars via REST and update the market cache."""
        if not symbols:
            return 0
        snap = self._submit(self._rest_get_latest_bars(symbols, timeout_sec=2.0))
        return self._update_cache_from_snapshot(snap)

    def _norm_bar(self, d: dict, default_t: float) -> dict:
        return {
            "o": float(d.get("o", 0.0)),
            "h": float(d.get("h", 0.0)),
            "l": float(d.get("l", 0.0)),
            "c": float(d.get("c", 0.0) or d.get("price", 0.0)),
            "v": float(d.get("v", 0.0)),
            "t": d.get("t", default_t),
        }

    def _update_cache_from_snapshot(self, snap_dict: Dict[str, Dict[str, Any]]) -> int:
        """Update the market cache with a snapshot of bar data."""
        if not snap_dict:
            return 0
        t_val = (time.time() / 86400.0)
        updated = 0
        with self._cache_lock:
            for sym, bar in snap_dict.items():
                self._market_cache[sym] = self._norm_bar(bar, t_val)
                updated += 1
        return updated
    
    async def _rest_get_latest_bars(self, symbols: List[str], *, timeout_sec: float = 1.0) -> Dict[str, Dict[str, Any]]:
        """Dispatcher to broker/asset-specific bar fetchers."""
        broker = (self.cfg.broker or "").lower()
        is_crypto = (self.cfg.asset_type or "").lower().startswith("crypto")
        if broker == "alpaca" and not is_crypto:
            return await self._rest_get_latest_bars_alpaca_stocks(symbols, timeout_sec=timeout_sec)
        if broker == "alpaca" and is_crypto:
            return await self._rest_get_latest_bars_alpaca_crypto(symbols, timeout_sec=timeout_sec)
        if broker == "binance":
            return await self._rest_get_latest_bars_binance(symbols, timeout_sec=timeout_sec)
        # Generic fallback (treat like Alpaca stocks path shape)
        return await self._rest_get_latest_bars_generic(symbols, timeout_sec=timeout_sec)

    # ------------------------------- per-broker helpers ------------------------------- #
    async def _rest_get_latest_bars_alpaca_stocks(self, symbols: List[str], *, timeout_sec: float) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        await self._ensure_session()
        timeframe = self.bars_timeframe
        base = self.data_rest_base  # .../v2/stocks
        sess = await self._ensure_session()
        for sym in symbols:
            try:
                await self._rl_rest.acquire()
                url = f"{base}/{sym}/bars?timeframe={timeframe}&limit=1"
                async with sess.get(url, timeout=timeout_sec) as r:
                    if r.status // 100 != 2:
                        continue
                    data = await r.json()
                bars = isinstance(data, dict) and (data.get("bars") or data.get("bar"))
                bar = (bars[-1] if isinstance(bars, list) and bars
                       else bars if isinstance(bars, dict) else None)
                if isinstance(bar, dict):
                    out[sym] = self._norm_bar(bar, time.time() / 86400.0)
            except Exception:
                continue
        return out

    async def _rest_get_latest_bars_alpaca_crypto(self, symbols: List[str], *, timeout_sec: float) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        await self._ensure_session()
        timeframe = self.bars_timeframe
        base = self.data_rest_base  # .../v1beta3/crypto/us
        sess = await self._ensure_session()
        for sym in symbols:
            try:
                await self._rl_rest.acquire()
                url = f"{base}/bars?symbols={sym}&timeframe={timeframe}&limit=1"
                async with sess.get(url, timeout=timeout_sec) as r:
                    if r.status // 100 != 2:
                        continue
                    data = await r.json()
                bars_by_sym = (isinstance(data, dict) and data.get("bars")) or {}
                seq = (isinstance(bars_by_sym, dict) and bars_by_sym.get(sym)) or []
                bar = seq[-1] if isinstance(seq, list) and seq else None
                if isinstance(bar, dict):
                    # normalize potential timestamp key variants
                    b = {"o": bar.get("o"), "h": bar.get("h"), "l": bar.get("l"),
                         "c": bar.get("c"), "v": bar.get("v"), "t": bar.get("t") or bar.get("timestamp")}
                    out[sym] = self._norm_bar(b, time.time() / 86400.0)
            except Exception:
                continue
        return out

    async def _rest_get_latest_bars_binance(self, symbols: List[str], *, timeout_sec: float) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        await self._ensure_session()
        base = self.data_rest_base  # https://api.binance.com/api
        sess = await self._ensure_session()
        # Map common tf to Binance intervals
        tf = self.bars_timeframe.lower()
        interval = {"1min": "1m", "1m": "1m", "5min": "5m", "5m": "5m"}.get(tf, "1m")
        for sym in symbols:
            try:
                await self._rl_rest.acquire()
                url = f"{base}/v3/klines?symbol={sym}&interval={interval}&limit=1"
                async with sess.get(url, timeout=timeout_sec) as r:
                    if r.status // 100 != 2:
                        continue
                    data = await r.json()
                if isinstance(data, list) and data:
                    k = data[-1]  # [ openTime, o, h, l, c, v, closeTime, ... ]
                    bar = {"o": float(k[1]), "h": float(k[2]), "l": float(k[3]),
                           "c": float(k[4]), "v": float(k[5]), "t": float(k[6]) / 86400_000.0}
                    out[sym] = self._norm_bar(bar, time.time() / 86400.0)
            except Exception:
                continue
        return out

    async def _rest_get_latest_bars_generic(self, symbols: List[str], *, timeout_sec: float) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        await self._ensure_session()
        timeframe = self.bars_timeframe
        base = self.data_rest_base
        sess = await self._ensure_session()
        for sym in symbols:
            try:
                await self._rl_rest.acquire()
                url = f"{base}/{sym}/bars?timeframe={timeframe}&limit=1"
                async with sess.get(url, timeout=timeout_sec) as r:
                    if r.status // 100 != 2:
                        continue
                    data = await r.json()
                bars = isinstance(data, dict) and (data.get("bars") or data.get("bar"))
                bar = (bars[-1] if isinstance(bars, list) and bars
                       else bars if isinstance(bars, dict) else None)
                if isinstance(bar, dict):
                    out[sym] = self._norm_bar(bar, time.time() / 86400.0)
            except Exception:
                continue
        return out

    # --------------------- local-mode episode reset (subscriptions/cache) --------------------- #
    def reset_subscriptions(self, symbols: list[str]) -> None:
        """
        Best-effort refresh for local mode: (un)subscribe to the provided symbols
        and clear per-episode account deltas. Safe to call in any mode.
        """
        if self.freeze_subscriptions:
            return
        try:
            symbols = list(dict.fromkeys(symbols or []))
            old = list(self._subscribed)
            # Unsubscribe symbols that are no longer used
            rm = [s for s in old if s not in symbols]
            if rm:
                self.unsubscribe(rm)
            # Subscribe any new symbols
            add = [s for s in symbols if s not in self._subscribed]
            if add:
                self.subscribe(add)
            # Reset previous NAV vectors used in paper/real (harmless in local)
            with self._cache_lock:
                self._account_cache.pop("_nav_prev_vec", None)
                self._account_cache.pop("_nav_prev_by_sym", None)
        except Exception:
            pass

    # ------------------------------- snapshot helpers ------------------------------- #
    def _ensure_bars(self, symbols: np.ndarray, timeout_sec: float = 1.0) -> None:
        """Ensure WS subscription, wait for cache fill, and backfill zeros once."""
        import numpy as np, time

        # Normalize symbols: ndarray -> list[str], preserve order, deduplicate
        if isinstance(symbols, np.ndarray):
            syms = symbols.tolist()
        else:
            syms = list(symbols)
        syms = [str(s) for s in syms]
        syms = list(dict.fromkeys(syms))
        if not syms:
            return

        deadline = time.time() + float(timeout_sec)

        # Subscribe any missing symbols (when subscriptions are not frozen)
        if not self.freeze_subscriptions:
            miss = [s for s in syms if s not in self._subscribed]
            if miss:
                try:
                    self.subscribe(miss)
                except Exception:
                    pass

        # Wait briefly until the cache has entries for subscribed symbols
        need = {s for s in syms if s in self._subscribed}
        while time.time() < deadline and need:
            with self._cache_lock:
                have = {s for s in need if s in self._market_cache}
            if have == need:
                break
            time.sleep(0.01)

        # If any closing price is zero, try a one-shot REST backfill
        zeros = []
        with self._cache_lock:
            for s in syms:
                bar = self._market_cache.get(s) or {}
                c = float(bar.get("c") or bar.get("price") or 0.0)
                if c == 0.0:
                    zeros.append(s)
        if zeros:
            try:
                self._backfill_from_snapshot(zeros)
            except Exception:
                pass


    # ------------------------------- public snapshots ------------------------------- #
    def get_cached_bars(self, symbols: np.ndarray) -> np.ndarray:
        """
        Return cached OHLCV data for each symbol as an (N, 6) float32 array
        ordered [o, h, l, c, v, t], preserving the original order and duplicates.

        Strategy (thread-safe against the cache lock):
        1) Read from cache and fill an (N,6) array.
        2) If any row has close==0.0 or cache-miss, backfill ONLY the needed unique symbols.
        3) Re-read from cache and return the fully aligned (N,6).
        """
        import time
        import numpy as np

        # EN: Normalize input to a Python list while preserving order & duplicates
        if isinstance(symbols, np.ndarray):
            syms = symbols.tolist()
        else:
            syms = list(symbols)

        n = len(syms)
        t_now = float(time.time() / 86400.0)
        out = np.zeros((n, 6), dtype=np.float32)  # [o,h,l,c,v,t]

        # EN: Helper to copy one bar dict -> row tuple with robust defaults
        def _bar_to_row(bar: dict) -> tuple:
            if not bar:
                return (0.0, 0.0, 0.0, 0.0, 0.0, t_now)
            o = float(bar.get("o", 0.0))
            h = float(bar.get("h", 0.0))
            l = float(bar.get("l", 0.0))
            c = float(bar.get("c", bar.get("price", 0.0)) or 0.0)
            v = float(bar.get("v", 0.0))
            t = bar.get("t", t_now)
            try:
                t = float(t)
            except Exception:
                t = t_now
            return (o, h, l, c, v, t)

        # EN: 1) Initial fill from cache
        missing_or_zero = set()  # EN: collect symbols that need backfill
        with self._cache_lock:
            for i, s in enumerate(syms):
                bar = dict(self._market_cache.get(s) or {})
                row = _bar_to_row(bar)
                out[i, :] = row
                if row[3] == 0.0:  # close==0.0
                    missing_or_zero.add(s)

        # EN: 2) If anything is missing/zero, backfill only the required unique symbols
        if missing_or_zero:
            # EN: keep original encounter order while deduping minimal set
            unique_needed = []
            seen = set()
            for s in syms:
                if s in missing_or_zero and s not in seen:
                    unique_needed.append(s)
                    seen.add(s)
            try:
                # EN: This populates the internal cache; network-safe in local/paper/real
                self._backfill_from_snapshot(unique_needed)
            except Exception:
                # EN: best effort; fall through
                pass

            # EN: 3) Re-read from cache to rebuild the aligned array
            with self._cache_lock:
                for i, s in enumerate(syms):
                    bar = dict(self._market_cache.get(s) or {})
                    out[i, :] = _bar_to_row(bar)

        return out


    def get_market_features(self, symbols: np.ndarray, timeout_sec: float = 1.0) -> np.ndarray:
        """Return (N,6) array [o,h,l,c,v,t] for the requested symbols."""
        self._ensure_bars(symbols, timeout_sec)
        return self.get_cached_bars(symbols)

    # ------------------------------- account snapshot ------------------------------- #
    def get_account_features(self, symbols: np.ndarray) -> np.ndarray:
        """
        Return (N,6) float32 array aligned with `symbols`.
        Columns match TradingVecEnv.ACCOUNT_FEATURE_KEYS order:
        [position_qty, cash, avg_entry_price, unrealized_pnl, exposure, asset_nav]
        """
        import numpy as np
        # --- fetch account & positions (unchanged networking) ---
        account, positions = self._submit(self._get_account_and_positions())

        # --- index positions by symbol for quick lookup ---
        pos_by_sym = {}
        for p in positions or []:
            s = str(p.get("symbol") or "")
            if not s:
                continue
            qty = float(p.get("qty") or p.get("quantity") or 0.0)
            avg = float(p.get("avg_entry_price") or p.get("avg_price") or 0.0)
            pos_by_sym[s] = {"qty": qty, "avg": avg}

        # --- latest prices from market cache (aligned to input symbols) ---
        syms = symbols.tolist() if isinstance(symbols, np.ndarray) else list(symbols)
        with self._cache_lock:
            px_by_sym = {s: float((self._market_cache.get(s, {}) or {}).get("c")
                                or (self._market_cache.get(s, {}) or {}).get("price")
                                or 0.0) for s in syms}

        # --- distribute cash equally per slot (same behavior as before) ---
        n = max(1, len(syms))
        cash_share = float(account.get("cash", 0.0)) / float(n)

        # --- build (N,6) ndarray in the canonical feature order ---
        out = np.zeros((n, 6), dtype=np.float32)
        for i, s in enumerate(syms):
            qty = float(pos_by_sym.get(s, {}).get("qty", 0.0))
            avg = float(pos_by_sym.get(s, {}).get("avg", 0.0))
            px  = float(px_by_sym.get(s, 0.0))
            exposure = qty * px
            unreal   = (px - avg) * qty
            nav_i    = cash_share + exposure

            # [position_qty, cash, avg_entry_price, unrealized_pnl, exposure, asset_nav]
            out[i, 0] = qty
            out[i, 1] = cash_share
            out[i, 2] = avg
            out[i, 3] = unreal
            out[i, 4] = exposure
            out[i, 5] = nav_i
        return out

    async def _get_account_and_positions(self) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        # Mirrors original logic with small helper usage
        status_a, account = await self._rest_get_json(self.rest_base, "/account", timeout=5)
        status_p, positions = await self._rest_get_json(self.rest_base, "/positions", timeout=5)
        account = account or {}
        positions = positions or []
        with self._cache_lock:
            self._account_cache.update(account or {})
        return account, positions

    # ------------------------------- orders ------------------------------- #
    def submit_orders(self, symbols: np.ndarray, sides: np.ndarray, qtys: np.ndarray, trade_mode: str) -> np.ndarray:
        assert len(symbols) == len(sides) == len(qtys), "length mismatch"
        mode = trade_mode or self.trade_mode

        results: List[Optional[dict]] = [None] * len(symbols)

        # Auto-subscribe only in local mode
        miss = [s for s in symbols if s not in self._subscribed]
        if miss and mode == "local":
            try:
                self.subscribe(miss)
            except Exception:
                pass

        uniq_syms: List[str] = []
        uniq_sides: List[str] = []
        uniq_qtys: List[int] = []
        first_idx: Dict[str, int] = {}
        for i, (s, side, qty) in enumerate(zip(symbols, sides, qtys)):
            if s not in self._subscribed:
                results[i] = {"symbol": s, "skipped": True, "reason": "not_subscribed"}
                continue
            if s in first_idx:
                results[i] = {"symbol": s, "skipped": True, "reason": "duplicate_lane"}
                continue
            first_idx[s] = i
            uniq_syms.append(s)
            uniq_sides.append(side)
            uniq_qtys.append(qty)

        if uniq_syms:
            _ = self.get_market_features(uniq_syms, timeout_sec=0.5)

        if mode == "local":
            side_to_action = {"hold": 0, "buy": 1, "sell": 2}
            now = int(time.time() * 1e6)
            with self._cache_lock:
                for j, s in enumerate(uniq_syms):
                    price = float(self._market_cache.get(s, {}).get("c") or self._market_cache.get(s, {}).get("price") or 0.0)
                    results[first_idx[s]] = {
                        "order_id": f"local-{s}-{now+j}",
                        "symbol": s,
                        "status": "filled",
                        "filled_avg_price": price,
                        "action": side_to_action.get(uniq_sides[j], 0),
                    }
        else:
            outs: List[dict] = []
            if uniq_syms:
                try:
                    outs = self._submit(self._submit_orders_async(uniq_syms, uniq_sides, uniq_qtys))
                except Exception as e:
                    outs = [{"error": str(e)} for _ in uniq_syms]
            for j, s in enumerate(uniq_syms):
                res = outs[j] if j < len(outs) else {}
                if not isinstance(res, dict):
                    res = {}
                if res.get("error"):
                    results[first_idx[s]] = {"symbol": s, "skipped": True, "reason": "error", "error": str(res.get("error"))}
                else:
                    res["symbol"] = s
                    results[first_idx[s]] = res

        for i, r in enumerate(results):
            if r is None:
                results[i] = {"symbol": symbols[i], "skipped": True, "reason": "no_order"}

        return results

    async def _submit_orders_async(self, symbols: List[str], sides: List[str], qtys: List[int]) -> List[dict]:
        outs: List[dict] = []
        url_base = self.rest_base
        for sym, side, qty in zip(symbols, sides, qtys):
            try:
                await self._rl_rest.acquire()
                payload = {"symbol": sym, "qty": int(qty), "side": side, "type": "market", "time_in_force": "day"}
                status, data = await self._rest_post_json(url_base, "/orders", payload, timeout=5)
                if status and status // 100 == 2 and isinstance(data, dict):
                    outs.append({"order_id": str(data.get("id", f"order-{int(time.time()*1e6)}"))})
                else:
                    tag = "err" if status else "exc"
                    outs.append({"order_id": f"{tag}-{sym}-{int(time.time()*1e6)}"})
            except Exception as e:
                outs.append({"error": str(e)})
        return outs

    def cancel_orders(self, order_ids: List[str]) -> None:
        self._submit(self._cancel_orders_async(order_ids))

    async def _cancel_orders_async(self, order_ids: List[str]) -> None:
        for oid in order_ids:
            await self._rl_rest.acquire()
            _ = await self._rest_delete(self.rest_base, f"/orders/{oid}", timeout=5)

    # ------------------------------- reward plumbing (paper/real) ------------------------------- #
    def step_account(self, order_results, symbols: Optional[List[str]] = None):
        """Compute per-slot NAV deltas for paper/real modes."""

        if symbols is not None:
            syms = list(symbols)
        else:
            syms = []
            for i, r in enumerate(order_results):
                sym = ""
                if isinstance(r, dict):
                    sym = str(r.get("symbol") or "")
                if not sym and i < len(self.init_symbols):
                    sym = self.init_symbols[i]
                syms.append(sym)

        n = len(syms)
        reward = np.zeros(n, dtype=np.float32)
        truncated = np.zeros(n, dtype=bool)
        terminated = np.zeros(n, dtype=bool)

        first_idx: Dict[str, int] = {}
        active_syms: List[str] = []
        for i, (s, r) in enumerate(zip(syms, order_results)):
            if not s or s not in self._subscribed:
                continue
            if isinstance(r, dict) and r.get("skipped"):
                continue
            if s in first_idx:
                continue
            first_idx[s] = i
            active_syms.append(s)

        if not active_syms:
            return reward, truncated, terminated

        account, positions = self._submit(self._get_account_and_positions())
        cash_total = float(account.get("cash", 0.0) or 0.0)
        cash_share = cash_total / float(len(active_syms)) if active_syms else 0.0
        qty_map = {
            str(p.get("symbol") or ""): float(p.get("qty") or p.get("quantity") or 0.0)
            for p in (positions or [])
        }
        with self._cache_lock:
            prices = {
                s: float((self._market_cache.get(s, {}) or {}).get("c")
                         or (self._market_cache.get(s, {}) or {}).get("price")
                         or 0.0)
                for s in active_syms
            }
            prev_map = dict(self._account_cache.get("_nav_prev_by_sym") or {})

        nav_map: Dict[str, float] = {}
        for s in active_syms:
            nav = cash_share + qty_map.get(s, 0.0) * prices.get(s, 0.0)
            nav_map[s] = nav
            prev = float(prev_map.get(s, 0.0) or 0.0)
            reward[first_idx[s]] = float(nav - prev)

        with self._cache_lock:
            self._account_cache["_nav_prev_by_sym"] = nav_map

        return reward, truncated, terminated

    # ------------------------------- parsers ------------------------------- #
    def _parse_market_message(self, msg):
        """
        Normalize Alpaca bar messages into a list of ``(symbol, {o,h,l,c,v,t})``
        pairs. Supports plain dictionaries, lists, and envelopes containing
        ``bars`` or ``data`` arrays.
        """
        import time
        items = []
        if isinstance(msg, list):
            items = [x for x in msg if isinstance(x, dict)]
        elif isinstance(msg, dict):
            if isinstance(msg.get("bars"), list):
                items = [x for x in msg["bars"] if isinstance(x, dict)]
            elif isinstance(msg.get("data"), list):
                items = [x for x in msg["data"] if isinstance(x, dict)]
            else:
                items = [msg]
        else:
            return []

        out = []
        now_fd = time.time() / 86400.0
        for it in items:
            sym = it.get("S") or it.get("symbol")
            if not sym:
                continue
            o = float(it.get("o", 0.0))
            h = float(it.get("h", 0.0))
            l = float(it.get("l", 0.0))
            c = float(it.get("c", 0.0) or it.get("price", 0.0))
            v = float(it.get("v", 0.0))
            t = it.get("t")
            if not isinstance(t, (int, float)):
                t = now_fd
            out.append((sym, {"o": o, "h": h, "l": l, "c": c, "v": v, "t": t}))
        return out

    def _parse_trade_message(self, msg: Any) -> Tuple[str, Dict[str, Any]]:
        try:
            item = (msg[0] if isinstance(msg, list) and msg else msg) if isinstance(msg, (list, dict)) else None
            if not isinstance(item, dict):
                return "", {}
            if "filled_avg_price" in item or "order" in (item.get("type") or ""):
                return "order", item
            if "cash" in item or "equity" in item or item.get("stream") == "account_updates":
                return "account", item
            return "other", item
        except Exception:
            return "", {}
