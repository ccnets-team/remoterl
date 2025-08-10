# alpaca_market_async.py (clean, threaded loop)
# Async WS + REST client that runs its own asyncio event-loop in a
# background thread, exposing a sync-friendly API for use from Gym envs.
#
# pip install aiohttp websockets
from __future__ import annotations
import asyncio
import contextlib
import json
import threading
import time
from typing import Any, Dict, List, Optional, Literal, Tuple

import aiohttp
import websockets
from websockets.legacy.client import WebSocketClientProtocol

from .alpaca_config import AlpacaConfig

TradeMode = Literal["local", "paper", "real"]


# ------------------------------ Async Token Bucket ------------------------------ #
class TokenBucketAsync:
    """Simple async token-bucket with asyncio.Lock.
    Used to throttle REST and WS pulls without blocking the loop.
    """
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = float(capacity)
        self.tokens = float(capacity)
        self.refill_rate = float(refill_rate)
        self._lock = asyncio.Lock()
        self._last = time.time()

    async def acquire(self, cost: float = 1.0, timeout: Optional[float] = None) -> bool:
        deadline = None if timeout is None else (asyncio.get_event_loop().time() + timeout)
        while True:
            async with self._lock:
                now = time.time()
                elapsed = now - self._last
                if elapsed > 0:
                    self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
                    self._last = now
                if self.tokens >= cost:
                    self.tokens -= cost
                    return True
            if deadline is not None and asyncio.get_event_loop().time() > deadline:
                return False
            await asyncio.sleep(0.01)


# ------------------------------ AlpacaMarketAsync ------------------------------ #
class AlpacaMarketAsync:
    """
    Async Alpaca market/trade client that:
      • Opens market & trades WebSockets and streams in background *async tasks*
      • Exposes sync-friendly methods by scheduling onto an internal loop
      • Keeps small in-memory caches updated for snapshots/orders/account

    This class mirrors AlpacaMarketSync's public surface closely while
    being safe to call from synchronous code (e.g., Gym VecEnv) thanks to
    an internal event-loop running in a daemon thread.
    """

    # ------------------------ lifecycle ------------------------ #
    def __init__(self, *, config: AlpacaConfig, symbols: Optional[List[str]] = None):
        self.cfg = config
        self.trade_mode: TradeMode = config.trade_mode
        self.init_symbols: List[str] = list(dict.fromkeys(symbols or []))

        # async resources (live only on the loop thread)
        self._session: Optional[aiohttp.ClientSession] = None
        self._market_ws: Optional[WebSocketClientProtocol] = None
        self._trades_ws: Optional[WebSocketClientProtocol] = None
        self._market_task: Optional[asyncio.Task] = None
        self._trades_task: Optional[asyncio.Task] = None
        self._stop_evt: Optional[asyncio.Event] = None

        # caches (shared; guard with a threading.Lock)
        self._cache_lock = threading.Lock()
        self._market_cache: Dict[str, Dict[str, Any]] = {}
        self._account_cache: Dict[str, Any] = {}
        self._orders_cache: Dict[str, Dict[str, Any]] = {}
        self._subscribed: set[str] = set()

        # rate limiters (async)
        self._rl_rest = TokenBucketAsync(capacity=self.cfg.rest_burst, refill_rate=self.cfg.rest_rps)
        self._rl_ws_pull = TokenBucketAsync(capacity=self.cfg.ws_pull_burst, refill_rate=self.cfg.ws_pull_rps)

        # dedicated asyncio loop in a background thread
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, name="Alpaca-AsyncLoop", daemon=True)
        self._thread.start()

    # ------------------------ runner ------------------------ #
    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_forever()
        finally:
            # best-effort graceful shutdown of pending tasks
            pending = asyncio.all_tasks(self._loop)
            for t in pending:
                t.cancel()
            if pending:
                self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            self._loop.run_until_complete(self._loop.shutdown_asyncgens())
            self._loop.close()

    # small helpers to bridge sync → async
    def _submit(self, coro, *, timeout: Optional[float] = None):
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result(timeout=timeout)

    # ------------------------ URLs ------------------------ #
    @property
    def rest_base(self) -> str:
        return self.cfg.paper_rest_base if self.trade_mode == "paper" else self.cfg.live_rest_base

    # ------------------------ connect/close (sync wrappers) ------------------------ #
    def connect_market(self) -> None:
        self._submit(self._connect_market_async())

    def connect_trades(self) -> None:
        self._submit(self._connect_trades_async())

    def close_market(self) -> None:
        self._submit(self._close_market_async())

    def close_trades(self) -> None:
        self._submit(self._close_trades_async())

    def close(self) -> None:
        # full shutdown, including loop/thread
        self._submit(self._close_all_async())
        # stop the loop thread after resources are closed
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    # ------------------------ async impls ------------------------ #
    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession(headers={
                "APCA-API-KEY-ID": self.cfg.api_key,
                "APCA-API-SECRET-KEY": self.cfg.secret_key,
                "Content-Type": "application/json",
            })
        return self._session

    async def _connect_market_async(self) -> None:
        await self._ensure_session()
        self._stop_evt = self._stop_evt or asyncio.Event()
        self._market_ws = await websockets.connect(self.cfg.market_ws_url)
        await self._send_ws(self._market_ws, {"action": "auth", "key": self.cfg.api_key, "secret": self.cfg.secret_key})
        _ = await self._recv_ws(self._market_ws, self.cfg.recv_timeout_sec)
        if self.init_symbols:
            await self._subscribe_async(self.init_symbols)
        # start background pump
        self._market_task = asyncio.create_task(self._market_loop())

    async def _connect_trades_async(self) -> None:
        await self._ensure_session()
        self._stop_evt = self._stop_evt or asyncio.Event()
        self._trades_ws = await websockets.connect(self.cfg.trades_ws_url)
        await self._send_ws(self._trades_ws, {"action": "auth", "key": self.cfg.api_key, "secret": self.cfg.secret_key})
        _ = await self._recv_ws(self._trades_ws, self.cfg.recv_timeout_sec)
        # start background pump
        self._trades_task = asyncio.create_task(self._trades_loop())

    async def _close_market_async(self) -> None:
        if self._market_task and not self._market_task.done():
            self._market_task.cancel()
            with contextlib.suppress(Exception):
                await self._market_task
        if self._market_ws:
            with contextlib.suppress(Exception):
                await self._market_ws.close()
        self._market_task = None
        self._market_ws = None

    async def _close_trades_async(self) -> None:
        if self._trades_task and not self._trades_task.done():
            self._trades_task.cancel()
            with contextlib.suppress(Exception):
                await self._trades_task
        if self._trades_ws:
            with contextlib.suppress(Exception):
                await self._trades_ws.close()
        self._trades_task = None
        self._trades_ws = None

    async def _close_all_async(self) -> None:
        if self._stop_evt:
            self._stop_evt.set()
        await self._close_trades_async()
        await self._close_market_async()
        if self._session:
            with contextlib.suppress(Exception):
                await self._session.close()
        self._session = None

    # ------------------------ subscribe (sync wrappers) ------------------------ #
    def subscribe(self, symbols: List[str]) -> None:
        self._submit(self._subscribe_async(symbols))

    def unsubscribe(self, symbols: List[str]) -> None:
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

    # ------------------------ Snapshots ------------------------ #
    def fetch_snapshot(self, symbols: List[str], timeout_sec: float = 1.0) -> Dict[str, dict]:
        """Return latest cached bars for *symbols*.
        Blocks the caller thread up to *timeout_sec* while the async pump fills.
        """
        deadline = time.time() + float(timeout_sec)
        # ensure subscription
        miss = [s for s in symbols if s not in self._subscribed]
        if miss:
            try:
                self.subscribe(miss)
            except Exception:
                pass
        # poll the shared cache (protected by lock) until all present or timeout
        need = set(symbols)
        have: set[str] = set()
        while time.time() < deadline:
            with self._cache_lock:
                have = {s for s in symbols if s in self._market_cache}
                if have == need:
                    break
            time.sleep(0.01)
        with self._cache_lock:
            return {s: dict(self._market_cache.get(s, {})) for s in symbols}

    # ------------------------ Orders (sync wrapper) ------------------------ #
    def submit_orders(self, symbols: List[str], sides: List[str], qtys: List[int], trade_mode: str) -> List[str]:
        return self._submit(self._submit_orders_async(symbols, sides, qtys, trade_mode))

    async def _submit_orders_async(self, symbols: List[str], sides: List[str], qtys: List[int], trade_mode: str) -> List[str]:
        assert len(symbols) == len(sides) == len(qtys), "symbols/sides/qtys length mismatch"
        mode = trade_mode or self.trade_mode
        if mode == "local":
            # make sure we have current prices
            _ = self.fetch_snapshot(list(set(symbols)), timeout_sec=0.5)
            order_ids: List[str] = []
            with self._cache_lock:
                for sym in symbols:
                    price = self._get_symbol_price_locked(sym)
                    oid = f"local-{sym}-{int(time.time()*1e6)}"
                    self._orders_cache[oid] = {"symbol": sym, "status": "filled", "filled_avg_price": price}
                    order_ids.append(oid)
            return order_ids

        # paper / real via aiohttp
        await self._rl_rest.acquire()
        sess = await self._ensure_session()
        url = f"{self.rest_base}/orders"
        order_ids: List[str] = []
        for sym, side, qty in zip(symbols, sides, qtys):
            await self._rl_rest.acquire()
            payload = {
                "symbol": sym,
                "qty": int(qty),
                "side": side,
                "type": "market",
                "time_in_force": "day",
            }
            try:
                async with sess.post(url, data=json.dumps(payload), timeout=5) as resp:
                    if resp.status // 100 == 2:
                        data = await resp.json()
                        oid = str(data.get("id", f"order-{int(time.time()*1e6)}"))
                        with self._cache_lock:
                            self._orders_cache[oid] = data
                        order_ids.append(oid)
                    else:
                        order_ids.append(f"err-{sym}-{int(time.time()*1e6)}")
            except Exception:
                order_ids.append(f"exc-{sym}-{int(time.time()*1e6)}")
        return order_ids

    # ------------------------ Account/Positions (sync wrappers) ------------------------ #
    def get_account(self) -> Dict[str, Any]:
        return self._submit(self._get_account_async())

    async def _get_account_async(self) -> Dict[str, Any]:
        await self._rl_rest.acquire()
        sess = await self._ensure_session()
        url = f"{self.rest_base}/account"
        try:
            async with sess.get(url, timeout=5) as resp:
                data = await resp.json() if resp.status // 100 == 2 else {}
                if data:
                    with self._cache_lock:
                        self._account_cache.update(data)
                return data or dict(self._account_cache)
        except Exception:
            with self._cache_lock:
                return dict(self._account_cache)

    def positions(self) -> List[Dict[str, Any]]:
        return self._submit(self._positions_async())

    async def _positions_async(self) -> List[Dict[str, Any]]:
        await self._rl_rest.acquire()
        sess = await self._ensure_session()
        url = f"{self.rest_base}/positions"
        try:
            async with sess.get(url, timeout=5) as resp:
                if resp.status // 100 == 2:
                    return await resp.json() or []
        except Exception:
            pass
        return []

    def cancel_orders(self, order_ids: List[str]) -> None:
        self._submit(self._cancel_orders_async(order_ids))

    async def _cancel_orders_async(self, order_ids: List[str]) -> None:
        sess = await self._ensure_session()
        for oid in order_ids:
            await self._rl_rest.acquire()
            url = f"{self.rest_base}/orders/{oid}"
            with contextlib.suppress(Exception):
                await sess.delete(url, timeout=5)

    # ------------------------ WS helpers ------------------------ #
    async def _send_ws(self, ws: WebSocketClientProtocol, obj: Dict[str, Any]) -> None:
        try:
            await ws.send(json.dumps(obj))
        except Exception:
            pass

    async def _recv_ws(self, ws: WebSocketClientProtocol, timeout: float) -> Optional[Any]:
        try:
            return await asyncio.wait_for(ws.recv(), timeout=timeout)
        except Exception:
            return None

    # ------------------------ WS pumps ------------------------ #
    async def _market_loop(self) -> None:
        ws = self._market_ws
        if not ws:
            return
        while self._stop_evt and not self._stop_evt.is_set():
            await self._rl_ws_pull.acquire()
            raw = await self._recv_ws(ws, self.cfg.recv_timeout_sec)
            if not raw:
                continue
            try:
                msg = json.loads(raw) if isinstance(raw, (bytes, bytearray, str)) else raw
                sym, payload = self._parse_market_message(msg)
                if sym:
                    with self._cache_lock:
                        self._market_cache[sym] = payload
            except Exception:
                continue

    async def _trades_loop(self) -> None:
        ws = self._trades_ws
        if not ws:
            return
        while self._stop_evt and not self._stop_evt.is_set():
            await self._rl_ws_pull.acquire()
            raw = await self._recv_ws(ws, self.cfg.recv_timeout_sec)
            if not raw:
                continue
            try:
                msg = json.loads(raw) if isinstance(raw, (bytes, bytearray, str)) else raw
                kind, payload = self._parse_trade_message(msg)
                with self._cache_lock:
                    if kind == "order" and payload:
                        oid = str(payload.get("id", f"order-{int(time.time()*1e6)}"))
                        self._orders_cache[oid] = payload
                    elif kind == "account" and payload:
                        self._account_cache.update(payload)
            except Exception:
                continue

    # ------------------------ Parsers (demo) ------------------------ #
    def _parse_market_message(self, msg: Any) -> Tuple[Optional[str], Dict[str, Any]]:
        """Accept either list-of-bars or single dict; return (symbol, payload)."""
        try:
            if isinstance(msg, list) and msg:
                item = msg[0]
            elif isinstance(msg, dict):
                item = msg
            else:
                return None, {}
            sym = item.get("S") or item.get("symbol")
            if not sym:
                return None, {}
            return sym, item
        except Exception:
            return None, {}

    def _parse_trade_message(self, msg: Any) -> Tuple[str, Dict[str, Any]]:
        try:
            if isinstance(msg, list) and msg:
                item = msg[0]
            elif isinstance(msg, dict):
                item = msg
            else:
                return "", {}
            if "filled_avg_price" in item or "order" in (item.get("type") or ""):
                return "order", item
            if "cash" in item or "equity" in item or item.get("stream") == "account_updates":
                return "account", item
            return "other", item
        except Exception:
            return "", {}

    # ------------------------ helpers ------------------------ #
    def _get_symbol_price_locked(self, symbol: str) -> float:
        snap = self._market_cache.get(symbol)
        if not snap:
            return 0.0
        return float(snap.get("c") or snap.get("price") or 0.0)
