# alpaca_market_sync.py
# pip install websocket-client requests
from __future__ import annotations
import json, time, threading
from typing import Any, Dict, List, Optional, Literal
from .alpaca_config import AlpacaConfig

import requests
import websocket  # websocket-client
from .stock_utils import COUNTRY_MAP, EXCHANGE_MAP, ASSET_TYPE_MAP
import numpy as np

TradeMode = Literal["local", "paper", "real"]


# ------------------------------ Rate Limiter ------------------------------ #
class SyncTokenBucket:
    """
    Simple token-bucket rate limiter.
    - capacity: maximum bucket size
    - refill_rate: tokens replenished per second
    """
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = float(capacity)
        self.tokens = float(capacity)
        self.refill_rate = float(refill_rate)
        self._lock = threading.Lock()
        self._last = time.time()

    def acquire(self, cost: float = 1.0, block: bool = True, timeout: Optional[float] = None) -> bool:
        deadline = None if timeout is None else time.time() + timeout
        while True:
            with self._lock:
                now = time.time()
                elapsed = now - self._last
                if elapsed > 0:
                    self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
                    self._last = now
                if self.tokens >= cost:
                    self.tokens -= cost
                    return True
            if not block:
                return False
            if deadline is not None and time.time() > deadline:
                return False
            time.sleep(0.01)

class AlpacaMarketSync:
    """
    WS * 2 + HTTP * 1
      - Market WS: receive bars/trades/quotes → _market_cache
      - Trades WS: receive order/fill/account updates → _orders_cache / _account_cache
      - REST: submit_orders / get_account / positions / cancel_orders
    """

    # ------------------------ lifecycle ------------------------ #
    def __init__(self, *, config: AlpacaConfig, symbols: Optional[List[str]] = None):
        self.cfg: AlpacaConfig = config
        self.trade_mode: TradeMode = config.trade_mode
        self.total_symbols: List[str] = list(dict.fromkeys(symbols or []))
        self._subscribed_symbols: set[str] = set()

        # WS objects
        self._market_ws: Optional[websocket.WebSocket] = None
        self._trades_ws: Optional[websocket.WebSocket] = None

        self.country_id = COUNTRY_MAP.get(self.cfg.country_code, 1)  # 1: US
        self.exchange_id = EXCHANGE_MAP.get(self.country_id, {}).get(self.cfg.exchange_code, 1)  # 1: XNYS
        self.asset_type = ASSET_TYPE_MAP.get(self.cfg.asset_type, 1)  # 1: ESXXXX

        # caches
        self._market_cache: Dict[tuple, Dict[str, Dict[str, Any]]] = {}    # symbol -> {o,h,l,c,v,t,...}
        self._account_cache: Dict[str, Any] = {}              # {cash, equity, buying_power, ...}
        self._orders_cache: Dict[str, Dict[str, Any]] = {}    # order_id -> {status, filled_avg_price, ...}
        
        # REST session
        self._rest = requests.Session()
        self._rest.headers.update({
            "APCA-API-KEY-ID": self.cfg.api_key,
            "APCA-API-SECRET-KEY": self.cfg.secret_key,
            "Content-Type": "application/json",
        })

        # rate limiters
        self._rl_rest = SyncTokenBucket(capacity=self.cfg.rest_burst, refill_rate=self.cfg.rest_rps)
        self._rl_ws_pull = SyncTokenBucket(capacity=self.cfg.ws_pull_burst, refill_rate=self.cfg.ws_pull_rps)

    @property
    def init_symbols(self) -> List[str]:
        return self.total_symbols

    # ------------------------ URLs ------------------------ #
    @property
    def rest_base(self) -> str:
        return self.cfg.paper_rest_base if self.trade_mode == "paper" else self.cfg.live_rest_base

    # ------------------------ connect/close ------------------------ #
    def connect_market(self) -> None:
        self._market_ws = websocket.create_connection(self.cfg.market_ws_url, timeout=self.cfg.recv_timeout_sec)
        self._send_ws(self._market_ws, {"action": "auth", "key": self.cfg.api_key, "secret": self.cfg.secret_key})
        _ = self._recv_ws(self._market_ws, self.cfg.recv_timeout_sec)  # auth ack (can be ignored)
        # initial subscription
        if self.total_symbols:
            self.subscribe(self.total_symbols)

    def connect_trades(self) -> None:
        # Order/account update WS (adjust endpoint for the environment)
        self._trades_ws = websocket.create_connection(self.cfg.trades_ws_url, timeout=self.cfg.recv_timeout_sec)
        self._send_ws(self._trades_ws, {"action": "auth", "key": self.cfg.api_key, "secret": self.cfg.secret_key})
        _ = self._recv_ws(self._trades_ws, self.cfg.recv_timeout_sec)  # auth ack
        
    def get_account_data(self) -> Dict[str, Any]:
        account_data = {}
        account_data["cash"] = self._account_cache.get("cash", 0.0)
        account_data["equity"] = self._account_cache.get("equity", 0.0)
        account_data["buying_power"] = self._account_cache.get("buying_power", 0.0)
        return account_data

    def close_market(self) -> None:
        try:
            if self._market_ws:
                self._market_ws.close()
        except Exception:
            pass
        self._market_ws = None

    def close_trades(self) -> None:
        try:
            if self._trades_ws:
                self._trades_ws.close()
        except Exception:
            pass
        self._trades_ws = None

    def close(self) -> None:
        self.unsubscribe(list(set(self.total_symbols)))
        self.close_trades()
        self.close_market()

    def subscribe(self, symbols: List[str]) -> None:
        if not symbols:
            return

        new_set = set(symbols)
        if new_set == self._subscribed_symbols:
            return  # no change

        # Remove any old subscriptions no longer needed
        if self._market_ws:
            self.unsubscribe(list(self._subscribed_symbols - new_set))

        self._subscribed_symbols = new_set

        if self._market_ws:
            self._send_ws(self._market_ws, {"action": "subscribe", "bars": symbols})

    def unsubscribe(self, symbols: List[str]) -> None:
        if not symbols:
            return
        rm = set(symbols) & self._subscribed_symbols
        if not rm:
            return
        self._subscribed_symbols -= rm
        if self._market_ws:
            self._send_ws(self._market_ws, {"action": "unsubscribe", "bars": list(rm)})

    # ------------------------ Snapshots ------------------------ #
    def fetch_snapshot(self, symbols: List[str], timeout_sec: float = 1.0) -> Dict[str, dict]:
        """
        Return the latest snapshot for requested symbols from the recent market WS cache.
        If absent, wait for WS messages until ``timeout_sec``.
        """
        need = set(symbols)
        have = {s for s in symbols if s in self._market_cache}
        deadline = time.time() + float(timeout_sec)

        # Subscribe immediately if any symbols are missing
        miss = [s for s in symbols if s not in self._subscribed_symbols]
        if miss:
            self.subscribe(miss)

        while have != need and time.time() < deadline:
            # Wait for the receiving thread to populate the cache
            time.sleep(0.01)

        return {s: self._market_cache.get(s, {}) for s in symbols}

    # ------------------------ Orders (REST) ------------------------ #
    def submit_orders(self, symbols: List[str], sides: List[str], qtys: List[int], trade_mode: str) -> List[dict]:
        """
        trade_mode:
            - "local":  Simulate fills using latest prices. Returns list of order
              result dicts consumable by ``LocalAccount`` with fields
              ``filled_avg_price`` and ``action``.
            - "paper"/"real": POST to REST API. Returns list of dicts containing
              at least the ``order_id`` for each submission.
        """
        assert len(symbols) == len(sides) == len(qtys), "symbols/sides/qtys length mismatch"
        mode = trade_mode or self.trade_mode
        order_results: List[dict] = []

        if mode == "local":
            # Simulated fills at current price; return records LocalAccount can consume.
            _ = self.fetch_snapshot(list(set(symbols)), timeout_sec=0.5)
            prices = [self._get_symbol_price(sym) for sym in symbols]
            side_to_action = {"hold": 0, "buy": 1, "sell": 2}
            for sym, side, price in zip(symbols, sides, prices):
                oid = f"local-{sym}-{int(time.time()*1e6)}"
                rec = {
                    "order_id": oid,
                    "symbol": sym,
                    "status": "filled",
                    "filled_avg_price": float(price or 0.0),
                    "action": side_to_action.get(side, 0),
                }
                self._orders_cache[oid] = rec
                order_results.append(rec)
            return order_results

        # paper / real
        url = f"{self.rest_base}/orders"
        for sym, side, qty in zip(symbols, sides, qtys):
            self._rl_rest.acquire()  # rate limit
            payload = {
                "symbol": sym,
                "qty": int(qty),
                "side": side,                 # "buy" or "sell"
                "type": "market",
                "time_in_force": "day",
            }
            try:
                resp = self._rest.post(url, data=json.dumps(payload), timeout=5)
                if resp.status_code // 100 == 2:
                    data = resp.json()
                    oid = str(data.get("id", f"order-{int(time.time()*1e6)}"))
                    self._orders_cache[oid] = data
                    order_results.append({"order_id": oid})
                else:
                    oid = f"err-{sym}-{int(time.time()*1e6)}"
                    order_results.append({"order_id": oid})
            except Exception:
                oid = f"exc-{sym}-{int(time.time()*1e6)}"
                order_results.append({"order_id": oid})
        return order_results

    def positions(self) -> List[Dict[str, Any]]:
        self._rl_rest.acquire()
        url = f"{self.rest_base}/positions"
        try:
            resp = self._rest.get(url, timeout=5)
            if resp.status_code // 100 == 2:
                return resp.json() or []
        except Exception:
            pass
        return []

    def cancel_orders(self, order_ids: List[str]) -> None:
        for oid in order_ids:
            self._rl_rest.acquire()
            url = f"{self.rest_base}/orders/{oid}"
            try:
                self._rest.delete(url, timeout=5)
            except Exception:
                pass

    def update_account(self, order_results):

        # number of parallel slots (aligns with env action/observation)
        n = len(order_results) if order_results is not None else len(self.total_symbols) or 1
        env_syms = self.total_symbols[:n]

        def _f(x, d=0.0):
            try:
                return float(x)
            except Exception:
                return float(d)

        # --- account cash (single pool) ---
        self._rl_rest.acquire()
        cash = 0.0
        try:
            resp = self._rest.get(f"{self.rest_base}/account", timeout=5)
            if resp.status_code // 100 == 2:
                acc = resp.json() or {}
                self._account_cache.update(acc)
                cash = _f(acc.get("cash", 0.0), 0.0)
        except Exception:
            pass

        # --- positions map: symbol -> qty ---
        qty_map = {}
        try:
            for p in self.positions():
                sym = p.get("symbol")
                if sym:
                    qty_map[sym] = _f(p.get("qty") or p.get("quantity") or 0.0, 0.0)
        except Exception:
            pass

        # --- NAV per slot: split cash equally + position value with WS price (fallback 0) ---
        cash_share = cash / float(n)
        nav_vec = np.zeros(n, dtype=np.float64)
        
        for i, sym in enumerate(env_syms):
            qty = qty_map.get(sym, 0.0)
            price = self._get_symbol_price(sym) or 0.0
            nav_vec[i] = cash_share + qty * price

        # --- reward = NAV_t - NAV_{t-1} (vector) ---
        prev_vec = np.asarray(self._account_cache.get("_nav_prev_vec", np.zeros(n, dtype=np.float64)), dtype=np.float64)
        if prev_vec.shape[0] != n:
            prev_vec = np.zeros(n, dtype=np.float64)
        reward = (nav_vec - prev_vec).astype(np.float32)

        # cache & flags
        self._account_cache["_nav_prev_vec"] = nav_vec
        truncated  = np.zeros(n, dtype=bool)
        terminated = np.zeros(n, dtype=bool)
        return reward, truncated, terminated

    # ------------------------ Helpers ------------------------ #
    def _get_symbol_price(self, symbol: str) -> float:
        snap = self._market_cache.get(symbol)
        if not snap:
            return 0.0
        # Priority: filled_avg_price-like → close → price
        return float(snap.get("c") or snap.get("price") or 0.0)

    def _get_symbol_prices(self, symbols: list[str]) -> list[float]:
        return [self._get_symbol_price(sym) for sym in symbols]

    def _send_ws(self, ws: websocket.WebSocket, obj: Dict[str, Any]) -> None:
        try:
            ws.send(json.dumps(obj))
        except Exception:
            pass

    def _recv_ws(self, ws: websocket.WebSocket, timeout: float) -> Optional[Any]:
        try:
            ws.settimeout(timeout)
            raw = ws.recv()
            return json.loads(raw) if raw else None
        except Exception:
            return None
