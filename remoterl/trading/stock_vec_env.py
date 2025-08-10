
from __future__ import annotations
import numpy as np
import gymnasium as gym
from .alpaca_market_sync import AlpacaMarketSync as AlpacaMarket
import numpy as np
from .stock_utils import NUM_COUNTRIES, NUM_EXCHANGES, NUM_ASSET_TYPES, NUM_LOCAL_SYMBOLS

class StockVecEnv(gym.Env):
    
    """
    Vectorized stock-trading env for RemoteRL + SB3/RLlib.

    Modes:
      - "real":   live orders via Alpaca (dry_run=False)
      - "local":  simulated fills w/ live prices (dry_run=True)
      - "paper": fully simulated (no order calls)
    """

    def __init__(
        self,
        num_envs: int,
        *,
        alpaca_market: AlpacaMarket,                 # shared client for AlpacaAccountSync/Async or LocalAccount
        **kwargs,
    ) -> None:
        super().__init__()
        self.num_envs = int(num_envs)
        self.alpaca_market = alpaca_market
        self.trade_mode = alpaca_market.trade_mode  # "local" | "paper" | "real"

        # Symbols are expected to be injected externally; otherwise use the first N from trade_client.init_symbols.
        init_symbols = list(getattr(alpaca_market, "init_symbols", []))
        assert len(init_symbols) >= self.num_envs, "init_symbols must cover num_envs"
        self.symbols = init_symbols[: self.num_envs]

        self.country_id  = alpaca_market.country_id
        self.exchange_id = alpaca_market.exchange_id
        self.asset_type  = alpaca_market.asset_type

        # Local-mode-only account simulator (holds cash/positions internally)
        # → use the injected symbols as-is
        from .local_account import LocalAccount  # if separated into another file
        self.local_account = LocalAccount(
            num_envs=self.num_envs,
            symbols=self.symbols,
            rng=np.random.default_rng(),
            **{k: v for k, v in kwargs.items() if k in {"num_stocks_range","budget_range","max_steps"}},
        ) if self.trade_mode == "local" else None

        # Observation feature keys
        self.external_feature_keys = ["o", "h", "l", "c", "v"]
        external_dims = len(self.external_feature_keys)
        # "external_state" is external market data, "internal_state" is internal state

        self.internal_feature_keys = [
            "position_qty",     # position quantity
            "cash",             # cash per account/slot
            "avg_entry_price",  # average entry price
            "unrealized_pnl",   # unrealized PnL
            "exposure",          # current exposure (price * qty)
            "group_NAV"         # average group net asset value
        ]
        internal_dim = len(self.internal_feature_keys)
        
        # Gym spaces 
        self.observation_space = gym.spaces.Dict({
            "asset_id": gym.spaces.MultiDiscrete(
                [NUM_COUNTRIES, NUM_EXCHANGES, NUM_ASSET_TYPES, NUM_LOCAL_SYMBOLS]
            ),
            "external_state": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(external_dims,), dtype=np.float32),
            "internal_state": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(internal_dim,), dtype=np.float32),
            "time": gym.spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32),
        })
        
        # Actions: per slot {0: hold, 1: buy, 2: sell}
        self.action_space = gym.spaces.MultiDiscrete([3] * self.num_envs)

        # Runtime
        self.step_idx = 0
    

    def _account_data_local_as_symbol_map(self) -> dict[str, dict]:
        """
        Convert LocalAccount arrays into a per‑symbol dict of pure floats,
        matching self.internal_feature_keys order.
        """
        assert self.trade_mode == "local" and self.local_account is not None
        acc = self.local_account.get_account_data()  # fetch-only

        out = {}
        for i, sym in enumerate(self.symbols):
            row = {}
            for k in self.internal_feature_keys:
                v = acc.get(k)
                if isinstance(v, np.ndarray):
                    row[k] = float(v[i])
                else:
                    row[k] = float(v)
            out[sym] = row
        return out


    def process_time(self, market_data, symbols):
        raw_time = np.array([market_data[symbol]["t"] for symbol in symbols], dtype=np.float32)
        
        # Define time features for day, week, month, quarter, year sin/cos embeddings
        day_sin = np.sin(2 * np.pi * (raw_time % 1) / 1)
        day_cos = np.cos(2 * np.pi * (raw_time % 1) / 1)

        day_of_week_sin = np.sin(2 * np.pi * (raw_time % 7) / 7)
        day_of_week_cos = np.cos(2 * np.pi * (raw_time % 7) / 7)

        month_of_year_sin = np.sin(2 * np.pi * (raw_time % 12) / 12)
        month_of_year_cos = np.cos(2 * np.pi * (raw_time % 12) / 12)

        quarter_of_year_sin = np.sin(2 * np.pi * (raw_time % 4) / 4)
        quarter_of_year_cos = np.cos(2 * np.pi * (raw_time % 4) / 4)

        year_sin = np.sin(2 * np.pi * (raw_time % 365) / 365)
        year_cos = np.cos(2 * np.pi * (raw_time % 365) / 365)

        time = np.concatenate([
            day_sin, day_cos,
            day_of_week_sin, day_of_week_cos,
            month_of_year_sin, month_of_year_cos,
            quarter_of_year_sin, quarter_of_year_cos,
            year_sin, year_cos
        ], axis=-1)
        
        return time

    def extract_features(self, market_data: dict[str, dict], account_data: dict) -> dict[str, np.ndarray]:
        
        symbols = list(market_data.keys())
        n = len(symbols)

        # ---- asset_id: (N, 4) int64  [country, venue, asset_type, local_symbol]
        local_symbol_ids = np.arange(n, dtype=np.int64) + 1  # replace with stable ID when ready
        asset_ids = np.stack([
            np.full(n, self.country_id, dtype=np.int64),
            np.full(n, self.exchange_id, dtype=np.int64),
            np.full(n, self.asset_type, dtype=np.int64),
            local_symbol_ids,
        ], axis=-1)

        # ---- external_state: (N, external_dims) float32  (o,h,l,c,v from market_data)
        external_state = np.asarray(
            [[float(market_data[s].get(k, 0.0)) for k in self.external_feature_keys] for s in symbols],
            dtype=np.float32,
        )

        # ---- internal_state: (N, internal_dim) float32
        # In local mode, account_data provides arrays per field; map them per-slot here.
        if self.trade_mode == "local" and self.local_account is not None and account_data:
            internal_rows = []
            for i, _sym in enumerate(symbols):
                row = [float(account_data.get(k, 0.0)[i] if isinstance(account_data.get(k), np.ndarray)
                            else account_data.get(k, 0.0))
                       for k in self.internal_feature_keys]
                internal_rows.append(row)
            internal_state = np.asarray(internal_rows, dtype=np.float32)
        else:
            # paper/real: fill later from remote account, or zeros as placeholder
            internal_state = np.zeros((n, len(self.internal_feature_keys)), dtype=np.float32)
        # ---- time: (N, 10) float32  (day, week, month, quarter, year → sin/cos)
        time = self.process_time(market_data, symbols).astype(np.float32)

        return {
            "asset_id": asset_ids,
            "external_state": external_state,
            "internal_state": internal_state,
            "time": time,
        }

    # ------------------------------------------------------------------ API
    def get_account_data(self):
        if self.trade_mode == "local":
            account_data = self.local_account.get_account_data()
        else:
            account_data = self.alpaca_market.get_account_data()   
        return account_data
            

    def fetch_observation(self, account_data):
        # market snapshot = external features only; internal comes from account_data        
        raw_obs = self.alpaca_market.fetch_snapshot(self.symbols)
        return self.extract_features(raw_obs, account_data)

    def reset(self, *, seed=None, options=None):
        self.step_idx = 0
        
        # Episode reset only for local mode (reset cash/positions)
        account_data = self.get_account_data()

        # Batch snapshot collection → build vector observation
        obs = self.fetch_observation(account_data)

        # Gymnasium: single dict for info is sufficient (can switch to per-slot list if needed)
        info = {}
        
        return obs, info
    
    def update_account(self, order_results):
        if self.trade_mode == "local":
            return self.local_account.update_account(order_results, self.step_idx)
        return self.alpaca_market.update_account(order_results)

    def step(self, action):
        self.step_idx += 1
        action = np.asarray(action, dtype=np.int64)

        sides = ["buy" if a == 1 else "sell" if a == 2 else "hold" for a in action]
        qtys = [1] * self.num_envs

        # submit_orders returns order results (local) or order IDs (paper/real)
        order_results = self.alpaca_market.submit_orders(
            self.symbols, sides, qtys, self.trade_mode
        )

        # Update account/reward based on executed orders
        reward, truncated, terminated = self.update_account(order_results)

        # Fetch latest account snapshot for observation construction
        account_data = self.get_account_data()
        observation = self.fetch_observation(account_data)

        info = {}

        return observation, reward, terminated, truncated, info

    def close(self):
        """Clean up resources and subscriptions based on trade_mode."""
        try:
            if self.trade_mode != "local":
                # paper/real mode: close connected sessions/WS if present
                if hasattr(self.alpaca_market, "close") and callable(self.alpaca_market.close):
                    self.alpaca_market.close()
        except Exception as e:
            # Logging output is optional
            print(f"[StockVecEnv.close] Cleanup failed: {e}")

        # Default Gymnasium close call
        super().close()
