
from __future__ import annotations
import numpy as np
import gymnasium as gym
from .trading_market import TradingMarket
from .asset_utils import NUM_COUNTRIES, NUM_EXCHANGES, NUM_ASSET_TYPES, NUM_LOCAL_SYMBOLS

class TradingVecEnv(gym.vector.VectorEnv):

    """
    Vectorized stock-trading environment compatible with RemoteRL, Stable-Baselines3,
    and RLlib.

    Modes
    -----
    - ``"real"``:   place live orders through Alpaca
    - ``"local"``:  simulate fills using live prices
    - ``"paper"``: fully simulated; no order submission
    """

    def __init__(
        self,
        num_envs: int,
        *,
        alpaca_market: TradingMarket,                 # Shared TradingMarket or LocalAccount client
        **kwargs,
    ) -> None:
        super().__init__()
        self.num_envs = int(num_envs)
        self.alpaca_market = alpaca_market
        self.trade_mode = alpaca_market.trade_mode  # "local" | "paper" | "real"

        # Symbols are expected to be provided externally; otherwise use the first N
        # from the client's initial subscription list.
        self.init_symbols = list(getattr(alpaca_market, "init_symbols", []))
        symbols = []
        if self.init_symbols:
            if len(self.init_symbols) >= self.num_envs:
                symbols = self.init_symbols[: self.num_envs]
            else:
                reps = (self.num_envs + len(self.init_symbols) - 1) // len(self.init_symbols)
                symbols = (self.init_symbols * reps)[: self.num_envs]
        self.symbols = np.array(symbols, dtype=object)
        
        self.country_id  = alpaca_market.country_id
        self.exchange_id = alpaca_market.exchange_id
        self.asset_type  = alpaca_market.asset_type

        # Local-mode-only account simulator that maintains cash and positions
        # internally; uses the injected symbols directly
        from .local_account import LocalAccount  # imported lazily
        self.local_account = LocalAccount(
            num_envs=self.num_envs,
            **{k: v for k, v in kwargs.items() if k in {"num_stocks_range","budget_range","max_step_range"}},
        ) if self.trade_mode == "local" else None

        # Observation feature keys
        self.MARKET_FEATURE_KEYS = ["o", "h", "l", "c", "v"]

        self.ACCOUNT_FEATURE_KEYS = [
            "position_qty",     # position quantity
            "cash",             # cash per slot
            "avg_entry_price",  # average entry price
            "unrealized_pnl",   # unrealized profit and loss
            "exposure",         # current exposure (price * qty)
            "asset_nav"         # per-asset net asset value (slot-level NAV)
        ]
        
        # Gym spaces
        n = self.num_envs
        self.observation_space = gym.spaces.Dict({
            "asset_id": gym.spaces.MultiDiscrete([NUM_COUNTRIES, NUM_EXCHANGES, NUM_ASSET_TYPES, NUM_LOCAL_SYMBOLS]),
            "market_features": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(n, len(self.MARKET_FEATURE_KEYS)), dtype=np.float32),
            "account_features": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(n, len(self.ACCOUNT_FEATURE_KEYS)), dtype=np.float32),
            "time_features": gym.spaces.Box(low=-1.0, high=1.0, shape=(n, 10), dtype=np.float32),
        })
        
        # Actions: per environment slot {0: hold, 1: buy, 2: sell}
        self.action_space = gym.spaces.MultiDiscrete([3] * self.num_envs)

        # Runtime state
        self.step_count = 0
            
    def _build_time_features(self, market_features, symbols):
        """
        Use time column (col=5) from market_features ndarray.
        """
        import numpy as np
        raw_time = np.asarray(market_features)[:, 5].astype(np.float32)  # fractional days
        feats = np.stack([
            np.sin(2*np.pi*(raw_time % 1)   / 1.0),  np.cos(2*np.pi*(raw_time % 1)   / 1.0),
            np.sin(2*np.pi*(raw_time % 7)   / 7.0),  np.cos(2*np.pi*(raw_time % 7)   / 7.0),
            np.sin(2*np.pi*(raw_time % 12) / 12.0),  np.cos(2*np.pi*(raw_time % 12) / 12.0),
            np.sin(2*np.pi*(raw_time % 4)   / 4.0),  np.cos(2*np.pi*(raw_time % 4)   / 4.0),
            np.sin(2*np.pi*(raw_time % 365)/365.0),  np.cos(2*np.pi*(raw_time % 365)/365.0),
        ], axis=-1).astype(np.float32)
        return feats

    def _build_observation_features(self, market_features, account_features):
        """
        Both inputs are ndarray:
        market_features: (N,5) or (N,6) [o,h,l,c,v,(t)]
        account_features: (N,6) per ACCOUNT_FEATURE_KEYS
        """
        n = len(self.symbols)
        asset_ids = np.stack([
            np.full(n, self.country_id,  dtype=np.int64),
            np.full(n, self.exchange_id, dtype=np.int64),
            np.full(n, self.asset_type,  dtype=np.int64),
            np.arange(n, dtype=np.int64) + 1,
        ], axis=-1)

        # Drop time for market_features box if you keep obs space at 5-cols; 
        # or keep 5-cols in obs and pass time via time_features only.
        market_feats = np.asarray(market_features[:, :5], dtype=np.float32)
        account_feats = np.asarray(account_features, dtype=np.float32)
        time_feats = self._build_time_features(market_features, self.symbols)

        return {
            "asset_id": asset_ids,
            "market_features": market_feats,
            "account_features": account_feats,
            "time_features": time_feats,
        }

    def get_account_features(self, symbols: np.ndarray) -> np.ndarray:
        """
        Return (N,6) ndarray for both local and non-local modes.
        """
        if self.trade_mode == "local":
            return self.local_account.get_account_features(symbols)
        else:
            return self.alpaca_market.get_account_features(symbols)

            
    def build_observation(self):
        # Retrieve current account snapshot
        account_features = self.get_account_features(self.symbols)
        
        # Market snapshot covers external features; internal metrics come from the account snapshot
        market_features = self.alpaca_market.get_market_features(self.symbols)
        
        return self._build_observation_features(market_features, account_features)

    def step_account(self, order_results):
        if self.trade_mode == "local":
            return self.local_account.step_account(order_results, self.step_count)
        return self.alpaca_market.step_account(order_results, self.symbols)
    
    def reset(self, *, seed=None, options=None):
        self.step_count = 0
        # Local-mode episode reset hooks
        if self.trade_mode == "local":
            # 1) re-randomize account state for current slot symbols
            self.symbols = self.local_account.reset_account(self.init_symbols)
            # 2) refresh market subscriptions (best-effort)
            self.alpaca_market.reset_subscriptions(self.symbols)
                        
        # Build initial observation
        obs = self.build_observation()
        info = {}
        return obs, info
    
        
    def step(self, action: np.ndarray):
        self.step_count += 1
        # Vectorized side mapping as ndarray
        sides = np.where(action == 1, "buy", np.where(action == 2, "sell", "hold"))
        qtys  = np.ones(self.num_envs, dtype=np.int64)  # vector of ones

        order_results = self.alpaca_market.submit_orders(
            self.symbols, sides, qtys, self.trade_mode
        )

        # Reward and done signals (per-slot)
        reward, truncated, terminated = self.step_account(order_results)

        # Build the next observation AFTER partial resets
        observation = self.build_observation()

        # Optional: sanitize obs to avoid NaN/Inf issues during training
        observation = {
            k: (np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0) if hasattr(v, "dtype") else v)
            for k, v in observation.items()
        }
        info = {}
        
        # --- Vector auto-reset semantics (local only) ---
        if self.trade_mode == "local":
            dones = truncated | terminated
            if np.any(dones):
                done_indices = np.flatnonzero(dones)
                self.symbols[done_indices] = self.local_account.reset_account(self.init_symbols, indices=done_indices)
                # Best-effort reconcile (un)subscriptions to the new overall set
                self.alpaca_market.reset_subscriptions(self.symbols)  # existing helper:contentReference[oaicite:1]{index=1}
                done_market_features = self.alpaca_market.get_market_features(self.symbols[done_indices])
                self.local_account.update_account(done_market_features, indices=done_indices)

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
