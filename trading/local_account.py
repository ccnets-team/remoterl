# Local account portfolio manager used for simulated trading.
from __future__ import annotations
import numpy as np
from typing import Iterable, Dict, Any

class LocalAccount:
    """
    Manage portfolio and account state when running in local simulation mode.

    All positions, cash balances, and average entry prices are tracked here.
    The trading environment merely reads this state and routes orders to the
    appropriate slot.
    """
    def __init__(
        self,
        num_envs: int,
        num_stocks_range=(0, 100),
        budget_range=(100.0, 10_000.0),
        max_step_range=(1_000, 10_000),
    ):
        self.rng: np.random.Generator = np.random.default_rng()
        self.num_envs = int(num_envs)

        # Configuration ranges
        assert num_stocks_range[0] <= num_stocks_range[1]
        assert budget_range[0] <= budget_range[1]
        assert max_step_range[0] <= max_step_range[1]
        self.num_stocks_range = tuple(num_stocks_range)
        self.budget_range = tuple(budget_range)
        self.max_step_range = tuple(max_step_range)


        # State arrays; each index corresponds to the same symbol across arrays
        self.cash = np.zeros(self.num_envs, dtype=np.float64)
        self.position_qty = np.zeros(self.num_envs, dtype=np.int64)
        self.avg_entry_price = np.zeros(self.num_envs, dtype=np.float64)
        self.prev_nav = np.zeros(self.num_envs, dtype=np.float64)
        self.max_steps = np.zeros(self.num_envs, dtype=np.int64)

        # Derived state variables
        self.unrealized_pnl = np.zeros(self.num_envs, dtype=np.float64)
        self.exposure = np.zeros(self.num_envs, dtype=np.float64)
        self.asset_nav = np.zeros(self.num_envs, dtype=np.float64)

        # ---- Initialize cash and NAV for local simulation ----
        # Seed each environment’s cash within the budget range so the first trade
        # does not immediately exhaust the account.
        low, high = self.budget_range
        if high <= low:
            high = low
        self.cash[:] = self.rng.uniform(low, high, size=self.num_envs)
        # With zero positions initially, net asset value equals cash
        self.prev_nav[:] = self.cash
        self.asset_nav[:] = self.prev_nav  # per-slot NAV = cash at start


    # ---- Snapshot helpers used by TradingVecEnv ----
    def get_account_features(self, symbols) -> np.ndarray:
        """
        Return (N,6) float32 array aligned with `symbols`.
        Columns match TradingVecEnv.ACCOUNT_FEATURE_KEYS order:
        [position_qty, cash, avg_entry_price, unrealized_pnl, exposure, asset_nav]
        """
        import numpy as np
        # Ensure 1D unique order is NOT forced here; keep exact lane order
        n = len(symbols)
        out = np.zeros((n, 6), dtype=np.float32)

        # Vectorized fill
        out[:, 0] = self.position_qty[:n]
        out[:, 1] = self.cash[:n]
        out[:, 2] = self.avg_entry_price[:n]
        out[:, 3] = self.unrealized_pnl[:n]
        out[:, 4] = self.exposure[:n]
        out[:, 5] = self.asset_nav[:n]
        return out


    # ---------- Apply actions under local fill rules ---------- #
    def apply_actions(self, actions, prices) -> np.ndarray:
        """
        Parameters
        ----------
        actions : array-like, shape (N,)
            Per-slot action where 0=hold, 1=buy, 2=sell.
        prices : array-like, shape (N,)
            Current market price for each slot.

        Applying rules
        ----------------
        - **buy**: increase quantity by one, decrease cash by price, and update
          ``avg_entry_price`` using a weighted average.
        - **sell**: if quantity ≥ 1, decrease quantity by one and add the price
          to cash; reset ``avg_entry_price`` when quantity reaches zero.

        Returns
        -------
        np.ndarray
            Reward computed as the change in net asset value (NAV).
        """
        actions = np.asarray(actions, dtype=np.int64)
        prices  = np.asarray(prices,  dtype=np.float64)

        buy  = actions == 1
        sell = (actions == 2) & (self.position_qty >= 1)

        # Preserve current quantities before applying changes
        old_qty = self.position_qty.copy()
        new_qty = old_qty + buy.astype(np.int64) - sell.astype(np.int64)

        # Update cash balances
        self.cash -= prices * buy
        self.cash += prices * sell

        # Commit updated quantities
        self.position_qty[:] = new_qty

        # Update average entry price for positions increased via buys
        inc_mask = buy
        with np.errstate(divide="ignore", invalid="ignore"):
            self.avg_entry_price[inc_mask] = (
                (self.avg_entry_price[inc_mask] * old_qty[inc_mask] + prices[inc_mask]) /
                (old_qty[inc_mask] + 1.0)
            )

        # Clear average price when the position is fully liquidated
        zero_mask = self.position_qty == 0
        self.avg_entry_price[zero_mask] = 0.0

        # Recompute derived variables in order: exposure → unrealized_pnl → NAV → reward → group NAV
        self.exposure[:] = self.position_qty * prices
        self.unrealized_pnl[:] = (prices - self.avg_entry_price) * self.position_qty

        nav = self.cash + self.exposure
        reward = (nav - self.prev_nav).astype(np.float32)
        self.prev_nav[:] = nav

        self.asset_nav[:] = nav  # per-slot NAV
        
        return reward

    # ------- episode reset for local simulation ------- #
    def reset_account(self, symbols: np.ndarray, indices: np.ndarray = None) -> np.ndarray:
        """
        Reset account state.

        - If indices is None: full reset across all lanes (current behavior).
        - If indices is provided: reset only those lanes and return K new symbols
        (dtype=object) aligned with the provided indices.
        """
        # Target lanes to reset
        target = np.arange(self.num_envs, dtype=int) if indices is None \
                else np.asarray(indices, dtype=int).reshape(-1)
        k = int(target.size)
        if k == 0:
            return np.asarray([], dtype=object)

        # Keep a scalar max_steps design: re-sample only on full reset
        if indices is None:
            self.max_steps[:] = self.rng.integers(self.max_step_range[0], self.max_step_range[1]+1, size=self.num_envs)
        else:
            self.max_steps[target] = self.rng.integers(self.max_step_range[0], self.max_step_range[1]+1, size=k)

        # Sample fresh cash for target lanes
        self.cash[target] = self.rng.uniform(self.budget_range[0], self.budget_range[1], size=k)

        # Sample initial position quantities for target lanes
        self.position_qty[target] = self.rng.integers(self.num_stocks_range[0], self.num_stocks_range[1] + 1, size=k)

        # Reset derived state for target lanes
        self.avg_entry_price[target] = 0.0
        self.exposure[target] = 0.0
        self.unrealized_pnl[target] = 0.0
        self.prev_nav[target] = self.cash[target]
        self.asset_nav[target] = self.prev_nav[target]

        # Choose K symbols (allow replacement if K > len(pool))
        chosen = np.asarray(self.rng.choice(symbols, size=k, replace=(k > len(symbols))), dtype=object)
        return chosen


    def update_account(self, market_features: np.ndarray, indices: np.ndarray):
        """
        Update exposure, PnL, and NAV for the given lanes using market features.

        Parameters
        ----------
        market_features : np.ndarray, shape (K, 6)
            Rows are [o, h, l, c, v, t]; only 'c' is used here.
        indices : np.ndarray, shape (K,)
            Target lane indices aligned with market_features rows.
        """
        import numpy as np
        idx = np.asarray(indices, dtype=int).reshape(-1)
        assert market_features.shape[0] == idx.size, "Market features must match indices"

        prices = market_features[:, 3].astype(np.float64)  # close column
        self.exposure[idx] = self.position_qty[idx] * prices
        self.unrealized_pnl[idx] = (prices - self.avg_entry_price[idx]) * self.position_qty[idx]
        self.asset_nav[idx] = self.cash[idx] + self.exposure[idx]
        # Sync prev_nav so the next reward = ΔNAV doesn't spike on reset frames
        self.prev_nav[idx] = self.asset_nav[idx]


    # ---------- Query ---------- #
    def step_account(self, order_results: Iterable[Dict[str, Any]], step_count: int):
        """
        Parameters
        ----------
        order_results : list[dict]
            Order results as returned from ``submit_orders``.
        step_count : int
            Current environment step index.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            ``(reward, truncated, terminated)`` where:
            - ``reward`` is the change in NAV for each environment slot.
            - ``truncated`` is True when ``max_steps`` is exceeded.
            - ``terminated`` is True when cash falls below zero or a position becomes invalid.
        """
        # Extract filled average prices from the order results
        prices = np.array(
            [o.get("filled_avg_price", 0.0) for o in order_results],
            dtype=np.float64
        )

        # Default to hold (0) when the action is unspecified
        actions = np.array(
            [o.get("action", 0) for o in order_results],
            dtype=np.int64
        )

        # Compute reward as the change in NAV
        reward = self.apply_actions(actions, prices)

        # Determine termination conditions
        terminated = (self.cash <= 0)
        truncated = np.full(self.num_envs, step_count >= self.max_steps, dtype=bool)

        return reward, truncated, terminated
