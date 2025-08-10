# local_account.py
from __future__ import annotations
import numpy as np

class LocalAccount:
    """
    Portfolio/account state manager for **local mode**.
    - Holds all state such as positions, cash, and average entry price here
    - The environment only reads state and delegates orders (common component)
    """
    def __init__(
        self,
        num_envs: int,
        symbols: list[str],
        rng,
        num_stocks_range=(0, 100),
        budget_range=(100.0, 10_000.0),
        max_steps=1_000,
    ):
        self.rng = rng
        self.num_envs = int(num_envs)
        self.symbols: list[str] = list(symbols)  # fixed slot order

        # Parameters
        self.num_stocks_range = tuple(num_stocks_range)
        self.budget_range = tuple(budget_range)
        self.max_steps = int(max_steps)

        # State (slot alignment matches the order of ``symbols``)
        self.cash = np.zeros(self.num_envs, dtype=np.float64)
        self.position_qty = np.zeros(self.num_envs, dtype=np.int64)
        self.avg_entry_price = np.zeros(self.num_envs, dtype=np.float64)
        self.net_worth_prev = np.zeros(self.num_envs, dtype=np.float64)

        # Newly added internal state variables
        self.unrealized_pnl = np.zeros(self.num_envs, dtype=np.float64)
        self.exposure = np.zeros(self.num_envs, dtype=np.float64)
        self.group_NAV = np.zeros(self.num_envs, dtype=np.float64)

    # ---------- Episode reset (local only) ---------- #
    def get_account_data(self) -> dict:
        """Return a snapshot of the current account state without modifying it."""
        return {
            "position_qty": self.position_qty.copy(),
            "cash": self.cash.copy(),
            "avg_entry_price": self.avg_entry_price.copy(),
            "net_worth_prev": self.net_worth_prev.copy(),
            "unrealized_pnl": getattr(self, "unrealized_pnl", np.zeros_like(self.cash)),
            "exposure": getattr(self, "exposure", np.zeros_like(self.cash)),
            "group_NAV": getattr(self, "group_NAV", np.zeros_like(self.cash)),
            "symbols": list(self.symbols),
        }

        
    # ---------- Apply actions (local fill rules) ---------- #
    def apply_actions(self, actions, prices) -> np.ndarray:
        """
        actions: (N,) {0: hold, 1: buy, 2: sell}
        prices : (N,) current price
        Execution rules (simple):
        - buy  : qty += 1, cash -= price, ``avg_entry_price`` becomes weighted average
        - sell : only if qty >= 1 → qty -= 1, cash += price (reset avg price to 0 when qty==0)
        Reward: NAV_t - NAV_{t-1}
        """
        actions = np.asarray(actions, dtype=np.int64)
        prices  = np.asarray(prices,  dtype=np.float64)

        buy  = actions == 1
        sell = (actions == 2) & (self.position_qty >= 1)

        # Backup quantity before applying buy/sell
        old_qty = self.position_qty.copy()
        new_qty = old_qty + buy.astype(np.int64) - sell.astype(np.int64)

        # Cash movement
        self.cash -= prices * buy
        self.cash += prices * sell

        # Apply quantity
        self.position_qty[:] = new_qty

        # Update average entry price (only buy)
        inc_mask = buy
        with np.errstate(divide="ignore", invalid="ignore"):
            self.avg_entry_price[inc_mask] = (
                (self.avg_entry_price[inc_mask] * old_qty[inc_mask] + prices[inc_mask]) /
                (old_qty[inc_mask] + 1.0)
            )

        # Reset average price when position is fully sold
        zero_mask = self.position_qty == 0
        self.avg_entry_price[zero_mask] = 0.0

        # Update derived internal variables (order: exposure → unrealized_pnl → NAV/reward → group_NAV)
        self.exposure[:] = self.position_qty * prices
        self.unrealized_pnl[:] = (prices - self.avg_entry_price) * self.position_qty

        nav = self.cash + self.exposure
        reward = (nav - self.net_worth_prev).astype(np.float32)
        self.net_worth_prev[:] = nav

        self.group_NAV[:] = float(np.mean(nav))  # broadcast

        return reward

    # ---------- Query ---------- #
    def update_account(self, order_results, step_idx: int):
        """
        order_results: list[dict] - order results returned from ``submit_orders()``
        step_idx: current environment step index
        return: (reward, truncated, terminated)
            reward: np.ndarray (num_envs,) - NAV change
            truncated: np.ndarray (num_envs,) - whether ``max_steps`` exceeded
            terminated: np.ndarray (num_envs,) - lack of cash or invalid position
        """
        # Extract only filled_avg_price from order_results
        prices = np.array(
            [o.get("filled_avg_price", 0.0) for o in order_results],
            dtype=np.float64
        )

        # Default to hold(0) when order_results lack action info
        actions = np.array(
            [o.get("action", 0) for o in order_results],
            dtype=np.int64
        )

        # Calculate reward (NAV change)
        reward = self.apply_actions(actions, prices)

        # Termination conditions
        terminated = (self.cash <= 0)
        truncated = np.full(self.num_envs, step_idx >= self.max_steps, dtype=bool)


        return reward, truncated, terminated
