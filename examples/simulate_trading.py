"""
Launch a RemoteRL stock-trading simulator.

This script hosts a **vectorized trading environment** that connects to live or
simulated market data from the Alpaca API and streams it to any connected
trainer process.

Run this script *before* starting a trainer job (e.g., ``train_sb3_stock.py`` or
``train_rllib_stock.py``) so the trainer can attach to the running market
environment.

Features
--------
- Supports ``local``, ``paper``, or ``real`` trading modes via Alpaca.
- Streams live OHLCV data and account state for a configurable set of symbols.
- Fully compatible with RemoteRL's distributed RL workflow.

Requirements
------------
- RemoteRL account and API key (``REMOTERL_API_KEY`` environment variable).
- Alpaca API key and secret (``ALPACA_API_KEY`` and ``ALPACA_SECRET_KEY`` variables).
- Python packages: ``remoterl``, ``gymnasium``, ``websocket-client``, ``requests``.

Obtain a RemoteRL API key at https://remoterl.com/user/dashboard
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import remoterl
import gymnasium as gym
from trading.trading_vec_env import TradingVecEnv
from trading.trading_market import TradingMarket
from trading.trading_config import TradingConfig
from trading.asset_utils import PRIMARY_STOCK_SYMBOLS

REMOTERL_API_KEY = os.getenv("REMOTERL_API_KEY")
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

def main() -> None:
    """Open a RemoteRL connection and serve the trading environment."""
    broker_config = TradingConfig(
        broker="alpaca",
        api_key=ALPACA_API_KEY,
        secret_key=ALPACA_SECRET_KEY,
        trade_mode="local",  # "local", "paper", or "real"
    )
    alpaca_client = TradingMarket(
        trading_config=broker_config,
        symbols=PRIMARY_STOCK_SYMBOLS,
    )
    gym.register(
        id="stock-trading-vec-env",
        vector_entry_point=lambda num_envs, **kwargs: TradingVecEnv(num_envs, alpaca_market=alpaca_client, **kwargs)
    )

    try:
        # 1) Connect to the RemoteRL service using your API key
        remoterl.init(api_key=REMOTERL_API_KEY, role="simulator", max_env_runners=1)  # initialization blocks until a trainer connects
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
