# alpaca_config.py
# pip install websocket-client requests
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

TradeMode = Literal["local", "paper", "real"]

# ------------------------------ Client ------------------------------ #
@dataclass
class AlpacaConfig:
    api_key: str
    secret_key: str
    trade_mode: TradeMode = "paper"  # "local" | "paper" | "real"

    country_code: str = "US"  # "US" | "KR"
    exchange_code: str = "XNYS"  # "XNYS" | "XNAS" | "ARCX" | "XASE" | "XCME" | "XCBF" | "XCEC"
    asset_type: str = "ESXXXX"  # "ESXXXX" | "EPXXXX" | "EDXXXX" | "EFXXXX" | "ECXXXX" | "EMXXXX" | "FXXXXX"

    # WS
    market_ws_url: str = "wss://stream.data.alpaca.markets/v2/iex"   # market data
    trades_ws_url: str = "wss://paper-api.alpaca.markets/stream"     # trade/account updates (example)

    # REST
    paper_rest_base: str = "https://paper-api.alpaca.markets/v2"
    live_rest_base: str  = "https://api.alpaca.markets/v2"

    # Features
    recv_timeout_sec: float = 1.0
    # Rate limiting sample: tuned for 5 calls/sec (~300 per minute)
    rest_rps: float = 5.0            # REST calls per second
    rest_burst: int = 10
    ws_pull_rps: float = 20.0        # WS messages processed per second
    ws_pull_burst: int = 50

