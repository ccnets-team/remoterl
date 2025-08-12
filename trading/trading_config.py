# Configuration settings for trading backends.
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional

TradeMode = Literal["local", "paper", "real"]
Broker    = Literal["alpaca", "binance", "ibkr"]

BROKER_DEFAULTS = {
    "alpaca": dict(
        market_ws_url="wss://stream.data.alpaca.markets/v2/iex",
        trades_ws_url="wss://paper-api.alpaca.markets/stream",
        paper_rest_base="https://paper-api.alpaca.markets/v2",
        live_rest_base="https://api.alpaca.markets/v2",
        data_rest_base=None,  # set in __post_init__ based on asset_type
    ),
    "binance": dict(
        market_ws_url="wss://stream.binance.com:9443/ws",
        trades_ws_url="wss://stream.binance.com:9443/ws",
        paper_rest_base="https://testnet.binance.vision/api",
        live_rest_base="https://api.binance.com/api",
        data_rest_base="https://api.binance.com/api",
    ),
    "ibkr": dict(
        market_ws_url="wss://your-ibkr-ws",
        trades_ws_url="wss://your-ibkr-trades",
        paper_rest_base="https://your-ibkr-paper-rest",
        live_rest_base="https://your-ibkr-live-rest",
        data_rest_base=None,
    ),
}

@dataclass
class TradingConfig:
    """Container for API credentials and endpoint configuration."""
    broker: Broker
    api_key: str
    secret_key: str
    trade_mode: TradeMode = "local"

    # Asset identity (for obs encoding):
    country_code: str = "US"
    exchange_code: str = "XNYS"
    asset_type:   str = "ESXXXX"  # e.g., "ESXXXX" stock, "Crypto/Spot", etc.

    # Endpoints; auto-filled from BROKER_DEFAULTS if left blank
    market_ws_url: str = ""
    trades_ws_url: str = ""
    paper_rest_base: str = ""
    live_rest_base: str  = ""
    data_rest_base: Optional[str] = None  # can be set/overridden here

    # Rate limits / timeouts
    recv_timeout_sec: float = 1.0
    rest_rps: float = 5.0
    rest_burst: int = 10
    ws_pull_rps: float = 20.0
    ws_pull_burst: int = 50

    # Optional bars timeframe for REST backfill
    bars_timeframe: str = "1Min"

    def __post_init__(self):
        if not self.broker:
            raise ValueError("broker is required")
        if not self.api_key:
            raise ValueError("api_key is required")
        if not self.secret_key:
            raise ValueError("secret_key is required")

        # Base defaults
        defaults = BROKER_DEFAULTS.get(self.broker.lower(), {})
        for field, value in defaults.items():
            if not getattr(self, field):
                setattr(self, field, value)

        # Smart default for data_rest_base when using Alpaca
        if self.broker.lower() == "alpaca" and not self.data_rest_base:
            # Crypto uses v1beta3; stocks use v2
            if str(self.asset_type).lower().startswith("crypto"):
                # scope "us" aligns with Alpaca’s public crypto routes
                self.data_rest_base = "https://data.alpaca.markets/v1beta3/crypto/us"
            else:
                self.data_rest_base = "https://data.alpaca.markets/v2/stocks"
