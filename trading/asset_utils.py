# Codes are structured as (Country, Exchange, Asset Type, Local Symbol)
# to uniquely identify assets around the world.
NUM_COUNTRIES = 1000        # Valid country IDs range from 1–999; 0 is reserved for unknown
NUM_EXCHANGES = 128         # Exchange codes span 1–127 within each country
NUM_ASSET_TYPES = 32        # Asset type identifiers from 1–31
NUM_LOCAL_SYMBOLS = 10_000  # Local symbol identifiers from 1–9,999

PRIMARY_STOCK_SYMBOLS = [
    # --- Mega-cap technology and communication services ---
    "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "META", "TSLA", "NVDA",
    # --- Large-cap financials ---
    "JPM", "BAC", "WFC", "C", "GS", "MS", "V", "MA", "PYPL", "AXP",
    # --- Healthcare and biotechnology ---
    "UNH", "JNJ", "PFE", "MRK", "ABBV", "LLY", "TMO", "CVS", "BMY", "MDT", "ABT",
    # --- Energy ---
    "CVX", "XOM", "COP", "OXY", "SLB", "PXD", "DVN",
    # --- Consumer staples and discretionary ---
    "HD", "LOW", "COST", "WMT", "TGT", "MCD", "SBUX", "KO", "PEP", "PM", "MO",
    "PG", "CL", "KMB", "NKE", "DIS",
    # --- Additional technology and semiconductors ---
    "ORCL", "IBM", "INTC", "AMD", "AVGO", "QCOM", "TXN", "MU", "ADBE",
    # --- Telecommunications and media ---
    "VZ", "T", "TMUS", "CMCSA",
    # --- Industrials and defense ---
    "BA", "LMT", "RTX", "NOC", "GD", "CAT", "DE", "GE", "HON", "MMM", "UPS", "FDX",
    # --- Materials ---
    "LIN", "APD", "NEM", "FCX", "AA", "NUE",
    # --- Utilities ---
    "NEE", "DUK", "SO", "AEP", "EXC", "XEL", "SRE", "PEG",
    # --- Real estate and specialty REITs ---
    "PLD", "AMT", "CCI", "SPG",
    # --- Automakers, restaurants, and retailers ---
    "GM", "F", "CMG", "TJX", "DPZ",
]

PRIMARY_CRYPTO_SYMBOLS = [
    "BTC/USD",
    "ETH/USD",
    "LTC/USD",
    "BCH/USD",
    "SOL/USD",
    "ADA/USD",
    "DOGE/USD",
    "MATIC/USD",
    "DOT/USD",
    "AVAX/USD",
]

# Map ISO country codes to numeric IDs (0 reserved for unknown).
COUNTRY_MAP = {"US": 1, "KR": 82}
# Reverse lookup example: COUNTRY_CODE_MAP = {v: k for k, v in COUNTRY_MAP.items()}

# Map country and exchange codes to exchange IDs.
EXCHANGE_MAP = {
    COUNTRY_MAP["US"]: {  # United States
        "XNYS": 1, "XNAS": 2, "ARCX": 3, "XASE": 4, "XCME": 5, "XCBF": 6, "XCEC": 7
    },
    COUNTRY_MAP["KR"]: {  # South Korea
        "XKRX": 1, "XKOS": 2, "XKNX": 3
    }
}

def EXCHANGE_ID(country_code: str, exchange_code: str) -> int:
    """Return the numeric exchange ID for a given country and exchange code."""
    return EXCHANGE_MAP[COUNTRY_MAP[country_code]][exchange_code]

# Map asset type strings to integer identifiers.
ASSET_TYPE_MAP = {
    "UNK": 0,
    "ESXXXX": 1,
    "EPXXXX": 2,
    "EDXXXX": 3,
    "EFXXXX": 4,
    "ECXXXX": 5,
    "EMXXXX": 6,
    "FXXXXX": 11,
    "Future/Interest Rate": 12,
    "Future/Commodity": 13,
    "Future/Currency": 14,
    "Option/Equity": 15,
    "Option/Index": 16,
    "Crypto/Spot": 20,
    "Crypto/Stablecoin": 21,
    "Crypto/Derivatives": 22
}

def SYMBOL_TYPE(country_code: str, exchange_code: str, asset_code: str) -> tuple[int, int, int]:
    """Convert string codes to their corresponding numeric identifiers."""
    country_id = COUNTRY_MAP[country_code]
    return country_id, EXCHANGE_MAP[country_id][exchange_code], ASSET_TYPE_MAP[asset_code]

# Global mapping of (country, exchange, asset type) to local symbol IDs.
WORLD_ASSET_MAP = {
    # United States, NASDAQ, common stock
    SYMBOL_TYPE("US", "XNYS", "ESXXXX"): {
            "AAPL": 1,
            "MSFT": 2,
            "AMZN": 3,
            "GOOGL": 4,
            "GOOG": 5,
            "META": 6,
            "TSLA": 7,
            "NVDA": 8,
            # more...
    },
    # United States, NASDAQ, exchange-traded funds
    SYMBOL_TYPE("US", "XNYS", "EFXXXX"): {
            "SPY": 1,
            "IVV": 2,
            "VOO": 3,
            # more...
    },
    # more..
}

def SYMBOL_ID(country_code: str, exchange_code: str, asset_code: str, symbol_code: str) -> int | None:
    """Return the local symbol ID for the specified asset components."""
    return WORLD_ASSET_MAP.get(SYMBOL_TYPE(country_code, exchange_code, asset_code), {}).get(symbol_code)
