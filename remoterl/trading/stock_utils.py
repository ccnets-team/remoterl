# (Country, Exchange, Asset Type, Local Symbol) <- Represent all global assets
NUM_COUNTRIES = 1000        # 1~999 global country codes like international calls
NUM_EXCHANGES = 128         # 1~127 exchange codes within each country
NUM_ASSET_TYPES = 32        # 1~31 asset type codes
NUM_LOCAL_SYMBOLS = 10_000  # 1~9999 local symbol codes of assets

PRIMARY_STOCK_SYMBOLS = [
    # --- Mega-cap tech & communication services ---
    "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "META", "TSLA", "NVDA",
    # --- Large-cap financials ---
    "JPM", "BAC", "WFC", "C", "GS", "MS", "V", "MA", "PYPL", "AXP",
    # --- Healthcare & biotech ---
    "UNH", "JNJ", "PFE", "MRK", "ABBV", "LLY", "TMO", "CVS", "BMY", "MDT", "ABT",
    # --- Energy ---
    "CVX", "XOM", "COP", "OXY", "SLB", "PXD", "DVN",
    # --- Consumer staples & discretionary ---
    "HD", "LOW", "COST", "WMT", "TGT", "MCD", "SBUX", "KO", "PEP", "PM", "MO",
    "PG", "CL", "KMB", "NKE", "DIS",
    # --- Additional tech & semis ---
    "ORCL", "IBM", "INTC", "AMD", "AVGO", "QCOM", "TXN", "MU", "ADBE",
    # --- Telecom & media ---
    "VZ", "T", "TMUS", "CMCSA",
    # --- Industrials & defense ---
    "BA", "LMT", "RTX", "NOC", "GD", "CAT", "DE", "GE", "HON", "MMM", "UPS", "FDX",
    # --- Materials ---
    "LIN", "APD", "NEM", "FCX", "AA", "NUE",
    # --- Utilities ---
    "NEE", "DUK", "SO", "AEP", "EXC", "XEL", "SRE", "PEG",
    # --- Real-estate / specialty REITs ---
    "PLD", "AMT", "CCI", "SPG",
    # --- Autos, restaurants, retailers ---
    "GM", "F", "CMG", "TJX", "DPZ",
]

# Country ID mapping(0=UNK, US=1, KR=82)
COUNTRY_MAP = {"US": 1, "KR": 82}
# COUNTRY_CODE_MAP = {v: k for k, v in COUNTRY_MAP.items()}

# Exchange ID mapping
EXCHANGE_MAP = {
    COUNTRY_MAP["US"]: {  # US
        "XNYS": 1, "XNAS": 2, "ARCX": 3, "XASE": 4, "XCME": 5, "XCBF": 6, "XCEC": 7
    },
    COUNTRY_MAP["KR"]: {  # KR
        "XKRX": 1, "XKOS": 2, "XKNX": 3
    }
}

def EXCHANGE_ID(country_code: str, exchange_code: str) -> int:
    return EXCHANGE_MAP[COUNTRY_MAP[country_code]][exchange_code]

# Asset type mapping
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
    country_id = COUNTRY_MAP[country_code]
    return country_id, EXCHANGE_MAP[country_id][exchange_code], ASSET_TYPE_MAP[asset_code]

# World Total Asset Map
WORLD_ASSET_MAP = {
    # US, NASDAQ, Common Stock
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
    # US, NASDAQ, ETF
    SYMBOL_TYPE("US", "XNYS", "EFXXXX"): {
            "SPY": 1,
            "IVV": 2,
            "VOO": 3,
            # more...
    },
    # more..
}

def SYMBOL_ID(country_code: str, exchange_code: str, asset_code: str, symbol_code: str) -> int | None:
    return WORLD_ASSET_MAP.get(SYMBOL_TYPE(country_code, exchange_code, asset_code), {}).get(symbol_code)
