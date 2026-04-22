"""
Config — shared + per-strategy parameters.
"""

# MT5 connection
MT5_PATH = None
MT5_LOGIN = None
MT5_PASSWORD = None
MT5_SERVER = None

# Trading
SYMBOL = "XAUUSD"
MAGIC_NUMBER = 234000
DEVIATION = 20
PIP_VALUE_PER_LOT = 100

# Risk (defaults — strategies can override via their own params)
MAX_POSITIONS = 4
MAX_RISK_PCT = 0.5
MAX_DRAWDOWN_PCT = 5.0
MAX_LOT = 1.0
MIN_LOT = 0.01
DAILY_TARGET_DOLLARS = 100.0
DAILY_LOSS_LIMIT_PCT = 2.0
MAX_CONSECUTIVE_LOSSES = 4
MIN_TRADE_COOLDOWN = 10.0
LOSS_STREAK_PAUSE = 900

# Session (UTC)
ASIAN_START = 0
LONDON_START = 7
NY_START = 13
SESSION_BLOCK_MINUTES = 3

# Data
MAX_CANDLES = 1000  # ~3 days M5, ~16 hours M1 — recent data only

# API
API_HOST = "127.0.0.1"
API_PORT = 8899

# Default strategy on startup
DEFAULT_STRATEGY = "AUTO"

# Tier 1 upgrades toggle
TIER1_ENABLED = True
