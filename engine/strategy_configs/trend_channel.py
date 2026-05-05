NAME = "TREND_CHANNEL"

DEFAULTS = {
    # Base
    "enabled": True,
    "risk_pct": 1.0,
    "fixed_lot": None,
    "max_active_trades": 2,
    "max_trades_day": 5,
    "spread_max": 0.6,
    "sl_min": 8.0,
    "sl_max": 20.0,
    "tp1_r": 1.5,
    "tp2_r": 2.5,
    "tp3_r": 3.0,
    "sessions": ["LONDON", "NEW_YORK"],
    "news_block": False,
    "cooldown_seconds": 120,
    "consecutive_loss_limit": 3,
    "loss_pause_seconds": 900,
    "notes": "",
    # Trend-specific
    "st_period": 10,
    "st_multiplier": 3.0,
    "min_rr": 1.5,
    "require_h1_align": True,
}

LOCKED = {}
DISABLED = {}

CFG_MAP = {
    "risk_pct": "SWING_RISK_PCT",
    "spread_max": "TREND_CHANNEL_MAX_SPREAD",
    "min_rr": "TREND_CHANNEL_MIN_RR",
    "st_period": "TREND_CHANNEL_ST_PERIOD",
    "st_multiplier": "TREND_CHANNEL_ST_MULTIPLIER",
    "require_h1_align": "TREND_CHANNEL_REQUIRE_H1_ALIGN",
}
