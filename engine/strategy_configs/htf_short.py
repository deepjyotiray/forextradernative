NAME = "HTF_SHORT"

DEFAULTS = {
    "enabled": True,
    "risk_pct": 1.0,
    "fixed_lot": None,
    "max_active_trades": 1,
    "max_trades_day": 2,
    "spread_max": 0.80,
    "sl_min": 15.0,
    "sl_max": 30.0,
    "min_rr": 1.8,
    "sessions": ["LONDON", "NEW_YORK", "ASIAN"],
    "news_block": True,
    "cooldown_seconds": 3600,
    "consecutive_loss_limit": 2,
    "loss_pause_seconds": 3600,
    "notes": "Higher-timeframe bias engine. SELL only. Macro auto-fetched from Yahoo Finance.",
}

LOCKED = {
    "max_active_trades": "HTF strategy holds one position at a time",
}

DISABLED = {}

CFG_MAP = {
    "enabled":    "HTF_SHORT_ENABLED",
    "spread_max": "HTF_SHORT_MAX_SPREAD",
    "min_rr":     "HTF_SHORT_MIN_RR",
}
