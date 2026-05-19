NAME = "SWING_ENGINE"

DEFAULTS = {
    # Base
    "enabled": True,
    "risk_pct": 0.5,
    "fixed_lot": None,
    "max_active_trades": 1,
    "max_trades_day": 5,
    "spread_max": 0.7,
    "min_confidence_pct": 80,
    "sl_min": 10.0,
    "sl_max": 25.0,
    "tp1_r": 0.25,
    "tp2_r": 0.35,
    "tp3_r": 0.5,
    "sessions": ["LONDON", "NEW_YORK"],
    "news_block": False,
    "cooldown_seconds": 120,
    "consecutive_loss_limit": 3,
    "loss_pause_seconds": 900,
    "notes": "",
}

# Shown in UI but not editable — spec-mandated
LOCKED = {
    "max_active_trades": "Spec: max 1 swing trade at a time",
}

# Not applicable — hidden in UI, null in JSON, ignored in code
DISABLED = {}

# Maps strategy config keys → cfg.* attribute names
CFG_MAP = {
    "risk_pct": "SWING_RISK_PCT",
}
