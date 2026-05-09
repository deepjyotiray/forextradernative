NAME = "INTRADAY_ENGINE"

DEFAULTS = {
    # Base
    "enabled": True,
    "risk_pct": 0.5,
    "fixed_lot": None,
    "max_active_trades": 5,
    "max_trades_day": 5,
    "spread_max": 0.25,
    "m5_volume_ratio_min": 1.2,
    "min_confidence_pct": 80,
    "sl_min": 1.5,
    "sl_max": 5.0,
    "tp1_r": 1.5,
    "tp2_r": 2.0,
    "tp3_r": None,          # not applicable — intraday exits at 2R
    "sessions": ["ASIAN", "LONDON", "NEW_YORK"],
    "news_block": True,
    "cooldown_seconds": 120,
    "consecutive_loss_limit": 3,
    "loss_pause_seconds": 900,
    "notes": "",
}

LOCKED = {
    "news_block": "Spec: always block on high-impact news",
}

DISABLED = {
    "tp3_r": "Intraday exits at 2R — no TP3",
}

CFG_MAP = {
    "risk_pct": "INTRADAY_RISK_PCT",
}
