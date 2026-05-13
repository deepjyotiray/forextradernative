NAME = "M15_ZONE_SCALP"

DEFAULTS = {
    "enabled": True,
    "risk_pct": 0.5,
    "fixed_lot": None,
    "max_active_trades": 1,
    "max_trades_day": 0,
    "spread_max": 0.5,
    "pip_size": 0.1,
    "tp_pips": 25,
    "sl_buffer_atr_mult": 0.35,
    "sl_floor_pips": 12,
    "sl_ceiling_pips": 35,
    "zone_touch_atr_mult": 0.25,
    "zone_touch_floor_pts": 0.8,
    "min_zone_strength": 0.15,
    "require_htf_bias": True,
    "bias_fallback_min_confidence": 0.35,
    "rejection_wick_ratio": 0.32,
    "displacement_body_ratio": 0.52,
    "close_near_extreme_ratio": 0.62,
    "sessions": ["LONDON", "NEW_YORK"],
    "news_block": True,
    "zone_detector_eps": 1.2,
    "zone_detector_min_samples": 2,
    "zone_detector_max_width": 5.0,
    "notes": "M15 scalp: bid/ask at DBSCAN zone, TP 20–30 pips typical (pip_size 0.1 for XAUUSD).",
}

LOCKED = {}
DISABLED = {}
CFG_MAP = {}
