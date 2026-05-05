NAME = "M15_SUPPORT_RESISTANCE_REJECTION_V1"

DEFAULTS = {
    # Base
    "enabled": True,
    "risk_pct": 1.0,
    "fixed_lot": None,
    "max_active_trades": 2,
    "max_trades_day": 4,
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
    # M15 SR-specific
    "lookback_hours": 2,
    "min_touches": 2,
    "zone_atr_mult": 0.25,
    "entry_buffer_atr": 0.1,
    "sl_buffer_atr": 0.35,
    "min_rr": 1.5,
    "require_candle_confirmation": True,
    "spike_zone_bypass": False,
}

LOCKED = {}
DISABLED = {}

CFG_MAP = {
    "risk_pct": "SWING_RISK_PCT",
    "spread_max": "M15_SR_MAX_SPREAD",
    "max_trades_day": "M15_SR_MAX_TRADES_PER_DAY",
    "lookback_hours": "M15_SR_LOOKBACK_HOURS",
    "min_touches": "M15_SR_MIN_TOUCHES",
    "zone_atr_mult": "M15_SR_ZONE_ATR_MULT",
    "entry_buffer_atr": "M15_SR_ENTRY_BUFFER_ATR",
    "sl_buffer_atr": "M15_SR_SL_BUFFER_ATR",
    "min_rr": "M15_SR_MIN_RR",
    "require_candle_confirmation": "M15_SR_REQUIRE_CANDLE_CONFIRMATION",
    "spike_zone_bypass": "M15_SR_SPIKE_ZONE_BYPASS",
}
