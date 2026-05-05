NAME = "SMC_CONFLUENCE"

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
    # SMC-specific
    "spread_mean_max": 0.6,
    "spread_vol_max": 0.15,
    "spread_percentile_max": 1.0,
    "spread_delta_max": 0.1,
    "threshold_with_trend": 0.7,
    "threshold_counter": 0.6,
    "threshold_counter_max": 0.35,
    "min_rr": 1.5,
    "ema_slope_min": 0.08,
    "ltf_tick_conflict_long_max": 0.3,
    "ltf_tick_conflict_short_min": 0.7,
    # Gate overrides
    "override_spread_mean": False,
    "override_spread_vol": False,
    "override_spread_percentile": False,
    "override_spread_widening": False,
    "override_execution_quality": False,
}

LOCKED = {}
DISABLED = {}

CFG_MAP = {
    "risk_pct": "SWING_RISK_PCT",
    "spread_mean_max": "SMC_SPREAD_MEAN_MAX",
    "spread_vol_max": "SMC_SPREAD_STD_MAX",
    "spread_percentile_max": "SMC_SPREAD_PERCENTILE_MAX",
    "spread_delta_max": "SMC_CURRENT_SPREAD_DELTA_MAX",
    "threshold_with_trend": "SMC_THRESHOLD_WITH_TREND",
    "threshold_counter": "SMC_THRESHOLD_COUNTER",
    "threshold_counter_max": "SMC_THRESHOLD_COUNTER_MAX",
    "min_rr": "SMC_MIN_RR",
    "ema_slope_min": "SMC_TIMEFRAME_EMA_SLOPE_MIN",
    "ltf_tick_conflict_long_max": "SMC_LTF_TICK_CONFLICT_LONG_MAX",
    "ltf_tick_conflict_short_min": "SMC_LTF_TICK_CONFLICT_SHORT_MIN",
    "override_spread_mean": "SPREAD_MEAN_GATE_OVERRIDE_ENABLED",
    "override_spread_vol": "SPREAD_VOLATILITY_GATE_OVERRIDE_ENABLED",
    "override_spread_percentile": "SPREAD_PERCENTILE_GATE_OVERRIDE_ENABLED",
    "override_spread_widening": "SPREAD_DELTA_GATE_OVERRIDE_ENABLED",
    "override_execution_quality": "EXECUTION_QUALITY_GATE_OVERRIDE_ENABLED",
}
