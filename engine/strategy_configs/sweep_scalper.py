NAME = "SWEEP_SCALPER"

DEFAULTS = {
    # Base
    "enabled": True,
    "risk_pct": 0.5,
    "fixed_lot": None,
    "max_active_trades": 5,
    "max_trades_day": 5,
    "spread_max": 0.5,
    "sl_min": 1.5,
    "sl_max": 3.0,
    "tp1_r": 1.5,
    "tp2_r": 2.0,
    "tp3_r": None,          # not applicable — scalper exits at 2R
    "sessions": ["LONDON", "NEW_YORK"],
    "news_block": True,
    "cooldown_seconds": 120,
    "consecutive_loss_limit": 3,
    "loss_pause_seconds": 900,
    "notes": "",
    # Scalper-specific
    "spread_mean_max": 0.5,
    "spread_vol_max": 0.12,
    "spread_percentile_max": 0.9,
    "spread_delta_max": 0.05,
    "quality_threshold": 0.55,
    "atr_min": 0.6,
    "atr_max": 4.0,
    "ema20_slope_min": 0.05,
    "body_ratio_min": 0.6,
    "tick_dir_threshold": 0.7,
    "sweep_lookback": 20,
    "sweep_tolerance": 0.25,
    "allow_asian_session": False,
    "max_trades_session": 5,
    "level_cooldown": 1800,
    # Gate overrides
    "override_post_signal_spread": False,
    "override_execution_quality": False,
    "override_tick_direction": False,
    "override_atr_rising": False,
    "override_compression": False,
}

LOCKED = {}
DISABLED = {
    "tp3_r": "Scalper exits at 2R — no TP3",
}

CFG_MAP = {
    "risk_pct": "INTRADAY_RISK_PCT",
    "spread_mean_max": "SCALPER_SPREAD_MEAN_MAX",
    "spread_vol_max": "SCALPER_SPREAD_STD_MAX",
    "spread_percentile_max": "SCALPER_SPREAD_PERCENTILE_MAX",
    "spread_delta_max": "SCALPER_CURRENT_SPREAD_DELTA_MAX",
    "quality_threshold": "SCALPER_QUALITY_THRESHOLD",
    "atr_min": "SCALPER_ATR_MIN",
    "atr_max": "SCALPER_ATR_MAX",
    "ema20_slope_min": "SCALPER_EMA20_SLOPE_MIN",
    "body_ratio_min": "SCALPER_BODY_RATIO_MIN",
    "tick_dir_threshold": "SCALPER_TICK_DIR_THRESHOLD",
    "sweep_lookback": "SCALPER_SWEEP_LOOKBACK",
    "sweep_tolerance": "SCALPER_SWEEP_TOLERANCE",
    "allow_asian_session": "SCALPER_ALLOW_ASIAN_SESSION",
    "max_trades_session": "SCALPER_MAX_TRADES_SESSION",
    "level_cooldown": "SCALPER_LEVEL_COOLDOWN",
    "override_post_signal_spread": "POST_SIGNAL_SPREAD_GATE_OVERRIDE_ENABLED",
    "override_execution_quality": "EXECUTION_QUALITY_GATE_OVERRIDE_ENABLED",
    "override_tick_direction": "TICK_DIRECTION_GATE_OVERRIDE_ENABLED",
    "override_atr_rising": "ATR_RISING_GATE_OVERRIDE_ENABLED",
    "override_compression": "COMPRESSION_RANGE_GATE_OVERRIDE_ENABLED",
}
