"""
Adaptive Parameter Engine — regime + performance aware parameter adjustment.

Design:
- Regime and performance are INDEPENDENT offsets applied to the same persisted
  config base values, then combined and clamped once.
- This prevents stacking: a -9 loss streak on top of RANGING can no longer push
  thresholds past the hard ceiling and make trading impossible.
- Streak tighten is capped at 2 steps (max +0.04) regardless of streak depth.

Never touched: DAILY_LOSS_LIMIT_PCT, MAX_CONSECUTIVE_LOSSES, TIER1_ENABLED,
               spread re-check flags, TIME_GATE_OVERRIDE_ENABLED, MAX_RISK_PCT.

Hard bounds (read live from ADAPT_* cfg keys so dashboard changes take effect):
  SMC_THRESHOLD_WITH_TREND : ADAPT_THRESHOLD_WTR_MIN – ADAPT_THRESHOLD_WTR_MAX
  SMC_THRESHOLD_COUNTER    : ADAPT_THRESHOLD_CTR_MIN – ADAPT_THRESHOLD_CTR_MAX
  SMC_MIN_RR               : ADAPT_MIN_RR_MIN        – ADAPT_MIN_RR_MAX
  SESSION_MAX_TRADES       : ADAPT_SESSION_TRADES_MIN – ADAPT_SESSION_TRADES_MAX
  MIN_TRADE_COOLDOWN       : ADAPT_COOLDOWN_MIN       – ADAPT_COOLDOWN_MAX
  SMC_SPREAD_MEAN_MAX      : 0.28 – 0.50
  SMC_EARLY_FAIL_RANGING   : 0.10 – 0.20
  SMC_EARLY_FAIL_TRENDING  : 0.20 – 0.35
"""
import time
import config as cfg

_last_apply_time = 0.0
_APPLY_INTERVAL  = 30.0   # seconds between recalculations

# Persisted baseline — captured once at import time from runtime_config.json.
# The adaptive engine always offsets from this snapshot, never from the live cfg
# values it previously wrote. This prevents drift across cycles.
_BASE: dict = {}


def _capture_base():
    """Snapshot the persisted config values. Called once after runtime_config loads."""
    global _BASE
    _BASE = {
        "SMC_THRESHOLD_WITH_TREND": cfg.SMC_THRESHOLD_WITH_TREND,
        "SMC_THRESHOLD_COUNTER":    cfg.SMC_THRESHOLD_COUNTER,
        "SMC_MIN_RR":               cfg.SMC_MIN_RR,
        "SESSION_MAX_TRADES":       cfg.SESSION_MAX_TRADES,
        "MIN_TRADE_COOLDOWN":       cfg.MIN_TRADE_COOLDOWN,
        "SMC_SPREAD_MEAN_MAX":      cfg.SMC_SPREAD_MEAN_MAX,
        "SMC_EARLY_FAIL_RANGING":   cfg.SMC_EARLY_FAIL_RANGING,
        "SMC_EARLY_FAIL_TRENDING":  cfg.SMC_EARLY_FAIL_TRENDING,
    }


# Capture immediately on import (runtime_config.json is already loaded by this point)
_capture_base()


def _bounds() -> dict:
    """Read live from cfg so dashboard ADAPT_* changes take effect without restart."""
    return {
        "SMC_THRESHOLD_WITH_TREND": (cfg.ADAPT_THRESHOLD_WTR_MIN, cfg.ADAPT_THRESHOLD_WTR_MAX),
        "SMC_THRESHOLD_COUNTER":    (cfg.ADAPT_THRESHOLD_CTR_MIN, cfg.ADAPT_THRESHOLD_CTR_MAX),
        "SMC_MIN_RR":               (cfg.ADAPT_MIN_RR_MIN,        cfg.ADAPT_MIN_RR_MAX),
        "SESSION_MAX_TRADES":       (cfg.ADAPT_SESSION_TRADES_MIN, cfg.ADAPT_SESSION_TRADES_MAX),
        "MIN_TRADE_COOLDOWN":       (cfg.ADAPT_COOLDOWN_MIN,       cfg.ADAPT_COOLDOWN_MAX),
        "SMC_SPREAD_MEAN_MAX":      (0.28, 0.50),
        "SMC_EARLY_FAIL_RANGING":   (0.10, 0.20),
        "SMC_EARLY_FAIL_TRENDING":  (0.20, 0.35),
    }


def _clamp(key: str, value, b: dict):
    lo, hi = b[key]
    return max(lo, min(hi, value))


def _set(key: str, value, b: dict):
    """Clamp, round floats to 2dp, and apply to cfg."""
    clamped = _clamp(key, value, b)
    if isinstance(clamped, float):
        clamped = round(clamped, 2)
    setattr(cfg, key, clamped)


def apply(regime: dict, risk_manager, perf_tracker) -> dict:
    """
    Called every ~30 cycles from the engine loop.
    Returns a dict describing what was adjusted (for logging).
    """
    global _last_apply_time
    now = time.time()
    if now - _last_apply_time < _APPLY_INTERVAL:
        return {}
    _last_apply_time = now

    b            = _bounds()
    regime_state = (regime or {}).get("state", "TRENDING")
    atr_ratio    = (regime or {}).get("atr_ratio", 1.0)
    daily_status = risk_manager.daily_status
    recent       = perf_tracker.recent_stats(10)
    streak       = recent.get("streak", 0)
    win_rate_10  = recent.get("win_rate", 0.5)
    daily_pnl    = daily_status.get("pnl", 0.0)
    start_bal    = risk_manager._start_balance or 1000.0
    daily_dd_pct = (-daily_pnl / start_bal * 100) if daily_pnl < 0 else 0.0

    changes = {}

    # ── Snapshot persisted base values ────────────────────────────────────────
    # Read from _BASE (captured at startup from runtime_config.json), not from
    # live cfg. This ensures each cycle offsets from the saved baseline, not from
    # whatever this function wrote in the previous cycle.
    cfg_wt     = _BASE["SMC_THRESHOLD_WITH_TREND"]
    cfg_ctr    = _BASE["SMC_THRESHOLD_COUNTER"]
    cfg_rr     = _BASE["SMC_MIN_RR"]
    cfg_trades = _BASE["SESSION_MAX_TRADES"]
    cfg_cool   = _BASE["MIN_TRADE_COOLDOWN"]
    cfg_spread = _BASE["SMC_SPREAD_MEAN_MAX"]
    cfg_ef_r   = _BASE["SMC_EARLY_FAIL_RANGING"]
    cfg_ef_t   = _BASE["SMC_EARLY_FAIL_TRENDING"]

    # ── 1. Regime offset (applied to base) ────────────────────────────────────
    if regime_state == "RANGING":
        r_wt  = cfg.SMC_RANGING_THRESHOLD_OFFSET
        r_ctr = cfg.SMC_RANGING_THRESHOLD_OFFSET
        r_rr  = 0.10
        r_spread = max(b["SMC_SPREAD_MEAN_MAX"][0], cfg_spread - 0.05)
        r_ef_r   = max(b["SMC_EARLY_FAIL_RANGING"][0], cfg_ef_r - 0.02)
        r_ef_t   = cfg_ef_t
    elif regime_state in ("NEWS_VOLATILITY", "EXPANSION"):
        r_wt  = 0.05
        r_ctr = 0.05
        r_rr  = 0.20
        r_spread = max(b["SMC_SPREAD_MEAN_MAX"][0], cfg_spread - 0.08)
        r_ef_r   = cfg_ef_r
        r_ef_t   = min(b["SMC_EARLY_FAIL_TRENDING"][1], cfg_ef_t + 0.05)
    else:  # TRENDING
        r_wt  = cfg.SMC_TRENDING_THRESHOLD_OFFSET
        r_ctr = cfg.SMC_TRENDING_THRESHOLD_OFFSET
        r_rr  = 0.0
        r_spread = cfg_spread
        r_ef_r   = cfg_ef_r
        r_ef_t   = cfg_ef_t

    # Low volatility proxy (independent additive offset)
    lv_off = 0.03 if atr_ratio < 0.6 else 0.0

    # ── 2. Performance offset (independent of regime) ─────────────────────────
    p_wt = p_ctr = p_rr = 0.0
    p_cool_add = p_trades_sub = 0

    if streak <= -2:
        # Cap at 2 steps (0.04) regardless of how deep the streak goes
        tighten      = min(2, abs(streak)) * 0.02
        p_wt         = tighten
        p_ctr        = tighten
        p_rr         = tighten * 0.5
        p_cool_add   = 300
        p_trades_sub = 1
        changes["loss_streak_tighten"] = streak
    elif win_rate_10 < 0.40:
        p_wt  = 0.02
        p_ctr = 0.02
        changes["low_winrate_tighten"] = round(win_rate_10, 2)

    # ── 3. Daily drawdown > 0.5% ──────────────────────────────────────────────
    dd_cool_add = dd_trades_sub = 0
    if daily_dd_pct > 0.5:
        dd_cool_add   = 300
        dd_trades_sub = 1
        changes["daily_dd_tighten"] = round(daily_dd_pct, 2)

    # ── 4. Combine all offsets onto base, then clamp once ─────────────────────
    _set("SMC_THRESHOLD_WITH_TREND", cfg_wt  + r_wt  + lv_off + p_wt,  b)
    _set("SMC_THRESHOLD_COUNTER",    cfg_ctr + r_ctr + lv_off + p_ctr, b)
    _set("SMC_MIN_RR",               cfg_rr  + r_rr  + p_rr,           b)
    _set("SESSION_MAX_TRADES",       int(cfg_trades - p_trades_sub - dd_trades_sub), b)
    _set("MIN_TRADE_COOLDOWN",       cfg_cool + p_cool_add + dd_cool_add, b)
    _set("SMC_SPREAD_MEAN_MAX",      r_spread,  b)
    _set("SMC_EARLY_FAIL_RANGING",   r_ef_r,    b)
    _set("SMC_EARLY_FAIL_TRENDING",  r_ef_t,    b)

    changes.update({
        "regime":                   regime_state,
        "SMC_THRESHOLD_WITH_TREND": cfg.SMC_THRESHOLD_WITH_TREND,
        "SMC_THRESHOLD_COUNTER":    cfg.SMC_THRESHOLD_COUNTER,
        "SMC_MIN_RR":               cfg.SMC_MIN_RR,
        "SESSION_MAX_TRADES":       cfg.SESSION_MAX_TRADES,
        "MIN_TRADE_COOLDOWN":       cfg.MIN_TRADE_COOLDOWN,
        "SMC_SPREAD_MEAN_MAX":      cfg.SMC_SPREAD_MEAN_MAX,
        "SMC_EARLY_FAIL_RANGING":   cfg.SMC_EARLY_FAIL_RANGING,
        "SMC_EARLY_FAIL_TRENDING":  cfg.SMC_EARLY_FAIL_TRENDING,
        "streak":                   streak,
        "daily_dd_pct":             round(daily_dd_pct, 2),
    })
    return changes
