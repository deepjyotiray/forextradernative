"""
Trade Manager — persistent state, correct P&L tracking.
Saves open trades to disk so restarts don't lose tracking.
Updates P&L from live positions every cycle.
Integrated with comprehensive order database and PnL validation.
"""
import time
import json
import os
import pandas as pd
import MetaTrader5 as mt5
from typing import Any, Dict, List
from datetime import datetime, timezone, timedelta
import config as cfg
from .order_database import OrderDatabase
from .pnl_validator import PnLValidator
from .decision_logger import log_closed_trade
from .strategy_activity_log import append_strategy_event

_IST = timezone(timedelta(hours=5, minutes=30))
_MT5_TZ = timezone(timedelta(hours=3))
_MT5_OFFSET_SECONDS = int(_MT5_TZ.utcoffset(None).total_seconds())

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_STATE_FILE = os.path.join(_BASE_DIR, "open_trades.json")

_EXIT_PROFILE_FIELDS = (
    "profile_name",
    "be_trigger_r",
    "breakeven_min_hold_seconds",
    "breakeven_volume_hold_ratio",
    "min_hold_seconds",
    "early_fail_points",
    "early_fail_min_ticks",
    "early_fail_max_ticks",
    "reversal_arm_r",
    "reversal_drawdown_pct",
    "reversal_floor_r",
    "profit_lock_1_arm_r",
    "profit_lock_1_r",
    "profit_lock_2_arm_r",
    "profit_lock_2_r",
    "timeout_seconds",
    "timeout_min_progress_r",
    "trail_activate_r",
    "trail_lock_r",
    "velocity_drop_enabled",
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
    if value is None:
        return bool(default)
    return bool(value)


def _risk_unit_dollars(sl_distance: float, volume: float) -> float:
    risk = _safe_float(sl_distance, 0.0) * _safe_float(volume, 0.0) * cfg.PIP_VALUE_PER_LOT
    return risk if risk > 0 else 1.0


def _mt5_ts_to_utc(ts: int) -> datetime:
    return datetime.fromtimestamp(int(ts) - _MT5_OFFSET_SECONDS, tz=timezone.utc)


def _infer_profile_name(strategy: str, scalp: bool, features: Dict[str, Any]) -> str:
    profile_name = str(
        (features or {}).get("profile_name")
        or (features or {}).get("_exit_profile")
        or ""
    ).strip().lower()
    if profile_name:
        return profile_name
    if scalp:
        return "scalp"
    return "swing_fast"


def _resolve_exit_profile(
    features: Dict[str, Any],
    *,
    strategy: str,
    scalp: bool,
    sl_distance: float,
    volume: float,
    be_trigger: float = 0.0,
    timeout: int = 0,
    early_fail: float = 0.0,
    tier1_min_ticks: int | None = None,
    tier1_max_ticks: int | None = None,
) -> Dict[str, Any]:
    feature_map = dict(features or {})
    profile_name = _infer_profile_name(strategy, scalp, feature_map)
    profile = cfg.get_exit_profile_config(profile_name)
    isolated_profile = profile_name in {"swing_engine"}
    risk_unit = _risk_unit_dollars(sl_distance, volume)
    legacy_be_trigger_r = (_safe_float(be_trigger, 0.0) / risk_unit) if _safe_float(be_trigger, 0.0) > 0 else 0.0
    be_trigger_override = feature_map["be_trigger_r"] if "be_trigger_r" in feature_map else feature_map.get("_be_trigger_r")

    resolved = {
        "profile_name": profile_name,
        "be_trigger_r": _safe_float(
            be_trigger_override,
            0.0 if isolated_profile else (legacy_be_trigger_r or _safe_float(profile.get("be_trigger_r"), 0.0)),
        ),
        "breakeven_min_hold_seconds": _safe_int(
            feature_map.get("breakeven_min_hold_seconds", feature_map.get("_breakeven_min_hold_seconds")),
            _safe_int(profile.get("breakeven_min_hold_seconds"), 0),
        ),
        "breakeven_volume_hold_ratio": _safe_float(
            feature_map.get("breakeven_volume_hold_ratio", feature_map.get("_breakeven_volume_hold_ratio")),
            _safe_float(profile.get("breakeven_volume_hold_ratio"), 0.0),
        ),
        "min_hold_seconds": _safe_int(
            feature_map.get("min_hold_seconds", feature_map.get("_min_hold_seconds")),
            _safe_int(profile.get("min_hold_seconds"), 0),
        ),
        "early_fail_points": _safe_float(
            feature_map.get("early_fail_points", feature_map.get("_early_fail_points", feature_map.get("_early_fail"))),
            _safe_float(profile.get("early_fail_points"), 0.0)
            if isolated_profile
            else _safe_float(early_fail, _safe_float(profile.get("early_fail_points"), 0.0)),
        ),
        "early_fail_min_ticks": _safe_int(
            feature_map.get("early_fail_min_ticks", feature_map.get("tier1_min_ticks", feature_map.get("_tier1_min_ticks"))),
            tier1_min_ticks if tier1_min_ticks is not None else _safe_int(profile.get("early_fail_min_ticks"), cfg.SMC_TIER1_MIN_TICKS),
        ),
        "early_fail_max_ticks": _safe_int(
            feature_map.get("early_fail_max_ticks", feature_map.get("tier1_max_ticks", feature_map.get("_tier1_max_ticks"))),
            tier1_max_ticks if tier1_max_ticks is not None else _safe_int(profile.get("early_fail_max_ticks"), cfg.SMC_TIER1_MAX_TICKS),
        ),
        "reversal_arm_r": _safe_float(
            feature_map.get("reversal_arm_r", feature_map.get("_reversal_arm_r")),
            _safe_float(profile.get("reversal_arm_r"), 0.0),
        ),
        "reversal_drawdown_pct": _safe_float(
            feature_map.get("reversal_drawdown_pct", feature_map.get("_reversal_drawdown_pct")),
            _safe_float(profile.get("reversal_drawdown_pct"), 0.0),
        ),
        "reversal_floor_r": _safe_float(
            feature_map.get("reversal_floor_r", feature_map.get("_reversal_floor_r")),
            _safe_float(profile.get("reversal_floor_r"), 0.0),
        ),
        "profit_lock_1_arm_r": _safe_float(
            feature_map.get("profit_lock_1_arm_r", feature_map.get("_profit_lock_1_arm_r")),
            _safe_float(profile.get("profit_lock_1_arm_r"), 0.0),
        ),
        "profit_lock_1_r": _safe_float(
            feature_map.get("profit_lock_1_r", feature_map.get("_profit_lock_1_r")),
            _safe_float(profile.get("profit_lock_1_r"), 0.0),
        ),
        "profit_lock_2_arm_r": _safe_float(
            feature_map.get("profit_lock_2_arm_r", feature_map.get("_profit_lock_2_arm_r")),
            _safe_float(profile.get("profit_lock_2_arm_r"), 0.0),
        ),
        "profit_lock_2_r": _safe_float(
            feature_map.get("profit_lock_2_r", feature_map.get("_profit_lock_2_r")),
            _safe_float(profile.get("profit_lock_2_r"), 0.0),
        ),
        "timeout_seconds": _safe_int(
            feature_map.get("timeout_seconds", feature_map.get("_timeout")),
            _safe_int(profile.get("timeout_seconds"), 0)
            if isolated_profile
            else (timeout or _safe_int(profile.get("timeout_seconds"), 0)),
        ),
        "timeout_min_progress_r": _safe_float(
            feature_map.get("timeout_min_progress_r", feature_map.get("_timeout_min_progress_r")),
            _safe_float(profile.get("timeout_min_progress_r"), 0.0),
        ),
        "trail_activate_r": _safe_float(
            feature_map.get("trail_activate_r", feature_map.get("_trail_activate_r")),
            _safe_float(profile.get("trail_activate_r"), 0.0),
        ),
        "trail_lock_r": _safe_float(
            feature_map.get("trail_lock_r", feature_map.get("_trail_lock_r")),
            _safe_float(profile.get("trail_lock_r"), 0.0),
        ),
        "velocity_drop_enabled": _safe_bool(
            feature_map.get("velocity_drop_enabled", feature_map.get("_velocity_drop_enabled")),
            _safe_bool(profile.get("velocity_drop_enabled"), False),
        ),
    }
    if _safe_bool(feature_map.get("anti_mode"), False):
        resolved["be_trigger_r"] = 0.0
        resolved["breakeven_min_hold_seconds"] = 0
        resolved["breakeven_volume_hold_ratio"] = 0.0
        resolved["early_fail_points"] = 9999.0
        resolved["profit_lock_1_arm_r"] = 0.0
        resolved["profit_lock_1_r"] = 0.0
        resolved["profit_lock_2_arm_r"] = 0.0
        resolved["profit_lock_2_r"] = 0.0
        resolved["reversal_arm_r"] = 9999.0
        resolved["reversal_drawdown_pct"] = 1.0
        resolved["reversal_floor_r"] = -9999.0
        resolved["timeout_seconds"] = max(_safe_int(resolved.get("timeout_seconds"), 0), 12 * 3600)
        resolved["timeout_min_progress_r"] = -9999.0
        resolved["trail_activate_r"] = 0.0
        resolved["trail_lock_r"] = 0.0
        resolved["velocity_drop_enabled"] = False
    return resolved


def _close_reason_category(close_reason: str, trade: "TradeRecord", exit_price: float = 0.0) -> str:
    reason = str(close_reason or "").strip()
    lower = reason.lower()
    if lower.startswith("m15 scalp fixed profit target"):
        return "profit_target"
    if reason.startswith("TIER1_EXIT"):
        return "tier1_fail"
    if lower.startswith("speed exit") or lower.startswith("time exit"):
        return "timeout"
    if lower.startswith("velocity drop"):
        return "velocity_drop"
    if "reversal" in lower:
        return "reversal"
    if reason == "MT5_CLOSED":
        tolerance = max(0.05, _safe_float(trade.sl_distance, 0.0) * 0.10)
        if exit_price > 0 and _safe_float(trade.tp, 0.0) > 0 and abs(exit_price - trade.tp) <= tolerance:
            return "tp"
        if exit_price > 0 and _safe_float(trade.sl, 0.0) > 0 and abs(exit_price - trade.sl) <= tolerance and trade.sl_breakeven:
            return "breakeven_stop"
        if exit_price > 0 and _safe_float(trade.sl, 0.0) > 0 and abs(exit_price - trade.sl) <= tolerance:
            return "sl"
        return "mt5_external"
    return "manual"


def _uses_isolated_exit_manager(trade: "TradeRecord") -> bool:
    profile_name = str(getattr(trade, "exit_profile", "") or "").strip().lower()
    strategy_name = str(getattr(trade, "strategy", "") or "").strip().upper()
    return profile_name in {"swing_engine"} or strategy_name in {"SWING_ENGINE"}


def _uses_m15_scalp_fixed_profit_target(trade: "TradeRecord") -> bool:
    if not _safe_bool(getattr(cfg, "M15_SCALP_FIXED_USD_TP_ENABLED", True), True):
        return False
    profile_name = str(getattr(trade, "exit_profile", "") or "").strip().lower()
    strategy_name = str(getattr(trade, "strategy", "") or "").strip().upper()
    return profile_name in {"m15_zone_scalp", "m15_scalp_deep", "m15_mean_reversion_fast"} or strategy_name in {
        "M15_ZONE_SCALP",
        "M15_ZONE_SCALP_INVERSE",
        "M15_SCALP_DEEP",
    }


def _fixed_profit_target_price(direction: str, entry: float, volume: float, target_usd: float) -> float | None:
    per_point_value = _safe_float(volume, 0.0) * _safe_float(getattr(cfg, "PIP_VALUE_PER_LOT", 0.0), 0.0)
    if per_point_value <= 0 or _safe_float(target_usd, 0.0) <= 0:
        return None
    target_points = _safe_float(target_usd, 0.0) / per_point_value
    if target_points <= 0:
        return None
    entry_price = _safe_float(entry, 0.0)
    if str(direction or "").strip().upper() == "SELL":
        return round(entry_price - target_points, 2)
    return round(entry_price + target_points, 2)


def _should_tighten_take_profit(direction: str, current_tp: float, candidate_tp: float | None) -> bool:
    if candidate_tp is None or candidate_tp <= 0:
        return False
    current = _safe_float(current_tp, 0.0)
    if current <= 0:
        return True
    if str(direction or "").strip().upper() == "SELL":
        return candidate_tp > current
    return candidate_tp < current


def _mt5_deal_reason_name(reason_code: Any) -> str:
    mapping = {
        getattr(mt5, "DEAL_REASON_CLIENT", None): "client",
        getattr(mt5, "DEAL_REASON_MOBILE", None): "mobile",
        getattr(mt5, "DEAL_REASON_WEB", None): "web",
        getattr(mt5, "DEAL_REASON_EXPERT", None): "expert",
        getattr(mt5, "DEAL_REASON_SL", None): "sl",
        getattr(mt5, "DEAL_REASON_TP", None): "tp",
        getattr(mt5, "DEAL_REASON_SO", None): "stopout",
        getattr(mt5, "DEAL_REASON_ROLLOVER", None): "rollover",
        getattr(mt5, "DEAL_REASON_VMARGIN", None): "variation_margin",
        getattr(mt5, "DEAL_REASON_SPLIT", None): "split",
    }
    return str(mapping.get(reason_code, "unknown"))


def _describe_mt5_close(close_data: Dict[str, Any]) -> str:
    reason_name = str(close_data.get("reason_name") or "").strip().lower()
    if reason_name == "sl":
        return "MT5_SL"
    if reason_name == "tp":
        return "MT5_TP"
    if reason_name == "stopout":
        return "MT5_STOPOUT"
    if reason_name == "expert":
        comment = str(close_data.get("deal_comment") or "").strip()
        if comment:
            return f"MT5_EXPERT:{comment}"
        return "MT5_EXPERT"
    return "MT5_CLOSED"


def _categorize_mt5_close(trade: "TradeRecord", close_data: Dict[str, Any], exit_price: float) -> str:
    reason_name = str(close_data.get("reason_name") or "").strip().lower()
    comment = str(close_data.get("deal_comment") or "").strip().upper()
    tolerance = max(0.05, _safe_float(trade.sl_distance, 0.0) * 0.10)
    if reason_name == "tp":
        return "tp"
    if reason_name == "sl":
        entry_tolerance = max(0.05, tolerance)
        if trade.sl_breakeven or (exit_price > 0 and abs(exit_price - trade.entry) <= entry_tolerance):
            return "breakeven_stop"
        return "sl"
    if reason_name == "stopout":
        return "stopout"
    if reason_name == "expert":
        if "PARTIAL" in comment:
            return "partial"
        if "FT_CLOSE" in comment:
            return "expert_close"
        return "manual"
    return _close_reason_category("MT5_CLOSED", trade, exit_price)


def _volume_ratio_from_m1(m1_df: Any, lookback: int) -> float:
    try:
        if m1_df is None or len(m1_df) < max(3, lookback + 1) or "volume" not in m1_df:
            return 0.0
        recent = m1_df["volume"].tail(lookback + 1).astype(float)
        last_volume = float(recent.iloc[-1] or 0.0)
        baseline = float(recent.iloc[:-1].mean() or 0.0)
        if baseline <= 0:
            return 0.0
        return round(last_volume / baseline, 3)
    except Exception:
        return 0.0


def _pressure_supports_direction(direction: str, tick_pressure: Dict[str, Any], threshold: float | None = None) -> bool:
    score = _safe_float((tick_pressure or {}).get("pressure_score"), 0.0)
    limit = _safe_float(threshold, _safe_float(getattr(cfg, "TICK_PRESSURE_BREAKEVEN_HOLD_MIN", 0.25), 0.25))
    direction = str(direction or "").upper()
    if direction == "BUY":
        return score >= limit or bool((tick_pressure or {}).get("favorable_long"))
    if direction == "SELL":
        return score <= -limit or bool((tick_pressure or {}).get("favorable_short"))
    return False


def _candle_body_ratio(candle: Any) -> float:
    try:
        high = _safe_float(candle.get("high"))
        low = _safe_float(candle.get("low"))
        open_price = _safe_float(candle.get("open"))
        close = _safe_float(candle.get("close"))
        rng = max(0.01, high - low)
        return round(abs(close - open_price) / rng, 3)
    except Exception:
        return 0.0


def _candle_direction(candle: Any) -> str:
    try:
        open_price = _safe_float(candle.get("open"))
        close = _safe_float(candle.get("close"))
        if close > open_price:
            return "BULLISH"
        if close < open_price:
            return "BEARISH"
    except Exception:
        pass
    return "NEUTRAL"


def _is_day_break_candle(df: Any, gap_threshold_hours: float = 4.0) -> bool:
    """Return True if the most recent candle opened after a market closure gap
    (weekend, holiday, or any gap > gap_threshold_hours between candles).
    In that case the candle body reflects the gap open, not a live reversal."""
    try:
        if df is None or len(df) < 2:
            return False
        import pandas as pd
        times = df["time"] if "time" in df.columns else df.index
        last_time = pd.Timestamp(times.iloc[-1])
        prev_time = pd.Timestamp(times.iloc[-2])
        gap_seconds = (last_time - prev_time).total_seconds()
        return gap_seconds > gap_threshold_hours * 3600
    except Exception:
        return False


def _last_candle_marker(df: Any) -> str:
    try:
        if df is None or len(df) < 1:
            return ""
        last = df.iloc[-1]
        times = df["time"] if "time" in df.columns else df.index
        ts = str(times.iloc[-1] if hasattr(times, "iloc") else times[-1])
        return (
            f"{ts}|"
            f"{_safe_float(last.get('open')):.2f}|"
            f"{_safe_float(last.get('high')):.2f}|"
            f"{_safe_float(last.get('low')):.2f}|"
            f"{_safe_float(last.get('close')):.2f}"
        )
    except Exception:
        return ""


class TradeRecord:
    __slots__ = (
        "ticket", "direction", "volume", "entry", "sl", "tp",
        "sl_distance", "strategy", "confidence", "reason",
        "open_time", "open_time_ist", "fill_ts", "peak_pnl", "live_pnl",
        "sl_breakeven", "partial_closed", "trail_active",
        "initial_volume", "scalp", "be_trigger_price", "timeout_seconds",
        "features", "manage_updates", "entry_tick_velocity", "current_price",
        "early_fail_points", "tier1_min_ticks", "tier1_max_ticks",
        "exit_profile", "be_trigger_r", "breakeven_min_hold_seconds",
        "breakeven_volume_hold_ratio", "min_hold_seconds",
        "reversal_arm_r", "reversal_drawdown_pct", "reversal_floor_r",
        "profit_lock_1_arm_r", "profit_lock_1_r", "profit_lock_2_arm_r", "profit_lock_2_r",
        "timeout_min_progress_r", "trail_activate_r", "trail_lock_r",
        "velocity_drop_enabled",
    )

    def __init__(self, ticket, direction, volume, entry, sl, tp, sl_distance,
                 strategy="", confidence=0, reason="",
                 scalp=False, be_trigger=0, timeout=0, early_fail=0, features=None,
                 tier1_min_ticks=None, tier1_max_ticks=None):
        self.ticket = ticket
        self.direction = direction
        self.volume = volume
        self.initial_volume = volume
        self.entry = entry
        self.sl = sl
        self.tp = tp
        self.sl_distance = sl_distance
        self.strategy = strategy
        self.confidence = confidence
        self.reason = reason
        now = datetime.now(timezone.utc)
        self.open_time = now.isoformat()
        self.open_time_ist = now.astimezone(_IST).isoformat()
        self.fill_ts = time.time()
        self.peak_pnl = 0.0
        self.live_pnl = 0.0
        self.sl_breakeven = False
        self.partial_closed = False
        self.trail_active = False
        self.scalp = scalp
        self.be_trigger_price = be_trigger
        resolved_profile = _resolve_exit_profile(
            features or {},
            strategy=strategy,
            scalp=scalp,
            sl_distance=sl_distance,
            volume=volume,
            be_trigger=be_trigger,
            timeout=timeout,
            early_fail=early_fail,
            tier1_min_ticks=tier1_min_ticks,
            tier1_max_ticks=tier1_max_ticks,
        )
        self.timeout_seconds = resolved_profile["timeout_seconds"]
        self.features = dict(features or {})
        self.features.update(resolved_profile)
        self.manage_updates = 0
        self.entry_tick_velocity = float(self.features.get("entry_tick_velocity", 0) or 0)
        self.current_price = entry
        self.early_fail_points = resolved_profile["early_fail_points"]
        self.tier1_min_ticks = resolved_profile["early_fail_min_ticks"]
        self.tier1_max_ticks = resolved_profile["early_fail_max_ticks"]
        self.exit_profile = resolved_profile["profile_name"]
        self.be_trigger_r = resolved_profile["be_trigger_r"]
        self.breakeven_min_hold_seconds = resolved_profile["breakeven_min_hold_seconds"]
        self.breakeven_volume_hold_ratio = resolved_profile["breakeven_volume_hold_ratio"]
        self.min_hold_seconds = resolved_profile["min_hold_seconds"]
        self.reversal_arm_r = resolved_profile["reversal_arm_r"]
        self.reversal_drawdown_pct = resolved_profile["reversal_drawdown_pct"]
        self.reversal_floor_r = resolved_profile["reversal_floor_r"]
        self.profit_lock_1_arm_r = resolved_profile["profit_lock_1_arm_r"]
        self.profit_lock_1_r = resolved_profile["profit_lock_1_r"]
        self.profit_lock_2_arm_r = resolved_profile["profit_lock_2_arm_r"]
        self.profit_lock_2_r = resolved_profile["profit_lock_2_r"]
        self.timeout_min_progress_r = resolved_profile["timeout_min_progress_r"]
        self.trail_activate_r = resolved_profile["trail_activate_r"]
        self.trail_lock_r = resolved_profile["trail_lock_r"]
        self.velocity_drop_enabled = resolved_profile["velocity_drop_enabled"]

    def to_dict(self) -> Dict:
        return {s: getattr(self, s) for s in self.__slots__}

    @classmethod
    def from_dict(cls, d: Dict) -> "TradeRecord":
        t = cls(d["ticket"], d["direction"], d["volume"], d["entry"],
                d["sl"], d["tp"], d["sl_distance"], d.get("strategy", ""),
                d.get("confidence", 0), d.get("reason", ""),
                d.get("scalp", False), d.get("be_trigger_price", 0),
                d.get("timeout_seconds", 0), d.get("early_fail_points", 0),
                d.get("features", {}), d.get("tier1_min_ticks"), d.get("tier1_max_ticks"))
        t.initial_volume = d.get("initial_volume", d["volume"])
        t.open_time = d.get("open_time", "")
        t.open_time_ist = d.get("open_time_ist", "")
        t.fill_ts = d.get("fill_ts", time.time())
        t.peak_pnl = d.get("peak_pnl", 0)
        t.live_pnl = d.get("live_pnl", 0)
        t.sl_breakeven = d.get("sl_breakeven", False)
        t.partial_closed = d.get("partial_closed", False)
        t.trail_active = d.get("trail_active", False)
        merged_features = dict(t.features)
        merged_features.update(d.get("features", {}))
        t.features = merged_features
        t.manage_updates = d.get("manage_updates", 0)
        t.entry_tick_velocity = float(d.get("entry_tick_velocity", t.features.get("entry_tick_velocity", 0)) or 0)
        t.current_price = d.get("current_price", t.entry)
        t.tier1_min_ticks = int(d.get("tier1_min_ticks", cfg.SMC_TIER1_MIN_TICKS))
        t.tier1_max_ticks = int(d.get("tier1_max_ticks", cfg.SMC_TIER1_MAX_TICKS))
        t.exit_profile = d.get("exit_profile", t.exit_profile)
        t.be_trigger_r = float(d.get("be_trigger_r", t.be_trigger_r))
        t.breakeven_min_hold_seconds = int(d.get("breakeven_min_hold_seconds", t.breakeven_min_hold_seconds))
        t.breakeven_volume_hold_ratio = float(d.get("breakeven_volume_hold_ratio", t.breakeven_volume_hold_ratio))
        t.min_hold_seconds = int(d.get("min_hold_seconds", t.min_hold_seconds))
        t.reversal_arm_r = float(d.get("reversal_arm_r", t.reversal_arm_r))
        t.reversal_drawdown_pct = float(d.get("reversal_drawdown_pct", t.reversal_drawdown_pct))
        t.reversal_floor_r = float(d.get("reversal_floor_r", t.reversal_floor_r))
        t.profit_lock_1_arm_r = float(d.get("profit_lock_1_arm_r", t.profit_lock_1_arm_r))
        t.profit_lock_1_r = float(d.get("profit_lock_1_r", t.profit_lock_1_r))
        t.profit_lock_2_arm_r = float(d.get("profit_lock_2_arm_r", t.profit_lock_2_arm_r))
        t.profit_lock_2_r = float(d.get("profit_lock_2_r", t.profit_lock_2_r))
        t.timeout_min_progress_r = float(d.get("timeout_min_progress_r", t.timeout_min_progress_r))
        t.trail_activate_r = float(d.get("trail_activate_r", t.trail_activate_r))
        t.trail_lock_r = float(d.get("trail_lock_r", t.trail_lock_r))
        t.velocity_drop_enabled = bool(d.get("velocity_drop_enabled", t.velocity_drop_enabled))
        return t


class TradeManager:
    def __init__(self, mt5_bridge):
        self.bridge = mt5_bridge
        self.open_trades: Dict[int, TradeRecord] = {}
        self.closed_trades: List[Dict] = []  # Deprecated - use MT5 history instead
        self._closing_tickets: set = set()
        self._pending_close_reasons: Dict[int, Dict[str, Any]] = {}
        self._strategy_lockouts: Dict[str, Dict[str, Any]] = {}
        self.order_db = OrderDatabase()
        self.pnl_validator = PnLValidator(mt5_bridge)
        self._pending_reentry_check: str | None = None
        self._load_state()

    def register_trade(self, ticket, direction, volume, entry, sl, tp, sl_distance,
                       strategy="", confidence=0, reason="",
                       scalp=False, be_trigger=0, timeout=0, early_fail=0, features=None,
                       tier1_min_ticks=None, tier1_max_ticks=None,
                       session_type="", market_phase=""):
        feature_snapshot = dict(features or {})
        feature_snapshot.update(
            _resolve_exit_profile(
                feature_snapshot,
                strategy=strategy,
                scalp=scalp,
                sl_distance=sl_distance,
                volume=volume,
                be_trigger=be_trigger,
                timeout=timeout,
                early_fail=early_fail,
                tier1_min_ticks=tier1_min_ticks,
                tier1_max_ticks=tier1_max_ticks,
            )
        )

        trade = TradeRecord(
            ticket, direction, volume, entry, sl, tp, sl_distance,
            strategy, confidence, reason, scalp, be_trigger, timeout, early_fail, feature_snapshot,
            tier1_min_ticks, tier1_max_ticks,
        )
        fixed_target = _safe_float(getattr(cfg, "M15_SCALP_FIXED_USD_TP", 1.5), 1.5)
        fixed_target_tp = _fixed_profit_target_price(direction, entry, volume, fixed_target)
        if _uses_m15_scalp_fixed_profit_target(trade) and _should_tighten_take_profit(direction, tp, fixed_target_tp):
            res = self.bridge.modify_trade(ticket, sl, fixed_target_tp)
            if res.get("success"):
                trade.tp = fixed_target_tp

        # Store in memory for active management
        self.open_trades[ticket] = trade
        strategy_key = str(strategy or "").strip().upper()
        if strategy_key:
            self._strategy_lockouts.pop(strategy_key, None)

        # Store in database for permanent record
        self.order_db.store_order(
            ticket, direction, volume, entry, sl, trade.tp, sl_distance,
            strategy, confidence, reason, scalp, be_trigger, timeout,
            feature_snapshot, session_type, market_phase
        )
        append_strategy_event(
            strategy,
            {
                "event": "OPEN",
                "ticket": ticket,
                "strategy": strategy,
                "direction": direction,
                "volume": round(_safe_float(volume), 2),
                "entry_price": round(_safe_float(entry), 2),
                "sl": round(_safe_float(sl), 2),
                "tp": round(_safe_float(trade.tp), 2),
                "reason": reason,
            },
        )
        
        self._save_state()

    def _persist_open_state(self):
        try:
            self._save_state()
        except Exception:
            pass

    def _set_post_tier1_lockout(self, trade: TradeRecord, reason: str) -> None:
        strategy_key = str(getattr(trade, "strategy", "") or "").strip().upper()
        if not strategy_key:
            return
        cooldown = max(
            0.0,
            _safe_float(
                getattr(cfg, "POST_TIER1_REENTRY_COOLDOWN_SECONDS", 180.0),
                180.0,
            ),
        )
        features = dict(getattr(trade, "features", {}) or {})
        setup_signature = str(features.get("context_hash") or features.get("signal_id") or "").strip()
        if cooldown <= 0 and not setup_signature:
            return
        self._strategy_lockouts[strategy_key] = {
            "expires_at": time.time() + cooldown if cooldown > 0 else 0.0,
            "setup_signature": setup_signature,
            "reason": str(reason or "TIER1_EXIT"),
            "source_ticket": getattr(trade, "ticket", 0),
            "created_at": time.time(),
        }

    def get_strategy_entry_lockout_reason(self, strategy_name: str, *, setup_signature: str = "") -> str:
        strategy_key = str(strategy_name or "").strip().upper()
        if not strategy_key:
            return ""
        lockout = self._strategy_lockouts.get(strategy_key)
        if not lockout:
            return ""

        now_ts = time.time()
        expires_at = _safe_float(lockout.get("expires_at"), 0.0)
        stored_signature = str(lockout.get("setup_signature") or "").strip()
        incoming_signature = str(setup_signature or "").strip()
        same_setup = bool(incoming_signature and stored_signature and incoming_signature == stored_signature)
        source_ticket = _safe_int(lockout.get("source_ticket"), 0)

        if expires_at > now_ts:
            remaining = max(0.0, expires_at - now_ts)
            suffix = " for same setup" if same_setup else ""
            return f"post-TIER1 cooldown active ({remaining:.0f}s remaining{suffix}, source #{source_ticket})"

        if same_setup:
            return f"same setup locked after recent TIER1 exit (source #{source_ticket})"

        self._strategy_lockouts.pop(strategy_key, None)
        return ""

    def _r_milestone_bucket(self, pnl: float, trade: TradeRecord) -> int:
        live_r = self._pnl_to_r(trade, pnl)
        if live_r >= 2.5:
            return 3
        if live_r >= 2.0:
            return 2
        if live_r >= 1.5:
            return 1
        return 0

    def _recent_swing_levels(
        self,
        df: pd.DataFrame | None,
        *,
        tail: int = 6,
        swing_window: int = 3,
    ) -> tuple[float | None, float | None]:
        if df is None or len(df) < max(tail, swing_window):
            return None, None
        recent = df.tail(tail)
        try:
            swing_high = round(_safe_float(recent["high"].tail(swing_window).max()), 2)
            swing_low = round(_safe_float(recent["low"].tail(swing_window).min()), 2)
        except Exception:
            return None, None
        return swing_high, swing_low

    def _apply_forward_stop(
        self,
        t: TradeRecord,
        candidate_sl: float | None,
        *,
        mark_trailing: bool = False,
    ) -> bool:
        if candidate_sl is None:
            return False
        current_sl = _safe_float(t.sl, 0.0)
        new_sl = round(_safe_float(candidate_sl), 2)
        if t.direction == "BUY":
            if current_sl > 0 and new_sl <= current_sl:
                return False
        else:
            if current_sl > 0 and new_sl >= current_sl:
                return False

        res = self.bridge.modify_trade(t.ticket, new_sl, t.tp)
        if not res.get("success"):
            return False

        t.sl = new_sl
        if (t.direction == "BUY" and new_sl >= _safe_float(t.entry)) or (
            t.direction == "SELL" and new_sl <= _safe_float(t.entry)
        ):
            t.sl_breakeven = True
        if mark_trailing:
            t.trail_active = True
        self.order_db.update_management_flags(
            t.ticket,
            sl_breakeven=t.sl_breakeven,
            partial_closed=t.partial_closed,
            trail_active=t.trail_active,
        )
        self._persist_open_state()
        return True

    def manage_all(self, live_positions: List[Dict], tick_metrics: Dict = None, market_context: Dict | None = None):
        live_map = {p["ticket"]: p for p in live_positions}
        closed = []
        state_dirty = False
        tick_metrics = tick_metrics or {}
        market_context = market_context or {}

        # Auto-adopt bot positions that exist in MT5 but aren't tracked
        for p in live_positions:
            if p["ticket"] not in self.open_trades and p.get("magic") == cfg.MAGIC_NUMBER:
                self.register_trade(
                    p["ticket"], p["type"], p["volume"], p["open_price"],
                    p.get("sl") or 0, p.get("tp") or 0,
                    abs(p["open_price"] - (p.get("sl") or p["open_price"])),
                    strategy="adopted", reason="auto-adopted from MT5",
                )
                self._log_adopt(p["ticket"])

        for ticket, trade in list(self.open_trades.items()):
            if ticket not in live_map:
                # Position gone — fetch P&L from MT5 deal history (authoritative)
                # Retry up to 3 times with short delay for deal propagation
                close_data = {
                    "pnl": 0.0,
                    "exit_price": 0.0,
                    "swap": 0.0,
                    "commission": 0.0,
                    "reason_code": None,
                    "reason_name": "",
                    "deal_comment": "",
                    "deal_ticket": 0,
                    "deal_time": "",
                }
                for _attempt in range(3):
                    close_data = self._fetch_closed_pnl(ticket)
                    if close_data["exit_price"] > 0:
                        break
                    time.sleep(0.3)

                pnl = close_data["pnl"]
                exit_price = close_data["exit_price"]

                # Fallback: compute P&L from prices if deal history unavailable
                if exit_price == 0.0 and trade.live_pnl != 0.0:
                    # Use last known live P&L as exit price hint
                    pnl = trade.live_pnl
                elif exit_price > 0 and pnl == 0.0:
                    # Have exit price but no deal P&L — compute it
                    if trade.direction == "BUY":
                        pnl = round((exit_price - trade.entry) * trade.initial_volume * cfg.PIP_VALUE_PER_LOT, 2)
                    else:
                        pnl = round((trade.entry - exit_price) * trade.initial_volume * cfg.PIP_VALUE_PER_LOT, 2)

                close_context = self._pending_close_reasons.pop(ticket, {}) or {}
                mt5_close_reason = _describe_mt5_close(close_data)
                close_reason = str(close_context.get("reason") or mt5_close_reason)
                close_reason_category = str(close_context.get("category") or _categorize_mt5_close(trade, close_data, exit_price))
                close_signal_live_pnl = _safe_float(close_context.get("close_signal_live_pnl"), trade.live_pnl)
                close_signal_live_r = _safe_float(close_context.get("close_signal_live_r"), self._pnl_to_r(trade, close_signal_live_pnl))
                peak_r = _safe_float(close_context.get("peak_r"), self._pnl_to_r(trade, trade.peak_pnl))
                final_r = self._pnl_to_r(trade, pnl)
                drawdown_from_peak_r = round(max(0.0, peak_r - close_signal_live_r), 3)
                drawdown_from_peak_pct = round((drawdown_from_peak_r / peak_r), 3) if peak_r > 0 else 0.0
                held_seconds = round(_safe_float(close_context.get("held_seconds"), time.time() - trade.fill_ts), 1)

                # Update database with final PnL + exit price
                self.order_db.close_order(
                    ticket, pnl, close_reason,
                    swap=close_data["swap"], commission=close_data["commission"],
                    exit_price=exit_price,
                    close_reason_category=close_reason_category,
                    close_signal_live_pnl=close_signal_live_pnl,
                    peak_r=peak_r,
                    final_r=final_r,
                    drawdown_from_peak_r=drawdown_from_peak_r,
                    drawdown_from_peak_pct=drawdown_from_peak_pct,
                    held_seconds=held_seconds,
                    profile_name=trade.exit_profile,
                    mt5_close_reason=close_data.get("reason_name", ""),
                    mt5_close_reason_code=close_data.get("reason_code"),
                    mt5_close_comment=close_data.get("deal_comment", ""),
                    mt5_close_deal_time=close_data.get("deal_time", ""),
                    mt5_close_deal_ticket=close_data.get("deal_ticket", 0),
                )
                self._log_closed_trade(
                    trade,
                    pnl,
                    exit_price,
                    close_reason,
                    close_reason_category=close_reason_category,
                    close_signal_live_pnl=close_signal_live_pnl,
                    close_signal_live_r=close_signal_live_r,
                    peak_r=peak_r,
                    final_r=final_r,
                    drawdown_from_peak_r=drawdown_from_peak_r,
                    drawdown_from_peak_pct=drawdown_from_peak_pct,
                    held_seconds=held_seconds,
                    mt5_close_reason=close_data.get("reason_name", ""),
                    mt5_close_comment=close_data.get("deal_comment", ""),
                )
                closed.append((ticket, pnl, pnl > 0))
                self._closing_tickets.discard(ticket)
                del self.open_trades[ticket]
            else:
                pos = live_map[ticket]
                pnl = pos["net_profit"]
                prev_peak_pnl = _safe_float(trade.peak_pnl, 0.0)
                prev_bucket = self._r_milestone_bucket(prev_peak_pnl, trade)
                trade.features["_prev_peak_pnl"] = prev_peak_pnl
                trade.live_pnl = pnl
                trade.volume = pos["volume"]
                trade.current_price = pos.get("current_price", trade.current_price)
                
                # Update SL/TP from MT5 (may have been modified by broker/EA)
                if pos.get("sl"):
                    trade.sl = pos["sl"]
                if pos.get("tp"):
                    trade.tp = pos["tp"]
                if pnl > trade.peak_pnl:
                    trade.peak_pnl = pnl
                if self._r_milestone_bucket(_safe_float(trade.peak_pnl, 0.0), trade) != prev_bucket:
                    state_dirty = True
                
                # Update database with live PnL
                self.order_db.update_live_pnl(
                    ticket, pnl, trade.volume, trade.sl, trade.tp
                )

        for ticket, trade in list(self.open_trades.items()):
            if ticket in self._closing_tickets:
                continue
            self._current_market_context = market_context
            self._manage_trade(trade, tick_metrics, market_context)

        if closed:
            self._save_state()
        elif state_dirty:
            self._persist_open_state()

        return closed

    def _fetch_closed_pnl(self, ticket: int) -> Dict:
        """Fetch real P&L + exit price from MT5 deal history - AUTHORITATIVE SOURCE.
        Sums profit+swap+commission from ALL deals for this position_id."""
        result = {
            "pnl": 0.0,
            "exit_price": 0.0,
            "swap": 0.0,
            "commission": 0.0,
            "reason_code": None,
            "reason_name": "",
            "deal_comment": "",
            "deal_ticket": 0,
            "deal_time": "",
        }
        try:
            now = datetime.now(timezone.utc)
            deals = mt5.history_deals_get(now - timedelta(days=7), now + timedelta(hours=1))
            if not deals:
                return result
            total = 0.0
            total_swap = 0.0
            total_comm = 0.0
            found = False
            last_out = None
            for d in deals:
                if d.position_id == ticket:
                    total += d.profit + d.swap + d.commission
                    total_swap += d.swap
                    total_comm += d.commission
                    if d.entry in (1, 2):  # OUT deal
                        found = True
                        if last_out is None or getattr(d, "time_msc", 0) >= getattr(last_out, "time_msc", 0):
                            last_out = d
            if found:
                result["pnl"] = round(total, 2)
                result["swap"] = round(total_swap, 2)
                result["commission"] = round(total_comm, 2)
                if last_out is not None:
                    result["exit_price"] = last_out.price
                    result["reason_code"] = getattr(last_out, "reason", None)
                    result["reason_name"] = _mt5_deal_reason_name(getattr(last_out, "reason", None))
                    result["deal_comment"] = str(getattr(last_out, "comment", "") or "")
                    result["deal_ticket"] = int(getattr(last_out, "ticket", 0) or 0)
                    result["deal_time"] = _mt5_ts_to_utc(getattr(last_out, "time", 0)).isoformat()
            return result
        except Exception:
            return result

    def _manage_trade(self, t: TradeRecord, tick_metrics: Dict, market_context: Dict | None = None):
        if not _uses_isolated_exit_manager(t):
            if self._apply_universal_management(t, tick_metrics, market_context):
                return
        if t.strategy == "SWING_ENGINE" or t.exit_profile == "swing_engine":
            self._manage_swing_engine(t, market_context or {})
            return
        if t.exit_profile == "scalp":
            self._manage_scalp(t)
        else:
            self._manage_swing(t)

    def _pnl_to_r(self, trade: TradeRecord, pnl: float) -> float:
        return round(_safe_float(pnl, 0.0) / _risk_unit_dollars(trade.sl_distance, trade.initial_volume), 3)

    def _profit_lock_price(self, trade: TradeRecord, lock_r: float) -> float:
        lock_points = trade.sl_distance * max(0.0, lock_r)
        if trade.direction == "BUY":
            return round(trade.entry + lock_points, 2)
        return round(trade.entry - lock_points, 2)

    def _locked_profit_r(self, trade: TradeRecord) -> float:
        current_sl = _safe_float(trade.sl, 0.0)
        if current_sl <= 0:
            return -999.0
        if trade.direction == "BUY":
            return round((current_sl - trade.entry) / max(0.01, trade.sl_distance), 3)
        return round((trade.entry - current_sl) / max(0.01, trade.sl_distance), 3)

    def _time_invested_profit_lock_r(self, trade: TradeRecord, live_r: float) -> float:
        if not bool(getattr(cfg, "TIME_INVESTED_PROFIT_LOCK_ENABLED", True)):
            return 0.0
        if not bool(getattr(trade, "scalp", False)):
            return 0.0
        min_age = max(0, _safe_int(getattr(cfg, "TIME_INVESTED_PROFIT_LOCK_SECONDS", 300), 300))
        if (time.time() - trade.fill_ts) < min_age:
            return 0.0
        desired_usd = max(0.0, _safe_float(getattr(cfg, "TIME_INVESTED_PROFIT_LOCK_USD", 0.75), 0.75))
        if desired_usd <= 0:
            return 0.0
        risk_dollars = _risk_unit_dollars(trade.sl_distance, trade.initial_volume)
        if risk_dollars <= 0:
            return 0.0
        desired_lock_r = desired_usd / risk_dollars
        if desired_lock_r <= 0:
            return 0.0
        min_buffer_r = max(0.0, _safe_float(getattr(cfg, "TIME_INVESTED_PROFIT_LOCK_MIN_BUFFER_R", 0.02), 0.02))
        max_lock_r = max(0.0, live_r - min_buffer_r)
        return round(min(desired_lock_r, max_lock_r), 3) if max_lock_r > 0 else 0.0

    def _apply_time_invested_profit_lock(self, trade: TradeRecord, live_r: float) -> bool:
        target_lock_r = self._time_invested_profit_lock_r(trade, live_r)
        if target_lock_r <= 0:
            return False
        if self._locked_profit_r(trade) >= target_lock_r:
            return False
        if not self._tighten_profit_lock(trade, target_lock_r):
            return False
        trade.sl_breakeven = True
        self.order_db.update_management_flags(
            trade.ticket,
            sl_breakeven=True,
            trail_active=True,
        )
        return True

    def _tighten_profit_lock(self, trade: TradeRecord, lock_r: float) -> bool:
        if lock_r <= 0:
            return False
        new_sl = self._profit_lock_price(trade, lock_r)
        current_sl = _safe_float(trade.sl, 0.0)
        if trade.direction == "BUY" and current_sl >= new_sl:
            return False
        if trade.direction == "SELL" and current_sl <= new_sl:
            return False
        res = self.bridge.modify_trade(trade.ticket, new_sl, trade.tp)
        if not res.get("success"):
            return False
        trade.sl = new_sl
        trade.trail_active = True
        self.order_db.update_management_flags(trade.ticket, trail_active=True)
        return True

    def _apply_universal_management(self, t: TradeRecord, tick_metrics: Dict, market_context: Dict | None = None) -> bool:
        age = time.time() - t.fill_ts
        t.manage_updates += 1
        market_context = market_context or {}
        current_price = t.current_price or t.entry
        points_move = (current_price - t.entry) if t.direction == "BUY" else (t.entry - current_price)
        current_tick_count = int(tick_metrics.get("tick_count", 0) or 0)
        entry_tick_count = int(t.features.get("entry_tick_count", 0) or 0)
        ticks_since_entry = (current_tick_count - entry_tick_count) if entry_tick_count and current_tick_count else t.manage_updates
        live_r = self._pnl_to_r(t, t.live_pnl)
        tick_pressure = market_context.get("tick_pressure") or {}
        current_volume_ratio = _volume_ratio_from_m1(
            market_context.get("m1_df"),
            int(getattr(cfg, "BREAKEVEN_VOLUME_LOOKBACK_CANDLES", 8) or 8),
        )
        anti_mode = _safe_bool((t.features or {}).get("anti_mode"), False)
        anti_profit_choke_r = _safe_float(
            (t.features or {}).get("anti_profit_choke_r"),
            _safe_float(getattr(cfg, "ANTI_MODE_PROFIT_CHOKE_R", 0.08), 0.08),
        )

        if anti_mode and live_r >= anti_profit_choke_r and age >= 1.0:
            self._close_early(
                t,
                f"ANTI profit choke ({live_r:.2f}R >= {anti_profit_choke_r:.2f}R)",
                category="anti_profit_choke",
            )
            return True

        if cfg.TIER1_ENABLED and t.tier1_min_ticks <= ticks_since_entry <= t.tier1_max_ticks and points_move <= -t.early_fail_points:
            self._close_early(
                t,
                f"TIER1_EXIT: Early fail protection triggered ({points_move:.2f} pts at tick {ticks_since_entry}, window {t.tier1_min_ticks}-{t.tier1_max_ticks})",
                category="tier1_fail",
            )
            return True

        fixed_target = _safe_float(getattr(cfg, "M15_SCALP_FIXED_USD_TP", 1.5), 1.5)
        live_pnl = _safe_float(t.live_pnl, 0.0)
        peak_pnl = max(_safe_float(t.peak_pnl, 0.0), live_pnl)
        if _uses_m15_scalp_fixed_profit_target(t) and fixed_target > 0 and peak_pnl >= fixed_target:
            reason = f"M15 scalp fixed profit target hit (${live_pnl:.2f} >= ${fixed_target:.2f})"
            if peak_pnl > live_pnl:
                reason = (
                    f"M15 scalp fixed profit target latched from peak "
                    f"(${peak_pnl:.2f} >= ${fixed_target:.2f}; live ${live_pnl:.2f})"
                )
            self._close_early(
                t,
                reason,
                category="profit_target",
            )
            return True

        if t.be_trigger_r > 0 and not t.sl_breakeven and live_r >= t.be_trigger_r:
            if age < t.breakeven_min_hold_seconds:
                return False
            if (
                t.breakeven_volume_hold_ratio > 0
                and current_volume_ratio >= t.breakeven_volume_hold_ratio
                and live_r < max(t.be_trigger_r + 0.20, 0.75)
            ):
                return False
            if (
                tick_pressure.get("ready")
                and _pressure_supports_direction(t.direction, tick_pressure)
                and live_r < max(t.be_trigger_r + 0.20, 0.75)
            ):
                return False
            new_sl = round(t.entry, 2)
            time_invested_lock_r = self._time_invested_profit_lock_r(t, live_r)
            if time_invested_lock_r > 0:
                new_sl = self._profit_lock_price(t, time_invested_lock_r)
            res = self.bridge.modify_trade(t.ticket, new_sl, t.tp)
            if res.get("success"):
                t.sl = new_sl
                t.sl_breakeven = True
                self.order_db.update_management_flags(t.ticket, sl_breakeven=True)
            return True

        if self._apply_time_invested_profit_lock(t, live_r):
            return True

        if age >= t.timeout_seconds and live_r < t.timeout_min_progress_r:
            self._close_early(
                t,
                f"Time exit {age:.0f}s, progress {live_r:.2f}R < {t.timeout_min_progress_r:.2f}R",
                category="timeout",
            )
            return True

        current_velocity = float(tick_metrics.get("velocity", 0) or 0)
        if t.velocity_drop_enabled and t.entry_tick_velocity > 0 and current_velocity > 0 and age >= 10 and live_r < 0.20:
            if current_velocity <= max(1.0, t.entry_tick_velocity * 0.5) and points_move < 0.30:
                pressure_score = _safe_float(tick_pressure.get("pressure_score"), 0.0)
                if tick_pressure.get("ready") and abs(pressure_score) > _safe_float(getattr(cfg, "TICK_PRESSURE_VELOCITY_EXIT_MAX", 0.05), 0.05):
                    return False
                self._close_early(
                    t,
                    f"Velocity drop: {current_velocity:.1f} from {t.entry_tick_velocity:.1f} ticks/s",
                    category="velocity_drop",
                )
                return True

        return False

    def _manage_scalp(self, t: TradeRecord):
        age = time.time() - t.fill_ts
        peak_r = self._pnl_to_r(t, t.peak_pnl)
        live_r = self._pnl_to_r(t, t.live_pnl)
        if age >= t.min_hold_seconds:
            if peak_r >= t.profit_lock_2_arm_r > 0 and self._tighten_profit_lock(t, t.profit_lock_2_r):
                return
            if peak_r >= t.profit_lock_1_arm_r > 0 and self._tighten_profit_lock(t, t.profit_lock_1_r):
                return
        if age < t.min_hold_seconds or peak_r < t.reversal_arm_r or peak_r <= 0:
            return
        giveback_pct = (peak_r - live_r) / peak_r
        if giveback_pct >= t.reversal_drawdown_pct and live_r <= t.reversal_floor_r:
            self._close_early(
                t,
                f"Scalp reversal ({peak_r:.2f}R -> {live_r:.2f}R, {giveback_pct:.0%} giveback)",
                category="reversal",
            )
            return

    def _manage_swing(self, t: TradeRecord):
        risk_unit = _risk_unit_dollars(t.sl_distance, t.initial_volume)
        pnl = t.live_pnl
        live_r = self._pnl_to_r(t, pnl)
        peak_r = self._pnl_to_r(t, t.peak_pnl)
        age = time.time() - t.fill_ts

        if live_r >= 1.0 and not t.partial_closed:
            t.partial_closed = True
            close_vol = round(t.initial_volume * 0.5, 2)
            if cfg.TIER1_ENABLED:
                # Tier 1: skip partial close at min lot, tighten SL instead
                if close_vol >= cfg.MIN_LOT and (t.volume - close_vol) >= cfg.MIN_LOT:
                    self._partial_close(t, close_vol, f"Partial 50% at 1R")
            else:
                if close_vol >= cfg.MIN_LOT:
                    self._partial_close(t, close_vol, f"Partial 50% at 1R")
            rr = 2.0
            if t.direction == "BUY":
                new_tp = round(t.entry + t.sl_distance * rr, 2)
            else:
                new_tp = round(t.entry - t.sl_distance * rr, 2)
            new_sl = self._profit_lock_price(t, t.trail_lock_r)
            self.bridge.modify_trade(t.ticket, new_sl, new_tp)
            t.tp = new_tp
            t.sl = new_sl
            self.order_db.update_management_flags(
                t.ticket, partial_closed=True, trail_active=t.trail_active
            )
            return

        if t.partial_closed and live_r >= t.trail_activate_r:
            trail_sl = self._profit_lock_price(t, t.trail_lock_r)
            if not t.trail_active or trail_sl != t.sl:
                self.bridge.modify_trade(t.ticket, trail_sl, t.tp)
                t.sl = trail_sl
            t.trail_active = True
            self.order_db.update_management_flags(t.ticket, partial_closed=True, trail_active=True)
            return

        if age < t.min_hold_seconds or peak_r < t.reversal_arm_r or peak_r <= 0:
            return
        giveback_pct = (peak_r - live_r) / peak_r
        if giveback_pct >= t.reversal_drawdown_pct and live_r <= t.reversal_floor_r:
            self._close_early(
                t,
                f"Swing reversal ({peak_r:.2f}R -> {live_r:.2f}R, {giveback_pct:.0%} giveback)",
                category="reversal",
            )
            return

    def _manage_swing_engine(self, t: TradeRecord, market_context: Dict[str, Any]):
        live_r = self._pnl_to_r(t, t.live_pnl)
        peak_pnl = max(_safe_float(t.peak_pnl, 0.0), _safe_float(t.live_pnl, 0.0))
        peak_r = self._pnl_to_r(t, peak_pnl)
        candidate_sls: list[tuple[float, bool]] = []

        if live_r >= 1.0:
            candidate_sls.append((round(_safe_float(t.entry), 2), False))

        # ── Granular profit lock: arm at 0.2R, lock in at every 0.1R step ──
        # Each level locks in 70% of the peak reached so far, floored at 0.
        # e.g. peak 0.2R → lock 0.0R (breakeven), peak 0.5R → lock 0.35R,
        #      peak 1.0R → lock 0.7R, peak 2.0R → lock 1.4R, etc.
        _LOCK_ARM_START = 0.2   # first arm threshold
        _LOCK_STEP      = 0.1   # re-evaluate every 0.1R of new peak
        _LOCK_RATIO     = 0.70  # lock in 70% of peak
        if peak_r >= _LOCK_ARM_START:
            # Snap peak_r down to nearest 0.1R step so we only move the lock
            # when a new 0.1R milestone is crossed (avoids tick-noise churn).
            snapped_peak = round(int(peak_r / _LOCK_STEP) * _LOCK_STEP, 2)
            target_lock_r = round(max(0.0, snapped_peak * _LOCK_RATIO), 2)
            if target_lock_r > 0 and self._locked_profit_r(t) < target_lock_r:
                candidate_sls.append((self._profit_lock_price(t, target_lock_r), True))
            elif target_lock_r == 0.0 and not t.sl_breakeven:
                # At 0.2R peak, just move to breakeven
                candidate_sls.append((round(_safe_float(t.entry), 2), False))

        h1_high, h1_low = self._recent_swing_levels(market_context.get("h1_df"))
        if t.direction == "BUY" and h1_low is not None:
            candidate_sls.append((h1_low, True))
        if t.direction == "SELL" and h1_high is not None:
            candidate_sls.append((h1_high, True))

        if peak_r >= 2.5:
            m15_high, m15_low = self._recent_swing_levels(market_context.get("m15_df"))
            if t.direction == "BUY" and m15_low is not None:
                candidate_sls.append((m15_low, True))
            if t.direction == "SELL" and m15_high is not None:
                candidate_sls.append((m15_high, True))

        if candidate_sls:
            if t.direction == "BUY":
                best_sl, trailing = max(candidate_sls, key=lambda item: item[0])
            else:
                best_sl, trailing = min(candidate_sls, key=lambda item: item[0])
            self._apply_forward_stop(t, best_sl, mark_trailing=trailing)

        candle_df = market_context.get("m15_df") if peak_r >= 2.5 else market_context.get("h1_df")
        if candle_df is not None and not candle_df.empty:
            last_candle = candle_df.iloc[-1]
            # Skip momentum reversal check if this candle opened after a day break
            # (weekend gap or holiday closure) — the body is stale context, not a reversal
            if not _is_day_break_candle(candle_df):
                body_ratio = _candle_body_ratio(last_candle)
                candle_direction = _candle_direction(last_candle)

                # Determine thresholds based on confidence and config
                min_peak_r = _safe_float(
                    getattr(cfg, "SWING_ENGINE_REVERSAL_MIN_PEAK_R", 1.5), 1.5
                )
                candles_required = _safe_int(
                    getattr(cfg, "SWING_ENGINE_REVERSAL_CANDLES_REQUIRED", 3), 3
                )
                high_conf_min = _safe_float(
                    getattr(cfg, "SWING_ENGINE_REVERSAL_HIGH_CONF_MIN", 0.80), 0.80
                )
                trade_confidence = _safe_float(
                    (t.features or {}).get("signal_confidence") or t.confidence, 0.0
                )
                if trade_confidence >= high_conf_min:
                    body_threshold = _safe_float(
                        getattr(cfg, "SWING_ENGINE_REVERSAL_HIGH_CONF_BODY", 0.85), 0.85
                    )
                else:
                    body_threshold = _safe_float(
                        getattr(cfg, "SWING_ENGINE_REVERSAL_BODY_THRESHOLD", 0.70), 0.70
                    )

                # Only consider reversal exit if trade has reached minimum profit
                if peak_r >= min_peak_r and body_ratio >= body_threshold:
                    is_counter = (
                        (t.direction == "BUY" and candle_direction == "BEARISH") or
                        (t.direction == "SELL" and candle_direction == "BULLISH")
                    )
                    if is_counter:
                        # Suppress reversal exit when D1/H4 HTF bias still agrees with
                        # the trade direction — short-term counter-candles are likely
                        # stop-loss chasers, not a genuine trend reversal.
                        from engine.swing_engine_strategy import _htf_bias_aligned
                        if _htf_bias_aligned(t.direction, market_context):
                            t.features["_reversal_candle_count"] = 0
                        else:
                            count = _safe_int(
                                (t.features or {}).get("_reversal_candle_count", 0), 0
                            ) + 1
                            t.features["_reversal_candle_count"] = count
                            if count >= candles_required:
                                direction_label = "bearish" if t.direction == "BUY" else "bullish"
                                self._close_early(
                                    t,
                                    f"Swing momentum reversal: {count} consecutive strong "
                                    f"{direction_label} candles (body {body_ratio:.2f}, "
                                    f"peak {peak_r:.2f}R, conf {trade_confidence:.0%})",
                                    category="momentum_reversal",
                                )
                                return
                    else:
                        # Aligned candle — reset counter
                        t.features["_reversal_candle_count"] = 0
                else:
                    # Below min_peak_r or body too weak — reset counter
                    t.features["_reversal_candle_count"] = 0

        # ── 6. Recovery exit: profit → meaningful dip → recovery ──────────
        # Requires a real peak, a real dip, and a real recovery — not tick noise.
        min_peak_to_arm = _safe_float(getattr(cfg, "SWING_RECOVERY_MIN_PEAK_R", 0.60), 0.60)
        min_dip_to_arm = _safe_float(getattr(cfg, "SWING_RECOVERY_MIN_DIP_R", 0.25), 0.25)
        min_recovery_r = _safe_float(getattr(cfg, "SWING_RECOVERY_MIN_RECOVERY_R", 0.15), 0.15)
        if peak_r >= min_peak_to_arm and live_r <= -min_dip_to_arm:
            t.features["_seen_loss_after_profit"] = True
        if t.features.get("_seen_loss_after_profit") and live_r >= min_recovery_r:
            self._close_early(
                t,
                f"Swing recovery exit: profit-then-loss-then-profit ({peak_r:.2f}R peak, {live_r:.2f}R now)",
                category="profit_recovery_exit",
            )
            self._pending_reentry_check = t.strategy
            return

        # ── 7. Profit giveback floor — only after a large peak ────────────
        # Requires a substantial peak before protecting against giveback.
        giveback_min_peak = _safe_float(getattr(cfg, "SWING_ENGINE_GIVEBACK_MIN_PEAK_R", 2.0), 2.0)
        giveback_floor = _safe_float(getattr(cfg, "SWING_ENGINE_GIVEBACK_FLOOR_R", 0.8), 0.8)
        if peak_r >= giveback_min_peak and live_r < (peak_r - giveback_floor):
            self._close_early(
                t,
                f"Swing profit giveback ({peak_r:.2f}R -> {live_r:.2f}R)",
                category="profit_giveback",
            )

    def _partial_close(self, t: TradeRecord, volume: float, reason: str):
        tick = mt5.symbol_info_tick(cfg.SYMBOL)
        if not tick:
            return
        close_type = mt5.ORDER_TYPE_SELL if t.direction == "BUY" else mt5.ORDER_TYPE_BUY
        price = tick.bid if t.direction == "BUY" else tick.ask
        request = {
            "action": mt5.TRADE_ACTION_DEAL, "symbol": cfg.SYMBOL,
            "volume": round(volume, 2), "type": close_type,
            "position": t.ticket, "price": price,
            "deviation": cfg.DEVIATION, "magic": cfg.MAGIC_NUMBER,
            "comment": "FT_PARTIAL", "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        mt5.order_send(request)

    def _close_early(self, t: TradeRecord, reason: str, category: str | None = None) -> Dict[str, Any]:
        self._closing_tickets.add(t.ticket)
        res = self.bridge.close_trade(t.ticket)
        if not res.get("success"):
            self._closing_tickets.discard(t.ticket)
        else:
            resolved_category = category or _close_reason_category(reason, t)
            if resolved_category == "tier1_fail":
                self._set_post_tier1_lockout(t, reason)
            # Store close-signal state; manage_all will write authoritative MT5 settlement.
            self._pending_close_reasons[t.ticket] = {
                "reason": reason,
                "category": resolved_category,
                "close_signal_live_pnl": round(t.live_pnl, 2),
                "close_signal_live_r": self._pnl_to_r(t, t.live_pnl),
                "peak_r": self._pnl_to_r(t, t.peak_pnl),
                "held_seconds": round(time.time() - t.fill_ts, 1),
                "profile_name": t.exit_profile,
            }
            close_price = res.get("close_price", 0)
            if close_price:
                self.order_db.update_exit_price(t.ticket, close_price)
        return res

    def _log_closed_trade(
        self,
        trade: TradeRecord,
        pnl: float,
        exit_price: float,
        close_reason: str,
        *,
        close_reason_category: str,
        close_signal_live_pnl: float,
        close_signal_live_r: float,
        peak_r: float,
        final_r: float,
        drawdown_from_peak_r: float,
        drawdown_from_peak_pct: float,
        held_seconds: float,
        mt5_close_reason: str,
        mt5_close_comment: str,
    ):
        """Append closed trade with MFE/MAE to closed_trades.jsonl."""
        try:
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "ticket": trade.ticket,
                "strategy": trade.strategy,
                "direction": trade.direction,
                "entry_price": trade.entry,
                "exit_price": exit_price,
                "sl": trade.sl,
                "tp": trade.tp,
                "volume": trade.initial_volume,
                "sl_distance": trade.sl_distance,
                "pnl": round(pnl, 2),
                "pnl_points": round(
                    (exit_price - trade.entry) if trade.direction == "BUY" else (trade.entry - exit_price), 4
                ) if exit_price else None,
                "duration_seconds": held_seconds,
                "mfe": round(trade.peak_pnl, 2),
                "mae": round(min(0.0, trade.live_pnl), 2),
                "mfe_r": peak_r,
                "peak_r": peak_r,
                "final_r": final_r,
                "close_signal_live_pnl": round(close_signal_live_pnl, 2),
                "close_signal_live_r": close_signal_live_r,
                "drawdown_from_peak_r": drawdown_from_peak_r,
                "drawdown_from_peak_pct": drawdown_from_peak_pct,
                "close_reason": close_reason,
                "close_reason_category": close_reason_category,
                "mt5_close_reason": mt5_close_reason,
                "mt5_close_comment": mt5_close_comment,
                "held_seconds": held_seconds,
                "profile_name": trade.exit_profile,
                "tier1_exit": close_reason_category == "tier1_fail",
            }
            log_closed_trade(entry)
            append_strategy_event(
                trade.strategy,
                {
                    "event": "CLOSE",
                    "ticket": trade.ticket,
                    "strategy": trade.strategy,
                    "direction": trade.direction,
                    "entry_price": round(_safe_float(trade.entry), 2),
                    "exit_price": round(_safe_float(exit_price), 2),
                    "pnl": round(_safe_float(pnl), 2),
                    "close_reason": close_reason,
                    "close_reason_category": close_reason_category,
                    "profile_name": trade.exit_profile,
                },
            )
        except Exception:
            pass

    def _log_adopt(self, ticket: int):
        try:
            with open(os.path.join(_BASE_DIR, "trader.log"), "a") as f:
                f.write(f"[ADOPT] Auto-adopted position #{ticket} from MT5\n")
        except Exception:
            pass
    
    def _log_pnl_validation(self, ticket: int, pnl_data: Dict):
        """Log PnL validation details for audit trail."""
        try:
            with open(os.path.join(_BASE_DIR, "pnl_validation.log"), "a") as f:
                timestamp = datetime.now(timezone.utc).isoformat()
                f.write(f"[{timestamp}] Ticket {ticket}: ")
                f.write(f"PnL=${pnl_data['final_pnl']}, ")
                f.write(f"Confidence={pnl_data['confidence']}, ")
                f.write(f"Notes={pnl_data['validation_notes']}\n")
        except Exception:
            pass

    def close_all(self):
        for ticket in list(self.open_trades.keys()):
            self.bridge.close_trade(ticket)

    def extend_trade_timeout(self, ticket: int, extend_seconds: int) -> Dict[str, Any] | None:
        trade = self.open_trades.get(int(ticket))
        if trade is None:
            return None
        extra = max(0, int(extend_seconds or 0))
        if extra <= 0:
            return {
                "ticket": trade.ticket,
                "timeout_seconds": int(trade.timeout_seconds),
                "manual_timeout_extension_seconds": int(
                    _safe_int((trade.features or {}).get("manual_timeout_extension_seconds"), 0)
                ),
                "remaining_seconds": max(0, int(trade.timeout_seconds - (time.time() - trade.fill_ts))),
            }

        trade.timeout_seconds = max(0, int(trade.timeout_seconds)) + extra
        trade.features["timeout_seconds"] = int(trade.timeout_seconds)
        trade.features["manual_timeout_extension_seconds"] = (
            _safe_int(trade.features.get("manual_timeout_extension_seconds"), 0) + extra
        )
        trade.features["manual_timeout_extended_at"] = datetime.now(timezone.utc).isoformat()
        self._save_state()
        return {
            "ticket": trade.ticket,
            "timeout_seconds": int(trade.timeout_seconds),
            "manual_timeout_extension_seconds": int(trade.features["manual_timeout_extension_seconds"]),
            "remaining_seconds": max(0, int(trade.timeout_seconds - (time.time() - trade.fill_ts))),
        }

    def manually_exit_trade(self, ticket: int, reason: str = "Manual dashboard exit") -> Dict[str, Any] | None:
        trade = self.open_trades.get(int(ticket))
        if trade is None:
            return None
        res = self._close_early(trade, reason, category="manual_exit")
        pending = self._pending_close_reasons.get(trade.ticket)
        if pending is None:
            return {
                "success": False,
                "ticket": trade.ticket,
                "error": res.get("error") or f"Failed to close trade #{trade.ticket}",
            }
        return {
            "success": True,
            "ticket": trade.ticket,
            "reason": pending["reason"],
            "close_reason_category": pending["category"],
            "close_signal_live_pnl": pending["close_signal_live_pnl"],
            "close_signal_live_r": pending["close_signal_live_r"],
            "profile_name": pending["profile_name"],
        }

    # --- Persistence ---

    def _save_state(self):
        try:
            data = {str(k): v.to_dict() for k, v in self.open_trades.items()}
            with open(_STATE_FILE, "w") as f:
                json.dump(data, f, default=str)
        except Exception:
            pass

    def _load_state(self):
        if not os.path.isfile(_STATE_FILE):
            return
        try:
            with open(_STATE_FILE) as f:
                data = json.load(f)
            for k, v in data.items():
                ticket = int(k)
                self.open_trades[ticket] = TradeRecord.from_dict(v)
        except Exception:
            pass

    @property
    def open_count(self) -> int:
        return len(self.open_trades)

    @property
    def status(self) -> Dict:
        today_stats = self.order_db.get_today_stats() if self.order_db else {}
        return {
            "open_trades": [t.to_dict() for t in self.open_trades.values()],
            "open_count": self.open_count,
            "today_stats": today_stats or {},
        }
    
    def get_order_history(self, days: int = 30) -> List[Dict]:
        """Get order history from database."""
        return self.order_db.get_closed_orders(days)
    
    def get_strategy_performance(self, days: int = 30) -> List[Dict]:
        """Get strategy performance breakdown."""
        return self.order_db.get_strategy_performance(days)
    
    def export_snapshot(self, filename: str = None) -> str:
        """Export complete order database snapshot."""
        return self.order_db.export_snapshot(filename)
    
    def get_pnl_validation_summary(self) -> Dict:
        """Get PnL validation summary for all open positions."""
        summary = {
            "total_positions": len(self.open_trades),
            "validation_results": [],
            "account_summary": self.pnl_validator.get_account_pnl_summary()
        }
        
        for ticket in self.open_trades.keys():
            pnl_data = self.pnl_validator.get_accurate_live_pnl(ticket)
            summary["validation_results"].append({
                "ticket": ticket,
                "pnl": pnl_data['pnl'],
                "confidence": pnl_data['confidence'],
                "source": pnl_data['source']
            })
        
        return summary
