"""
Trade Attribution Engine - Comprehensive logging for every trade decision.

Logs for every decision (taken or skipped):
- setup.direction, bias.direction, trade_type
- quality_score, threshold_used
- spread metrics, ATR, compression/LTF flags
- tick metrics, session

For executed trades, also logs:
- entry/exit prices, SL/TP, outcome, PnL
- trade_duration, exit_reason
"""
import json
import time
import math
import shutil
from datetime import datetime, timezone
from typing import Dict, Optional, List
import os
try:
    import numpy as np
except Exception:  # pragma: no cover - numpy should exist, but keep logger resilient
    np = None
from .backtest_context import get_backtest_now, is_backtest_mode
from .time_utils import isoformat_ist

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ATTRIBUTION_FILE = os.path.join(_BASE_DIR, "trade_attribution.jsonl")
_BACKUP_DIR = os.path.join(_BASE_DIR, "backup_logs")
_ATTRIBUTION_MAX_BYTES = int(os.getenv("TRADE_ATTRIBUTION_MAX_BYTES", str(512 * 1024 * 1024)))
_ATTRIBUTION_LINE_MAX_BYTES = int(os.getenv("TRADE_ATTRIBUTION_LINE_MAX_BYTES", str(512 * 1024)))
_ATTRIBUTION_FALLBACK_FILE = os.path.join(_BASE_DIR, "trade_attribution_fallback.jsonl")

os.makedirs(_BACKUP_DIR, exist_ok=True)


def _json_safe(value):
    """Recursively normalize NumPy/scalar values into JSON-serializable types."""
    if np is not None:
        if isinstance(value, (np.bool_, np.integer, np.floating)):
            return value.item()
        if isinstance(value, np.ndarray):
            return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return value


def _is_finite_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _shrink_for_log(value, *, max_depth: int = 6, max_items: int = 40, max_string: int = 2000):
    """Keep attribution payloads informative but bounded for reliable file writes."""
    if max_depth <= 0:
        text = str(value)
        return text[:max_string] + ("...<truncated>" if len(text) > max_string else "")

    if value is None or isinstance(value, (bool, int)):
        return value

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, str):
        return value[:max_string] + ("...<truncated>" if len(value) > max_string else "")

    if isinstance(value, dict):
        items = list(value.items())
        shrunk = {
            str(k): _shrink_for_log(v, max_depth=max_depth - 1, max_items=max_items, max_string=max_string)
            for k, v in items[:max_items]
        }
        if len(items) > max_items:
            shrunk["__truncated_items__"] = len(items) - max_items
        return shrunk

    if isinstance(value, (list, tuple, set)):
        seq = list(value)
        shrunk = [
            _shrink_for_log(v, max_depth=max_depth - 1, max_items=max_items, max_string=max_string)
            for v in seq[:max_items]
        ]
        if len(seq) > max_items:
            shrunk.append(f"...<{len(seq) - max_items} more items>")
        return shrunk

    try:
        return _shrink_for_log(_json_safe(value), max_depth=max_depth - 1, max_items=max_items, max_string=max_string)
    except Exception:
        text = str(value)
        return text[:max_string] + ("...<truncated>" if len(text) > max_string else "")


def _serialize_attribution(attribution: Dict) -> bytes:
    shrunk = _shrink_for_log(_json_safe(attribution))
    payload = json.dumps(shrunk, default=str, ensure_ascii=False, separators=(",", ":"))
    encoded = payload.encode("utf-8", errors="replace")
    if len(encoded) <= _ATTRIBUTION_LINE_MAX_BYTES:
        return encoded

    minimal = {
        "timestamp": shrunk.get("timestamp"),
        "unix_time": shrunk.get("unix_time"),
        "strategy": shrunk.get("strategy"),
        "setup_direction": shrunk.get("setup_direction"),
        "bias_direction": shrunk.get("bias_direction"),
        "trade_type": shrunk.get("trade_type"),
        "decision": shrunk.get("decision"),
        "reason": shrunk.get("reason"),
        "price": shrunk.get("price"),
        "trade_id": shrunk.get("trade_id"),
        "trade_completed": shrunk.get("trade_completed", False),
        "payload_truncated": True,
        "payload_original_bytes": len(encoded),
    }
    return json.dumps(minimal, default=str, ensure_ascii=False, separators=(",", ":")).encode("utf-8", errors="replace")


def _rotate_if_needed(path: str) -> None:
    try:
        if not os.path.exists(path):
            return
        size = os.path.getsize(path)
        if size < _ATTRIBUTION_MAX_BYTES:
            return
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        dest = os.path.join(_BACKUP_DIR, f"trade_attribution_{stamp}.jsonl")
        shutil.move(path, dest)
    except Exception:
        pass


def _append_bytes(path: str, data: bytes) -> None:
    with open(path, "ab") as f:
        f.write(data)
        f.write(b"\n")


def _candidate_attribution_paths() -> List[str]:
    paths: List[str] = []
    for candidate in [_ATTRIBUTION_FILE, _ATTRIBUTION_FALLBACK_FILE]:
        if os.path.exists(candidate):
            paths.append(candidate)
    try:
        rotated = sorted(
            [
                os.path.join(_BACKUP_DIR, name)
                for name in os.listdir(_BACKUP_DIR)
                if name.startswith("trade_attribution_") and name.endswith(".jsonl")
            ],
            key=lambda item: os.path.getmtime(item),
            reverse=True,
        )
        paths.extend(rotated[:7])
    except Exception:
        pass
    return paths


class TradeAttributionEngine:
    def __init__(self):
        self._active_trades = {}  # {trade_id: attribution_data}
        self._trade_counter = 0
    
    def log_decision(
        self,
        strategy: str,
        setup_direction: Optional[str],
        bias_direction: Optional[str],
        quality_score: float,
        threshold_used: float,
        spread_mean: float,
        spread_std: float,
        spread_percentile: float,
        atr_value: float,
        compression_flag: bool,
        ltf_conflict_flag: bool,
        tick_ratio: float,
        tick_velocity: float,
        session: str,
        decision: str,  # "TRADE_TAKEN" or "TRADE_SKIPPED"
        reason: str,
        price: float,
        additional_data: Optional[Dict] = None
    ) -> Optional[str]:
        """Log a trade decision. Returns trade_id if trade was taken."""
        
        # Determine trade type
        trade_type = "NEUTRAL"
        if setup_direction and bias_direction and bias_direction != "NEUTRAL":
            if setup_direction == bias_direction:
                trade_type = "WITH_TREND"
            else:
                trade_type = "COUNTER_TREND"
        
        attribution = {
            "timestamp": isoformat_ist(_now_utc()),
            "unix_time": _now_utc().timestamp(),
            "strategy": strategy,
            "setup_direction": setup_direction,
            "bias_direction": bias_direction,
            "trade_type": trade_type,
            "quality_score": round(quality_score, 3),
            "threshold_used": round(threshold_used, 3),
            "spread_mean": round(spread_mean, 4),
            "spread_std": round(spread_std, 4),
            "spread_percentile": round(spread_percentile, 3),
            "atr_value": round(atr_value, 4),
            "compression_flag": compression_flag,
            "ltf_conflict_flag": ltf_conflict_flag,
            "tick_ratio": round(tick_ratio, 3),
            "tick_velocity": round(tick_velocity, 2),
            "session": session,
            "decision": decision,
            "reason": reason,
            "price": round(price, 2),
        }
        
        if additional_data:
            attribution.update(additional_data)
        
        trade_id = None
        if decision == "TRADE_TAKEN":
            self._trade_counter += 1
            trade_id = f"{strategy}_{int(_now_utc().timestamp())}_{self._trade_counter}"
            attribution["trade_id"] = trade_id
            self._active_trades[trade_id] = attribution
        
        self._write_attribution(attribution)
        return trade_id
    
    def log_trade_outcome(
        self,
        trade_id: str,
        entry_price: float,
        exit_price: float,
        sl: float,
        tp: float,
        outcome: str,  # "WIN", "LOSS", "BE"
        pnl: float,
        trade_duration: int,  # seconds
        exit_reason: str  # "TP", "SL", "TIMEOUT", "EARLY_EXIT"
    ):
        """Log the outcome of an executed trade."""
        
        if trade_id not in self._active_trades:
            return
        
        attribution = self._active_trades[trade_id].copy()
        attribution.update({
            "trade_completed": True,
            "entry_price": round(entry_price, 2),
            "exit_price": round(exit_price, 2),
            "sl": round(sl, 2),
            "tp": round(tp, 2),
            "outcome": outcome,
            "pnl": round(pnl, 2),
            "trade_duration": trade_duration,
            "exit_reason": exit_reason,
            "completion_time": isoformat_ist(_now_utc()),
        })
        
        self._write_attribution(attribution)
        del self._active_trades[trade_id]
    
    def _write_attribution(self, attribution: Dict):
        """Write attribution data to file."""
        if is_backtest_mode():
            return
        payload = None
        try:
            _rotate_if_needed(_ATTRIBUTION_FILE)
            payload = _serialize_attribution(attribution)
            _append_bytes(_ATTRIBUTION_FILE, payload)
        except Exception as e:
            try:
                _rotate_if_needed(_ATTRIBUTION_FILE)
                if payload is None:
                    payload = _serialize_attribution(attribution)
                _append_bytes(_ATTRIBUTION_FALLBACK_FILE, payload)
                print(f"[ATTRIBUTION WARN] Primary write failed, used fallback file: {e}")
            except Exception as fallback_error:
                print(f"[ATTRIBUTION ERROR] Failed to write: {e} | fallback failed: {fallback_error}")
    
    def get_recent_attributions(self, hours: int = 24) -> List[Dict]:
        """Get recent attribution data."""
        if is_backtest_mode():
            return []
        candidate_paths = _candidate_attribution_paths()
        if not candidate_paths:
            return []
        
        cutoff = time.time() - (hours * 3600)
        attributions = []
        
        try:
            for path in candidate_paths:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        try:
                            attr = json.loads(line.strip())
                            if attr.get("unix_time", 0) >= cutoff:
                                attributions.append(attr)
                        except Exception:
                            continue
        except Exception:
            pass
        
        attributions.sort(key=lambda row: row.get("unix_time", 0))
        return attributions
    
    def get_active_trades(self) -> Dict:
        """Get currently active trades."""
        return self._active_trades.copy()


# Singleton instance
attribution_engine = TradeAttributionEngine()


def log_trade_decision(
    strategy: str,
    setup_direction: Optional[str],
    bias_direction: Optional[str],
    quality_score: float,
    threshold_used: float,
    spread_data: Dict,
    atr_value: float,
    compression_flag: bool,
    ltf_conflict_flag: bool,
    tick_data: Dict,
    session: str,
    decision: str,
    reason: str,
    price: float,
    additional_data: Optional[Dict] = None
) -> Optional[str]:
    """Convenience function to log trade decisions."""
    
    return attribution_engine.log_decision(
        strategy=strategy,
        setup_direction=setup_direction,
        bias_direction=bias_direction,
        quality_score=quality_score,
        threshold_used=threshold_used,
        spread_mean=spread_data.get("mean", 0),
        spread_std=spread_data.get("std", 0),
        spread_percentile=spread_data.get("percentile", 0),
        atr_value=atr_value,
        compression_flag=compression_flag,
        ltf_conflict_flag=ltf_conflict_flag,
        tick_ratio=tick_data.get("ratio", 0.5),
        tick_velocity=tick_data.get("velocity", 0),
        session=session,
        decision=decision,
        reason=reason,
        price=price,
        additional_data=additional_data
    )


def log_trade_outcome(
    trade_id: str,
    entry_price: float,
    exit_price: float,
    sl: float,
    tp: float,
    outcome: str,
    pnl: float,
    trade_duration: int,
    exit_reason: str
):
    """Convenience function to log trade outcomes."""
    
    attribution_engine.log_trade_outcome(
        trade_id=trade_id,
        entry_price=entry_price,
        exit_price=exit_price,
        sl=sl,
        tp=tp,
        outcome=outcome,
        pnl=pnl,
        trade_duration=trade_duration,
        exit_reason=exit_reason
    )


def get_recent_attributions(hours: int = 24) -> List[Dict]:
    """Get recent attribution data."""
    return attribution_engine.get_recent_attributions(hours)


def get_active_trades() -> Dict:
    """Get currently active trades."""
    return attribution_engine.get_active_trades()


def _now_utc() -> datetime:
    override = get_backtest_now()
    if isinstance(override, datetime):
        return override.astimezone(timezone.utc)
    return datetime.now(timezone.utc)
