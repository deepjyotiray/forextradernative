"""
Decision Logger - Mandatory logging for all trading decisions.

Logs per decision:
- setup.direction
- bias.direction  
- quality_score
- threshold used
- spread_mean/std
- compression flag
- LTF conflict flag
- trade decision (taken/skipped + reason)
"""
import json
import time
from datetime import datetime, timezone
from typing import Dict, Optional
import os
try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None
from .backtest_context import emit_backtest_decision, is_backtest_mode

_LOG_FILE = "decision_log.jsonl"
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOG_PATH = os.path.join(_BASE_DIR, _LOG_FILE)


def _json_safe(value):
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


def log_decision(
    strategy: str,
    setup_direction: Optional[str],
    bias_direction: Optional[str], 
    quality_score: float,
    threshold: float,
    spread_mean: float,
    spread_std: float,
    compression_ok: bool,
    ltf_conflict: bool,
    decision: str,
    reason: str,
    price: float,
    additional_data: Optional[Dict] = None
):
    """Log a trading decision with all required context."""
    sim_now = None
    if additional_data:
        sim_now = additional_data.get("_sim_now")
    now_dt = datetime.now(timezone.utc)
    if isinstance(sim_now, datetime):
        now_dt = sim_now.astimezone(timezone.utc)

    log_entry = {
        "timestamp": now_dt.isoformat(),
        "unix_time": now_dt.timestamp(),
        "strategy": strategy,
        "setup_direction": setup_direction,
        "bias_direction": bias_direction,
        "quality_score": round(quality_score, 3),
        "threshold": round(threshold, 3),
        "spread_mean": round(spread_mean, 4),
        "spread_std": round(spread_std, 4),
        "compression_ok": compression_ok,
        "ltf_conflict": ltf_conflict,
        "decision": decision,  # "TRADE_TAKEN", "TRADE_SKIPPED"
        "reason": reason,
        "price": round(price, 2),
        "counter_trend": setup_direction != bias_direction if setup_direction and bias_direction else False,
        "session": _get_session(now_dt),
    }
    
    if additional_data:
        log_entry.update(additional_data)

    if is_backtest_mode():
        emit_backtest_decision(log_entry)
        return

    try:
        with open(_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(_json_safe(log_entry), default=str) + "\n")
    except Exception as e:
        print(f"[LOG ERROR] Failed to write decision log: {e}")


def log_smc_decision(
    setup_direction: Optional[str],
    bias_direction: Optional[str],
    quality_score: float, 
    threshold: float,
    spread_mean: float,
    spread_std: float,
    compression_ok: bool,
    ltf_conflict: bool,
    decision: str,
    reason: str,
    price: float,
    setup_features: Optional[Dict] = None,
    signal_data: Optional[Dict] = None
):
    """Log SMC strategy decision."""
    additional = {}
    if setup_features:
        additional["setup_features"] = setup_features
    if signal_data:
        additional["signal_data"] = {
            "sl": signal_data.get("sl"),
            "tp": signal_data.get("tp"), 
            "rr": signal_data.get("rr"),
            "confidence": signal_data.get("confidence")
        }
    
    log_decision(
        strategy="SMC_CONFLUENCE",
        setup_direction=setup_direction,
        bias_direction=bias_direction,
        quality_score=quality_score,
        threshold=threshold,
        spread_mean=spread_mean,
        spread_std=spread_std,
        compression_ok=compression_ok,
        ltf_conflict=ltf_conflict,
        decision=decision,
        reason=reason,
        price=price,
        additional_data=additional
    )


def log_scalper_decision(
    setup_direction: Optional[str],
    quality_score: float,
    spread_mean: float,
    spread_std: float,
    compression_ok: bool,
    decision: str,
    reason: str,
    price: float,
    sweep_level: Optional[float] = None,
    signal_data: Optional[Dict] = None
):
    """Log Sweep Scalper decision."""
    additional = {}
    if sweep_level:
        additional["sweep_level"] = round(sweep_level, 2)
    if signal_data:
        additional["signal_data"] = {
            "sl": signal_data.get("sl"),
            "tp": signal_data.get("tp"),
            "rr": signal_data.get("rr"),
            "confidence": signal_data.get("confidence")
        }
    
    log_decision(
        strategy="SWEEP_SCALPER",
        setup_direction=setup_direction,
        bias_direction=None,  # Scalper doesn't use bias
        quality_score=quality_score,
        threshold=0.65,  # Fixed threshold for scalper
        spread_mean=spread_mean,
        spread_std=spread_std,
        compression_ok=compression_ok,
        ltf_conflict=False,  # Scalper doesn't check LTF conflict
        decision=decision,
        reason=reason,
        price=price,
        additional_data=additional
    )


def _get_session(now_dt: Optional[datetime] = None) -> str:
    """Get current trading session."""
    dt = now_dt or datetime.now(timezone.utc)
    h = dt.hour
    if 7 <= h < 9:
        return "LONDON"
    elif 12 <= h < 13:
        return "OVERLAP" 
    elif 13 <= h < 15:
        return "NY"
    else:
        return "CLOSED"


def get_session_stats(hours: int = 24) -> Dict:
    """Get decision statistics for the last N hours."""
    if not os.path.exists(_LOG_PATH):
        return {"error": "No log file found"}
    
    cutoff = time.time() - (hours * 3600)
    stats = {
        "total_decisions": 0,
        "trades_taken": 0,
        "trades_skipped": 0,
        "by_strategy": {},
        "by_session": {},
        "avg_quality_score": 0,
        "counter_trend_ratio": 0,
    }
    
    try:
        with open(_LOG_PATH, "r", encoding="utf-8") as f:
            entries = []
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    if entry.get("unix_time", 0) >= cutoff:
                        entries.append(entry)
                except:
                    continue
        
        if not entries:
            return stats
        
        stats["total_decisions"] = len(entries)
        stats["trades_taken"] = sum(1 for e in entries if e.get("decision") == "TRADE_TAKEN")
        stats["trades_skipped"] = sum(1 for e in entries if e.get("decision") == "TRADE_SKIPPED")
        
        # By strategy
        for entry in entries:
            strat = entry.get("strategy", "UNKNOWN")
            if strat not in stats["by_strategy"]:
                stats["by_strategy"][strat] = {"taken": 0, "skipped": 0}
            if entry.get("decision") == "TRADE_TAKEN":
                stats["by_strategy"][strat]["taken"] += 1
            else:
                stats["by_strategy"][strat]["skipped"] += 1
        
        # By session
        for entry in entries:
            sess = entry.get("session", "UNKNOWN")
            if sess not in stats["by_session"]:
                stats["by_session"][sess] = {"taken": 0, "skipped": 0}
            if entry.get("decision") == "TRADE_TAKEN":
                stats["by_session"][sess]["taken"] += 1
            else:
                stats["by_session"][sess]["skipped"] += 1
        
        # Quality scores
        quality_scores = [e.get("quality_score", 0) for e in entries]
        stats["avg_quality_score"] = round(sum(quality_scores) / len(quality_scores), 3)
        
        # Counter-trend ratio
        counter_trends = sum(1 for e in entries if e.get("counter_trend", False))
        stats["counter_trend_ratio"] = round(counter_trends / len(entries), 3)
        
    except Exception as e:
        stats["error"] = str(e)
    
    return stats


def get_recent_decisions(limit: int = 20) -> list:
    """Return the most recent decision log entries, newest first."""
    if not os.path.exists(_LOG_PATH):
        return []
    entries = []
    try:
        with open(_LOG_PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entries.append(json.loads(line.strip()))
                except Exception:
                    continue
    except Exception:
        return []
    return list(reversed(entries[-limit:]))


def get_live_blockers(limit: int = 20) -> Dict:
    """Summarize the latest blockers so the UI can explain why no trades are firing."""
    recent = get_recent_decisions(limit)
    skipped = [e for e in recent if e.get("decision") == "TRADE_SKIPPED"]
    latest_by_strategy = {}
    counts = {}

    for entry in skipped:
        strategy = entry.get("strategy", "UNKNOWN")
        if strategy not in latest_by_strategy:
            latest_by_strategy[strategy] = {
                "reason": entry.get("reason", ""),
                "timestamp": entry.get("timestamp"),
                "spread_mean": entry.get("spread_mean"),
                "compression_ok": entry.get("compression_ok"),
            }
        reason = entry.get("reason", "Unknown")
        counts[reason] = counts.get(reason, 0) + 1

    top_reasons = [
        {"reason": reason, "count": count}
        for reason, count in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:5]
    ]

    return {
        "latest_by_strategy": latest_by_strategy,
        "top_reasons": top_reasons,
        "recent_skipped_count": len(skipped),
    }
