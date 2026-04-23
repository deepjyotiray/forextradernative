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

_LOG_FILE = "decision_log.jsonl"
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOG_PATH = os.path.join(_BASE_DIR, _LOG_FILE)


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
    
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "unix_time": time.time(),
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
        "session": _get_session(),
    }
    
    if additional_data:
        log_entry.update(additional_data)
    
    try:
        with open(_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
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


def _get_session() -> str:
    """Get current trading session."""
    h = datetime.now(timezone.utc).hour
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