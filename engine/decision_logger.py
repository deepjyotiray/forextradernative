"""
Decision Logger - Mandatory logging for all trading decisions.

Daily log files: logs/decision_log_YYYY-MM-DD.jsonl
Auto-rotates to backup_logs/ at day boundary.
"""
import json
import time
import shutil
from datetime import datetime, timezone
from typing import Dict, Optional
import os
import config as cfg
try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None
from .backtest_context import emit_backtest_decision, is_backtest_mode
from .time_utils import date_str_ist, isoformat_ist

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOGS_DIR = os.path.join(_BASE_DIR, "logs")
_BACKUP_DIR = os.path.join(_BASE_DIR, "backup_logs")
_CLOSED_TRADES_FILE = "closed_trades.jsonl"
_CLOSED_TRADES_PATH = os.path.join(_BASE_DIR, _CLOSED_TRADES_FILE)

os.makedirs(_LOGS_DIR, exist_ok=True)
os.makedirs(_BACKUP_DIR, exist_ok=True)

# Track which IST date the current log file is for
_current_log_date: str = ""
_current_log_path: str = ""


def _today_ist() -> str:
    return date_str_ist()


def _get_log_path() -> str:
    """Return today's log path, rotating yesterday's file to backup if the date changed."""
    global _current_log_date, _current_log_path
    today = _today_ist()
    if _current_log_date == today:
        return _current_log_path

    # Date has changed — rotate the previous day's file to backup
    if _current_log_path and os.path.isfile(_current_log_path):
        dest = os.path.join(_BACKUP_DIR, os.path.basename(_current_log_path))
        try:
            shutil.move(_current_log_path, dest)
        except Exception:
            pass

    _current_log_date = today
    _current_log_path = os.path.join(_LOGS_DIR, f"decision_log_{today}.jsonl")
    return _current_log_path


# Also migrate the legacy flat decision_log.jsonl on first import
_LEGACY_LOG = os.path.join(_BASE_DIR, "decision_log.jsonl")
if os.path.isfile(_LEGACY_LOG):
    _legacy_dest = os.path.join(_BACKUP_DIR, "decision_log_pre_rotation.jsonl")
    try:
        shutil.move(_LEGACY_LOG, _legacy_dest)
    except Exception:
        pass

# Initialise current path
_get_log_path()


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
        "timestamp": isoformat_ist(now_dt),
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
        with open(_get_log_path(), "a", encoding="utf-8") as f:
            f.write(json.dumps(_json_safe(log_entry), default=str) + "\n")
    except Exception as e:
        print(f"[LOG ERROR] Failed to write decision log: {e}")


def log_closed_trade(entry: Dict) -> None:
    """Append a closed-trade record to closed_trades.jsonl."""
    if is_backtest_mode():
        return
    try:
        with open(_CLOSED_TRADES_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(_json_safe(entry), default=str) + "\n")
    except Exception as e:
        print(f"[LOG ERROR] Failed to write closed_trades.jsonl: {e}")


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
    """Log scalp-family decision."""
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
        strategy="M15_SCALP_DEEP",
        setup_direction=setup_direction,
        bias_direction=None,  # Scalper doesn't use bias
        quality_score=quality_score,
        threshold=float(getattr(cfg, "SCALPER_QUALITY_THRESHOLD", 0.65) or 0.65),
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
    from engine.session_filter import get_session_state_at

    dt = now_dt or datetime.now(timezone.utc)
    state = get_session_state_at(dt)
    return str(state.get("label") or state.get("session") or "CLOSED")


def get_session_stats(hours: int = 24) -> Dict:
    """Get decision statistics for the last N hours."""
    log_path = _get_log_path()
    if not os.path.exists(log_path):
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
        with open(log_path, "r", encoding="utf-8") as f:
            entries = []
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    if entry.get("unix_time", 0) >= cutoff:
                        entries.append(entry)
                except Exception:
                    continue

        if not entries:
            return stats

        stats["total_decisions"] = len(entries)
        stats["trades_taken"] = sum(1 for e in entries if e.get("decision") == "TRADE_TAKEN")
        stats["trades_skipped"] = sum(1 for e in entries if e.get("decision") == "TRADE_SKIPPED")

        for entry in entries:
            strat = entry.get("strategy", "UNKNOWN")
            if strat not in stats["by_strategy"]:
                stats["by_strategy"][strat] = {"taken": 0, "skipped": 0}
            if entry.get("decision") == "TRADE_TAKEN":
                stats["by_strategy"][strat]["taken"] += 1
            else:
                stats["by_strategy"][strat]["skipped"] += 1

        for entry in entries:
            sess = entry.get("session", "UNKNOWN")
            if sess not in stats["by_session"]:
                stats["by_session"][sess] = {"taken": 0, "skipped": 0}
            if entry.get("decision") == "TRADE_TAKEN":
                stats["by_session"][sess]["taken"] += 1
            else:
                stats["by_session"][sess]["skipped"] += 1

        quality_scores = [e.get("quality_score", 0) for e in entries]
        stats["avg_quality_score"] = round(sum(quality_scores) / len(quality_scores), 3)

        counter_trends = sum(1 for e in entries if e.get("counter_trend", False))
        stats["counter_trend_ratio"] = round(counter_trends / len(entries), 3)

    except Exception as e:
        stats["error"] = str(e)

    return stats


def get_recent_decisions(limit: int = 20) -> list:
    """Return the most recent decision log entries from today's log, newest first."""
    log_path = _get_log_path()
    if not os.path.exists(log_path):
        return []
    entries = []
    try:
        # Read only the tail of the file — no need to load the whole thing
        with open(log_path, "rb") as f:
            f.seek(0, 2)
            file_size = f.tell()
            # Read last 256KB at most — enough for hundreds of recent entries
            read_size = min(file_size, 256 * 1024)
            f.seek(file_size - read_size)
            raw = f.read().decode("utf-8", errors="ignore")
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
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


def get_recent_ai_reviews(limit: int = 20) -> list:
    """Return recent decisions that carried an AI trade review transcript."""
    recent = get_recent_decisions(max(limit * 5, 50))
    rows = []
    for entry in recent:
        ai_status = entry.get("ai_review_status")
        validation_result = entry.get("validation_result") or {}
        ai_review = validation_result.get("ai_trade_review") or {}
        has_transcript = bool(ai_review.get("request_text") or ai_review.get("response_text"))
        if not ai_status and not has_transcript:
            continue
        rows.append({
            "timestamp": entry.get("timestamp"),
            "strategy": entry.get("strategy"),
            "decision": entry.get("decision"),
            "reason": entry.get("reason"),
            "symbol": entry.get("symbol") or cfg.SYMBOL,
            "price": entry.get("price"),
            "signal": entry.get("direction"),
            "signal_confidence": entry.get("signal_confidence"),
            "ai_review_status": entry.get("ai_review_status") or ai_review.get("status"),
            "ai_review_used": bool(entry.get("ai_review_used", ai_review.get("used", False))),
            "ai_review_decision": entry.get("ai_review_decision") or ai_review.get("decision"),
            "ai_review_confidence": entry.get("ai_review_confidence", ai_review.get("confidence")),
            "ai_review_reason": entry.get("ai_review_reason") or ai_review.get("reason"),
            "ai_review_trigger": entry.get("ai_review_trigger") or ai_review.get("review_trigger"),
            "ai_review_model": entry.get("ai_review_model") or ai_review.get("model"),
            "request_text": ai_review.get("request_text") or "",
            "response_text": ai_review.get("response_text") or "",
            "system_text": ai_review.get("system_text") or "",
            "should_block": bool(ai_review.get("should_block", False)),
        })
        if len(rows) >= limit:
            break
    return rows
