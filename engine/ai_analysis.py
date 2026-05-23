"""
Optional AI commentary for trading analytics.

Builds a compact snapshot from the existing local analytics pipeline and, when
configured with an OpenAI or NVIDIA API key, asks a model for a short
interpretation.
"""
from __future__ import annotations

import json
import os
import statistics
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import requests

from .data_pipeline import get_decision_summary
from .order_database import OrderDatabase
from .reporting_dashboard import generate_comprehensive_report
from .trade_attribution import get_recent_attributions


_IST = timezone(timedelta(hours=5, minutes=30))


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _attribution_file_path() -> str:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, "trade_attribution.jsonl")


def _should_use_lightweight_snapshot() -> bool:
    threshold_bytes = int(os.getenv("AI_ANALYSIS_MAX_ATTRIBUTION_BYTES", str(100 * 1024 * 1024)))
    try:
        path = _attribution_file_path()
        if not os.path.exists(path):
            return False
        return os.path.getsize(path) > threshold_bytes
    except Exception:
        return False


def _top_items(rows: List[Dict[str, Any]], limit: int = 3) -> List[Dict[str, Any]]:
    return [dict(row) for row in (rows or [])[:limit]]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _parse_ist_date(ist_date: str) -> datetime:
    return datetime.strptime(str(ist_date).strip(), "%Y-%m-%d").replace(tzinfo=_IST)


def _parse_iso_datetime(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        normalized = str(value).replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _mean(values: List[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _normalize_session(session_value: Any) -> str:
    session = str(session_value or "UNKNOWN").upper().strip()
    if session in {"NEW_YORK", "NY"}:
        return "NY"
    return session or "UNKNOWN"


def _orders_to_completed_trades(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    trades: List[Dict[str, Any]] = []
    for row in rows:
        features = row.get("features") or {}
        close_dt = _parse_iso_datetime(row.get("close_time_ist")) or _parse_iso_datetime(row.get("close_time"))
        if not close_dt:
            continue
        pnl = _safe_float(row.get("final_pnl"), 0.0)
        if pnl > 0:
            outcome = "WIN"
        elif pnl < 0:
            outcome = "LOSS"
        else:
            outcome = "BE"

        direction = str(row.get("direction") or "").upper()
        setup_direction = "LONG" if direction in {"BUY", "LONG"} else "SHORT"
        quality_score = features.get("quality_score")
        if quality_score is None:
            quality_score = features.get("signal_confidence")
        trade_type = "COUNTER_TREND" if "counter-trend" in str(row.get("reason") or "").lower() else "WITH_TREND"

        trades.append(
            {
                "timestamp": close_dt.isoformat(),
                "unix_time": close_dt.astimezone(timezone.utc).timestamp(),
                "trade_id": f"order_{row.get('ticket')}",
                "strategy": row.get("strategy") or "UNKNOWN",
                "setup_direction": setup_direction,
                "trade_type": trade_type,
                "quality_score": _safe_float(quality_score, 0.0),
                "tick_ratio": _safe_float(features.get("tick_ratio"), 0.0),
                "tick_velocity": _safe_float(features.get("entry_tick_velocity"), 0.0),
                "spread_mean": _safe_float(features.get("spread"), 0.0),
                "atr_value": _safe_float(features.get("atr"), 0.0),
                "session": _normalize_session(features.get("session") or row.get("session_type")),
                "decision": "TRADE_TAKEN",
                "reason": row.get("reason") or "",
                "entry_price": _safe_float(row.get("entry_price"), 0.0),
                "exit_price": _safe_float(row.get("exit_price"), 0.0),
                "outcome": outcome,
                "pnl": pnl,
                "trade_duration": _safe_float(row.get("held_seconds"), 0.0),
                "completion_time": close_dt.isoformat(),
                "trade_completed": True,
            }
        )
    return trades


def _build_performance_summary(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    winners = [t for t in trades if t.get("outcome") == "WIN"]
    losers = [t for t in trades if t.get("outcome") == "LOSS"]
    be_trades = [t for t in trades if t.get("outcome") == "BE"]
    pnls = [_safe_float(t.get("pnl"), 0.0) for t in trades]
    win_pnls = [_safe_float(t.get("pnl"), 0.0) for t in winners]
    loss_pnls = [_safe_float(t.get("pnl"), 0.0) for t in losers]

    cumulative = []
    running_total = 0.0
    for trade in sorted(trades, key=lambda row: row.get("unix_time", 0)):
        running_total += _safe_float(trade.get("pnl"), 0.0)
        cumulative.append(running_total)

    peak = cumulative[0] if cumulative else 0.0
    max_drawdown = 0.0
    for value in cumulative:
        if value > peak:
            peak = value
        max_drawdown = max(max_drawdown, peak - value)

    best_trade = max(trades, key=lambda row: _safe_float(row.get("pnl"), 0.0), default=None)
    worst_trade = min(trades, key=lambda row: _safe_float(row.get("pnl"), 0.0), default=None)
    loss_total = sum(loss_pnls)

    return {
        "total_trades": len(trades),
        "win_count": len(winners),
        "loss_count": len(losers),
        "be_count": len(be_trades),
        "win_rate": round(len(winners) / len(trades), 3) if trades else 0.0,
        "total_pnl": round(sum(pnls), 2),
        "profit_factor": round(abs(sum(win_pnls) / loss_total), 2) if loss_total else (float("inf") if win_pnls else 0.0),
        "max_drawdown": round(max_drawdown, 2),
        "avg_win": round(_mean(win_pnls), 2),
        "avg_loss": round(_mean(loss_pnls), 2),
        "best_trade": {
            "pnl": round(_safe_float(best_trade.get("pnl"), 0.0), 2),
            "setup": best_trade.get("setup_direction"),
            "session": best_trade.get("session"),
        } if best_trade else None,
        "worst_trade": {
            "pnl": round(_safe_float(worst_trade.get("pnl"), 0.0), 2),
            "setup": worst_trade.get("setup_direction"),
            "session": worst_trade.get("session"),
        } if worst_trade else None,
    }


def _build_directional_analysis(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    def _analyze(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not rows:
            return {"trade_count": 0}
        winners = [t for t in rows if t.get("outcome") == "WIN"]
        pnls = [_safe_float(t.get("pnl"), 0.0) for t in rows]
        durations = [_safe_float(t.get("trade_duration"), 0.0) for t in rows if t.get("trade_duration") is not None]
        return {
            "trade_count": len(rows),
            "win_rate": round(len(winners) / len(rows), 3),
            "total_pnl": round(sum(pnls), 2),
            "avg_pnl": round(_mean(pnls), 2),
            "avg_duration": round(_mean(durations), 0) if durations else 0,
            "best_trade_pnl": round(max(pnls), 2),
            "worst_trade_pnl": round(min(pnls), 2),
        }

    long_trades = [t for t in trades if t.get("setup_direction") == "LONG"]
    short_trades = [t for t in trades if t.get("setup_direction") == "SHORT"]
    long_wr = len([t for t in long_trades if t.get("outcome") == "WIN"]) / len(long_trades) if long_trades else 0.0
    short_wr = len([t for t in short_trades if t.get("outcome") == "WIN"]) / len(short_trades) if short_trades else 0.0

    better_direction = None
    if long_trades and short_trades:
        better_direction = "LONG" if long_wr >= short_wr else "SHORT"
    elif long_trades:
        better_direction = "LONG"
    elif short_trades:
        better_direction = "SHORT"

    return {
        "LONG": _analyze(long_trades),
        "SHORT": _analyze(short_trades),
        "direction_balance": len(long_trades) / max(len(short_trades), 1) if short_trades else (float("inf") if long_trades else 0.0),
        "better_direction": better_direction,
    }


def _build_session_analysis(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    sessions: Dict[str, Dict[str, Any]] = {}
    for session_name in ["LONDON", "OVERLAP", "NY", "UNKNOWN"]:
        rows = [t for t in trades if (t.get("session") or "UNKNOWN") == session_name]
        if rows:
            winners = [t for t in rows if t.get("outcome") == "WIN"]
            pnls = [_safe_float(t.get("pnl"), 0.0) for t in rows]
            sessions[session_name] = {
                "trade_count": len(rows),
                "win_rate": round(len(winners) / len(rows), 3),
                "total_pnl": round(sum(pnls), 2),
                "avg_pnl": round(_mean(pnls), 2),
            }
        else:
            sessions[session_name] = {"trade_count": 0}

    active = {name: stats for name, stats in sessions.items() if stats.get("trade_count", 0) > 0}
    best = max(active.items(), key=lambda item: item[1].get("win_rate", 0.0)) if active else None
    worst = min(active.items(), key=lambda item: item[1].get("win_rate", 0.0)) if active else None
    return {
        "sessions": sessions,
        "best_session": {
            "name": best[0],
            "win_rate": best[1]["win_rate"],
            "trade_count": best[1]["trade_count"],
        } if best else None,
        "worst_session": {
            "name": worst[0],
            "win_rate": worst[1]["win_rate"],
            "trade_count": worst[1]["trade_count"],
        } if worst else None,
    }


def _build_quality_analysis(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    def _analyze(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not rows:
            return {"trade_count": 0}
        winners = [t for t in rows if t.get("outcome") == "WIN"]
        scores = [_safe_float(t.get("quality_score"), 0.0) for t in rows]
        return {
            "trade_count": len(rows),
            "win_rate": round(len(winners) / len(rows), 3),
            "avg_score": round(_mean(scores), 3),
        }

    high = [t for t in trades if _safe_float(t.get("quality_score"), 0.0) >= 0.75]
    medium = [t for t in trades if 0.65 <= _safe_float(t.get("quality_score"), 0.0) < 0.75]
    low = [t for t in trades if _safe_float(t.get("quality_score"), 0.0) < 0.65]
    return {
        "high_quality": _analyze(high),
        "medium_quality": _analyze(medium),
        "low_quality": _analyze(low),
        "quality_effectiveness": bool(high and medium and (_analyze(high)["win_rate"] >= _analyze(medium)["win_rate"])),
    }


def _build_risk_analysis(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    max_consecutive_losses = 0
    current_losses = 0
    durations = []
    for trade in sorted(trades, key=lambda row: row.get("unix_time", 0)):
        if trade.get("outcome") == "LOSS":
            current_losses += 1
            max_consecutive_losses = max(max_consecutive_losses, current_losses)
        else:
            current_losses = 0
        if trade.get("trade_duration") is not None:
            durations.append(_safe_float(trade.get("trade_duration"), 0.0))

    return {
        "max_consecutive_losses": max_consecutive_losses,
        "avg_trade_duration": round(_mean(durations), 0) if durations else 0,
        "longest_trade": round(max(durations), 0) if durations else 0,
        "shortest_trade": round(min(durations), 0) if durations else 0,
        "risk_events": 0,
    }


def _build_day_recommendations(
    completed_trades: List[Dict[str, Any]],
    order_review: Dict[str, Any],
    directional_analysis: Dict[str, Any],
    session_analysis: Dict[str, Any],
    risk_analysis: Dict[str, Any],
) -> List[str]:
    recommendations: List[str] = []
    summary = order_review.get("summary") or {}
    breakeven_review = order_review.get("breakeven_review") or {}
    total_trades = int(summary.get("trades") or 0)
    breakeven_count = int(breakeven_review.get("count") or 0)
    if total_trades and breakeven_count / total_trades >= 0.25:
        recommendations.append(
            f"Breakeven exits were {breakeven_count}/{total_trades} trades; review whether stops moved too aggressively."
        )

    best_session = session_analysis.get("best_session") or {}
    worst_session = session_analysis.get("worst_session") or {}
    if best_session.get("name") and worst_session.get("name") and best_session.get("name") != worst_session.get("name"):
        recommendations.append(
            f"Session edge was uneven: {best_session['name']} outperformed {worst_session['name']}; compare entry quality and spreads."
        )

    long_count = int((directional_analysis.get("LONG") or {}).get("trade_count") or 0)
    short_count = int((directional_analysis.get("SHORT") or {}).get("trade_count") or 0)
    better_direction = directional_analysis.get("better_direction")
    if better_direction and long_count and short_count:
        recommendations.append(
            f"{better_direction} setups led the day; review whether the weaker side showed avoidable counter-bias entries."
        )

    if int(risk_analysis.get("max_consecutive_losses") or 0) >= 2:
        recommendations.append("Losses clustered intraday; inspect whether trade quality degraded after the first losing streak.")

    if not recommendations and completed_trades:
        recommendations.append("Use this day as a replay sample and compare the strongest and weakest trades side by side.")
    return recommendations[:3]


def _build_day_decision_summary(attributions: List[Dict[str, Any]]) -> Dict[str, Any]:
    taken_ids = {str(item.get("trade_id")) for item in attributions if item.get("decision") == "TRADE_TAKEN" and item.get("trade_id")}
    completed_ids = {str(item.get("trade_id")) for item in attributions if item.get("trade_completed") and item.get("trade_id")}
    skipped = [item for item in attributions if item.get("decision") == "TRADE_SKIPPED"]
    skip_reasons: Dict[str, int] = {}
    for item in skipped:
        reason = str(item.get("reason") or "Unknown").strip() or "Unknown"
        skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
    top_skip_reasons = dict(sorted(skip_reasons.items(), key=lambda kv: kv[1], reverse=True)[:10])
    total_decisions = len(taken_ids) + len(skipped)
    return {
        "total_decisions": total_decisions,
        "trades_taken": len(taken_ids),
        "trades_skipped": len(skipped),
        "trades_completed": len(completed_ids),
        "conversion_rate": round(len(taken_ids) / total_decisions, 3) if total_decisions else 0.0,
        "completion_rate": round(len(completed_ids) / len(taken_ids), 3) if taken_ids else 0.0,
        "top_skip_reasons": top_skip_reasons,
    }


def _filter_attributions_for_ist_date(ist_date: str) -> List[Dict[str, Any]]:
    day_start = _parse_ist_date(ist_date)
    day_end = day_start + timedelta(days=1)
    now_utc = datetime.now(timezone.utc)
    lookup_hours = max(24, int((now_utc - day_start.astimezone(timezone.utc)).total_seconds() // 3600) + 24)
    attributions = get_recent_attributions(hours=lookup_hours)
    filtered: List[Dict[str, Any]] = []
    for item in attributions:
        timestamp = _parse_iso_datetime(item.get("timestamp")) or _parse_iso_datetime(item.get("completion_time"))
        if not timestamp:
            continue
        timestamp_ist = timestamp.astimezone(_IST)
        if day_start <= timestamp_ist < day_end:
            filtered.append(item)
    return filtered


def build_analysis_snapshot(days: int) -> Dict[str, Any]:
    """Collect a compact, model-friendly snapshot from existing reports."""
    order_review = OrderDatabase().get_trade_outcome_review(days) or {}
    decision_summary = get_decision_summary(days) or {}
    report_error = None

    if _should_use_lightweight_snapshot():
        report = {}
        report_error = (
            "Skipped full attribution-backed report because trade_attribution.jsonl is too large "
            "for inline AI summary generation."
        )
    else:
        report = generate_comprehensive_report(period_days=days) or {}
        report_error = report.get("error")

    performance_summary = dict(report.get("performance_summary") or {})
    directional_analysis = dict(report.get("directional_analysis") or {})
    session_analysis = dict(report.get("session_analysis") or {})
    quality_analysis = dict(report.get("quality_analysis") or {})
    risk_analysis = dict(report.get("risk_analysis") or {})

    return {
        "generated_at": _utc_now_iso(),
        "period_days": days,
        "performance_summary": {
            "total_trades": performance_summary.get("total_trades", order_review.get("summary", {}).get("trades", 0)),
            "win_rate": performance_summary.get("win_rate", order_review.get("summary", {}).get("win_rate", 0.0)),
            "total_pnl": performance_summary.get("total_pnl", order_review.get("summary", {}).get("total_pnl", 0.0)),
            "profit_factor": performance_summary.get("profit_factor"),
            "max_drawdown": performance_summary.get("max_drawdown"),
            "avg_win": performance_summary.get("avg_win"),
            "avg_loss": performance_summary.get("avg_loss"),
            "best_trade": performance_summary.get("best_trade"),
            "worst_trade": performance_summary.get("worst_trade"),
        },
        "decision_summary": {
            "total_decisions": decision_summary.get("total_decisions", 0),
            "trades_taken": decision_summary.get("trades_taken", 0),
            "trades_skipped": decision_summary.get("trades_skipped", 0),
            "trades_completed": decision_summary.get("trades_completed", 0),
            "conversion_rate": decision_summary.get("conversion_rate", 0.0),
            "completion_rate": decision_summary.get("completion_rate", 0.0),
            "top_skip_reasons": decision_summary.get("top_skip_reasons", {}),
        },
        "directional_analysis": {
            "LONG": directional_analysis.get("LONG", {}),
            "SHORT": directional_analysis.get("SHORT", {}),
            "better_direction": directional_analysis.get("better_direction"),
            "direction_balance": directional_analysis.get("direction_balance"),
        },
        "session_analysis": {
            "best_session": session_analysis.get("best_session"),
            "worst_session": session_analysis.get("worst_session"),
            "sessions": session_analysis.get("sessions", {}),
        },
        "quality_analysis": quality_analysis,
        "risk_analysis": risk_analysis,
        "recommendations": [
            str(item.get("message", "")).strip()
            for item in (report.get("recommendations") or [])
            if str(item.get("message", "")).strip()
        ][:3],
        "order_review": {
            "summary": dict(order_review.get("summary") or {}),
            "close_reason_categories": _top_items(order_review.get("close_reason_categories") or []),
            "strategies": _top_items(order_review.get("strategies") or []),
            "profiles": _top_items(order_review.get("profiles") or []),
            "volume_ratio_buckets": _top_items(order_review.get("volume_ratio_buckets") or []),
            "pressure_score_buckets": _top_items(order_review.get("pressure_score_buckets") or []),
            "breakeven_review": dict(order_review.get("breakeven_review") or {}),
        },
        "report_error": report_error,
    }


def build_analysis_snapshot_for_date(ist_date: str) -> Dict[str, Any]:
    normalized_date = _parse_ist_date(ist_date).strftime("%Y-%m-%d")
    order_db = OrderDatabase()
    day_orders = order_db.get_closed_orders_for_ist_date(normalized_date) or []
    order_review = order_db.get_trade_outcome_review_for_ist_date(normalized_date) or {}
    attributions = _filter_attributions_for_ist_date(normalized_date)
    completed_trades = _orders_to_completed_trades(day_orders)

    performance_summary = _build_performance_summary(completed_trades)
    directional_analysis = _build_directional_analysis(completed_trades)
    session_analysis = _build_session_analysis(completed_trades)
    quality_analysis = _build_quality_analysis(completed_trades)
    risk_analysis = _build_risk_analysis(completed_trades)
    recommendations = _build_day_recommendations(
        completed_trades,
        order_review,
        directional_analysis,
        session_analysis,
        risk_analysis,
    )

    return {
        "generated_at": _utc_now_iso(),
        "period_days": 1,
        "target_ist_date": normalized_date,
        "scope": "ist_day",
        "performance_summary": performance_summary,
        "decision_summary": _build_day_decision_summary(attributions),
        "directional_analysis": directional_analysis,
        "session_analysis": session_analysis,
        "quality_analysis": quality_analysis,
        "risk_analysis": risk_analysis,
        "recommendations": recommendations,
        "order_review": {
            "summary": dict(order_review.get("summary") or {}),
            "close_reason_categories": _top_items(order_review.get("close_reason_categories") or []),
            "strategies": _top_items(order_review.get("strategies") or []),
            "profiles": _top_items(order_review.get("profiles") or []),
            "volume_ratio_buckets": _top_items(order_review.get("volume_ratio_buckets") or []),
            "pressure_score_buckets": _top_items(order_review.get("pressure_score_buckets") or []),
            "breakeven_review": dict(order_review.get("breakeven_review") or {}),
        },
        "report_error": (
            None
            if attributions
            else "Decision-attribution records were unavailable for the selected IST day; completed-trade performance was built from orders.db."
        ),
    }


def _build_analysis_prompt(snapshot: Dict[str, Any]) -> str:
    perf = snapshot.get("performance_summary") or {}
    decision = snapshot.get("decision_summary") or {}
    directional = snapshot.get("directional_analysis") or {}
    session = snapshot.get("session_analysis") or {}
    order_review = snapshot.get("order_review") or {}
    strategies = order_review.get("strategies") or []
    close_reasons = order_review.get("close_reason_categories") or []
    recommendations = snapshot.get("recommendations") or []

    def _format_top_rows(rows: List[Dict[str, Any]], key_name: str) -> str:
        formatted = []
        for row in rows[:3]:
            label = row.get(key_name) or "unknown"
            trades = row.get("trades", 0)
            pnl = row.get("total_pnl", 0)
            win_rate = row.get("win_rate")
            if win_rate is None:
                formatted.append(f"{label}: trades={trades}, pnl={pnl}")
            else:
                formatted.append(f"{label}: trades={trades}, win_rate={win_rate}, pnl={pnl}")
        return "; ".join(formatted) if formatted else "none"

    return (
        "Review this automated trading analytics snapshot.\n"
        "Base every statement on the supplied numbers only.\n"
        "Do not give discretionary market predictions or suggest increasing risk.\n"
        "Keep the output under 160 words.\n"
        "Format:\n"
        "- One short overall assessment.\n"
        "- Two evidence-based observations.\n"
        "- Two practical next checks or tuning ideas.\n"
        "Finish with a final line starting with 'Priority:'.\n\n"
        f"Target IST date: {snapshot.get('target_ist_date') or 'n/a'}\n"
        f"Period days: {snapshot.get('period_days')}\n"
        f"Performance: trades={perf.get('total_trades')}, win_rate={perf.get('win_rate')}, total_pnl={perf.get('total_pnl')}, "
        f"profit_factor={perf.get('profit_factor')}, max_drawdown={perf.get('max_drawdown')}\n"
        f"Decisions: total={decision.get('total_decisions')}, taken={decision.get('trades_taken')}, skipped={decision.get('trades_skipped')}, "
        f"completed={decision.get('trades_completed')}\n"
        f"Directional better side: {directional.get('better_direction')}\n"
        f"Best session: {json.dumps(session.get('best_session'))}\n"
        f"Worst session: {json.dumps(session.get('worst_session'))}\n"
        f"Top strategies: {_format_top_rows(strategies, 'strategy')}\n"
        f"Top close reasons: {_format_top_rows(close_reasons, 'close_reason_category')}\n"
        f"Recommendations: {'; '.join(str(r) for r in recommendations[:3]) if recommendations else 'none'}\n"
        f"Report note: {snapshot.get('report_error') or 'none'}"
    )


def _extract_response_text(payload: Dict[str, Any]) -> str:
    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    parts: List[str] = []
    for item in payload.get("output", []) or []:
        for content in item.get("content", []) or []:
            if content.get("type") == "output_text":
                text = str(content.get("text", "")).strip()
                if text:
                    parts.append(text)
    return "\n".join(parts).strip()


def _extract_chat_completion_text(payload: Dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                text = str(item.get("text") or item.get("content") or "").strip()
                if text:
                    parts.append(text)
        return "\n".join(parts).strip()
    return ""


class AIAnalysisService:
    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        enabled: Optional[bool] = None,
        model: Optional[str] = None,
        endpoint: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        cache_ttl_seconds: Optional[int] = None,
    ):
        env_enabled = os.getenv("OPENAI_AI_ANALYSIS_ENABLED", "1").strip().lower()
        openai_key = os.getenv("OPENAI_API_KEY", "").strip()
        nvidia_key = os.getenv("NVIDIA_API_KEY", os.getenv("NVAPI_KEY", "")).strip()
        resolved_api_key = (api_key if api_key is not None else (openai_key or nvidia_key)).strip()
        self.enabled = (
            enabled
            if enabled is not None
            else env_enabled not in {"0", "false", "off", "no"}
        )
        endpoint_hint = str(endpoint or "").strip().lower()
        if "integrate.api.nvidia.com" in endpoint_hint:
            self.provider = "nvidia"
        elif api_key is not None:
            self.provider = "openai"
        elif openai_key:
            self.provider = "openai"
        elif nvidia_key:
            self.provider = "nvidia"
        else:
            self.provider = "openai"
        self.api_key = resolved_api_key
        default_model = (
            os.getenv("OPENAI_AI_ANALYSIS_MODEL", "gpt-5.4-mini")
            if self.provider == "openai"
            else os.getenv("NVIDIA_AI_ANALYSIS_MODEL", "mistralai/mistral-medium-3.5-128b")
        )
        self.model = (model or default_model).strip()
        default_endpoint = (
            os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
            if self.provider == "openai"
            else os.getenv("NVIDIA_AI_ANALYSIS_ENDPOINT", "https://integrate.api.nvidia.com/v1/chat/completions")
        )
        self.endpoint = (endpoint or default_endpoint).rstrip("/")
        default_timeout = (
            os.getenv("OPENAI_AI_ANALYSIS_TIMEOUT_SECONDS", "20")
            if self.provider == "openai"
            else os.getenv("NVIDIA_AI_ANALYSIS_TIMEOUT_SECONDS", "60")
        )
        self.timeout_seconds = float(timeout_seconds or default_timeout)
        self.cache_ttl_seconds = int(cache_ttl_seconds or os.getenv("OPENAI_AI_ANALYSIS_CACHE_TTL_SECONDS", "300"))
        self._cache: Dict[str, Dict[str, Any]] = {}

    def _disabled_payload(self, days: int, reason: str) -> Dict[str, Any]:
        return {
            "enabled": False,
            "status": "disabled",
            "days": days,
            "model": self.model,
            "provider": self.provider,
            "generated_at": _utc_now_iso(),
            "message": reason,
            "analysis": "",
        }

    def _disabled_day_payload(self, ist_date: str, reason: str) -> Dict[str, Any]:
        return {
            "enabled": False,
            "status": "disabled",
            "target_ist_date": ist_date,
            "model": self.model,
            "provider": self.provider,
            "generated_at": _utc_now_iso(),
            "message": reason,
            "analysis": "",
        }

    def _request_openai_analysis(self, snapshot: Dict[str, Any]) -> str:
        payload = {
            "model": self.model,
            "instructions": (
                "You are a trading system analyst. "
                "Summarize performance, identify patterns, and suggest low-risk follow-up checks."
            ),
            "input": _build_analysis_prompt(snapshot),
            "max_output_tokens": 240,
        }
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.endpoint}/responses",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI API error {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"OpenAI API connection failed: {exc.reason}") from exc

        text = _extract_response_text(response_payload)
        if not text:
            raise RuntimeError("OpenAI API returned no text output")
        return text

    def _request_nvidia_analysis(self, snapshot: Dict[str, Any]) -> str:
        payload = {
            "model": self.model,
            "reasoning_effort": "none",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a trading system analyst. "
                        "Summarize performance, identify patterns, and suggest low-risk follow-up checks."
                    ),
                },
                {
                    "role": "user",
                    "content": _build_analysis_prompt(snapshot),
                },
            ],
            "max_tokens": 240,
            "temperature": 0.20,
            "top_p": 1.00,
            "stream": False,
        }
        try:
            response = requests.post(
                self.endpoint,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            response_payload = response.json()
        except requests.HTTPError as exc:
            detail = ""
            try:
                detail = exc.response.text
            except Exception:
                detail = str(exc)
            raise RuntimeError(f"NVIDIA API error: {detail}") from exc
        except requests.RequestException as exc:
            raise RuntimeError(f"NVIDIA API connection failed: {exc}") from exc

        text = _extract_chat_completion_text(response_payload)
        if not text:
            raise RuntimeError("NVIDIA API returned no text output")
        return text

    def generate_summary(self, days: int = 30, refresh: bool = False) -> Dict[str, Any]:
        if not self.enabled:
            return self._disabled_payload(days, "AI analysis is disabled. Set OPENAI_AI_ANALYSIS_ENABLED=1 to enable it.")
        if not self.api_key:
            return self._disabled_payload(days, "Set OPENAI_API_KEY or NVIDIA_API_KEY to enable AI analysis.")

        cache_key = f"days:{days}"
        cached = self._cache.get(cache_key)
        now = time.time()
        if cached and not refresh and (now - float(cached.get("_cached_at", 0))) < self.cache_ttl_seconds:
            result = dict(cached)
            result["cached"] = True
            result.pop("_cached_at", None)
            return result

        snapshot = build_analysis_snapshot(days)
        try:
            analysis = (
                self._request_openai_analysis(snapshot)
                if self.provider == "openai"
                else self._request_nvidia_analysis(snapshot)
            )
            result = {
                "enabled": True,
                "status": "ready",
                "days": days,
                "model": self.model,
                "provider": self.provider,
                "generated_at": _utc_now_iso(),
                "cached": False,
                "analysis": analysis,
                "snapshot": snapshot,
            }
            self._cache[cache_key] = {**result, "_cached_at": now}
            return result
        except Exception as exc:
            return {
                "enabled": True,
                "status": "error",
                "days": days,
                "model": self.model,
                "provider": self.provider,
                "generated_at": _utc_now_iso(),
                "cached": False,
                "message": "AI analysis failed. Base analytics are still available.",
                "error": str(exc),
                "analysis": "",
                "snapshot": snapshot,
            }

    def generate_day_summary(self, ist_date: str, refresh: bool = False) -> Dict[str, Any]:
        normalized_date = _parse_ist_date(ist_date).strftime("%Y-%m-%d")
        if not self.enabled:
            return self._disabled_day_payload(normalized_date, "AI analysis is disabled. Set OPENAI_AI_ANALYSIS_ENABLED=1 to enable it.")
        if not self.api_key:
            return self._disabled_day_payload(normalized_date, "Set OPENAI_API_KEY or NVIDIA_API_KEY to enable AI analysis.")

        cache_key = f"date:{normalized_date}"
        cached = self._cache.get(cache_key)
        now = time.time()
        if cached and not refresh and (now - float(cached.get("_cached_at", 0))) < self.cache_ttl_seconds:
            result = dict(cached)
            result["cached"] = True
            result.pop("_cached_at", None)
            return result

        snapshot = build_analysis_snapshot_for_date(normalized_date)
        try:
            analysis = (
                self._request_openai_analysis(snapshot)
                if self.provider == "openai"
                else self._request_nvidia_analysis(snapshot)
            )
            result = {
                "enabled": True,
                "status": "ready",
                "target_ist_date": normalized_date,
                "model": self.model,
                "provider": self.provider,
                "generated_at": _utc_now_iso(),
                "cached": False,
                "analysis": analysis,
                "snapshot": snapshot,
            }
            self._cache[cache_key] = {**result, "_cached_at": now}
            return result
        except Exception as exc:
            return {
                "enabled": True,
                "status": "error",
                "target_ist_date": normalized_date,
                "model": self.model,
                "provider": self.provider,
                "generated_at": _utc_now_iso(),
                "cached": False,
                "message": "AI analysis failed. Base analytics are still available.",
                "error": str(exc),
                "analysis": "",
                "snapshot": snapshot,
            }


ai_analysis_service = AIAnalysisService()
