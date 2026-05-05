"""
Optional AI commentary for trading analytics.

Builds a compact snapshot from the existing local analytics pipeline and, when
configured with an OpenAI or NVIDIA API key, asks a model for a short
interpretation.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests

from .data_pipeline import get_decision_summary
from .order_database import OrderDatabase
from .reporting_dashboard import generate_comprehensive_report


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
        self._cache: Dict[int, Dict[str, Any]] = {}

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

        cached = self._cache.get(days)
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
            self._cache[days] = {**result, "_cached_at": now}
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


ai_analysis_service = AIAnalysisService()
