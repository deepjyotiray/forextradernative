"""
Optional NVIDIA-backed AI review for live trade candidates.

This service is intentionally conservative:
- It never creates trades or changes direction.
- It only reviews trades that already passed deterministic gates.
- It only blocks when the model explicitly rejects with sufficient confidence.
- It fails open by default so API issues do not halt the trader.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests

import config as cfg


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _clip_text(value: Any, limit: int = 160) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _extract_message_text(payload: Dict[str, Any]) -> str:
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


def _extract_json_object(text: str) -> Dict[str, Any]:
    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("AI response did not contain a JSON object")
    return json.loads(cleaned[start : end + 1])


def build_trade_review_context(
    strategy: str,
    signal: Dict[str, Any],
    market_state: Dict[str, Any],
    gate_result: Dict[str, Any],
) -> Dict[str, Any]:
    tick = market_state.get("tick") or {}
    indicators = market_state.get("indicators") or {}
    bias = market_state.get("bias") or {}
    regime = market_state.get("regime") or {}
    tick_pressure = market_state.get("tick_pressure") or {}
    positions = market_state.get("positions") or []
    account = market_state.get("account") or {}
    m15_context = gate_result.get("m15_context") or {}

    action = str(signal.get("signal") or "").upper()
    fallback_entry = tick.get("ask") if action == "BUY" else tick.get("bid")

    return {
        "generated_at": _utc_now_iso(),
        "strategy": strategy,
        "symbol": market_state.get("symbol") or cfg.SYMBOL,
        "setup": {
            "action": action,
            "confidence": round(_safe_float(signal.get("confidence"), 0.0), 4),
            "entry": round(_safe_float(signal.get("entry"), fallback_entry), 3),
            "sl": round(_safe_float(signal.get("sl"), 0.0), 3),
            "tp": round(_safe_float(signal.get("tp"), 0.0), 3),
            "rr": round(_safe_float(signal.get("rr"), 0.0), 3),
            "sl_distance": round(_safe_float(signal.get("sl_distance"), 0.0), 3),
            "xgb_prob": round(_safe_float(signal.get("xgb_prob", signal.get("_xgb_prob")), 0.0), 4),
            "xgb_trained": bool(signal.get("_xgb_trained", False)),
            "entry_volume_ratio": round(_safe_float(signal.get("_entry_volume_ratio"), 0.0), 3),
            "tick_pressure_score": round(
                _safe_float(
                    signal.get("_entry_tick_pressure_score"),
                    _safe_float(tick_pressure.get("pressure_score"), 0.0),
                ),
                3,
            ),
            "tick_pressure_bias": _clip_text(tick_pressure.get("directional_bias"), 24),
            "reason": _clip_text(signal.get("reason"), 120),
        },
        "gate": {
            "score": _safe_int(gate_result.get("score"), 0),
            "required_score": _safe_int(gate_result.get("required_score"), 0),
            "confirmation_count": _safe_int(gate_result.get("confirmation_count"), 0),
            "required_rr": round(_safe_float(gate_result.get("required_rr"), 0.0), 3),
            "counter_trend": bool(gate_result.get("counter_trend")),
            "premium_only": bool(gate_result.get("premium_only")),
            "session_reason": _clip_text((gate_result.get("session_state") or {}).get("reason"), 40),
            "m15_reason": _clip_text(m15_context.get("reason"), 48),
        },
        "market": {
            "session": _clip_text((gate_result.get("session_state") or {}).get("session"), 24),
            "spread": round(_safe_float(tick.get("spread"), 0.0), 4),
            "spread_mean": round(_safe_float(tick.get("spread_mean", tick.get("spread")), 0.0), 4),
            "spread_percentile": round(_safe_float(tick.get("spread_pctl", tick.get("spread_percentile")), 0.5), 4),
            "atr": round(_safe_float(indicators.get("atr14", indicators.get("atr")), 0.0), 4),
            "atr_ratio": round(_safe_float(indicators.get("atr_ratio"), 0.0), 4),
            "body_ratio": round(_safe_float(indicators.get("body_ratio"), 0.0), 4),
            "rsi": round(_safe_float(indicators.get("rsi"), 0.0), 2),
            "ema9_slope": round(_safe_float(indicators.get("ema9_slope"), 0.0), 4),
            "bias_direction": _clip_text(bias.get("direction"), 16),
            "bias_confidence": round(_safe_float(bias.get("confidence"), 0.0), 4),
            "regime": _clip_text(regime.get("state"), 24),
            "open_positions": len(positions),
            "balance": round(_safe_float(account.get("balance"), 0.0), 2),
            "equity": round(_safe_float(account.get("equity"), 0.0), 2),
        },
    }


class AITradeAdvisorService:
    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        enabled: Optional[bool] = None,
        model: Optional[str] = None,
        endpoint: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        min_seconds_between_calls: Optional[float] = None,
        review_max_confidence: Optional[float] = None,
        review_score_buffer: Optional[int] = None,
        reject_confidence_min: Optional[float] = None,
        fail_open: Optional[bool] = None,
    ):
        env_enabled = os.getenv("NVIDIA_AI_TRADE_ENABLED", "").strip().lower()
        cfg_enabled = bool(getattr(cfg, "AI_TRADE_ADVISOR_ENABLED", False))
        if enabled is not None:
            self.enabled = bool(enabled)
        elif env_enabled:
            self.enabled = env_enabled not in {"0", "false", "off", "no"}
        else:
            self.enabled = cfg_enabled

        self.api_key = (
            api_key
            if api_key is not None
            else os.getenv("NVIDIA_API_KEY", os.getenv("NVAPI_KEY", ""))
        ).strip()
        self.model = (
            model
            or os.getenv("NVIDIA_AI_TRADE_MODEL", "mistralai/mistral-medium-3.5-128b")
        ).strip()
        self.endpoint = (
            endpoint
            or os.getenv("NVIDIA_AI_TRADE_ENDPOINT", "https://integrate.api.nvidia.com/v1/chat/completions")
        ).strip()
        self.timeout_seconds = float(
            timeout_seconds
            if timeout_seconds is not None
            else os.getenv("NVIDIA_AI_TRADE_TIMEOUT_SECONDS", str(getattr(cfg, "AI_TRADE_ADVISOR_TIMEOUT_SECONDS", 3.0)))
        )
        self.min_seconds_between_calls = float(
            min_seconds_between_calls
            if min_seconds_between_calls is not None
            else os.getenv(
                "NVIDIA_AI_TRADE_MIN_SECONDS_BETWEEN_CALLS",
                str(getattr(cfg, "AI_TRADE_ADVISOR_MIN_SECONDS_BETWEEN_CALLS", 20)),
            )
        )
        self.review_max_confidence = float(
            review_max_confidence
            if review_max_confidence is not None
            else os.getenv(
                "NVIDIA_AI_TRADE_MAX_CONFIDENCE",
                str(getattr(cfg, "AI_TRADE_ADVISOR_MAX_CONFIDENCE", 0.72)),
            )
        )
        self.review_score_buffer = int(
            review_score_buffer
            if review_score_buffer is not None
            else os.getenv(
                "NVIDIA_AI_TRADE_SCORE_BUFFER",
                str(getattr(cfg, "AI_TRADE_ADVISOR_SCORE_BUFFER", 5)),
            )
        )
        self.reject_confidence_min = float(
            reject_confidence_min
            if reject_confidence_min is not None
            else os.getenv(
                "NVIDIA_AI_TRADE_REJECT_CONFIDENCE_MIN",
                str(getattr(cfg, "AI_TRADE_ADVISOR_REJECT_CONFIDENCE_MIN", 0.70)),
            )
        )
        cfg_fail_open = bool(getattr(cfg, "AI_TRADE_ADVISOR_FAIL_OPEN", True))
        if fail_open is not None:
            self.fail_open = bool(fail_open)
        else:
            env_fail_open = os.getenv("NVIDIA_AI_TRADE_FAIL_OPEN", "").strip().lower()
            self.fail_open = cfg_fail_open if not env_fail_open else env_fail_open not in {"0", "false", "off", "no"}

        self._last_call_at = 0.0

    def _base_payload(self, strategy: str, signal: Dict[str, Any], gate_result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "model": self.model,
            "strategy": strategy,
            "signal": str(signal.get("signal") or "").upper(),
            "status": "skipped",
            "used": False,
            "should_block": False,
            "decision": "ABSTAIN",
            "confidence": 0.0,
            "reason": "",
            "risk_flags": [],
            "latency_ms": 0,
            "review_trigger": "",
            "gate_score": _safe_int(gate_result.get("score"), 0),
            "required_score": _safe_int(gate_result.get("required_score"), 0),
            "generated_at": _utc_now_iso(),
            "system_text": "",
            "request_text": "",
            "response_text": "",
        }

    def should_review_trade(self, signal: Dict[str, Any], gate_result: Dict[str, Any]) -> Tuple[bool, str]:
        action = str(signal.get("signal") or "").upper()
        if action not in {"BUY", "SELL"}:
            return False, "non_trade_signal"

        confidence = _safe_float(signal.get("confidence"), 0.0)
        gate_score = _safe_int(gate_result.get("score"), 0)
        required_score = _safe_int(gate_result.get("required_score"), gate_score)
        counter_trend = bool(gate_result.get("counter_trend"))
        xgb_trained = bool(signal.get("_xgb_trained", False))
        xgb_prob = _safe_float(signal.get("xgb_prob", signal.get("_xgb_prob")), 0.0)

        if counter_trend:
            return True, "counter_trend_setup"
        if confidence <= self.review_max_confidence:
            return True, f"confidence={confidence:.2f}"
        if gate_score <= required_score + self.review_score_buffer:
            return True, f"score_margin={gate_score - required_score}"
        if xgb_trained and 0.0 < xgb_prob < 0.50:
            return True, f"xgb_prob={xgb_prob:.2f}"
        return False, "clear_high_quality_setup"

    def _build_messages(self, context: Dict[str, Any]) -> List[Dict[str, str]]:
        schema = {
            "decision": "APPROVE | REJECT | ABSTAIN",
            "confidence": "0.0 to 1.0",
            "reason": "short phrase, max 20 words",
            "risk_flags": ["optional short tags"],
        }
        return [
            {
                "role": "system",
                "content": (
                    "You are a conservative trade reviewer for an automated trading system. "
                    "The trade already passed deterministic rules. "
                    "You may only APPROVE, REJECT, or ABSTAIN. "
                    "Never suggest a new trade, never change direction, and never increase risk. "
                    "Base the answer only on the supplied structured data. "
                    "Return JSON only with this schema: "
                    f"{json.dumps(schema)}"
                ),
            },
            {
                "role": "user",
                "content": (
                    "Review this pre-approved trade candidate. "
                    "Reject only if the structured evidence suggests elevated execution or context risk.\n\n"
                    f"{json.dumps(context, sort_keys=True)}"
                ),
            },
        ]

    def _request_review(self, context: Dict[str, Any]) -> Dict[str, Any]:
        messages = self._build_messages(context)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "reasoning_effort": "none",
            "messages": messages,
            "max_tokens": 400,
            "temperature": 0.10,
            "top_p": 1.00,
            "stream": False,
        }
        response = requests.post(
            self.endpoint,
            headers=headers,
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload_json = response.json()
        text = _extract_message_text(payload_json)
        if not text:
            raise RuntimeError("NVIDIA API returned no message content")
        parsed = _extract_json_object(text)
        parsed["_raw_text"] = text
        parsed["_system_text"] = str((messages[0] or {}).get("content") or "").strip()
        parsed["_request_text"] = str((messages[1] or {}).get("content") or "").strip()
        return parsed

    def review_trade(
        self,
        *,
        strategy: str,
        signal: Dict[str, Any],
        market_state: Dict[str, Any],
        gate_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        result = self._base_payload(strategy, signal, gate_result)

        if not self.enabled:
            result["status"] = "disabled"
            result["reason"] = "AI trade advisor disabled"
            return result
        if not self.api_key:
            result["status"] = "disabled"
            result["reason"] = "Set NVIDIA_API_KEY to enable AI trade advisor"
            return result

        should_review, trigger = self.should_review_trade(signal, gate_result)
        result["review_trigger"] = trigger
        if not should_review:
            result["reason"] = trigger
            return result

        now = time.time()
        if self.min_seconds_between_calls > 0 and (now - self._last_call_at) < self.min_seconds_between_calls:
            result["reason"] = "rate_limited"
            return result

        context = build_trade_review_context(strategy, signal, market_state, gate_result)
        started = time.time()
        self._last_call_at = started
        try:
            review = self._request_review(context)
            latency_ms = int((time.time() - started) * 1000)
            decision = str(review.get("decision") or "ABSTAIN").strip().upper()
            if decision not in {"APPROVE", "REJECT", "ABSTAIN"}:
                decision = "ABSTAIN"
            confidence = max(0.0, min(1.0, _safe_float(review.get("confidence"), 0.0)))
            reason = _clip_text(review.get("reason") or review.get("_raw_text") or decision, 160)
            risk_flags = [
                _clip_text(item, 40)
                for item in (review.get("risk_flags") or [])
                if str(item or "").strip()
            ][:4]
            should_block = decision == "REJECT" and confidence >= self.reject_confidence_min
            result.update(
                {
                    "status": "ready",
                    "used": True,
                    "should_block": should_block,
                    "decision": decision,
                    "confidence": round(confidence, 4),
                    "reason": reason,
                    "risk_flags": risk_flags,
                    "latency_ms": latency_ms,
                    "context": context,
                    "system_text": str(review.get("_system_text") or ""),
                    "request_text": str(review.get("_request_text") or ""),
                    "response_text": str(review.get("_raw_text") or ""),
                }
            )
            return result
        except Exception as exc:
            result.update(
                {
                    "status": "error",
                    "used": True,
                    "should_block": False,
                    "decision": "ABSTAIN",
                    "reason": _clip_text(str(exc), 160),
                    "latency_ms": int((time.time() - started) * 1000),
                }
            )
            if not self.fail_open:
                result["should_block"] = True
            return result


ai_trade_advisor_service = AITradeAdvisorService()
