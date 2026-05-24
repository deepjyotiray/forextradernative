"""
On-demand OpenAI-backed manual trade ideas for manual execution.

This service is advisory only:
- It never auto-places trades.
- It returns a single manual idea or WAIT.
- Execution remains a separate explicit API action.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

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


def infer_pending_order_type(action: str, entry: float, tick: Dict[str, Any]) -> str:
    side = str(action or "").upper()
    bid = _safe_float((tick or {}).get("bid"), 0.0)
    ask = _safe_float((tick or {}).get("ask"), 0.0)
    if side == "BUY":
        return "BUY_LIMIT" if entry < max(ask, bid) else "BUY_STOP"
    if side == "SELL":
        return "SELL_LIMIT" if entry > min(bid or entry, ask or entry) else "SELL_STOP"
    return "NONE"


def _compress_candles(candles: List[Dict[str, Any]], keep: int = 24) -> List[Dict[str, float]]:
    compact: List[Dict[str, float]] = []
    for candle in (candles or [])[-keep:]:
        compact.append(
            {
                "o": round(_safe_float(candle.get("open"), 0.0), 3),
                "h": round(_safe_float(candle.get("high"), 0.0), 3),
                "l": round(_safe_float(candle.get("low"), 0.0), 3),
                "c": round(_safe_float(candle.get("close"), 0.0), 3),
                "v": round(_safe_float(candle.get("volume"), 0.0), 0),
            }
        )
    return compact


def build_manual_trade_context(
    *,
    symbol: str,
    timeframe: str,
    tick: Dict[str, Any],
    session: str,
    market_open: bool,
    indicators: Dict[str, Any],
    regime: Dict[str, Any],
    bias: Dict[str, Any],
    liquidity: Dict[str, Any],
    tick_pressure: Dict[str, Any],
    xgb_live_prediction: Dict[str, Any],
    account: Dict[str, Any],
    candles: List[Dict[str, Any]],
    support_zones: List[Dict[str, Any]],
    resistance_zones: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "generated_at": _utc_now_iso(),
        "symbol": symbol,
        "timeframe": timeframe,
        "market": {
            "session": str(session or ""),
            "market_open": bool(market_open),
            "bid": round(_safe_float(tick.get("bid"), 0.0), 3),
            "ask": round(_safe_float(tick.get("ask"), 0.0), 3),
            "spread": round(_safe_float(tick.get("spread"), 0.0), 3),
        },
        "indicators": {
            "ema9": round(_safe_float(indicators.get("ema9"), 0.0), 3),
            "ema15": round(_safe_float(indicators.get("ema15"), 0.0), 3),
            "ema20": round(_safe_float(indicators.get("ema20"), 0.0), 3),
            "ema9_slope": round(_safe_float(indicators.get("ema9_slope"), 0.0), 4),
            "ema20_slope": round(_safe_float(indicators.get("ema20_slope"), 0.0), 4),
            "atr14": round(_safe_float(indicators.get("atr14", indicators.get("atr")), 0.0), 4),
            "atr_ratio": round(_safe_float(indicators.get("atr_ratio"), 0.0), 4),
            "rsi": round(_safe_float(indicators.get("rsi"), 0.0), 2),
            "body_ratio": round(_safe_float(indicators.get("body_ratio"), 0.0), 4),
            "range_10": round(_safe_float(indicators.get("range_10"), 0.0), 4),
        },
        "regime": {
            "state": _clip_text(regime.get("state"), 24),
            "direction": _clip_text(regime.get("direction"), 16),
            "trade_allowed": bool(regime.get("trade_allowed", True)),
            "atr_ratio": round(_safe_float(regime.get("atr_ratio"), 0.0), 4),
            "reasons": [_clip_text(item, 48) for item in (regime.get("reasons") or [])[:4]],
        },
        "bias": {
            "direction": _clip_text(bias.get("direction"), 16),
            "confidence": round(_safe_float(bias.get("confidence"), 0.0), 4),
            "reasons": [_clip_text(item, 48) for item in (bias.get("reasons") or [])[:4]],
        },
        "tick_pressure": {
            "ready": bool(tick_pressure.get("ready")),
            "directional_bias": _clip_text(tick_pressure.get("directional_bias"), 16),
            "pressure_score": round(_safe_float(tick_pressure.get("pressure_score"), 0.0), 4),
            "burst_rate": round(_safe_float(tick_pressure.get("burst_rate"), 0.0), 2),
            "spread_mean": round(_safe_float(tick_pressure.get("spread_mean"), 0.0), 3),
            "spread_std": round(_safe_float(tick_pressure.get("spread_std"), 0.0), 4),
            "spread_shock": round(_safe_float(tick_pressure.get("spread_shock"), 0.0), 3),
            "favorable_long": bool(tick_pressure.get("favorable_long")),
            "favorable_short": bool(tick_pressure.get("favorable_short")),
        },
        "xgb": {
            "available": bool(xgb_live_prediction.get("available")),
            "buy_probability": round(_safe_float(xgb_live_prediction.get("buy_probability"), 0.0), 4),
            "sell_probability": round(_safe_float(xgb_live_prediction.get("sell_probability"), 0.0), 4),
            "sentiment": _clip_text(xgb_live_prediction.get("sentiment"), 16),
            "reason": _clip_text(xgb_live_prediction.get("reason"), 64),
        },
        "account": {
            "balance": round(_safe_float(account.get("balance"), 0.0), 2),
            "equity": round(_safe_float(account.get("equity"), 0.0), 2),
            "free_margin": round(_safe_float(account.get("free_margin"), 0.0), 2),
        },
        "levels": {
            "session_high": round(_safe_float((liquidity.get("key_levels") or {}).get("session_high"), 0.0), 3),
            "session_low": round(_safe_float((liquidity.get("key_levels") or {}).get("session_low"), 0.0), 3),
            "support_zones": [
                {
                    "low": round(_safe_float(zone.get("zone_low"), 0.0), 3),
                    "high": round(_safe_float(zone.get("zone_high"), 0.0), 3),
                    "strength": round(_safe_float(zone.get("strength"), 0.0), 3),
                }
                for zone in (support_zones or [])[:3]
            ],
            "resistance_zones": [
                {
                    "low": round(_safe_float(zone.get("zone_low"), 0.0), 3),
                    "high": round(_safe_float(zone.get("zone_high"), 0.0), 3),
                    "strength": round(_safe_float(zone.get("strength"), 0.0), 3),
                }
                for zone in (resistance_zones or [])[:3]
            ],
        },
        "recent_candles": _compress_candles(candles),
    }


def normalize_manual_trade_idea(raw: Dict[str, Any], tick: Dict[str, Any]) -> Dict[str, Any]:
    action = str(raw.get("action") or "WAIT").strip().upper()
    if action not in {"BUY", "SELL", "WAIT"}:
        action = "WAIT"

    confidence = max(0.0, min(1.0, _safe_float(raw.get("confidence"), 0.0)))
    entry = round(_safe_float(raw.get("entry"), 0.0), 3)
    sl = round(_safe_float(raw.get("sl"), 0.0), 3)
    tp = round(_safe_float(raw.get("tp"), 0.0), 3)
    rr = round(_safe_float(raw.get("rr"), 0.0), 3)
    setup_type = str(raw.get("setup_type") or "").strip().upper() or ("WAIT" if action == "WAIT" else "LIMIT")
    reasoning = _clip_text(raw.get("reasoning") or raw.get("reason") or "", 220)
    invalidation = _clip_text(raw.get("invalidation") or "", 120)
    checklist = [_clip_text(item, 64) for item in (raw.get("checklist") or []) if str(item or "").strip()][:4]
    current_price = round(
        _safe_float((tick or {}).get("ask") if action == "BUY" else (tick or {}).get("bid"), 0.0),
        3,
    )

    actionable = False
    if action in {"BUY", "SELL"} and entry > 0 and sl > 0 and tp > 0:
        if action == "BUY" and sl < entry < tp:
            actionable = True
        elif action == "SELL" and tp < entry < sl:
            actionable = True

    if actionable and rr <= 0:
        risk = abs(entry - sl)
        reward = abs(tp - entry)
        rr = round(reward / risk, 3) if risk > 0 else 0.0

    pending_order_type = infer_pending_order_type(action, entry, tick) if actionable else "NONE"

    if not actionable:
        action = "WAIT"
        confidence = 0.0 if confidence < 0 else confidence
        entry = 0.0
        sl = 0.0
        tp = 0.0
        rr = 0.0
        setup_type = "WAIT"
        pending_order_type = "NONE"

    return {
        "action": action,
        "confidence": round(confidence, 4),
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "rr": rr,
        "setup_type": setup_type,
        "reasoning": reasoning,
        "invalidation": invalidation,
        "checklist": checklist,
        "actionable": actionable,
        "current_price": current_price,
        "pending_order_type": pending_order_type,
    }


class AIManualTradeIdeaService:
    def __init__(
        self,
        api_key: str | None = None,
        *,
        enabled: bool | None = None,
        model: str | None = None,
        endpoint: str | None = None,
        timeout_seconds: float | None = None,
    ):
        env_enabled = os.getenv("OPENAI_MANUAL_TRADE_ENABLED", "").strip().lower()
        if enabled is not None:
            self.enabled = bool(enabled)
        elif env_enabled:
            self.enabled = env_enabled not in {"0", "false", "off", "no"}
        else:
            self.enabled = True

        self.api_key = (api_key if api_key is not None else os.getenv("OPENAI_API_KEY", "")).strip()
        self.model = (model or os.getenv("OPENAI_MANUAL_TRADE_MODEL", "gpt-5.4-mini")).strip()
        self.endpoint = (endpoint or os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")).rstrip("/")
        self.timeout_seconds = float(
            timeout_seconds if timeout_seconds is not None else os.getenv("OPENAI_MANUAL_TRADE_TIMEOUT_SECONDS", "12")
        )
        self.provider = "openai"

    @property
    def is_enabled(self) -> bool:
        env_enabled = os.getenv("OPENAI_MANUAL_TRADE_ENABLED", "").strip().lower()
        if env_enabled:
            default = env_enabled not in {"0", "false", "off", "no"}
        else:
            default = self.enabled
        return bool(getattr(cfg, "AI_MANUAL_TRADE_IDEAS_ENABLED", default))

    def _base_payload(self) -> Dict[str, Any]:
        return {
            "enabled": self.is_enabled,
            "provider": self.provider,
            "model": self.model,
            "status": "disabled",
            "actionable": False,
            "action": "WAIT",
            "confidence": 0.0,
            "entry": 0.0,
            "sl": 0.0,
            "tp": 0.0,
            "rr": 0.0,
            "setup_type": "WAIT",
            "reasoning": "",
            "invalidation": "",
            "checklist": [],
            "pending_order_type": "NONE",
            "current_price": 0.0,
            "latency_ms": 0,
            "generated_at": _utc_now_iso(),
            "request_text": "",
            "response_text": "",
        }

    def _schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "action": {"type": "string", "enum": ["BUY", "SELL", "WAIT"]},
                "confidence": {"type": "number"},
                "entry": {"type": "number"},
                "sl": {"type": "number"},
                "tp": {"type": "number"},
                "rr": {"type": "number"},
                "setup_type": {"type": "string", "enum": ["LIMIT", "STOP", "MARKET", "WAIT"]},
                "reasoning": {"type": "string"},
                "invalidation": {"type": "string"},
                "checklist": {"type": "array", "items": {"type": "string"}},
            },
            "required": [
                "action",
                "confidence",
                "entry",
                "sl",
                "tp",
                "rr",
                "setup_type",
                "reasoning",
                "invalidation",
                "checklist",
            ],
        }

    def _build_request(self, context: Dict[str, Any]) -> Dict[str, Any]:
        instructions = (
            "You are a conservative manual trade idea assistant for XAUUSD. "
            "Return exactly one actionable BUY/SELL setup or WAIT. "
            "Base the answer only on the structured data provided. "
            "Use the entry as the pending-order price. "
            "The dashboard may also optionally place a market order now in the same direction, "
            "so the direction, SL, and TP must still make sense around current price. "
            "Rules: for BUY use sl < entry < tp; for SELL use tp < entry < sl; "
            "if no clean setup exists, return WAIT with numeric fields set to 0. "
            "Prefer RR >= 1.2 and keep the reasoning brief."
        )
        user_payload = {
            "task": "Produce one manual trade idea for the current market.",
            "context": context,
        }
        return {
            "model": self.model,
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": instructions}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": json.dumps(user_payload, sort_keys=True)}],
                },
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "manual_trade_idea",
                    "strict": True,
                    "schema": self._schema(),
                }
            },
            "max_output_tokens": 500,
        }

    def generate_trade_idea(self, context: Dict[str, Any], tick: Dict[str, Any]) -> Dict[str, Any]:
        result = self._base_payload()
        result["current_price"] = round(
            _safe_float((tick or {}).get("ask") or (tick or {}).get("bid"), 0.0),
            3,
        )

        if not self.is_enabled:
            result["status"] = "disabled"
            result["reasoning"] = "Manual AI trade ideas are disabled"
            return result
        if not self.api_key:
            result["status"] = "disabled"
            result["reasoning"] = "Set OPENAI_API_KEY to enable manual AI trade ideas"
            return result

        payload = self._build_request(context)
        started = time.time()
        try:
            response = requests.post(
                f"{self.endpoint}/responses",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload_json = response.json()
            raw_text = _extract_response_text(payload_json)
            if not raw_text:
                raise RuntimeError("OpenAI API returned no text output")
            parsed = _extract_json_object(raw_text)
            normalized = normalize_manual_trade_idea(parsed, tick)
            result.update(normalized)
            result["status"] = "ready"
            result["latency_ms"] = int((time.time() - started) * 1000)
            result["request_text"] = json.dumps(payload.get("input", [None, {"content": []}])[1], default=str)
            result["response_text"] = raw_text
            return result
        except Exception as exc:
            result["status"] = "error"
            result["reasoning"] = _clip_text(str(exc), 160)
            result["latency_ms"] = int((time.time() - started) * 1000)
            result["request_text"] = json.dumps(payload.get("input", [None, {"content": []}])[1], default=str)
            return result


ai_manual_trade_idea_service = AIManualTradeIdeaService()
