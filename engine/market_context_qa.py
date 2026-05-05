"""
Custom question answering over live market + strategy context.

This is intentionally advisory only:
- It answers user questions about the current market state.
- It does not place trades.
- It does not auto-veto or auto-approve strategy execution.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from openai import OpenAI


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _trim_snapshot(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only the fields the model actually needs to answer trading questions."""
    market = snapshot.get("market", {})
    guidance = snapshot.get("guidance", {})
    return {
        "symbol": snapshot.get("symbol"),
        "session": snapshot.get("session"),
        "engine_enabled": snapshot.get("engine_enabled"),
        "guidance": {
            "market_summary": guidance.get("market_summary"),
            "action_now": guidance.get("action_now"),
            "wait_advice": guidance.get("wait_advice"),
            "lead_strategy": guidance.get("lead_strategy"),
        },
        "market": {
            "tick": market.get("tick"),
            "regime": market.get("regime"),
            "bias": market.get("bias"),
            "calendar": {k: market.get("calendar", {}).get(k) for k in ("blocked", "advisory", "reason", "next_in_min")},
            "tick_pressure": {k: market.get("tick_pressure", {}).get(k) for k in ("ready", "directional_bias", "pressure_score")},
            "positions": market.get("positions", [])[:3],
            "blockers": {"top_reasons": (market.get("blockers") or {}).get("top_reasons", [])[:3]},
            "liquidity": market.get("liquidity"),
            "zones": {
                "support": (market.get("zones") or {}).get("support", [])[:2],
                "resistance": (market.get("zones") or {}).get("resistance", [])[:2],
            },
        },
    }


def _build_prompt(question: str, snapshot: Dict[str, Any]) -> str:
    return (
        "You are a live market context assistant inside a trading dashboard.\n"
        "Answer the user's custom question using only the supplied JSON context.\n"
        "Be concrete, concise, and conservative.\n"
        "If the context is insufficient, say exactly what is missing.\n"
        "Do not claim to execute trades, open orders, or change settings.\n"
        "Do not invent indicators, prices, or triggers that are not present.\n"
        "When the user asks what to do now, prefer waiting over forcing a trade.\n"
        "Format with short sections when useful.\n\n"
        f"User question:\n{question.strip()}\n\n"
        f"Context JSON:\n{json.dumps(snapshot, separators=(',', ':'), ensure_ascii=True)}"
    )


class MarketContextQAService:
    def __init__(self):
        api_key = os.getenv("NVIDIA_API_KEY", os.getenv("NVAPI_KEY", "")).strip()
        if not api_key:
            api_key = "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"
        self.api_key = api_key
        self.model = os.getenv("NVIDIA_MARKET_QA_MODEL", "meta/llama-3.1-8b-instruct")
        self.timeout_seconds = float(os.getenv("NVIDIA_MARKET_QA_TIMEOUT_SECONDS", "25"))
        self.provider = "nvidia"
        self.enabled = os.getenv("OPENAI_MARKET_QA_ENABLED", "1").strip().lower() not in {"0", "false", "off", "no"}
        self._client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=self.api_key,
        )

    def answer_question(self, question: str, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        clean_question = str(question or "").strip()
        if not clean_question:
            return {
                "enabled": True, "status": "error", "provider": self.provider,
                "model": self.model, "generated_at": _utc_now_iso(),
                "question": "", "message": "Question cannot be empty.", "answer": "",
            }
        if not self.enabled:
            return {
                "enabled": False, "status": "disabled", "provider": self.provider,
                "model": self.model, "generated_at": _utc_now_iso(),
                "question": clean_question, "message": "Market Q&A is disabled.", "answer": "",
            }

        started = time.time()
        try:
            trimmed = _trim_snapshot(snapshot)
            completion = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You answer custom questions about a live trading dashboard context. "
                            "Stay grounded in the supplied JSON and keep the answer concise."
                        ),
                    },
                    {
                        "role": "user",
                        "content": _build_prompt(clean_question, trimmed),
                    },
                ],
                temperature=0.15,
                top_p=1,
                max_tokens=400,
                stream=False,
                timeout=self.timeout_seconds,
            )

            answer = (completion.choices[0].message.content or "").strip()
            if not answer:
                raise RuntimeError("Model returned no text output")

            return {
                "enabled": True, "status": "ready", "provider": self.provider,
                "model": self.model, "generated_at": _utc_now_iso(),
                "latency_ms": int((time.time() - started) * 1000),
                "question": clean_question, "answer": answer,
            }
        except Exception as exc:
            return {
                "enabled": True, "status": "error", "provider": self.provider,
                "model": self.model, "generated_at": _utc_now_iso(),
                "latency_ms": int((time.time() - started) * 1000),
                "question": clean_question,
                "message": "Custom market question failed.",
                "error": str(exc), "answer": "",
            }


market_context_qa_service = MarketContextQAService()
