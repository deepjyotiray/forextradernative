"""
AI market bias layer for XAUUSD.

This service adds a bounded macro/news bias overlay on top of the existing
strategy engine. It never creates trades directly; it may only block one side,
block all new entries, or reduce the final lot size.
"""
from __future__ import annotations

import copy
import json
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

import config as cfg
from local_env import load_local_env_file
from engine.session_filter import is_market_open_at


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


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


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _clip_text(value: Any, limit: int = 240) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _json_safe(value: Any):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return value


def _coerce_utc_datetime(value: Any) -> Optional[datetime]:
    try:
        if value is None:
            return None
        if isinstance(value, datetime):
            dt = value
        elif hasattr(value, "to_pydatetime"):
            dt = value.to_pydatetime()
        else:
            text = str(value).strip()
            if not text:
                return None
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt
    except Exception:
        return None


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


def _iso_after_seconds(seconds: int) -> str:
    return (_utc_now() + timedelta(seconds=max(0, int(seconds)))).isoformat()


def _symbol_in_scope(symbol: str) -> bool:
    return str(symbol or "").upper() == "XAUUSD"


def _automation_enabled(cfg_module=cfg) -> bool:
    env_value = os.getenv("AI_AUTOMATION_ENABLED", "").strip().lower()
    if env_value:
        return env_value not in {"0", "false", "off", "no"}
    return bool(getattr(cfg_module, "AI_AUTOMATION_ENABLED", True))


def _market_is_open(market_data: Dict[str, Any], now: datetime) -> bool:
    try:
        candidate = market_data.get("now_utc")
        dt = _coerce_utc_datetime(candidate) or now
        return bool(is_market_open_at(dt))
    except Exception:
        return bool(is_market_open_at(now))


class AIMarketBiasService:
    def __init__(
        self,
        base_dir: str | Path,
        cfg_module=cfg,
        *,
        enabled: Optional[bool] = None,
        alpha_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        model: Optional[str] = None,
        endpoint: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
    ):
        self.base_dir = Path(base_dir)
        load_local_env_file(self.base_dir)
        self.cfg_module = cfg_module
        self.state_path = self.base_dir / "ai_market_bias_state.json"
        env_enabled = (
            os.getenv("AI_MARKET_BIAS_ENABLED", "").strip().lower()
            or os.getenv("OPENAI_AI_MARKET_BIAS_ENABLED", "").strip().lower()
        )
        self._enabled_override = enabled
        self._env_enabled = env_enabled
        self.alpha_api_key = (
            alpha_api_key if alpha_api_key is not None else os.getenv("ALPHAVANTAGE_API_KEY", "")
        ).strip()
        self.openai_api_key = (
            openai_api_key if openai_api_key is not None else os.getenv("OPENAI_API_KEY", "")
        ).strip()
        self.model = (model or os.getenv("OPENAI_AI_MARKET_BIAS_MODEL", "gpt-5.4-mini")).strip()
        self.endpoint = (endpoint or os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")).rstrip("/")
        self.timeout_seconds = float(timeout_seconds or os.getenv("OPENAI_AI_MARKET_BIAS_TIMEOUT_SECONDS", "20"))
        self.provider = "openai"
        self._lock = threading.Lock()
        self._snapshot = self._load_persisted_state() or self._default_snapshot(
            status="disabled" if not self.is_enabled else "idle",
            reason="AI market bias service not initialized yet.",
        )
        self._last_refresh_started_at = 0.0

    @property
    def is_enabled(self) -> bool:
        if self._enabled_override is not None:
            return bool(self._enabled_override)
        if self._env_enabled:
            return self._env_enabled not in {"0", "false", "off", "no"}
        return bool(getattr(self.cfg_module, "AI_MARKET_BIAS_ENABLED", False))

    @property
    def automation_enabled(self) -> bool:
        return _automation_enabled(self.cfg_module)

    @property
    def alpha_enabled(self) -> bool:
        env_value = os.getenv("AI_MARKET_BIAS_ALPHA_ENABLED", "").strip().lower()
        if env_value:
            return env_value not in {"0", "false", "off", "no"}
        return bool(getattr(self.cfg_module, "AI_MARKET_BIAS_ALPHA_ENABLED", True))

    @property
    def openai_enabled(self) -> bool:
        env_value = os.getenv("AI_MARKET_BIAS_OPENAI_ENABLED", "").strip().lower()
        if env_value:
            return env_value not in {"0", "false", "off", "no"}
        return bool(getattr(self.cfg_module, "AI_MARKET_BIAS_OPENAI_ENABLED", True))

    @property
    def enabled(self) -> bool:
        return self.is_enabled

    @property
    def refresh_seconds(self) -> int:
        return max(60, _safe_int(getattr(self.cfg_module, "AI_MARKET_BIAS_REFRESH_SECONDS", 300), 300))

    @property
    def news_lookback_minutes(self) -> int:
        return max(30, _safe_int(getattr(self.cfg_module, "AI_MARKET_BIAS_NEWS_LOOKBACK_MINUTES", 180), 180))

    @property
    def direction_block_threshold(self) -> float:
        return min(1.0, max(0.0, _safe_float(getattr(self.cfg_module, "AI_MARKET_BIAS_DIRECTION_BLOCK_THRESHOLD", 0.75), 0.75)))

    @property
    def reduce_threshold(self) -> float:
        return min(1.0, max(0.0, _safe_float(getattr(self.cfg_module, "AI_MARKET_BIAS_REDUCE_THRESHOLD", 0.60), 0.60)))

    @property
    def fail_open(self) -> bool:
        return bool(getattr(self.cfg_module, "AI_MARKET_BIAS_FAIL_OPEN", True))

    @property
    def max_news_items(self) -> int:
        return max(10, min(100, _safe_int(getattr(self.cfg_module, "AI_MARKET_BIAS_MAX_NEWS_ITEMS", 40), 40)))

    def _default_source_health(self) -> Dict[str, Dict[str, Any]]:
        return {
            "gold_spot": {"ok": False, "updated_at": "", "error": "", "item_count": 0},
            "gold_history": {"ok": False, "updated_at": "", "error": "", "item_count": 0},
            "news_macro": {"ok": False, "updated_at": "", "error": "", "item_count": 0},
            "news_usd": {"ok": False, "updated_at": "", "error": "", "item_count": 0},
            "market_status": {"ok": False, "updated_at": "", "error": "", "item_count": 0},
            "openai": {"ok": False, "updated_at": "", "error": "", "item_count": 0},
        }

    def _default_snapshot(self, *, status: str = "idle", reason: str = "") -> Dict[str, Any]:
        return {
            "enabled": bool(self.is_enabled),
            "status": status,
            "provider": self.provider,
            "model": self.model,
            "generated_at": "",
            "expires_at": "",
            "directional_bias": "NEUTRAL",
            "confidence": 0.0,
            "risk_mode": "NORMAL",
            "allow_long": True,
            "allow_short": True,
            "risk_multiplier": 1.0,
            "summary": "",
            "macro_drivers": [],
            "news_drivers": [],
            "source_health": self._default_source_health(),
            "last_error": reason,
            "last_refresh_reason": "",
            "last_applied_action": "",
            "last_applied_reason": "",
            "last_applied_at": "",
            "symbol_scope": "XAUUSD",
            "fail_open": bool(self.fail_open),
            "alpha_configured": bool(self.alpha_api_key),
            "openai_configured": bool(self.openai_api_key),
        }

    def _load_persisted_state(self) -> Optional[Dict[str, Any]]:
        try:
            if not self.state_path.exists():
                return None
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return None
            snapshot = self._default_snapshot(status=str(payload.get("status") or "idle"))
            snapshot.update(payload)
            snapshot["source_health"] = {
                **self._default_source_health(),
                **dict(payload.get("source_health") or {}),
            }
            return snapshot
        except Exception:
            return None

    def _persist_state(self) -> None:
        try:
            self.state_path.write_text(
                json.dumps(_json_safe(self._snapshot), indent=2, sort_keys=True, default=str),
                encoding="utf-8",
            )
        except Exception:
            pass

    def get_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._snapshot)

    def get_dashboard_status(self) -> Dict[str, Any]:
        return self.get_snapshot()

    def _should_refresh(self, now: Optional[datetime] = None) -> bool:
        now = now or _utc_now()
        snapshot = self._snapshot or {}
        generated = _coerce_utc_datetime(snapshot.get("generated_at"))
        if generated is None:
            return True
        return (now - generated).total_seconds() >= self.refresh_seconds

    def _is_snapshot_fresh(self, snapshot: Dict[str, Any], now: Optional[datetime] = None) -> bool:
        now = now or _utc_now()
        expires_at = _coerce_utc_datetime(snapshot.get("expires_at"))
        generated_at = _coerce_utc_datetime(snapshot.get("generated_at"))
        if generated_at is None:
            return False
        if expires_at is not None and now > expires_at:
            return False
        return (now - generated_at).total_seconds() <= max(self.refresh_seconds * 3, 900)

    def refresh_if_due(self, market_data: Dict[str, Any], *, force: bool = False) -> Dict[str, Any]:
        with self._lock:
            if not force and not self._should_refresh():
                return copy.deepcopy(self._snapshot)
            return copy.deepcopy(self._refresh_locked(market_data, force=force))

    def _refresh_locked(self, market_data: Dict[str, Any], *, force: bool = False) -> Dict[str, Any]:
        now = _utc_now()
        symbol = str(market_data.get("symbol") or getattr(self.cfg_module, "SYMBOL", "")).upper()
        self._last_refresh_started_at = time.time()
        current = copy.deepcopy(self._snapshot)

        if not self.is_enabled:
            self._snapshot = self._default_snapshot(status="disabled", reason="AI market bias disabled.")
            self._snapshot["last_refresh_reason"] = "disabled"
            self._persist_state()
            return self._snapshot

        if not self.automation_enabled:
            self._snapshot = self._default_snapshot(
                status="disabled",
                reason="AI automation bypass is enabled; live strategy is running without AI market bias.",
            )
            self._snapshot["last_refresh_reason"] = "automation_bypassed"
            self._persist_state()
            return self._snapshot

        if not self.alpha_enabled or not self.openai_enabled:
            disabled_sources = []
            if not self.alpha_enabled:
                disabled_sources.append("Alpha Vantage")
            if not self.openai_enabled:
                disabled_sources.append("OpenAI")
            self._snapshot = self._default_snapshot(
                status="disabled",
                reason=f"AI market bias source disabled: {', '.join(disabled_sources)}",
            )
            self._snapshot["last_refresh_reason"] = "source_disabled"
            self._persist_state()
            return self._snapshot

        if not _symbol_in_scope(symbol):
            self._snapshot = self._default_snapshot(status="disabled", reason=f"Out of scope for symbol {symbol or '-'}")
            self._snapshot["last_refresh_reason"] = "out_of_scope"
            self._persist_state()
            return self._snapshot

        if not _market_is_open(market_data, now):
            fallback = copy.deepcopy(current or self._default_snapshot(status="idle"))
            fallback["enabled"] = True
            fallback["status"] = "idle"
            fallback["last_error"] = "AI market bias idle while market is closed."
            fallback["last_refresh_reason"] = "market_closed"
            self._snapshot = fallback
            self._persist_state()
            return self._snapshot

        if not self.alpha_api_key or not self.openai_api_key:
            reason = "Missing ALPHAVANTAGE_API_KEY or OPENAI_API_KEY"
            self._snapshot = self._default_snapshot(status="disabled", reason=reason)
            self._snapshot["last_refresh_reason"] = "missing_credentials"
            self._persist_state()
            return self._snapshot

        try:
            collected = self._collect_context(market_data, now)
            ai_decision = self._request_openai_decision(collected)
            snapshot = self._build_snapshot_from_decision(ai_decision, collected, now)
            self._snapshot = snapshot
            self._persist_state()
            return self._snapshot
        except Exception as exc:
            fallback = copy.deepcopy(current or self._default_snapshot(status="error"))
            fallback["enabled"] = True
            fallback["status"] = "stale" if self._is_snapshot_fresh(fallback, now) else "error"
            fallback["last_error"] = _clip_text(exc, 400)
            fallback["last_refresh_reason"] = "refresh_failed"
            if fallback["status"] == "error":
                fallback["allow_long"] = True
                fallback["allow_short"] = True
                fallback["risk_mode"] = "NORMAL"
                fallback["risk_multiplier"] = 1.0
                fallback["directional_bias"] = "NEUTRAL"
                fallback["confidence"] = 0.0
                fallback["summary"] = ""
                fallback["generated_at"] = fallback.get("generated_at") or now.isoformat()
                fallback["expires_at"] = fallback.get("expires_at") or _iso_after_seconds(self.refresh_seconds)
            self._snapshot = fallback
            self._persist_state()
            return self._snapshot

    def _alpha_get(self, params: Dict[str, Any]) -> Dict[str, Any]:
        query = dict(params or {})
        query["apikey"] = self.alpha_api_key
        response = requests.get(
            "https://www.alphavantage.co/query",
            params=query,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("Alpha Vantage returned a non-object payload")
        if payload.get("Error Message"):
            raise RuntimeError(str(payload.get("Error Message")))
        if payload.get("Information"):
            raise RuntimeError(str(payload.get("Information")))
        if payload.get("Note"):
            raise RuntimeError(str(payload.get("Note")))
        return payload

    def _mark_health(
        self,
        health: Dict[str, Dict[str, Any]],
        key: str,
        *,
        ok: bool,
        updated_at: str,
        item_count: int = 0,
        error: str = "",
    ) -> None:
        health[key] = {
            "ok": bool(ok),
            "updated_at": updated_at,
            "item_count": max(0, int(item_count)),
            "error": _clip_text(error, 240),
        }

    def _collect_context(self, market_data: Dict[str, Any], now: datetime) -> Dict[str, Any]:
        health = self._default_source_health()
        updated_at = now.isoformat()
        external: Dict[str, Any] = {
            "gold_spot": {},
            "gold_history": [],
            "news_macro": [],
            "news_usd": [],
            "market_status": {},
            "weekend_intel": {},
        }
        errors: List[str] = []

        def _try_load(key: str, loader, *, item_counter=None):
            try:
                payload = loader()
                external[key] = payload
                count = item_counter(payload) if item_counter is not None else (len(payload) if isinstance(payload, list) else 1)
                self._mark_health(health, key, ok=True, updated_at=updated_at, item_count=count)
            except Exception as exc:
                errors.append(f"{key}: {exc}")
                external[key] = [] if key.startswith("news_") or key == "gold_history" else {}
                self._mark_health(health, key, ok=False, updated_at=updated_at, error=str(exc))

        _try_load(
            "gold_spot",
            lambda: self._alpha_get({"function": "GOLD_SILVER_SPOT", "symbol": "GOLD"}),
        )
        _try_load(
            "gold_history",
            lambda: self._alpha_get({"function": "GOLD_SILVER_HISTORY", "symbol": "GOLD", "interval": "daily"}),
            item_counter=lambda payload: len(payload.get("data") or []),
        )

        time_from = (now - timedelta(minutes=self.news_lookback_minutes)).strftime("%Y%m%dT%H%M")
        macro_topics = "financial_markets,economy_macro,economy_monetary"
        _try_load(
            "news_macro",
            lambda: self._alpha_get(
                {
                    "function": "NEWS_SENTIMENT",
                    "topics": macro_topics,
                    "time_from": time_from,
                    "sort": "LATEST",
                    "limit": min(1000, max(50, self.max_news_items)),
                }
            ),
            item_counter=lambda payload: len(payload.get("feed") or []),
        )
        _try_load(
            "news_usd",
            lambda: self._alpha_get(
                {
                    "function": "NEWS_SENTIMENT",
                    "tickers": "FOREX:USD",
                    "time_from": time_from,
                    "sort": "LATEST",
                    "limit": min(1000, max(50, self.max_news_items)),
                }
            ),
            item_counter=lambda payload: len(payload.get("feed") or []),
        )
        _try_load(
            "market_status",
            lambda: self._alpha_get({"function": "MARKET_STATUS"}),
        )

        compact = self._build_compact_market_context(market_data)
        deduped_news = self._merge_news_items(
            external.get("news_macro", {}).get("feed") or [],
            external.get("news_usd", {}).get("feed") or [],
        )
        external["news_items"] = deduped_news
        return {
            "generated_at": updated_at,
            "symbol": str(market_data.get("symbol") or getattr(self.cfg_module, "SYMBOL", "XAUUSD")).upper(),
            "internal_market": compact,
            "external_market": {
                "gold_spot": self._compact_gold_spot(external.get("gold_spot") or {}),
                "gold_history": self._compact_gold_history(external.get("gold_history") or {}),
                "market_status": self._compact_market_status(external.get("market_status") or {}),
                "news_items": deduped_news,
                "weekend_intel": self._compact_weekend_intel(market_data.get("weekend_intel") or {}),
            },
            "source_health": health,
            "source_errors": errors,
        }

    def _compact_gold_spot(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            return {}
        candidate = payload.get("data") if isinstance(payload.get("data"), dict) else payload
        return {
            "symbol": str(candidate.get("symbol") or "GOLD"),
            "price": _safe_float(candidate.get("price") or candidate.get("spot_price") or candidate.get("value"), 0.0),
            "unit": str(candidate.get("unit") or candidate.get("currency") or "USD").upper(),
            "last_refreshed": str(candidate.get("last_refreshed") or candidate.get("updated_at") or ""),
        }

    def _compact_gold_history(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        rows = []
        data = payload.get("data") or []
        if isinstance(data, dict):
            data = data.get("data") or []
        for item in list(data or [])[:15]:
            if not isinstance(item, dict):
                continue
            rows.append(
                {
                    "date": str(item.get("date") or item.get("timestamp") or ""),
                    "close": _safe_float(item.get("value") or item.get("close"), 0.0),
                }
            )
        return rows

    def _compact_market_status(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        markets = payload.get("markets") or payload.get("data") or payload.get("market_status") or []
        if not isinstance(markets, list):
            return {}
        fx = next(
            (
                item
                for item in markets
                if str(item.get("market_type") or item.get("type") or "").strip().lower() == "forex"
            ),
            {},
        )
        return {
            "market_type": str(fx.get("market_type") or fx.get("type") or ""),
            "region": str(fx.get("region") or fx.get("market") or ""),
            "current_status": str(fx.get("current_status") or fx.get("status") or ""),
            "local_open": str(fx.get("local_open") or fx.get("open") or ""),
            "local_close": str(fx.get("local_close") or fx.get("close") or ""),
        }

    def _merge_news_items(self, macro_items: Iterable[Dict[str, Any]], usd_items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        merged: List[Dict[str, Any]] = []
        seen: set[str] = set()
        half = max(1, self.max_news_items // 2)
        selected = list(macro_items or [])[:half] + list(usd_items or [])[:half]
        for item in selected:
            if not isinstance(item, dict):
                continue
            key = str(item.get("url") or item.get("title") or item.get("time_published") or "").strip()
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(
                {
                    "title": _clip_text(item.get("title"), 180),
                    "summary": _clip_text(item.get("summary"), 260),
                    "source": _clip_text(item.get("source"), 60),
                    "time_published": str(item.get("time_published") or ""),
                    "overall_sentiment_score": _safe_float(item.get("overall_sentiment_score"), 0.0),
                    "overall_sentiment_label": _clip_text(item.get("overall_sentiment_label"), 40),
                    "topics": [
                        _clip_text((topic or {}).get("topic"), 40)
                        for topic in list(item.get("topics") or [])[:4]
                        if isinstance(topic, dict)
                    ],
                }
            )
            if len(merged) >= self.max_news_items:
                break
        return merged

    def _build_compact_market_context(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        tick = dict(market_data.get("tick") or {})
        indicators = dict(market_data.get("indicators") or {})
        bias = dict(market_data.get("bias") or {})
        regime = dict(market_data.get("regime") or {})
        pressure = dict(market_data.get("tick_pressure") or {})
        candles = {
            "M15": self._candle_summary(market_data.get("m15_df")),
            "H1": self._candle_summary(market_data.get("h1_df")),
            "D1": self._candle_summary(market_data.get("d1_df")),
        }
        return {
            "session": str(market_data.get("session") or ""),
            "tick": {
                "bid": _safe_float(tick.get("bid"), 0.0),
                "ask": _safe_float(tick.get("ask"), 0.0),
                "spread": _safe_float(tick.get("spread"), 0.0),
            },
            "bias": {
                "direction": _clip_text(bias.get("direction"), 20),
                "confidence": _safe_float(bias.get("confidence"), 0.0),
            },
            "regime": {
                "state": _clip_text(regime.get("state"), 24),
                "direction": _clip_text(regime.get("direction"), 20),
            },
            "tick_pressure": {
                "directional_bias": _clip_text(pressure.get("directional_bias"), 20),
                "pressure_score": _safe_float(pressure.get("pressure_score"), 0.0),
                "burst_rate": _safe_float(pressure.get("burst_rate"), 0.0),
            },
            "indicators": {
                "atr": _safe_float(indicators.get("atr14", indicators.get("atr")), 0.0),
                "rsi": _safe_float(indicators.get("rsi"), 0.0),
                "ema9_slope": _safe_float(indicators.get("ema9_slope"), 0.0),
                "body_ratio": _safe_float(indicators.get("body_ratio"), 0.0),
            },
            "candles": candles,
        }

    def _compact_weekend_intel(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            return {}
        return {
            "status": _clip_text(payload.get("status"), 20),
            "generated_at": str(payload.get("generated_at") or ""),
            "summary": _clip_text(payload.get("summary"), 240),
            "sample_count": _safe_int(payload.get("sample_count"), 0),
            "preopen_window": _safe_bool(payload.get("preopen_window"), False),
            "latest_x_counts": [
                {
                    "label": _clip_text(item.get("label"), 30),
                    "total_count": _safe_int(item.get("total_count"), 0),
                }
                for item in list(payload.get("latest_x_counts") or [])[:3]
                if isinstance(item, dict)
            ],
            "latest_x_posts": [
                {
                    "query_label": _clip_text(item.get("query_label"), 30),
                    "created_at": str(item.get("created_at") or ""),
                    "text": _clip_text(item.get("text"), 180),
                    "like_count": _safe_int(item.get("like_count"), 0),
                    "retweet_count": _safe_int(item.get("retweet_count"), 0),
                }
                for item in list(payload.get("latest_x_posts") or [])[:5]
                if isinstance(item, dict)
            ],
            "latest_alpha_news": [
                {
                    "title": _clip_text(item.get("title"), 160),
                    "source": _clip_text(item.get("source"), 40),
                    "time_published": str(item.get("time_published") or ""),
                    "overall_sentiment_label": _clip_text(item.get("overall_sentiment_label"), 40),
                }
                for item in list(payload.get("latest_alpha_news") or [])[:5]
                if isinstance(item, dict)
            ],
            "latest_gold_history": [
                {
                    "date": str(item.get("date") or ""),
                    "close": _safe_float(item.get("close"), 0.0),
                }
                for item in list(payload.get("latest_gold_history") or [])[:5]
                if isinstance(item, dict)
            ],
        }

    def _candle_summary(self, df) -> Dict[str, Any]:
        try:
            if df is None or len(df) < 2:
                return {}
            row = df.iloc[-1]
            prev = df.iloc[-2]
            current = _safe_float(row.get("close"), 0.0)
            prior = _safe_float(prev.get("close"), 0.0)
            open_price = _safe_float(row.get("open"), 0.0)
            high = _safe_float(row.get("high"), 0.0)
            low = _safe_float(row.get("low"), 0.0)
            direction = "UP" if current > prior else "DOWN" if current < prior else "FLAT"
            return {
                "time": str(row.get("datetime") or row.get("time") or ""),
                "open": open_price,
                "high": high,
                "low": low,
                "close": current,
                "close_vs_prev": round(current - prior, 3),
                "body": round(current - open_price, 3),
                "range": round(high - low, 3),
                "direction": direction,
            }
        except Exception:
            return {}

    def _openai_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "directional_bias": {"type": "string", "enum": ["LONG", "SHORT", "NEUTRAL"]},
                "confidence": {"type": "number"},
                "risk_mode": {"type": "string", "enum": ["NORMAL", "REDUCED", "BLOCK_ALL"]},
                "risk_multiplier": {"type": "number"},
                "allow_long": {"type": "boolean"},
                "allow_short": {"type": "boolean"},
                "ttl_minutes": {"type": "integer"},
                "summary": {"type": "string"},
                "macro_drivers": {"type": "array", "items": {"type": "string"}},
                "news_drivers": {"type": "array", "items": {"type": "string"}},
            },
            "required": [
                "directional_bias",
                "confidence",
                "risk_mode",
                "risk_multiplier",
                "allow_long",
                "allow_short",
                "ttl_minutes",
                "summary",
                "macro_drivers",
                "news_drivers",
            ],
        }

    def _request_openai_decision(self, context: Dict[str, Any]) -> Dict[str, Any]:
        instructions = (
            "You are a conservative gold macro/news bias evaluator for an automated XAUUSD trading system. "
            "You do not generate entries, exits, or targets. "
            "You only decide whether current macro/news context favors LONG, SHORT, NEUTRAL, reduced risk, or blocking all new trades. "
            "Be cautious. If evidence is mixed or weak, prefer NEUTRAL and NORMAL risk. "
            "Only use BLOCK_ALL for elevated macro/news uncertainty or event risk. "
            "Only allow one-sided blocking when the directional case is strong and confidence is high."
        )
        user_payload = {
            "task": "Evaluate XAUUSD macro/news bias for live strategy gating.",
            "rules": {
                "scope": "XAUUSD only",
                "no_direct_trades": True,
                "direction_block_threshold": self.direction_block_threshold,
                "risk_reduce_threshold": self.reduce_threshold,
                "risk_multiplier_bounds": [0.50, 1.00],
            },
            "context": context,
        }
        payload = {
            "model": self.model,
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": instructions}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": json.dumps(user_payload, default=str)}],
                },
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "ai_market_bias",
                    "strict": True,
                    "schema": self._openai_schema(),
                }
            },
            "max_output_tokens": 500,
        }
        response = requests.post(
            f"{self.endpoint}/responses",
            headers={
                "Authorization": f"Bearer {self.openai_api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        response_payload = response.json()
        text = _extract_response_text(response_payload)
        if not text:
            raise RuntimeError("OpenAI API returned no text output")
        parsed = _extract_json_object(text)
        return self._validate_decision_payload(parsed)

    def _validate_decision_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("AI market bias payload must be an object")
        directional_bias = str(payload.get("directional_bias") or "NEUTRAL").strip().upper()
        if directional_bias not in {"LONG", "SHORT", "NEUTRAL"}:
            raise ValueError(f"Invalid directional_bias: {directional_bias}")
        risk_mode = str(payload.get("risk_mode") or "NORMAL").strip().upper()
        if risk_mode not in {"NORMAL", "REDUCED", "BLOCK_ALL"}:
            raise ValueError(f"Invalid risk_mode: {risk_mode}")
        confidence = min(1.0, max(0.0, _safe_float(payload.get("confidence"), 0.0)))
        risk_multiplier = min(1.0, max(0.0, _safe_float(payload.get("risk_multiplier"), 1.0)))
        ttl_minutes = max(1, min(1440, _safe_int(payload.get("ttl_minutes"), self.refresh_seconds // 60 or 5)))
        macro_drivers = [_clip_text(item, 120) for item in list(payload.get("macro_drivers") or [])[:6]]
        news_drivers = [_clip_text(item, 120) for item in list(payload.get("news_drivers") or [])[:6]]
        return {
            "directional_bias": directional_bias,
            "confidence": confidence,
            "risk_mode": risk_mode,
            "risk_multiplier": risk_multiplier,
            "allow_long": bool(payload.get("allow_long", True)),
            "allow_short": bool(payload.get("allow_short", True)),
            "ttl_minutes": ttl_minutes,
            "summary": _clip_text(payload.get("summary"), 400),
            "macro_drivers": macro_drivers,
            "news_drivers": news_drivers,
        }

    def _build_snapshot_from_decision(self, decision: Dict[str, Any], context: Dict[str, Any], now: datetime) -> Dict[str, Any]:
        ttl_minutes = max(1, _safe_int(decision.get("ttl_minutes"), self.refresh_seconds // 60 or 5))
        expires_at = now + timedelta(minutes=ttl_minutes)
        risk_multiplier = min(1.0, max(0.50, _safe_float(decision.get("risk_multiplier"), 1.0)))
        snapshot = self._default_snapshot(status="ready")
        snapshot.update(
            {
                "enabled": True,
                "status": "ready",
                "generated_at": now.isoformat(),
                "expires_at": expires_at.isoformat(),
                "directional_bias": decision["directional_bias"],
                "confidence": round(_safe_float(decision["confidence"], 0.0), 4),
                "risk_mode": decision["risk_mode"],
                "allow_long": bool(decision["allow_long"]),
                "allow_short": bool(decision["allow_short"]),
                "risk_multiplier": round(risk_multiplier, 4),
                "summary": decision["summary"],
                "macro_drivers": list(decision["macro_drivers"] or []),
                "news_drivers": list(decision["news_drivers"] or []),
                "source_health": copy.deepcopy(context.get("source_health") or self._default_source_health()),
                "last_error": "",
                "last_refresh_reason": "refreshed",
                "alpha_configured": True,
                "openai_configured": True,
            }
        )
        snapshot["source_health"]["openai"] = {
            "ok": True,
            "updated_at": now.isoformat(),
            "item_count": 1,
            "error": "",
        }
        return snapshot

    def _record_application(self, action: str, reason: str) -> None:
        self._snapshot["last_applied_action"] = action
        self._snapshot["last_applied_reason"] = _clip_text(reason, 320)
        self._snapshot["last_applied_at"] = _utc_now_iso()
        self._persist_state()

    def apply_to_signal(self, strategy_name: str, signal: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        action = str((signal or {}).get("signal") or "").upper()
        snapshot = self.get_snapshot()
        if action not in {"BUY", "SELL"}:
            return {"allowed": True, "reason": "", "adjusted_signal": dict(signal or {}), "snapshot": snapshot}
        if not self.is_enabled or not _symbol_in_scope(market_data.get("symbol") or getattr(self.cfg_module, "SYMBOL", "")):
            return {"allowed": True, "reason": "", "adjusted_signal": dict(signal or {}), "snapshot": snapshot}
        if snapshot.get("status") != "ready" or not self._is_snapshot_fresh(snapshot):
            return {"allowed": True, "reason": "", "adjusted_signal": dict(signal or {}), "snapshot": snapshot}

        adjusted = dict(signal or {})
        adjusted["_ai_feature_overrides"] = dict(adjusted.get("_ai_feature_overrides") or {})
        adjusted["_ai_market_bias_status"] = snapshot.get("status")
        adjusted["_ai_market_bias_directional_bias"] = snapshot.get("directional_bias")
        adjusted["_ai_market_bias_confidence"] = snapshot.get("confidence")
        adjusted["_ai_market_bias_risk_mode"] = snapshot.get("risk_mode")
        adjusted["_ai_market_bias_risk_multiplier"] = snapshot.get("risk_multiplier")
        adjusted["_ai_market_bias_summary"] = snapshot.get("summary")
        adjusted["_ai_feature_overrides"].update(
            {
                "ai_market_bias_directional_bias": snapshot.get("directional_bias"),
                "ai_market_bias_confidence": snapshot.get("confidence"),
                "ai_market_bias_risk_mode": snapshot.get("risk_mode"),
                "ai_market_bias_risk_multiplier": snapshot.get("risk_multiplier"),
                "ai_market_bias_summary": snapshot.get("summary"),
            }
        )

        risk_mode = str(snapshot.get("risk_mode") or "NORMAL").upper()
        confidence = _safe_float(snapshot.get("confidence"), 0.0)
        directional_bias = str(snapshot.get("directional_bias") or "NEUTRAL").upper()
        direction = "LONG" if action == "BUY" else "SHORT"

        if risk_mode == "BLOCK_ALL":
            reason = f"AI market bias blocked all new trades: {snapshot.get('summary') or 'macro/news risk elevated'}"
            self._record_application("BLOCK_ALL", reason)
            adjusted["_ai_market_bias_action"] = "BLOCK_ALL"
            return {"allowed": False, "reason": reason, "adjusted_signal": adjusted, "snapshot": snapshot}

        if confidence >= self.direction_block_threshold:
            if directional_bias == "LONG" and direction == "SHORT":
                reason = f"AI market bias blocked SHORT trades: {snapshot.get('summary') or 'strong LONG macro/news bias'}"
                self._record_application("BLOCK_DIRECTION", reason)
                adjusted["_ai_market_bias_action"] = "BLOCK_SHORT"
                return {"allowed": False, "reason": reason, "adjusted_signal": adjusted, "snapshot": snapshot}
            if directional_bias == "SHORT" and direction == "LONG":
                reason = f"AI market bias blocked LONG trades: {snapshot.get('summary') or 'strong SHORT macro/news bias'}"
                self._record_application("BLOCK_DIRECTION", reason)
                adjusted["_ai_market_bias_action"] = "BLOCK_LONG"
                return {"allowed": False, "reason": reason, "adjusted_signal": adjusted, "snapshot": snapshot}

        if risk_mode == "REDUCED" and confidence >= self.reduce_threshold:
            reduced_multiplier = min(1.0, max(0.50, _safe_float(snapshot.get("risk_multiplier"), 1.0)))
            adjusted["_ai_market_bias_action"] = "REDUCE_RISK"
            adjusted["_ai_market_bias_risk_multiplier"] = reduced_multiplier
            adjusted["_ai_feature_overrides"]["ai_market_bias_applied_action"] = "REDUCE_RISK"
            self._record_application("REDUCE_RISK", snapshot.get("summary") or "Reduced risk due to mixed macro/news context")
            return {
                "allowed": True,
                "reason": snapshot.get("summary") or "Reduced risk due to AI market bias",
                "adjusted_signal": adjusted,
                "snapshot": snapshot,
            }

        adjusted["_ai_market_bias_action"] = "PASS"
        return {"allowed": True, "reason": "", "adjusted_signal": adjusted, "snapshot": snapshot}
