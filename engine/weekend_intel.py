"""
Weekend intel collector for XAUUSD.

This service runs only during the weekend market-closed window. It polls low-cost
external context sources, persists compact snapshots, and hands that context to
the first live AI market-bias refresh after the market reopens.
"""
from __future__ import annotations

import copy
import json
import os
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests

import config as cfg
from engine.session_filter import is_market_open_at
from local_env import load_local_env_file


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _clip_text(value: Any, limit: int = 220) -> str:
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


def _market_is_open(market_data: Dict[str, Any], now: datetime) -> bool:
    try:
        candidate = market_data.get("now_utc")
        dt = _coerce_utc_datetime(candidate) or now
        return bool(is_market_open_at(dt))
    except Exception:
        return bool(is_market_open_at(now))


def _automation_enabled(cfg_module=cfg) -> bool:
    env_value = os.getenv("AI_AUTOMATION_ENABLED", "").strip().lower()
    if env_value:
        return env_value not in {"0", "false", "off", "no"}
    return bool(getattr(cfg_module, "AI_AUTOMATION_ENABLED", True))


def _is_weekend_window(now: datetime) -> bool:
    dt = now.astimezone(timezone.utc)
    if bool(is_market_open_at(dt)):
        return False
    weekday = dt.weekday()
    if weekday == 4 and dt.hour >= 22:
        return True
    if weekday == 5:
        return True
    if weekday == 6 and dt.hour < 22:
        return True
    return False


class WeekendIntelService:
    def __init__(
        self,
        base_dir: str | Path,
        cfg_module=cfg,
        *,
        enabled: Optional[bool] = None,
        alpha_api_key: Optional[str] = None,
        x_bearer_token: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
    ):
        self.base_dir = Path(base_dir)
        load_local_env_file(self.base_dir)
        self.cfg_module = cfg_module
        self.state_path = self.base_dir / "weekend_intel_state.json"
        self._enabled_override = enabled
        env_enabled = os.getenv("WEEKEND_INTEL_ENABLED", "").strip().lower()
        self._env_enabled = env_enabled
        self.alpha_api_key = (
            alpha_api_key if alpha_api_key is not None else os.getenv("ALPHAVANTAGE_API_KEY", "")
        ).strip()
        self.x_bearer_token = (
            x_bearer_token
            if x_bearer_token is not None
            else (os.getenv("X_API_BEARER_TOKEN", "") or os.getenv("TWITTER_BEARER_TOKEN", ""))
        ).strip()
        self.timeout_seconds = float(timeout_seconds or os.getenv("WEEKEND_INTEL_TIMEOUT_SECONDS", "20"))
        self._lock = threading.Lock()
        self._snapshot = self._load_persisted_state() or self._default_snapshot(
            status="disabled" if not self.is_enabled else "idle",
            reason="Weekend intel service not initialized yet.",
        )

    @property
    def is_enabled(self) -> bool:
        if self._enabled_override is not None:
            return bool(self._enabled_override)
        if self._env_enabled:
            return self._env_enabled not in {"0", "false", "off", "no"}
        return bool(getattr(self.cfg_module, "WEEKEND_INTEL_ENABLED", True))

    @property
    def automation_enabled(self) -> bool:
        return _automation_enabled(self.cfg_module)

    @property
    def alpha_enabled(self) -> bool:
        env_value = os.getenv("WEEKEND_INTEL_ALPHA_ENABLED", "").strip().lower()
        if env_value:
            return env_value not in {"0", "false", "off", "no"}
        return bool(getattr(self.cfg_module, "WEEKEND_INTEL_ALPHA_ENABLED", True))

    @property
    def x_enabled(self) -> bool:
        env_value = os.getenv("WEEKEND_INTEL_X_ENABLED", "").strip().lower()
        if env_value:
            return env_value not in {"0", "false", "off", "no"}
        return bool(getattr(self.cfg_module, "WEEKEND_INTEL_X_ENABLED", True))

    @property
    def refresh_hours(self) -> int:
        return max(1, min(24, _safe_int(getattr(self.cfg_module, "WEEKEND_INTEL_REFRESH_HOURS", 4), 4)))

    @property
    def preopen_hours(self) -> int:
        return max(1, min(12, _safe_int(getattr(self.cfg_module, "WEEKEND_INTEL_PREOPEN_HOURS", 6), 6)))

    @property
    def x_spike_post_count(self) -> int:
        return max(10, _safe_int(getattr(self.cfg_module, "WEEKEND_INTEL_X_SPIKE_POST_COUNT", 80), 80))

    @property
    def x_max_posts(self) -> int:
        return max(3, min(40, _safe_int(getattr(self.cfg_module, "WEEKEND_INTEL_X_MAX_POSTS", 12), 12)))

    @property
    def alpha_news_limit(self) -> int:
        return max(5, min(50, _safe_int(getattr(self.cfg_module, "WEEKEND_INTEL_ALPHA_NEWS_LIMIT", 12), 12)))

    def _default_source_health(self) -> Dict[str, Dict[str, Any]]:
        return {
            "market_status": {"ok": False, "updated_at": "", "error": "", "item_count": 0},
            "news_macro": {"ok": False, "updated_at": "", "error": "", "item_count": 0},
            "news_usd": {"ok": False, "updated_at": "", "error": "", "item_count": 0},
            "gold_history": {"ok": False, "updated_at": "", "error": "", "item_count": 0},
            "x_counts": {"ok": False, "updated_at": "", "error": "", "item_count": 0},
            "x_posts": {"ok": False, "updated_at": "", "error": "", "item_count": 0},
        }

    def _default_snapshot(self, *, status: str = "idle", reason: str = "") -> Dict[str, Any]:
        return {
            "enabled": bool(self.is_enabled),
            "status": status,
            "generated_at": "",
            "last_checked_at": "",
            "next_due_at": "",
            "last_error": reason,
            "last_refresh_reason": "",
            "market_open": False,
            "weekend_window": False,
            "preopen_window": False,
            "refresh_hours": self.refresh_hours,
            "alpha_configured": bool(self.alpha_api_key),
            "x_configured": bool(self.x_bearer_token),
            "source_health": self._default_source_health(),
            "latest_alpha_news": [],
            "latest_x_counts": [],
            "latest_x_posts": [],
            "latest_gold_history": [],
            "recent_samples": [],
            "sample_count": 0,
            "summary": "",
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
            snapshot["recent_samples"] = list(payload.get("recent_samples") or [])[:24]
            snapshot["sample_count"] = len(snapshot["recent_samples"])
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

    def _next_due_iso(self, now: datetime) -> str:
        return (now + timedelta(hours=self.refresh_hours)).isoformat()

    def _is_preopen_window(self, now: datetime) -> bool:
        dt = now.astimezone(timezone.utc)
        return dt.weekday() == 6 and (22 - self.preopen_hours) <= dt.hour < 22

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
            "error": _clip_text(error, 220),
        }

    def _should_refresh(self, now: Optional[datetime] = None) -> bool:
        now = now or _utc_now()
        snapshot = self._snapshot or {}
        last_checked = _coerce_utc_datetime(snapshot.get("last_checked_at")) or _coerce_utc_datetime(
            snapshot.get("generated_at")
        )
        if last_checked is None:
            return True
        return (now - last_checked).total_seconds() >= self.refresh_hours * 3600

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

    def _x_get(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        response = requests.get(
            f"https://api.x.com{path}",
            params=params,
            headers={"Authorization": f"Bearer {self.x_bearer_token}"},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("X API returned a non-object payload")
        if payload.get("errors"):
            first = next((item for item in payload.get("errors") or [] if isinstance(item, dict)), {})
            raise RuntimeError(str(first.get("detail") or first.get("title") or "X API error"))
        return payload

    def _x_queries(self) -> List[Dict[str, str]]:
        return [
            {
                "label": "gold_core",
                "query": '(gold OR xauusd OR "spot gold") lang:en -is:retweet',
            },
            {
                "label": "macro_fed",
                "query": '((fed OR fomc OR powell OR inflation OR cpi OR rates) (gold OR xauusd)) lang:en -is:retweet',
            },
            {
                "label": "usd_yields",
                "query": '((usd OR dollar OR dxy OR yields OR treasury OR us10y) (gold OR xauusd)) lang:en -is:retweet',
            },
        ]

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._snapshot)

    def get_dashboard_status(self) -> Dict[str, Any]:
        return self.get_status()

    def refresh_if_due(self, market_data: Dict[str, Any], *, force: bool = False) -> Dict[str, Any]:
        with self._lock:
            if not force and not self._should_refresh():
                return copy.deepcopy(self._snapshot)
            return copy.deepcopy(self._refresh_locked(market_data))

    def _refresh_locked(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        now = _utc_now()
        reference_dt = _coerce_utc_datetime((market_data or {}).get("now_utc")) or now
        market_open = _market_is_open(market_data, now)
        weekend_window = _is_weekend_window(reference_dt)
        preopen_window = self._is_preopen_window(reference_dt)

        if not self.is_enabled:
            snapshot = self._default_snapshot(status="disabled", reason="Weekend intel disabled.")
            snapshot["last_checked_at"] = now.isoformat()
            snapshot["next_due_at"] = self._next_due_iso(now)
            snapshot["last_refresh_reason"] = "disabled"
            self._snapshot = snapshot
            self._persist_state()
            return self._snapshot

        if not self.automation_enabled:
            snapshot = self._default_snapshot(
                status="disabled",
                reason="AI automation bypass is enabled; weekend intel polling is paused.",
            )
            snapshot["last_checked_at"] = now.isoformat()
            snapshot["next_due_at"] = self._next_due_iso(now)
            snapshot["last_refresh_reason"] = "automation_bypassed"
            self._snapshot = snapshot
            self._persist_state()
            return self._snapshot

        if not self.alpha_enabled and not self.x_enabled:
            snapshot = self._default_snapshot(
                status="disabled",
                reason="Weekend intel sources disabled: Alpha Vantage and X",
            )
            snapshot["last_checked_at"] = now.isoformat()
            snapshot["next_due_at"] = self._next_due_iso(now)
            snapshot["last_refresh_reason"] = "source_disabled"
            self._snapshot = snapshot
            self._persist_state()
            return self._snapshot

        if not self.alpha_api_key and not self.x_bearer_token:
            snapshot = self._default_snapshot(status="disabled", reason="Missing Alpha Vantage and X API credentials.")
            snapshot["last_checked_at"] = now.isoformat()
            snapshot["next_due_at"] = self._next_due_iso(now)
            snapshot["last_refresh_reason"] = "missing_credentials"
            self._snapshot = snapshot
            self._persist_state()
            return self._snapshot

        if market_open:
            snapshot = copy.deepcopy(self._snapshot or self._default_snapshot(status="idle"))
            snapshot["enabled"] = True
            snapshot["status"] = "idle"
            snapshot["market_open"] = True
            snapshot["weekend_window"] = False
            snapshot["preopen_window"] = False
            snapshot["last_checked_at"] = now.isoformat()
            snapshot["next_due_at"] = self._next_due_iso(now)
            snapshot["last_refresh_reason"] = "market_open"
            snapshot["last_error"] = ""
            self._snapshot = snapshot
            self._persist_state()
            return self._snapshot

        if not weekend_window:
            snapshot = copy.deepcopy(self._snapshot or self._default_snapshot(status="idle"))
            snapshot["enabled"] = True
            snapshot["status"] = "idle"
            snapshot["market_open"] = False
            snapshot["weekend_window"] = False
            snapshot["preopen_window"] = False
            snapshot["last_checked_at"] = now.isoformat()
            snapshot["next_due_at"] = self._next_due_iso(now)
            snapshot["last_refresh_reason"] = "out_of_window"
            snapshot["last_error"] = ""
            self._snapshot = snapshot
            self._persist_state()
            return self._snapshot

        try:
            sample = self._collect_sample(now, preopen_window=preopen_window)
            recent_samples = list((self._snapshot or {}).get("recent_samples") or [])
            recent_samples.insert(0, sample)
            recent_samples = recent_samples[:24]

            snapshot = self._default_snapshot(status="ready")
            snapshot.update(
                {
                    "enabled": True,
                    "status": "ready",
                    "generated_at": now.isoformat(),
                    "last_checked_at": now.isoformat(),
                    "next_due_at": self._next_due_iso(now),
                    "market_open": False,
                    "weekend_window": True,
                    "preopen_window": bool(preopen_window),
                    "refresh_hours": self.refresh_hours,
                    "alpha_configured": bool(self.alpha_api_key),
                    "x_configured": bool(self.x_bearer_token),
                    "source_health": sample["source_health"],
                    "latest_alpha_news": list(sample["alpha_news"] or []),
                    "latest_x_counts": list(sample["x_counts"] or []),
                    "latest_x_posts": list(sample["x_posts"] or []),
                    "latest_gold_history": list(sample["gold_history"] or []),
                    "recent_samples": recent_samples,
                    "sample_count": len(recent_samples),
                    "summary": sample["summary"],
                    "last_error": "",
                    "last_refresh_reason": "refreshed",
                }
            )
            self._snapshot = snapshot
            self._persist_state()
            return self._snapshot
        except Exception as exc:
            fallback = copy.deepcopy(self._snapshot or self._default_snapshot(status="error"))
            fallback["enabled"] = True
            fallback["status"] = "stale" if fallback.get("generated_at") else "error"
            fallback["market_open"] = False
            fallback["weekend_window"] = True
            fallback["preopen_window"] = bool(preopen_window)
            fallback["last_checked_at"] = now.isoformat()
            fallback["next_due_at"] = self._next_due_iso(now)
            fallback["last_refresh_reason"] = "refresh_failed"
            fallback["last_error"] = _clip_text(exc, 320)
            self._snapshot = fallback
            self._persist_state()
            return self._snapshot

    def _collect_sample(self, now: datetime, *, preopen_window: bool) -> Dict[str, Any]:
        updated_at = now.isoformat()
        health = self._default_source_health()
        alpha_news: List[Dict[str, Any]] = []
        x_counts: List[Dict[str, Any]] = []
        x_posts: List[Dict[str, Any]] = []
        gold_history: List[Dict[str, Any]] = []
        errors: List[str] = []

        if self.alpha_enabled and self.alpha_api_key:
            try:
                market_status = self._alpha_get({"function": "MARKET_STATUS"})
                self._mark_health(health, "market_status", ok=True, updated_at=updated_at, item_count=1)
            except Exception as exc:
                market_status = {}
                errors.append(f"market_status: {exc}")
                self._mark_health(health, "market_status", ok=False, updated_at=updated_at, error=str(exc))

            time_from = (now - timedelta(hours=self.refresh_hours)).strftime("%Y%m%dT%H%M")
            try:
                macro = self._alpha_get(
                    {
                        "function": "NEWS_SENTIMENT",
                        "topics": "financial_markets,economy_macro,economy_monetary",
                        "time_from": time_from,
                        "sort": "LATEST",
                        "limit": max(self.alpha_news_limit, 10),
                    }
                )
                macro_feed = list(macro.get("feed") or [])
                self._mark_health(
                    health,
                    "news_macro",
                    ok=True,
                    updated_at=updated_at,
                    item_count=len(macro_feed),
                )
            except Exception as exc:
                macro_feed = []
                errors.append(f"news_macro: {exc}")
                self._mark_health(health, "news_macro", ok=False, updated_at=updated_at, error=str(exc))

            try:
                usd = self._alpha_get(
                    {
                        "function": "NEWS_SENTIMENT",
                        "tickers": "FOREX:USD",
                        "time_from": time_from,
                        "sort": "LATEST",
                        "limit": max(self.alpha_news_limit, 10),
                    }
                )
                usd_feed = list(usd.get("feed") or [])
                self._mark_health(
                    health,
                    "news_usd",
                    ok=True,
                    updated_at=updated_at,
                    item_count=len(usd_feed),
                )
            except Exception as exc:
                usd_feed = []
                errors.append(f"news_usd: {exc}")
                self._mark_health(health, "news_usd", ok=False, updated_at=updated_at, error=str(exc))

            alpha_news = self._merge_alpha_news(macro_feed, usd_feed)

            if preopen_window or not (self._snapshot or {}).get("latest_gold_history"):
                try:
                    history = self._alpha_get(
                        {"function": "GOLD_SILVER_HISTORY", "symbol": "GOLD", "interval": "daily"}
                    )
                    gold_history = self._compact_gold_history(history)
                    self._mark_health(
                        health,
                        "gold_history",
                        ok=True,
                        updated_at=updated_at,
                        item_count=len(gold_history),
                    )
                except Exception as exc:
                    gold_history = list((self._snapshot or {}).get("latest_gold_history") or [])
                    errors.append(f"gold_history: {exc}")
                    self._mark_health(health, "gold_history", ok=False, updated_at=updated_at, error=str(exc))
            else:
                gold_history = list((self._snapshot or {}).get("latest_gold_history") or [])
                self._mark_health(
                    health,
                    "gold_history",
                    ok=bool(gold_history),
                    updated_at=updated_at,
                    item_count=len(gold_history),
                )
        else:
            market_status = {}
            macro_feed = []
            usd_feed = []

        if self.x_enabled and self.x_bearer_token:
            try:
                x_counts = self._collect_x_counts(now)
                self._mark_health(
                    health,
                    "x_counts",
                    ok=True,
                    updated_at=updated_at,
                    item_count=len(x_counts),
                )
            except Exception as exc:
                x_counts = []
                errors.append(f"x_counts: {exc}")
                self._mark_health(health, "x_counts", ok=False, updated_at=updated_at, error=str(exc))

            should_fetch_posts = preopen_window or any(
                int(item.get("total_count") or 0) >= self.x_spike_post_count for item in x_counts
            )
            if should_fetch_posts:
                try:
                    x_posts = self._collect_x_posts(now)
                    self._mark_health(
                        health,
                        "x_posts",
                        ok=True,
                        updated_at=updated_at,
                        item_count=len(x_posts),
                    )
                except Exception as exc:
                    x_posts = []
                    errors.append(f"x_posts: {exc}")
                    self._mark_health(health, "x_posts", ok=False, updated_at=updated_at, error=str(exc))
            else:
                self._mark_health(health, "x_posts", ok=True, updated_at=updated_at, item_count=0)

        summary = self._build_summary(alpha_news, x_counts, x_posts, preopen_window=preopen_window)
        return {
            "generated_at": updated_at,
            "source_health": health,
            "alpha_news": alpha_news,
            "x_counts": x_counts,
            "x_posts": x_posts,
            "gold_history": gold_history,
            "market_status": market_status,
            "summary": summary,
            "errors": errors,
        }

    def _merge_alpha_news(self, macro_items: Iterable[Dict[str, Any]], usd_items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        merged: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for item in list(macro_items or [])[: self.alpha_news_limit] + list(usd_items or [])[: self.alpha_news_limit]:
            if not isinstance(item, dict):
                continue
            key = str(item.get("url") or item.get("title") or item.get("time_published") or "").strip()
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(
                {
                    "title": _clip_text(item.get("title"), 160),
                    "summary": _clip_text(item.get("summary"), 220),
                    "source": _clip_text(item.get("source"), 50),
                    "time_published": str(item.get("time_published") or ""),
                    "overall_sentiment_score": round(_safe_float(item.get("overall_sentiment_score"), 0.0), 3),
                    "overall_sentiment_label": _clip_text(item.get("overall_sentiment_label"), 40),
                    "topics": [
                        _clip_text((topic or {}).get("topic"), 40)
                        for topic in list(item.get("topics") or [])[:4]
                        if isinstance(topic, dict)
                    ],
                }
            )
            if len(merged) >= self.alpha_news_limit:
                break
        return merged

    def _collect_x_counts(self, now: datetime) -> List[Dict[str, Any]]:
        start_time = (now - timedelta(hours=self.refresh_hours)).isoformat().replace("+00:00", "Z")
        end_time = now.isoformat().replace("+00:00", "Z")
        rows: List[Dict[str, Any]] = []
        for item in self._x_queries():
            payload = self._x_get(
                "/2/tweets/counts/recent",
                {
                    "query": item["query"],
                    "granularity": "hour",
                    "start_time": start_time,
                    "end_time": end_time,
                },
            )
            series = list(payload.get("data") or [])
            total_count = int((payload.get("meta") or {}).get("total_tweet_count") or 0)
            rows.append(
                {
                    "label": item["label"],
                    "query": item["query"],
                    "total_count": total_count,
                    "series": [
                        {
                            "start": str(bucket.get("start") or ""),
                            "end": str(bucket.get("end") or ""),
                            "tweet_count": int(bucket.get("tweet_count") or 0),
                        }
                        for bucket in series[-6:]
                        if isinstance(bucket, dict)
                    ],
                }
            )
        return rows

    def _collect_x_posts(self, now: datetime) -> List[Dict[str, Any]]:
        start_time = (now - timedelta(hours=self.refresh_hours)).isoformat().replace("+00:00", "Z")
        rows: List[Dict[str, Any]] = []
        seen_ids: set[str] = set()
        per_query_limit = max(3, min(10, self.x_max_posts // max(1, len(self._x_queries()))))
        for item in self._x_queries():
            payload = self._x_get(
                "/2/tweets/search/recent",
                {
                    "query": item["query"],
                    "start_time": start_time,
                    "max_results": per_query_limit,
                    "tweet.fields": "created_at,public_metrics,author_id",
                },
            )
            for post in list(payload.get("data") or []):
                if not isinstance(post, dict):
                    continue
                post_id = str(post.get("id") or "").strip()
                if not post_id or post_id in seen_ids:
                    continue
                seen_ids.add(post_id)
                metrics = dict(post.get("public_metrics") or {})
                rows.append(
                    {
                        "id": post_id,
                        "query_label": item["label"],
                        "author_id": str(post.get("author_id") or ""),
                        "created_at": str(post.get("created_at") or ""),
                        "text": _clip_text(post.get("text"), 220),
                        "retweet_count": int(metrics.get("retweet_count") or 0),
                        "reply_count": int(metrics.get("reply_count") or 0),
                        "like_count": int(metrics.get("like_count") or 0),
                        "quote_count": int(metrics.get("quote_count") or 0),
                    }
                )
                if len(rows) >= self.x_max_posts:
                    return rows
        return rows

    def _compact_gold_history(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        rows = []
        data = payload.get("data") or []
        if isinstance(data, dict):
            data = data.get("data") or []
        for item in list(data or [])[:7]:
            if not isinstance(item, dict):
                continue
            rows.append(
                {
                    "date": str(item.get("date") or item.get("timestamp") or ""),
                    "close": _safe_float(item.get("value") or item.get("close"), 0.0),
                }
            )
        return rows

    def _build_summary(
        self,
        alpha_news: List[Dict[str, Any]],
        x_counts: List[Dict[str, Any]],
        x_posts: List[Dict[str, Any]],
        *,
        preopen_window: bool,
    ) -> str:
        total_x = sum(int(item.get("total_count") or 0) for item in x_counts)
        news_count = len(alpha_news)
        posts_count = len(x_posts)
        phase = "pre-open" if preopen_window else "weekend"
        return (
            f"Collected {phase} intel: {news_count} Alpha news items, "
            f"{total_x} X count hits across {len(x_counts)} queries, "
            f"{posts_count} X posts stored."
        )
