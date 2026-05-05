"""
Economic Calendar — event-aware trading filter.
Fetches high-impact USD events, blocks trading in blackout windows.
Uses nofap.cc free forex calendar API (no key needed).
Falls back to hardcoded recurring events if API fails.

Blackout rules:
  - configurable minutes BEFORE high-impact event: block new trades
  - configurable minutes AFTER high-impact event: block new trades
  - Open trades get tightened SL during blackout
"""
import time
import json
import os
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
from urllib.request import urlopen, Request
from urllib.error import URLError
import config as cfg


# High-impact USD events that move gold hard
_CRITICAL_KEYWORDS = {
    "FOMC", "Federal Funds Rate", "Interest Rate Decision",
    "Non-Farm", "Nonfarm", "NFP",
    "CPI", "Consumer Price Index",
    "PPI", "Producer Price Index",
    "GDP", "Gross Domestic Product",
    "PCE", "Core PCE",
    "Unemployment", "Jobless Claims",
    "Retail Sales",
    "ISM Manufacturing", "ISM Services",
    "Powell", "Fed Chair",
}

_POLL_INTERVAL = 3600         # re-fetch every hour
_CACHE_FILE = "calendar_cache.json"


class EconomicCalendar:
    def __init__(self):
        self._events: List[Dict] = []
        self._last_fetch: float = 0
        self._lock = threading.Lock()
        self._load_cache()

    def poll(self):
        """Fetch events if stale. Non-blocking (runs in background)."""
        if time.time() - self._last_fetch < _POLL_INTERVAL:
            return
        threading.Thread(target=self._fetch, daemon=True).start()

    def check(self) -> Dict:
        """
        Check if we're in a blackout window.
        Returns: {blocked: bool, reason: str, next_event: dict or None, events_today: int}
        """
        now = datetime.now(timezone.utc)
        blackout_before = max(0, int(getattr(cfg, "CALENDAR_BLOCK_BEFORE_MINUTES", 30) or 0)) * 60
        blackout_after = max(0, int(getattr(cfg, "CALENDAR_BLOCK_AFTER_MINUTES", 15) or 0)) * 60
        closest = None
        closest_dist = float("inf")

        for ev in self._events:
            ev_time = self._parse_time(ev.get("time_utc", ""))
            if ev_time is None:
                continue
            dist = (ev_time - now).total_seconds()

            # Within blackout window?
            if -blackout_after <= dist <= blackout_before:
                if dist > 0:
                    return {
                        "blocked": True,
                        "reason": f"Event in {dist/60:.0f}min: {ev.get('title','')}",
                        "next_event": ev,
                        "phase": "PRE_EVENT",
                    }
                else:
                    return {
                        "blocked": True,
                        "reason": f"Post-event ({-dist/60:.0f}min ago): {ev.get('title','')}",
                        "next_event": ev,
                        "phase": "POST_EVENT",
                    }

            # Track closest upcoming
            if 0 < dist < closest_dist:
                closest_dist = dist
                closest = ev

        today_count = sum(1 for ev in self._events
                          if self._is_today(ev.get("time_utc", "")))

        return {
            "blocked": False,
            "reason": "",
            "next_event": closest,
            "next_in_min": round(closest_dist / 60, 1) if closest else None,
            "events_today": today_count,
        }

    def get_events(self) -> List[Dict]:
        return self._events

    def _fetch(self):
        """Fetch from free forex calendar API."""
        try:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            tomorrow = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d")

            events = []
            # Try forexfactory-style free API
            for url in [
                f"https://nfs.faireconomy.media/ff_calendar_thisweek.json",
            ]:
                try:
                    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
                    resp = urlopen(req, timeout=10)
                    data = json.loads(resp.read().decode())
                    if isinstance(data, list):
                        for item in data:
                            if item.get("country", "") != "USD":
                                continue
                            if item.get("impact", "") not in ("High", "Medium"):
                                continue
                            title = item.get("title", "")
                            # Filter to critical events only
                            is_critical = any(kw.lower() in title.lower() for kw in _CRITICAL_KEYWORDS)
                            if not is_critical and item.get("impact") != "High":
                                continue
                            events.append({
                                "title": title,
                                "time_utc": item.get("date", ""),
                                "impact": item.get("impact", ""),
                                "forecast": item.get("forecast", ""),
                                "previous": item.get("previous", ""),
                                "critical": is_critical,
                            })
                        break
                except Exception:
                    continue

            if not events:
                events = self._hardcoded_recurring()

            with self._lock:
                self._events = events
                self._last_fetch = time.time()
            self._save_cache(events)

        except Exception as e:
            print(f"[CALENDAR] Fetch error: {e}")

    def _hardcoded_recurring(self) -> List[Dict]:
        """Fallback: known recurring high-impact times."""
        now = datetime.now(timezone.utc)
        events = []
        # FOMC: typically 2pm ET (18:00 UTC) on scheduled dates
        # NFP: first Friday of month, 8:30am ET (12:30 UTC)
        # CPI: ~8:30am ET (12:30 UTC) mid-month
        # These are approximate — the API fetch is preferred
        dow = now.weekday()
        day = now.day

        # First Friday NFP
        if dow == 4 and day <= 7:
            nfp_time = now.replace(hour=12, minute=30, second=0, microsecond=0)
            events.append({
                "title": "Non-Farm Payrolls (estimated)",
                "time_utc": nfp_time.isoformat(),
                "impact": "High", "critical": True,
            })

        # CPI mid-month
        if 10 <= day <= 15 and dow < 5:
            cpi_time = now.replace(hour=12, minute=30, second=0, microsecond=0)
            events.append({
                "title": "CPI (estimated)",
                "time_utc": cpi_time.isoformat(),
                "impact": "High", "critical": True,
            })

        return events

    def _parse_time(self, time_str: str) -> Optional[datetime]:
        if not time_str:
            return None
        try:
            # Handle various formats
            for fmt in [
                "%Y-%m-%dT%H:%M:%S%z",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S",
                "%b %d, %Y %I:%M%p",
            ]:
                try:
                    dt = datetime.strptime(time_str, fmt)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    return dt
                except ValueError:
                    continue
            # pandas-style
            from datetime import datetime as _dt
            dt = _dt.fromisoformat(time_str.replace("Z", "+00:00"))
            return dt
        except Exception:
            return None

    def _is_today(self, time_str: str) -> bool:
        dt = self._parse_time(time_str)
        if not dt:
            return False
        return dt.date() == datetime.now(timezone.utc).date()

    def _save_cache(self, events):
        try:
            with open(_CACHE_FILE, "w") as f:
                json.dump({"events": events, "ts": time.time()}, f)
        except Exception:
            pass

    def _load_cache(self):
        if not os.path.isfile(_CACHE_FILE):
            return
        try:
            with open(_CACHE_FILE) as f:
                data = json.load(f)
            if time.time() - data.get("ts", 0) < 86400:  # cache valid 24h
                self._events = data.get("events", [])
                self._last_fetch = data.get("ts", 0)
        except Exception:
            pass


# Singleton
calendar = EconomicCalendar()
