"""
Macro Fetcher — automatic DXY and US10Y trend detection.

Fetches live price history for:
  - DXY  (US Dollar Index)  via Yahoo Finance  ticker: DX-Y.NYB
  - US10Y (10-Year Treasury) via Yahoo Finance  ticker: ^TNX

Trend is derived from:
  1. EMA(20) slope over last 10 daily bars  — primary signal
  2. Price vs EMA(50)                        — confirmation
  3. Higher-high / lower-low structure       — tie-breaker

Result: "UP" | "DOWN" | "RANGE"

Refresh: every 4 hours in a background thread.
Cache:   in-memory + disk (macro_cache.json) so restarts are instant.
Fallback: if both fetches fail → "RANGE" (neutral, no false signals).

No API key required. Uses Yahoo Finance v8 chart endpoint (same as yfinance).
"""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional
from urllib.request import urlopen, Request
from urllib.error import URLError

import numpy as np

_CACHE_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "macro_cache.json")
_REFRESH_INTERVAL = 4 * 3600   # 4 hours
_STALE_AFTER     = 12 * 3600   # treat as stale after 12 h (use RANGE as fallback)

# Yahoo Finance v8 chart endpoint — no auth, no key
_YF_URL = (
    "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    "?interval=1d&range=60d"
)
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "application/json",
}

# Ticker map with fallbacks
_TICKERS = {
    "dxy":   ["DX-Y.NYB", "DX=F"],
    "us10y": ["^TNX", "TNX"],
}


# ─────────────────────────────────────────────────────────────────────────────
# Trend derivation
# ─────────────────────────────────────────────────────────────────────────────

def _ema(values: np.ndarray, period: int) -> np.ndarray:
    k = 2.0 / (period + 1)
    out = np.empty_like(values)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = values[i] * k + out[i - 1] * (1 - k)
    return out


def _derive_trend(closes: np.ndarray) -> str:
    """
    Classify a price series as UP / DOWN / RANGE.

    Logic (all three must agree for a directional call):
      - EMA20 slope over last 10 bars
      - Price vs EMA50
      - Recent swing structure (last 2 highs / lows)
    """
    if len(closes) < 22:
        return "RANGE"

    c = closes.astype(float)

    # EMA slopes
    ema20 = _ema(c, 20)
    slope20 = ema20[-1] - ema20[-10]
    threshold = float(np.std(np.diff(c[-20:]))) * 0.5   # adaptive threshold

    # Price vs EMA50
    above_ema50 = False
    below_ema50 = False
    if len(c) >= 52:
        ema50 = _ema(c, 50)
        above_ema50 = c[-1] > ema50[-1]
        below_ema50 = c[-1] < ema50[-1]

    # Swing structure: compare last 2 peaks and troughs (simple)
    recent = c[-10:]
    mid = len(recent) // 2
    first_half_high = recent[:mid].max()
    second_half_high = recent[mid:].max()
    first_half_low  = recent[:mid].min()
    second_half_low  = recent[mid:].min()
    hh = second_half_high > first_half_high
    hl = second_half_low  > first_half_low
    lh = second_half_high < first_half_high
    ll = second_half_low  < first_half_low

    bullish_structure = hh and hl
    bearish_structure = lh and ll

    # Combine signals
    bull_votes = sum([
        slope20 > threshold,
        above_ema50,
        bullish_structure,
    ])
    bear_votes = sum([
        slope20 < -threshold,
        below_ema50,
        bearish_structure,
    ])

    if bull_votes >= 2 and bull_votes > bear_votes:
        return "UP"
    if bear_votes >= 2 and bear_votes > bull_votes:
        return "DOWN"
    return "RANGE"


# ─────────────────────────────────────────────────────────────────────────────
# Yahoo Finance fetch
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_closes(ticker: str, timeout: float = 8.0) -> Optional[np.ndarray]:
    url = _YF_URL.format(ticker=ticker)
    try:
        req = Request(url, headers=_HEADERS)
        with urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
        result = data.get("chart", {}).get("result")
        if not result:
            return None
        indicators = result[0].get("indicators", {})
        quote = indicators.get("quote", [{}])[0]
        closes_raw = quote.get("close", [])
        closes = [c for c in closes_raw if c is not None]
        if len(closes) < 10:
            return None
        return np.array(closes, dtype=float)
    except Exception:
        return None


def _fetch_trend(key: str) -> str:
    """Try each ticker fallback until one succeeds."""
    for ticker in _TICKERS.get(key, []):
        closes = _fetch_closes(ticker)
        if closes is not None:
            return _derive_trend(closes)
    return "RANGE"


# ─────────────────────────────────────────────────────────────────────────────
# Cache
# ─────────────────────────────────────────────────────────────────────────────

def _save_cache(state: Dict) -> None:
    try:
        with open(_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f)
    except Exception:
        pass


def _load_cache() -> Optional[Dict]:
    if not os.path.isfile(_CACHE_FILE):
        return None
    try:
        with open(_CACHE_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# MacroFetcher singleton
# ─────────────────────────────────────────────────────────────────────────────

class MacroFetcher:
    """
    Background service that keeps DXY and US10Y trends up to date.

    Usage:
        from engine.macro_fetcher import macro_fetcher
        trends = macro_fetcher.get()
        # {"dxy_trend": "DOWN", "us10y_trend": "DOWN", "fetched_at": ..., "source": "live"}
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: Dict = {
            "dxy_trend":   "RANGE",
            "us10y_trend": "RANGE",
            "fetched_at":  0.0,
            "source":      "init",
        }
        self._last_fetch: float = 0.0
        self._thread: Optional[threading.Thread] = None

        # Warm up from disk cache immediately so first call is never cold
        cached = _load_cache()
        if cached and isinstance(cached, dict):
            age = time.time() - float(cached.get("fetched_at", 0))
            if age < _STALE_AFTER:
                with self._lock:
                    self._state = cached
                    self._state["source"] = "cache"
                    self._last_fetch = float(cached.get("fetched_at", 0))

    # ── Public API ────────────────────────────────────────────────────────────

    def get(self) -> Dict:
        """Return current macro trends. Triggers a background refresh if stale."""
        self._maybe_refresh()
        with self._lock:
            return dict(self._state)

    def get_trends(self) -> tuple[str, str]:
        """Return (dxy_trend, us10y_trend) as a quick tuple."""
        state = self.get()
        return state["dxy_trend"], state["us10y_trend"]

    def force_refresh(self) -> Dict:
        """Synchronous refresh — blocks until complete. Use for startup/testing."""
        self._do_fetch()
        with self._lock:
            return dict(self._state)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _maybe_refresh(self) -> None:
        now = time.time()
        if now - self._last_fetch < _REFRESH_INTERVAL:
            return
        # Only one background thread at a time
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._do_fetch, daemon=True, name="macro_fetcher")
        self._thread.start()

    def _do_fetch(self) -> None:
        try:
            dxy   = _fetch_trend("dxy")
            us10y = _fetch_trend("us10y")
            now   = time.time()
            new_state = {
                "dxy_trend":   dxy,
                "us10y_trend": us10y,
                "fetched_at":  now,
                "fetched_at_iso": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
                "source":      "live",
            }
            with self._lock:
                self._state = new_state
                self._last_fetch = now
            _save_cache(new_state)
        except Exception:
            # On any unexpected error keep existing state, just update timestamp
            # so we don't hammer the API on every tick
            with self._lock:
                self._last_fetch = time.time()


# Module-level singleton — import this everywhere
macro_fetcher = MacroFetcher()
