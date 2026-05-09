"""
Higher-Timeframe Bias Engine (HTF).

Generates LONG-only trade opportunities based on:
  Step 1 — Macro filter        (DXY + US10Y)
  Step 2 — Weekly sentiment    (primary)
  Step 3 — Weekly entry logic
  Step 4 — Daily/H1 fallback
  Step 5 — Hard filters
  Step 6 — Confidence scoring

Returns a structured decision dict compatible with BaseStrategy.generate_signal().
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _no(reason: str) -> Dict[str, Any]:
    return {
        "strategy": "NONE",
        "decision": "NO_TRADE",
        "confidence": 0,
        "entry_zone": None,
        "sl": None,
        "tp": [],
        "reason": reason,
    }


def _structure(data: Dict, tf: str) -> str:
    return str(data.get("structure", {}).get(tf, "RANGE")).upper()


def _ohlc(data: Dict, tf: str) -> Dict[str, float]:
    return data.get("ohlc", {}).get(tf, {})


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Macro filter
# ─────────────────────────────────────────────────────────────────────────────

def _macro_bias(data: Dict) -> str:
    macro = data.get("macro", {})
    dxy = str(macro.get("dxy_trend", "RANGE")).upper()
    us10y = str(macro.get("us10y_trend", "RANGE")).upper()

    if dxy == "DOWN" and us10y == "DOWN":
        return "STRONG_BUY"
    if dxy == "DOWN" or us10y == "DOWN":
        return "WEAK_BUY"
    return "NEUTRAL_OR_SELL"


def _long_macro_block_reason(data: Dict, macro: str) -> Optional[str]:
    if macro != "NEUTRAL_OR_SELL":
        return None

    macro_data = data.get("macro", {})
    dxy = str(macro_data.get("dxy_trend", "RANGE")).upper()
    us10y = str(macro_data.get("us10y_trend", "RANGE")).upper()
    neutral_values = {"RANGE", "NEUTRAL", ""}

    if dxy in neutral_values and us10y in neutral_values:
        return "Macro not supportive for long (neutral: neither DXY nor US10Y is trending down)"

    if dxy == "UP" and us10y == "UP":
        return "Macro not supportive for long (bearish: DXY and US10Y are both trending up)"

    return "Macro not supportive for long (mixed: DXY and US10Y are not aligned for buys)"


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Weekly sentiment
# ─────────────────────────────────────────────────────────────────────────────

def _weekly_bias(data: Dict) -> str:
    price = float(data.get("price", 0.0))
    weekly = data.get("weekly_range", {})
    mid_7d = float(weekly.get("mid_7d", 0.0))
    low_7d = float(weekly.get("low_7d", 0.0))
    high_7d = float(weekly.get("high_7d", 0.0))

    d1 = _structure(data, "D1")
    h4 = _structure(data, "H4")
    d1_body = float(data.get("momentum", {}).get("D1_body_ratio", 0.0))

    # Strong bullish: price above mid, D1+H4 up, momentum present
    if (
        mid_7d > 0
        and price > mid_7d
        and d1 == "UP"
        and h4 == "UP"
        and d1_body >= 0.5
    ):
        return "STRONG_BULLISH"

    # Early bullish: price near weekly low with bounce, D1 not bearish
    if high_7d > low_7d > 0:
        range_size = high_7d - low_7d
        near_low = price <= low_7d + range_size * 0.20
        if near_low and d1 != "DOWN":
            return "EARLY_BULLISH"

    return "UNCLEAR"


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Weekly entry
# ─────────────────────────────────────────────────────────────────────────────

def _weekly_entry(data: Dict, macro: str) -> Optional[Dict[str, Any]]:
    weekly = data.get("weekly_range", {})
    high_7d = float(weekly.get("high_7d", 0.0))
    low_7d = float(weekly.get("low_7d", 0.0))
    price = float(data.get("price", 0.0))
    pullback_pct = float(data.get("pullback", {}).get("depth_pct", 0.0))
    h1 = _structure(data, "H1")

    # Pullback depth 20–50%
    if not (0.20 <= pullback_pct <= 0.50):
        return None

    # Price not at extreme highs (within 5% of 7d high = extended)
    if high_7d > 0 and price >= high_7d * 0.995:
        return None

    # H1 higher-low formation (UP structure is the proxy)
    if h1 != "UP":
        return None

    entry = price
    sl_distance = 20.0 if macro == "STRONG_BUY" else 25.0
    sl = round(entry - sl_distance, 2)
    tp1 = round(entry + sl_distance * 2, 2)
    tp2 = round(entry + sl_distance * 3, 2)
    tp3 = None  # trail runner — managed externally

    return {
        "strategy": "WEEKLY_LONG",
        "entry_zone": round(entry, 2),
        "sl": sl,
        "tp": [tp1, tp2, tp3],
        "sl_distance": sl_distance,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Daily / H1 fallback
# ─────────────────────────────────────────────────────────────────────────────

def _daily_entry(data: Dict) -> Optional[Dict[str, Any]]:
    h4 = _structure(data, "H4")
    h1 = _structure(data, "H1")

    if not (h4 == "UP" and h1 == "UP"):
        return None

    pullback_pct = float(data.get("pullback", {}).get("depth_pct", 0.0))
    h4_body = float(data.get("momentum", {}).get("H4_body_ratio", 0.0))

    if not (0.20 <= pullback_pct <= 0.40):
        return None

    if h4_body < 0.6:
        return None

    price = float(data.get("price", 0.0))
    entry = price
    sl_distance = 12.0
    sl = round(entry - sl_distance, 2)
    tp1 = round(entry + sl_distance * 1.5, 2)
    tp2 = round(entry + sl_distance * 2.0, 2)

    return {
        "strategy": "DAILY_LONG",
        "entry_zone": round(entry, 2),
        "sl": sl,
        "tp": [tp1, tp2],
        "sl_distance": sl_distance,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Hard filters
# ─────────────────────────────────────────────────────────────────────────────

def _hard_filter(data: Dict, macro: str) -> Optional[str]:
    news = data.get("news", {})
    if news.get("high_impact_soon"):
        return "High-impact news imminent"

    d1 = _structure(data, "D1")
    d1_body = float(data.get("momentum", {}).get("D1_body_ratio", 0.0))
    if d1 == "RANGE" and d1_body < 0.3:
        return "D1 ranging with no momentum"

    pullback_pct = float(data.get("pullback", {}).get("depth_pct", 0.0))
    if pullback_pct < 0.10:
        return "Pullback < 10% — chasing price"

    weekly = data.get("weekly_range", {})
    high_7d = float(weekly.get("high_7d", 0.0))
    price = float(data.get("price", 0.0))
    if high_7d > 0 and price >= high_7d * 0.998:
        return "Price at extreme weekly high — overextended"

    macro_block = _long_macro_block_reason(data, macro)
    if macro_block:
        return macro_block

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — Confidence scoring
# ─────────────────────────────────────────────────────────────────────────────

def _confidence(weekly_bias: str, macro: str, pullback_pct: float, h1: str) -> int:
    score = 0

    if weekly_bias == "STRONG_BULLISH":
        score += 40
    elif weekly_bias == "EARLY_BULLISH":
        score += 20

    if macro == "STRONG_BUY":
        score += 30
    elif macro == "WEAK_BUY":
        score += 15

    if 0.20 <= pullback_pct <= 0.50:
        score += 20

    if h1 == "UP":
        score += 10

    return min(score, 100)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run all 6 steps and return a structured decision dict.

    Args:
        data: Input dict matching the HTF bias engine schema.

    Returns:
        {
            "strategy": "WEEKLY_LONG" | "DAILY_LONG" | "NONE",
            "decision": "BUY" | "NO_TRADE",
            "confidence": int 0-100,
            "entry_zone": float | None,
            "sl": float | None,
            "tp": list[float | None],
            "reason": str,
        }
    """
    # Step 1
    macro = _macro_bias(data)

    # Step 5 — hard filters run before any entry logic
    block = _hard_filter(data, macro)
    if block:
        return _no(block)

    # Step 2
    w_bias = _weekly_bias(data)

    pullback_pct = float(data.get("pullback", {}).get("depth_pct", 0.0))
    h1 = _structure(data, "H1")
    conf = _confidence(w_bias, macro, pullback_pct, h1)

    # Step 3 — weekly path
    if w_bias != "UNCLEAR" and macro != "NEUTRAL_OR_SELL":
        entry = _weekly_entry(data, macro)
        if entry:
            return {
                **entry,
                "decision": "BUY",
                "confidence": conf,
                "reason": (
                    f"Weekly bias={w_bias} | macro={macro} | "
                    f"pullback={pullback_pct:.0%} | H1={h1}"
                ),
            }
        return _no(
            f"Weekly bias={w_bias} but entry conditions not met "
            f"(pullback={pullback_pct:.0%} H1={h1})"
        )

    # Step 4 — daily fallback
    entry = _daily_entry(data)
    if entry:
        return {
            **entry,
            "decision": "BUY",
            "confidence": conf,
            "reason": (
                f"Daily fallback | macro={macro} | "
                f"pullback={pullback_pct:.0%} | H1={h1}"
            ),
        }

    return _no(f"Weekly bias unclear and daily fallback conditions not met (macro={macro})")
