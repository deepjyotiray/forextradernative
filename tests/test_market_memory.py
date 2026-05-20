import pandas as pd

from engine.market_memory import assess_signal, compute_market_memory
from engine.market_state import compute_market_state


def _frame_from_closes(closes, *, wiggle=0.35):
    rows = []
    prev = float(closes[0]) - 0.2
    for close in closes:
        close = float(close)
        open_price = prev
        high = max(open_price, close) + wiggle
        low = min(open_price, close) - wiggle
        rows.append({"open": open_price, "high": high, "low": low, "close": close})
        prev = close
    return pd.DataFrame(rows)


def test_compute_market_memory_detects_bullish_four_hour_alignment():
    m1 = _frame_from_closes([100.0 + i * 0.08 for i in range(240)], wiggle=0.18)
    m5 = _frame_from_closes([100.0 + i * 0.35 for i in range(48)], wiggle=0.28)
    m15 = _frame_from_closes([100.0 + i * 0.95 for i in range(16)], wiggle=0.42)

    memory = compute_market_memory({"M1": m1, "M5": m5, "M15": m15})

    assert memory["ready"] is True
    assert memory["direction"] == "LONG"
    assert memory["confidence"] > 0.45
    assert memory["alignment"] > 0.60
    assert memory["range_position"] > 0.80


def test_compute_market_state_embeds_market_memory():
    m1 = _frame_from_closes([100.0 + i * 0.04 for i in range(240)], wiggle=0.15)
    m5 = _frame_from_closes([100.0 + i * 0.18 for i in range(48)], wiggle=0.22)
    m15 = _frame_from_closes([100.0 + i * 0.55 for i in range(16)], wiggle=0.30)
    h1 = _frame_from_closes([100.0 + i * 1.0 for i in range(60)], wiggle=0.55)
    h4 = _frame_from_closes([100.0 + i * 2.4 for i in range(60)], wiggle=0.80)
    d1 = _frame_from_closes([100.0 + i * 3.2 for i in range(60)], wiggle=1.10)

    state = compute_market_state({"M1": m1, "M5": m5, "M15": m15, "H1": h1, "H4": h4, "D1": d1})

    assert "market_memory" in state
    assert state["market_memory"]["ready"] is True
    assert state["market_memory"]["direction"] == "LONG"


def test_assess_signal_penalizes_buying_into_bearish_memory():
    market_state = {
        "market_memory": {
            "ready": True,
            "direction": "SHORT",
            "impulse_direction": "SHORT",
            "phase": "EXPANSION",
            "confidence": 0.82,
            "alignment": 0.78,
            "range_position": 0.18,
            "high_rejecting": True,
            "low_rejecting": False,
        }
    }

    assessment = assess_signal({"signal": "BUY", "_signal_family": "M15"}, market_state)

    assert assessment["enabled"] is True
    assert assessment["score_bonus"] < -4.0
    assert assessment["direction"] == "SHORT"
