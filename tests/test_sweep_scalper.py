from unittest.mock import patch
import pandas as pd

from engine import decision_logger
from engine.sweep_scalper import SweepScalper, _ema_slope_alignment_ok, _marginal_setup_has_directional_support
from engine.session_filter import get_session_at
from datetime import datetime, timezone


def test_log_scalper_decision_uses_live_quality_threshold():
    with patch.object(decision_logger.cfg, "SCALPER_QUALITY_THRESHOLD", 0.4):
        with patch("engine.decision_logger.log_decision") as mock_log_decision:
            decision_logger.log_scalper_decision(
                setup_direction="SHORT",
                quality_score=0.55,
                spread_mean=0.15,
                spread_std=0.03,
                compression_ok=True,
                decision="TRADE_TAKEN",
                reason="test",
                price=4558.0,
                sweep_level=4560.54,
            )

    assert mock_log_decision.call_args.kwargs["threshold"] == 0.4


def test_marginal_short_requires_negative_pressure_or_non_positive_slope():
    assert not _marginal_setup_has_directional_support("SHORT", 0.55, 0.189, 0.10)
    assert _marginal_setup_has_directional_support("SHORT", 0.55, -0.05, 0.10)
    assert _marginal_setup_has_directional_support("SHORT", 0.55, 0.189, -0.02)


def test_stronger_quality_setup_skips_marginal_support_gate():
    assert _marginal_setup_has_directional_support("SHORT", 0.70, 0.189, 0.10)


def test_ema15_alignment_blocks_long_when_fast_slope_is_negative():
    assert not _ema_slope_alignment_ok("LONG", -0.12, 0.09)


def test_ema15_alignment_blocks_short_when_fast_slope_is_positive():
    assert not _ema_slope_alignment_ok("SHORT", 0.12, -0.09)


def test_ema15_alignment_requires_both_slopes_to_support_direction():
    assert _ema_slope_alignment_ok("LONG", 0.12, 0.09)
    assert _ema_slope_alignment_ok("SHORT", -0.12, -0.09)


def test_session_classifier_marks_midnight_utc_as_asian():
    now_utc = datetime(2026, 4, 30, 0, 16, tzinfo=timezone.utc)

    assert get_session_at(now_utc) == "ASIAN"


def _m15_box_df(direction="LONG"):
    rows = []
    base = 100.0
    for idx in range(24):
        if direction == "LONG":
            close = base + idx * 0.18
        else:
            close = base - idx * 0.18
        open_price = close - 0.05 if direction == "LONG" else close + 0.05
        rows.append(
            {
                "datetime": pd.Timestamp("2026-01-01") + pd.Timedelta(minutes=15 * idx),
                "open": float(open_price),
                "high": float(max(open_price, close) + 0.25),
                "low": float(min(open_price, close) - 0.25),
                "close": float(close),
                "volume": 1000.0 + idx,
            }
        )

    if direction == "LONG":
        box = [
            {"open": 104.55, "high": 104.70, "low": 104.48, "close": 104.62},
            {"open": 104.60, "high": 104.74, "low": 104.52, "close": 104.66},
            {"open": 104.64, "high": 104.76, "low": 104.54, "close": 104.68},
            {"open": 104.66, "high": 104.78, "low": 104.56, "close": 104.70},
        ]
    else:
        box = [
            {"open": 95.45, "high": 95.52, "low": 95.30, "close": 95.38},
            {"open": 95.40, "high": 95.48, "low": 95.26, "close": 95.34},
            {"open": 95.36, "high": 95.44, "low": 95.22, "close": 95.30},
            {"open": 95.32, "high": 95.40, "low": 95.18, "close": 95.26},
        ]
    for i, row in enumerate(box):
        row["datetime"] = pd.Timestamp("2026-01-02") + pd.Timedelta(minutes=15 * i)
        row["volume"] = 2000.0 + i
        rows.append(row)
    return pd.DataFrame(rows)


def test_bias_consolidation_detects_long_setup_near_box_low():
    scalper = SweepScalper()
    m15 = _m15_box_df("LONG")
    m30 = _m15_box_df("LONG")

    with patch("engine.sweep_scalper.cfg.SCALPER_BIAS_CONSOLIDATION_ENABLED", True):
        setup = scalper._detect_bias_consolidation_setup(
            m15_df=m15,
            m30_df=m30,
            bias={"direction": "LONG"},
            regime={"state": "RANGING"},
            price=104.56,
            spread=0.18,
        )

    assert setup is not None
    assert setup["direction"] == "LONG"
    assert setup["anchor_level"] == round(float(m15.tail(4)["low"].min()), 2)


def test_bias_consolidation_rejects_long_when_price_not_near_box_low():
    scalper = SweepScalper()
    m15 = _m15_box_df("LONG")
    m30 = _m15_box_df("LONG")

    with patch("engine.sweep_scalper.cfg.SCALPER_BIAS_CONSOLIDATION_ENABLED", True):
        setup = scalper._detect_bias_consolidation_setup(
            m15_df=m15,
            m30_df=m30,
            bias={"direction": "LONG"},
            regime={"state": "RANGING"},
            price=104.92,
            spread=0.18,
        )

    assert setup is None


def test_sideways_range_guard_blocks_long_at_top_of_box():
    scalper = SweepScalper()
    m15 = _m15_box_df("LONG")
    m30 = _m15_box_df("LONG")

    guard = scalper._validate_sideways_range_entry(
        direction="LONG",
        price=104.92,
        spread=0.18,
        regime={"state": "RANGING"},
        m15_df=m15,
        m30_df=m30,
    )

    assert guard is not None
    assert guard["allowed"] is False
    assert "near range low" in guard["reason"]


def test_sideways_range_guard_allows_short_near_range_high():
    scalper = SweepScalper()
    m15 = _m15_box_df("SHORT")
    m30 = _m15_box_df("SHORT")

    guard = scalper._validate_sideways_range_entry(
        direction="SHORT",
        price=95.36,
        spread=0.18,
        regime={"state": "RANGING"},
        m15_df=m15,
        m30_df=m30,
    )

    assert guard is not None
    assert guard["allowed"] is True
    assert guard["label"] == "M15/M30 box"


def test_m15_trade_direction_prefers_bias_structure_first():
    scalper = SweepScalper()
    m15 = _m15_box_df("LONG")

    decision = scalper._resolve_m15_trade_direction(
        m15,
        {
            "m15_structure": {"pattern": "BULLISH", "bos": True, "bos_direction": "LONG"},
            "m15_pullback": {"active": False, "direction": None},
        },
    )

    assert decision["direction"] == "LONG"
    assert "BOS LONG" in decision["reason"]


def test_m15_trade_direction_rejects_mixed_context_without_clear_trend():
    scalper = SweepScalper()
    m15 = _m15_box_df("LONG")
    m15["close"] = 100.0
    m15["open"] = 100.0
    m15["high"] = 100.1
    m15["low"] = 99.9

    decision = scalper._resolve_m15_trade_direction(
        m15,
        {
            "m15_structure": {"pattern": "MIXED", "bos": False, "bos_direction": None},
            "m15_pullback": {"active": False, "direction": None},
        },
    )

    assert decision["direction"] == "NEUTRAL"
    assert "mixed" in decision["reason"].lower()


def test_ltf_candle_stats_classifies_high_close_correctly():
    scalper = SweepScalper()
    df = pd.DataFrame(
        [
            {
                "datetime": pd.Timestamp("2026-01-02 10:00:00"),
                "open": 100.10,
                "high": 100.80,
                "low": 100.00,
                "close": 100.72,
                "volume": 1200.0,
            }
        ]
    )

    stats = scalper._ltf_candle_stats(df)

    assert stats["close_position"] == "HIGH"
    assert stats["body_ratio"] > 0.6


def test_micro_trade_plan_stays_within_spec_bounds():
    scalper = SweepScalper()

    sl, tp, sl_dist = scalper._compute_micro_trade_plan(
        price=2500.0,
        direction="LONG",
        body_ratio=0.78,
        volume_ratio=1.55,
        candle_range=0.92,
        spread=0.18,
    )

    assert 0.50 <= sl_dist <= 0.70
    assert 0.50 <= round(tp - 2500.0, 2) <= 1.00
    assert sl < 2500.0
    assert tp > 2500.0


def test_range_setup_candle_confirmation_allows_mid_close_if_directional():
    scalper = SweepScalper()

    ok, reason = scalper._setup_candle_confirmation_ok(
        setup_type="RANGE_EDGE",
        direction="LONG",
        close_position="MID",
        last_open=100.0,
        last_close=100.2,
    )

    assert ok is True
    assert reason == ""


def test_sweep_setup_still_requires_edge_close():
    scalper = SweepScalper()

    ok, reason = scalper._setup_candle_confirmation_ok(
        setup_type="SWEEP",
        direction="SHORT",
        close_position="MID",
        last_open=100.2,
        last_close=100.0,
    )

    assert ok is False
    assert "close near candle low" in reason


def _wide_range_df(direction="SHORT"):
    rows = []
    for idx in range(24):
        base = 100.0 + idx * 0.05
        rows.append(
            {
                "datetime": pd.Timestamp("2026-01-01") + pd.Timedelta(minutes=15 * idx),
                "open": base,
                "high": base + 0.35,
                "low": base - 0.35,
                "close": base + 0.02,
                "volume": 1400.0 + idx,
            }
        )
    if direction == "SHORT":
        box = [
            {"open": 102.40, "high": 103.80, "low": 101.90, "close": 103.10},
            {"open": 103.10, "high": 104.05, "low": 102.10, "close": 103.55},
            {"open": 103.55, "high": 104.20, "low": 102.30, "close": 103.88},
            {"open": 103.88, "high": 104.35, "low": 102.40, "close": 104.02},
            {"open": 104.02, "high": 104.50, "low": 102.55, "close": 104.10},
            {"open": 104.10, "high": 104.60, "low": 102.60, "close": 104.18},
        ]
    else:
        box = [
            {"open": 102.50, "high": 104.10, "low": 100.80, "close": 101.20},
            {"open": 101.20, "high": 103.95, "low": 100.55, "close": 100.98},
            {"open": 100.98, "high": 103.70, "low": 100.35, "close": 100.85},
            {"open": 100.85, "high": 103.55, "low": 100.20, "close": 100.72},
            {"open": 100.72, "high": 103.45, "low": 100.05, "close": 100.60},
            {"open": 100.60, "high": 103.20, "low": 99.95, "close": 100.48},
        ]
    for i, row in enumerate(box):
        row["datetime"] = pd.Timestamp("2026-01-03") + pd.Timedelta(minutes=30 * i)
        row["volume"] = 2200.0 + i
        rows.append(row)
    return pd.DataFrame(rows)


def test_range_edge_setup_detects_short_near_wide_range_high():
    scalper = SweepScalper()
    m15 = _wide_range_df("SHORT")
    m30 = _wide_range_df("SHORT")

    setup = scalper._detect_range_edge_scalp_setup(
        m15_df=m15,
        m30_df=m30,
        bias={"direction": "SHORT"},
        regime={"state": "RANGING"},
        price=104.28,
        spread=0.18,
    )

    assert setup is not None
    assert setup["direction"] == "SHORT"
    assert setup["label"] == "M15/M30 wide range"


def test_range_edge_setup_rejects_when_not_near_range_edge():
    scalper = SweepScalper()
    m15 = _wide_range_df("SHORT")
    m30 = _wide_range_df("SHORT")

    setup = scalper._detect_range_edge_scalp_setup(
        m15_df=m15,
        m30_df=m30,
        bias={"direction": "SHORT"},
        regime={"state": "RANGING"},
        price=103.10,
        spread=0.18,
    )

    assert setup is None
