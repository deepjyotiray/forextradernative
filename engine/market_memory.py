"""
Four-hour market memory built from recent M1/M5/M15 candles.

This module gives the engine a compact view of the last four hours so it can
reason about directional alignment, expansion/compression, range location, and
possible exhaustion before choosing a trade.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .indicators import atr, ema

_WINDOW_BARS = {"M1": 240, "M5": 48, "M15": 16}
_TF_WEIGHTS = {"M1": 0.20, "M5": 0.35, "M15": 0.45}


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _direction_from_signal(signal: Dict[str, Any]) -> str:
    action = str(signal.get("signal") or "").upper()
    if action == "BUY":
        return "LONG"
    if action == "SELL":
        return "SHORT"
    return "NEUTRAL"


def _body_pressure(df: pd.DataFrame, lookback: int = 12) -> float:
    sub = df.tail(min(len(df), max(3, lookback)))
    if sub.empty:
        return 0.0
    body = (sub["close"].astype(float) - sub["open"].astype(float)).to_numpy()
    rng = (sub["high"].astype(float) - sub["low"].astype(float)).replace(0, np.nan).to_numpy()
    ratios = np.divide(np.abs(body), rng, out=np.zeros_like(body, dtype=float), where=~np.isnan(rng))
    signed = np.sign(body) * ratios
    return _clip(float(np.mean(signed)) if len(signed) else 0.0, -1.0, 1.0)


def _atr_value(df: pd.DataFrame, period: int = 14) -> float:
    if df is None or len(df) < 3:
        return 0.0
    h = df["high"].astype(float).to_numpy()
    l = df["low"].astype(float).to_numpy()
    c = df["close"].astype(float).to_numpy()
    return float(atr(h, l, c, period)[-1])


def _timeframe_memory(tf: str, df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    bars = int(_WINDOW_BARS.get(tf, 0) or 0)
    if df is None or len(df) < max(8, min(12, bars // 3 or 8)):
        return {"ready": False}

    sub = df.tail(min(len(df), bars)).copy()
    c = sub["close"].astype(float).to_numpy()
    h = sub["high"].astype(float).to_numpy()
    l = sub["low"].astype(float).to_numpy()
    close_now = float(c[-1])
    range_high = float(np.max(h))
    range_low = float(np.min(l))
    range_width = max(0.01, range_high - range_low)

    ema_fast = ema(c, 9)
    ema_slow = ema(c, 20)
    atr_last = max(0.05, _atr_value(sub, 14))

    net_move = float(close_now - c[0])
    slope = float(ema_fast[-1] - ema_fast[-3]) if len(ema_fast) >= 3 else net_move
    gap = float(ema_fast[-1] - ema_slow[-1])
    close_position = _clip((close_now - range_low) / range_width, 0.0, 1.0)

    move_score = _clip(net_move / max(atr_last * 3.0, 0.25), -1.0, 1.0)
    slope_score = _clip(slope / max(atr_last * 1.5, 0.25), -1.0, 1.0)
    gap_score = _clip(gap / max(atr_last * 2.0, 0.25), -1.0, 1.0)
    position_score = _clip((close_position - 0.5) * 2.0, -1.0, 1.0)
    body_score = _body_pressure(sub)

    score = (
        move_score * 0.35
        + slope_score * 0.25
        + gap_score * 0.25
        + body_score * 0.10
        + position_score * 0.05
    )

    half = max(4, len(sub) // 2)
    early = sub.head(half)
    recent = sub.tail(half)
    early_atr = max(0.05, _atr_value(early, min(14, max(3, len(early) - 1))))
    recent_atr = max(0.05, _atr_value(recent, min(14, max(3, len(recent) - 1))))
    expansion_score = _clip((recent_atr / early_atr) - 1.0, -1.0, 1.0)

    trend_efficiency = _clip(abs(net_move) / range_width, 0.0, 1.0)
    latest = sub.iloc[-1]
    tolerance = max(atr_last * 0.20, range_width * 0.03, 0.15)
    high_rejecting = bool(float(latest["high"]) >= (range_high - tolerance) and float(latest["close"]) < (range_high - tolerance * 0.35))
    low_rejecting = bool(float(latest["low"]) <= (range_low + tolerance) and float(latest["close"]) > (range_low + tolerance * 0.35))

    direction = "LONG" if score >= 0.18 else "SHORT" if score <= -0.18 else "NEUTRAL"
    return {
        "ready": True,
        "bars": len(sub),
        "score": round(float(score), 4),
        "direction": direction,
        "close_position": round(close_position, 4),
        "range_high": round(range_high, 3),
        "range_low": round(range_low, 3),
        "range_width": round(range_width, 3),
        "atr": round(atr_last, 4),
        "net_move": round(net_move, 4),
        "trend_efficiency": round(trend_efficiency, 4),
        "body_pressure": round(body_score, 4),
        "expansion_score": round(float(expansion_score), 4),
        "high_rejecting": high_rejecting,
        "low_rejecting": low_rejecting,
    }


def compute_market_memory(candles: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    tf_data = {
        "M1": _timeframe_memory("M1", candles.get("M1")),
        "M5": _timeframe_memory("M5", candles.get("M5")),
        "M15": _timeframe_memory("M15", candles.get("M15")),
    }
    valid = {tf: payload for tf, payload in tf_data.items() if payload.get("ready")}
    if not valid:
        return {
            "enabled": True,
            "ready": False,
            "window_hours": 4,
            "direction": "NEUTRAL",
            "confidence": 0.0,
            "alignment": 0.0,
            "phase": "BALANCED",
            "range_position": 0.5,
            "summary": "4h market memory unavailable",
            "timeframes": tf_data,
        }

    total_weight = sum(_TF_WEIGHTS.get(tf, 0.0) for tf in valid) or 1.0
    weighted_score = sum(_safe_float(payload.get("score")) * _TF_WEIGHTS.get(tf, 0.0) for tf, payload in valid.items()) / total_weight
    impulse_weights = {"M1": 0.50, "M5": 0.35, "M15": 0.15}
    impulse_total = sum(impulse_weights.get(tf, 0.0) for tf in valid) or 1.0
    impulse_score = sum(_safe_float(payload.get("score")) * impulse_weights.get(tf, 0.0) for tf, payload in valid.items()) / impulse_total
    expansion_score = sum(_safe_float(payload.get("expansion_score")) * _TF_WEIGHTS.get(tf, 0.0) for tf, payload in valid.items()) / total_weight

    score_sign = 1 if weighted_score > 0.12 else -1 if weighted_score < -0.12 else 0
    agreement_weight = 0.0
    if score_sign != 0:
        agreement_weight = sum(
            _TF_WEIGHTS.get(tf, 0.0)
            for tf, payload in valid.items()
            if (_safe_float(payload.get("score")) > 0.08 and score_sign > 0)
            or (_safe_float(payload.get("score")) < -0.08 and score_sign < 0)
        )
    alignment = _clip(agreement_weight / total_weight if total_weight > 0 else 0.0, 0.0, 1.0)

    ref = valid.get("M1") or valid.get("M5") or valid.get("M15") or next(iter(valid.values()))
    range_position = _clip(_safe_float(ref.get("close_position"), 0.5), 0.0, 1.0)
    range_high = _safe_float(ref.get("range_high"), 0.0)
    range_low = _safe_float(ref.get("range_low"), 0.0)

    phase = "BALANCED"
    if expansion_score >= 0.20:
        phase = "EXPANSION"
    elif expansion_score <= -0.20:
        phase = "COMPRESSION"

    direction = "NEUTRAL"
    if weighted_score >= 0.18 and alignment >= 0.45:
        direction = "LONG"
    elif weighted_score <= -0.18 and alignment >= 0.45:
        direction = "SHORT"
    elif abs(weighted_score) >= 0.28:
        direction = "LONG" if weighted_score > 0 else "SHORT"

    impulse_direction = "LONG" if impulse_score >= 0.12 else "SHORT" if impulse_score <= -0.12 else "NEUTRAL"
    confidence = _clip(abs(weighted_score) * 0.75 + alignment * 0.35 + min(0.20, abs(impulse_score) * 0.15), 0.0, 1.0)
    high_rejecting = any(bool(payload.get("high_rejecting")) for payload in valid.values())
    low_rejecting = any(bool(payload.get("low_rejecting")) for payload in valid.values())

    summary = (
        f"4h {direction.lower()} memory | phase={phase.lower()} | "
        f"conf={confidence:.2f} | align={alignment:.2f} | pos={range_position:.2f}"
    )
    return {
        "enabled": True,
        "ready": True,
        "window_hours": 4,
        "direction": direction,
        "confidence": round(confidence, 3),
        "alignment": round(alignment, 3),
        "phase": phase,
        "score": round(weighted_score, 4),
        "impulse_score": round(impulse_score, 4),
        "impulse_direction": impulse_direction,
        "expansion_score": round(expansion_score, 4),
        "range_position": round(range_position, 4),
        "range_high": round(range_high, 3),
        "range_low": round(range_low, 3),
        "high_rejecting": high_rejecting,
        "low_rejecting": low_rejecting,
        "summary": summary,
        "timeframes": tf_data,
    }


def assess_signal(signal: Dict[str, Any], market_state: Dict[str, Any]) -> Dict[str, Any]:
    nested_state = market_state.get("market_state") or {}
    memory = market_state.get("market_memory") or nested_state.get("market_memory") or {}
    if not memory.get("ready"):
        return {
            "enabled": False,
            "score_bonus": 0.0,
            "direction": "NEUTRAL",
            "phase": "BALANCED",
            "confidence": 0.0,
            "alignment": 0.0,
            "range_position": 0.5,
            "reason": "4h memory unavailable",
        }

    trade_dir = _direction_from_signal(signal)
    if trade_dir not in {"LONG", "SHORT"}:
        return {
            "enabled": True,
            "score_bonus": 0.0,
            "direction": str(memory.get("direction") or "NEUTRAL"),
            "phase": str(memory.get("phase") or "BALANCED"),
            "confidence": _safe_float(memory.get("confidence"), 0.0),
            "alignment": _safe_float(memory.get("alignment"), 0.0),
            "range_position": _safe_float(memory.get("range_position"), 0.5),
            "reason": "No directional trade to score",
        }

    dominant = str(memory.get("direction") or "NEUTRAL").upper()
    impulse = str(memory.get("impulse_direction") or "NEUTRAL").upper()
    phase = str(memory.get("phase") or "BALANCED").upper()
    confidence = _safe_float(memory.get("confidence"), 0.0)
    alignment = _safe_float(memory.get("alignment"), 0.0)
    range_position = _safe_float(memory.get("range_position"), 0.5)
    signal_family = str(signal.get("_signal_family") or "").upper()
    high_rejecting = bool(memory.get("high_rejecting"))
    low_rejecting = bool(memory.get("low_rejecting"))

    score = 0.0
    reasons = []

    if dominant == trade_dir:
        aligned_bonus = 2.0 + confidence * 3.0 + alignment * 1.5
        if signal_family in {"TREND", "SWING", "INTRADAY"}:
            aligned_bonus += 0.6
        score += aligned_bonus
        reasons.append(f"4h direction aligns {dominant}")
    elif dominant in {"LONG", "SHORT"}:
        against_penalty = 2.2 + confidence * 3.2
        if signal_family in {"TREND", "SWING", "INTRADAY"}:
            against_penalty += 0.6
        score -= against_penalty
        reasons.append(f"4h direction opposes {dominant}")
    else:
        reasons.append("4h direction neutral")

    if impulse == trade_dir:
        score += 1.2
        reasons.append(f"short-term impulse supports {trade_dir}")
    elif impulse in {"LONG", "SHORT"}:
        score -= 0.9
        reasons.append(f"short-term impulse opposes {trade_dir}")

    if phase == "EXPANSION":
        if dominant == trade_dir:
            score += 0.9
            reasons.append("expansion favors continuation")
        elif dominant in {"LONG", "SHORT"}:
            score -= 0.6
    elif phase == "COMPRESSION" and dominant == "NEUTRAL":
        score -= 0.5
        reasons.append("4h compression lacks clear directional edge")

    if trade_dir == "LONG":
        if range_position >= 0.90:
            score -= 1.8
            reasons.append("long too near 4h high")
        elif dominant == "LONG" and range_position <= 0.35:
            score += 1.0
            reasons.append("bullish pullback location")
        if low_rejecting:
            score += 1.0
            reasons.append("4h low reclaim supports long")
        if high_rejecting:
            score -= 0.8
    else:
        if range_position <= 0.10:
            score -= 1.8
            reasons.append("short too near 4h low")
        elif dominant == "SHORT" and range_position >= 0.65:
            score += 1.0
            reasons.append("bearish pullback location")
        if high_rejecting:
            score += 1.0
            reasons.append("4h high rejection supports short")
        if low_rejecting:
            score -= 0.8

    score = _clip(score, -8.0, 8.0)
    return {
        "enabled": True,
        "score_bonus": round(score, 3),
        "direction": dominant,
        "phase": phase,
        "confidence": round(confidence, 3),
        "alignment": round(alignment, 3),
        "range_position": round(range_position, 3),
        "impulse_direction": impulse,
        "high_rejecting": high_rejecting,
        "low_rejecting": low_rejecting,
        "reason": "; ".join(reasons[:4]),
    }
