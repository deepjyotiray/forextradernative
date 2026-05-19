from __future__ import annotations

from typing import Any, Dict, List, Optional

import config as cfg


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _resolve_tf_state(market_state: Dict[str, Any], tf: str) -> str:
    trend = market_state.get("trend") or {}
    structure = market_state.get("structure") or {}
    struct_value = str(structure.get(tf, "RANGE") or "RANGE").upper()
    if struct_value != "RANGE":
        return struct_value
    return str(trend.get(tf, "RANGE") or "RANGE").upper()


def _resolve_m15_structure(m15_df) -> str:
    if m15_df is None or len(m15_df) < 5:
        return "RANGE"
    frame = m15_df.rename(columns=lambda c: str(c).lower())
    if "high" not in frame or "low" not in frame or "close" not in frame:
        return "RANGE"
    closed = frame.iloc[-5:-1]
    if len(closed) < 4:
        return "RANGE"
    highs = [float(v) for v in closed["high"].tolist()]
    lows = [float(v) for v in closed["low"].tolist()]
    closes = [float(v) for v in closed["close"].tolist()]
    rising = all(highs[i] >= highs[i - 1] and lows[i] >= lows[i - 1] for i in range(1, len(highs)))
    falling = all(highs[i] <= highs[i - 1] and lows[i] <= lows[i - 1] for i in range(1, len(highs)))
    if rising and closes[-1] >= closes[0]:
        return "UP"
    if falling and closes[-1] <= closes[0]:
        return "DOWN"
    return "RANGE"


def _confidence_label(score: float) -> str:
    magnitude = abs(score)
    if magnitude >= 55:
        return "HIGH"
    if magnitude >= 30:
        return "MEDIUM"
    return "LOW"


def _confidence_rank(label: str) -> int:
    return {"LOW": 0, "MEDIUM": 1, "HIGH": 2}.get(str(label or "").upper(), 0)


def _direction_from_score(score: float) -> str:
    if score >= 18:
        return "LONG"
    if score <= -18:
        return "SHORT"
    return "NEUTRAL"


def _current_price(data: Dict[str, Any]) -> float:
    tick = data.get("tick") or {}
    return _safe_float(tick.get("bid"), _safe_float(tick.get("ask"), 0.0))


def _zone_hold_score(zone_type: str, price: float, zone_low: float, zone_high: float, zone_mid: float) -> tuple[float, str]:
    zone_type = str(zone_type or "").upper()
    if price <= 0:
        return 0.0, "price unavailable"
    if zone_type == "DEMAND":
        if price >= zone_high:
            return 12.0, "holding above demand"
        if price >= zone_mid:
            return 8.0, "holding upper-half of demand"
        if price < zone_low:
            return -12.0, "lost demand"
        return 2.0, "inside lower-half of demand"
    if zone_type == "SUPPLY":
        if price > zone_high:
            return 14.0, "reclaimed above supply"
        if price > zone_mid:
            return 9.0, "holding upper-half of supply"
        if price <= zone_low:
            return -10.0, "holding below supply"
        return -2.0, "inside upper-half of supply"
    return 0.0, "zone hold neutral"


def _zone_reclaim_breakdown_score(zone_type: str, zone_low: float, zone_high: float, zone_mid: float, m15_df) -> tuple[float, str]:
    if m15_df is None or len(m15_df) < 3:
        return 0.0, "no reclaim/breakdown context"
    frame = m15_df.rename(columns=lambda c: str(c).lower())
    if "close" not in frame:
        return 0.0, "no close context"
    closed = frame.iloc[-3:-1]
    if len(closed) < 2:
        return 0.0, "no reclaim/breakdown context"
    prev_close = float(closed.iloc[0]["close"])
    last_close = float(closed.iloc[1]["close"])
    zone_type = str(zone_type or "").upper()
    if zone_type == "DEMAND":
        if last_close > zone_mid and prev_close > zone_mid:
            return 10.0, "demand reclaim holding"
        if last_close < zone_low:
            return -14.0, "demand breakdown"
    elif zone_type == "SUPPLY":
        if last_close > zone_mid and prev_close > zone_mid:
            return 12.0, "supply reclaimed"
        if last_close < zone_low and prev_close < zone_mid:
            return -10.0, "supply rejected lower"
    return 0.0, "reclaim/breakdown neutral"


def _final_action_from_bias(
    setup_direction: str,
    bias_direction: str,
    score: float,
    confidence_label: str,
    cfg_module,
) -> tuple[str, str]:
    min_conf = str(getattr(cfg_module, "M15_ZONE_MICRO_BIAS_MIN_CONFIDENCE", "MEDIUM") or "MEDIUM").upper()
    min_score = max(0.0, _safe_float(getattr(cfg_module, "M15_ZONE_MICRO_BIAS_MIN_SCORE", 30.0), 30.0))
    if bias_direction == "NEUTRAL":
        return "BLOCK", "NONE"
    if abs(_safe_float(score, 0.0)) < min_score:
        return "BLOCK", "NONE"
    if _confidence_rank(confidence_label) < _confidence_rank(min_conf):
        return "BLOCK", "NONE"

    base_direction = str(setup_direction or "").upper()
    wanted = "LONG" if base_direction == "BUY" else "SHORT"
    if bias_direction == wanted:
        return "ALLOW_BASE_TRADE", base_direction

    if bool(getattr(cfg_module, "M15_ZONE_ROUTE_AGAINST_BIAS_TO_INVERSE", True)):
        return "ROUTE_TO_INVERSE", "BUY" if bias_direction == "LONG" else "SELL"
    return "BLOCK", "NONE"


def resolve_m15_zone_micro_bias(
    context: Dict[str, Any],
    setup_direction: str,
    zone_type: str,
    cfg_module=cfg,
) -> Dict[str, Any]:
    data = context.get("data") or {}
    signal = context.get("signal") or {}
    market_state = data.get("market_state") or {}
    indicators = data.get("indicators") or {}
    tick_pressure = data.get("tick_pressure") or {}
    m15_df = data.get("m15_df")

    d1 = _resolve_tf_state(market_state, "D1")
    h4 = _resolve_tf_state(market_state, "H4")
    h1 = _resolve_tf_state(market_state, "H1")
    m15_structure = _resolve_m15_structure(m15_df)
    htf_state = {"D1": d1, "H4": h4, "H1": h1, "M15": m15_structure}

    pressure_score = _safe_float(tick_pressure.get("pressure_score"), 0.0)
    pressure_bias = str(tick_pressure.get("directional_bias") or tick_pressure.get("bias") or "NEUTRAL").upper()
    favorable_long = bool(tick_pressure.get("favorable_long"))
    favorable_short = bool(tick_pressure.get("favorable_short"))
    burst_rate = _safe_float(tick_pressure.get("burst_rate"), 0.0)

    body_ratio = _safe_float(signal.get("_body_ratio"), _safe_float(indicators.get("body_ratio"), 0.0))
    bullish_reaction = bool(signal.get("_bullish_reaction"))
    bearish_reaction = bool(signal.get("_bearish_reaction"))
    zone_low = _safe_float(signal.get("_zone_low"))
    zone_high = _safe_float(signal.get("_zone_high"))
    zone_mid = _safe_float(signal.get("_zone_mid"))
    price = _current_price(data)
    atr_ratio = _safe_float(indicators.get("atr_ratio"), _safe_float((data.get("regime") or {}).get("atr_ratio"), 1.0))

    score = 0.0
    reasons: List[str] = []

    score += {"UP": 22.0, "DOWN": -22.0}.get(d1, 0.0)
    score += {"UP": 12.0, "DOWN": -12.0}.get(h4, 0.0)
    score += {"UP": 8.0, "DOWN": -8.0}.get(h1, 0.0)
    score += {"UP": 12.0, "DOWN": -12.0}.get(m15_structure, 0.0)
    reasons.append(f"HTF D1:{d1} H4:{h4} H1:{h1} M15:{m15_structure}")

    pressure_component = _clamp(pressure_score * 100.0, -30.0, 30.0)
    score += pressure_component
    if pressure_bias == "LONG":
        score += 8.0
    elif pressure_bias == "SHORT":
        score -= 8.0
    if favorable_long:
        score += 6.0
    if favorable_short:
        score -= 6.0
    reasons.append(f"pressure {pressure_bias} {pressure_score:+.3f} burst {burst_rate:.2f}")

    hold_score, hold_reason = _zone_hold_score(zone_type, price, zone_low, zone_high, zone_mid)
    score += hold_score
    reasons.append(hold_reason)

    reclaim_score, reclaim_reason = _zone_reclaim_breakdown_score(zone_type, zone_low, zone_high, zone_mid, m15_df)
    score += reclaim_score
    reasons.append(reclaim_reason)

    if bullish_reaction and not bearish_reaction:
        score += 12.0
        reasons.append("bullish candle reaction")
    elif bearish_reaction and not bullish_reaction:
        score -= 12.0
        reasons.append("bearish candle reaction")
    elif bullish_reaction and bearish_reaction:
        reasons.append("mixed candle reaction")
    else:
        reasons.append("no directional candle reaction")

    if body_ratio >= 0.85:
        if bullish_reaction and not bearish_reaction:
            score += 8.0
            reasons.append(f"strong bullish body {body_ratio:.2f}")
        elif bearish_reaction and not bullish_reaction:
            score -= 8.0
            reasons.append(f"strong bearish body {body_ratio:.2f}")
        else:
            if setup_direction == "BUY":
                score += 2.0
            elif setup_direction == "SELL":
                score -= 2.0
            reasons.append(f"strong generic body {body_ratio:.2f}")
    elif body_ratio >= 0.60:
        reasons.append(f"usable body {body_ratio:.2f}")
    else:
        reasons.append(f"weak body {body_ratio:.2f}")

    if 0.75 <= atr_ratio <= 1.4:
        score += 2.0
        reasons.append(f"healthy atr_ratio {atr_ratio:.2f}")
    elif atr_ratio >= 1.9:
        score -= 2.0 if setup_direction == "SELL" else 0.0
        reasons.append(f"high atr_ratio {atr_ratio:.2f}")

    score = _clamp(score, -100.0, 100.0)
    micro_bias_direction = _direction_from_score(score)
    micro_bias_confidence = _confidence_label(score)

    blocked_reason: Optional[str] = None
    sell_short_pressure_ok = pressure_score <= float(getattr(cfg_module, "M15_ZONE_SELL_MIN_NEGATIVE_PRESSURE_SCORE", -0.15))
    buy_long_pressure_ok = pressure_score >= abs(float(getattr(cfg_module, "M15_ZONE_SELL_MIN_NEGATIVE_PRESSURE_SCORE", -0.15)))
    sell_body_min_if_d1_up = float(getattr(cfg_module, "M15_ZONE_SELL_MIN_BODY_RATIO_IF_D1_UP", 0.85))
    buy_body_min_if_d1_down = float(getattr(cfg_module, "M15_ZONE_BUY_MIN_BODY_RATIO_IF_D1_DOWN", 0.85))

    if micro_bias_direction == "SHORT":
        strong_bear_body = bearish_reaction and body_ratio >= (sell_body_min_if_d1_up if d1 == "UP" else 0.70)
        short_structure_confirmed = m15_structure == "DOWN"
        bullish_conflict = d1 == "UP" and h4 == "DOWN"

        if bool(getattr(cfg_module, "M15_ZONE_SELL_BLOCK_IF_PRESSURE_LONG", True)) and pressure_bias == "LONG" and pressure_score > 0:
            blocked_reason = "short blocked: long pressure active"
        elif bool(getattr(cfg_module, "M15_ZONE_SELL_REQUIRE_SHORT_PRESSURE", True)) and not (
            sell_short_pressure_ok or strong_bear_body or short_structure_confirmed
        ):
            blocked_reason = "short blocked: missing bearish micro confirmation"
        elif bullish_conflict and not (sell_short_pressure_ok or strong_bear_body or short_structure_confirmed):
            blocked_reason = "short blocked: D1 up / H4 down conflict"

    if micro_bias_direction == "LONG":
        strong_bull_body = bullish_reaction and body_ratio >= (buy_body_min_if_d1_down if d1 == "DOWN" else 0.70)
        if bool(getattr(cfg_module, "M15_ZONE_BUY_BLOCK_IF_PRESSURE_SHORT", True)) and pressure_bias == "SHORT" and pressure_score < 0:
            blocked_reason = "long blocked: short pressure active"
        elif bool(getattr(cfg_module, "M15_ZONE_BUY_REQUIRE_LONG_PRESSURE", False)) and not (
            buy_long_pressure_ok or strong_bull_body
        ):
            blocked_reason = "long blocked: missing bullish micro confirmation"
        elif d1 == "DOWN" and not (buy_long_pressure_ok or strong_bull_body):
            blocked_reason = "long blocked: D1 down without strong bullish reclaim"

    if blocked_reason:
        reasons.append(blocked_reason)
        return {
            "micro_bias_direction": micro_bias_direction,
            "micro_bias_score": round(score, 2),
            "micro_bias_confidence": micro_bias_confidence,
            "recommended_action": "BLOCK",
            "final_direction": "NONE",
            "reasons": reasons,
            "pressure_bias": pressure_bias,
            "pressure_score": round(pressure_score, 3),
            "burst_rate": round(burst_rate, 2),
            "htf_state": htf_state,
        }

    recommended_action, final_direction = _final_action_from_bias(
        setup_direction=setup_direction,
        bias_direction=micro_bias_direction,
        score=score,
        confidence_label=micro_bias_confidence,
        cfg_module=cfg_module,
    )

    if recommended_action == "BLOCK":
        if micro_bias_direction == "NEUTRAL":
            reasons.append("blocked: neutral micro bias")
        elif abs(_safe_float(score, 0.0)) < max(0.0, _safe_float(getattr(cfg_module, "M15_ZONE_MICRO_BIAS_MIN_SCORE", 30.0), 30.0)):
            reasons.append("blocked: score below configured minimum")
        else:
            reasons.append("blocked: confidence below configured minimum")

    return {
        "micro_bias_direction": micro_bias_direction,
        "micro_bias_score": round(score, 2),
        "micro_bias_confidence": micro_bias_confidence,
        "recommended_action": recommended_action,
        "final_direction": final_direction,
        "reasons": reasons,
        "pressure_bias": pressure_bias,
        "pressure_score": round(pressure_score, 3),
        "burst_rate": round(burst_rate, 2),
        "htf_state": htf_state,
    }


def build_m15_zone_decision_reason(
    strategy_label: str,
    base_direction: str,
    resolver: Dict[str, Any],
    signal: Optional[Dict[str, Any]] = None,
) -> str:
    signal = signal or {}
    htf = resolver.get("htf_state") or {}
    reason_tail = "; ".join((resolver.get("reasons") or [])[:3])
    return (
        f"{strategy_label} | base={str(base_direction or '').upper()} | "
        f"micro_bias={resolver.get('micro_bias_direction', 'NEUTRAL')} "
        f"score={_safe_float(resolver.get('micro_bias_score')):+.0f} "
        f"confidence={resolver.get('micro_bias_confidence', 'LOW')} | "
        f"action={resolver.get('recommended_action', 'BLOCK')} | "
        f"final={resolver.get('final_direction', 'NONE')} | "
        f"D1:{htf.get('D1', 'RANGE')} H4:{htf.get('H4', 'RANGE')} "
        f"H1:{htf.get('H1', 'RANGE')} M15:{htf.get('M15', 'RANGE')} | "
        f"pressure={resolver.get('pressure_bias', 'NEUTRAL')} "
        f"{_safe_float(resolver.get('pressure_score')):+.2f} | "
        f"body={_safe_float(signal.get('_body_ratio')):.2f} | "
        f"{reason_tail}"
    )


def mirror_zone_signal(source: Dict[str, Any], data: Dict[str, Any], strategy_name: str) -> Dict[str, Any]:
    tick = data.get("tick") or {}
    bid = _safe_float(tick.get("bid"))
    ask = _safe_float(tick.get("ask") or tick.get("bid"))
    source_signal = str(source.get("signal") or "").upper()
    direction = "SELL" if source_signal == "BUY" else "BUY"
    entry = bid if direction == "SELL" else (ask if ask > 0 else bid)
    sl = round(_safe_float(source.get("tp")), 2)
    tp = round(_safe_float(source.get("sl")), 2)
    sl_dist = abs(entry - sl)
    tp_dist = abs(tp - entry)
    rr = tp_dist / max(sl_dist, 1e-6)
    mirrored = dict(source)
    mirrored.update(
        {
            "signal": direction,
            "entry": round(entry, 2),
            "sl": sl,
            "tp": tp,
            "tp_levels": [tp],
            "sl_distance": round(sl_dist, 2),
            "rr": round(rr, 2),
            "_strategy_name": strategy_name,
            "strategy": strategy_name,
            "decision": direction,
            "_setup_direction": "LONG" if direction == "BUY" else "SHORT",
            "_bias_direction": "LONG" if direction == "BUY" else "SHORT",
            "_source_strategy_name": "M15_ZONE_SCALP",
            "_source_signal": source_signal,
            "_source_setup_direction": "LONG" if source_signal == "BUY" else "SHORT",
            "_tp_levels": [tp],
            "_tp_pips": round(tp_dist / max(0.01, _safe_float(source.get("_pip_size"), 0.1)), 2),
        }
    )
    return mirrored
