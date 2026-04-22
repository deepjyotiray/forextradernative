"""
SMC Confluence Strategy — Smart Money Concepts.
Combines: regime, MTF bias, liquidity sweeps, order blocks, FVG, zones,
          tick micro-execution, rejection candles.
Only fires on high-confluence setups (score >= 0.7).
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional
from .strategies.base_strategy import BaseStrategy
from .regime import classify_regime
from .mtf_bias import compute_bias
from .liquidity import compute_liquidity
from .indicators import ema, atr, rsi, compute_indicators
from .tick_processor import TickProcessor
from .session_filter import get_session


class SMCStrategy(BaseStrategy):
    name = "SMC_CONFLUENCE"

    def __init__(self):
        self.tick_proc = TickProcessor(buffer_size=200)
        self._min_score = 0.55
        self._min_rr = 1.2

    def generate_signal(self, data: Dict) -> Dict:
        tick = data.get("tick")
        if not tick:
            return {"signal": "NO_TRADE", "reason": "No tick"}

        # Feed tick processor
        self.tick_proc.feed(tick)
        tick_snap = self.tick_proc.snapshot()

        m1 = data.get("m1_df")
        m5 = data.get("m5_df")
        m15 = data.get("m15_df")
        h1 = data.get("h1_df")
        h4 = data.get("h4_df")
        d1 = data.get("d1_df")
        zones = data.get("zones", {})
        ind = data.get("indicators", {})

        if m5 is None or len(m5) < 50:
            return {"signal": "NO_TRADE", "reason": "Insufficient M5 data"}

        price = tick["bid"]
        spread = tick.get("spread", 0)

        # === 1. Session filter — only London + NY ===
        session = get_session()
        if session == "ASIAN":
            return {"signal": "NO_TRADE", "reason": "Asian session — no trading"}

        # === 2. Market regime ===
        regime = classify_regime(h4, h1, m15, tick_snap)
        if not regime["trade_allowed"]:
            return {"signal": "NO_TRADE", "reason": f"Regime: {regime['state']}",
                    "regime": regime}

        # === 3. MTF bias ===
        bias = compute_bias(h4, h1, m15)
        if bias["direction"] == "NEUTRAL":
            return {"signal": "NO_TRADE", "reason": "No directional bias",
                    "bias": bias, "regime": regime}

        direction = bias["direction"]  # LONG or SHORT

        # === 4. Liquidity analysis ===
        liq = compute_liquidity(m5, m15, h1, d1)

        # === 5. Tick micro-execution check ===
        if tick_snap.get("ready") and self.tick_proc.is_chaotic():
            return {"signal": "NO_TRADE", "reason": "Tick chaos — news-like behavior",
                    "regime": regime, "bias": bias}

        # Spread filter — use spread model
        atr_val = ind.get("atr", 1)
        spread_ok, spread_reason = self.tick_proc.check_spread_ok(0.30, max_pctl=0.6)
        if not spread_ok:
            return {"signal": "NO_TRADE", "reason": f"Spread: {spread_reason}"}

        # === 6. Score confluence ===
        score = 0.0
        reasons = []

        # 6a. Bias alignment (0-0.25)
        bias_score = min(0.25, bias["confidence"] * 0.3)
        score += bias_score
        reasons.append(f"Bias: {direction} ({bias['confidence']:.0%})")

        # 6b. Regime alignment (0-0.15)
        if regime["state"] == "TRENDING" and regime.get("direction") == direction:
            score += 0.15
            reasons.append(f"Regime: TRENDING {direction}")
        elif regime["state"] == "EXPANSION":
            score += 0.10
            reasons.append("Regime: EXPANSION")

        # 6c. Price at key level (0-0.25)
        at_zone, zone_type = self._price_at_zone(price, direction, zones, liq)
        if at_zone:
            score += 0.25
            reasons.append(f"At {zone_type}")

        # 6d. Liquidity sweep confirmation (0-0.20)
        if direction == "LONG" and liq.get("recent_buy_sweep"):
            score += 0.20
            reasons.append("Liquidity sweep below (BUY)")
        elif direction == "SHORT" and liq.get("recent_sell_sweep"):
            score += 0.20
            reasons.append("Liquidity sweep above (SELL)")

        # 6e. Rejection candle on M5/M15 (0-0.15)
        rejection = self._check_rejection(m5, direction)
        if rejection:
            score += 0.15
            reasons.append(f"Rejection candle: {rejection}")

        # 6f. RSI confirmation (0-0.05 bonus)
        rsi_val = ind.get("rsi", 50)
        if direction == "LONG" and 30 <= rsi_val <= 50:
            score += 0.05
            reasons.append(f"RSI {rsi_val:.0f} (oversold zone)")
        elif direction == "SHORT" and 50 <= rsi_val <= 70:
            score += 0.05
            reasons.append(f"RSI {rsi_val:.0f} (overbought zone)")

        # 6g. Tick momentum alignment (0-0.05)
        if tick_snap.get("ready"):
            mom = tick_snap.get("momentum", 0)
            if (direction == "LONG" and mom > 0) or (direction == "SHORT" and mom < 0):
                score += 0.05
                reasons.append(f"Tick momentum aligned ({mom:+.3f})")

        # 6h. Correlation confirmation — DXY/US10Y (0-0.10)
        corr = data.get("correlation", {})
        corr_bias = corr.get("bias")
        if corr_bias == direction:
            corr_conf = corr.get("confidence", 0)
            bonus = min(0.10, corr_conf * 0.15)
            score += bonus
            corr_reasons = corr.get("reasons", [])
            reasons.append(f"Correlation: {', '.join(corr_reasons[:2])}" if corr_reasons
                           else f"Correlation confirms {direction}")
        elif corr_bias and corr_bias != "NEUTRAL" and corr_bias != direction:
            score -= 0.05  # penalty for conflicting correlation
            reasons.append(f"Correlation AGAINST: {corr_bias}")

        score = round(min(1.0, score), 3)

        if score < self._min_score:
            return {"signal": "NO_TRADE", "reason": f"Score {score:.0%} < {self._min_score:.0%}",
                    "score": score, "reasons": reasons, "regime": regime, "bias": bias}

        # === 7. Compute SL/TP ===
        signal = "BUY" if direction == "LONG" else "SELL"
        sl, tp, sl_dist = self._compute_sl_tp(price, signal, zones, liq, ind)
        if sl is None or tp is None:
            return {"signal": "NO_TRADE", "reason": "Cannot compute valid SL/TP"}

        # RR check
        tp_dist = abs(tp - price)
        rr = tp_dist / sl_dist if sl_dist > 0 else 0
        if rr < self._min_rr:
            return {"signal": "NO_TRADE", "reason": f"RR {rr:.1f} < {self._min_rr}"}

        return {
            "signal": signal,
            "entry": price,
            "sl": sl,
            "tp": tp,
            "sl_distance": round(sl_dist, 2),
            "confidence": score,
            "reason": " | ".join(reasons),
            "reasons": reasons,
            "rr": round(rr, 2),
            "regime": regime,
            "bias": bias,
            "indicators": ind,
        }

    def _price_at_zone(self, price: float, direction: str, zones: Dict,
                       liq: Dict) -> tuple:
        """Check if price is at a relevant zone for the direction."""
        # Check S/R zones
        if direction == "LONG":
            for z in zones.get("support", []):
                if z["zone_low"] - 1 <= price <= z["zone_high"] + 2:
                    return True, f"Support [{z['zone_low']}-{z['zone_high']}]"
        else:
            for z in zones.get("resistance", []):
                if z["zone_low"] - 2 <= price <= z["zone_high"] + 1:
                    return True, f"Resistance [{z['zone_low']}-{z['zone_high']}]"

        # Check order blocks
        for ob in liq.get("order_blocks", []):
            if direction == "LONG" and ob["type"] == "BULLISH_OB":
                if ob["zone_low"] - 1 <= price <= ob["zone_high"] + 1:
                    return True, f"Bullish OB [{ob['zone_low']}-{ob['zone_high']}]"
            if direction == "SHORT" and ob["type"] == "BEARISH_OB":
                if ob["zone_low"] - 1 <= price <= ob["zone_high"] + 1:
                    return True, f"Bearish OB [{ob['zone_low']}-{ob['zone_high']}]"

        # Check FVGs
        for fvg in liq.get("fvg", []):
            if direction == "LONG" and fvg["type"] == "BULLISH_FVG":
                if fvg["zone_low"] - 0.5 <= price <= fvg["zone_high"] + 0.5:
                    return True, f"Bullish FVG [{fvg['zone_low']}-{fvg['zone_high']}]"
            if direction == "SHORT" and fvg["type"] == "BEARISH_FVG":
                if fvg["zone_low"] - 0.5 <= price <= fvg["zone_high"] + 0.5:
                    return True, f"Bearish FVG [{fvg['zone_low']}-{fvg['zone_high']}]"

        return False, ""

    def _check_rejection(self, df: pd.DataFrame, direction: str) -> Optional[str]:
        """Check last 2 candles for rejection pattern."""
        if df is None or len(df) < 2:
            return None
        c = df.iloc[-1]
        rng = c["high"] - c["low"]
        if rng == 0:
            return None
        lower_wick = (min(c["open"], c["close"]) - c["low"]) / rng
        upper_wick = (c["high"] - max(c["open"], c["close"])) / rng

        if direction == "LONG" and lower_wick > 0.55 and c["close"] > c["open"]:
            return "Bullish rejection (long lower wick)"
        if direction == "SHORT" and upper_wick > 0.55 and c["close"] < c["open"]:
            return "Bearish rejection (long upper wick)"
        return None

    def _compute_sl_tp(self, price: float, signal: str, zones: Dict,
                       liq: Dict, ind: Dict) -> tuple:
        """Compute SL/TP. Clamped to max $5 SL, max $10 TP for XAUUSD."""
        atr_val = ind.get("atr14", ind.get("atr", 2.0))
        if atr_val <= 0:
            atr_val = 2.0

        _SL_MAX = atr_val * 3  # max SL = 3x ATR
        _TP_MAX = atr_val * 5  # max TP = 5x ATR

        if signal == "BUY":
            sl = price - atr_val * 1.5
            for z in zones.get("support", []):
                if z["zone_low"] < price and price - z["zone_low"] < _SL_MAX:
                    sl = z["zone_low"] - 0.5
                    break
            for s in liq.get("sweeps", []):
                if s["type"] == "BUY_SWEEP" and s.get("wick_low", 0) < price:
                    candidate = s["wick_low"] - 0.3
                    if price - candidate < _SL_MAX:
                        sl = min(sl, candidate)

            sl_dist = price - sl
            if sl_dist > _SL_MAX:
                sl = round(price - _SL_MAX, 2)
                sl_dist = _SL_MAX

            tp = price + sl_dist * 2.0
            for z in zones.get("resistance", []):
                if z["zone_low"] > price and z["zone_low"] - price < _TP_MAX:
                    tp = z["zone_low"] - 0.5
                    break
            levels = liq.get("key_levels", {})
            for lv in [levels.get("session_high"), levels.get("prev_day_high")]:
                if lv and lv > price + sl_dist and lv < tp:
                    tp = lv - 0.3
            if tp - price > _TP_MAX:
                tp = round(price + _TP_MAX, 2)

        else:
            sl = price + atr_val * 1.5
            for z in zones.get("resistance", []):
                if z["zone_high"] > price and z["zone_high"] - price < _SL_MAX:
                    sl = z["zone_high"] + 0.5
                    break
            for s in liq.get("sweeps", []):
                if s["type"] == "SELL_SWEEP" and s.get("wick_high", 0) > price:
                    candidate = s["wick_high"] + 0.3
                    if candidate - price < _SL_MAX:
                        sl = max(sl, candidate)

            sl_dist = sl - price
            if sl_dist > _SL_MAX:
                sl = round(price + _SL_MAX, 2)
                sl_dist = _SL_MAX

            tp = price - sl_dist * 2.0
            for z in zones.get("support", []):
                if z["zone_high"] < price and price - z["zone_high"] < _TP_MAX:
                    tp = z["zone_high"] + 0.5
                    break
            levels = liq.get("key_levels", {})
            for lv in [levels.get("session_low"), levels.get("prev_day_low")]:
                if lv and lv < price - sl_dist and lv > tp:
                    tp = lv + 0.3
            if price - tp > _TP_MAX:
                tp = round(price - _TP_MAX, 2)

        sl = round(sl, 2)
        tp = round(tp, 2)
        sl_dist = round(abs(price - sl), 2)

        if sl_dist < 0.5:
            return None, None, 0

        return sl, tp, sl_dist
