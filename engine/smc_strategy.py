"""
SMC Confluence Strategy — Setup-First Directional Arbitration.

Direction is determined by setup signals (zone interaction, sweep, rejection),
NOT by higher-timeframe bias. Bias sets the quality threshold:
  - With-trend:    quality_score >= 0.65
  - Counter-trend:  quality_score >= 0.80 + sweep OR rejection required
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from .strategies.base_strategy import BaseStrategy
from .regime import classify_regime
from .mtf_bias import compute_bias
from .liquidity import compute_liquidity
from .indicators import ema, atr, rsi, compute_indicators
from .tick_processor import TickProcessor
from .session_filter import get_session
import config as cfg

_THRESHOLD_WITH_TREND = 0.65
_THRESHOLD_COUNTER = 0.80


class SMCStrategy(BaseStrategy):
    name = "SMC_CONFLUENCE"

    def __init__(self):
        self.tick_proc = TickProcessor(buffer_size=200)
        self._min_rr = 1.2

    def generate_signal(self, data: Dict) -> Dict:
        tick = data.get("tick")
        if not tick:
            return _no("No tick")

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
            return _no("Insufficient M5 data")

        price = tick["bid"]
        spread = tick.get("spread", 0)

        # === 1. Session filter ===
        session = get_session()
        if session == "ASIAN":
            return _no("Asian session — no trading")

        # === 2. Regime ===
        regime = classify_regime(h4, h1, m15, tick_snap)
        if not regime["trade_allowed"]:
            return _no(f"Regime: {regime['state']}", regime=regime)

        # === 3. Tick chaos ===
        if tick_snap.get("ready") and self.tick_proc.is_chaotic():
            return _no("Tick chaos — news-like behavior", regime=regime)

        # === 4. Spread filter ===
        if cfg.TIER1_ENABLED:
            spread_ok, spread_reason = self.tick_proc.check_spread_ok(0.30, max_pctl=0.6)
            if not spread_ok:
                return _no(f"Spread: {spread_reason}")
        else:
            if spread > 0.50:
                return _no(f"Spread {spread:.2f} > 0.50")

        # === 5. Liquidity analysis ===
        liq = compute_liquidity(m5, m15, h1, d1)

        # === 6. SETUP-FIRST: detect direction from price action ===
        setup = self._detect_setup(price, zones, liq, m5, ind)
        if not setup:
            return _no("No setup detected", regime=regime)

        direction = setup["direction"]

        # === 7. Compute bias separately ===
        bias = compute_bias(h4, h1, m15)
        counter_trend = bias["direction"] != "NEUTRAL" and bias["direction"] != direction
        threshold = _THRESHOLD_COUNTER if counter_trend else _THRESHOLD_WITH_TREND

        # === 8. Counter-trend guard: require sweep OR rejection ===
        if counter_trend:
            has_sweep = setup.get("has_sweep", False)
            has_rejection = setup.get("has_rejection", False)
            if not has_sweep and not has_rejection:
                return _no(
                    f"Counter-trend {direction} vs bias {bias['direction']}: no sweep or rejection",
                    regime=regime, bias=bias,
                    log_extra={"setup_dir": direction, "bias_dir": bias["direction"],
                               "quality": 0, "threshold": threshold, "counter": True,
                               "pullback": False}
                )

        # === 9. Extension filter for counter-trend ===
        if counter_trend and h1 is not None and len(h1) >= 55:
            c_h1 = h1["close"].values.astype(float)
            h_h1 = h1["high"].values.astype(float)
            l_h1 = h1["low"].values.astype(float)
            ema50_h1 = ema(c_h1, 50)
            atr14_h1 = atr(h_h1, l_h1, c_h1, 14)
            dist_from_eq = abs(price - ema50_h1[-1])
            min_extension = atr14_h1[-1] * 1.5
            if dist_from_eq < min_extension:
                return _no(
                    f"Counter-trend: price too close to H1 EMA50 ({dist_from_eq:.2f} < {min_extension:.2f})",
                    regime=regime, bias=bias,
                    log_extra={"setup_dir": direction, "bias_dir": bias["direction"],
                               "quality": 0, "threshold": threshold, "counter": True,
                               "pullback": False}
                )

        # === 10. Quality score ===
        score = 0.0
        reasons = []

        # 10a. Setup components (already detected)
        score += setup.get("zone_score", 0)
        if setup.get("zone_type"):
            reasons.append(f"At {setup['zone_type']}")

        if setup.get("has_sweep"):
            score += 0.20
            sweep_dir = "below (BUY)" if direction == "LONG" else "above (SELL)"
            reasons.append(f"Liquidity sweep {sweep_dir}")

        if setup.get("has_rejection"):
            score += 0.15
            reasons.append(f"Rejection: {setup['rejection_detail']}")

        # 10b. Bias alignment (0-0.25) — with-trend gets bonus, counter gets 0
        if not counter_trend and bias["direction"] != "NEUTRAL":
            bias_score = min(0.25, bias["confidence"] * 0.3)
            score += bias_score
            reasons.append(f"Bias aligned: {bias['direction']} ({bias['confidence']:.0%})")
        elif counter_trend:
            reasons.append(f"Counter-trend: setup {direction} vs bias {bias['direction']}")

        # 10c. Regime alignment (0-0.15)
        if regime["state"] == "TRENDING" and regime.get("direction") == direction:
            score += 0.15
            reasons.append(f"Regime: TRENDING {direction}")
        elif regime["state"] == "EXPANSION":
            score += 0.10
            reasons.append("Regime: EXPANSION")

        # 10d. RSI confirmation (0-0.05)
        rsi_val = ind.get("rsi", 50)
        if direction == "LONG" and 30 <= rsi_val <= 50:
            score += 0.05
            reasons.append(f"RSI {rsi_val:.0f} (oversold zone)")
        elif direction == "SHORT" and 50 <= rsi_val <= 70:
            score += 0.05
            reasons.append(f"RSI {rsi_val:.0f} (overbought zone)")

        # 10e. Tick momentum alignment (0-0.05)
        if tick_snap.get("ready"):
            mom = tick_snap.get("momentum", 0)
            if (direction == "LONG" and mom > 0) or (direction == "SHORT" and mom < 0):
                score += 0.05
                reasons.append(f"Tick momentum aligned ({mom:+.3f})")

        # 10f. Correlation confirmation (0-0.10)
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
            score -= 0.05
            reasons.append(f"Correlation AGAINST: {corr_bias}")

        score = round(min(1.0, score), 3)

        # === 11. LTF pullback blocker ===
        pullback = False
        if not counter_trend:
            pullback = self._is_pullback_active(direction, m5, m1, ind, tick_snap)
            if pullback:
                return _no(
                    f"Pullback active against {direction} entry",
                    score=score, regime=regime, bias=bias,
                    log_extra={"setup_dir": direction, "bias_dir": bias["direction"],
                               "quality": score, "threshold": threshold, "counter": False,
                               "pullback": True}
                )

        # === 12. Threshold check ===
        if score < threshold:
            return _no(
                f"Score {score:.0%} < {'counter' if counter_trend else 'trend'} threshold {threshold:.0%}",
                score=score, reasons=reasons, regime=regime, bias=bias,
                log_extra={"setup_dir": direction, "bias_dir": bias["direction"],
                           "quality": score, "threshold": threshold,
                           "counter": counter_trend, "pullback": False}
            )

        # === 13. SL/TP ===
        signal = "BUY" if direction == "LONG" else "SELL"
        sl, tp, sl_dist = self._compute_sl_tp(price, signal, zones, liq, ind)
        if sl is None or tp is None:
            return _no("Cannot compute valid SL/TP")

        tp_dist = abs(tp - price)
        rr = tp_dist / sl_dist if sl_dist > 0 else 0
        if rr < self._min_rr:
            return _no(f"RR {rr:.1f} < {self._min_rr}")

        # === 14. Log decision context ===
        reasons.append(f"[Q:{score:.0%} T:{threshold:.0%} {'CTR' if counter_trend else 'WTR'}]")

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
            "_setup_direction": direction,
            "_bias_direction": bias["direction"],
            "_counter_trend": counter_trend,
            "_quality_score": score,
            "_threshold": threshold,
        }

    # ------------------------------------------------------------------
    # Setup detection — direction-agnostic
    # ------------------------------------------------------------------

    def _detect_setup(self, price: float, zones: Dict, liq: Dict,
                      m5: pd.DataFrame, ind: Dict) -> Optional[Dict]:
        """Detect setup from price action. Returns best setup or None."""
        setups: List[Dict] = []

        # Check BOTH sides for zone interaction
        for zone_dir, zone_score, zone_type in self._scan_zones(price, zones, liq):
            s = {"direction": zone_dir, "zone_score": zone_score, "zone_type": zone_type,
                 "has_sweep": False, "has_rejection": False, "rejection_detail": ""}

            # Check sweep for this direction
            if zone_dir == "LONG" and liq.get("recent_buy_sweep"):
                s["has_sweep"] = True
            elif zone_dir == "SHORT" and liq.get("recent_sell_sweep"):
                s["has_sweep"] = True

            # Check rejection for this direction
            rej = self._check_rejection_both(m5)
            if rej and rej["direction"] == zone_dir:
                s["has_rejection"] = True
                s["rejection_detail"] = rej["detail"]

            # Total setup strength for ranking
            s["total"] = zone_score + (0.20 if s["has_sweep"] else 0) + (0.15 if s["has_rejection"] else 0)
            setups.append(s)

        # Also check sweep-only setups (no zone required if sweep + rejection)
        for sweep_dir in ("LONG", "SHORT"):
            has_sweep = (sweep_dir == "LONG" and liq.get("recent_buy_sweep")) or \
                        (sweep_dir == "SHORT" and liq.get("recent_sell_sweep"))
            if not has_sweep:
                continue
            # Skip if we already have a zone setup in this direction
            if any(s["direction"] == sweep_dir for s in setups):
                continue
            rej = self._check_rejection_both(m5)
            if rej and rej["direction"] == sweep_dir:
                setups.append({
                    "direction": sweep_dir, "zone_score": 0, "zone_type": "",
                    "has_sweep": True, "has_rejection": True,
                    "rejection_detail": rej["detail"],
                    "total": 0.35,
                })

        if not setups:
            return None

        # Pick strongest setup
        setups.sort(key=lambda s: s["total"], reverse=True)
        return setups[0]

    def _scan_zones(self, price: float, zones: Dict, liq: Dict) -> List[Tuple[str, float, str]]:
        """Scan all zones for price interaction. Returns list of (direction, score, type)."""
        hits = []

        # Support zones → LONG
        for z in zones.get("support", []):
            if z["zone_low"] - 1 <= price <= z["zone_high"] + 2:
                hits.append(("LONG", 0.25, f"Support [{z['zone_low']}-{z['zone_high']}]"))

        # Resistance zones → SHORT
        for z in zones.get("resistance", []):
            if z["zone_low"] - 2 <= price <= z["zone_high"] + 1:
                hits.append(("SHORT", 0.25, f"Resistance [{z['zone_low']}-{z['zone_high']}]"))

        # Bullish OB → LONG
        for ob in liq.get("order_blocks", []):
            if ob["type"] == "BULLISH_OB" and ob["zone_low"] - 1 <= price <= ob["zone_high"] + 1:
                hits.append(("LONG", 0.25, f"Bullish OB [{ob['zone_low']}-{ob['zone_high']}]"))
            elif ob["type"] == "BEARISH_OB" and ob["zone_low"] - 1 <= price <= ob["zone_high"] + 1:
                hits.append(("SHORT", 0.25, f"Bearish OB [{ob['zone_low']}-{ob['zone_high']}]"))

        # FVGs
        for fvg in liq.get("fvg", []):
            if fvg["type"] == "BULLISH_FVG" and fvg["zone_low"] - 0.5 <= price <= fvg["zone_high"] + 0.5:
                hits.append(("LONG", 0.20, f"Bullish FVG [{fvg['zone_low']}-{fvg['zone_high']}]"))
            elif fvg["type"] == "BEARISH_FVG" and fvg["zone_low"] - 0.5 <= price <= fvg["zone_high"] + 0.5:
                hits.append(("SHORT", 0.20, f"Bearish FVG [{fvg['zone_low']}-{fvg['zone_high']}]"))

        return hits

    def _check_rejection_both(self, df: pd.DataFrame) -> Optional[Dict]:
        """Check last 2 candles for rejection in EITHER direction."""
        if df is None or len(df) < 2:
            return None
        c = df.iloc[-1]
        rng = c["high"] - c["low"]
        if rng == 0:
            return None
        lower_wick = (min(c["open"], c["close"]) - c["low"]) / rng
        upper_wick = (c["high"] - max(c["open"], c["close"])) / rng

        if lower_wick > 0.55 and c["close"] > c["open"]:
            return {"direction": "LONG", "detail": "Bullish rejection (long lower wick)"}
        if upper_wick > 0.55 and c["close"] < c["open"]:
            return {"direction": "SHORT", "detail": "Bearish rejection (long upper wick)"}

        # Check previous candle too
        p = df.iloc[-2]
        rng_p = p["high"] - p["low"]
        if rng_p == 0:
            return None
        lower_wick_p = (min(p["open"], p["close"]) - p["low"]) / rng_p
        upper_wick_p = (p["high"] - max(p["open"], p["close"])) / rng_p

        if lower_wick_p > 0.55 and p["close"] > p["open"]:
            return {"direction": "LONG", "detail": "Bullish rejection (prev candle)"}
        if upper_wick_p > 0.55 and p["close"] < p["open"]:
            return {"direction": "SHORT", "detail": "Bearish rejection (prev candle)"}

        return None

    # ------------------------------------------------------------------
    # LTF Pullback blocker
    # ------------------------------------------------------------------

    def _is_pullback_active(self, direction: str, m5: pd.DataFrame,
                            m1: pd.DataFrame, ind: Dict, tick_snap: Dict) -> bool:
        """Return True if short-term momentum opposes the trade direction."""
        signals_against = 0

        # M5: price vs EMA20
        if m5 is not None and len(m5) >= 20:
            c5 = m5["close"].values.astype(float)
            ema20_m5 = ema(c5, 20)
            if direction == "LONG" and c5[-1] < ema20_m5[-1]:
                signals_against += 1
            elif direction == "SHORT" and c5[-1] > ema20_m5[-1]:
                signals_against += 1

        # M1: last 3 candles sequence
        if m1 is not None and len(m1) >= 5:
            c1 = m1["close"].values.astype(float)
            o1 = m1["open"].values.astype(float)
            bearish_count = sum(1 for i in range(-3, 0) if c1[i] < o1[i])
            bullish_count = sum(1 for i in range(-3, 0) if c1[i] > o1[i])
            if direction == "LONG" and bearish_count >= 3:
                signals_against += 1
            elif direction == "SHORT" and bullish_count >= 3:
                signals_against += 1

        # Tick direction ratio
        if tick_snap.get("ready"):
            dir_pct = tick_snap.get("dir_pct", 0.5)
            mom = tick_snap.get("momentum", 0)
            if direction == "LONG" and (dir_pct < 0.35 or mom < -0.5):
                signals_against += 1
            elif direction == "SHORT" and (dir_pct > 0.65 or mom > 0.5):
                signals_against += 1

        # Need 2+ signals against to confirm active pullback
        return signals_against >= 2

    # ------------------------------------------------------------------
    # SL/TP (unchanged)
    # ------------------------------------------------------------------

    def _compute_sl_tp(self, price: float, signal: str, zones: Dict,
                       liq: Dict, ind: Dict) -> tuple:
        atr_val = ind.get("atr14", ind.get("atr", 2.0))
        if atr_val <= 0:
            atr_val = 2.0

        _SL_MAX = atr_val * 3
        _TP_MAX = atr_val * 5

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


def _no(reason: str, **kwargs) -> Dict:
    result = {"signal": "NO_TRADE", "reason": reason}
    result["score"] = kwargs.get("score", 0)
    if "reasons" in kwargs:
        result["reasons"] = kwargs["reasons"]
    if "regime" in kwargs:
        result["regime"] = kwargs["regime"]
    if "bias" in kwargs:
        result["bias"] = kwargs["bias"]
    if "log_extra" in kwargs:
        result["_arb_log"] = kwargs["log_extra"]
    return result
