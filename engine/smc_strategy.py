"""
SMC Confluence Strategy — Setup-First Directional Arbitration.

Direction is determined by setup signals (zone interaction, sweep, rejection),
NOT by higher-timeframe bias. Bias sets the quality threshold:
  - With-trend:    quality_score >= 0.65
  - Counter-trend:  quality_score >= 0.70-0.75 + sweep OR rejection required
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
from .decision_logger import log_smc_decision
from .anti_starvation import anti_starvation
from .signal_quality import quality_score
import config as cfg

class SMCStrategy(BaseStrategy):
    name = "SMC_CONFLUENCE"

    def __init__(self):
        self.tick_proc = TickProcessor(buffer_size=300)

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
        zones = data.get("zones", {})
        ind = data.get("indicators", {})

        price = tick["bid"]
        spread = tick.get("spread", 0)
        spread_mean = tick_snap.get("spread_mean", spread)
        spread_std = tick_snap.get("spread_std", 0)

        def skip(reason: str,
                 setup_direction: Optional[str] = None,
                 bias_direction: Optional[str] = None,
                 quality_score_value: float = 0.0,
                 threshold: float = 0.0,
                 compression_ok: bool = False,
                 ltf_conflict: bool = False,
                 setup_features: Optional[Dict] = None,
                 log_extra: Optional[Dict] = None) -> Dict:
            log_smc_decision(
                setup_direction=setup_direction,
                bias_direction=bias_direction,
                quality_score=quality_score_value,
                threshold=threshold,
                spread_mean=spread_mean,
                spread_std=spread_std,
                compression_ok=compression_ok,
                ltf_conflict=ltf_conflict,
                decision="TRADE_SKIPPED",
                reason=reason,
                price=price,
                setup_features=setup_features,
            )
            payload = {"score": quality_score_value}
            if bias_direction:
                payload["bias"] = {"direction": bias_direction}
            if log_extra:
                payload["log_extra"] = log_extra
            return _no(reason, **payload)

        if m1 is None or len(m1) < 50 or m5 is None or len(m5) < 50:
            return skip("Insufficient M1/M5 data")

        # === 1. Session filter (tightened windows) ===
        from datetime import datetime, timezone
        now = data.get("now_utc")
        if not isinstance(now, datetime):
            now = datetime.now(timezone.utc)
        elif now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        else:
            now = now.astimezone(timezone.utc)
        h = now.hour
        in_window = (
            cfg.TRADE_WINDOW_LONDON_START <= h < cfg.TRADE_WINDOW_LONDON_END
            or cfg.TRADE_WINDOW_OVERLAP_START <= h < cfg.TRADE_WINDOW_OVERLAP_END
        )
        if not in_window and not (cfg.SESSION_OVERRIDE_ENABLED or cfg.TIME_GATE_OVERRIDE_ENABLED or cfg.ALL_GATES_OVERRIDE_ENABLED):
            return skip(f"Outside trade window (UTC {h}:xx)")

        # === 2. Spread engine (execution quality) ===
        spread_ok, spread_reason = self._check_spread_quality(spread, tick_snap)
        if not spread_ok:
            return skip(f"Spread: {spread_reason}")

        # === 3. Compression gate ===
        compression_ok, comp_reason = self._check_compression_gate(m1, ind)
        if not compression_ok:
            return skip(f"Compression: {comp_reason}")

        # === 4. Liquidity analysis ===
        liq = compute_liquidity(m5, m15, h1, None)

        # === 5. SETUP-FIRST: detect direction from price action ===
        setup = self._detect_smc_setup(price, zones, liq, m1, m5, ind)
        if not setup:
            return skip("No setup detected", compression_ok=True)

        direction = setup["direction"]

        # === 6. Compute bias + regime separately ===
        bias = compute_bias(h4, h1, m15)
        regime = classify_regime(h4, h1, m15, tick_snap)
        counter_trend = bias["direction"] != "NEUTRAL" and bias["direction"] != direction
        threshold = self._resolve_threshold(bias, counter_trend, regime)

        # === 7. Counter-trend guard: require sweep OR rejection ===
        if counter_trend:
            has_sweep = setup.get("has_sweep", False)
            has_rejection = setup.get("has_rejection", False)
            if not has_sweep and not has_rejection:
                return skip(
                    f"Counter-trend {direction} vs bias {bias['direction']}: no sweep or rejection",
                    setup_direction=direction,
                    bias_direction=bias["direction"],
                    threshold=threshold,
                    compression_ok=True,
                    setup_features=setup,
                    log_extra={"setup_dir": direction, "bias_dir": bias["direction"],
                               "quality": 0, "threshold": threshold, "counter": True,
                               "pullback": False}
                )

        # === 8. LTF conflict blocker ===
        ltf_conflict = self._check_ltf_conflict(direction, bias, m1, m5, tick_snap)
        if ltf_conflict:
            return skip(
                f"LTF conflict: {direction} blocked by opposite momentum",
                setup_direction=direction,
                bias_direction=bias["direction"],
                threshold=threshold,
                compression_ok=True,
                ltf_conflict=True,
                setup_features=setup,
                log_extra={"setup_dir": direction, "bias_dir": bias["direction"],
                           "quality": 0, "threshold": threshold, "counter": counter_trend,
                           "ltf_conflict": True}
            )

        directional_guard_reason = self._check_directional_entry_guard(
            direction=direction,
            setup=setup,
            m5=m5,
            bias=bias,
        )
        if directional_guard_reason:
            return skip(
                directional_guard_reason,
                setup_direction=direction,
                bias_direction=bias["direction"],
                threshold=threshold,
                compression_ok=True,
                setup_features=setup,
                log_extra={"setup_dir": direction, "bias_dir": bias["direction"],
                           "quality": 0, "threshold": threshold, "counter": counter_trend,
                           "directional_guard": True}
            )

        # === 9. Quality score (single function) ===
        score, reasons = self._quality_score(setup, direction, m1, m5, ind, tick_snap, counter_trend, threshold)

        # === 10. Threshold check ===
        if score < threshold:
            return skip(
                f"Score {score:.0%} < {'counter' if counter_trend else 'trend'} threshold {threshold:.0%}",
                setup_direction=direction,
                bias_direction=bias["direction"],
                quality_score_value=score,
                threshold=threshold,
                compression_ok=True,
                setup_features=setup,
                log_extra={"setup_dir": direction, "bias_dir": bias["direction"],
                           "quality": score, "threshold": threshold,
                           "counter": counter_trend, "ltf_conflict": False}
            )

        # === 11. SL/TP (cost-aware) ===
        signal = "BUY" if direction == "LONG" else "SELL"
        sl, tp, sl_dist = self._compute_cost_aware_sl_tp(price, signal, ind, spread)
        if sl is None or tp is None:
            return skip("Cannot compute valid SL/TP", setup_direction=direction,
                        bias_direction=bias["direction"], quality_score_value=score,
                        threshold=threshold, compression_ok=True, setup_features=setup)

        tp_dist = abs(tp - price)
        rr = tp_dist / sl_dist if sl_dist > 0 else 0
        if rr < cfg.SMC_MIN_RR:
            return skip(f"RR {rr:.1f} < {cfg.SMC_MIN_RR}", setup_direction=direction,
                        bias_direction=bias["direction"], quality_score_value=score,
                        threshold=threshold, compression_ok=True, setup_features=setup)

        # High-confidence TP extension
        if score >= cfg.SMC_HIGH_CONF_THRESHOLD:
            extended_tp_dist = tp_dist * cfg.SMC_HIGH_CONF_RR_MULTIPLIER
            if signal == "BUY":
                tp = round(price + extended_tp_dist, 2)
            else:
                tp = round(price - extended_tp_dist, 2)
            tp_dist = extended_tp_dist
            rr = round(tp_dist / sl_dist, 2)

        # === 12. Log decision context ===
        reasons.append(f"[Q:{score:.0%} T:{threshold:.0%} {'CTR' if counter_trend else 'WTR'}]")

        signal_result = {
            "signal": signal,
            "entry": price,
            "sl": sl,
            "tp": tp,
            "sl_distance": round(sl_dist, 2),
            "confidence": score,
            "reason": " | ".join(reasons),
            "reasons": reasons,
            "rr": round(rr, 2),
            "bias": bias,
            "indicators": ind,
            "_setup_direction": direction,
            "_bias_direction": bias["direction"],
            "_counter_trend": counter_trend,
            "_quality_score": score,
            "_threshold": threshold,
            "_ltf_conflict": False,
            "_be_trigger": 0.30,
            "_timeout": 60,
            "_early_fail": self._resolve_early_fail(regime, score),
            "_high_conf": score >= cfg.SMC_HIGH_CONF_THRESHOLD,
            "_tier1_min_ticks": cfg.SMC_TIER1_MIN_TICKS,
            "_tier1_max_ticks": cfg.SMC_TIER1_MAX_TICKS,
            "_entry_spread": spread,
            "_entry_tick_velocity": tick_snap.get("velocity", 0),
        }

        # Log successful trade decision
        log_smc_decision(
            setup_direction=direction, bias_direction=bias["direction"], quality_score=score, threshold=threshold,
            spread_mean=spread_mean, spread_std=spread_std, compression_ok=True,
            ltf_conflict=False, decision="TRADE_TAKEN", 
            reason=" | ".join(reasons), price=price,
            setup_features=setup, signal_data=signal_result
        )

        return signal_result

    # ------------------------------------------------------------------
    # Setup detection — improved LONG detection with ±3 pip tolerance
    # ------------------------------------------------------------------

    def _detect_smc_setup(self, price: float, zones: Dict, liq: Dict,
                          m1: pd.DataFrame, m5: pd.DataFrame, ind: Dict) -> Optional[Dict]:
        """Detect SMC setup from price action with improved LONG detection."""
        setups: List[Dict] = []

        # Check BOTH sides for zone interaction (±3 pip tolerance)
        for zone_dir, zone_score, zone_type in self._scan_zones_expanded(price, zones, liq):
            s = {"direction": zone_dir, "zone_score": zone_score, "zone_type": zone_type,
                 "has_sweep": False, "has_rejection": False, "rejection_detail": "",
                 "has_reversal_seq": False}

            # Check sweep for this direction
            if zone_dir == "LONG" and liq.get("recent_buy_sweep"):
                s["has_sweep"] = True
            elif zone_dir == "SHORT" and liq.get("recent_sell_sweep"):
                s["has_sweep"] = True

            # Check multi-candle rejection patterns
            rej = self._check_multi_candle_rejection(m1, m5, zone_dir)
            if rej:
                s["has_rejection"] = True
                s["rejection_detail"] = rej["detail"]

            # Check 2-3 candle reversal sequence
            rev_seq = self._check_reversal_sequence(m1, m5, zone_dir)
            if rev_seq:
                s["has_reversal_seq"] = True
                s["reversal_detail"] = rev_seq["detail"]

            # Count setup conditions. LONG and SHORT both require 2+ confirmations.
            conditions = sum([
                s["zone_score"] > 0,
                s["has_sweep"],
                s["has_rejection"],
                s["has_reversal_seq"]
            ])

            if zone_dir == "LONG" and conditions >= 2:
                s["total"] = zone_score + (0.20 if s["has_sweep"] else 0) + \
                            (0.15 if s["has_rejection"] else 0) + \
                            (0.10 if s["has_reversal_seq"] else 0)
                setups.append(s)
            elif zone_dir == "SHORT" and conditions >= 2:
                s["total"] = zone_score + (0.20 if s["has_sweep"] else 0) + \
                            (0.15 if s["has_rejection"] else 0) + \
                            (0.10 if s["has_reversal_seq"] else 0)
                setups.append(s)

        if not setups:
            return None

        # Pick strongest setup
        setups.sort(key=lambda s: s["total"], reverse=True)
        return setups[0]

    def _scan_zones_expanded(self, price: float, zones: Dict, liq: Dict) -> List[Tuple[str, float, str]]:
        """Scan zones with ±3 pip tolerance for better LONG detection."""
        hits = []
        tolerance = 3.0  # ±3 pips

        # Support zones → LONG (expanded tolerance)
        for z in zones.get("support", []):
            if z["zone_low"] - tolerance <= price <= z["zone_high"] + tolerance:
                hits.append(("LONG", 0.25, f"Support [{z['zone_low']}-{z['zone_high']}]"))

        # Resistance zones → SHORT
        for z in zones.get("resistance", []):
            if z["zone_low"] - tolerance <= price <= z["zone_high"] + tolerance:
                hits.append(("SHORT", 0.25, f"Resistance [{z['zone_low']}-{z['zone_high']}]"))

        # Order blocks with expanded tolerance
        for ob in liq.get("order_blocks", []):
            if ob["type"] == "BULLISH_OB" and ob["zone_low"] - tolerance <= price <= ob["zone_high"] + tolerance:
                hits.append(("LONG", 0.25, f"Bullish OB [{ob['zone_low']}-{ob['zone_high']}]"))
            elif ob["type"] == "BEARISH_OB" and ob["zone_low"] - tolerance <= price <= ob["zone_high"] + tolerance:
                hits.append(("SHORT", 0.25, f"Bearish OB [{ob['zone_low']}-{ob['zone_high']}]"))

        # FVGs with smaller tolerance
        for fvg in liq.get("fvg", []):
            if fvg["type"] == "BULLISH_FVG" and fvg["zone_low"] - 1.5 <= price <= fvg["zone_high"] + 1.5:
                hits.append(("LONG", 0.20, f"Bullish FVG [{fvg['zone_low']}-{fvg['zone_high']}]"))
            elif fvg["type"] == "BEARISH_FVG" and fvg["zone_low"] - 1.5 <= price <= fvg["zone_high"] + 1.5:
                hits.append(("SHORT", 0.20, f"Bearish FVG [{fvg['zone_low']}-{fvg['zone_high']}]"))

        return hits

    def _check_multi_candle_rejection(self, m1: pd.DataFrame, m5: pd.DataFrame, direction: str) -> Optional[Dict]:
        """Check for multi-candle rejection patterns."""
        # Check M5 first (stronger signal)
        if m5 is not None and len(m5) >= 3:
            for i in range(-3, 0):  # Last 3 candles
                c = m5.iloc[i]
                rng = c["high"] - c["low"]
                if rng == 0:
                    continue
                    
                body = abs(c["close"] - c["open"])
                lower_wick = (min(c["open"], c["close"]) - c["low"]) / rng
                upper_wick = (c["high"] - max(c["open"], c["close"])) / rng
                body_ratio = body / rng

                # Bullish rejection patterns
                if direction == "LONG":
                    if (lower_wick > 0.55 and c["close"] > c["open"]) or \
                       (body_ratio < 0.3 and lower_wick > 0.4):  # Doji with long lower wick
                        return {"detail": f"M5 bullish rejection (candle {i})"}
                    prev = m5.iloc[i - 1] if abs(i - 1) <= len(m5) else None
                    if prev is not None and prev["close"] < prev["open"] and c["close"] > c["open"] \
                       and c["close"] >= prev["open"] and c["open"] <= prev["close"]:
                        return {"detail": f"M5 bullish engulfing (candle {i})"}
                
                # Bearish rejection patterns
                elif direction == "SHORT":
                    if (upper_wick > 0.55 and c["close"] < c["open"]) or \
                       (body_ratio < 0.3 and upper_wick > 0.4):  # Doji with long upper wick
                        return {"detail": f"M5 bearish rejection (candle {i})"}
                    prev = m5.iloc[i - 1] if abs(i - 1) <= len(m5) else None
                    if prev is not None and prev["close"] > prev["open"] and c["close"] < c["open"] \
                       and c["open"] >= prev["close"] and c["close"] <= prev["open"]:
                        return {"detail": f"M5 bearish engulfing (candle {i})"}

        # Check M1 for additional confirmation
        if m1 is not None and len(m1) >= 5:
            for i in range(-5, 0):  # Last 5 M1 candles
                c = m1.iloc[i]
                rng = c["high"] - c["low"]
                if rng == 0:
                    continue
                    
                lower_wick = (min(c["open"], c["close"]) - c["low"]) / rng
                upper_wick = (c["high"] - max(c["open"], c["close"])) / rng

                if direction == "LONG" and lower_wick > 0.6 and c["close"] > c["open"]:
                    return {"detail": f"M1 bullish rejection (candle {i})"}
                elif direction == "SHORT" and upper_wick > 0.6 and c["close"] < c["open"]:
                    return {"detail": f"M1 bearish rejection (candle {i})"}
                prev = m1.iloc[i - 1] if abs(i - 1) <= len(m1) else None
                if direction == "LONG" and prev is not None and prev["close"] < prev["open"] and \
                   c["close"] > c["open"] and c["close"] >= prev["open"] and c["open"] <= prev["close"]:
                    return {"detail": f"M1 bullish engulfing (candle {i})"}
                if direction == "SHORT" and prev is not None and prev["close"] > prev["open"] and \
                   c["close"] < c["open"] and c["open"] >= prev["close"] and c["close"] <= prev["open"]:
                    return {"detail": f"M1 bearish engulfing (candle {i})"}

        return None

    def _check_reversal_sequence(self, m1: pd.DataFrame, m5: pd.DataFrame, direction: str) -> Optional[Dict]:
        """Check for 2-3 candle reversal sequences."""
        # M5 reversal sequence (stronger)
        if m5 is not None and len(m5) >= 3:
            last_3 = m5.iloc[-3:]
            closes = last_3["close"].values
            opens = last_3["open"].values
            
            if direction == "LONG":
                # Look for: bearish -> bearish -> bullish or bearish -> doji -> bullish
                if closes[0] < opens[0] and closes[2] > opens[2] and closes[2] > closes[0]:
                    return {"detail": "M5 bullish reversal sequence"}
            elif direction == "SHORT":
                # Look for: bullish -> bullish -> bearish or bullish -> doji -> bearish
                if closes[0] > opens[0] and closes[2] < opens[2] and closes[2] < closes[0]:
                    return {"detail": "M5 bearish reversal sequence"}

        # M1 reversal sequence (weaker confirmation)
        if m1 is not None and len(m1) >= 3:
            last_3 = m1.iloc[-3:]
            closes = last_3["close"].values
            opens = last_3["open"].values
            
            if direction == "LONG":
                bearish_count = sum(1 for i in range(2) if closes[i] < opens[i])
                if bearish_count >= 2 and closes[2] > opens[2]:
                    return {"detail": "M1 bullish reversal sequence"}
            elif direction == "SHORT":
                bullish_count = sum(1 for i in range(2) if closes[i] > opens[i])
                if bullish_count >= 2 and closes[2] < opens[2]:
                    return {"detail": "M1 bearish reversal sequence"}

        return None

    # ------------------------------------------------------------------
    # Quality scoring (single function)
    # ------------------------------------------------------------------

    def _quality_score(
        self,
        setup: Dict,
        direction: str,
        m1: pd.DataFrame,
        m5: pd.DataFrame,
        ind: Dict,
        tick_snap: Dict,
        counter_trend: bool,
        threshold: float,
    ) -> Tuple[float, List[str]]:
        """Compute the requested quality score from existing signals."""
        reasons = []

        if setup.get("zone_type"):
            reasons.append(f"At {setup['zone_type']}")
        m1_aligned = self._check_timeframe_alignment(m1, direction, "M1")
        m5_aligned = self._check_timeframe_alignment(m5, direction, "M5")
        relaxed_params = anti_starvation.get_relaxed_params()
        tick_threshold = relaxed_params.get("tick_ratio_threshold", cfg.SCALPER_TICK_DIR_THRESHOLD) if relaxed_params["active"] else cfg.SCALPER_TICK_DIR_THRESHOLD
        compression_ok = (
            ind.get("range_10", 0) > 0 and
            ind.get("atr14", ind.get("atr", 2.0)) > 0 and
            ind.get("range_10", 0) < cfg.COMPRESSION_ATR_MULTIPLIER * ind.get("atr14", ind.get("atr", 2.0)) and
            ind.get("atr_slope", 0) > 0
        )
        score, score_reasons = quality_score(
            direction=direction,
            sweep_present=setup.get("has_sweep", False),
            body_ratio=ind.get("body_ratio", 0),
            tick_ratio=tick_snap.get("dir_pct", 0.5),
            tick_velocity_increasing=tick_snap.get("vel_increasing", False),
            m1_aligned=m1_aligned,
            m5_aligned=m5_aligned,
            compression_ok=compression_ok,
            tick_ratio_threshold=tick_threshold,
        )
        reasons.extend(score_reasons)

        if setup.get("has_rejection"):
            reasons.append(f"Rejection: {setup['rejection_detail']}")
        if setup.get("has_reversal_seq"):
            reasons.append(f"Reversal: {setup.get('reversal_detail', '')}")
        if counter_trend:
            reasons.append(f"Counter-trend threshold {threshold:.0%}")
        if relaxed_params["active"] and relaxed_params.get("type") == "tick_ratio":
            reasons.append(f"Anti-starvation tick ratio {tick_threshold:.0%}")

        score = round(min(1.0, score), 3)
        return score, reasons

    def _check_timeframe_alignment(self, df: pd.DataFrame, direction: str, tf: str) -> bool:
        """Check if timeframe is aligned with setup direction (price vs EMA20 + slope)."""
        if df is None or len(df) < 25:
            return False
            
        c = df["close"].values.astype(float)
        ema20 = ema(c, 20)
        price = c[-1]
        ema_val = ema20[-1]
        slope = ema20[-1] - ema20[-3] if len(ema20) >= 3 else 0

        if direction == "LONG":
            return price >= ema_val and slope > cfg.SMC_TIMEFRAME_EMA_SLOPE_MIN
        else:  # SHORT
            return price <= ema_val and slope < -cfg.SMC_TIMEFRAME_EMA_SLOPE_MIN

    def _check_directional_entry_guard(
        self,
        direction: str,
        setup: Dict,
        m5: pd.DataFrame,
        bias: Dict,
    ) -> Optional[str]:
        """Apply asymmetric guards for weaker directional setups before scoring."""
        if direction != "SHORT":
            return None

        if not (setup.get("has_sweep") or setup.get("has_rejection")):
            return "SHORT setup needs sweep or rejection"

        if not self._check_timeframe_alignment(m5, direction, "M5"):
            return "SHORT setup requires M5 alignment"

        if bias.get("direction") == "LONG" and not setup.get("has_sweep"):
            return "Counter-bias SHORT needs sweep confirmation"

        return None

    def _resolve_threshold(self, bias: Dict, counter_trend: bool, regime: Dict) -> float:
        regime_state = regime.get("state", "TRENDING")
        if regime_state == "RANGING":
            offset = cfg.SMC_RANGING_THRESHOLD_OFFSET
        elif regime_state == "TRENDING":
            offset = cfg.SMC_TRENDING_THRESHOLD_OFFSET
        else:
            offset = 0.0
        if not counter_trend:
            return round(min(1.0, cfg.SMC_THRESHOLD_WITH_TREND + offset), 2)
        bias_conf = max(0.0, min(1.0, float(bias.get("confidence", 0) or 0)))
        base = round(min(cfg.SMC_THRESHOLD_COUNTER_MAX, cfg.SMC_THRESHOLD_COUNTER + max(0.0, bias_conf - 0.5) * 0.1), 2)
        return round(min(1.0, base + offset), 2)

    def _resolve_early_fail(self, regime: Dict, score: float = 0.0) -> float:
        regime_state = regime.get("state", "TRENDING")
        if regime_state == "RANGING":
            base = cfg.SMC_EARLY_FAIL_RANGING
        elif regime_state == "TRENDING":
            base = cfg.SMC_EARLY_FAIL_TRENDING
        else:
            base = cfg.SMC_EARLY_FAIL_POINTS
        # Confidence adjustment: high-conf gets more room, low-conf gets cut faster
        if score >= cfg.SMC_HIGH_CONF_THRESHOLD:
            base = base * cfg.SMC_EARLY_FAIL_CONF_HIGH_MULT
        elif score < 0.65:
            base = base * cfg.SMC_EARLY_FAIL_CONF_LOW_MULT
        return round(base, 3)

    # ------------------------------------------------------------------
    # Spread engine (execution quality)
    # ------------------------------------------------------------------

    def _check_spread_quality(self, spread: float, tick_snap: Dict) -> Tuple[bool, str]:
        """Check spread quality using rolling window statistics with anti-starvation."""
        if cfg.ALL_GATES_OVERRIDE_ENABLED or cfg.SPREAD_GATE_OVERRIDE_ENABLED:
            return True, "Spread gate overridden"

        if not tick_snap.get("ready"):
            # Fallback to simple check with anti-starvation
            relaxed_params = anti_starvation.get_relaxed_params()
            spread_limit = cfg.SMC_SPREAD_MEAN_MAX
            if relaxed_params["active"] and relaxed_params["type"] == "spread_tolerance":
                spread_limit = min(spread_limit + relaxed_params["spread_tolerance_bonus"], cfg.SMC_SPREAD_MEAN_MAX)
            
            if spread >= spread_limit and not cfg.SPREAD_MEAN_GATE_OVERRIDE_ENABLED:
                return False, f"Spread {spread:.3f} >= {spread_limit:.3f}"
            return True, "OK"

        spread_mean = tick_snap.get("spread_mean", spread)
        spread_std = tick_snap.get("spread_std", 0)
        spread_pctl = tick_snap.get("spread_pctl", 0.5)

        # Apply anti-starvation to spread limits
        relaxed_params = anti_starvation.get_relaxed_params()
        spread_limit = cfg.SMC_SPREAD_MEAN_MAX
        if relaxed_params["active"] and relaxed_params["type"] == "spread_tolerance":
            spread_limit = min(spread_limit + relaxed_params["spread_tolerance_bonus"], cfg.SMC_SPREAD_MEAN_MAX)

        # SMC conditions with relaxation
        if spread_mean >= spread_limit and not cfg.SPREAD_MEAN_GATE_OVERRIDE_ENABLED:
            return False, f"Mean spread {spread_mean:.3f} >= {spread_limit:.3f}"
        if spread_std > cfg.SMC_SPREAD_STD_MAX and not cfg.SPREAD_VOLATILITY_GATE_OVERRIDE_ENABLED:
            return False, f"Spread volatility {spread_std:.3f} > {cfg.SMC_SPREAD_STD_MAX:.3f}"
        if spread_pctl > cfg.SMC_SPREAD_PERCENTILE_MAX and not cfg.SPREAD_PERCENTILE_GATE_OVERRIDE_ENABLED:
            return False, f"Spread percentile {spread_pctl:.0%} > {cfg.SMC_SPREAD_PERCENTILE_MAX:.0%}"
        
        # Pre-send check: current spread vs entry spread
        if spread > spread_mean + cfg.SMC_CURRENT_SPREAD_DELTA_MAX and not cfg.SPREAD_DELTA_GATE_OVERRIDE_ENABLED:
            return False, f"Current spread {spread:.3f} > entry+{cfg.SMC_CURRENT_SPREAD_DELTA_MAX:.3f}"

        return True, "Spread OK"

    # ------------------------------------------------------------------
    # Compression gate
    # ------------------------------------------------------------------

    def _check_compression_gate(self, m1: pd.DataFrame, ind: Dict) -> Tuple[bool, str]:
        """Check compression gate: RangeN < configured ATR multiple AND ATR slope rising."""
        if cfg.ALL_GATES_OVERRIDE_ENABLED or cfg.COMPRESSION_GATE_OVERRIDE_ENABLED:
            return True, "Compression gate overridden"

        lookback = max(2, int(cfg.COMPRESSION_RANGE_LOOKBACK))
        if m1 is None or len(m1) < lookback + 5:
            return False, "Insufficient M1 data for compression check"

        recent = m1.iloc[-lookback:]
        range_n = float(recent["high"].max() - recent["low"].min())
        
        atr_val = ind.get("atr14", ind.get("atr", 2.0))
        atr_threshold = cfg.COMPRESSION_ATR_MULTIPLIER * atr_val
        
        if range_n >= atr_threshold and not cfg.COMPRESSION_RANGE_GATE_OVERRIDE_ENABLED:
            return False, f"Range{lookback} {range_n:.1f} >= {atr_threshold:.1f} (too wide)"

        # Check ATR slope (rising)
        atr_slope = ind.get("atr_slope", 0)
        if atr_slope <= 0 and not cfg.ATR_RISING_GATE_OVERRIDE_ENABLED:
            return False, f"ATR slope {atr_slope:.3f} not rising"

        return True, f"Compression OK (R{lookback}:{range_n:.1f} < {atr_threshold:.1f}, ATR+)"

    # ------------------------------------------------------------------
    # LTF conflict blocker
    # ------------------------------------------------------------------

    def _check_ltf_conflict(self, direction: str, bias: Dict, m1: pd.DataFrame,
                            m5: pd.DataFrame, tick_snap: Dict) -> bool:
        """Check for LTF momentum conflict that should block trades."""
        if bias["direction"] == "NEUTRAL":
            return False

        bias_dir = bias["direction"]
        
        # Only block if trading in bias direction but LTF shows strong opposite momentum
        if direction != bias_dir:
            return False  # Counter-trend trades handled by asymmetric threshold

        m5_conflict = False
        if m5 is not None and len(m5) >= 25:
            c5 = m5["close"].values.astype(float)
            ema20_m5 = ema(c5, 20)
            price = c5[-1]
            slope = ema20_m5[-1] - ema20_m5[-3] if len(ema20_m5) >= 3 else 0
            if bias_dir == "SHORT":
                m5_conflict = price > ema20_m5[-1] and slope > 0
            else:
                m5_conflict = price < ema20_m5[-1] and slope < 0

        m1_conflict = False
        if m1 is not None and len(m1) >= 5:
            last_5 = m1.iloc[-5:]
            lows = last_5["low"].values
            highs = last_5["high"].values
            if bias_dir == "SHORT":
                m1_conflict = all(lows[i] > lows[i - 1] for i in range(1, len(lows)))
            else:
                m1_conflict = all(highs[i] < highs[i - 1] for i in range(1, len(highs)))

        tick_conflict = False
        if tick_snap.get("ready"):
            dir_pct = tick_snap.get("dir_pct", 0.5)
            if bias_dir == "SHORT":
                tick_conflict = dir_pct >= cfg.SMC_LTF_TICK_CONFLICT_SHORT_MIN
            else:
                tick_conflict = dir_pct <= cfg.SMC_LTF_TICK_CONFLICT_LONG_MAX

        return m5_conflict and m1_conflict and tick_conflict

    # ------------------------------------------------------------------
    # SL/TP (cost-aware)
    # ------------------------------------------------------------------

    def _compute_cost_aware_sl_tp(self, price: float, signal: str, ind: Dict, spread: float) -> tuple:
        """Cost-aware SL/TP calculation."""
        atr_val = ind.get("atr14", ind.get("atr", 2.0))
        if atr_val <= 0:
            atr_val = 2.0

        # SL: clamp(ATR * 0.6, 0.80, 1.50)
        sl_dist = max(0.80, min(1.50, atr_val * 0.6))
        
        # TP: max(ATR * 0.8, spread * 2.2) capped at ATR * 1.5
        tp_dist = max(atr_val * 0.8, spread * 2.2)
        tp_dist = min(tp_dist, atr_val * 1.5)
        tp_dist = max(tp_dist, spread * 2.2)  # Ensure minimum spread coverage

        if signal == "BUY":
            sl = round(price - sl_dist, 2)
            tp = round(price + tp_dist, 2)
        else:
            sl = round(price + sl_dist, 2)
            tp = round(price - tp_dist, 2)

        sl_dist = round(sl_dist, 2)

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
