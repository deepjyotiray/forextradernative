"""
Conservative M15 support/resistance rejection strategy for XAUUSD.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

import config as cfg
from .decision_logger import log_decision
from .indicators import atr, ema
from .session_filter import get_session_at
from .strategies.base_strategy import BaseStrategy
from .tick_processor import TickProcessor
from .time_utils import date_str_ist


class M15SupportResistanceStrategy(BaseStrategy):
    name = "M15_SUPPORT_RESISTANCE_REJECTION_V1"
    _POSITION_TAG = "M15SR"

    FAMILY_TREND = "M15_SR_TREND_ALIGNED_REJECTION"
    FAMILY_BOUNCE = "M15_SR_MEAN_REVERSION_BOUNCE"
    FAMILY_TRUE_REVERSAL = "M15_SR_TRUE_REVERSAL"

    def __init__(self):
        self.tick_proc = TickProcessor(buffer_size=400)
        self._daily_trade_counts: Dict[str, int] = {}
        self._last_trade_candle_by_direction: Dict[str, datetime] = {}
        self._spike_block_until: Optional[datetime] = None
        self._spike_anchor_time: Optional[datetime] = None
        self._asian_counter_bias_counts: Dict[str, int] = {}
        self._pending_signals: Dict[str, Dict] = {}
        self._pending_by_zone_side: Dict[Tuple[str, str], str] = {}

    def generate_signal(self, data: Dict) -> Dict:
        if not bool(getattr(cfg, "M15_SR_ENABLED", True)):
            return self._no("Strategy disabled")
        if bool(getattr(cfg, "M15_SR_AS_CONTEXT_ONLY", False)):
            return self._no("M15 context only mode")

        symbol = str(data.get("symbol") or cfg.SYMBOL).upper()
        if symbol != "XAUUSD":
            return self._no(f"Strategy limited to XAUUSD (got {symbol})")

        tick = data.get("tick") or {}
        if not tick:
            return self._no("No tick")
        self.tick_proc.feed(tick)
        tick_snap = self.tick_proc.snapshot()
        tick_pressure = data.get("tick_pressure") or {}

        m15 = data.get("m15_df")
        if m15 is None or len(m15) < 10:
            return self._no("Insufficient M15 data")

        frame = m15.copy().reset_index(drop=True)
        closed = self._closed_frame(frame)
        if len(closed) < max(8, int(cfg.M15_SR_LOOKBACK_HOURS) * 4):
            return self._no("Need at least 8 closed M15 candles")

        atr_m15 = self._atr_m15(closed)
        if atr_m15 <= 0:
            return self._no("Invalid M15 ATR")

        now_utc = self._resolve_now(data, tick, closed)
        current_price = self._mid_price(tick)
        last_closed = closed.iloc[-1]
        confirmation = self._candle_snapshot(last_closed)
        support_zones, resistance_zones, lookback_used = self._detect_nearby_zones(
            closed=closed,
            atr_m15=atr_m15,
            current_price=current_price,
        )
        nearest_support = self._nearest_zone(support_zones, current_price)
        nearest_resistance = self._nearest_zone(resistance_zones, current_price)
        market_ctx = self._build_market_context(data, closed, now_utc)
        candle_metrics = self.candle_metrics(last_closed)

        filter_results = {
            "spread": {"passed": True, "reason": "OK", "current": round(float(tick.get("spread", 0.0) or 0.0), 4)},
            "manipulation": {"passed": True, "reason": "OK"},
            "tick_pressure": {"passed": True, "reason": "OK"},
            "zone_quality": {"passed": True, "reason": "OK"},
            "context": {"passed": True, "reason": "OK"},
        }

        spread_ok, spread_reason = self._check_spread_filter(tick, tick_snap)
        filter_results["spread"] = {"passed": spread_ok, "reason": spread_reason}
        if not spread_ok:
            return self._skip(
                reason=spread_reason,
                price=current_price,
                atr_m15=atr_m15,
                support_zones=support_zones,
                resistance_zones=resistance_zones,
                confirmation=confirmation,
                filter_results=filter_results,
                now_utc=now_utc,
                block_category="spread",
                lookback_used=lookback_used,
                market_context=market_ctx,
                candle_metrics=candle_metrics,
            )

        spike_ok, spike_reason = self._check_spike_block(closed, atr_m15, now_utc)
        if not spike_ok:
            filter_results["manipulation"] = {"passed": False, "reason": spike_reason}
            return self._skip(
                reason=spike_reason,
                price=current_price,
                atr_m15=atr_m15,
                support_zones=support_zones,
                resistance_zones=resistance_zones,
                confirmation=confirmation,
                filter_results=filter_results,
                now_utc=now_utc,
                block_category="manipulation",
                lookback_used=lookback_used,
                market_context=market_ctx,
                candle_metrics=candle_metrics,
            )

        if int(getattr(cfg, "M15_SR_MAX_TRADES_PER_DAY", 0) or 0) > 0 and self._daily_trade_count(now_utc) >= int(cfg.M15_SR_MAX_TRADES_PER_DAY):
            return self._skip(
                reason=f"Daily trade cap reached ({cfg.M15_SR_MAX_TRADES_PER_DAY})",
                price=current_price,
                atr_m15=atr_m15,
                support_zones=support_zones,
                resistance_zones=resistance_zones,
                confirmation=confirmation,
                filter_results=filter_results,
                now_utc=now_utc,
                block_category="risk",
                lookback_used=lookback_used,
                market_context=market_ctx,
                candle_metrics=candle_metrics,
            )

        positions = data.get("positions") or []
        candidate_signals: List[Dict] = []

        if nearest_support:
            outcome = self._evaluate_directional_setup(
                direction="BUY",
                active_zone=nearest_support,
                opposite_zones=resistance_zones,
                closed=closed,
                tick=tick,
                current_price=current_price,
                atr_m15=atr_m15,
                positions=positions,
                now_utc=now_utc,
                filter_results=filter_results,
                nearest_support=nearest_support,
                nearest_resistance=nearest_resistance,
                tick_pressure=tick_pressure,
                market_context=market_ctx,
            )
            if outcome.get("signal") == "BUY":
                candidate_signals.append(outcome)
            elif outcome.get("_terminal"):
                return outcome

        if nearest_resistance:
            outcome = self._evaluate_directional_setup(
                direction="SELL",
                active_zone=nearest_resistance,
                opposite_zones=support_zones,
                closed=closed,
                tick=tick,
                current_price=current_price,
                atr_m15=atr_m15,
                positions=positions,
                now_utc=now_utc,
                filter_results=filter_results,
                nearest_support=nearest_support,
                nearest_resistance=nearest_resistance,
                tick_pressure=tick_pressure,
                market_context=market_ctx,
            )
            if outcome.get("signal") == "SELL":
                candidate_signals.append(outcome)
            elif outcome.get("_terminal"):
                return outcome

        if not candidate_signals:
            return self._skip(
                reason="No clean M15 zone rejection setup",
                price=current_price,
                atr_m15=atr_m15,
                support_zones=support_zones,
                resistance_zones=resistance_zones,
                active_zone=self._zone_payload(nearest_support or nearest_resistance),
                zone_touch_count=(nearest_support or nearest_resistance or {}).get("touches"),
                confirmation=confirmation,
                filter_results=filter_results,
                now_utc=now_utc,
                block_category="setup",
                lookback_used=lookback_used,
                market_context=market_ctx,
                candle_metrics=candle_metrics,
            )

        candidate_signals.sort(
            key=lambda item: (
                float(item.get("confidence", 0.0)),
                float(item.get("rr", 0.0)),
                1.0 if item.get("_signal_family") == self.FAMILY_TREND else 0.0,
            ),
            reverse=True,
        )
        signal = candidate_signals[0]
        self._register_pending_signal(signal)
        self._log_decision(
            decision="TRADE_TAKEN",
            reason=signal.get("reason", "M15 SR setup"),
            price=current_price,
            atr_m15=atr_m15,
            support_zones=support_zones,
            resistance_zones=resistance_zones,
            active_zone=self._zone_payload(signal.get("_active_zone")),
            zone_touch_count=(signal.get("_active_zone") or {}).get("touches"),
            entry_direction=signal.get("signal"),
            confirmation=confirmation,
            filter_results=filter_results,
            now_utc=now_utc,
            signal_data=signal,
            lookback_used=lookback_used,
            market_context=market_ctx,
            candle_metrics=candle_metrics,
        )
        return signal

    def evaluate_spike_context(self, data: Dict) -> Dict:
        m15 = data.get("m15_df")
        tick = data.get("tick") or {}
        if m15 is None or len(m15) < 10:
            return {"allowed": True, "reason": "SPIKE_PASS", "spike_detected": False}
        closed = self._closed_frame(m15.copy().reset_index(drop=True))
        atr_m15 = self._atr_m15(closed)
        now_utc = self._resolve_now(data, tick, closed)
        allowed, reason = self._check_spike_block(closed, atr_m15, now_utc)
        return {
            "allowed": allowed,
            "reason": "SPIKE_PASS" if allowed else f"SPIKE_BLOCK: {reason}",
            "spike_detected": not allowed,
            "atr_m15": round(float(atr_m15), 4) if atr_m15 else 0.0,
        }

    def evaluate_trade_context(self, data: Dict, direction: str, signal: Optional[Dict] = None) -> Dict:
        if not bool(getattr(cfg, "M15_SR_ENABLED", True)):
            return {"allowed": True, "reason": "M15_CONTEXT_PASS", "distance_atr": None, "candle_confirmation": False}

        tick = data.get("tick") or {}
        m15 = data.get("m15_df")
        if not tick or m15 is None or len(m15) < 10:
            return {
                "allowed": False,
                "reason": "M15_CONTEXT_BLOCK: insufficient M15 data",
                "distance_atr": None,
                "candle_confirmation": False,
            }

        frame = m15.copy().reset_index(drop=True)
        closed = self._closed_frame(frame)
        atr_m15 = self._atr_m15(closed)
        if atr_m15 <= 0:
            return {
                "allowed": False,
                "reason": "M15_CONTEXT_BLOCK: invalid M15 ATR",
                "distance_atr": None,
                "candle_confirmation": False,
            }

        current_price = self._mid_price(tick)
        confirmation_candle = closed.iloc[-1]
        support_zones, resistance_zones, lookback_used = self._detect_nearby_zones(
            closed=closed,
            atr_m15=atr_m15,
            current_price=current_price,
        )
        nearest_support = self._nearest_zone(support_zones, current_price)
        nearest_resistance = self._nearest_zone(resistance_zones, current_price)
        entry_direction = str(direction or "").upper()

        if entry_direction == "BUY":
            relevant_zone = nearest_support
            breakout_zone = nearest_resistance
            candle_confirmation = bool(relevant_zone and self._is_buy_rejection(confirmation_candle, relevant_zone))
            breakout_confirmation = bool(
                breakout_zone
                and float(confirmation_candle["close"]) > float(breakout_zone["zone_high"])
                and self._is_strong_bullish(confirmation_candle)
            )
        else:
            relevant_zone = nearest_resistance
            breakout_zone = nearest_support
            candle_confirmation = bool(relevant_zone and self._is_sell_rejection(confirmation_candle, relevant_zone))
            breakout_confirmation = bool(
                breakout_zone
                and float(confirmation_candle["close"]) < float(breakout_zone["zone_low"])
                and self._is_strong_bearish(confirmation_candle)
            )

        if relevant_zone:
            distance_atr = round(float(relevant_zone.get("distance", 0.0) or 0.0) / max(atr_m15, 0.0001), 3)
        else:
            distance_atr = None

        context = {
            "allowed": False,
            "reason": "M15_CONTEXT_BLOCK: direction not aligned",
            "distance_atr": distance_atr,
            "candle_confirmation": candle_confirmation or breakout_confirmation,
            "zone": self._zone_payload(relevant_zone or breakout_zone),
            "support_zone": self._zone_payload(nearest_support),
            "resistance_zone": self._zone_payload(nearest_resistance),
            "lookback_hours_used": lookback_used,
            "breakout_confirmation": breakout_confirmation,
        }

        max_distance = float(getattr(cfg, "M15_SR_MAX_DISTANCE_FROM_ZONE_ATR", 0.25))
        if relevant_zone is None and not breakout_confirmation:
            return context
        if relevant_zone and distance_atr is not None and distance_atr > max_distance and not breakout_confirmation:
            context["reason"] = "M15_CONTEXT_BLOCK: price too far from zone"
            return context
        if not (candle_confirmation or breakout_confirmation):
            return context

        context["allowed"] = True
        context["reason"] = "M15_CONTEXT_PASS"
        return context

    def confirm_trade_executed(self, signal: Dict):
        direction = str(signal.get("signal") or "").upper()
        candle_time = self._coerce_dt(signal.get("_signal_candle_time"))
        if direction in ("BUY", "SELL") and candle_time is not None:
            self._last_trade_candle_by_direction[direction] = candle_time
            day_key = date_str_ist(candle_time)
            self._daily_trade_counts[day_key] = self._daily_trade_counts.get(day_key, 0) + 1
        signal_id = str(signal.get("_signal_id") or "")
        if signal_id:
            pending = self._pending_signals.pop(signal_id, None)
            if pending:
                zone_id = str(pending.get("zone_id") or "")
                side = str(pending.get("side") or "")
                self._pending_by_zone_side.pop((zone_id, side), None)
        if str(signal.get("_signal_family") or "") == self.FAMILY_BOUNCE and str(signal.get("_session") or "") == "ASIAN":
            session_key = self._session_key(candle_time or datetime.now(timezone.utc), "ASIAN")
            self._asian_counter_bias_counts[session_key] = self._asian_counter_bias_counts.get(session_key, 0) + 1

    def pre_send_revalidate(self, signal: Dict, data: Dict) -> Dict:
        if str(signal.get("_strategy_name") or self.name) != self.name:
            return {"allowed": True, "reason": "PRE_SEND_PASS"}

        tick = data.get("tick") or {}
        m15 = data.get("m15_df")
        closed = self._closed_frame(m15.copy().reset_index(drop=True)) if m15 is not None and len(m15) >= 2 else None
        now_utc = self._resolve_now(data, tick, closed or pd.DataFrame([{"datetime": signal.get("_decision_time")}]))
        live_ctx = self._build_market_context(data, closed, now_utc) if closed is not None and len(closed) else self._build_market_context(data, pd.DataFrame(), now_utc)
        decision_time = self._coerce_dt(signal.get("_decision_time")) or now_utc
        age_seconds = max(0.0, (now_utc - decision_time).total_seconds())
        ttl_seconds = int(signal.get("_ttl_seconds", getattr(cfg, "M15_SR_SIGNAL_TTL_SECONDS", 20)) or 20)
        spread = float(tick.get("spread", 0.0) or 0.0)
        planned_entry = float(signal.get("entry", 0.0) or 0.0)
        planned_sl = float(signal.get("sl", 0.0) or 0.0)
        planned_tp = float(signal.get("tp", 0.0) or 0.0)
        atr_m15 = float(signal.get("_atr14", signal.get("indicators", {}).get("atr14_m15", 0.0)) or 0.0)
        max_spread = float(signal.get("_max_spread", getattr(cfg, "M15_SR_MAX_SPREAD", 0.45)) or 0.45)
        current_price = self._mid_price(tick)
        drift_abs = abs(current_price - planned_entry)
        drift_atr = drift_abs / max(atr_m15, 0.0001)
        recalculated_rr = self._calculate_rr(
            side=str(signal.get("signal") or ""),
            entry=current_price,
            sl=planned_sl,
            tp=planned_tp,
        )
        side = str(signal.get("signal") or "").upper()
        pressure_score = float((data.get("tick_pressure") or {}).get("pressure_score", 0.0) or 0.0)
        pressure_bias = str((data.get("tick_pressure") or {}).get("directional_bias", "NEUTRAL") or "NEUTRAL").upper()
        context_hash = str(signal.get("_context_hash") or "")
        reason = ""

        if age_seconds > ttl_seconds:
            reason = "STALE_SIGNAL"
        elif spread > max_spread:
            reason = "SPREAD_TOO_HIGH_AT_SEND"
        elif drift_atr > float(signal.get("_max_entry_drift_atr", getattr(cfg, "M15_SR_MAX_ENTRY_DRIFT_ATR", 0.10)) or 0.10):
            reason = "ENTRY_PRICE_DRIFT_TOO_LARGE"
        elif recalculated_rr < float(signal.get("_min_rr_required", 0.0) or 0.0):
            reason = "RR_DEGRADED_BEFORE_SEND"
        elif self._tick_pressure_flipped_against(side, pressure_score, pressure_bias):
            reason = "TICK_PRESSURE_FLIPPED_BEFORE_SEND"
        elif self.context_worsened(side, context_hash, live_ctx, signal=signal):
            reason = "CONTEXT_WORSENED_BEFORE_SEND"
        elif self.newer_no_signal_invalidated(self.name, str(signal.get("_zone_id") or ""), side, signal_id=str(signal.get("_signal_id") or "")):
            reason = "SIGNAL_SUPERSEDED_BY_NEWER_NO_SIGNAL"

        if reason:
            self._log_pre_send_rejection(
                signal=signal,
                age_seconds=age_seconds,
                current_price=current_price,
                planned_entry=planned_entry,
                drift_atr=drift_atr,
                spread=spread,
                recalculated_rr=recalculated_rr,
                pressure_score=pressure_score,
                pressure_bias=pressure_bias,
                live_ctx=live_ctx,
                reason=reason,
                now_utc=now_utc,
            )
            return {"allowed": False, "reason": reason}
        return {"allowed": True, "reason": "PRE_SEND_PASS"}

    def newer_no_signal_invalidated(self, strategy_name: str, zone_id: str, side: str, signal_id: str = "") -> bool:
        if strategy_name != self.name or not zone_id or not side:
            return False
        latest_id = self._pending_by_zone_side.get((zone_id, side))
        if signal_id and latest_id and latest_id != signal_id:
            latest = self._pending_signals.get(latest_id) or {}
            return bool(latest.get("invalidated"))
        pending = self._pending_signals.get(signal_id) if signal_id else None
        return bool(pending and pending.get("invalidated"))

    def context_worsened(self, side: str, context_hash: str, live_ctx: Dict, signal: Optional[Dict] = None) -> bool:
        if not signal:
            return False
        old_class = str(signal.get("_context_classification") or "")
        new_class = self.classify_buy_context(live_ctx) if side == "BUY" else self.classify_sell_context(live_ctx)
        old_rank = self._classification_rank(old_class)
        new_rank = self._classification_rank(new_class)
        if new_rank > old_rank:
            return True
        old_bias = str(signal.get("_status_bias") or "")
        new_bias = str(live_ctx.get("status_bias") or "")
        if side == "BUY" and old_bias != "SHORT" and new_bias == "SHORT" and str(live_ctx.get("h1_direction") or "") != "UP":
            return True
        if side == "SELL" and old_bias != "LONG" and new_bias == "LONG" and str(live_ctx.get("h1_direction") or "") != "DOWN":
            return True
        new_hash = self._context_hash(live_ctx)
        return bool(context_hash and new_hash != context_hash and new_rank >= old_rank and str(live_ctx.get("regime") or "") == "RANGING")

    def evaluate_rejection_setup(
        self,
        direction: str,
        zone: Dict,
        candle: pd.Series,
        tick: Dict,
        atr_m15: float,
        market_context: Dict,
        opposite_zones: List[Dict],
        closed: Optional[pd.DataFrame] = None,
        sweep_context: Optional[Dict] = None,
        now_utc: Optional[datetime] = None,
    ) -> Dict:
        direction = str(direction or "").upper()
        now_utc = now_utc or datetime.now(timezone.utc)
        tick_pressure = market_context.get("tick_pressure") or {}
        tick_score = float(tick_pressure.get("pressure_score", 0.0) or 0.0)
        tick_bias = str(tick_pressure.get("directional_bias", "NEUTRAL") or "NEUTRAL").upper()
        session = str(market_context.get("session") or "UNKNOWN").upper()
        candidate_class = self.classify_buy_context(market_context) if direction == "BUY" else self.classify_sell_context(market_context)
        signal_family_candidate = self._signal_family_for_classification(candidate_class)
        rejection_reasons: List[str] = []

        if candidate_class == "BLOCK":
            rejection_reasons.append(f"HTF_CONTEXT_BLOCKED_{direction}")
            return self._no(" | ".join(rejection_reasons))

        if candidate_class == "TREND_ALIGNED":
            if direction == "BUY":
                if not self.buy_candle_quality_normal(candle, zone, atr_m15):
                    rejection_reasons.append("WEAK_BUY_REJECTION_CANDLE")
                if tick_score < float(getattr(cfg, "M15_SR_TICK_MIN_TREND_ALIGNED", -0.10)):
                    rejection_reasons.append("TICK_PRESSURE_AGAINST_BUY")
            else:
                if not self.sell_candle_quality_normal(candle, zone, atr_m15):
                    rejection_reasons.append("WEAK_SELL_REJECTION_CANDLE")
                if tick_score > 0.10:
                    rejection_reasons.append("TICK_PRESSURE_AGAINST_SELL")

        elif candidate_class == "MEAN_REVERSION_BOUNCE_ONLY":
            min_touches = int(getattr(cfg, "M15_SR_ASIAN_COUNTER_BIAS_MIN_TOUCHES", 3) if session == "ASIAN" else 3)
            min_tick = float(
                getattr(cfg, "M15_SR_TICK_MIN_ASIAN_COUNTER_BIAS_BOUNCE", 0.35)
                if session == "ASIAN"
                else getattr(cfg, "M15_SR_TICK_MIN_COUNTER_BIAS_BOUNCE", 0.30)
            )
            if int(zone.get("touches", 0) or 0) < min_touches:
                rejection_reasons.append(
                    "COUNTER_BIAS_SUPPORT_NOT_STRONG_ENOUGH" if direction == "BUY" else "COUNTER_BIAS_RESISTANCE_NOT_STRONG_ENOUGH"
                )
            if direction == "BUY":
                if bool(getattr(cfg, "M15_SR_BLOCK_COUNTER_BIAS_BUY_IF_TICK_NEGATIVE", True)) and tick_score < 0:
                    rejection_reasons.append("COUNTER_BIAS_BUY_TICK_NEGATIVE")
                if tick_bias == "SHORT":
                    rejection_reasons.append("COUNTER_BIAS_BUY_TICK_BIAS_SHORT")
                if tick_score < min_tick or tick_bias != "LONG":
                    rejection_reasons.append("COUNTER_BIAS_BUY_NEEDS_STRONG_LONG_TICK_PRESSURE")
                if not self.buy_candle_quality_counter_bias(candle, zone, atr_m15):
                    rejection_reasons.append("COUNTER_BIAS_BUY_CANDLE_NOT_A_PLUS")
                if session == "ASIAN" and self._asian_counter_bias_trade_count(now_utc) >= int(getattr(cfg, "M15_SR_ASIAN_COUNTER_BIAS_MAX_TRADES_PER_SESSION", 1)):
                    rejection_reasons.append("ASIAN_COUNTER_BIAS_TRADE_LIMIT_REACHED")
            else:
                if bool(getattr(cfg, "M15_SR_BLOCK_COUNTER_BIAS_SELL_IF_TICK_POSITIVE", True)) and tick_score > 0:
                    rejection_reasons.append("COUNTER_BIAS_SELL_TICK_POSITIVE")
                if tick_bias == "LONG":
                    rejection_reasons.append("COUNTER_BIAS_SELL_TICK_BIAS_LONG")
                if tick_score > -min_tick or tick_bias != "SHORT":
                    rejection_reasons.append("COUNTER_BIAS_SELL_NEEDS_STRONG_SHORT_TICK_PRESSURE")
                if not self.sell_candle_quality_counter_bias(candle, zone, atr_m15):
                    rejection_reasons.append("COUNTER_BIAS_SELL_CANDLE_NOT_A_PLUS")
                if session == "ASIAN" and self._asian_counter_bias_trade_count(now_utc) >= int(getattr(cfg, "M15_SR_ASIAN_COUNTER_BIAS_MAX_TRADES_PER_SESSION", 1)):
                    rejection_reasons.append("ASIAN_COUNTER_BIAS_TRADE_LIMIT_REACHED")

        elif candidate_class == "TRUE_REVERSAL_REQUIRED":
            if direction == "BUY":
                if str(market_context.get("h1_direction") or "") == "DOWN" and bool(getattr(cfg, "M15_SR_REQUIRE_TRUE_REVERSAL_FOR_H1_OPPOSITE", True)):
                    rejection_reasons.append("TRUE_REVERSAL_H1_STILL_BEARISH")
                if not self.bullish_m15_structure_shift_confirmed(market_context):
                    rejection_reasons.append("NO_BULLISH_STRUCTURE_SHIFT")
                if tick_score < float(getattr(cfg, "M15_SR_TICK_MIN_TRUE_REVERSAL", 0.40)) or tick_bias != "LONG":
                    rejection_reasons.append("TRUE_REVERSAL_NEEDS_STRONG_LONG_TICK_PRESSURE")
                if not self.buy_candle_quality_true_reversal(candle, zone, atr_m15):
                    rejection_reasons.append("TRUE_REVERSAL_BUY_CANDLE_NOT_STRONG_ENOUGH")
                if not self.price_reclaimed_ema20_or_vwap(direction, candle, market_context):
                    rejection_reasons.append("NO_EMA20_OR_VWAP_RECLAIM")
                if not self.higher_low_confirmed(closed):
                    rejection_reasons.append("NO_HIGHER_LOW_CONFIRMED")
            else:
                if str(market_context.get("h1_direction") or "") == "UP" and bool(getattr(cfg, "M15_SR_REQUIRE_TRUE_REVERSAL_FOR_H1_OPPOSITE", True)):
                    rejection_reasons.append("TRUE_REVERSAL_H1_STILL_BULLISH")
                if not self.bearish_m15_structure_shift_confirmed(market_context):
                    rejection_reasons.append("NO_BEARISH_STRUCTURE_SHIFT")
                if tick_score > -float(getattr(cfg, "M15_SR_TICK_MIN_TRUE_REVERSAL", 0.40)) or tick_bias != "SHORT":
                    rejection_reasons.append("TRUE_REVERSAL_NEEDS_STRONG_SHORT_TICK_PRESSURE")
                if not self.sell_candle_quality_true_reversal(candle, zone, atr_m15):
                    rejection_reasons.append("TRUE_REVERSAL_SELL_CANDLE_NOT_STRONG_ENOUGH")
                if not self.price_reclaimed_ema20_or_vwap(direction, candle, market_context):
                    rejection_reasons.append("NO_EMA20_OR_VWAP_LOSS")
                if not self.lower_high_confirmed(closed):
                    rejection_reasons.append("NO_LOWER_HIGH_CONFIRMED")

        if rejection_reasons:
            return self._rejected_setup_payload(
                direction=direction,
                zone=zone,
                candle=candle,
                atr_m15=atr_m15,
                market_context=market_context,
                reason=" | ".join(rejection_reasons),
                signal_family_candidate=signal_family_candidate,
                rejection_reasons=rejection_reasons,
                intended_execution_time=now_utc,
            )

        plan = self._build_trade_plan(
            direction=direction,
            zone=zone,
            opposite_zones=opposite_zones,
            tick=tick,
            atr_m15=atr_m15,
            sweep_context=sweep_context,
            signal_family=signal_family_candidate,
            market_context=market_context,
            closed=closed,
        )
        if plan is None:
            return self._rejected_setup_payload(
                direction=direction,
                zone=zone,
                candle=candle,
                atr_m15=atr_m15,
                market_context=market_context,
                reason="Poor RR: no valid M15 SR trade plan",
                signal_family_candidate=signal_family_candidate,
                rejection_reasons=["NO_VALID_M15_SR_TRADE_PLAN"],
                intended_execution_time=now_utc,
            )

        confidence = self._confidence_score(zone, candle, plan, signal_family_candidate)
        signal_id = self._make_signal_id(direction, zone, now_utc)
        context_hash = self._context_hash(market_context)
        ttl_seconds = int(
            getattr(cfg, "M15_SR_COUNTER_BIAS_SIGNAL_TTL_SECONDS", 10)
            if signal_family_candidate == self.FAMILY_BOUNCE
            else getattr(cfg, "M15_SR_SIGNAL_TTL_SECONDS", 20)
        )
        signal = {
            "signal": direction,
            "entry": plan["entry"],
            "sl": plan["sl"],
            "tp": plan["tp"],
            "sl_distance": plan["sl_distance"],
            "rr": plan["rr"],
            "confidence": confidence,
            "reason": f"{direction} rejection from {zone['zone_id']} | touches {zone['touches']} | RR {plan['rr']:.2f}",
            "indicators": {
                "atr14_m15": round(atr_m15, 4),
                "spread": round(float(tick.get("spread", 0.0) or 0.0), 4),
            },
            "_active_zone": zone,
            "_signal_candle_time": candle["datetime"].isoformat(),
            "_comment": f"FT_{self._POSITION_TAG}_{zone['zone_id'][:8]}",
            "_setup_direction": "LONG" if direction == "BUY" else "SHORT",
            "_quality_score": confidence,
            "_threshold": 0.55,
            "_signal_family": signal_family_candidate,
            "_m15_signal_class": signal_family_candidate,
            "_sweep_confirmed": bool(plan.get("_sweep_reclaim")),
            "_candle_confirmation": True,
            "_exit_profile": self.select_exit_profile_for_signal_family(signal_family_candidate, market_context),
            "_entry_spread": float(tick.get("spread", 0.0) or 0.0),
            "_entry_tick_pressure_score": float((market_context.get("tick_pressure") or {}).get("pressure_score", 0.0) or 0.0),
            "_entry_tick_pressure_bias": str((market_context.get("tick_pressure") or {}).get("directional_bias", "NEUTRAL") or "NEUTRAL"),
            "_be_trigger": 0.30,
            "_be_trigger_r": self._breakeven_trigger_r(signal_family_candidate),
            "_timeout": self._timeout_seconds(signal_family_candidate),
            "_timeout_min_progress_r": self._timeout_min_progress(signal_family_candidate),
            "_min_hold_seconds": self._min_hold_seconds(signal_family_candidate),
            "_early_fail": max(0.12, round(atr_m15 * (0.15 if signal_family_candidate == self.FAMILY_BOUNCE else 0.20), 3)),
            "_tier1_min_ticks": cfg.SMC_TIER1_MIN_TICKS,
            "_tier1_max_ticks": cfg.SMC_TIER1_MAX_TICKS,
            "_reversal_arm_r": self._reversal_arm_r(signal_family_candidate),
            "_reversal_drawdown_pct": self._reversal_drawdown_pct(signal_family_candidate),
            "_reversal_floor_r": self._reversal_floor_r(signal_family_candidate),
            "_trail_activate_r": self._trail_activate_r(signal_family_candidate),
            "_trail_lock_r": self._trail_lock_r(signal_family_candidate),
            "_velocity_drop_enabled": self._velocity_drop_enabled(signal_family_candidate),
            "_signal_id": signal_id,
            "_zone_id": zone["zone_id"],
            "_decision_time": now_utc.isoformat(),
            "_decision_bar_time": candle["datetime"].isoformat(),
            "_planned_entry": plan["entry"],
            "_planned_sl": plan["sl"],
            "_planned_tp": plan["tp"],
            "_atr14": round(atr_m15, 4),
            "_context_hash": context_hash,
            "_ttl_seconds": ttl_seconds,
            "_max_entry_drift_atr": float(getattr(cfg, "M15_SR_MAX_ENTRY_DRIFT_ATR", 0.10) or 0.10),
            "_min_rr_required": float(plan.get("_min_rr_required", 0.0) or 0.0),
            "_max_spread": float(getattr(cfg, "M15_SR_MAX_SPREAD", 0.45) or 0.45),
            "_context_classification": candidate_class,
            "_status_bias": market_context.get("status_bias"),
            "_session": session,
            "_regime": market_context.get("regime"),
            "_htf_macro_support": market_context.get("macro_support_status", "UNKNOWN"),
        }
        signal.update(plan)
        return signal

    def _evaluate_directional_setup(
        self,
        direction: str,
        active_zone: Dict,
        opposite_zones: List[Dict],
        closed: pd.DataFrame,
        tick: Dict,
        current_price: float,
        atr_m15: float,
        positions: List[Dict],
        now_utc: datetime,
        filter_results: Dict,
        nearest_support: Optional[Dict],
        nearest_resistance: Optional[Dict],
        tick_pressure: Optional[Dict] = None,
        market_context: Optional[Dict] = None,
    ) -> Dict:
        confirmation_candle = closed.iloc[-1]
        confirmation = self._candle_snapshot(confirmation_candle)
        candle_metrics = self.candle_metrics(confirmation_candle)
        market_context = dict(market_context or {})
        market_context["tick_pressure"] = tick_pressure or {}
        block_reason = self._zone_quality_issue(active_zone)
        if block_reason:
            filter_results["zone_quality"] = {"passed": False, "reason": block_reason}
            return self._skip(
                reason=block_reason,
                price=current_price,
                atr_m15=atr_m15,
                support_zones=[nearest_support] if nearest_support else [],
                resistance_zones=[nearest_resistance] if nearest_resistance else [],
                active_zone=self._zone_payload(active_zone),
                zone_touch_count=active_zone.get("touches"),
                entry_direction=direction,
                confirmation=confirmation,
                filter_results=filter_results,
                now_utc=now_utc,
                block_category="manipulation",
                lookback_used=active_zone.get("lookback_hours"),
                terminal=True,
                market_context=market_context,
                candle_metrics=candle_metrics,
                signal_family_candidate=self._signal_family_for_classification(
                    self.classify_buy_context(market_context) if direction == "BUY" else self.classify_sell_context(market_context)
                ),
                zone_id=str(active_zone.get("zone_id") or ""),
                side=direction,
                invalidate_pending=True,
            )

        if self._has_open_direction_position(positions, direction):
            return self._skip(
                reason=f"{direction} position already open for {self.name}",
                price=current_price,
                atr_m15=atr_m15,
                support_zones=[nearest_support] if nearest_support else [],
                resistance_zones=[nearest_resistance] if nearest_resistance else [],
                active_zone=self._zone_payload(active_zone),
                zone_touch_count=active_zone.get("touches"),
                entry_direction=direction,
                confirmation=confirmation,
                filter_results=filter_results,
                now_utc=now_utc,
                block_category="risk",
                lookback_used=active_zone.get("lookback_hours"),
                terminal=True,
                market_context=market_context,
                candle_metrics=candle_metrics,
            )

        if self._has_open_zone_position(positions, active_zone):
            return self._skip(
                reason=f"Active {direction} zone already traded ({active_zone['zone_id']})",
                price=current_price,
                atr_m15=atr_m15,
                support_zones=[nearest_support] if nearest_support else [],
                resistance_zones=[nearest_resistance] if nearest_resistance else [],
                active_zone=self._zone_payload(active_zone),
                zone_touch_count=active_zone.get("touches"),
                entry_direction=direction,
                confirmation=confirmation,
                filter_results=filter_results,
                now_utc=now_utc,
                block_category="risk",
                lookback_used=active_zone.get("lookback_hours"),
                terminal=True,
                market_context=market_context,
                candle_metrics=candle_metrics,
            )

        cooldown_reason = self._cooldown_reason(direction, confirmation_candle["datetime"])
        if cooldown_reason:
            return self._skip(
                reason=cooldown_reason,
                price=current_price,
                atr_m15=atr_m15,
                support_zones=[nearest_support] if nearest_support else [],
                resistance_zones=[nearest_resistance] if nearest_resistance else [],
                active_zone=self._zone_payload(active_zone),
                zone_touch_count=active_zone.get("touches"),
                entry_direction=direction,
                confirmation=confirmation,
                filter_results=filter_results,
                now_utc=now_utc,
                block_category="cooldown",
                lookback_used=active_zone.get("lookback_hours"),
                terminal=True,
                market_context=market_context,
                candle_metrics=candle_metrics,
            )

        if self._is_choppy_market(closed, nearest_support, nearest_resistance):
            filter_results["manipulation"] = {
                "passed": False,
                "reason": "Choppy market: both nearest support and resistance touched in last 3 candles",
            }
            return self._skip(
                reason=filter_results["manipulation"]["reason"],
                price=current_price,
                atr_m15=atr_m15,
                support_zones=[nearest_support] if nearest_support else [],
                resistance_zones=[nearest_resistance] if nearest_resistance else [],
                active_zone=self._zone_payload(active_zone),
                zone_touch_count=active_zone.get("touches"),
                entry_direction=direction,
                confirmation=confirmation,
                filter_results=filter_results,
                now_utc=now_utc,
                block_category="manipulation",
                lookback_used=active_zone.get("lookback_hours"),
                terminal=True,
                market_context=market_context,
                candle_metrics=candle_metrics,
                zone_id=str(active_zone.get("zone_id") or ""),
                side=direction,
                invalidate_pending=True,
            )

        sweep_context = self._recent_sweep_context(closed, active_zone, atr_m15, direction)

        if direction == "BUY":
            if not self._price_near_zone(current_price, active_zone):
                return self._skip(
                    reason="Support zone not near price",
                    price=current_price,
                    atr_m15=atr_m15,
                    support_zones=[nearest_support] if nearest_support else [],
                    resistance_zones=[nearest_resistance] if nearest_resistance else [],
                    active_zone=self._zone_payload(active_zone),
                    zone_touch_count=active_zone.get("touches"),
                    entry_direction=direction,
                    confirmation=confirmation,
                    filter_results=filter_results,
                    now_utc=now_utc,
                    block_category="setup",
                    lookback_used=active_zone.get("lookback_hours"),
                    terminal=True,
                    market_context=market_context,
                    candle_metrics=candle_metrics,
                    zone_id=str(active_zone.get("zone_id") or ""),
                    side=direction,
                    invalidate_pending=True,
                )
            if self._recent_close_break(active_zone, closed, atr_m15, below=True):
                return self._skip(
                    reason="Support recently broken",
                    price=current_price,
                    atr_m15=atr_m15,
                    support_zones=[nearest_support] if nearest_support else [],
                    resistance_zones=[nearest_resistance] if nearest_resistance else [],
                    active_zone=self._zone_payload(active_zone),
                    zone_touch_count=active_zone.get("touches"),
                    entry_direction=direction,
                    confirmation=confirmation,
                    filter_results=filter_results,
                    now_utc=now_utc,
                    block_category="setup",
                    lookback_used=active_zone.get("lookback_hours"),
                    terminal=True,
                    market_context=market_context,
                    candle_metrics=candle_metrics,
                    zone_id=str(active_zone.get("zone_id") or ""),
                    side=direction,
                    invalidate_pending=True,
                )
            if self._strong_momentum_against_trade(closed, "BUY"):
                return self._skip(
                    reason="Momentum against BUY: last 2 M15 closes strongly bearish near lows",
                    price=current_price,
                    atr_m15=atr_m15,
                    support_zones=[nearest_support] if nearest_support else [],
                    resistance_zones=[nearest_resistance] if nearest_resistance else [],
                    active_zone=self._zone_payload(active_zone),
                    zone_touch_count=active_zone.get("touches"),
                    entry_direction=direction,
                    confirmation=confirmation,
                    filter_results={**filter_results, "manipulation": {"passed": False, "reason": "Momentum against BUY"}},
                    now_utc=now_utc,
                    block_category="manipulation",
                    lookback_used=active_zone.get("lookback_hours"),
                    terminal=True,
                    market_context=market_context,
                    candle_metrics=candle_metrics,
                    zone_id=str(active_zone.get("zone_id") or ""),
                    side=direction,
                    invalidate_pending=True,
                )
            if self._is_fake_breakout_candle(confirmation_candle, active_zone, atr_m15, "BUY"):
                return self._skip(
                    reason="Awaiting one extra candle after support sweep/fake breakout",
                    price=current_price,
                    atr_m15=atr_m15,
                    support_zones=[nearest_support] if nearest_support else [],
                    resistance_zones=[nearest_resistance] if nearest_resistance else [],
                    active_zone=self._zone_payload(active_zone),
                    zone_touch_count=active_zone.get("touches"),
                    entry_direction=direction,
                    confirmation=confirmation,
                    filter_results={**filter_results, "manipulation": {"passed": False, "reason": "Awaiting extra confirmation after sweep"}},
                    now_utc=now_utc,
                    block_category="manipulation",
                    lookback_used=active_zone.get("lookback_hours"),
                    terminal=True,
                    market_context=market_context,
                    candle_metrics=candle_metrics,
                    zone_id=str(active_zone.get("zone_id") or ""),
                    side=direction,
                    invalidate_pending=True,
                )
            if sweep_context and not sweep_context.get("valid_overshoot", False):
                return self._skip(
                    reason="Support sweep was too deep to treat as a clean reclaim",
                    price=current_price,
                    atr_m15=atr_m15,
                    support_zones=[nearest_support] if nearest_support else [],
                    resistance_zones=[nearest_resistance] if nearest_resistance else [],
                    active_zone=self._zone_payload(active_zone),
                    zone_touch_count=active_zone.get("touches"),
                    entry_direction=direction,
                    confirmation=confirmation,
                    filter_results=filter_results,
                    now_utc=now_utc,
                    block_category="manipulation",
                    lookback_used=active_zone.get("lookback_hours"),
                    terminal=True,
                    market_context=market_context,
                    candle_metrics=candle_metrics,
                    zone_id=str(active_zone.get("zone_id") or ""),
                    side=direction,
                    invalidate_pending=True,
                )
            if sweep_context and not sweep_context.get("reclaimed", False):
                return self._skip(
                    reason="Support sweep did not reclaim the zone",
                    price=current_price,
                    atr_m15=atr_m15,
                    support_zones=[nearest_support] if nearest_support else [],
                    resistance_zones=[nearest_resistance] if nearest_resistance else [],
                    active_zone=self._zone_payload(active_zone),
                    zone_touch_count=active_zone.get("touches"),
                    entry_direction=direction,
                    confirmation=confirmation,
                    filter_results=filter_results,
                    now_utc=now_utc,
                    block_category="manipulation",
                    lookback_used=active_zone.get("lookback_hours"),
                    terminal=True,
                    market_context=market_context,
                    candle_metrics=candle_metrics,
                    zone_id=str(active_zone.get("zone_id") or ""),
                    side=direction,
                    invalidate_pending=True,
                )
            if sweep_context and not self._is_buy_rejection(confirmation_candle, active_zone):
                sweep_reason = "Awaiting strong bullish reclaim after support sweep" if int(sweep_context.get("bars_ago", 0)) == 0 else "Post-sweep candle did not confirm BUY"
                return self._skip(
                    reason=sweep_reason,
                    price=current_price,
                    atr_m15=atr_m15,
                    support_zones=[nearest_support] if nearest_support else [],
                    resistance_zones=[nearest_resistance] if nearest_resistance else [],
                    active_zone=self._zone_payload(active_zone),
                    zone_touch_count=active_zone.get("touches"),
                    entry_direction=direction,
                    confirmation=confirmation,
                    filter_results=filter_results,
                    now_utc=now_utc,
                    block_category="manipulation",
                    lookback_used=active_zone.get("lookback_hours"),
                    terminal=True,
                    market_context=market_context,
                    candle_metrics=candle_metrics,
                    zone_id=str(active_zone.get("zone_id") or ""),
                    side=direction,
                    invalidate_pending=True,
                )
        else:
            if not self._price_near_zone(current_price, active_zone):
                return self._skip(
                    reason="Resistance zone not near price",
                    price=current_price,
                    atr_m15=atr_m15,
                    support_zones=[nearest_support] if nearest_support else [],
                    resistance_zones=[nearest_resistance] if nearest_resistance else [],
                    active_zone=self._zone_payload(active_zone),
                    zone_touch_count=active_zone.get("touches"),
                    entry_direction=direction,
                    confirmation=confirmation,
                    filter_results=filter_results,
                    now_utc=now_utc,
                    block_category="setup",
                    lookback_used=active_zone.get("lookback_hours"),
                    terminal=True,
                    market_context=market_context,
                    candle_metrics=candle_metrics,
                    zone_id=str(active_zone.get("zone_id") or ""),
                    side=direction,
                    invalidate_pending=True,
                )
            if self._recent_close_break(active_zone, closed, atr_m15, below=False):
                return self._skip(
                    reason="Resistance recently broken",
                    price=current_price,
                    atr_m15=atr_m15,
                    support_zones=[nearest_support] if nearest_support else [],
                    resistance_zones=[nearest_resistance] if nearest_resistance else [],
                    active_zone=self._zone_payload(active_zone),
                    zone_touch_count=active_zone.get("touches"),
                    entry_direction=direction,
                    confirmation=confirmation,
                    filter_results=filter_results,
                    now_utc=now_utc,
                    block_category="setup",
                    lookback_used=active_zone.get("lookback_hours"),
                    terminal=True,
                    market_context=market_context,
                    candle_metrics=candle_metrics,
                    zone_id=str(active_zone.get("zone_id") or ""),
                    side=direction,
                    invalidate_pending=True,
                )
            if self._strong_momentum_against_trade(closed, "SELL"):
                return self._skip(
                    reason="Momentum against SELL: last 2 M15 closes strongly bullish near highs",
                    price=current_price,
                    atr_m15=atr_m15,
                    support_zones=[nearest_support] if nearest_support else [],
                    resistance_zones=[nearest_resistance] if nearest_resistance else [],
                    active_zone=self._zone_payload(active_zone),
                    zone_touch_count=active_zone.get("touches"),
                    entry_direction=direction,
                    confirmation=confirmation,
                    filter_results={**filter_results, "manipulation": {"passed": False, "reason": "Momentum against SELL"}},
                    now_utc=now_utc,
                    block_category="manipulation",
                    lookback_used=active_zone.get("lookback_hours"),
                    terminal=True,
                    market_context=market_context,
                    candle_metrics=candle_metrics,
                    zone_id=str(active_zone.get("zone_id") or ""),
                    side=direction,
                    invalidate_pending=True,
                )
            if self._is_fake_breakout_candle(confirmation_candle, active_zone, atr_m15, "SELL"):
                return self._skip(
                    reason="Awaiting one extra candle after resistance sweep/fake breakout",
                    price=current_price,
                    atr_m15=atr_m15,
                    support_zones=[nearest_support] if nearest_support else [],
                    resistance_zones=[nearest_resistance] if nearest_resistance else [],
                    active_zone=self._zone_payload(active_zone),
                    zone_touch_count=active_zone.get("touches"),
                    entry_direction=direction,
                    confirmation=confirmation,
                    filter_results={**filter_results, "manipulation": {"passed": False, "reason": "Awaiting extra confirmation after sweep"}},
                    now_utc=now_utc,
                    block_category="manipulation",
                    lookback_used=active_zone.get("lookback_hours"),
                    terminal=True,
                    market_context=market_context,
                    candle_metrics=candle_metrics,
                    zone_id=str(active_zone.get("zone_id") or ""),
                    side=direction,
                    invalidate_pending=True,
                )
            if sweep_context and not sweep_context.get("valid_overshoot", False):
                return self._skip(
                    reason="Resistance sweep was too deep to treat as a clean reclaim",
                    price=current_price,
                    atr_m15=atr_m15,
                    support_zones=[nearest_support] if nearest_support else [],
                    resistance_zones=[nearest_resistance] if nearest_resistance else [],
                    active_zone=self._zone_payload(active_zone),
                    zone_touch_count=active_zone.get("touches"),
                    entry_direction=direction,
                    confirmation=confirmation,
                    filter_results=filter_results,
                    now_utc=now_utc,
                    block_category="manipulation",
                    lookback_used=active_zone.get("lookback_hours"),
                    terminal=True,
                    market_context=market_context,
                    candle_metrics=candle_metrics,
                    zone_id=str(active_zone.get("zone_id") or ""),
                    side=direction,
                    invalidate_pending=True,
                )
            if sweep_context and not sweep_context.get("reclaimed", False):
                return self._skip(
                    reason="Resistance sweep did not reclaim the zone",
                    price=current_price,
                    atr_m15=atr_m15,
                    support_zones=[nearest_support] if nearest_support else [],
                    resistance_zones=[nearest_resistance] if nearest_resistance else [],
                    active_zone=self._zone_payload(active_zone),
                    zone_touch_count=active_zone.get("touches"),
                    entry_direction=direction,
                    confirmation=confirmation,
                    filter_results=filter_results,
                    now_utc=now_utc,
                    block_category="manipulation",
                    lookback_used=active_zone.get("lookback_hours"),
                    terminal=True,
                    market_context=market_context,
                    candle_metrics=candle_metrics,
                    zone_id=str(active_zone.get("zone_id") or ""),
                    side=direction,
                    invalidate_pending=True,
                )
            if sweep_context and not self._is_sell_rejection(confirmation_candle, active_zone):
                sweep_reason = "Awaiting strong bearish reclaim after resistance sweep" if int(sweep_context.get("bars_ago", 0)) == 0 else "Post-sweep candle did not confirm SELL"
                return self._skip(
                    reason=sweep_reason,
                    price=current_price,
                    atr_m15=atr_m15,
                    support_zones=[nearest_support] if nearest_support else [],
                    resistance_zones=[nearest_resistance] if nearest_resistance else [],
                    active_zone=self._zone_payload(active_zone),
                    zone_touch_count=active_zone.get("touches"),
                    entry_direction=direction,
                    confirmation=confirmation,
                    filter_results=filter_results,
                    now_utc=now_utc,
                    block_category="manipulation",
                    lookback_used=active_zone.get("lookback_hours"),
                    terminal=True,
                    market_context=market_context,
                    candle_metrics=candle_metrics,
                    zone_id=str(active_zone.get("zone_id") or ""),
                    side=direction,
                    invalidate_pending=True,
                )

        outcome = self.evaluate_rejection_setup(
            direction=direction,
            zone=active_zone,
            candle=confirmation_candle,
            tick=tick,
            atr_m15=atr_m15,
            market_context=market_context,
            opposite_zones=opposite_zones,
            closed=closed,
            sweep_context=sweep_context,
            now_utc=now_utc,
        )
        if outcome.get("signal") in {"BUY", "SELL"}:
            return outcome
        return self._skip(
            reason=str(outcome.get("reason") or "No valid M15 SR trade plan"),
            price=current_price,
            atr_m15=atr_m15,
            support_zones=[nearest_support] if nearest_support else [],
            resistance_zones=[nearest_resistance] if nearest_resistance else [],
            active_zone=self._zone_payload(active_zone),
            zone_touch_count=active_zone.get("touches"),
            entry_direction=direction,
            confirmation=confirmation,
            filter_results=filter_results,
            now_utc=now_utc,
            block_category="setup",
            lookback_used=active_zone.get("lookback_hours"),
            terminal=True,
            market_context=market_context,
            candle_metrics=candle_metrics,
            signal_family_candidate=str(outcome.get("_signal_family_candidate") or ""),
            rejection_reasons=outcome.get("_rejection_reasons"),
            zone_id=str(active_zone.get("zone_id") or ""),
            side=direction,
            invalidate_pending=True,
            intended_execution_time=now_utc,
        )

    def _detect_nearby_zones(
        self,
        closed: pd.DataFrame,
        atr_m15: float,
        current_price: float,
    ) -> Tuple[List[Dict], List[Dict], int]:
        primary = max(8, int(cfg.M15_SR_LOOKBACK_HOURS) * 4)
        fallback = max(primary, int(cfg.M15_SR_FALLBACK_LOOKBACK_HOURS) * 4)
        extended = max(fallback, int(cfg.M15_SR_EXTENDED_LOOKBACK_HOURS) * 4)
        for bars, hours in (
            (primary, cfg.M15_SR_LOOKBACK_HOURS),
            (fallback, cfg.M15_SR_FALLBACK_LOOKBACK_HOURS),
            (extended, cfg.M15_SR_EXTENDED_LOOKBACK_HOURS),
        ):
            window = closed.iloc[-bars:].reset_index(drop=True)
            support = self._build_zones(window, atr_m15, "support", hours, current_price)
            resistance = self._build_zones(window, atr_m15, "resistance", hours, current_price)
            if support or resistance:
                return support, resistance, int(hours)
        return [], [], int(cfg.M15_SR_EXTENDED_LOOKBACK_HOURS)

    def _build_zones(
        self,
        window: pd.DataFrame,
        atr_m15: float,
        zone_type: str,
        lookback_hours: int,
        current_price: float,
    ) -> List[Dict]:
        if len(window) < 4:
            return []
        swing_points = self._swing_points(window, zone_type)
        if not swing_points:
            return []
        cluster_distance = max(0.05, atr_m15 * float(cfg.M15_SR_ZONE_ATR_MULT))
        clusters: List[List[Dict]] = []
        for point in sorted(swing_points, key=lambda item: item["price"]):
            placed = False
            for cluster in clusters:
                cluster_mid = sum(p["price"] for p in cluster) / len(cluster)
                if abs(point["price"] - cluster_mid) <= cluster_distance:
                    cluster.append(point)
                    placed = True
                    break
            if not placed:
                clusters.append([point])

        half_width = cluster_distance / 2.0
        clear_break = atr_m15 * float(cfg.M15_SR_ENTRY_BUFFER_ATR)
        max_distance = atr_m15 * float(cfg.M15_SR_MAX_DISTANCE_FROM_ZONE_ATR)
        recent_closes = window["close"].tail(5).tolist()
        zones: List[Dict] = []
        for cluster in clusters:
            unique_indices = sorted({int(p["idx"]) for p in cluster})
            touch_count = len(unique_indices)
            if touch_count < int(cfg.M15_SR_MIN_TOUCHES):
                continue
            prices = [float(p["price"]) for p in cluster]
            zone_mid = sum(prices) / len(prices)
            zone_low = round(zone_mid - half_width, 2)
            zone_high = round(zone_mid + half_width, 2)
            source_ranges = [float(window.iloc[idx]["high"] - window.iloc[idx]["low"]) for idx in unique_indices]
            single_extreme = len(unique_indices) <= 1
            spike_based = max(source_ranges or [0.0]) > atr_m15 * float(cfg.M15_SR_SPIKE_RANGE_ATR_MULT)
            random_cross = self._has_random_closes(recent_closes, zone_low, zone_high, zone_type)
            invalidated = (
                any(float(c) < (zone_low - clear_break) for c in recent_closes[-3:])
                if zone_type == "support"
                else any(float(c) > (zone_high + clear_break) for c in recent_closes[-3:])
            )
            distance = self._zone_distance(current_price, zone_low, zone_high)
            if invalidated or distance > max_distance:
                continue
            zones.append(
                {
                    "zone_id": f"{zone_type.upper()}_{lookback_hours}H_{round(zone_mid, 2):.2f}",
                    "type": zone_type,
                    "zone_low": zone_low,
                    "zone_high": zone_high,
                    "zone_mid": round(zone_mid, 2),
                    "touches": touch_count,
                    "touch_indices": unique_indices,
                    "distance": round(distance, 4),
                    "max_distance_abs": round(max_distance, 4),
                    "lookback_hours": int(lookback_hours),
                    "single_extreme": single_extreme,
                    "spike_based": spike_based,
                    "random_cross": random_cross,
                }
            )
        zones.sort(key=lambda item: (item["distance"], -item["touches"]))
        return zones

    def _swing_points(self, df: pd.DataFrame, zone_type: str) -> List[Dict]:
        result: List[Dict] = []
        for idx in range(1, len(df) - 1):
            row = df.iloc[idx]
            prev_row = df.iloc[idx - 1]
            next_row = df.iloc[idx + 1]
            if zone_type == "support":
                if float(row["low"]) < float(prev_row["low"]) and float(row["low"]) < float(next_row["low"]):
                    result.append({"idx": idx, "price": float(row["low"])})
            else:
                if float(row["high"]) > float(prev_row["high"]) and float(row["high"]) > float(next_row["high"]):
                    result.append({"idx": idx, "price": float(row["high"])})
        return result

    def _build_trade_plan(
        self,
        direction: str,
        zone: Dict,
        opposite_zones: List[Dict],
        tick: Dict,
        atr_m15: float,
        sweep_context: Optional[Dict] = None,
        signal_family: str = "",
        market_context: Optional[Dict] = None,
        closed: Optional[pd.DataFrame] = None,
    ) -> Optional[Dict]:
        entry = round(float(tick["ask"] if direction == "BUY" else tick["bid"]), 2)
        if direction == "BUY":
            sl_anchor = float(zone["zone_low"])
            if sweep_context and sweep_context.get("direction") == "BUY":
                sl_anchor = min(sl_anchor, float(sweep_context.get("extreme", sl_anchor)))
            sl = round(sl_anchor - atr_m15 * float(cfg.M15_SR_SL_BUFFER_ATR), 2)
            risk = entry - sl
            preferred = next((z for z in opposite_zones if z["zone_mid"] > entry), None)
        else:
            sl_anchor = float(zone["zone_high"])
            if sweep_context and sweep_context.get("direction") == "SELL":
                sl_anchor = max(sl_anchor, float(sweep_context.get("extreme", sl_anchor)))
            sl = round(sl_anchor + atr_m15 * float(cfg.M15_SR_SL_BUFFER_ATR), 2)
            risk = sl - entry
            preferred = next((z for z in opposite_zones if z["zone_mid"] < entry), None)

        if risk <= 0:
            return None

        target_source = "fixed_rr"
        min_rr = 1.40 if not signal_family else float(getattr(cfg, "M15_SR_NORMAL_MIN_RR", 1.5))
        max_rr_allowed: Optional[float] = None
        if signal_family == self.FAMILY_BOUNCE:
            min_rr = float(getattr(cfg, "M15_SR_BOUNCE_MIN_RR", 0.8))
            max_rr_allowed = float(getattr(cfg, "M15_SR_BOUNCE_MAX_RR", 1.2))
        elif signal_family == self.FAMILY_TRUE_REVERSAL:
            min_rr = float(getattr(cfg, "M15_SR_TRUE_REVERSAL_MIN_RR", 1.8))

        tp = None
        obstacle = None
        if signal_family == self.FAMILY_BOUNCE:
            obstacle = self._mean_reversion_target(direction, entry, risk, preferred, market_context or {}, closed)
            if obstacle is not None:
                tp = round(float(obstacle), 2)
                target_source = "nearest_obstacle"
        elif preferred is not None:
            tp = round(float(preferred["zone_mid"]), 2)
            target_source = "opposite_zone"

        if tp is None:
            default_rr = float(cfg.M15_SR_DEFAULT_RR)
            if max_rr_allowed is not None:
                default_rr = min(default_rr, max_rr_allowed)
            if signal_family == self.FAMILY_TRUE_REVERSAL:
                default_rr = max(default_rr, min_rr)
            tp = round(entry + (risk * default_rr), 2) if direction == "BUY" else round(entry - (risk * default_rr), 2)

        reward = (tp - entry) if direction == "BUY" else (entry - tp)
        rr = reward / risk if risk > 0 else 0.0
        if max_rr_allowed is not None and rr > max_rr_allowed:
            tp = round(entry + (risk * max_rr_allowed), 2) if direction == "BUY" else round(entry - (risk * max_rr_allowed), 2)
            reward = (tp - entry) if direction == "BUY" else (entry - tp)
            rr = reward / risk if risk > 0 else 0.0
            target_source = "capped_bounce_rr"
        if rr < min_rr:
            return None

        return {
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "sl_distance": round(abs(risk), 2),
            "rr": round(rr, 2),
            "_target_source": target_source,
            "_opposite_zone_id": preferred.get("zone_id") if preferred and target_source == "opposite_zone" else None,
            "_sl_anchor": round(sl_anchor, 2),
            "_sweep_reclaim": bool(sweep_context),
            "_sweep_extreme": round(float(sweep_context["extreme"]), 2) if sweep_context and sweep_context.get("extreme") is not None else None,
            "_obstacle_target": round(float(obstacle), 2) if obstacle is not None else None,
            "_min_rr_required": round(min_rr, 2),
            "_max_rr_allowed": round(max_rr_allowed, 2) if max_rr_allowed is not None else None,
        }

    def _check_spread_filter(self, tick: Dict, tick_snap: Dict) -> Tuple[bool, str]:
        spread = float(tick.get("spread", 0.0) or 0.0)
        if spread > float(cfg.M15_SR_MAX_SPREAD):
            return False, f"Spread {spread:.3f} > M15_SR_MAX_SPREAD {cfg.M15_SR_MAX_SPREAD:.3f}"
        if tick_snap.get("ready") and not tick_snap.get("spread_stable", True):
            return False, "Execution instability: spread unstable"
        return True, "OK"

    def _check_spike_block(self, closed: pd.DataFrame, atr_m15: float, now_utc: datetime) -> Tuple[bool, str]:
        last_candle = closed.iloc[-1]
        candle_time = self._coerce_dt(last_candle["datetime"])
        if candle_time is None:
            candle_time = now_utc

        candle_range = float(last_candle["high"] - last_candle["low"])
        recent_volume = closed["volume"].tail(8)
        avg_volume = float(recent_volume.iloc[:-1].mean()) if len(recent_volume) > 1 else float(recent_volume.mean() or 0.0)
        last_volume = float(last_candle.get("volume", 0.0) or 0.0)
        spike_range = candle_range > atr_m15 * float(cfg.M15_SR_SPIKE_RANGE_ATR_MULT)
        spike_volume = avg_volume > 0 and last_volume > avg_volume * float(cfg.M15_SR_SPIKE_VOLUME_MULT)
        if spike_range and spike_volume:
            block_candles = int(cfg.M15_SR_BLOCK_AFTER_SPIKE_CANDLES)
            self._spike_anchor_time = candle_time
            self._spike_block_until = candle_time + timedelta(minutes=15 * block_candles)

        if self._spike_block_until is not None and candle_time <= self._spike_block_until:
            anchor = self._spike_anchor_time.isoformat() if self._spike_anchor_time else "unknown"
            return False, f"Spike/manipulation block active after candle {anchor}"
        if self._spike_block_until is not None and candle_time > self._spike_block_until:
            self._spike_block_until = None
            self._spike_anchor_time = None
        return True, "OK"

    def _zone_quality_issue(self, zone: Dict) -> Optional[str]:
        if int(zone.get("touches", 0)) < int(cfg.M15_SR_MIN_TOUCHES):
            return "Zone quality failed: fewer than minimum touches"
        if zone.get("single_extreme"):
            return "Zone quality failed: formed by a single extreme wick"
        if zone.get("spike_based") and not bool(getattr(cfg, "M15_SR_SPIKE_ZONE_BYPASS", False)):
            return "Zone quality failed: spike-dominated zone"
        if zone.get("random_cross"):
            return "Zone quality failed: recent closes are crossing both sides of the zone"
        return None

    def _recent_close_break(self, zone: Dict, closed: pd.DataFrame, atr_m15: float, below: bool) -> bool:
        buffer = atr_m15 * float(cfg.M15_SR_ENTRY_BUFFER_ATR)
        recent = closed["close"].tail(2).tolist()
        if below:
            return any(float(close) < float(zone["zone_low"]) - buffer for close in recent)
        return any(float(close) > float(zone["zone_high"]) + buffer for close in recent)

    def _is_buy_rejection(self, candle: pd.Series, zone: Dict) -> bool:
        stats = self._wick_stats(candle)
        strong_close = stats["close_pct"] >= 0.60
        return (
            stats["lower_wick"] > max(stats["body"], stats["range"] * 0.20)
            and float(candle["close"]) > float(zone["zone_mid"])
            and (float(candle["close"]) > float(candle["open"]) or strong_close)
        )

    def _is_sell_rejection(self, candle: pd.Series, zone: Dict) -> bool:
        stats = self._wick_stats(candle)
        strong_close = stats["close_pct"] <= 0.40
        return (
            stats["upper_wick"] > max(stats["body"], stats["range"] * 0.20)
            and float(candle["close"]) < float(zone["zone_mid"])
            and (float(candle["close"]) < float(candle["open"]) or strong_close)
        )

    def _is_fake_breakout_candle(self, candle: pd.Series, zone: Dict, atr_m15: float, direction: str) -> bool:
        stats = self._wick_stats(candle)
        abnormal_range = stats["range"] > atr_m15 * 1.2
        if direction == "BUY":
            return (
                float(candle["low"]) < float(zone["zone_low"])
                and float(candle["close"]) >= float(zone["zone_low"])
                and abnormal_range
            )
        return (
            float(candle["high"]) > float(zone["zone_high"])
            and float(candle["close"]) <= float(zone["zone_high"])
            and abnormal_range
        )

    def _prior_candle_was_fake_breakout(self, closed: pd.DataFrame, zone: Dict, atr_m15: float, direction: str) -> bool:
        if len(closed) < 2:
            return False
        return self._is_fake_breakout_candle(closed.iloc[-2], zone, atr_m15, direction)

    def _recent_sweep_context(self, closed: pd.DataFrame, zone: Dict, atr_m15: float, direction: str) -> Optional[Dict]:
        if len(closed) < 2:
            return None
        zone_low = float(zone["zone_low"])
        zone_high = float(zone["zone_high"])
        max_overshoot = max(0.05, atr_m15 * 0.45)
        recent = closed.tail(2).reset_index(drop=True)
        sweeps: List[Dict] = []
        for idx in range(len(recent)):
            candle = recent.iloc[idx]
            if direction == "BUY":
                overshoot = zone_low - float(candle["low"])
                if overshoot <= 0:
                    continue
                sweeps.append(
                    {
                        "direction": direction,
                        "bars_ago": len(recent) - 1 - idx,
                        "extreme": float(candle["low"]),
                        "overshoot": round(overshoot, 4),
                        "reclaimed": float(candle["close"]) >= zone_low,
                        "valid_overshoot": overshoot <= max_overshoot,
                    }
                )
            else:
                overshoot = float(candle["high"]) - zone_high
                if overshoot <= 0:
                    continue
                sweeps.append(
                    {
                        "direction": direction,
                        "bars_ago": len(recent) - 1 - idx,
                        "extreme": float(candle["high"]),
                        "overshoot": round(overshoot, 4),
                        "reclaimed": float(candle["close"]) <= zone_high,
                        "valid_overshoot": overshoot <= max_overshoot,
                    }
                )
        if not sweeps:
            return None
        sweeps.sort(key=lambda item: (item["bars_ago"], -item["overshoot"]))
        return sweeps[0]

    def _is_choppy_market(self, closed: pd.DataFrame, support_zone: Optional[Dict], resistance_zone: Optional[Dict]) -> bool:
        if not support_zone or not resistance_zone:
            return False
        recent = closed.tail(3).reset_index(drop=True)
        support_touch_indices = {idx for idx, row in recent.iterrows() if self._candle_touches_zone(row, support_zone)}
        resistance_touch_indices = {idx for idx, row in recent.iterrows() if self._candle_touches_zone(row, resistance_zone)}
        return bool(support_touch_indices and resistance_touch_indices and support_touch_indices != resistance_touch_indices)

    def _strong_momentum_against_trade(self, closed: pd.DataFrame, direction: str) -> bool:
        if len(closed) < 2:
            return False
        recent = closed.tail(2)
        if direction == "BUY":
            return all(self._is_strong_bearish(row) for _, row in recent.iterrows())
        return all(self._is_strong_bullish(row) for _, row in recent.iterrows())

    def _cooldown_reason(self, direction: str, signal_candle_time: datetime) -> Optional[str]:
        last_trade_time = self._last_trade_candle_by_direction.get(direction)
        if last_trade_time is None:
            return None
        elapsed = int((signal_candle_time - last_trade_time).total_seconds() // 900)
        if elapsed < int(cfg.M15_SR_COOLDOWN_CANDLES):
            return f"Cooldown active for {direction}: {elapsed}/{cfg.M15_SR_COOLDOWN_CANDLES} candles elapsed"
        return None

    def _daily_trade_count(self, now_utc: datetime) -> int:
        day_key = date_str_ist(now_utc)
        return int(self._daily_trade_counts.get(day_key, 0))

    def _has_open_direction_position(self, positions: List[Dict], direction: str) -> bool:
        for pos in positions:
            ptype = str(pos.get("type") or pos.get("direction") or "").upper()
            if ptype != direction:
                continue
            strategy = str(pos.get("strategy") or "")
            comment = str(pos.get("comment") or "")
            if strategy == self.name or comment.startswith(f"FT_{self._POSITION_TAG}"):
                return True
        return False

    def _has_open_zone_position(self, positions: List[Dict], zone: Dict) -> bool:
        tag = zone["zone_id"][:8]
        for pos in positions:
            comment = str(pos.get("comment") or "")
            if comment == f"FT_{self._POSITION_TAG}_{tag}":
                return True
        return False

    def _log_decision(
        self,
        decision: str,
        reason: str,
        price: float,
        atr_m15: float,
        support_zones: List[Dict],
        resistance_zones: List[Dict],
        active_zone: Optional[Dict] = None,
        zone_touch_count: Optional[int] = None,
        entry_direction: Optional[str] = None,
        confirmation: Optional[Dict] = None,
        filter_results: Optional[Dict] = None,
        now_utc: Optional[datetime] = None,
        signal_data: Optional[Dict] = None,
        block_category: Optional[str] = None,
        lookback_used: Optional[int] = None,
        market_context: Optional[Dict] = None,
        candle_metrics: Optional[Dict] = None,
        signal_family_candidate: Optional[str] = None,
        rejection_reasons: Optional[List[str]] = None,
        intended_execution_time: Optional[datetime] = None,
    ):
        additional = {
            "_sim_now": now_utc,
            "current_price": round(float(price), 2),
            "atr_m15": round(float(atr_m15), 4),
            "support_zones": [self._zone_payload(zone) for zone in support_zones[:4]],
            "resistance_zones": [self._zone_payload(zone) for zone in resistance_zones[:4]],
            "active_zone": active_zone,
            "zone_touch_count": zone_touch_count,
            "entry_direction": entry_direction,
            "confirmation_candle": confirmation,
            "filter_results": filter_results or {},
            "block_category": block_category,
            "lookback_hours_used": lookback_used,
            "signal_family_candidate": signal_family_candidate,
            "rejection_reasons": rejection_reasons or [],
            "candle_metrics": candle_metrics or {},
            "market_context": market_context or {},
            "intended_execution_time": intended_execution_time.isoformat() if hasattr(intended_execution_time, "isoformat") else intended_execution_time,
        }
        if signal_data:
            additional["signal_data"] = {
                "entry": signal_data.get("entry"),
                "sl": signal_data.get("sl"),
                "tp": signal_data.get("tp"),
                "rr": signal_data.get("rr"),
                "lot_size": signal_data.get("lot_size"),
                "zone_id": (signal_data.get("_active_zone") or {}).get("zone_id"),
                "signal_family": signal_data.get("_signal_family"),
                "exit_profile": signal_data.get("_exit_profile"),
                "ttl_seconds": signal_data.get("_ttl_seconds"),
                "context_hash": signal_data.get("_context_hash"),
                "trend_classification": signal_data.get("_context_classification"),
            }
        log_decision(
            strategy=self.name,
            setup_direction="LONG" if entry_direction == "BUY" else "SHORT" if entry_direction == "SELL" else None,
            bias_direction=None,
            quality_score=float(signal_data.get("confidence", 0.0) if signal_data else 0.0),
            threshold=0.55,
            spread_mean=float((filter_results or {}).get("spread", {}).get("current", 0.0) or 0.0),
            spread_std=0.0,
            compression_ok=True,
            ltf_conflict=False,
            decision=decision,
            reason=reason,
            price=price,
            additional_data=additional,
        )

    def _skip(
        self,
        reason: str,
        price: float,
        atr_m15: float,
        support_zones: List[Dict],
        resistance_zones: List[Dict],
        active_zone: Optional[Dict] = None,
        zone_touch_count: Optional[int] = None,
        entry_direction: Optional[str] = None,
        confirmation: Optional[Dict] = None,
        filter_results: Optional[Dict] = None,
        now_utc: Optional[datetime] = None,
        block_category: Optional[str] = None,
        lookback_used: Optional[int] = None,
        terminal: bool = False,
        market_context: Optional[Dict] = None,
        candle_metrics: Optional[Dict] = None,
        signal_family_candidate: Optional[str] = None,
        rejection_reasons: Optional[List[str]] = None,
        zone_id: str = "",
        side: str = "",
        invalidate_pending: bool = False,
        intended_execution_time: Optional[datetime] = None,
    ) -> Dict:
        if invalidate_pending and zone_id and side:
            self._invalidate_pending_for_zone(zone_id, side, reason, now_utc or datetime.now(timezone.utc))
        self._log_decision(
            decision="TRADE_SKIPPED",
            reason=reason,
            price=price,
            atr_m15=atr_m15,
            support_zones=support_zones,
            resistance_zones=resistance_zones,
            active_zone=active_zone,
            zone_touch_count=zone_touch_count,
            entry_direction=entry_direction,
            confirmation=confirmation,
            filter_results=filter_results,
            now_utc=now_utc,
            block_category=block_category,
            lookback_used=lookback_used,
            market_context=market_context,
            candle_metrics=candle_metrics,
            signal_family_candidate=signal_family_candidate,
            rejection_reasons=rejection_reasons,
            intended_execution_time=intended_execution_time,
        )
        payload = self._no(reason)
        payload["_signal_family_candidate"] = signal_family_candidate
        payload["_rejection_reasons"] = rejection_reasons or []
        if terminal:
            payload["_terminal"] = True
        return payload

    @staticmethod
    def candle_metrics(candle: pd.Series) -> Optional[Dict[str, float]]:
        high = float(candle["high"])
        low = float(candle["low"])
        open_px = float(candle["open"])
        close_px = float(candle["close"])
        candle_range = high - low
        if candle_range <= 0:
            return None
        body = abs(close_px - open_px)
        lower_wick = min(open_px, close_px) - low
        upper_wick = high - max(open_px, close_px)
        close_pos = (close_px - low) / candle_range
        return {
            "range": candle_range,
            "body": body,
            "lower_wick": lower_wick,
            "upper_wick": upper_wick,
            "close_pos": close_pos,
            "lower_wick_pct": lower_wick / candle_range,
            "upper_wick_pct": upper_wick / candle_range,
            "upper_to_lower": upper_wick / max(lower_wick, 0.0001),
            "lower_to_upper": lower_wick / max(upper_wick, 0.0001),
            "is_green": close_px >= open_px,
            "is_red": close_px < open_px,
        }

    def buy_candle_quality_normal(self, candle: pd.Series, zone: Dict, atr_m15: float) -> bool:
        m = self.candle_metrics(candle)
        if m is None:
            return False
        return (
            float(candle["low"]) <= float(zone["zone_high"])
            and float(candle["close"]) > float(zone["zone_high"]) + 0.03 * atr_m15
            and m["close_pos"] >= 0.65
            and m["lower_wick_pct"] >= 0.40
            and m["upper_wick"] <= 0.75 * m["lower_wick"]
        )

    def buy_candle_quality_counter_bias(self, candle: pd.Series, zone: Dict, atr_m15: float) -> bool:
        m = self.candle_metrics(candle)
        if m is None:
            return False
        return (
            float(candle["low"]) <= float(zone["zone_high"])
            and float(candle["close"]) > float(zone["zone_high"]) + 0.03 * atr_m15
            and float(candle["close"]) >= float(candle["open"])
            and m["close_pos"] >= 0.75
            and m["lower_wick_pct"] >= 0.45
            and m["upper_wick"] <= 0.60 * m["lower_wick"]
            and int(zone.get("touches", 0) or 0) >= 3
        )

    def buy_candle_quality_true_reversal(self, candle: pd.Series, zone: Dict, atr_m15: float) -> bool:
        m = self.candle_metrics(candle)
        if m is None:
            return False
        return (
            float(candle["low"]) <= float(zone["zone_high"])
            and float(candle["close"]) > float(zone["zone_high"]) + 0.05 * atr_m15
            and float(candle["close"]) > float(candle["open"])
            and m["close_pos"] >= 0.80
            and m["lower_wick_pct"] >= 0.40
            and m["upper_wick"] <= 0.50 * m["lower_wick"]
        )

    def sell_candle_quality_normal(self, candle: pd.Series, zone: Dict, atr_m15: float) -> bool:
        m = self.candle_metrics(candle)
        if m is None:
            return False
        return (
            float(candle["high"]) >= float(zone["zone_low"])
            and float(candle["close"]) < float(zone["zone_low"]) - 0.03 * atr_m15
            and m["close_pos"] <= 0.35
            and m["upper_wick_pct"] >= 0.40
            and m["lower_wick"] <= 0.75 * m["upper_wick"]
        )

    def sell_candle_quality_counter_bias(self, candle: pd.Series, zone: Dict, atr_m15: float) -> bool:
        m = self.candle_metrics(candle)
        if m is None:
            return False
        return (
            float(candle["high"]) >= float(zone["zone_low"])
            and float(candle["close"]) < float(zone["zone_low"]) - 0.03 * atr_m15
            and float(candle["close"]) <= float(candle["open"])
            and m["close_pos"] <= 0.25
            and m["upper_wick_pct"] >= 0.45
            and m["lower_wick"] <= 0.60 * m["upper_wick"]
            and int(zone.get("touches", 0) or 0) >= 3
        )

    def sell_candle_quality_true_reversal(self, candle: pd.Series, zone: Dict, atr_m15: float) -> bool:
        m = self.candle_metrics(candle)
        if m is None:
            return False
        return (
            float(candle["high"]) >= float(zone["zone_low"])
            and float(candle["close"]) < float(zone["zone_low"]) - 0.05 * atr_m15
            and float(candle["close"]) < float(candle["open"])
            and m["close_pos"] <= 0.20
            and m["upper_wick_pct"] >= 0.40
            and m["lower_wick"] <= 0.50 * m["upper_wick"]
        )

    def classify_buy_context(self, ctx: Dict) -> str:
        h1 = str(ctx.get("h1_direction") or "RANGE").upper()
        m15 = str(ctx.get("m15_direction") or "RANGE").upper()
        bias = str(ctx.get("status_bias") or "NEUTRAL").upper()
        macro_long = ctx.get("macro_long_supportive")

        if bool(getattr(cfg, "M15_SR_H1_FILTER_ENABLED", True)) and h1 == "DOWN" and bool(getattr(cfg, "M15_SR_BLOCK_BUY_WHEN_H1_DOWN", True)):
            return "TRUE_REVERSAL_REQUIRED"
        if h1 == "UP" and m15 != "DOWN":
            if macro_long is False and bias == "SHORT":
                return "MEAN_REVERSION_BOUNCE_ONLY"
            return "TREND_ALIGNED"
        if h1 == "RANGE" and m15 == "DOWN" and bias == "SHORT":
            return "MEAN_REVERSION_BOUNCE_ONLY"
        if macro_long is False and bias == "SHORT":
            return "MEAN_REVERSION_BOUNCE_ONLY"
        if h1 == "RANGE":
            if m15 == "DOWN" and bool(getattr(cfg, "M15_SR_BLOCK_NORMAL_BUY_WHEN_M15_DOWN", True)):
                return "MEAN_REVERSION_BOUNCE_ONLY"
            return "TREND_ALIGNED" if bias != "SHORT" else "MEAN_REVERSION_BOUNCE_ONLY"
        return "BLOCK"

    def classify_sell_context(self, ctx: Dict) -> str:
        h1 = str(ctx.get("h1_direction") or "RANGE").upper()
        m15 = str(ctx.get("m15_direction") or "RANGE").upper()
        bias = str(ctx.get("status_bias") or "NEUTRAL").upper()
        macro_short = ctx.get("macro_short_supportive")

        if bool(getattr(cfg, "M15_SR_H1_FILTER_ENABLED", True)) and h1 == "UP" and bool(getattr(cfg, "M15_SR_BLOCK_SELL_WHEN_H1_UP", True)):
            return "TRUE_REVERSAL_REQUIRED"
        if h1 == "DOWN" and m15 != "UP":
            if macro_short is False and bias == "LONG":
                return "MEAN_REVERSION_BOUNCE_ONLY"
            return "TREND_ALIGNED"
        if h1 == "RANGE" and m15 == "UP" and bias == "LONG":
            return "MEAN_REVERSION_BOUNCE_ONLY"
        if macro_short is False and bias == "LONG":
            return "MEAN_REVERSION_BOUNCE_ONLY"
        if h1 == "RANGE":
            if m15 == "UP" and bool(getattr(cfg, "M15_SR_BLOCK_NORMAL_SELL_WHEN_M15_UP", True)):
                return "MEAN_REVERSION_BOUNCE_ONLY"
            return "TREND_ALIGNED" if bias != "LONG" else "MEAN_REVERSION_BOUNCE_ONLY"
        return "BLOCK"

    def bullish_m15_structure_shift_confirmed(self, ctx: Dict) -> bool:
        struct = (ctx.get("bias") or {}).get("m15_structure") or {}
        pattern = str(struct.get("pattern") or "").upper()
        bos_direction = str(struct.get("bos_direction") or "").upper()
        return bos_direction == "LONG" or pattern == "BULLISH" or str(ctx.get("m15_direction") or "") == "UP"

    def bearish_m15_structure_shift_confirmed(self, ctx: Dict) -> bool:
        struct = (ctx.get("bias") or {}).get("m15_structure") or {}
        pattern = str(struct.get("pattern") or "").upper()
        bos_direction = str(struct.get("bos_direction") or "").upper()
        return bos_direction == "SHORT" or pattern == "BEARISH" or str(ctx.get("m15_direction") or "") == "DOWN"

    def price_reclaimed_ema20_or_vwap(self, direction: str, candle: pd.Series, ctx: Dict) -> bool:
        close_px = float(candle["close"])
        ema20_val = float(ctx.get("ema20", 0.0) or 0.0)
        vwap_val = float(ctx.get("vwap", 0.0) or 0.0)
        if direction == "BUY":
            return (ema20_val > 0 and close_px >= ema20_val) or (vwap_val > 0 and close_px >= vwap_val)
        return (ema20_val > 0 and close_px <= ema20_val) or (vwap_val > 0 and close_px <= vwap_val)

    def higher_low_confirmed(self, closed: Optional[pd.DataFrame]) -> bool:
        swings = self._swing_points(closed if closed is not None else pd.DataFrame(), "support")
        return len(swings) >= 2 and float(swings[-1]["price"]) > float(swings[-2]["price"])

    def lower_high_confirmed(self, closed: Optional[pd.DataFrame]) -> bool:
        swings = self._swing_points(closed if closed is not None else pd.DataFrame(), "resistance")
        return len(swings) >= 2 and float(swings[-1]["price"]) < float(swings[-2]["price"])

    @staticmethod
    def _wick_stats(candle: pd.Series) -> Dict:
        high = float(candle["high"])
        low = float(candle["low"])
        open_px = float(candle["open"])
        close_px = float(candle["close"])
        candle_range = max(0.0001, high - low)
        body = abs(close_px - open_px)
        return {
            "range": candle_range,
            "body": body,
            "upper_wick": high - max(open_px, close_px),
            "lower_wick": min(open_px, close_px) - low,
            "close_pct": (close_px - low) / candle_range,
        }

    @staticmethod
    def _zone_distance(price: float, zone_low: float, zone_high: float) -> float:
        if zone_low <= price <= zone_high:
            return 0.0
        return min(abs(price - zone_low), abs(price - zone_high))

    def _price_near_zone(self, price: float, zone: Dict) -> bool:
        return self._zone_distance(price, float(zone["zone_low"]), float(zone["zone_high"])) <= float(zone.get("max_distance_abs", 0.0) or 0.0)

    @staticmethod
    def _has_random_closes(closes: List[float], zone_low: float, zone_high: float, zone_type: str = "") -> bool:
        recent = closes[-3:]
        above = any(float(c) > zone_high for c in recent)
        below = any(float(c) < zone_low for c in recent)
        if not (above and below):
            return False
        last = float(recent[-1])
        if zone_type == "support" and last > zone_high:
            return False
        if zone_type == "resistance" and last < zone_low:
            return False
        return True

    @staticmethod
    def _nearest_zone(zones: List[Dict], current_price: float) -> Optional[Dict]:
        if not zones:
            return None
        return sorted(zones, key=lambda item: (item["distance"], -item["touches"], abs(item["zone_mid"] - current_price)))[0]

    @staticmethod
    def _candle_touches_zone(candle: pd.Series, zone: Dict) -> bool:
        return float(candle["low"]) <= float(zone["zone_high"]) and float(candle["high"]) >= float(zone["zone_low"])

    def _is_strong_bearish(self, candle: pd.Series) -> bool:
        stats = self._wick_stats(candle)
        return float(candle["close"]) < float(candle["open"]) and stats["close_pct"] <= 0.35 and stats["body"] >= stats["range"] * 0.45

    def _is_strong_bullish(self, candle: pd.Series) -> bool:
        stats = self._wick_stats(candle)
        return float(candle["close"]) > float(candle["open"]) and stats["close_pct"] >= 0.65 and stats["body"] >= stats["range"] * 0.45

    def _confidence_score(self, zone: Dict, candle: pd.Series, plan: Dict, signal_family: str) -> float:
        stats = self._wick_stats(candle)
        wick_bonus = min(0.08, max(stats["upper_wick"], stats["lower_wick"]) / max(stats["range"], 0.0001) * 0.08)
        touch_bonus = min(0.12, max(0, int(zone.get("touches", 0)) - int(cfg.M15_SR_MIN_TOUCHES)) * 0.04)
        rr_bonus = min(0.10, max(0.0, float(plan.get("rr", 0.0)) - float(plan.get("_min_rr_required", cfg.M15_SR_MIN_RR))) * 0.08)
        family_bonus = 0.02 if signal_family == self.FAMILY_TREND else 0.0
        if signal_family == self.FAMILY_TRUE_REVERSAL:
            family_bonus = 0.04
        return round(min(0.92, 0.58 + wick_bonus + touch_bonus + rr_bonus + family_bonus), 3)

    @staticmethod
    def _candle_snapshot(candle: pd.Series) -> Dict:
        return {
            "time": candle["datetime"].isoformat() if hasattr(candle["datetime"], "isoformat") else str(candle["datetime"]),
            "open": round(float(candle["open"]), 2),
            "high": round(float(candle["high"]), 2),
            "low": round(float(candle["low"]), 2),
            "close": round(float(candle["close"]), 2),
            "volume": round(float(candle.get("volume", 0.0) or 0.0), 2),
        }

    @staticmethod
    def _zone_payload(zone: Optional[Dict]) -> Optional[Dict]:
        if not zone:
            return None
        return {
            "zone_id": zone.get("zone_id"),
            "type": zone.get("type"),
            "zone_low": zone.get("zone_low"),
            "zone_high": zone.get("zone_high"),
            "zone_mid": zone.get("zone_mid"),
            "touches": zone.get("touches"),
            "distance": zone.get("distance"),
            "lookback_hours": zone.get("lookback_hours"),
        }

    @staticmethod
    def _closed_frame(frame: pd.DataFrame) -> pd.DataFrame:
        if frame is None or frame.empty:
            return pd.DataFrame()
        if "datetime" in frame.columns and len(frame) >= 2:
            return frame.iloc[:-1].reset_index(drop=True)
        return frame.copy().reset_index(drop=True)

    @staticmethod
    def _mid_price(tick: Dict) -> float:
        bid = float(tick.get("bid", 0.0) or 0.0)
        ask = float(tick.get("ask", bid) or bid)
        if bid > 0 and ask > 0:
            return round((bid + ask) / 2.0, 2)
        return round(max(bid, ask), 2)

    @staticmethod
    def _coerce_dt(value) -> Optional[datetime]:
        if isinstance(value, pd.Timestamp):
            value = value.to_pydatetime()
        if isinstance(value, datetime):
            return value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
        if isinstance(value, str):
            try:
                dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
                return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
            except Exception:
                return None
        return None

    @staticmethod
    def _resolve_now(data: Dict, tick: Dict, closed: pd.DataFrame) -> datetime:
        now_utc = data.get("now_utc")
        if isinstance(now_utc, datetime):
            return now_utc.astimezone(timezone.utc) if now_utc.tzinfo else now_utc.replace(tzinfo=timezone.utc)
        tick_time = tick.get("time")
        parsed = M15SupportResistanceStrategy._coerce_dt(tick_time)
        if parsed is not None:
            return parsed
        if closed is not None and not closed.empty:
            candle_time = M15SupportResistanceStrategy._coerce_dt(closed.iloc[-1]["datetime"])
            if candle_time is not None:
                return candle_time
        return datetime.now(timezone.utc)

    @staticmethod
    def _atr_m15(closed: pd.DataFrame) -> float:
        highs = closed["high"].astype(float).to_numpy()
        lows = closed["low"].astype(float).to_numpy()
        closes = closed["close"].astype(float).to_numpy()
        return round(float(atr(highs, lows, closes, 14)[-1]), 4)

    @staticmethod
    def _no(reason: str) -> Dict:
        return {"signal": "NO_TRADE", "reason": reason, "score": 0.0}

    def _build_market_context(self, data: Dict, closed: Optional[pd.DataFrame], now_utc: datetime) -> Dict:
        bias = dict(data.get("bias") or {})
        regime = dict(data.get("regime") or {})
        market_state = dict(data.get("market_state") or {})
        trend = dict(market_state.get("trend") or {})
        structure = dict(market_state.get("structure") or {})
        indicators = dict(data.get("indicators") or {})
        tf_context = dict(data.get("tf_context") or {})
        tick_pressure = dict(data.get("tick_pressure") or {})
        h1_direction = self._direction_label(
            trend.get("H1"),
            structure.get("H1"),
            (bias.get("h1_structure") or {}).get("pattern"),
            tf_context.get("h1_ema50_slope"),
        )
        m15_direction = self._direction_label(
            trend.get("M15"),
            (bias.get("m15_structure") or {}).get("pattern"),
            None,
            indicators.get("ema20_slope"),
        )
        session = str(data.get("session") or get_session_at(now_utc)).upper()
        ema20_val = 0.0
        vwap_val = 0.0
        range_mid = 0.0
        if closed is not None and len(closed) >= 20:
            closes = closed["close"].astype(float).to_numpy()
            ema20_val = float(ema(closes, 20)[-1])
            recent = closed.tail(20)
            volume = recent["volume"].astype(float)
            typical = (recent["high"].astype(float) + recent["low"].astype(float) + recent["close"].astype(float)) / 3.0
            vol_sum = float(volume.sum() or 0.0)
            if vol_sum > 0:
                vwap_val = float((typical * volume).sum() / vol_sum)
            range_mid = float((recent["high"].astype(float).max() + recent["low"].astype(float).min()) / 2.0)
        macro_long = self._macro_supportive("BUY", bias, market_state, h1_direction)
        macro_short = self._macro_supportive("SELL", bias, market_state, h1_direction)
        return {
            "bias": bias,
            "regime": str(regime.get("state") or "UNKNOWN").upper(),
            "market_state": market_state,
            "h1_direction": h1_direction,
            "m15_direction": m15_direction,
            "status_bias": self._normalize_bias_direction(bias.get("direction")),
            "session": session,
            "tick_pressure": tick_pressure,
            "ema20": round(ema20_val, 4) if ema20_val else 0.0,
            "vwap": round(vwap_val, 4) if vwap_val else 0.0,
            "range_mid": round(range_mid, 4) if range_mid else 0.0,
            "macro_long_supportive": macro_long,
            "macro_short_supportive": macro_short,
            "macro_support_status": self._macro_support_status(macro_long, macro_short),
        }

    @staticmethod
    def _normalize_bias_direction(direction: Any) -> str:
        raw = str(direction or "NEUTRAL").upper()
        if raw == "BUY":
            return "LONG"
        if raw == "SELL":
            return "SHORT"
        return raw if raw in {"LONG", "SHORT", "NEUTRAL"} else "NEUTRAL"

    @staticmethod
    def _direction_label(primary: Any, secondary: Any, tertiary: Any, slope: Any) -> str:
        for value in (primary, secondary, tertiary):
            raw = str(value or "").upper()
            if raw in {"UP", "DOWN", "RANGE"}:
                return raw
            if raw in {"BULLISH", "LONG"}:
                return "UP"
            if raw in {"BEARISH", "SHORT"}:
                return "DOWN"
        try:
            slope_val = float(slope or 0.0)
            if slope_val > 0:
                return "UP"
            if slope_val < 0:
                return "DOWN"
        except Exception:
            pass
        return "RANGE"

    def _macro_supportive(self, side: str, bias: Dict, market_state: Dict, h1_direction: str) -> Optional[bool]:
        trend = dict((market_state.get("trend") or {}))
        structure = dict((market_state.get("structure") or {}))
        h4 = self._direction_label(trend.get("H4"), structure.get("H4"), None, None)
        d1 = self._direction_label(trend.get("D1"), structure.get("D1"), None, None)
        bias_dir = self._normalize_bias_direction((bias or {}).get("direction"))
        if side == "BUY":
            up_votes = sum(1 for item in (h1_direction, h4, d1) if item == "UP")
            down_votes = sum(1 for item in (h1_direction, h4, d1) if item == "DOWN")
            if bias_dir == "SHORT" and down_votes >= 2:
                return False
            if up_votes >= 2 and bias_dir != "SHORT":
                return True
            return None
        up_votes = sum(1 for item in (h1_direction, h4, d1) if item == "UP")
        down_votes = sum(1 for item in (h1_direction, h4, d1) if item == "DOWN")
        if bias_dir == "LONG" and up_votes >= 2:
            return False
        if down_votes >= 2 and bias_dir != "LONG":
            return True
        return None

    @staticmethod
    def _macro_support_status(macro_long: Optional[bool], macro_short: Optional[bool]) -> str:
        if macro_long is False and macro_short is True:
            return "SHORT_SUPPORTIVE"
        if macro_short is False and macro_long is True:
            return "LONG_SUPPORTIVE"
        if macro_long is False or macro_short is False:
            return "OPPOSING"
        if macro_long is True or macro_short is True:
            return "SUPPORTIVE"
        return "UNKNOWN"

    def _signal_family_for_classification(self, classification: str) -> str:
        if classification == "TREND_ALIGNED":
            return self.FAMILY_TREND
        if classification == "MEAN_REVERSION_BOUNCE_ONLY":
            return self.FAMILY_BOUNCE
        if classification == "TRUE_REVERSAL_REQUIRED":
            return self.FAMILY_TRUE_REVERSAL
        return "M15"

    def select_exit_profile_for_signal_family(self, signal_family: str, market_context: Dict) -> str:
        if signal_family == self.FAMILY_BOUNCE:
            return "m15_mean_reversion_fast"
        if signal_family == self.FAMILY_TRUE_REVERSAL:
            return "swing_structured"
        return "structured_intraday" if str(market_context.get("h1_direction") or "") == "RANGE" else "swing_structured"

    @staticmethod
    def _classification_rank(classification: str) -> int:
        ranks = {
            "TREND_ALIGNED": 0,
            "MEAN_REVERSION_BOUNCE_ONLY": 1,
            "TRUE_REVERSAL_REQUIRED": 2,
            "BLOCK": 3,
        }
        return ranks.get(str(classification or "").upper(), 3)

    def _mean_reversion_target(
        self,
        direction: str,
        entry: float,
        risk: float,
        preferred_zone: Optional[Dict],
        market_context: Dict,
        closed: Optional[pd.DataFrame],
    ) -> Optional[float]:
        candidates: List[float] = []
        if preferred_zone:
            candidates.append(float(preferred_zone.get("zone_mid", 0.0) or 0.0))
        for key in ("vwap", "ema20", "range_mid"):
            value = float(market_context.get(key, 0.0) or 0.0)
            if value > 0:
                candidates.append(value)
        if closed is not None and len(closed):
            recent = closed.tail(8)
            if direction == "BUY":
                candidates.append(float(recent["high"].astype(float).max()))
            else:
                candidates.append(float(recent["low"].astype(float).min()))
        filtered = []
        for candidate in candidates:
            if direction == "BUY" and candidate > entry:
                filtered.append(candidate)
            elif direction == "SELL" and candidate < entry:
                filtered.append(candidate)
        if not filtered:
            rr = float(getattr(cfg, "M15_SR_BOUNCE_MIN_RR", 0.8))
            return entry + (risk * rr) if direction == "BUY" else entry - (risk * rr)
        return min(filtered) if direction == "BUY" else max(filtered)

    def _tick_pressure_flipped_against(self, side: str, score: float, bias: str) -> bool:
        if side == "BUY":
            return score < 0 or bias == "SHORT"
        if side == "SELL":
            return score > 0 or bias == "LONG"
        return False

    def _calculate_rr(self, side: str, entry: float, sl: float, tp: float) -> float:
        side = str(side or "").upper()
        if side == "BUY":
            risk = entry - sl
            reward = tp - entry
        else:
            risk = sl - entry
            reward = entry - tp
        if risk <= 0:
            return 0.0
        return round(reward / risk, 3)

    def _make_signal_id(self, direction: str, zone: Dict, now_utc: datetime) -> str:
        return f"{self.name}:{direction}:{zone.get('zone_id')}:{int(now_utc.timestamp())}"

    def _context_hash(self, context: Dict) -> str:
        payload = {
            "h1": context.get("h1_direction"),
            "m15": context.get("m15_direction"),
            "bias": context.get("status_bias"),
            "session": context.get("session"),
            "regime": context.get("regime"),
            "macro": context.get("macro_support_status"),
            "tick_bias": (context.get("tick_pressure") or {}).get("directional_bias"),
        }
        return hashlib.md5(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def _register_pending_signal(self, signal: Dict) -> None:
        signal_id = str(signal.get("_signal_id") or "")
        zone_id = str(signal.get("_zone_id") or "")
        side = str(signal.get("signal") or "").upper()
        if not signal_id or not zone_id or side not in {"BUY", "SELL"}:
            return
        self._pending_signals[signal_id] = {
            "signal_id": signal_id,
            "zone_id": zone_id,
            "side": side,
            "decision_time": signal.get("_decision_time"),
            "invalidated": False,
            "invalidate_reason": "",
        }
        self._pending_by_zone_side[(zone_id, side)] = signal_id

    def _invalidate_pending_for_zone(self, zone_id: str, side: str, reason: str, now_utc: datetime) -> None:
        key = (str(zone_id or ""), str(side or "").upper())
        signal_id = self._pending_by_zone_side.get(key)
        if not signal_id:
            return
        pending = self._pending_signals.get(signal_id)
        if not pending:
            return
        pending["invalidated"] = True
        pending["invalidate_reason"] = reason
        pending["invalidated_at"] = now_utc.isoformat()

    def _rejected_setup_payload(
        self,
        direction: str,
        zone: Dict,
        candle: pd.Series,
        atr_m15: float,
        market_context: Dict,
        reason: str,
        signal_family_candidate: str,
        rejection_reasons: List[str],
        intended_execution_time: Optional[datetime] = None,
    ) -> Dict:
        payload = self._no(reason)
        payload["_signal_family_candidate"] = signal_family_candidate
        payload["_rejection_reasons"] = rejection_reasons
        payload["_decision_context"] = {
            "zone_id": zone.get("zone_id"),
            "touches": zone.get("touches"),
            "atr_m15": atr_m15,
            "candle_metrics": self.candle_metrics(candle),
            "market_context": market_context,
            "intended_execution_time": intended_execution_time.isoformat() if hasattr(intended_execution_time, "isoformat") else intended_execution_time,
        }
        return payload

    def _asian_counter_bias_trade_count(self, now_utc: datetime) -> int:
        return int(self._asian_counter_bias_counts.get(self._session_key(now_utc, "ASIAN"), 0))

    @staticmethod
    def _session_key(now_utc: datetime, session: str) -> str:
        return f"{date_str_ist(now_utc)}:{session}"

    @staticmethod
    def _breakeven_trigger_r(signal_family: str) -> float:
        if signal_family == M15SupportResistanceStrategy.FAMILY_BOUNCE:
            return float(getattr(cfg, "EXIT_PROFILE_M15_BOUNCE_BE_TRIGGER_R", 0.22))
        if signal_family == M15SupportResistanceStrategy.FAMILY_TREND:
            return float(getattr(cfg, "EXIT_PROFILE_STRUCTURED_INTRADAY_BE_TRIGGER_R", 0.35))
        return float(getattr(cfg, "EXIT_PROFILE_M15_BE_TRIGGER_R", 0.60))

    @staticmethod
    def _timeout_seconds(signal_family: str) -> int:
        if signal_family == M15SupportResistanceStrategy.FAMILY_BOUNCE:
            candles = int(getattr(cfg, "M15_SR_BOUNCE_MAX_HOLD_CANDLES", 4) or 4)
            return max(60, candles * 15 * 60)
        if signal_family == M15SupportResistanceStrategy.FAMILY_TREND:
            return int(getattr(cfg, "EXIT_PROFILE_STRUCTURED_INTRADAY_TIMEOUT_SECONDS", 1800) or 1800)
        return int(getattr(cfg, "EXIT_PROFILE_M15_TIMEOUT_SECONDS", 300) or 300)

    @staticmethod
    def _timeout_min_progress(signal_family: str) -> float:
        if signal_family == M15SupportResistanceStrategy.FAMILY_BOUNCE:
            return float(getattr(cfg, "EXIT_PROFILE_M15_BOUNCE_TIMEOUT_MIN_PROGRESS_R", 0.05))
        if signal_family == M15SupportResistanceStrategy.FAMILY_TREND:
            return float(getattr(cfg, "EXIT_PROFILE_STRUCTURED_INTRADAY_TIMEOUT_MIN_PROGRESS_R", 0.15))
        return float(getattr(cfg, "EXIT_PROFILE_M15_TIMEOUT_MIN_PROGRESS_R", 0.30))

    @staticmethod
    def _min_hold_seconds(signal_family: str) -> int:
        if signal_family == M15SupportResistanceStrategy.FAMILY_BOUNCE:
            return int(getattr(cfg, "EXIT_PROFILE_M15_BOUNCE_MIN_HOLD_SECONDS", 15) or 15)
        if signal_family == M15SupportResistanceStrategy.FAMILY_TREND:
            return int(getattr(cfg, "EXIT_PROFILE_STRUCTURED_INTRADAY_MIN_HOLD_SECONDS", 30) or 30)
        return int(getattr(cfg, "EXIT_PROFILE_M15_MIN_HOLD_SECONDS", 60) or 60)

    @staticmethod
    def _reversal_arm_r(signal_family: str) -> float:
        if signal_family == M15SupportResistanceStrategy.FAMILY_BOUNCE:
            return float(getattr(cfg, "EXIT_PROFILE_M15_BOUNCE_REVERSAL_ARM_R", 0.80))
        if signal_family == M15SupportResistanceStrategy.FAMILY_TREND:
            return float(getattr(cfg, "EXIT_PROFILE_STRUCTURED_INTRADAY_REVERSAL_ARM_R", 1.00))
        return float(getattr(cfg, "EXIT_PROFILE_M15_REVERSAL_ARM_R", 1.50))

    @staticmethod
    def _reversal_drawdown_pct(signal_family: str) -> float:
        if signal_family == M15SupportResistanceStrategy.FAMILY_BOUNCE:
            return float(getattr(cfg, "EXIT_PROFILE_M15_BOUNCE_REVERSAL_DRAWDOWN_PCT", 0.60))
        if signal_family == M15SupportResistanceStrategy.FAMILY_TREND:
            return float(getattr(cfg, "EXIT_PROFILE_STRUCTURED_INTRADAY_REVERSAL_DRAWDOWN_PCT", 0.70))
        return float(getattr(cfg, "EXIT_PROFILE_M15_REVERSAL_DRAWDOWN_PCT", 0.75))

    @staticmethod
    def _reversal_floor_r(signal_family: str) -> float:
        if signal_family == M15SupportResistanceStrategy.FAMILY_BOUNCE:
            return float(getattr(cfg, "EXIT_PROFILE_M15_BOUNCE_REVERSAL_FLOOR_R", 0.10))
        if signal_family == M15SupportResistanceStrategy.FAMILY_TREND:
            return float(getattr(cfg, "EXIT_PROFILE_STRUCTURED_INTRADAY_REVERSAL_FLOOR_R", 0.20))
        return float(getattr(cfg, "EXIT_PROFILE_M15_REVERSAL_FLOOR_R", 0.50))

    @staticmethod
    def _trail_activate_r(signal_family: str) -> float:
        if signal_family == M15SupportResistanceStrategy.FAMILY_BOUNCE:
            return float(getattr(cfg, "EXIT_PROFILE_M15_BOUNCE_TRAIL_ACTIVATE_R", 0.90))
        if signal_family == M15SupportResistanceStrategy.FAMILY_TREND:
            return float(getattr(cfg, "EXIT_PROFILE_STRUCTURED_INTRADAY_TRAIL_ACTIVATE_R", 1.10))
        return float(getattr(cfg, "EXIT_PROFILE_M15_TRAIL_ACTIVATE_R", 1.50))

    @staticmethod
    def _trail_lock_r(signal_family: str) -> float:
        if signal_family == M15SupportResistanceStrategy.FAMILY_BOUNCE:
            return float(getattr(cfg, "EXIT_PROFILE_M15_BOUNCE_TRAIL_LOCK_R", 0.20))
        if signal_family == M15SupportResistanceStrategy.FAMILY_TREND:
            return float(getattr(cfg, "EXIT_PROFILE_STRUCTURED_INTRADAY_TRAIL_LOCK_R", 0.35))
        return float(getattr(cfg, "EXIT_PROFILE_M15_TRAIL_LOCK_R", 0.75))

    @staticmethod
    def _velocity_drop_enabled(signal_family: str) -> bool:
        if signal_family == M15SupportResistanceStrategy.FAMILY_BOUNCE:
            return bool(getattr(cfg, "EXIT_PROFILE_M15_BOUNCE_VELOCITY_DROP_ENABLED", True))
        if signal_family == M15SupportResistanceStrategy.FAMILY_TREND:
            return bool(getattr(cfg, "EXIT_PROFILE_STRUCTURED_INTRADAY_VELOCITY_DROP_ENABLED", True))
        return bool(getattr(cfg, "EXIT_PROFILE_M15_VELOCITY_DROP_ENABLED", False))

    def _log_pre_send_rejection(
        self,
        signal: Dict,
        age_seconds: float,
        current_price: float,
        planned_entry: float,
        drift_atr: float,
        spread: float,
        recalculated_rr: float,
        pressure_score: float,
        pressure_bias: str,
        live_ctx: Dict,
        reason: str,
        now_utc: datetime,
    ) -> None:
        log_decision(
            strategy=self.name,
            setup_direction="LONG" if str(signal.get("signal") or "").upper() == "BUY" else "SHORT",
            bias_direction=None,
            quality_score=float(signal.get("confidence", 0.0) or 0.0),
            threshold=0.55,
            spread_mean=spread,
            spread_std=0.0,
            compression_ok=True,
            ltf_conflict=False,
            decision="TRADE_SKIPPED",
            reason=reason,
            price=current_price,
            additional_data={
                "_sim_now": now_utc,
                "pre_send_rejection": {
                    "signal_age_seconds": round(age_seconds, 2),
                    "current_price": round(current_price, 2),
                    "planned_entry": round(planned_entry, 2),
                    "price_drift_atr": round(drift_atr, 3),
                    "current_spread": round(spread, 4),
                    "recalculated_rr": round(recalculated_rr, 3),
                    "current_tick_pressure_score": round(pressure_score, 3),
                    "current_tick_pressure_bias": pressure_bias,
                    "current_context": live_ctx,
                    "rejection_reason": reason,
                    "signal_id": signal.get("_signal_id"),
                    "signal_family": signal.get("_signal_family"),
                }
            },
        )
