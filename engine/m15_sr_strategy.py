"""
Conservative M15 support/resistance rejection strategy for XAUUSD.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd

import config as cfg
from .decision_logger import log_decision
from .indicators import atr
from .strategies.base_strategy import BaseStrategy
from .tick_processor import TickProcessor, entry_pressure_block_reason
from .time_utils import date_str_ist


class M15SupportResistanceStrategy(BaseStrategy):
    name = "M15_SUPPORT_RESISTANCE_REJECTION_V1"
    _POSITION_TAG = "M15SR"

    def __init__(self):
        self.tick_proc = TickProcessor(buffer_size=400)
        self._daily_trade_counts: Dict[str, int] = {}
        self._last_trade_candle_by_direction: Dict[str, datetime] = {}
        self._spike_block_until: Optional[datetime] = None
        self._spike_anchor_time: Optional[datetime] = None

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

        filter_results = {
            "spread": {"passed": True, "reason": "OK", "current": round(float(tick.get("spread", 0.0) or 0.0), 4)},
            "manipulation": {"passed": True, "reason": "OK"},
            "tick_pressure": {"passed": True, "reason": "OK"},
            "zone_quality": {"passed": True, "reason": "OK"},
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
            )

        candidate_signals.sort(key=lambda item: (float(item.get("confidence", 0.0)), float(item.get("rr", 0.0))), reverse=True)
        signal = candidate_signals[0]
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
            context["reason"] = "M15_CONTEXT_BLOCK: direction not aligned"
            return context
        if relevant_zone and distance_atr is not None and distance_atr > max_distance and not breakout_confirmation:
            context["reason"] = "M15_CONTEXT_BLOCK: price too far from zone"
            return context
        if not (candle_confirmation or breakout_confirmation):
            context["reason"] = "M15_CONTEXT_BLOCK: direction not aligned"
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
    ) -> Dict:
        confirmation_candle = closed.iloc[-1]
        confirmation = self._candle_snapshot(confirmation_candle)
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
            )

        sweep_context = self._recent_sweep_context(closed, active_zone, atr_m15, direction)

        if direction == "BUY":
            if not self._price_near_zone(current_price, active_zone):
                return self._no("Support zone not near price")
            if self._recent_close_break(active_zone, closed, atr_m15, below=True):
                return self._no("Support recently broken")
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
                )
            if sweep_context and not sweep_context.get("valid_overshoot", False):
                return self._no("Support sweep was too deep to treat as a clean reclaim")
            if sweep_context and not sweep_context.get("reclaimed", False):
                return self._no("Support sweep did not reclaim the zone")
            if sweep_context and not self._is_buy_rejection(confirmation_candle, active_zone):
                if int(sweep_context.get("bars_ago", 0)) == 0:
                    return self._skip(
                        reason="Awaiting strong bullish reclaim after support sweep",
                        price=current_price,
                        atr_m15=atr_m15,
                        support_zones=[nearest_support] if nearest_support else [],
                        resistance_zones=[nearest_resistance] if nearest_resistance else [],
                        active_zone=self._zone_payload(active_zone),
                        zone_touch_count=active_zone.get("touches"),
                        entry_direction=direction,
                        confirmation=confirmation,
                        filter_results={**filter_results, "manipulation": {"passed": False, "reason": "Weak reclaim after support sweep"}},
                        now_utc=now_utc,
                        block_category="manipulation",
                        lookback_used=active_zone.get("lookback_hours"),
                        terminal=True,
                    )
                return self._no("Post-sweep candle did not confirm BUY")
            if cfg.M15_SR_REQUIRE_CANDLE_CONFIRMATION and not sweep_context and not self._is_buy_rejection(confirmation_candle, active_zone):
                return self._no("No bullish rejection at support")
            if cfg.M15_SR_REQUIRE_CANDLE_CONFIRMATION and self._prior_candle_was_fake_breakout(closed, active_zone, atr_m15, "BUY"):
                if not self._is_buy_rejection(confirmation_candle, active_zone):
                    return self._no("Post-sweep candle did not confirm BUY")
        else:
            if not self._price_near_zone(current_price, active_zone):
                return self._no("Resistance zone not near price")
            if self._recent_close_break(active_zone, closed, atr_m15, below=False):
                return self._no("Resistance recently broken")
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
                )
            if sweep_context and not sweep_context.get("valid_overshoot", False):
                return self._no("Resistance sweep was too deep to treat as a clean reclaim")
            if sweep_context and not sweep_context.get("reclaimed", False):
                return self._no("Resistance sweep did not reclaim the zone")
            if sweep_context and not self._is_sell_rejection(confirmation_candle, active_zone):
                if int(sweep_context.get("bars_ago", 0)) == 0:
                    return self._skip(
                        reason="Awaiting strong bearish reclaim after resistance sweep",
                        price=current_price,
                        atr_m15=atr_m15,
                        support_zones=[nearest_support] if nearest_support else [],
                        resistance_zones=[nearest_resistance] if nearest_resistance else [],
                        active_zone=self._zone_payload(active_zone),
                        zone_touch_count=active_zone.get("touches"),
                        entry_direction=direction,
                        confirmation=confirmation,
                        filter_results={**filter_results, "manipulation": {"passed": False, "reason": "Weak reclaim after resistance sweep"}},
                        now_utc=now_utc,
                        block_category="manipulation",
                        lookback_used=active_zone.get("lookback_hours"),
                        terminal=True,
                    )
                return self._no("Post-sweep candle did not confirm SELL")
            if cfg.M15_SR_REQUIRE_CANDLE_CONFIRMATION and not sweep_context and not self._is_sell_rejection(confirmation_candle, active_zone):
                return self._no("No bearish rejection at resistance")
            if cfg.M15_SR_REQUIRE_CANDLE_CONFIRMATION and self._prior_candle_was_fake_breakout(closed, active_zone, atr_m15, "SELL"):
                if not self._is_sell_rejection(confirmation_candle, active_zone):
                    return self._no("Post-sweep candle did not confirm SELL")

        plan = self._build_trade_plan(
            direction=direction,
            zone=active_zone,
            opposite_zones=opposite_zones,
            tick=tick,
            atr_m15=atr_m15,
            sweep_context=sweep_context,
        )
        if plan is None:
            return self._skip(
                reason="Poor RR: no valid M15 SR trade plan",
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
                block_category="poor_rr",
                lookback_used=active_zone.get("lookback_hours"),
                terminal=True,
            )

        confidence = self._confidence_score(active_zone, confirmation_candle, plan)
        tick_pressure = tick_pressure or {}
        pressure_reason = entry_pressure_block_reason(
            direction,
            tick_pressure,
            strong_threshold=float(getattr(cfg, "TICK_PRESSURE_ENTRY_BLOCK_THRESHOLD", 0.35) or 0.35),
            short_positive_veto_threshold=float(getattr(cfg, "TICK_PRESSURE_SHORT_ENTRY_VETO_THRESHOLD", 0.05) or 0.05),
        )
        if pressure_reason:
            filter_results["tick_pressure"] = {"passed": False, "reason": pressure_reason}
            return self._skip(
                reason=pressure_reason,
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
                block_category="tick_pressure",
                lookback_used=active_zone.get("lookback_hours"),
                terminal=True,
            )
        signal = {
            "signal": direction,
            "entry": plan["entry"],
            "sl": plan["sl"],
            "tp": plan["tp"],
            "sl_distance": plan["sl_distance"],
            "rr": plan["rr"],
            "confidence": confidence,
            "reason": f"{direction} rejection from {active_zone['zone_id']} | touches {active_zone['touches']} | RR {plan['rr']:.2f}",
            "indicators": {
                "atr14_m15": round(atr_m15, 4),
                "spread": round(float(tick.get("spread", 0.0) or 0.0), 4),
            },
            "_active_zone": active_zone,
            "_signal_candle_time": confirmation_candle["datetime"].isoformat(),
            "_comment": f"FT_{self._POSITION_TAG}_{active_zone['zone_id'][:8]}",
            "_setup_direction": "LONG" if direction == "BUY" else "SHORT",
            "_quality_score": confidence,
            "_threshold": 0.55,
            "_signal_family": "M15",
            "_sweep_confirmed": bool(plan.get("_sweep_reclaim")),
            "_candle_confirmation": True,
            "_exit_profile": "swing_structured",
            "_entry_spread": float(tick.get("spread", 0.0) or 0.0),
            "_entry_tick_pressure_score": float(tick_pressure.get("pressure_score", 0.0) or 0.0),
            "_entry_tick_pressure_bias": str(tick_pressure.get("directional_bias", "NEUTRAL") or "NEUTRAL"),
            "_be_trigger": 0.30,
            "_be_trigger_r": cfg.EXIT_PROFILE_M15_BE_TRIGGER_R,
            "_timeout": 300,
            "_timeout_min_progress_r": cfg.EXIT_PROFILE_M15_TIMEOUT_MIN_PROGRESS_R,
            "_min_hold_seconds": cfg.EXIT_PROFILE_M15_MIN_HOLD_SECONDS,
            "_early_fail": max(0.15, round(atr_m15 * 0.20, 3)),
            "_tier1_min_ticks": cfg.SMC_TIER1_MIN_TICKS,
            "_tier1_max_ticks": cfg.SMC_TIER1_MAX_TICKS,
            "_reversal_arm_r": cfg.EXIT_PROFILE_M15_REVERSAL_ARM_R,
            "_reversal_drawdown_pct": cfg.EXIT_PROFILE_M15_REVERSAL_DRAWDOWN_PCT,
            "_reversal_floor_r": cfg.EXIT_PROFILE_M15_REVERSAL_FLOOR_R,
            "_trail_activate_r": cfg.EXIT_PROFILE_M15_TRAIL_ACTIVATE_R,
            "_trail_lock_r": cfg.EXIT_PROFILE_M15_TRAIL_LOCK_R,
            "_velocity_drop_enabled": cfg.EXIT_PROFILE_M15_VELOCITY_DROP_ENABLED,
        }
        signal.update(plan)
        return signal

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
    ) -> Optional[Dict]:
        entry = round(float(tick["ask"] if direction == "BUY" else tick["bid"]), 2)
        if direction == "BUY":
            sl_anchor = float(zone["zone_low"])
            if sweep_context and sweep_context.get("direction") == "BUY":
                sl_anchor = min(sl_anchor, float(sweep_context.get("extreme", sl_anchor)))
            sl = round(sl_anchor - atr_m15 * float(cfg.M15_SR_SL_BUFFER_ATR), 2)
            risk = entry - sl
            preferred = next((z for z in opposite_zones if z["zone_mid"] > entry), None)
            preferred_tp = round(float(preferred["zone_mid"]), 2) if preferred else None
        else:
            sl_anchor = float(zone["zone_high"])
            if sweep_context and sweep_context.get("direction") == "SELL":
                sl_anchor = max(sl_anchor, float(sweep_context.get("extreme", sl_anchor)))
            sl = round(sl_anchor + atr_m15 * float(cfg.M15_SR_SL_BUFFER_ATR), 2)
            risk = sl - entry
            preferred = next((z for z in opposite_zones if z["zone_mid"] < entry), None)
            preferred_tp = round(float(preferred["zone_mid"]), 2) if preferred else None

        if risk <= 0:
            return None

        tp = preferred_tp
        rr = 0.0
        target_source = "fixed_rr"
        if tp is not None:
            reward = (tp - entry) if direction == "BUY" else (entry - tp)
            rr = reward / risk if risk > 0 else 0.0
            if rr >= float(cfg.M15_SR_MIN_RR):
                target_source = "opposite_zone"
            else:
                tp = None

        if tp is None:
            rr = float(cfg.M15_SR_DEFAULT_RR)
            tp = round(entry + (risk * rr), 2) if direction == "BUY" else round(entry - (risk * rr), 2)

        reward = (tp - entry) if direction == "BUY" else (entry - tp)
        rr = reward / risk if risk > 0 else 0.0
        if rr < float(cfg.M15_SR_MIN_RR):
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
        # Only choppy if touches occur on different candles (not a single wide-range candle spanning both zones)
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
        }
        if signal_data:
            additional["signal_data"] = {
                "entry": signal_data.get("entry"),
                "sl": signal_data.get("sl"),
                "tp": signal_data.get("tp"),
                "rr": signal_data.get("rr"),
                "lot_size": signal_data.get("lot_size"),
                "zone_id": (signal_data.get("_active_zone") or {}).get("zone_id"),
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
    ) -> Dict:
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
        )
        payload = self._no(reason)
        if terminal:
            payload["_terminal"] = True
        return payload

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
        # Only inspect the 3 most recent closes to avoid flagging historically-straddled zones
        recent = closes[-3:]
        above = any(float(c) > zone_high for c in recent)
        below = any(float(c) < zone_low for c in recent)
        if not (above and below):
            return False
        # If the last close is cleanly on the expected side, the zone still has directional bias
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

    def _confidence_score(self, zone: Dict, candle: pd.Series, plan: Dict) -> float:
        stats = self._wick_stats(candle)
        wick_bonus = min(0.08, max(stats["upper_wick"], stats["lower_wick"]) / max(stats["range"], 0.0001) * 0.08)
        touch_bonus = min(0.12, max(0, int(zone.get("touches", 0)) - int(cfg.M15_SR_MIN_TOUCHES)) * 0.04)
        rr_bonus = min(0.10, max(0.0, float(plan.get("rr", 0.0)) - float(cfg.M15_SR_MIN_RR)) * 0.08)
        return round(min(0.90, 0.58 + wick_bonus + touch_bonus + rr_bonus), 3)

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
        candle_time = M15SupportResistanceStrategy._coerce_dt(closed.iloc[-1]["datetime"])
        return candle_time or datetime.now(timezone.utc)

    @staticmethod
    def _atr_m15(closed: pd.DataFrame) -> float:
        highs = closed["high"].astype(float).to_numpy()
        lows = closed["low"].astype(float).to_numpy()
        closes = closed["close"].astype(float).to_numpy()
        return round(float(atr(highs, lows, closes, 14)[-1]), 4)

    @staticmethod
    def _no(reason: str) -> Dict:
        return {"signal": "NO_TRADE", "reason": reason, "score": 0.0}
