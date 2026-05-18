"""
M15 zone scalp inverse — exact opposite execution of M15_ZONE_SCALP.

Uses the same setup detection and micro-bias resolution as M15_ZONE_SCALP.
It only activates when the shared resolver says the opposite side has the
stronger real-time entry bias.
"""
from __future__ import annotations

from typing import Dict

from engine.m15_zone_scalp_strategy import (
    M15ZoneScalpStrategy,
    _safe_float,
)
from engine.m15_zone_micro_bias import (
    build_m15_zone_decision_reason,
    mirror_zone_signal,
)


class M15ZoneScalpInverseStrategy(M15ZoneScalpStrategy):
    name = "M15_ZONE_SCALP_INVERSE"

    def generate_signal(self, data: Dict) -> Dict:
        candidates, blocked = self._family_candidates(data)
        if blocked is not None:
            return blocked

        best_signal = None
        best_weight = float("-inf")
        best_blocked = None
        best_block_weight = float("-inf")

        for candidate in candidates:
            resolver = self._resolve_micro_bias(data, candidate)
            candidate_weight = abs(_safe_float(resolver.get("micro_bias_score"))) + float(candidate.get("_zone_strength", 0.0)) * 20.0
            expected_inverse_dir = "SELL" if str(candidate.get("signal") or "").upper() == "BUY" else "BUY"
            if (
                resolver.get("recommended_action") == "ROUTE_TO_INVERSE"
                and resolver.get("final_direction") == expected_inverse_dir
            ):
                routed = mirror_zone_signal(candidate, data, self.name)
                routed = self._attach_micro_bias(routed, resolver)
                routed["_route_type"] = "ROUTED_TO_INVERSE"
                routed["reason"] = build_m15_zone_decision_reason("M15 zone scalp inverse", candidate.get("signal"), resolver, routed)
                if candidate_weight > best_weight:
                    best_signal = routed
                    best_weight = candidate_weight
            else:
                blocked_sig = self._no(
                    build_m15_zone_decision_reason("M15 zone scalp inverse", candidate.get("signal"), resolver, candidate),
                    _micro_bias_direction=resolver.get("micro_bias_direction"),
                    _micro_bias_score=resolver.get("micro_bias_score"),
                    _micro_bias_confidence=resolver.get("micro_bias_confidence"),
                    _micro_bias_action=resolver.get("recommended_action"),
                    _micro_bias_reasons=list(resolver.get("reasons") or []),
                    _pressure_bias=resolver.get("pressure_bias"),
                    _pressure_score=resolver.get("pressure_score"),
                    _route_type="BLOCKED_BASE_WON" if resolver.get("recommended_action") == "ALLOW_BASE_TRADE" else "BLOCKED_NEUTRAL_OR_WEAK",
                    _base_setup_direction="LONG" if candidate.get("signal") == "BUY" else "SHORT",
                    _zone_type=candidate.get("_zone_type"),
                )
                if candidate_weight > best_block_weight:
                    best_blocked = blocked_sig
                    best_block_weight = candidate_weight

        return best_signal or best_blocked or self._no("No actionable inverse M15 zone family candidate")
