"""
M15 zone scalp inverse — exact opposite execution of M15_ZONE_SCALP.

Uses the same setup detection as M15_ZONE_SCALP, but flips the final trade
direction and mirrors the stop/target so the trade thesis is the opposite of
the base strategy's thesis.
"""
from __future__ import annotations

from typing import Dict

from engine import strategy_configs as _scfg
from engine.m15_zone_scalp_strategy import (
    M15ZoneScalpStrategy,
    _calc_scalp_lot,
    _safe_float,
)


class M15ZoneScalpInverseStrategy(M15ZoneScalpStrategy):
    name = "M15_ZONE_SCALP_INVERSE"

    def generate_signal(self, data: Dict) -> Dict:
        source = super().generate_signal(data)
        source_signal = str(source.get("signal") or "").upper()
        if source_signal not in ("BUY", "SELL"):
            return source

        tick = data.get("tick") or {}
        bid = _safe_float(tick.get("bid"))
        ask = _safe_float(tick.get("ask") or tick.get("bid"))
        if bid <= 0:
            return self._no("No tick data")

        direction = "SELL" if source_signal == "BUY" else "BUY"
        entry = bid if direction == "SELL" else (ask if ask > 0 else bid)
        sl = round(_safe_float(source.get("tp")), 2)
        tp = round(_safe_float(source.get("sl")), 2)

        if direction == "BUY":
            if not (sl < entry < tp):
                return self._no("Inverse BUY produced invalid levels")
        else:
            if not (tp < entry < sl):
                return self._no("Inverse SELL produced invalid levels")

        sl_dist = abs(entry - sl)
        tp_dist = abs(tp - entry)
        rr = tp_dist / max(sl_dist, 1e-6)

        s_cfg = _scfg.get(self.name)
        balance = _safe_float((data.get("account") or {}).get("balance"))
        if balance <= 0:
            return self._no("Account balance unavailable")
        lot = _calc_scalp_lot(balance, s_cfg, sl_dist)

        confidence = min(0.99, _safe_float(source.get("confidence"), 0.0))
        pip = max(0.01, _safe_float(source.get("_pip_size"), _safe_float(s_cfg.get("pip_size"), 0.1)))
        setup_direction = "LONG" if direction == "BUY" else "SHORT"
        source_setup = "LONG" if source_signal == "BUY" else "SHORT"
        source_reason = str(source.get("reason") or "").strip()
        reason = f"{direction} M15 zone scalp inverse | opposite of {source_signal} setup | {source_reason}"
        tp_levels = [tp]

        return {
            "signal": direction,
            "entry": round(entry, 2),
            "sl": sl,
            "tp": tp,
            "tp_levels": tp_levels,
            "sl_distance": round(sl_dist, 2),
            "lot": round(lot, 2),
            "confidence": confidence,
            "confidence_pct": int(round(confidence * 100)),
            "reason": reason,
            "rr": round(rr, 2),
            "_strategy_name": self.name,
            "strategy": self.name,
            "decision": direction,
            "_signal_family": "M15",
            "_setup_direction": setup_direction,
            "_bias_direction": str(source.get("_bias_direction") or source_setup).upper(),
            "_source_strategy_name": "M15_ZONE_SCALP",
            "_source_signal": source_signal,
            "_source_setup_direction": source_setup,
            "_sweep_confirmed": bool(source.get("_sweep_confirmed", False)),
            "_m15_zone_confirmed": bool(source.get("_m15_zone_confirmed", True)),
            "_candle_confirmation": bool(source.get("_candle_confirmation", True)),
            "_body_ratio": round(_safe_float(source.get("_body_ratio")), 3),
            "_exit_profile": str(source.get("_exit_profile") or "m15_zone_scalp"),
            "_scalp": bool(source.get("_scalp", True)),
            "_tp_levels": tp_levels,
            "_zone_mid": _safe_float(source.get("_zone_mid")),
            "_tp_pips": round(tp_dist / pip, 2),
            "_pip_size": pip,
            "_session": source.get("_session"),
        }
