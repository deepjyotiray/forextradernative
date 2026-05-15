"""
Strategy Manager — registry, hot-switch, and AUTO arbitration.

Modes:
  Single strategy: only the active strategy runs
  Multi-select: only the selected strategies are eligible, best signal wins
  AUTO: all registered strategies are eligible, best signal wins
"""
from typing import Any, Dict, Iterable, List, Optional
import time
from engine.strategies.base_strategy import BaseStrategy
from engine.master_control import pre_trade_validation, log_trade_decision_comprehensive
from engine.master_trade_gate import master_trade_gate
from engine import strategy_configs
from engine.decision_logger import log_decision

_NO_TRADE_LOG_INTERVAL = 60.0  # seconds between logging same NO_TRADE reason
_no_trade_log_times: Dict[str, float] = {}

_AUTO = "AUTO"
_SIGNAL_FAMILY_BY_STRATEGY = {
    "SMC_CONFLUENCE": "SMC",
    "SWEEP_SCALPER": "SWEEP",
    "M15_SUPPORT_RESISTANCE_REJECTION_V1": "M15",
    "M15_SCALP_DEEP": "M15",
    "M15_ZONE_SCALP": "M15",
    "TREND_CHANNEL": "TREND",
}



def _normalize_signal(sig, strategy_name: str) -> Dict:
    if isinstance(sig, dict):
        normalized = dict(sig)
        normalized.setdefault("_strategy_name", strategy_name)
        family = normalized.get("_signal_family") or _SIGNAL_FAMILY_BY_STRATEGY.get(strategy_name)
        if family:
            normalized["_signal_family"] = family

        action = str(normalized.get("signal") or "").upper()
        if action in ("BUY", "SELL"):
            if family == "SWEEP":
                normalized.setdefault("_sweep_confirmed", True)
                normalized.setdefault("_candle_confirmation", True)
            elif family == "M15":
                normalized.setdefault("_sweep_confirmed", bool(normalized.get("_sweep_reclaim")))
                normalized.setdefault("_candle_confirmation", True)
        return normalized
    if sig is None:
        reason = f"{strategy_name} returned no signal payload"
    else:
        reason = f"{strategy_name} returned invalid signal type: {type(sig).__name__}"
    return {"signal": "NO_TRADE", "reason": reason}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _resolve_trade_direction(signal: Dict, fallback: str = "NEUTRAL") -> str:
    setup_direction = str(signal.get("_setup_direction") or "").upper()
    if setup_direction in ("LONG", "SHORT"):
        return setup_direction
    action = str(signal.get("signal") or "").upper()
    if action == "BUY":
        return "LONG"
    if action == "SELL":
        return "SHORT"
    return fallback


def _resolve_bias_direction(signal: Dict, market_data: Dict) -> str:
    bias = signal.get("bias") or market_data.get("bias") or {}
    direction = str(bias.get("direction") or signal.get("_bias_direction") or "").upper()
    if direction == "BUY":
        return "LONG"
    if direction == "SELL":
        return "SHORT"
    return direction or "NEUTRAL"


def _pressure_bonus(trade_dir: str, pressure_score: float) -> float:
    if trade_dir == "LONG":
        if pressure_score > 0:
            return min(8.0, pressure_score * 12.0)
        if pressure_score < 0:
            return max(-8.0, pressure_score * 12.0)
    elif trade_dir == "SHORT":
        if pressure_score < 0:
            return min(8.0, abs(pressure_score) * 12.0)
        if pressure_score > 0:
            return max(-8.0, -pressure_score * 12.0)
    return 0.0


def _build_auto_market_state(data: Dict) -> Dict:
    market_state = dict(data or {})
    market_state.setdefault("bias", data.get("bias") or {})
    market_state.setdefault("regime", data.get("regime") or {})
    market_state.setdefault("tick_snapshot", data.get("tick_snapshot") or {})
    market_state.setdefault("tick_pressure", data.get("tick_pressure") or {})
    return market_state


def _summarize_auto_candidate(signal: Dict, strategy_name: str, gate_result: Optional[Dict], market_data: Dict) -> Dict:
    trade_dir = _resolve_trade_direction(signal)
    bias_dir = _resolve_bias_direction(signal, market_data)
    confidence = _safe_float(signal.get("confidence"), 0.0)
    rr = _safe_float(signal.get("rr"), 0.0)
    volume_ratio = _safe_float(signal.get("_entry_volume_ratio"), 0.0)
    pressure_score = _safe_float(
        signal.get("_entry_tick_pressure_score"),
        _safe_float((market_data.get("tick_pressure") or {}).get("pressure_score"), 0.0),
    )
    gate_allowed = bool((gate_result or {}).get("allowed", False))
    gate_reason = str((gate_result or {}).get("reason") or signal.get("reason") or "")
    gate_score = _safe_int((gate_result or {}).get("score"), 0)
    confirmations = _safe_int((gate_result or {}).get("confirmation_count"), 0)
    counter_trend = bool((gate_result or {}).get("counter_trend"))

    arbitration_score = float(gate_score)
    arbitration_score += min(max(confidence, 0.0), 1.0) * 20.0
    arbitration_score += min(max(rr, 0.0), 3.0) * 4.0
    arbitration_score += min(max(confirmations, 0), 4) * 5.0
    arbitration_score += max(0.0, min(volume_ratio - 1.0, 1.0)) * 10.0
    arbitration_score += _pressure_bonus(trade_dir, pressure_score)
    if trade_dir in ("LONG", "SHORT"):
        if bias_dir == trade_dir:
            arbitration_score += 8.0
        elif counter_trend:
            arbitration_score -= 6.0

    return {
        "strategy": strategy_name,
        "signal": signal,
        "trade_dir": trade_dir,
        "bias_dir": bias_dir,
        "gate_allowed": gate_allowed,
        "gate_reason": gate_reason,
        "gate_score": gate_score,
        "confirmation_count": confirmations,
        "counter_trend": counter_trend,
        "confidence": round(confidence, 4),
        "rr": round(rr, 2),
        "volume_ratio": round(volume_ratio, 3),
        "pressure_score": round(pressure_score, 3),
        "arbitration_score": round(arbitration_score, 3),
    }


def _candidate_sort_key(candidate: Dict) -> tuple:
    return (
        _safe_float(candidate.get("arbitration_score"), 0.0),
        _safe_int(candidate.get("gate_score"), 0),
        _safe_float(candidate.get("confidence"), 0.0),
        _safe_float(candidate.get("rr"), 0.0),
    )


class StrategyManager:
    def __init__(self):
        self._strategies: Dict[str, BaseStrategy] = {}
        self._active: str = ""
        self._selected: List[str] = []
        self._selected_tokens: List[str] = []
        self._auto: bool = True

    def register(self, strategy: BaseStrategy):
        self._strategies[strategy.name] = strategy
        if not self._selected:
            self._selected = [strategy.name]
            self._selected_tokens = [strategy.name]
            self._active = strategy.name

    def _expand_token(self, token: str) -> Optional[List[str]]:
        name = str(token or "").strip().upper()
        if not name:
            return []
        if name in self._strategies:
            return [name]
        return None

    def set_active(self, name: str) -> bool:
        name = name.upper()
        if name == _AUTO:
            self._auto = True
            self._selected = list(self._strategies.keys())
            self._selected_tokens = list(self._strategies.keys())
            self._active = _AUTO
            return True
        expanded = self._expand_token(name)
        if expanded is None or not expanded:
            return False
        self._auto = False
        self._selected = expanded
        self._selected_tokens = [name]
        self._active = name
        return True

    def set_selection(self, names: Iterable[str]) -> bool:
        requested: List[str] = []
        requested_tokens: List[str] = []
        auto_requested = False
        for raw in names or []:
            name = str(raw or "").strip().upper()
            if not name:
                continue
            if name == _AUTO:
                auto_requested = True
                continue
            expanded = self._expand_token(name)
            if expanded is None or not expanded:
                return False
            if name not in requested_tokens:
                requested_tokens.append(name)
            for strategy_name in expanded:
                if strategy_name not in requested:
                    requested.append(strategy_name)
        if auto_requested or not requested:
            return self.set_active(_AUTO)
        self._auto = False
        self._selected = requested
        self._selected_tokens = requested_tokens
        self._active = ",".join(requested_tokens)
        return True

    @property
    def active_name(self) -> str:
        return _AUTO if self._auto else ",".join(self._selected_tokens)

    @property
    def is_auto(self) -> bool:
        return self._auto

    @property
    def active(self) -> Optional[BaseStrategy]:
        if self._auto or len(self._selected) != 1 or len(self._selected_tokens) != 1:
            return None
        return self._strategies.get(self._selected[0])

    @property
    def selected(self) -> List[str]:
        return list(self._selected_tokens if not self._auto else self._strategies.keys())

    @property
    def available(self) -> List[str]:
        return [_AUTO] + list(self._strategies.keys())

    def get(self, name: str) -> Optional[BaseStrategy]:
        return self._strategies.get(name)

    def enabled_strategies(self) -> List[str]:
        return list(self._strategies.keys()) if self._auto else list(self._selected)

    def evaluate_all(self, data: Dict, strategy_names: Optional[Iterable[str]] = None) -> Dict:
        """
        Run ALL strategies, return the best actionable signal.
        Returns: {signal_dict, strategy_name, all_results}
        """
        results = {}
        candidates = []
        market_state = _build_auto_market_state(data)
        risk_manager = data.get("_risk_manager") or data.get("risk_manager")
        m15_context_provider = (
            data.get("_m15_context_provider")
            or data.get("m15_context_provider")
            or self.get("M15_SUPPORT_RESISTANCE_REJECTION_V1")
        )

        enabled = list(strategy_names or self._strategies.keys())
        tick = data.get("tick") or {}
        _log_price = _safe_float(tick.get("bid"), 0.0)
        for name in enabled:
            strat = self._strategies[name]
            # Per-strategy enabled check
            try:
                scfg = strategy_configs.get(name)
                if not scfg.get("enabled", True):
                    results[name] = {"signal": "NO_TRADE", "reason": "DISABLED", "confidence": 0.0}
                    continue
            except Exception:
                pass
            try:
                sig = _normalize_signal(strat.generate_signal(data), name)
            except Exception as e:
                sig = {"signal": "NO_TRADE", "reason": str(e)}

            action = sig.get("signal", "NO_TRADE")
            row = {
                "signal": action,
                "confidence": _safe_float(sig.get("confidence"), 0.0),
                "reason": str(sig.get("reason", ""))[:120],
                "price": _safe_float(sig.get("entry"), 0.0) or None,
            }
            # Log NO_TRADE decisions so all strategies have visibility (throttled)
            if action == "NO_TRADE":
                _nt_key = f"{name}:{str(sig.get('reason',''))[:80]}"
                _nt_now = time.time()
                if _nt_now - _no_trade_log_times.get(_nt_key, 0.0) >= _NO_TRADE_LOG_INTERVAL:
                    _no_trade_log_times[_nt_key] = _nt_now
                    log_decision(
                        strategy=name,
                        setup_direction=None,
                        bias_direction=None,
                        quality_score=0.0,
                        threshold=0.0,
                        spread_mean=0.0,
                        spread_std=0.0,
                        compression_ok=True,
                        ltf_conflict=False,
                        decision="TRADE_SKIPPED",
                        reason=str(sig.get("reason", "No signal"))[:200],
                        price=_log_price,
                    )
            if action in ("BUY", "SELL"):
                try:
                    gate_result = master_trade_gate(
                        sig,
                        market_state,
                        risk_manager=risk_manager,
                        m15_context_provider=m15_context_provider,
                    )
                except Exception as e:
                    gate_result = {
                        "allowed": False,
                        "reason": f"AUTO_GATE_ERR: {e}",
                        "score": 0,
                        "confirmation_count": 0,
                        "counter_trend": False,
                    }
                sig["_master_gate_preview"] = gate_result
                candidate = _summarize_auto_candidate(sig, name, gate_result, market_state)
                candidates.append(candidate)
                row.update(
                    {
                        "gate_allowed": candidate["gate_allowed"],
                        "gate_reason": candidate["gate_reason"],
                        "gate_score": candidate["gate_score"],
                        "confirmation_count": candidate["confirmation_count"],
                        "counter_trend": candidate["counter_trend"],
                        "rr": candidate["rr"],
                        "volume_ratio": candidate["volume_ratio"],
                        "pressure_score": candidate["pressure_score"],
                        "arb_score": candidate["arbitration_score"],
                    }
                )
                if candidate["gate_allowed"]:
                    row["reason"] = str(sig.get("reason", ""))[:120]
                else:
                    row["reason"] = candidate["gate_reason"][:120]
            results[name] = row

        allowed_candidates = [candidate for candidate in candidates if candidate.get("gate_allowed")]
        if allowed_candidates:
            winner = max(allowed_candidates, key=_candidate_sort_key)
            best_sig = dict(winner["signal"])
            best_name = str(winner["strategy"])
            best_sig["_arb_log"] = {
                "strategy": best_name,
                "setup_dir": winner["trade_dir"],
                "bias_dir": winner["bias_dir"],
                "quality": winner["confidence"],
                "threshold": 0.0,
                "counter": winner["counter_trend"],
                "confirmation_count": winner["confirmation_count"],
                "gate_score": winner["gate_score"],
                "arb_score": winner["arbitration_score"],
                "volume_ratio": winner["volume_ratio"],
                "pressure_score": winner["pressure_score"],
            }
            best_sig["_master_gate_preview"] = best_sig.get("_master_gate_preview") or winner["signal"].get("_master_gate_preview")
            selected_signal = best_sig
            selected_strategy = best_name
        elif candidates:
            best_blocked = max(candidates, key=_candidate_sort_key)
            selected_signal = {
                "signal": "NO_TRADE",
                "reason": best_blocked.get("gate_reason") or "No strategy passed arbitration gate",
                "_auto_blocked_candidate": best_blocked.get("strategy"),
            }
            selected_strategy = str(best_blocked.get("strategy") or "AUTO")
        else:
            selected_signal = {"signal": "NO_TRADE", "reason": "No strategy produced a signal"}
            selected_strategy = ",".join(self._selected_tokens) if self._selected_tokens else _AUTO

        return {
            "signal": selected_signal,
            "strategy": selected_strategy,
            "all_results": results,
        }

    def generate_signal(self, data: Dict) -> Dict:
        """AUTO mode: pick the single best signal. Non-AUTO: use generate_signals for independent evaluation."""
        if self._auto:
            evaluation = self.evaluate_all(data, strategy_names=self.enabled_strategies())
            strat_name = str(evaluation.get("strategy") or _AUTO)
            sig = dict(evaluation.get("signal") or {"signal": "NO_TRADE", "reason": "Empty signal"})
            skip_global = bool(sig.get("_skip_global_filters"))
            trade_id = None

            if not skip_global and sig.get("signal") in ("BUY", "SELL"):
                allowed, validation_result = pre_trade_validation(sig, strat_name, data)
                if allowed:
                    trade_id = log_trade_decision_comprehensive(
                        strategy=strat_name,
                        signal=sig,
                        market_data=data,
                        decision="TRADE_TAKEN",
                        reason=sig.get("reason", "Signal generated"),
                        validation_result=validation_result,
                    )
                    sig["_trade_id"] = trade_id
                    sig["_validation_result"] = validation_result
                else:
                    skip_reasons = [block["reason"] for block in validation_result.get("blocks", [])]
                    skip_reason = " | ".join(skip_reasons)
                    log_trade_decision_comprehensive(
                        strategy=strat_name,
                        signal=sig,
                        market_data=data,
                        decision="TRADE_SKIPPED",
                        reason=f"Validation failed: {skip_reason}",
                        validation_result=validation_result,
                    )
                    sig = {
                        "signal": "NO_TRADE",
                        "reason": f"Blocked by validation: {skip_reason}",
                        "_original_signal": sig,
                        "_validation_result": validation_result,
                        "_strategy_name": strat_name,
                    }
            elif not skip_global:
                log_trade_decision_comprehensive(
                    strategy=strat_name,
                    signal=sig,
                    market_data=data,
                    decision="TRADE_SKIPPED",
                    reason=sig.get("reason", "No signal generated"),
                )

            return {"strategy": strat_name, "signal": sig, "trade_id": trade_id}

        # Non-AUTO: return first valid signal; caller should use generate_signals for all.
        results = self.generate_signals(data)
        if results:
            return results[0]
        return {
            "strategy": self._selected[0] if self._selected else _AUTO,
            "signal": {"signal": "NO_TRADE", "reason": "No strategy produced a signal"},
            "trade_id": None,
        }

    def generate_signals(self, data: Dict) -> List[Dict]:
        results: List[Dict] = []
        for name in self.enabled_strategies():
            strat = self._strategies[name]
            try:
                sig = _normalize_signal(strat.generate_signal(data), strat.name)
            except Exception as e:
                sig = {"signal": "NO_TRADE", "reason": str(e), "_strategy_name": name}

            skip_global = bool(sig.get("_skip_global_filters"))
            trade_id = None

            if not skip_global and sig.get("signal") in ("BUY", "SELL"):
                allowed, validation_result = pre_trade_validation(sig, name, data)
                if allowed:
                    trade_id = log_trade_decision_comprehensive(
                        strategy=name,
                        signal=sig,
                        market_data=data,
                        decision="TRADE_TAKEN",
                        reason=sig.get("reason", "Signal generated"),
                        validation_result=validation_result,
                    )
                    sig["_trade_id"] = trade_id
                    sig["_validation_result"] = validation_result
                else:
                    skip_reasons = [block["reason"] for block in validation_result.get("blocks", [])]
                    skip_reason = " | ".join(skip_reasons)
                    log_trade_decision_comprehensive(
                        strategy=name,
                        signal=sig,
                        market_data=data,
                        decision="TRADE_SKIPPED",
                        reason=f"Validation failed: {skip_reason}",
                        validation_result=validation_result,
                    )
                    sig = {
                        "signal": "NO_TRADE",
                        "reason": f"Blocked by validation: {skip_reason}",
                        "_original_signal": sig,
                        "_validation_result": validation_result,
                        "_strategy_name": name,
                    }
            elif not skip_global:
                log_trade_decision_comprehensive(
                    strategy=name,
                    signal=sig,
                    market_data=data,
                    decision="TRADE_SKIPPED",
                    reason=sig.get("reason", "No signal generated"),
                )

            results.append({"strategy": name, "signal": sig, "trade_id": trade_id})
        return results

    def status(self) -> Dict:
        return {
            "active": self.active_name,
            "available": [_AUTO] + list(self._strategies.keys()),
            "is_auto": self.is_auto,
            "selected": self.selected,
            "enabled": self.enabled_strategies(),
        }
