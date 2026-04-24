"""
Strategy Manager — registry, hot-switch, and AUTO mode with integrated control systems.

Modes:
  Single strategy: only the active strategy runs
  AUTO: all strategies run every cycle, best signal wins
  
Integrated with:
  - Master Control System for validation and logging
  - Risk controls and trade pacing
  - Parameter calibration and over-filtering detection
"""
from typing import Dict, List, Optional
from engine.strategies.base_strategy import BaseStrategy
from engine.master_control import pre_trade_validation, log_trade_decision_comprehensive

_AUTO = "AUTO"


def _normalize_signal(sig, strategy_name: str) -> Dict:
    if isinstance(sig, dict):
        return sig
    if sig is None:
        reason = f"{strategy_name} returned no signal payload"
    else:
        reason = f"{strategy_name} returned invalid signal type: {type(sig).__name__}"
    return {"signal": "NO_TRADE", "reason": reason}


class StrategyManager:
    def __init__(self):
        self._strategies: Dict[str, BaseStrategy] = {}
        self._active: str = ""

    def register(self, strategy: BaseStrategy):
        self._strategies[strategy.name] = strategy
        if not self._active:
            self._active = strategy.name

    def set_active(self, name: str) -> bool:
        name = name.upper()
        if name == _AUTO:
            self._active = _AUTO
            return True
        if name not in self._strategies:
            return False
        self._active = name
        return True

    @property
    def active_name(self) -> str:
        return self._active

    @property
    def is_auto(self) -> bool:
        return self._active == _AUTO

    @property
    def active(self) -> Optional[BaseStrategy]:
        if self._active == _AUTO:
            return None
        return self._strategies.get(self._active)

    @property
    def available(self) -> List[str]:
        return [_AUTO] + list(self._strategies.keys())

    def get(self, name: str) -> Optional[BaseStrategy]:
        return self._strategies.get(name)

    def evaluate_all(self, data: Dict) -> Dict:
        """
        Run ALL strategies, return the best actionable signal.
        Returns: {signal_dict, strategy_name, all_results}
        """
        results = {}
        best_sig = None
        best_name = None
        best_conf = -1

        for name, strat in self._strategies.items():
            try:
                sig = _normalize_signal(strat.generate_signal(data), name)
            except Exception as e:
                sig = {"signal": "NO_TRADE", "reason": str(e)}
            results[name] = sig

            action = sig.get("signal", "NO_TRADE")
            if action not in ("BUY", "SELL"):
                continue

            conf = sig.get("confidence", 0)
            if conf > best_conf:
                best_conf = conf
                best_sig = sig
                best_name = name

        return {
            "signal": best_sig or {"signal": "NO_TRADE", "reason": "No strategy produced a signal"},
            "strategy": best_name,
            "all_results": {n: {"signal": s.get("signal", "NO_TRADE"),
                                "confidence": s.get("confidence", 0),
                                "reason": s.get("reason", "")[:80]}
                           for n, s in results.items()},
        }

    def generate_signal(self, data: Dict) -> tuple:
        """
        Main entry point with integrated control systems.
        Returns (signal_dict, strategy_name, trade_id)
        """
        if self._active == _AUTO:
            result = self.evaluate_all(data)
            sig = result["signal"]
            name = result["strategy"] or "AUTO"
            # Tag the signal with auto-selection metadata
            if sig.get("signal") in ("BUY", "SELL"):
                sig["_auto_selected"] = True
                sig["_auto_results"] = result["all_results"]
        else:
            strat = self._strategies[self._active]
            sig = _normalize_signal(strat.generate_signal(data), strat.name)
            name = strat.name
        
        trade_id = None
        
        # Apply comprehensive validation and logging
        if sig.get("signal") in ("BUY", "SELL"):
            # Pre-trade validation
            allowed, validation_result = pre_trade_validation(sig, name, data)
            
            if allowed:
                # Log trade taken
                trade_id = log_trade_decision_comprehensive(
                    strategy=name,
                    signal=sig,
                    market_data=data,
                    decision="TRADE_TAKEN",
                    reason=sig.get("reason", "Signal generated"),
                    validation_result=validation_result
                )
                
                # Add trade management info
                sig["_trade_id"] = trade_id
                sig["_validation_result"] = validation_result
                
            else:
                # Log trade skipped
                skip_reasons = [block["reason"] for block in validation_result.get("blocks", [])]
                skip_reason = " | ".join(skip_reasons)
                
                log_trade_decision_comprehensive(
                    strategy=name,
                    signal=sig,
                    market_data=data,
                    decision="TRADE_SKIPPED",
                    reason=f"Validation failed: {skip_reason}",
                    validation_result=validation_result
                )
                
                # Convert to NO_TRADE
                sig = {
                    "signal": "NO_TRADE",
                    "reason": f"Blocked by validation: {skip_reason}",
                    "_original_signal": sig,
                    "_validation_result": validation_result
                }
        
        elif sig.get("signal") == "NO_TRADE":
            # Log decision skipped
            log_trade_decision_comprehensive(
                strategy=name,
                signal=sig,
                market_data=data,
                decision="TRADE_SKIPPED",
                reason=sig.get("reason", "No signal generated")
            )
        
        return sig, name, trade_id

    def status(self) -> Dict:
        return {
            "active": self._active,
            "available": self.available,
            "is_auto": self.is_auto,
        }
