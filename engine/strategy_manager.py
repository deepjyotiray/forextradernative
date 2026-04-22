"""
Strategy Manager — registry, hot-switch, and AUTO mode.

Modes:
  Single strategy: only the active strategy runs
  AUTO: all strategies run every cycle, best signal wins
"""
from typing import Dict, List, Optional
from engine.strategies.base_strategy import BaseStrategy

_AUTO = "AUTO"


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
                sig = strat.generate_signal(data)
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
        Main entry point. Returns (signal_dict, strategy_name).
        In AUTO mode, evaluates all and picks best.
        In single mode, runs only the active strategy.
        """
        if self._active == _AUTO:
            result = self.evaluate_all(data)
            sig = result["signal"]
            name = result["strategy"] or "AUTO"
            # Tag the signal with auto-selection metadata
            if sig.get("signal") in ("BUY", "SELL"):
                sig["_auto_selected"] = True
                sig["_auto_results"] = result["all_results"]
            return sig, name
        else:
            strat = self._strategies[self._active]
            sig = strat.generate_signal(data)
            return sig, strat.name

    def status(self) -> Dict:
        return {
            "active": self._active,
            "available": self.available,
            "is_auto": self.is_auto,
        }
