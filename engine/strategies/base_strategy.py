"""
Base Strategy — all strategies implement this interface.
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional


class BaseStrategy(ABC):
    name: str = "BASE"

    @abstractmethod
    def generate_signal(self, data: Dict) -> Dict:
        """
        Generate a trading signal.

        Args:
            data: {
                "m1_df": DataFrame, "m5_df": DataFrame, "m15_df": DataFrame,
                "h1_df": DataFrame, "h4_df": DataFrame, "d1_df": DataFrame,
                "tick": {"bid", "ask", "spread"},
                "zones": {"support": [...], "resistance": [...]},
                "indicators": {M1 indicators dict},
                "account": {balance, equity, ...},
                "positions": [list of open positions],
            }

        Returns:
            {
                "signal": "BUY" | "SELL" | "NO_TRADE",
                "sl": float or None,
                "tp": float or None,
                "confidence": float 0-1,
                "reason": str,
                "sl_distance": float,
                "entry": float,
                "indicators": dict,
            }
        """
        ...
