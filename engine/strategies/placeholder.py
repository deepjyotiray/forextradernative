"""
Placeholder Strategy — returns NO_TRADE until user implements their own.
Replace the generate_signal method with your trading logic.
"""
from typing import Dict
from .base_strategy import BaseStrategy


class PlaceholderStrategy(BaseStrategy):
    name = "PLACEHOLDER"

    def generate_signal(self, data: Dict) -> Dict:
        """
        Replace this with your strategy logic.

        Available in `data`:
            data["m1_df"]       — M1 candles DataFrame (up to 10K)
            data["m5_df"]       — M5 candles DataFrame
            data["m15_df"]      — M15 candles DataFrame
            data["h1_df"]       — H1 candles DataFrame
            data["h4_df"]       — H4 candles DataFrame
            data["d1_df"]       — D1 candles DataFrame
            data["tick"]        — {"bid", "ask", "spread"}
            data["zones"]       — {"support": [...], "resistance": [...]}
            data["indicators"]  — latest M1 indicators (ema9, ema15, atr, rsi, etc.)
            data["account"]     — {"balance", "equity", "free_margin", ...}
            data["positions"]   — list of open positions

        Return format:
            {
                "signal": "BUY" or "SELL" or "NO_TRADE",
                "sl": stop_loss_price,
                "tp": take_profit_price,
                "confidence": 0.0 to 1.0,
                "reason": "why this trade",
                "sl_distance": abs(entry - sl),
                "entry": entry_price,
                "indicators": data["indicators"],
            }
        """
        return {
            "signal": "NO_TRADE",
            "reason": "Placeholder — implement your strategy in engine/strategies/placeholder.py",
        }
