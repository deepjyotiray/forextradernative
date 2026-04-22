"""
Performance Tracker — tracks win rate, RR, drawdown, expectancy.
Provides adaptive feedback for risk adjustment.
"""
import json
import os
from typing import Dict, List
from collections import deque


class PerformanceTracker:
    def __init__(self, path: str = "trade_history.json"):
        self._path = path
        self.trades: List[Dict] = []
        self._recent: deque = deque(maxlen=50)
        self._peak_balance = 0.0
        self._ticket_set: set = set()  # fast lookup for dedup
        self._load()

    def record(self, trade: Dict):
        ticket = trade.get("ticket")
        if ticket and ticket in self._ticket_set:
            # Update existing record with richer data (MT5 sync may enrich)
            for i, t in enumerate(self.trades):
                if t.get("ticket") == ticket:
                    self.trades[i] = {**t, **trade}  # merge, new data wins
                    break
        else:
            self.trades.append(trade)
            if ticket:
                self._ticket_set.add(ticket)
            self._recent.append(trade)
        self._save()

    def sync_from_mt5(self, mt5_closed: List[Dict]):
        """Full sync: ensure every MT5 closed trade is in the tracker with correct P&L."""
        for mt5t in mt5_closed:
            ticket = mt5t.get("ticket")
            if not ticket:
                continue
            record = {
                "ticket": ticket,
                "pnl": mt5t.get("pnl", 0),
                "won": mt5t.get("won", False),
                "time": mt5t.get("close_time", ""),
                "direction": mt5t.get("direction", ""),
                "volume": mt5t.get("volume", 0),
                "entry_price": mt5t.get("entry_price", 0),
                "exit_price": mt5t.get("exit_price", 0),
                "symbol": mt5t.get("symbol", ""),
                "comment": mt5t.get("comment", ""),
            }
            if ticket in self._ticket_set:
                # Update P&L to match MT5 (authoritative)
                for i, t in enumerate(self.trades):
                    if t.get("ticket") == ticket:
                        if t.get("pnl") != record["pnl"]:
                            self.trades[i] = {**t, **record}
                        break
            else:
                self.trades.append(record)
                self._ticket_set.add(ticket)
                self._recent.append(record)
        self._save()

    def get_stats(self) -> Dict:
        if not self.trades:
            return {"total": 0}
        pnls = [t.get("pnl", 0) for t in self.trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = abs(sum(losses) / len(losses)) if losses else 0
        win_rate = len(wins) / len(pnls) if pnls else 0
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

        cum = 0
        peak = 0
        max_dd = 0
        for p in pnls:
            cum += p
            if cum > peak:
                peak = cum
            dd = peak - cum
            if dd > max_dd:
                max_dd = dd

        return {
            "total": len(pnls),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(win_rate, 3),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "expectancy": round(expectancy, 2),
            "total_pnl": round(sum(pnls), 2),
            "max_drawdown": round(max_dd, 2),
            "avg_rr": round(avg_win / avg_loss, 2) if avg_loss > 0 else 0,
        }

    def recent_stats(self, n: int = 20) -> Dict:
        """Stats over last N trades for adaptive risk."""
        recent = list(self._recent)[-n:]
        if not recent:
            return {"total": 0, "win_rate": 0.5}
        pnls = [t.get("pnl", 0) for t in recent]
        wins = sum(1 for p in pnls if p > 0)
        return {
            "total": len(pnls),
            "win_rate": round(wins / len(pnls), 3),
            "streak": self._current_streak(),
        }

    def risk_multiplier(self) -> float:
        """Adaptive risk: reduce after drawdown, slight increase on streak."""
        rs = self.recent_stats(20)
        wr = rs.get("win_rate", 0.5)
        streak = rs.get("streak", 0)
        if wr < 0.3:
            return 0.5  # halve risk
        if wr < 0.4:
            return 0.75
        if streak >= 3 and wr > 0.6:
            return min(1.25, 1.0 + streak * 0.05)  # bounded increase
        return 1.0

    def _current_streak(self) -> int:
        """Positive = win streak, negative = loss streak."""
        if not self._recent:
            return 0
        streak = 0
        last_won = self._recent[-1].get("pnl", 0) > 0
        for t in reversed(self._recent):
            if (t.get("pnl", 0) > 0) == last_won:
                streak += 1
            else:
                break
        return streak if last_won else -streak

    def _save(self):
        try:
            with open(self._path, "w") as f:
                json.dump(self.trades[-500:], f, default=str)
        except Exception:
            pass

    def _load(self):
        if os.path.isfile(self._path):
            try:
                with open(self._path) as f:
                    self.trades = json.load(f)
                self._ticket_set = {t.get("ticket") for t in self.trades if t.get("ticket")}
                for t in self.trades[-50:]:
                    self._recent.append(t)
            except Exception:
                pass
