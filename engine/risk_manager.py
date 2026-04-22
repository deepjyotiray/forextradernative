"""
Risk Manager — lot sizing, position limits, drawdown, daily P&L, adaptive risk.
"""
import time
from datetime import datetime, timezone
from typing import Dict
import config as cfg


class RiskManager:
    def __init__(self):
        self._daily_pnl = 0.0
        self._daily_trades = 0
        self._daily_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self._daily_target_hit = False
        self._daily_loss_hit = False
        self._consecutive_losses = 0
        self._last_trade_time = 0.0
        self._start_balance = 0.0
        self._risk_multiplier = 1.0  # set by performance tracker

    def set_start_balance(self, balance: float):
        self._start_balance = balance

    def seed_from_mt5(self, today_pnl: Dict):
        """Seed daily P&L from MT5 deal history on startup."""
        self._daily_pnl = today_pnl.get("pnl", 0.0)
        self._daily_trades = today_pnl.get("trades", 0)
        self._consecutive_losses = today_pnl.get("losses", 0)  # conservative
        if self._daily_pnl >= cfg.DAILY_TARGET_DOLLARS:
            self._daily_target_hit = True
        balance = self._start_balance if self._start_balance > 0 else 1000
        if self._daily_pnl <= -(balance * cfg.DAILY_LOSS_LIMIT_PCT / 100):
            self._daily_loss_hit = True

    def set_risk_multiplier(self, mult: float):
        """Called by performance tracker for adaptive risk."""
        self._risk_multiplier = max(0.5, min(1.25, mult))

    def check_daily_reset(self):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._daily_date:
            self._daily_pnl = 0.0
            self._daily_trades = 0
            self._daily_target_hit = False
            self._daily_loss_hit = False
            self._daily_date = today
            self._consecutive_losses = 0

    def can_trade(self, account: Dict, open_count: int) -> tuple:
        """Returns (allowed: bool, reason: str)."""
        self.check_daily_reset()

        if self._daily_target_hit:
            return False, f"Daily target ${cfg.DAILY_TARGET_DOLLARS} hit"
        if self._daily_loss_hit:
            return False, "Daily loss limit hit"

        balance = account.get("balance", 0)
        equity = account.get("equity", balance)

        if open_count >= cfg.MAX_POSITIONS:
            return False, f"Max positions ({cfg.MAX_POSITIONS})"

        if balance > 0:
            dd = (balance - equity) / balance * 100
            if dd >= cfg.MAX_DRAWDOWN_PCT:
                return False, f"Drawdown {dd:.1f}% >= {cfg.MAX_DRAWDOWN_PCT}%"

        if self._consecutive_losses >= cfg.MAX_CONSECUTIVE_LOSSES:
            elapsed = time.time() - self._last_trade_time
            if elapsed < cfg.LOSS_STREAK_PAUSE:
                return False, f"Loss streak pause ({cfg.LOSS_STREAK_PAUSE - elapsed:.0f}s remaining)"
            self._consecutive_losses = 0

        if time.time() - self._last_trade_time < cfg.MIN_TRADE_COOLDOWN:
            return False, "Cooldown"

        return True, "OK"

    def calculate_lot(self, account: Dict, sl_distance: float) -> float:
        balance = account.get("balance", 0)
        if balance <= 0 or sl_distance <= 0:
            return cfg.MIN_LOT
        risk_pct = cfg.MAX_RISK_PCT * self._risk_multiplier
        risk_amount = balance * (risk_pct / 100)
        lot = risk_amount / (sl_distance * cfg.PIP_VALUE_PER_LOT)
        return max(cfg.MIN_LOT, min(cfg.MAX_LOT, round(lot, 2)))

    def record_trade_result(self, pnl: float, won: bool):
        self._daily_pnl += pnl
        self._daily_trades += 1
        self._last_trade_time = time.time()
        if won:
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1
        if self._daily_pnl >= cfg.DAILY_TARGET_DOLLARS:
            self._daily_target_hit = True
        # Use actual balance for daily loss limit
        balance = self._start_balance if self._start_balance > 0 else 1000
        if self._daily_pnl <= -(balance * cfg.DAILY_LOSS_LIMIT_PCT / 100):
            self._daily_loss_hit = True

    def record_trade_opened(self):
        self._last_trade_time = time.time()

    @property
    def daily_status(self) -> Dict:
        return {
            "pnl": round(self._daily_pnl, 2),
            "trades": self._daily_trades,
            "target_hit": self._daily_target_hit,
            "loss_limit_hit": self._daily_loss_hit,
            "consecutive_losses": self._consecutive_losses,
            "date": self._daily_date,
            "risk_multiplier": round(self._risk_multiplier, 2),
        }
