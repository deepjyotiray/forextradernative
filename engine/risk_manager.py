"""
Risk Manager — lot sizing, position limits, drawdown, daily P&L, adaptive risk.
"""
import time
from datetime import datetime, timezone, timedelta
from typing import Dict
import config as cfg
from .backtest_context import get_backtest_now

_IST = timezone(timedelta(hours=5, minutes=30))


class RiskManager:
    def __init__(self):
        self._daily_pnl = 0.0
        self._daily_trades = 0
        self._daily_date = datetime.now(_IST).strftime("%Y-%m-%d")
        self._daily_target_hit = False
        self._daily_loss_hit = False
        self._consecutive_losses = 0
        self._last_trade_time = 0.0
        self._start_balance = 0.0
        self._risk_multiplier = 1.0  # set by performance tracker
        self._floating_pnl = 0.0  # live unrealized P&L from open positions
        self._loss_streak_pause_until = 0.0  # absolute timestamp when pause ends
        self._last_account: Dict = {}  # latest account snapshot for drawdown

    @staticmethod
    def _now_utc() -> datetime:
        override = get_backtest_now()
        if isinstance(override, datetime):
            return override.astimezone(timezone.utc)
        return datetime.now(timezone.utc)

    @staticmethod
    def _now_ts() -> float:
        return RiskManager._now_utc().timestamp()

    def set_start_balance(self, balance: float):
        self._start_balance = balance

    def seed_from_mt5(self, today_pnl: Dict, closed_trades: list = None):
        """Seed daily P&L from MT5 deal history on startup."""
        # Reset day FIRST so yesterday's data doesn't re-trigger flags
        self.check_daily_reset()

        self._daily_pnl = today_pnl.get("pnl", 0.0)
        self._daily_trades = today_pnl.get("trades", 0)
        # Compute trailing consecutive losses from actual trade sequence
        # Skip if pause is active OR was already served (streak was reset to 0)
        if self._loss_streak_pause_until > 0:
            pass  # pause active or already served this day — don't restore streak
        elif closed_trades:
            streak = 0
            for t in reversed(closed_trades):
                if t.get("pnl", 0) >= 0:
                    break
                streak += 1
            self._consecutive_losses = streak
        else:
            self._consecutive_losses = 0
        # Re-evaluate flags from fresh MT5 data every reseed (clear then set)
        self._daily_target_hit = self._daily_pnl >= cfg.DAILY_TARGET_DOLLARS
        balance = self._start_balance if self._start_balance > 0 else 1000
        self._daily_loss_hit = self._daily_pnl <= -(balance * cfg.DAILY_LOSS_LIMIT_PCT / 100)

    def set_risk_multiplier(self, mult: float):
        """Called by performance tracker for adaptive risk."""
        self._risk_multiplier = max(0.5, min(1.25, mult))

    def check_daily_reset(self):
        today = self._now_utc().astimezone(_IST).strftime("%Y-%m-%d")
        if today != self._daily_date:
            self._daily_pnl = 0.0
            self._daily_trades = 0
            self._daily_target_hit = False
            self._daily_loss_hit = False
            self._daily_date = today
            self._consecutive_losses = 0
            self._loss_streak_pause_until = 0.0

    def update_floating_pnl(self, floating: float):
        """Called each cycle with total floating P&L from open positions."""
        self._floating_pnl = floating

    def should_close_all(self) -> bool:
        """True if realized + floating hits daily target or daily loss limit."""
        combined = self._daily_pnl + self._floating_pnl
        # Daily target
        if cfg.DAILY_TARGET_ENABLED and combined >= cfg.DAILY_TARGET_DOLLARS:
            return True
        # Daily loss limit
        balance = self._start_balance if self._start_balance > 0 else 1000
        loss_limit = balance * cfg.DAILY_LOSS_LIMIT_PCT / 100
        if combined <= -loss_limit:
            return True
        return False

    def should_close_all_reason(self) -> str:
        """Return reason for close-all, or empty string."""
        combined = self._daily_pnl + self._floating_pnl
        if cfg.DAILY_TARGET_ENABLED and combined >= cfg.DAILY_TARGET_DOLLARS:
            return f"Daily target ${cfg.DAILY_TARGET_DOLLARS} reached (realized+floating ${combined:.2f})"
        balance = self._start_balance if self._start_balance > 0 else 1000
        loss_limit = balance * cfg.DAILY_LOSS_LIMIT_PCT / 100
        if combined <= -loss_limit:
            return f"Daily loss limit -${loss_limit:.2f} breached (realized+floating ${combined:.2f})"
        return ""

    def should_close_drawdown(self, account: Dict) -> bool:
        """True if equity drawdown exceeds MAX_DRAWDOWN_PCT."""
        balance = account.get("balance", 0)
        equity = account.get("equity", balance)
        if balance <= 0:
            return False
        dd = (balance - equity) / balance * 100
        return dd >= cfg.MAX_DRAWDOWN_PCT

    def can_trade(self, account: Dict, open_count: int) -> tuple:
        """Returns (allowed: bool, reason: str)."""
        self.check_daily_reset()

        if cfg.DAILY_TARGET_ENABLED:
            if self._daily_target_hit:
                return False, f"Daily target ${cfg.DAILY_TARGET_DOLLARS} hit"
            combined = self._daily_pnl + self._floating_pnl
            if combined >= cfg.DAILY_TARGET_DOLLARS:
                return False, f"Daily target ${cfg.DAILY_TARGET_DOLLARS} hit (realized+floating ${combined:.2f})"
        if self._daily_loss_hit:
            return False, "Daily loss limit hit"
        # Block new trades if realized + floating breaches loss limit
        balance_ref = self._start_balance if self._start_balance > 0 else 1000
        loss_limit = balance_ref * cfg.DAILY_LOSS_LIMIT_PCT / 100
        combined_loss = self._daily_pnl + self._floating_pnl
        if combined_loss <= -loss_limit:
            return False, f"Daily loss limit (realized+floating ${combined_loss:.2f})"

        balance = account.get("balance", 0)
        equity = account.get("equity", balance)
        self._last_account = account

        if open_count >= cfg.MAX_POSITIONS:
            return False, f"Max positions ({cfg.MAX_POSITIONS})"

        if balance > 0:
            dd = (balance - equity) / balance * 100
            if dd >= cfg.MAX_DRAWDOWN_PCT:
                return False, f"Drawdown {dd:.1f}% >= {cfg.MAX_DRAWDOWN_PCT}%"

        # Loss streak pause — use absolute timestamp so reseed can't bypass
        now = self._now_ts()
        if now < self._loss_streak_pause_until:
            remaining = self._loss_streak_pause_until - now
            return False, f"Loss streak pause ({remaining:.0f}s remaining)"
        if self._consecutive_losses >= cfg.MAX_CONSECUTIVE_LOSSES:
            self._loss_streak_pause_until = now + cfg.LOSS_STREAK_PAUSE
            self._consecutive_losses = 0  # reset so reseed doesn't re-trigger after pause
            return False, f"Loss streak pause ({cfg.LOSS_STREAK_PAUSE}s)"

        if now - self._last_trade_time < cfg.MIN_TRADE_COOLDOWN:
            return False, "Cooldown"

        return True, "OK"

    def calculate_lot(self, account: Dict, sl_distance: float, high_conf: bool = False) -> float:
        balance = account.get("balance", 0)
        if balance <= 0 or sl_distance <= 0:
            return cfg.MIN_LOT
        risk_pct = cfg.MAX_RISK_PCT * self._risk_multiplier
        if high_conf:
            risk_pct = risk_pct * cfg.SMC_HIGH_CONF_RISK_MULTIPLIER
        risk_amount = balance * (risk_pct / 100)
        lot = risk_amount / (sl_distance * cfg.PIP_VALUE_PER_LOT)
        return max(cfg.MIN_LOT, min(cfg.MAX_LOT, round(lot, 2)))

    def record_trade_result(self, pnl: float, won: bool):
        self._daily_pnl += pnl
        self._daily_trades += 1
        self._last_trade_time = self._now_ts()
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
        self._last_trade_time = self._now_ts()

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
