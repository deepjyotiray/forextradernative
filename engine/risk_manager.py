"""
Risk Manager — lot sizing, position limits, drawdown, daily P&L, adaptive risk.
"""
import time
from datetime import datetime, timezone, timedelta
from typing import Dict
import config as cfg
from .backtest_context import get_backtest_now
from engine import strategy_configs as _scfg_store

_IST = timezone(timedelta(hours=5, minutes=30))


class RiskManager:
    def __init__(self):
        self._daily_pnl = 0.0
        self._daily_trades = 0
        _now_ist = datetime.now(_IST)
        self._daily_date = _now_ist.strftime("%Y-%m-%d")
        self._daily_target_hit = False
        self._daily_loss_hit = False
        self._consecutive_losses = 0
        self._last_trade_time = 0.0
        self._start_balance = 0.0
        self._risk_multiplier = 1.0  # set by performance tracker
        self._floating_pnl = 0.0  # live unrealized P&L from open positions
        self._loss_streak_pause_until = 0.0  # absolute timestamp when pause ends
        self._manual_reset_cutoff_ts = 0.0  # ignore prior trades for streak rebuilding after reset
        self._manual_hard_reset_cutoff_ts = 0.0  # ignore prior trades for daily P&L/count after hard reset
        self._last_account: Dict = {}  # latest account snapshot for drawdown
        self._last_trade_outcome = ""

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

        today_trades = [t for t in (closed_trades or []) if self._trade_in_today(t)]
        if self._manual_hard_reset_cutoff_ts > 0:
            eligible = [t for t in today_trades if self._trade_close_ts(t) >= self._manual_hard_reset_cutoff_ts]
            self._daily_pnl = round(sum(float(t.get("pnl", 0.0) or 0.0) for t in eligible), 2)
            self._daily_trades = len(eligible)
        else:
            self._daily_pnl = today_pnl.get("pnl", 0.0)
            self._daily_trades = today_pnl.get("trades", 0)
        # Compute trailing consecutive losses from actual trade sequence
        # Skip if pause is active OR was already served (streak was reset to 0)
        if self._loss_streak_pause_until > 0:
            pass  # pause active or already served this day — don't restore streak
        elif today_trades:
            streak_cutoff_ts = self._manual_reset_cutoff_ts
            today_trades = [t for t in today_trades if self._trade_close_ts(t) >= streak_cutoff_ts]
            streak = 0
            for t in reversed(today_trades):
                if t.get("pnl", 0) >= 0:
                    break
                streak += 1
            self._consecutive_losses = streak
            if today_trades:
                last_trade = today_trades[-1]
                self._last_trade_outcome = "LOSS" if float(last_trade.get("pnl", 0.0) or 0.0) < 0 else "WIN"
            else:
                self._last_trade_outcome = ""
        else:
            self._consecutive_losses = 0
            self._last_trade_outcome = ""
        # Re-evaluate flags from fresh MT5 data every reseed (clear then set)
        self._daily_target_hit = self._daily_pnl >= cfg.DAILY_TARGET_DOLLARS
        balance = self._start_balance if self._start_balance > 0 else 1000
        self._daily_loss_hit = self._daily_pnl <= -(balance * cfg.DAILY_LOSS_LIMIT_PCT / 100)

    def set_risk_multiplier(self, mult: float):
        """Called by performance tracker for adaptive risk."""
        self._risk_multiplier = max(0.5, min(1.25, mult))

    @staticmethod
    def _trading_day(dt_ist: datetime) -> str:
        """Trading day resets at midnight IST."""
        return dt_ist.strftime("%Y-%m-%d")

    def _trade_in_today(self, trade: dict) -> bool:
        """True if the trade's close_time falls within today's calendar day (midnight IST)."""
        raw = trade.get("close_time") or trade.get("time", "")
        if not raw:
            return False
        try:
            from .time_utils import coerce_utc
            dt = coerce_utc(raw)
            if dt is None:
                return False
            now_ist = dt.astimezone(_IST)
            today_ist = datetime.now(_IST).date()
            return now_ist.date() == today_ist
        except Exception:
            return False

    def check_daily_reset(self):
        today = self._trading_day(self._now_utc().astimezone(_IST))
        if today != self._daily_date:
            self._daily_pnl = 0.0
            self._daily_trades = 0
            self._daily_target_hit = False
            self._daily_loss_hit = False
            self._daily_date = today
            self._consecutive_losses = 0
            self._loss_streak_pause_until = 0.0
            self._manual_reset_cutoff_ts = 0.0
            self._manual_hard_reset_cutoff_ts = 0.0

    def update_floating_pnl(self, floating: float):
        """Called each cycle with total floating P&L from open positions."""
        self._floating_pnl = floating

    def _trade_close_ts(self, trade: dict) -> float:
        raw = trade.get("close_time") or trade.get("time", "")
        if not raw:
            return 0.0
        try:
            from .time_utils import coerce_utc
            dt = coerce_utc(raw)
            if dt is None:
                return 0.0
            return dt.timestamp()
        except Exception:
            return 0.0

    def should_close_all(self) -> bool:
        """True if realized + floating hits daily target or daily loss limit."""
        combined = self._daily_pnl + self._floating_pnl
        # Daily target
        if cfg.DAILY_TARGET_ENABLED and float(cfg.DAILY_TARGET_DOLLARS) > 0 and combined >= cfg.DAILY_TARGET_DOLLARS:
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
        if cfg.DAILY_TARGET_ENABLED and float(cfg.DAILY_TARGET_DOLLARS) > 0 and combined >= cfg.DAILY_TARGET_DOLLARS:
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

        max_open_trades = int(getattr(cfg, "MAX_OPEN_TRADES", cfg.MAX_POSITIONS))
        if open_count >= max_open_trades:
            return False, "RISK_BLOCK: max open trades reached"

        max_trades_per_day = int(getattr(cfg, "MAX_TRADES_PER_DAY", self._daily_trades or 0) or 0)
        if max_trades_per_day > 0 and self._daily_trades >= max_trades_per_day:
            return False, "RISK_BLOCK: max trades reached"

        if cfg.DAILY_TARGET_ENABLED and float(cfg.DAILY_TARGET_DOLLARS) > 0:
            if self._daily_target_hit:
                return False, f"Daily target ${cfg.DAILY_TARGET_DOLLARS} hit"
            combined = self._daily_pnl + self._floating_pnl
            if combined >= cfg.DAILY_TARGET_DOLLARS:
                return False, f"Daily target ${cfg.DAILY_TARGET_DOLLARS} hit (realized+floating ${combined:.2f})"
        if bool(getattr(cfg, "DAILY_PROFIT_LOCK", False)):
            combined_profit = self._daily_pnl + self._floating_pnl
            if combined_profit >= float(getattr(cfg, "DAILY_PROFIT_LOCK_AFTER", 0.0) or 0.0):
                return False, "RISK_LOCK: daily profit target reached"
        if self._daily_loss_hit:
            return False, "Daily loss limit hit"
        balance_ref = self._start_balance if self._start_balance > 0 else 1000
        fixed_daily_max_loss = float(getattr(cfg, "DAILY_MAX_LOSS", 0.0) or 0.0)
        if fixed_daily_max_loss < 0 and self._daily_pnl <= fixed_daily_max_loss:
            return False, "RISK_BLOCK: daily max loss reached"
        daily_max_loss_pct = float(getattr(cfg, "DAILY_MAX_LOSS_PCT_OF_ACCOUNT", 0.0) or 0.0)
        if daily_max_loss_pct > 0:
            account_daily_max_loss = balance_ref * daily_max_loss_pct / 100.0
            if self._daily_pnl <= -account_daily_max_loss:
                return False, f"RISK_BLOCK: daily max loss ${account_daily_max_loss:.2f} reached"
        # Block new trades if realized + floating breaches loss limit
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
        streak_limit = int(getattr(cfg, "STOP_AFTER_CONSECUTIVE_LOSSES", cfg.MAX_CONSECUTIVE_LOSSES))
        if self._consecutive_losses >= streak_limit:
            self._loss_streak_pause_until = now + cfg.LOSS_STREAK_PAUSE
            self._consecutive_losses = 0  # reset so reseed doesn't re-trigger after pause
            self._last_trade_outcome = "LOSS"
            return False, "RISK_BLOCK: consecutive losses reached"

        if now - self._last_trade_time < cfg.MIN_TRADE_COOLDOWN:
            return False, "Cooldown"

        return True, "OK"

    def calculate_lot(self, account: Dict, sl_distance: float, high_conf: bool = False, strategy: str = "") -> float:
        balance = account.get("balance", 0)
        if balance <= 0 or sl_distance <= 0:
            return cfg.MIN_LOT
        strategy_name = str(strategy or "").upper()
        # Read risk_pct from per-strategy config store; fall back to cfg globals
        try:
            scfg = _scfg_store.get(strategy_name)
            base_risk_pct = float(scfg.get("risk_pct") or 0.0) if scfg else 0.0
            if base_risk_pct <= 0:
                raise ValueError
        except Exception:
            intraday_strategies = {"SWEEP_SCALPER", "INTRADAY_ENGINE"}
            swing_strategies = {"SMC_CONFLUENCE", "M15_SUPPORT_RESISTANCE_REJECTION_V1", "TREND_CHANNEL", "SWING_ENGINE"}
            if strategy_name in intraday_strategies:
                base_risk_pct = float(getattr(cfg, "INTRADAY_RISK_PCT", cfg.MAX_RISK_PCT))
            elif strategy_name in swing_strategies:
                base_risk_pct = float(getattr(cfg, "SWING_RISK_PCT", cfg.MAX_RISK_PCT))
            else:
                base_risk_pct = float(getattr(cfg, "MAX_RISK_PCT", 0.0))
        risk_pct = base_risk_pct * self._risk_multiplier
        if high_conf:
            risk_pct = risk_pct * cfg.SMC_HIGH_CONF_RISK_MULTIPLIER
        risk_amount = balance * (risk_pct / 100)
        lot = risk_amount / (sl_distance * cfg.PIP_VALUE_PER_LOT)
        return max(cfg.MIN_LOT, min(cfg.MAX_LOT, round(lot, 2)))

    def record_trade_result(self, pnl: float, won: bool):
        self._daily_pnl += pnl
        self._daily_trades += 1
        self._last_trade_time = self._now_ts()
        self._last_trade_outcome = "WIN" if won else "LOSS"
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

    def reset_gates(self, hard: bool = False) -> Dict:
        """
        Reset all blocking risk gates so trading can resume immediately.
        hard=False  — clears pause/streak/cooldown only (safe reset)
        hard=True   — also clears daily loss flag and trade counter (full reset)
        """
        now = self._now_ts()
        self._loss_streak_pause_until = 0.0
        self._consecutive_losses = 0
        self._manual_reset_cutoff_ts = now
        self._last_trade_time = 0.0          # clear cooldown
        self._last_trade_outcome = ""
        cleared = [
            "loss_streak_pause",
            "consecutive_losses",
            "trade_cooldown",
        ]
        if hard:
            self._daily_loss_hit = False
            self._daily_target_hit = False
            self._daily_trades = 0
            self._daily_pnl = 0.0
            self._manual_hard_reset_cutoff_ts = now
            cleared += ["daily_loss_flag", "daily_target_flag", "daily_trade_count", "daily_pnl"]
        else:
            self._manual_hard_reset_cutoff_ts = 0.0
        return {"cleared": cleared, "hard": hard, "timestamp": now}

    def record_trade_opened(self):
        self._last_trade_time = self._now_ts()

    @property
    def last_trade_time(self) -> float:
        return self._last_trade_time

    @property
    def last_trade_outcome(self) -> str:
        return self._last_trade_outcome

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
