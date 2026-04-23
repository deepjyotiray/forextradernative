"""
Session-Level Risk Control System - Mandatory risk limits.

Implements:
- max_consecutive_losses = 2
- max_daily_loss = 2% of account  
- max_trades_per_session = 5

Rules:
- if consecutive_losses >= 2: stop trading for session
- if daily_loss >= 2%: stop trading completely
- Block new trades if limits exceeded
"""
import json
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
import os
from .trade_attribution import get_recent_attributions


_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RISK_STATE_FILE = os.path.join(_BASE_DIR, "risk_control_state.json")


class SessionRiskController:
    def __init__(self):
        self._max_consecutive_losses = 2
        self._max_daily_loss_pct = 0.02  # 2%
        self._max_trades_per_session = 5
        self._account_balance = 10000.0  # Default, should be updated
        
        self._current_session = ""
        self._current_date = ""
        self._session_trades = 0
        self._consecutive_losses = 0
        self._daily_pnl = 0.0
        self._last_trade_outcome = None
        
        self._risk_blocks = {
            "consecutive_losses": False,
            "daily_loss": False,
            "session_trades": False,
        }
        
        self._risk_history = []
        self._load_state()
    
    def update_account_balance(self, balance: float):
        """Update account balance for risk calculations."""
        self._account_balance = balance
        self._save_state()
    
    def update_session(self):
        """Update current session and reset counters if needed."""
        now = datetime.now(timezone.utc)
        h = now.hour
        date = now.strftime("%Y-%m-%d")
        
        if 7 <= h < 9:
            session = "LONDON"
        elif 12 <= h < 13:
            session = "OVERLAP"
        elif 13 <= h < 15:
            session = "NY"
        else:
            session = "CLOSED"
        
        # Reset session trades if new session
        if session != self._current_session and session != "CLOSED":
            self._session_trades = 0
            self._risk_blocks["session_trades"] = False
        
        # Reset daily counters if new day
        if date != self._current_date:
            self._daily_pnl = 0.0
            self._consecutive_losses = 0
            self._risk_blocks["daily_loss"] = False
            self._risk_blocks["consecutive_losses"] = False
            
            # Recalculate daily PnL from attribution data
            self._recalculate_daily_metrics(date)
        
        self._current_session = session
        self._current_date = date
        self._save_state()
    
    def check_trade_allowed(self) -> Dict:
        """Check if a new trade is allowed based on risk limits."""
        self.update_session()
        
        if self._current_session == "CLOSED":
            return {
                "allowed": False,
                "reason": "Outside trading hours",
                "risk_status": self.get_risk_status()
            }
        
        # Check consecutive losses
        if self._risk_blocks["consecutive_losses"]:
            return {
                "allowed": False,
                "reason": f"Consecutive losses limit reached ({self._consecutive_losses}/{self._max_consecutive_losses})",
                "risk_status": self.get_risk_status()
            }
        
        # Check daily loss
        if self._risk_blocks["daily_loss"]:
            return {
                "allowed": False,
                "reason": f"Daily loss limit reached ({self._daily_pnl:.2f} / {self._max_daily_loss_pct*100:.0f}%)",
                "risk_status": self.get_risk_status()
            }
        
        # Check session trades
        if self._session_trades >= self._max_trades_per_session:
            self._risk_blocks["session_trades"] = True
            return {
                "allowed": False,
                "reason": f"Session trade limit reached ({self._session_trades}/{self._max_trades_per_session})",
                "risk_status": self.get_risk_status()
            }
        
        return {
            "allowed": True,
            "risk_status": self.get_risk_status()
        }
    
    def record_trade_taken(self):
        """Record that a trade was taken."""
        self.update_session()
        self._session_trades += 1
        self._save_state()
    
    def record_trade_outcome(self, pnl: float, outcome: str):
        """Record trade outcome and update risk metrics."""
        self.update_session()
        
        self._daily_pnl += pnl
        self._last_trade_outcome = outcome
        
        # Update consecutive losses
        if outcome == "LOSS":
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0
        
        # Check risk limits
        if self._consecutive_losses >= self._max_consecutive_losses:
            self._risk_blocks["consecutive_losses"] = True
            self._log_risk_event("consecutive_losses", f"{self._consecutive_losses} consecutive losses")
        
        daily_loss_limit = self._account_balance * self._max_daily_loss_pct
        if self._daily_pnl <= -daily_loss_limit:
            self._risk_blocks["daily_loss"] = True
            self._log_risk_event("daily_loss", f"Daily loss {self._daily_pnl:.2f} exceeds limit {daily_loss_limit:.2f}")
        
        self._save_state()
    
    def get_risk_status(self) -> Dict:
        """Get current risk status."""
        self.update_session()
        
        daily_loss_limit = self._account_balance * self._max_daily_loss_pct
        
        return {
            "current_session": self._current_session,
            "current_date": self._current_date,
            "session_trades": self._session_trades,
            "max_session_trades": self._max_trades_per_session,
            "consecutive_losses": self._consecutive_losses,
            "max_consecutive_losses": self._max_consecutive_losses,
            "daily_pnl": round(self._daily_pnl, 2),
            "daily_loss_limit": round(daily_loss_limit, 2),
            "daily_loss_pct": round((abs(self._daily_pnl) / self._account_balance) * 100, 2) if self._daily_pnl < 0 else 0,
            "account_balance": self._account_balance,
            "risk_blocks": self._risk_blocks.copy(),
            "last_trade_outcome": self._last_trade_outcome,
            "trades_remaining_session": max(0, self._max_trades_per_session - self._session_trades),
        }
    
    def get_risk_limits(self) -> Dict:
        """Get configured risk limits."""
        return {
            "max_consecutive_losses": self._max_consecutive_losses,
            "max_daily_loss_pct": self._max_daily_loss_pct,
            "max_trades_per_session": self._max_trades_per_session,
            "account_balance": self._account_balance,
        }
    
    def update_risk_limits(self, **kwargs) -> Dict:
        """Update risk limits (use with caution)."""
        updated = []
        
        if "max_consecutive_losses" in kwargs:
            old_val = self._max_consecutive_losses
            self._max_consecutive_losses = max(1, min(5, kwargs["max_consecutive_losses"]))
            updated.append(f"max_consecutive_losses: {old_val} -> {self._max_consecutive_losses}")
        
        if "max_daily_loss_pct" in kwargs:
            old_val = self._max_daily_loss_pct
            self._max_daily_loss_pct = max(0.01, min(0.05, kwargs["max_daily_loss_pct"]))
            updated.append(f"max_daily_loss_pct: {old_val:.1%} -> {self._max_daily_loss_pct:.1%}")
        
        if "max_trades_per_session" in kwargs:
            old_val = self._max_trades_per_session
            self._max_trades_per_session = max(1, min(10, kwargs["max_trades_per_session"]))
            updated.append(f"max_trades_per_session: {old_val} -> {self._max_trades_per_session}")
        
        if updated:
            self._save_state()
        
        return {
            "updated": updated,
            "current_limits": self.get_risk_limits()
        }
    
    def reset_session_blocks(self) -> Dict:
        """Reset session-level blocks (use with extreme caution)."""
        old_blocks = self._risk_blocks.copy()
        
        # Only reset session-level blocks, not daily blocks
        self._risk_blocks["consecutive_losses"] = False
        self._risk_blocks["session_trades"] = False
        
        self._log_risk_event("manual_reset", "Session blocks manually reset")
        self._save_state()
        
        return {
            "message": "Session blocks reset",
            "old_blocks": old_blocks,
            "new_blocks": self._risk_blocks.copy()
        }
    
    def get_risk_history(self, days: int = 7) -> List[Dict]:
        """Get risk event history."""
        cutoff = time.time() - (days * 24 * 3600)
        return [event for event in self._risk_history if event.get("timestamp", 0) >= cutoff]
    
    def _recalculate_daily_metrics(self, date: str):
        """Recalculate daily metrics from attribution data."""
        # Get today's completed trades
        attributions = get_recent_attributions(hours=24)
        
        daily_pnl = 0.0
        consecutive_losses = 0
        last_outcome = None
        
        # Filter for today's completed trades
        today_trades = []
        for attr in attributions:
            if not attr.get("trade_completed"):
                continue
            
            attr_date = attr.get("timestamp", "")[:10]  # YYYY-MM-DD
            if attr_date == date:
                today_trades.append(attr)
        
        # Sort by completion time
        today_trades.sort(key=lambda x: x.get("unix_time", 0))
        
        # Calculate metrics
        for trade in today_trades:
            pnl = trade.get("pnl", 0)
            outcome = trade.get("outcome", "")
            
            daily_pnl += pnl
            last_outcome = outcome
            
            if outcome == "LOSS":
                consecutive_losses += 1
            else:
                consecutive_losses = 0
        
        self._daily_pnl = daily_pnl
        self._consecutive_losses = consecutive_losses
        self._last_trade_outcome = last_outcome
        
        # Update blocks based on recalculated metrics
        daily_loss_limit = self._account_balance * self._max_daily_loss_pct
        if self._daily_pnl <= -daily_loss_limit:
            self._risk_blocks["daily_loss"] = True
        
        if self._consecutive_losses >= self._max_consecutive_losses:
            self._risk_blocks["consecutive_losses"] = True
    
    def _log_risk_event(self, event_type: str, description: str):
        """Log a risk management event."""
        event = {
            "timestamp": time.time(),
            "datetime": datetime.now(timezone.utc).isoformat(),
            "type": event_type,
            "description": description,
            "session": self._current_session,
            "date": self._current_date,
            "risk_status": self.get_risk_status()
        }
        
        self._risk_history.append(event)
        
        # Keep only last 100 events
        self._risk_history = self._risk_history[-100:]
    
    def _save_state(self):
        """Save risk control state."""
        try:
            state = {
                "account_balance": self._account_balance,
                "current_session": self._current_session,
                "current_date": self._current_date,
                "session_trades": self._session_trades,
                "consecutive_losses": self._consecutive_losses,
                "daily_pnl": self._daily_pnl,
                "last_trade_outcome": self._last_trade_outcome,
                "risk_blocks": self._risk_blocks,
                "risk_limits": {
                    "max_consecutive_losses": self._max_consecutive_losses,
                    "max_daily_loss_pct": self._max_daily_loss_pct,
                    "max_trades_per_session": self._max_trades_per_session,
                },
                "risk_history": self._risk_history,
                "last_update": time.time()
            }
            
            with open(_RISK_STATE_FILE, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            print(f"[RISK CONTROL ERROR] Failed to save state: {e}")
    
    def _load_state(self):
        """Load risk control state."""
        try:
            if os.path.exists(_RISK_STATE_FILE):
                with open(_RISK_STATE_FILE, "r") as f:
                    state = json.load(f)
                
                self._account_balance = state.get("account_balance", 10000.0)
                self._current_session = state.get("current_session", "")
                self._current_date = state.get("current_date", "")
                self._session_trades = state.get("session_trades", 0)
                self._consecutive_losses = state.get("consecutive_losses", 0)
                self._daily_pnl = state.get("daily_pnl", 0.0)
                self._last_trade_outcome = state.get("last_trade_outcome")
                self._risk_blocks = state.get("risk_blocks", {
                    "consecutive_losses": False,
                    "daily_loss": False,
                    "session_trades": False,
                })
                
                # Load risk limits
                limits = state.get("risk_limits", {})
                self._max_consecutive_losses = limits.get("max_consecutive_losses", 2)
                self._max_daily_loss_pct = limits.get("max_daily_loss_pct", 0.02)
                self._max_trades_per_session = limits.get("max_trades_per_session", 5)
                
                self._risk_history = state.get("risk_history", [])
                
        except Exception as e:
            print(f"[RISK CONTROL ERROR] Failed to load state: {e}")


# Singleton instance
risk_controller = SessionRiskController()


def check_trade_allowed() -> Dict:
    """Check if a new trade is allowed."""
    return risk_controller.check_trade_allowed()


def record_trade_taken():
    """Record that a trade was taken."""
    risk_controller.record_trade_taken()


def record_trade_outcome(pnl: float, outcome: str):
    """Record trade outcome."""
    risk_controller.record_trade_outcome(pnl, outcome)


def get_risk_status() -> Dict:
    """Get current risk status."""
    return risk_controller.get_risk_status()


def update_account_balance(balance: float):
    """Update account balance."""
    risk_controller.update_account_balance(balance)


def get_risk_limits() -> Dict:
    """Get risk limits."""
    return risk_controller.get_risk_limits()


def update_risk_limits(**kwargs) -> Dict:
    """Update risk limits."""
    return risk_controller.update_risk_limits(**kwargs)


def reset_session_blocks() -> Dict:
    """Reset session blocks (use with caution)."""
    return risk_controller.reset_session_blocks()


def get_risk_history(days: int = 7) -> List[Dict]:
    """Get risk event history."""
    return risk_controller.get_risk_history(days)