"""
Anti-Starvation Module - Controlled parameter relaxation.

If no trades in session:
- Relax ONE parameter only:
  * tick_ratio: 65% → 60%
    OR
  * spread tolerance: +0.02
"""
import time
from datetime import datetime, timezone
from typing import Dict, Optional
import json
import os
from .session_filter import get_session, is_market_open
from .backtest_context import get_backtest_now, is_backtest_mode

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_STATE_FILE = os.path.join(_BASE_DIR, "anti_starvation_state.json")


class AntiStarvationManager:
    def __init__(self):
        self._session_trades = {}  # {session_date: trade_count}
        self._current_session = ""
        self._current_date = ""
        self._relaxation_active = False
        self._relaxation_type = None  # "tick_ratio" or "spread_tolerance"
        self._load_state()
    
    def update_session(self):
        """Update current session and date."""
        now = _now_utc()
        date = now.strftime("%Y-%m-%d")
        session = get_session() if is_market_open() else "CLOSED"
        
        session_key = f"{date}_{session}"
        
        # Reset if new session
        if session_key != f"{self._current_date}_{self._current_session}":
            self._current_session = session
            self._current_date = date
            if session_key not in self._session_trades:
                self._session_trades[session_key] = 0
            self._check_starvation()
            if not is_backtest_mode():
                self._save_state()
    
    def record_trade(self):
        """Record a trade taken in current session."""
        self.update_session()
        if self._current_session != "CLOSED":
            session_key = f"{self._current_date}_{self._current_session}"
            self._session_trades[session_key] = self._session_trades.get(session_key, 0) + 1
            
            # Disable relaxation if we got a trade
            if self._relaxation_active:
                self._relaxation_active = False
                self._relaxation_type = None
            
            if not is_backtest_mode():
                self._save_state()
    
    def _check_starvation(self):
        """Check if we should activate anti-starvation for current session."""
        if self._current_session == "CLOSED":
            self._relaxation_active = False
            self._relaxation_type = None
            return

        session_key = f"{self._current_date}_{self._current_session}"
        trade_count = self._session_trades.get(session_key, 0)

        # Only relax after the session has had enough time to prove starvation.
        if trade_count == 0 and self._minutes_since_session_start() >= 30:
            if not self._relaxation_active:
                last_type = self._get_last_relaxation_type()
                if last_type == "tick_ratio":
                    self._relaxation_type = "spread_tolerance"
                else:
                    self._relaxation_type = "tick_ratio"
                self._relaxation_active = True
        else:
            self._relaxation_active = False
            self._relaxation_type = None

    def _minutes_since_session_start(self) -> int:
        """Measure how long the current configured session has been open."""
        now = _now_utc()
        start_hour = {"ASIAN": 0, "LONDON": 7, "NEW_YORK": 13}.get(self._current_session)
        if start_hour is None:
            return 0
        session_start = now.replace(hour=start_hour, minute=0, second=0, microsecond=0)
        return max(0, int((now - session_start).total_seconds() // 60))
    
    def _get_last_relaxation_type(self) -> str:
        """Get the last relaxation type used."""
        # Look at recent sessions to see what was used last
        sessions = sorted(self._session_trades.keys(), reverse=True)
        for session in sessions[:5]:  # Check last 5 sessions
            if self._session_trades[session] > 0:
                # This session had trades, check if relaxation was used before it
                continue
        return "spread_tolerance"  # Default
    
    def get_relaxed_params(self) -> Dict:
        """Get relaxed parameters if anti-starvation is active."""
        self.update_session()
        
        if not self._relaxation_active or self._current_session == "CLOSED":
            return {"active": False}
        
        params = {"active": True, "type": self._relaxation_type}
        
        if self._relaxation_type == "tick_ratio":
            params["tick_ratio_threshold"] = 0.60  # Relaxed from 0.65
            params["spread_tolerance_bonus"] = 0.0
        elif self._relaxation_type == "spread_tolerance":
            params["tick_ratio_threshold"] = 0.65  # Normal
            params["spread_tolerance_bonus"] = 0.02  # +0.02 to spread limits
        
        return params
    
    def get_status(self) -> Dict:
        """Get current anti-starvation status."""
        self.update_session()
        session_key = f"{self._current_date}_{self._current_session}"
        
        return {
            "current_session": self._current_session,
            "current_date": self._current_date,
            "session_trades": self._session_trades.get(session_key, 0),
            "relaxation_active": self._relaxation_active,
            "relaxation_type": self._relaxation_type,
            "recent_sessions": {k: v for k, v in sorted(self._session_trades.items(), reverse=True)[:10]}
        }
    
    def _save_state(self):
        """Save state to file."""
        if is_backtest_mode():
            return
        try:
            state = {
                "session_trades": self._session_trades,
                "current_session": self._current_session,
                "current_date": self._current_date,
                "relaxation_active": self._relaxation_active,
                "relaxation_type": self._relaxation_type,
                "last_update": time.time()
            }
            with open(_STATE_FILE, "w") as f:
                json.dump(state, f, indent=2)
        except Exception:
            pass  # Fail silently
    
    def _load_state(self):
        """Load state from file."""
        if is_backtest_mode():
            return
        try:
            if os.path.exists(_STATE_FILE):
                with open(_STATE_FILE, "r") as f:
                    state = json.load(f)
                
                self._session_trades = state.get("session_trades", {})
                self._current_session = state.get("current_session", "")
                self._current_date = state.get("current_date", "")
                self._relaxation_active = state.get("relaxation_active", False)
                self._relaxation_type = state.get("relaxation_type", None)
                
                # Clean old sessions (keep last 30 days)
                cutoff_time = time.time() - (30 * 24 * 3600)
                cutoff_date = datetime.fromtimestamp(cutoff_time, timezone.utc).strftime("%Y-%m-%d")
                
                self._session_trades = {
                    k: v for k, v in self._session_trades.items() 
                    if k.split("_")[0] >= cutoff_date
                }
        except Exception:
            pass  # Start fresh if load fails


# Singleton instance
anti_starvation = AntiStarvationManager()


def apply_anti_starvation_to_signal(signal: Dict, strategy: str) -> Dict:
    """Apply anti-starvation parameter relaxation to a signal."""
    if signal.get("signal") not in ("BUY", "SELL"):
        return signal
    
    params = anti_starvation.get_relaxed_params()
    if not params["active"]:
        return signal
    
    # Add relaxation info to signal
    signal["_anti_starvation"] = {
        "active": True,
        "type": params["type"],
        "original_reason": signal.get("reason", "")
    }
    
    # Note: The actual parameter relaxation should be applied in the strategy
    # generation logic, not here. This just tags the signal.
    
    return signal


def record_trade_taken():
    """Record that a trade was taken."""
    anti_starvation.record_trade()


def get_anti_starvation_status() -> Dict:
    """Get current anti-starvation status."""
    return anti_starvation.get_status()


def _now_utc() -> datetime:
    override = get_backtest_now()
    if isinstance(override, datetime):
        return override.astimezone(timezone.utc)
    return datetime.now(timezone.utc)
