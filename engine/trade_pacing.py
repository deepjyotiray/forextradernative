"""
Trade Pacing Control System - Prevents overtrading.

Implements:
- minimum_time_between_trades = 2-5 minutes
- Block new trades if previous trade too recent
"""
import time
from datetime import datetime, timezone
from typing import Dict, Optional
import json
import os


_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PACING_STATE_FILE = os.path.join(_BASE_DIR, "trade_pacing_state.json")


class TradePacingController:
    def __init__(self):
        self._min_time_between_trades = 120  # 2 minutes in seconds
        self._max_time_between_trades = 300  # 5 minutes in seconds
        self._last_trade_time = 0
        self._trade_history = []  # Recent trade timestamps
        self._adaptive_pacing = True  # Adjust pacing based on recent performance
        self._load_state()
    
    def set_pacing_interval(self, min_seconds: int, max_seconds: int = None):
        """Set minimum time between trades."""
        self._min_time_between_trades = max(60, min(600, min_seconds))  # 1-10 minutes
        if max_seconds:
            self._max_time_between_trades = max(self._min_time_between_trades, min(900, max_seconds))
        else:
            self._max_time_between_trades = self._min_time_between_trades * 2
        self._save_state()
    
    def check_pacing_allowed(self) -> Dict:
        """Check if a new trade is allowed based on pacing rules."""
        current_time = time.time()
        
        if self._last_trade_time == 0:
            return {
                "allowed": True,
                "reason": "First trade of session",
                "time_since_last": 0,
                "min_interval": self._min_time_between_trades
            }
        
        time_since_last = current_time - self._last_trade_time
        required_interval = self._get_required_interval()
        
        if time_since_last < required_interval:
            time_remaining = required_interval - time_since_last
            return {
                "allowed": False,
                "reason": f"Pacing limit: {time_remaining:.0f}s remaining",
                "time_since_last": round(time_since_last),
                "time_remaining": round(time_remaining),
                "required_interval": required_interval,
                "min_interval": self._min_time_between_trades
            }
        
        return {
            "allowed": True,
            "time_since_last": round(time_since_last),
            "required_interval": required_interval,
            "min_interval": self._min_time_between_trades
        }
    
    def record_trade_taken(self, outcome: Optional[str] = None):
        """Record that a trade was taken."""
        current_time = time.time()
        self._last_trade_time = current_time
        
        # Keep trade history for adaptive pacing
        trade_record = {
            "timestamp": current_time,
            "datetime": datetime.now(timezone.utc).isoformat(),
            "outcome": outcome
        }
        
        self._trade_history.append(trade_record)
        
        # Keep only last 20 trades
        self._trade_history = self._trade_history[-20:]
        
        self._save_state()
    
    def _get_required_interval(self) -> int:
        """Get required interval based on adaptive pacing."""
        if not self._adaptive_pacing:
            return self._min_time_between_trades
        
        # Analyze recent performance to adjust pacing
        recent_trades = self._trade_history[-10:]  # Last 10 trades
        
        if len(recent_trades) < 3:
            return self._min_time_between_trades
        
        # Count recent losses
        recent_losses = sum(1 for trade in recent_trades if trade.get("outcome") == "LOSS")
        loss_rate = recent_losses / len(recent_trades)
        
        # Increase pacing if high loss rate
        if loss_rate > 0.6:  # More than 60% losses
            # Use maximum pacing
            return self._max_time_between_trades
        elif loss_rate > 0.4:  # More than 40% losses
            # Use medium pacing
            return int(self._min_time_between_trades * 1.5)
        else:
            # Use minimum pacing
            return self._min_time_between_trades
    
    def get_pacing_status(self) -> Dict:
        """Get current pacing status."""
        current_time = time.time()
        time_since_last = current_time - self._last_trade_time if self._last_trade_time > 0 else 0
        required_interval = self._get_required_interval()
        
        # Analyze recent trading frequency
        recent_trades = [t for t in self._trade_history if current_time - t["timestamp"] < 3600]  # Last hour
        trades_per_hour = len(recent_trades)
        
        # Calculate average interval between recent trades
        avg_interval = 0
        if len(self._trade_history) >= 2:
            intervals = []
            for i in range(1, min(len(self._trade_history), 11)):  # Last 10 intervals
                interval = self._trade_history[-i]["timestamp"] - self._trade_history[-i-1]["timestamp"]
                intervals.append(interval)
            avg_interval = sum(intervals) / len(intervals) if intervals else 0
        
        return {
            "last_trade_time": self._last_trade_time,
            "time_since_last": round(time_since_last),
            "required_interval": required_interval,
            "min_interval": self._min_time_between_trades,
            "max_interval": self._max_time_between_trades,
            "adaptive_pacing": self._adaptive_pacing,
            "trades_last_hour": trades_per_hour,
            "avg_interval_recent": round(avg_interval),
            "recent_trades_count": len(self._trade_history),
            "pacing_allowed": time_since_last >= required_interval,
        }
    
    def get_pacing_settings(self) -> Dict:
        """Get current pacing settings."""
        return {
            "min_time_between_trades": self._min_time_between_trades,
            "max_time_between_trades": self._max_time_between_trades,
            "adaptive_pacing": self._adaptive_pacing,
        }
    
    def update_pacing_settings(self, **kwargs) -> Dict:
        """Update pacing settings."""
        updated = []
        
        if "min_time_between_trades" in kwargs:
            old_val = self._min_time_between_trades
            self._min_time_between_trades = max(60, min(600, kwargs["min_time_between_trades"]))
            updated.append(f"min_time: {old_val}s -> {self._min_time_between_trades}s")
        
        if "max_time_between_trades" in kwargs:
            old_val = self._max_time_between_trades
            self._max_time_between_trades = max(self._min_time_between_trades, min(900, kwargs["max_time_between_trades"]))
            updated.append(f"max_time: {old_val}s -> {self._max_time_between_trades}s")
        
        if "adaptive_pacing" in kwargs:
            old_val = self._adaptive_pacing
            self._adaptive_pacing = bool(kwargs["adaptive_pacing"])
            updated.append(f"adaptive: {old_val} -> {self._adaptive_pacing}")
        
        if updated:
            self._save_state()
        
        return {
            "updated": updated,
            "current_settings": self.get_pacing_settings()
        }
    
    def reset_pacing_history(self) -> Dict:
        """Reset pacing history (use with caution)."""
        old_count = len(self._trade_history)
        self._trade_history = []
        self._last_trade_time = 0
        self._save_state()
        
        return {
            "message": f"Reset pacing history ({old_count} trades cleared)",
            "status": self.get_pacing_status()
        }
    
    def get_trade_frequency_analysis(self, hours: int = 24) -> Dict:
        """Analyze trade frequency over specified period."""
        current_time = time.time()
        cutoff_time = current_time - (hours * 3600)
        
        # Filter trades within time period
        period_trades = [t for t in self._trade_history if t["timestamp"] >= cutoff_time]
        
        if len(period_trades) < 2:
            return {
                "period_hours": hours,
                "trade_count": len(period_trades),
                "analysis": "Insufficient data"
            }
        
        # Calculate intervals
        intervals = []
        for i in range(1, len(period_trades)):
            interval = period_trades[i]["timestamp"] - period_trades[i-1]["timestamp"]
            intervals.append(interval)
        
        # Group by outcome
        wins = [t for t in period_trades if t.get("outcome") == "WIN"]
        losses = [t for t in period_trades if t.get("outcome") == "LOSS"]
        
        return {
            "period_hours": hours,
            "trade_count": len(period_trades),
            "avg_interval": round(sum(intervals) / len(intervals)),
            "min_interval": round(min(intervals)),
            "max_interval": round(max(intervals)),
            "trades_per_hour": round(len(period_trades) / hours, 2),
            "win_count": len(wins),
            "loss_count": len(losses),
            "win_rate": round(len(wins) / len(period_trades), 3) if period_trades else 0,
            "intervals": [round(i) for i in intervals[-10:]]  # Last 10 intervals
        }
    
    def _save_state(self):
        """Save pacing state."""
        try:
            state = {
                "min_time_between_trades": self._min_time_between_trades,
                "max_time_between_trades": self._max_time_between_trades,
                "last_trade_time": self._last_trade_time,
                "trade_history": self._trade_history,
                "adaptive_pacing": self._adaptive_pacing,
                "last_update": time.time()
            }
            
            with open(_PACING_STATE_FILE, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            print(f"[PACING ERROR] Failed to save state: {e}")
    
    def _load_state(self):
        """Load pacing state."""
        try:
            if os.path.exists(_PACING_STATE_FILE):
                with open(_PACING_STATE_FILE, "r") as f:
                    state = json.load(f)
                
                self._min_time_between_trades = state.get("min_time_between_trades", 120)
                self._max_time_between_trades = state.get("max_time_between_trades", 300)
                self._last_trade_time = state.get("last_trade_time", 0)
                self._trade_history = state.get("trade_history", [])
                self._adaptive_pacing = state.get("adaptive_pacing", True)
                
                # Clean old history (keep last 30 days)
                cutoff_time = time.time() - (30 * 24 * 3600)
                self._trade_history = [
                    t for t in self._trade_history 
                    if t.get("timestamp", 0) >= cutoff_time
                ]
                
        except Exception as e:
            print(f"[PACING ERROR] Failed to load state: {e}")


# Singleton instance
pacing_controller = TradePacingController()


def check_pacing_allowed() -> Dict:
    """Check if trade pacing allows new trade."""
    return pacing_controller.check_pacing_allowed()


def record_trade_taken(outcome: Optional[str] = None):
    """Record that a trade was taken."""
    pacing_controller.record_trade_taken(outcome)


def get_pacing_status() -> Dict:
    """Get current pacing status."""
    return pacing_controller.get_pacing_status()


def set_pacing_interval(min_seconds: int, max_seconds: int = None):
    """Set pacing interval."""
    pacing_controller.set_pacing_interval(min_seconds, max_seconds)


def get_pacing_settings() -> Dict:
    """Get pacing settings."""
    return pacing_controller.get_pacing_settings()


def update_pacing_settings(**kwargs) -> Dict:
    """Update pacing settings."""
    return pacing_controller.update_pacing_settings(**kwargs)


def get_trade_frequency_analysis(hours: int = 24) -> Dict:
    """Get trade frequency analysis."""
    return pacing_controller.get_trade_frequency_analysis(hours)