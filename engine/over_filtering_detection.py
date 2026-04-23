"""
Over-Filtering Detection System - Identifies when system is too restrictive.

Tracks:
- signals_detected_per_session
- trades_executed_per_session  
- conversion_rate = trades / signals

If conversion_rate < 10% OR trades_per_session < 1 consistently:
- System is over-filtered

Action:
- Relax ONE filter slightly:
  tick_ratio: 65% → 60% OR spread tolerance: +0.02
"""
import json
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional
import os
from collections import defaultdict, deque
from .trade_attribution import get_recent_attributions


_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FILTER_STATE_FILE = os.path.join(_BASE_DIR, "over_filtering_state.json")


class OverFilteringDetector:
    def __init__(self):
        self._session_data = defaultdict(lambda: {"signals": 0, "trades": 0})
        self._daily_conversion_rates = deque(maxlen=10)  # Last 10 days
        self._filter_relaxations = []
        self._current_relaxation = None
        self._min_conversion_rate = 0.10  # 10%
        self._min_trades_per_session = 1.0
        self._evaluation_days = 5  # Evaluate over 5 days
        self._load_state()
    
    def update_session_stats(self):
        """Update session statistics from attribution data."""
        # Get recent attributions (last 7 days)
        attributions = get_recent_attributions(hours=168)
        
        # Clear current session data
        self._session_data.clear()
        
        # Group by session
        for attr in attributions:
            timestamp = attr.get("timestamp", "")
            if not timestamp:
                continue
            
            try:
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                date = dt.strftime("%Y-%m-%d")
                hour = dt.hour
                
                if 7 <= hour < 9:
                    session = "LONDON"
                elif 12 <= hour < 13:
                    session = "OVERLAP"
                elif 13 <= hour < 15:
                    session = "NY"
                else:
                    continue  # Skip non-trading hours
                
                session_key = f"{date}_{session}"
                
                # Count all signals
                self._session_data[session_key]["signals"] += 1
                
                # Count executed trades
                if attr.get("decision") == "TRADE_TAKEN":
                    self._session_data[session_key]["trades"] += 1
                    
            except Exception:
                continue
        
        self._save_state()
    
    def analyze_filtering_status(self) -> Dict:
        """Analyze current filtering status and detect over-filtering."""
        self.update_session_stats()
        
        if not self._session_data:
            return {"status": "insufficient_data", "message": "No session data available"}
        
        # Calculate conversion rates by session
        session_stats = {}
        total_signals = 0
        total_trades = 0
        
        for session_key, data in self._session_data.items():
            signals = data["signals"]
            trades = data["trades"]
            conversion_rate = trades / signals if signals > 0 else 0
            
            session_stats[session_key] = {
                "signals": signals,
                "trades": trades,
                "conversion_rate": conversion_rate
            }
            
            total_signals += signals
            total_trades += trades
        
        overall_conversion_rate = total_trades / total_signals if total_signals > 0 else 0
        
        # Calculate average trades per session
        sessions_with_data = len([s for s in session_stats.values() if s["signals"] > 0])
        avg_trades_per_session = total_trades / sessions_with_data if sessions_with_data > 0 else 0
        
        # Detect over-filtering
        is_over_filtered = (
            overall_conversion_rate < self._min_conversion_rate or
            avg_trades_per_session < self._min_trades_per_session
        )
        
        # Get recent sessions (last 5 days)
        recent_sessions = self._get_recent_sessions(5)
        recent_conversion_rates = [
            s["trades"] / s["signals"] if s["signals"] > 0 else 0 
            for s in recent_sessions.values()
        ]
        
        consistent_low_conversion = (
            len(recent_conversion_rates) >= 3 and
            sum(1 for rate in recent_conversion_rates if rate < self._min_conversion_rate) >= len(recent_conversion_rates) * 0.7
        )
        
        analysis = {
            "status": "over_filtered" if is_over_filtered else "normal",
            "overall_conversion_rate": round(overall_conversion_rate, 3),
            "avg_trades_per_session": round(avg_trades_per_session, 2),
            "total_signals": total_signals,
            "total_trades": total_trades,
            "sessions_analyzed": len(session_stats),
            "is_over_filtered": is_over_filtered,
            "consistent_low_conversion": consistent_low_conversion,
            "thresholds": {
                "min_conversion_rate": self._min_conversion_rate,
                "min_trades_per_session": self._min_trades_per_session
            },
            "recent_sessions": recent_sessions,
            "current_relaxation": self._current_relaxation,
        }
        
        return analysis
    
    def recommend_filter_relaxation(self) -> Dict:
        """Recommend filter relaxation if over-filtering detected."""
        analysis = self.analyze_filtering_status()
        
        if not analysis["is_over_filtered"]:
            return {
                "recommendation": "none",
                "message": "No over-filtering detected",
                "analysis": analysis
            }
        
        if self._current_relaxation:
            return {
                "recommendation": "wait",
                "message": f"Already relaxing {self._current_relaxation['type']}",
                "current_relaxation": self._current_relaxation,
                "analysis": analysis
            }
        
        # Determine which filter to relax
        # Alternate between tick_ratio and spread_tolerance
        last_relaxation_type = None
        if self._filter_relaxations:
            last_relaxation_type = self._filter_relaxations[-1].get("type")
        
        if last_relaxation_type == "tick_ratio":
            recommended_type = "spread_tolerance"
            recommended_change = "+0.02 to spread limits"
        else:
            recommended_type = "tick_ratio"
            recommended_change = "65% → 60%"
        
        return {
            "recommendation": "relax_filter",
            "type": recommended_type,
            "change": recommended_change,
            "reason": f"Conversion rate {analysis['overall_conversion_rate']:.1%} < {self._min_conversion_rate:.0%} or avg trades/session {analysis['avg_trades_per_session']:.1f} < {self._min_trades_per_session}",
            "analysis": analysis
        }
    
    def apply_filter_relaxation(self, relaxation_type: str) -> Dict:
        """Apply filter relaxation."""
        if self._current_relaxation:
            return {"error": f"Already relaxing {self._current_relaxation['type']}"}
        
        if relaxation_type not in ["tick_ratio", "spread_tolerance"]:
            return {"error": f"Invalid relaxation type: {relaxation_type}"}
        
        self._current_relaxation = {
            "type": relaxation_type,
            "start_time": time.time(),
            "start_timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        relaxation_record = self._current_relaxation.copy()
        self._filter_relaxations.append(relaxation_record)
        
        self._save_state()
        
        return {
            "success": True,
            "message": f"Applied {relaxation_type} relaxation",
            "relaxation": self._current_relaxation
        }
    
    def evaluate_relaxation_effectiveness(self, min_hours: int = 24) -> Dict:
        """Evaluate effectiveness of current relaxation."""
        if not self._current_relaxation:
            return {"error": "No active relaxation"}
        
        hours_since_start = (time.time() - self._current_relaxation["start_time"]) / 3600
        if hours_since_start < min_hours:
            return {
                "status": "evaluating",
                "hours_remaining": min_hours - hours_since_start,
                "current_relaxation": self._current_relaxation
            }
        
        # Compare before/after metrics
        analysis = self.analyze_filtering_status()
        
        # Simple evaluation: if conversion rate improved, keep relaxation
        improved = analysis["overall_conversion_rate"] > self._min_conversion_rate
        
        if improved:
            # Keep relaxation permanently
            self._current_relaxation["status"] = "permanent"
            self._current_relaxation["end_time"] = time.time()
            self._current_relaxation["effectiveness"] = "positive"
        else:
            # Remove relaxation
            self._current_relaxation["status"] = "removed"
            self._current_relaxation["end_time"] = time.time()
            self._current_relaxation["effectiveness"] = "negative"
        
        result = self._current_relaxation.copy()
        self._current_relaxation = None
        self._save_state()
        
        return {
            "status": "completed",
            "result": result,
            "analysis": analysis
        }
    
    def get_active_relaxations(self) -> Dict:
        """Get currently active filter relaxations."""
        relaxations = {}
        
        if self._current_relaxation:
            relaxations[self._current_relaxation["type"]] = True
        
        # Check for permanent relaxations
        for relaxation in self._filter_relaxations:
            if relaxation.get("status") == "permanent":
                relaxations[relaxation["type"]] = True
        
        return relaxations
    
    def _get_recent_sessions(self, days: int) -> Dict:
        """Get session data for recent days."""
        cutoff_time = time.time() - (days * 24 * 3600)
        cutoff_date = datetime.fromtimestamp(cutoff_time, timezone.utc).strftime("%Y-%m-%d")
        
        recent_sessions = {}
        for session_key, data in self._session_data.items():
            session_date = session_key.split("_")[0]
            if session_date >= cutoff_date:
                recent_sessions[session_key] = data
        
        return recent_sessions
    
    def _save_state(self):
        """Save state to file."""
        try:
            state = {
                "session_data": dict(self._session_data),
                "filter_relaxations": self._filter_relaxations,
                "current_relaxation": self._current_relaxation,
                "last_update": time.time()
            }
            with open(_FILTER_STATE_FILE, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            print(f"[FILTER DETECTOR ERROR] Failed to save state: {e}")
    
    def _load_state(self):
        """Load state from file."""
        try:
            if os.path.exists(_FILTER_STATE_FILE):
                with open(_FILTER_STATE_FILE, "r") as f:
                    state = json.load(f)
                
                # Convert back to defaultdict
                session_data = state.get("session_data", {})
                for key, value in session_data.items():
                    self._session_data[key] = value
                
                self._filter_relaxations = state.get("filter_relaxations", [])
                self._current_relaxation = state.get("current_relaxation")
                
                # Clean old data (keep last 30 days)
                cutoff_time = time.time() - (30 * 24 * 3600)
                cutoff_date = datetime.fromtimestamp(cutoff_time, timezone.utc).strftime("%Y-%m-%d")
                
                # Clean session data
                keys_to_remove = []
                for key in self._session_data.keys():
                    session_date = key.split("_")[0]
                    if session_date < cutoff_date:
                        keys_to_remove.append(key)
                
                for key in keys_to_remove:
                    del self._session_data[key]
                
        except Exception as e:
            print(f"[FILTER DETECTOR ERROR] Failed to load state: {e}")


# Singleton instance
filter_detector = OverFilteringDetector()


def analyze_filtering_status() -> Dict:
    """Analyze current filtering status."""
    return filter_detector.analyze_filtering_status()


def recommend_filter_relaxation() -> Dict:
    """Recommend filter relaxation if needed."""
    return filter_detector.recommend_filter_relaxation()


def apply_filter_relaxation(relaxation_type: str) -> Dict:
    """Apply filter relaxation."""
    return filter_detector.apply_filter_relaxation(relaxation_type)


def get_active_relaxations() -> Dict:
    """Get currently active filter relaxations."""
    return filter_detector.get_active_relaxations()


def evaluate_relaxation_effectiveness(min_hours: int = 24) -> Dict:
    """Evaluate effectiveness of current relaxation."""
    return filter_detector.evaluate_relaxation_effectiveness(min_hours)