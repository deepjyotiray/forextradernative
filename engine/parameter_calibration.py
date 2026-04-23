"""
Parameter Calibration System - Controlled tuning of specific parameters.

Allows tuning ONLY these parameters:
- quality_threshold_with_trend (default 0.65)
- quality_threshold_counter_trend (default 0.70-0.75)
- tick_ratio_threshold (default 0.65)
- SL multiplier (ATR * 0.5-0.7 range)
- TP multiplier (ATR * 0.8-1.2 range)

Rules:
- Adjust ONLY one parameter at a time
- Use minimum 30 trades before evaluating change
- Accept change ONLY if win_rate improves OR drawdown decreases
"""
import json
import time
from datetime import datetime, timezone
from typing import Dict, Optional, List
import os
from .performance_analytics import get_performance_metrics


_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CALIBRATION_FILE = os.path.join(_BASE_DIR, "parameter_calibration.json")


class ParameterCalibrationSystem:
    def __init__(self):
        self._parameters = {
            "quality_threshold_with_trend": 0.65,
            "quality_threshold_counter_trend": 0.70,
            "tick_ratio_threshold": 0.65,
            "sl_multiplier": 0.6,  # ATR * 0.6
            "tp_multiplier": 1.0,  # ATR * 1.0
        }
        
        self._parameter_ranges = {
            "quality_threshold_with_trend": (0.55, 0.75),
            "quality_threshold_counter_trend": (0.65, 0.80),
            "tick_ratio_threshold": (0.55, 0.75),
            "sl_multiplier": (0.5, 0.7),
            "tp_multiplier": (0.8, 1.2),
        }
        
        self._calibration_history = []
        self._current_test = None
        self._min_trades_for_evaluation = 30
        self._load_state()
    
    def get_current_parameters(self) -> Dict:
        """Get current parameter values."""
        return self._parameters.copy()
    
    def start_parameter_test(self, parameter_name: str, new_value: float, reason: str = "") -> Dict:
        """Start testing a new parameter value."""
        
        if parameter_name not in self._parameters:
            return {"error": f"Unknown parameter: {parameter_name}"}
        
        min_val, max_val = self._parameter_ranges[parameter_name]
        if not (min_val <= new_value <= max_val):
            return {"error": f"Value {new_value} outside range [{min_val}, {max_val}]"}
        
        if self._current_test:
            return {"error": f"Already testing parameter: {self._current_test['parameter']}"}
        
        # Record baseline performance
        baseline_metrics = get_performance_metrics(lookback_trades=50)
        
        self._current_test = {
            "parameter": parameter_name,
            "old_value": self._parameters[parameter_name],
            "new_value": new_value,
            "start_time": time.time(),
            "start_timestamp": datetime.now(timezone.utc).isoformat(),
            "reason": reason,
            "baseline_metrics": {
                "win_rate": baseline_metrics.get("overall", {}).get("win_rate", 0),
                "max_drawdown": baseline_metrics.get("overall", {}).get("max_drawdown", 0),
                "profit_factor": baseline_metrics.get("overall", {}).get("profit_factor", 1),
                "total_trades": baseline_metrics.get("overall", {}).get("total_trades", 0),
            },
            "trades_at_start": baseline_metrics.get("actual_trades", 0),
        }
        
        # Apply new parameter
        self._parameters[parameter_name] = new_value
        self._save_state()
        
        return {
            "success": True,
            "message": f"Started testing {parameter_name}: {self._current_test['old_value']} -> {new_value}",
            "test_info": self._current_test
        }
    
    def evaluate_current_test(self, force_evaluation: bool = False) -> Dict:
        """Evaluate the current parameter test."""
        
        if not self._current_test:
            return {"error": "No active test"}
        
        current_metrics = get_performance_metrics(lookback_trades=50)
        current_trades = current_metrics.get("actual_trades", 0)
        trades_since_start = current_trades - self._current_test["trades_at_start"]
        
        if not force_evaluation and trades_since_start < self._min_trades_for_evaluation:
            return {
                "status": "testing",
                "trades_needed": self._min_trades_for_evaluation - trades_since_start,
                "trades_since_start": trades_since_start,
                "current_test": self._current_test
            }
        
        # Compare performance
        baseline = self._current_test["baseline_metrics"]
        current = {
            "win_rate": current_metrics.get("overall", {}).get("win_rate", 0),
            "max_drawdown": current_metrics.get("overall", {}).get("max_drawdown", 0),
            "profit_factor": current_metrics.get("overall", {}).get("profit_factor", 1),
            "total_trades": current_metrics.get("overall", {}).get("total_trades", 0),
        }
        
        # Decision criteria: accept if win_rate improves OR drawdown decreases
        win_rate_improved = current["win_rate"] > baseline["win_rate"]
        drawdown_decreased = current["max_drawdown"] < baseline["max_drawdown"]
        
        accept_change = win_rate_improved or drawdown_decreased
        
        # Record result
        test_result = {
            "parameter": self._current_test["parameter"],
            "old_value": self._current_test["old_value"],
            "new_value": self._current_test["new_value"],
            "start_time": self._current_test["start_time"],
            "end_time": time.time(),
            "duration_hours": (time.time() - self._current_test["start_time"]) / 3600,
            "trades_evaluated": trades_since_start,
            "baseline_metrics": baseline,
            "test_metrics": current,
            "win_rate_change": current["win_rate"] - baseline["win_rate"],
            "drawdown_change": current["max_drawdown"] - baseline["max_drawdown"],
            "accepted": accept_change,
            "reason": self._current_test["reason"],
        }
        
        self._calibration_history.append(test_result)
        
        if not accept_change:
            # Revert parameter
            self._parameters[self._current_test["parameter"]] = self._current_test["old_value"]
        
        self._current_test = None
        self._save_state()
        
        return {
            "status": "completed",
            "accepted": accept_change,
            "result": test_result,
            "message": f"Parameter test {'ACCEPTED' if accept_change else 'REJECTED'}"
        }
    
    def get_calibration_history(self, limit: int = 20) -> List[Dict]:
        """Get calibration history."""
        return self._calibration_history[-limit:]
    
    def get_parameter_recommendations(self) -> Dict:
        """Get parameter tuning recommendations based on performance data."""
        metrics = get_performance_metrics(lookback_trades=100)
        
        if metrics.get("error"):
            return {"error": "Insufficient data for recommendations"}
        
        recommendations = []
        
        # Analyze quality score effectiveness
        quality_analysis = metrics.get("quality_score", {})
        if quality_analysis.get("win_rate_low_score", 0) > quality_analysis.get("win_rate_high_score", 0):
            recommendations.append({
                "parameter": "quality_threshold_with_trend",
                "suggestion": "decrease",
                "current": self._parameters["quality_threshold_with_trend"],
                "recommended": max(0.55, self._parameters["quality_threshold_with_trend"] - 0.05),
                "reason": "Low quality scores performing better than high scores"
            })
        
        # Analyze execution filters
        filter_analysis = metrics.get("execution_filters", {})
        if filter_analysis.get("tick_ratio_impact", 0) < 0:
            recommendations.append({
                "parameter": "tick_ratio_threshold",
                "suggestion": "decrease",
                "current": self._parameters["tick_ratio_threshold"],
                "recommended": max(0.55, self._parameters["tick_ratio_threshold"] - 0.05),
                "reason": "High tick ratio filter may be too restrictive"
            })
        
        # Analyze duration vs performance
        duration_analysis = metrics.get("duration", {})
        if duration_analysis.get("avg_duration_losses", 0) > duration_analysis.get("avg_duration_wins", 0) * 1.5:
            recommendations.append({
                "parameter": "sl_multiplier",
                "suggestion": "decrease",
                "current": self._parameters["sl_multiplier"],
                "recommended": max(0.5, self._parameters["sl_multiplier"] - 0.05),
                "reason": "Losses taking too long, tighter SL may help"
            })
        
        return {
            "recommendations": recommendations,
            "analysis_based_on": f"{metrics.get('actual_trades', 0)} trades",
            "current_parameters": self._parameters.copy()
        }
    
    def _save_state(self):
        """Save calibration state to file."""
        try:
            state = {
                "parameters": self._parameters,
                "calibration_history": self._calibration_history,
                "current_test": self._current_test,
                "last_update": time.time()
            }
            with open(_CALIBRATION_FILE, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            print(f"[CALIBRATION ERROR] Failed to save state: {e}")
    
    def _load_state(self):
        """Load calibration state from file."""
        try:
            if os.path.exists(_CALIBRATION_FILE):
                with open(_CALIBRATION_FILE, "r") as f:
                    state = json.load(f)
                
                self._parameters.update(state.get("parameters", {}))
                self._calibration_history = state.get("calibration_history", [])
                self._current_test = state.get("current_test")
                
                # Clean old history (keep last 100 entries)
                self._calibration_history = self._calibration_history[-100:]
        except Exception as e:
            print(f"[CALIBRATION ERROR] Failed to load state: {e}")


# Singleton instance
calibration_system = ParameterCalibrationSystem()


def get_current_parameters() -> Dict:
    """Get current parameter values."""
    return calibration_system.get_current_parameters()


def start_parameter_test(parameter_name: str, new_value: float, reason: str = "") -> Dict:
    """Start testing a new parameter value."""
    return calibration_system.start_parameter_test(parameter_name, new_value, reason)


def evaluate_current_test(force_evaluation: bool = False) -> Dict:
    """Evaluate the current parameter test."""
    return calibration_system.evaluate_current_test(force_evaluation)


def get_calibration_history(limit: int = 20) -> List[Dict]:
    """Get calibration history."""
    return calibration_system.get_calibration_history(limit)


def get_parameter_recommendations() -> Dict:
    """Get parameter tuning recommendations."""
    return calibration_system.get_parameter_recommendations()