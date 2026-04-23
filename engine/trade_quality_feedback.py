"""
Trade Quality Feedback Loop - Continuous improvement system.

After each 20-50 trades:
- Analyze which quality_score range performs best
- Analyze which setups produce highest win rate  
- Analyze which filters reject profitable trades
- Adjust thresholds ONLY based on data
"""
import json
import os
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import statistics
from collections import defaultdict
from .trade_attribution import get_recent_attributions
from .performance_analytics import get_performance_metrics


class TradeQualityFeedbackLoop:
    def __init__(self):
        self._analysis_intervals = [20, 50]  # Analyze after 20 and 50 trades
        self._last_analysis_trade_count = 0
        self._feedback_history = []
        self._recommendations = []
        self._load_state()
    
    def check_analysis_needed(self) -> Dict:
        """Check if quality analysis is needed."""
        # Get recent completed trades
        attributions = get_recent_attributions(hours=168)  # 7 days
        completed_trades = [a for a in attributions if a.get("trade_completed", False)]
        current_trade_count = len(completed_trades)
        
        trades_since_last = current_trade_count - self._last_analysis_trade_count
        
        # Check if we've hit an analysis interval
        analysis_needed = False
        interval_hit = None
        
        for interval in self._analysis_intervals:
            if trades_since_last >= interval:
                analysis_needed = True
                interval_hit = interval
                break
        
        return {
            "analysis_needed": analysis_needed,
            "current_trade_count": current_trade_count,
            "trades_since_last_analysis": trades_since_last,
            "interval_hit": interval_hit,
            "next_analysis_at": self._last_analysis_trade_count + min(self._analysis_intervals)
        }
    
    def perform_quality_analysis(self, force: bool = False) -> Dict:
        """Perform comprehensive quality analysis."""
        check = self.check_analysis_needed()
        
        if not check["analysis_needed"] and not force:
            return {
                "status": "not_needed",
                "check_result": check
            }
        
        # Get recent completed trades
        attributions = get_recent_attributions(hours=168)
        completed_trades = [a for a in attributions if a.get("trade_completed", False)]
        
        if len(completed_trades) < 10:
            return {
                "status": "insufficient_data",
                "trade_count": len(completed_trades)
            }
        
        # Take last 50 trades for analysis
        recent_trades = sorted(completed_trades, key=lambda x: x.get("unix_time", 0))[-50:]
        
        analysis = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trade_count": len(recent_trades),
            "analysis_type": "quality_feedback",
            
            # A. Quality Score Range Analysis
            "quality_score_analysis": self._analyze_quality_score_ranges(recent_trades),
            
            # B. Setup Type Analysis
            "setup_analysis": self._analyze_setup_types(recent_trades),
            
            # C. Filter Effectiveness Analysis
            "filter_analysis": self._analyze_filter_effectiveness(recent_trades),
            
            # D. Threshold Optimization
            "threshold_analysis": self._analyze_threshold_optimization(recent_trades),
            
            # E. Session/Time Analysis
            "session_analysis": self._analyze_session_performance(recent_trades),
        }
        
        # Generate recommendations
        recommendations = self._generate_recommendations(analysis)
        analysis["recommendations"] = recommendations
        
        # Update state
        self._last_analysis_trade_count = len(completed_trades)
        self._feedback_history.append(analysis)
        self._recommendations = recommendations
        
        # Keep only last 20 analyses
        self._feedback_history = self._feedback_history[-20:]
        
        self._save_state()
        
        return {
            "status": "completed",
            "analysis": analysis
        }
    
    def _analyze_quality_score_ranges(self, trades: List[Dict]) -> Dict:
        """Analyze performance by quality score ranges."""
        ranges = {
            "high": (0.75, 1.0),
            "medium": (0.65, 0.75),
            "low": (0.0, 0.65)
        }
        
        range_analysis = {}
        
        for range_name, (min_score, max_score) in ranges.items():
            range_trades = [
                t for t in trades 
                if min_score <= t.get("quality_score", 0) < max_score
            ]
            
            if range_trades:
                winners = [t for t in range_trades if t.get("outcome") == "WIN"]
                win_rate = len(winners) / len(range_trades)
                avg_pnl = statistics.mean([t.get("pnl", 0) for t in range_trades])
                
                range_analysis[range_name] = {
                    "trade_count": len(range_trades),
                    "win_rate": round(win_rate, 3),
                    "avg_pnl": round(avg_pnl, 2),
                    "score_range": f"{min_score:.2f}-{max_score:.2f}"
                }
        
        # Find best performing range
        best_range = max(range_analysis.items(), key=lambda x: x[1]["win_rate"]) if range_analysis else None
        
        return {
            "ranges": range_analysis,
            "best_performing_range": best_range[0] if best_range else None,
            "best_win_rate": best_range[1]["win_rate"] if best_range else 0
        }
    
    def _analyze_setup_types(self, trades: List[Dict]) -> Dict:
        """Analyze performance by setup types."""
        setup_performance = defaultdict(lambda: {"trades": [], "win_rate": 0, "avg_pnl": 0})
        
        for trade in trades:
            # Analyze setup direction
            setup_dir = trade.get("setup_direction", "UNKNOWN")
            setup_performance[f"direction_{setup_dir}"]["trades"].append(trade)
            
            # Analyze trade type (with/counter trend)
            trade_type = trade.get("trade_type", "UNKNOWN")
            setup_performance[f"type_{trade_type}"]["trades"].append(trade)
            
            # Analyze session
            session = trade.get("session", "UNKNOWN")
            setup_performance[f"session_{session}"]["trades"].append(trade)
        
        # Calculate metrics for each setup type
        results = {}
        for setup_type, data in setup_performance.items():
            if data["trades"]:
                winners = [t for t in data["trades"] if t.get("outcome") == "WIN"]
                win_rate = len(winners) / len(data["trades"])
                avg_pnl = statistics.mean([t.get("pnl", 0) for t in data["trades"]])
                
                results[setup_type] = {
                    "trade_count": len(data["trades"]),
                    "win_rate": round(win_rate, 3),
                    "avg_pnl": round(avg_pnl, 2)
                }
        
        # Find best setups
        best_setup = max(results.items(), key=lambda x: x[1]["win_rate"]) if results else None
        
        return {
            "setup_performance": results,
            "best_setup": best_setup[0] if best_setup else None,
            "best_setup_win_rate": best_setup[1]["win_rate"] if best_setup else 0
        }
    
    def _analyze_filter_effectiveness(self, trades: List[Dict]) -> Dict:
        """Analyze which filters might be rejecting profitable setups."""
        # Get all recent decisions (taken and skipped)
        all_attributions = get_recent_attributions(hours=168)
        
        # Separate taken and skipped
        taken = [a for a in all_attributions if a.get("decision") == "TRADE_TAKEN"]
        skipped = [a for a in all_attributions if a.get("decision") == "TRADE_SKIPPED"]
        
        # Analyze skip reasons
        skip_reasons = defaultdict(list)
        for attr in skipped:
            reason = attr.get("reason", "Unknown")
            main_reason = reason.split("|")[0].strip()
            skip_reasons[main_reason].append(attr)
        
        # Analyze filter impact
        filter_analysis = {}
        
        for reason, skipped_trades in skip_reasons.items():
            if len(skipped_trades) < 5:  # Skip if too few samples
                continue
            
            # Estimate what win rate these skipped trades might have had
            # by comparing their quality scores to taken trades
            skipped_scores = [t.get("quality_score", 0) for t in skipped_trades if t.get("quality_score")]
            
            if skipped_scores:
                avg_skipped_score = statistics.mean(skipped_scores)
                
                # Find taken trades with similar scores
                similar_taken = [
                    t for t in taken 
                    if abs(t.get("quality_score", 0) - avg_skipped_score) < 0.1
                ]
                
                if similar_taken:
                    # Get outcomes for similar taken trades
                    completed_similar = [t for t in trades if t.get("trade_id") in [s.get("trade_id") for s in similar_taken]]
                    
                    if completed_similar:
                        similar_winners = [t for t in completed_similar if t.get("outcome") == "WIN"]
                        estimated_win_rate = len(similar_winners) / len(completed_similar)
                        
                        filter_analysis[reason] = {
                            "skipped_count": len(skipped_trades),
                            "avg_quality_score": round(avg_skipped_score, 3),
                            "estimated_win_rate": round(estimated_win_rate, 3),
                            "potentially_profitable": estimated_win_rate > 0.5
                        }
        
        return {
            "filter_effectiveness": filter_analysis,
            "total_skipped": len(skipped),
            "total_taken": len(taken),
            "conversion_rate": len(taken) / (len(taken) + len(skipped)) if (taken or skipped) else 0
        }
    
    def _analyze_threshold_optimization(self, trades: List[Dict]) -> Dict:
        """Analyze optimal thresholds based on performance."""
        # Analyze quality score thresholds
        thresholds_to_test = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
        threshold_analysis = {}
        
        for threshold in thresholds_to_test:
            above_threshold = [t for t in trades if t.get("quality_score", 0) >= threshold]
            
            if above_threshold:
                winners = [t for t in above_threshold if t.get("outcome") == "WIN"]
                win_rate = len(winners) / len(above_threshold)
                avg_pnl = statistics.mean([t.get("pnl", 0) for t in above_threshold])
                
                threshold_analysis[threshold] = {
                    "trade_count": len(above_threshold),
                    "win_rate": round(win_rate, 3),
                    "avg_pnl": round(avg_pnl, 2)
                }
        
        # Find optimal threshold (best win rate with reasonable trade count)
        optimal_threshold = None
        best_score = 0
        
        for threshold, metrics in threshold_analysis.items():
            if metrics["trade_count"] >= 5:  # Minimum trade count
                # Score = win_rate * log(trade_count) to balance win rate and volume
                import math
                score = metrics["win_rate"] * math.log(metrics["trade_count"])
                if score > best_score:
                    best_score = score
                    optimal_threshold = threshold
        
        return {
            "threshold_analysis": threshold_analysis,
            "optimal_threshold": optimal_threshold,
            "current_threshold_performance": threshold_analysis.get(0.65, {}),  # Default threshold
        }
    
    def _analyze_session_performance(self, trades: List[Dict]) -> Dict:
        """Analyze performance by session and time."""
        session_performance = defaultdict(list)
        
        for trade in trades:
            session = trade.get("session", "UNKNOWN")
            session_performance[session].append(trade)
        
        results = {}
        for session, session_trades in session_performance.items():
            if session_trades:
                winners = [t for t in session_trades if t.get("outcome") == "WIN"]
                win_rate = len(winners) / len(session_trades)
                avg_pnl = statistics.mean([t.get("pnl", 0) for t in session_trades])
                
                results[session] = {
                    "trade_count": len(session_trades),
                    "win_rate": round(win_rate, 3),
                    "avg_pnl": round(avg_pnl, 2)
                }
        
        best_session = max(results.items(), key=lambda x: x[1]["win_rate"]) if results else None
        
        return {
            "session_performance": results,
            "best_session": best_session[0] if best_session else None,
            "best_session_win_rate": best_session[1]["win_rate"] if best_session else 0
        }
    
    def _generate_recommendations(self, analysis: Dict) -> List[Dict]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        # Quality score threshold recommendations
        threshold_analysis = analysis.get("threshold_analysis", {})
        optimal_threshold = threshold_analysis.get("optimal_threshold")
        
        if optimal_threshold and optimal_threshold != 0.65:
            recommendations.append({
                "type": "threshold_adjustment",
                "parameter": "quality_threshold_with_trend",
                "current_value": 0.65,
                "recommended_value": optimal_threshold,
                "reason": f"Optimal threshold analysis suggests {optimal_threshold} for better performance",
                "priority": "high"
            })
        
        # Filter effectiveness recommendations
        filter_analysis = analysis.get("filter_analysis", {})
        filter_effectiveness = filter_analysis.get("filter_effectiveness", {})
        
        for filter_reason, metrics in filter_effectiveness.items():
            if metrics.get("potentially_profitable") and metrics.get("skipped_count", 0) > 10:
                recommendations.append({
                    "type": "filter_relaxation",
                    "filter": filter_reason,
                    "reason": f"Filter '{filter_reason}' may be rejecting profitable trades (est. {metrics['estimated_win_rate']:.1%} win rate)",
                    "skipped_count": metrics["skipped_count"],
                    "priority": "medium"
                })
        
        # Setup type recommendations
        setup_analysis = analysis.get("setup_analysis", {})
        best_setup = setup_analysis.get("best_setup")
        
        if best_setup and "direction_" in best_setup:
            direction = best_setup.replace("direction_", "")
            if direction in ["LONG", "SHORT"]:
                recommendations.append({
                    "type": "setup_focus",
                    "setup_type": direction,
                    "reason": f"{direction} setups showing best performance ({setup_analysis['best_setup_win_rate']:.1%} win rate)",
                    "priority": "low"
                })
        
        # Session recommendations
        session_analysis = analysis.get("session_analysis", {})
        best_session = session_analysis.get("best_session")
        
        if best_session and best_session != "UNKNOWN":
            recommendations.append({
                "type": "session_focus",
                "session": best_session,
                "reason": f"{best_session} session showing best performance ({session_analysis['best_session_win_rate']:.1%} win rate)",
                "priority": "low"
            })
        
        return recommendations
    
    def get_current_recommendations(self) -> List[Dict]:
        """Get current recommendations."""
        return self._recommendations.copy()
    
    def get_feedback_history(self, limit: int = 10) -> List[Dict]:
        """Get feedback analysis history."""
        return self._feedback_history[-limit:]
    
    def _save_state(self):
        """Save feedback loop state."""
        try:
            state = {
                "last_analysis_trade_count": self._last_analysis_trade_count,
                "feedback_history": self._feedback_history,
                "recommendations": self._recommendations,
                "last_update": time.time()
            }
            
            with open(f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/feedback_loop_state.json", "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            print(f"[FEEDBACK LOOP ERROR] Failed to save state: {e}")
    
    def _load_state(self):
        """Load feedback loop state."""
        try:
            state_file = f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/feedback_loop_state.json"
            if os.path.exists(state_file):
                with open(state_file, "r") as f:
                    state = json.load(f)
                
                self._last_analysis_trade_count = state.get("last_analysis_trade_count", 0)
                self._feedback_history = state.get("feedback_history", [])
                self._recommendations = state.get("recommendations", [])
                
        except Exception as e:
            print(f"[FEEDBACK LOOP ERROR] Failed to load state: {e}")


# Singleton instance
feedback_loop = TradeQualityFeedbackLoop()


def check_analysis_needed() -> Dict:
    """Check if quality analysis is needed."""
    return feedback_loop.check_analysis_needed()


def perform_quality_analysis(force: bool = False) -> Dict:
    """Perform quality analysis."""
    return feedback_loop.perform_quality_analysis(force)


def get_current_recommendations() -> List[Dict]:
    """Get current recommendations."""
    return feedback_loop.get_current_recommendations()


def get_feedback_history(limit: int = 10) -> List[Dict]:
    """Get feedback history."""
    return feedback_loop.get_feedback_history(limit)
