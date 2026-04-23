"""
Reporting Dashboard - Generate periodic performance reports.

Generates comprehensive reports including:
- Total trades, win rate, profit factor
- Average win/loss, max drawdown
- Best/worst session, setup type
- Filter effectiveness, parameter performance
"""
import json
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
import statistics
from .trade_attribution import get_recent_attributions
from .performance_analytics import get_performance_metrics, get_filter_effectiveness_report
from .parameter_calibration import get_calibration_history
from .session_risk_control import get_risk_history
from .trade_quality_feedback import get_feedback_history


class ReportingDashboard:
    def __init__(self):
        self._report_cache = {}
        self._cache_duration = 300  # 5 minutes
    
    def generate_comprehensive_report(self, period_days: int = 7) -> Dict:
        """Generate comprehensive performance report."""
        
        # Check cache
        cache_key = f"comprehensive_{period_days}"
        now = time.time()
        if cache_key in self._report_cache:
            cached_report, cache_time = self._report_cache[cache_key]
            if now - cache_time < self._cache_duration:
                return cached_report
        
        # Get data
        attributions = get_recent_attributions(hours=period_days * 24)
        completed_trades = [a for a in attributions if a.get("trade_completed", False)]
        
        if not completed_trades:
            return {"error": "No completed trades in period", "period_days": period_days}
        
        # Generate report sections
        report = {
            "report_timestamp": datetime.now(timezone.utc).isoformat(),
            "period_days": period_days,
            "period_start": (datetime.now(timezone.utc) - timedelta(days=period_days)).isoformat(),
            "period_end": datetime.now(timezone.utc).isoformat(),
            
            # Core Performance Metrics
            "performance_summary": self._generate_performance_summary(completed_trades),
            
            # Detailed Analytics
            "directional_analysis": self._generate_directional_analysis(completed_trades),
            "session_analysis": self._generate_session_analysis(completed_trades),
            "setup_analysis": self._generate_setup_analysis(completed_trades),
            "quality_analysis": self._generate_quality_analysis(completed_trades),
            "risk_analysis": self._generate_risk_analysis(completed_trades),
            
            # Filter and System Analysis
            "filter_effectiveness": get_filter_effectiveness_report(),
            "parameter_changes": self._get_recent_parameter_changes(),
            "system_health": self._generate_system_health_report(),
            
            # Recommendations
            "recommendations": self._generate_report_recommendations(completed_trades),
        }
        
        # Cache report
        self._report_cache[cache_key] = (report, now)
        
        return report
    
    def _generate_performance_summary(self, trades: List[Dict]) -> Dict:
        """Generate core performance summary."""
        winners = [t for t in trades if t.get("outcome") == "WIN"]
        losers = [t for t in trades if t.get("outcome") == "LOSS"]
        be_trades = [t for t in trades if t.get("outcome") == "BE"]
        
        pnls = [t.get("pnl", 0) for t in trades]
        win_pnls = [t.get("pnl", 0) for t in winners]
        loss_pnls = [t.get("pnl", 0) for t in losers]
        
        # Calculate drawdown
        cumulative_pnl = []
        running_total = 0
        for trade in sorted(trades, key=lambda x: x.get("unix_time", 0)):
            running_total += trade.get("pnl", 0)
            cumulative_pnl.append(running_total)
        
        peak = cumulative_pnl[0] if cumulative_pnl else 0
        max_drawdown = 0
        for pnl in cumulative_pnl:
            if pnl > peak:
                peak = pnl
            drawdown = peak - pnl
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # Find best and worst trades
        best_trade = max(trades, key=lambda x: x.get("pnl", 0)) if trades else None
        worst_trade = min(trades, key=lambda x: x.get("pnl", 0)) if trades else None
        
        return {
            "total_trades": len(trades),
            "win_count": len(winners),
            "loss_count": len(losers),
            "be_count": len(be_trades),
            "win_rate": round(len(winners) / len(trades), 3) if trades else 0,
            "total_pnl": round(sum(pnls), 2),
            "avg_win": round(statistics.mean(win_pnls), 2) if win_pnls else 0,
            "avg_loss": round(statistics.mean(loss_pnls), 2) if loss_pnls else 0,
            "profit_factor": round(abs(sum(win_pnls) / sum(loss_pnls)), 2) if loss_pnls and sum(loss_pnls) != 0 else float('inf'),
            "max_drawdown": round(max_drawdown, 2),
            "best_trade": {
                "pnl": round(best_trade.get("pnl", 0), 2),
                "setup": best_trade.get("setup_direction"),
                "session": best_trade.get("session")
            } if best_trade else None,
            "worst_trade": {
                "pnl": round(worst_trade.get("pnl", 0), 2),
                "setup": worst_trade.get("setup_direction"),
                "session": worst_trade.get("session")
            } if worst_trade else None,
        }
    
    def _generate_directional_analysis(self, trades: List[Dict]) -> Dict:
        """Analyze LONG vs SHORT performance."""
        long_trades = [t for t in trades if t.get("setup_direction") == "LONG"]
        short_trades = [t for t in trades if t.get("setup_direction") == "SHORT"]
        
        def analyze_direction(direction_trades, direction_name):
            if not direction_trades:
                return {"trade_count": 0}
            
            winners = [t for t in direction_trades if t.get("outcome") == "WIN"]
            pnls = [t.get("pnl", 0) for t in direction_trades]
            durations = [t.get("trade_duration", 0) for t in direction_trades if t.get("trade_duration")]
            
            return {
                "trade_count": len(direction_trades),
                "win_rate": round(len(winners) / len(direction_trades), 3),
                "total_pnl": round(sum(pnls), 2),
                "avg_pnl": round(statistics.mean(pnls), 2),
                "avg_duration": round(statistics.mean(durations), 0) if durations else 0,
                "best_trade_pnl": round(max(pnls), 2) if pnls else 0,
                "worst_trade_pnl": round(min(pnls), 2) if pnls else 0,
            }
        
        return {
            "LONG": analyze_direction(long_trades, "LONG"),
            "SHORT": analyze_direction(short_trades, "SHORT"),
            "direction_balance": len(long_trades) / max(len(short_trades), 1) if short_trades else float('inf'),
            "better_direction": "LONG" if (long_trades and short_trades and 
                                         (len([t for t in long_trades if t.get("outcome") == "WIN"]) / len(long_trades)) > 
                                         (len([t for t in short_trades if t.get("outcome") == "WIN"]) / len(short_trades))) else "SHORT"
        }
    
    def _generate_session_analysis(self, trades: List[Dict]) -> Dict:
        """Analyze performance by trading session."""
        sessions = ["LONDON", "OVERLAP", "NY"]
        session_analysis = {}
        
        for session in sessions:
            session_trades = [t for t in trades if t.get("session") == session]
            
            if session_trades:
                winners = [t for t in session_trades if t.get("outcome") == "WIN"]
                pnls = [t.get("pnl", 0) for t in session_trades]
                
                session_analysis[session] = {
                    "trade_count": len(session_trades),
                    "win_rate": round(len(winners) / len(session_trades), 3),
                    "total_pnl": round(sum(pnls), 2),
                    "avg_pnl": round(statistics.mean(pnls), 2),
                }
            else:\n                session_analysis[session] = {"trade_count": 0}
        
        # Find best and worst sessions
        active_sessions = {k: v for k, v in session_analysis.items() if v["trade_count"] > 0}
        best_session = max(active_sessions.items(), key=lambda x: x[1]["win_rate"]) if active_sessions else None
        worst_session = min(active_sessions.items(), key=lambda x: x[1]["win_rate"]) if active_sessions else None
        
        return {
            "sessions": session_analysis,
            "best_session": {
                "name": best_session[0],
                "win_rate": best_session[1]["win_rate"],
                "trade_count": best_session[1]["trade_count"]
            } if best_session else None,
            "worst_session": {
                "name": worst_session[0],
                "win_rate": worst_session[1]["win_rate"],
                "trade_count": worst_session[1]["trade_count"]
            } if worst_session else None,
        }
    
    def _generate_setup_analysis(self, trades: List[Dict]) -> Dict:
        """Analyze performance by setup types."""
        # Analyze by trade type (with/counter trend)
        with_trend = [t for t in trades if t.get("trade_type") == "WITH_TREND"]
        counter_trend = [t for t in trades if t.get("trade_type") == "COUNTER_TREND"]
        
        def analyze_trade_type(type_trades):
            if not type_trades:
                return {"trade_count": 0}
            
            winners = [t for t in type_trades if t.get("outcome") == "WIN"]
            pnls = [t.get("pnl", 0) for t in type_trades]
            
            return {
                "trade_count": len(type_trades),
                "win_rate": round(len(winners) / len(type_trades), 3),
                "avg_pnl": round(statistics.mean(pnls), 2),
                "total_pnl": round(sum(pnls), 2),
            }
        
        return {
            "with_trend": analyze_trade_type(with_trend),
            "counter_trend": analyze_trade_type(counter_trend),
            "trend_alignment_ratio": len(with_trend) / max(len(counter_trend), 1) if counter_trend else float('inf'),
        }
    
    def _generate_quality_analysis(self, trades: List[Dict]) -> Dict:
        """Analyze quality score effectiveness."""
        # Group by quality score ranges
        high_quality = [t for t in trades if t.get("quality_score", 0) >= 0.75]
        med_quality = [t for t in trades if 0.65 <= t.get("quality_score", 0) < 0.75]
        low_quality = [t for t in trades if t.get("quality_score", 0) < 0.65]
        
        def analyze_quality_range(range_trades):
            if not range_trades:
                return {"trade_count": 0}
            
            winners = [t for t in range_trades if t.get("outcome") == "WIN"]
            return {
                "trade_count": len(range_trades),
                "win_rate": round(len(winners) / len(range_trades), 3),
                "avg_score": round(statistics.mean([t.get("quality_score", 0) for t in range_trades]), 3),
            }
        
        return {
            "high_quality": analyze_quality_range(high_quality),
            "medium_quality": analyze_quality_range(med_quality),
            "low_quality": analyze_quality_range(low_quality),
            "quality_effectiveness": high_quality and med_quality and 
                                   (len([t for t in high_quality if t.get("outcome") == "WIN"]) / len(high_quality)) > 
                                   (len([t for t in med_quality if t.get("outcome") == "WIN"]) / len(med_quality))
        }
    
    def _generate_risk_analysis(self, trades: List[Dict]) -> Dict:
        """Analyze risk metrics and patterns."""
        # Consecutive loss analysis
        consecutive_losses = 0
        max_consecutive_losses = 0
        current_streak = 0
        
        for trade in sorted(trades, key=lambda x: x.get("unix_time", 0)):
            if trade.get("outcome") == "LOSS":
                current_streak += 1
                max_consecutive_losses = max(max_consecutive_losses, current_streak)
            else:
                current_streak = 0
        
        # Duration analysis
        durations = [t.get("trade_duration", 0) for t in trades if t.get("trade_duration")]
        
        return {
            "max_consecutive_losses": max_consecutive_losses,
            "avg_trade_duration": round(statistics.mean(durations), 0) if durations else 0,
            "longest_trade": max(durations) if durations else 0,
            "shortest_trade": min(durations) if durations else 0,
            "risk_events": len(get_risk_history(days=7)),
        }
    
    def _get_recent_parameter_changes(self) -> List[Dict]:
        """Get recent parameter calibration changes."""
        history = get_calibration_history(limit=10)
        return [
            {
                "parameter": h.get("parameter"),
                "old_value": h.get("old_value"),
                "new_value": h.get("new_value"),
                "accepted": h.get("accepted"),
                "date": datetime.fromtimestamp(h.get("end_time", 0), timezone.utc).strftime("%Y-%m-%d") if h.get("end_time") else None
            }
            for h in history
        ]
    
    def _generate_system_health_report(self) -> Dict:
        """Generate system health indicators."""
        # Get recent feedback analysis
        feedback_history = get_feedback_history(limit=3)
        
        # Get filter effectiveness
        filter_report = get_filter_effectiveness_report()
        
        return {
            "recent_analyses": len(feedback_history),
            "conversion_rate": filter_report.get("conversion_rate", 0),
            "system_status": "healthy" if filter_report.get("conversion_rate", 0) > 0.1 else "over_filtered",
            "last_feedback_analysis": feedback_history[0].get("timestamp") if feedback_history else None,
        }
    
    def _generate_report_recommendations(self, trades: List[Dict]) -> List[Dict]:
        """Generate actionable recommendations based on report analysis."""
        recommendations = []
        
        # Performance-based recommendations
        if len(trades) >= 20:
            win_rate = len([t for t in trades if t.get("outcome") == "WIN"]) / len(trades)
            
            if win_rate < 0.4:
                recommendations.append({
                    "type": "performance_alert",
                    "priority": "high",
                    "message": f"Win rate {win_rate:.1%} is below 40%. Consider parameter adjustment or strategy review.",
                })
            
            # Directional imbalance
            long_trades = [t for t in trades if t.get("setup_direction") == "LONG"]
            short_trades = [t for t in trades if t.get("setup_direction") == "SHORT"]
            
            if len(long_trades) > len(short_trades) * 3:
                recommendations.append({
                    "type": "directional_imbalance",
                    "priority": "medium",
                    "message": f"Heavy LONG bias detected ({len(long_trades)} LONG vs {len(short_trades)} SHORT). Review setup detection.",
                })
            elif len(short_trades) > len(long_trades) * 3:
                recommendations.append({
                    "type": "directional_imbalance",
                    "priority": "medium",
                    "message": f"Heavy SHORT bias detected ({len(short_trades)} SHORT vs {len(long_trades)} LONG). Review setup detection.",
                })
        
        return recommendations
    
    def generate_daily_summary(self) -> Dict:
        """Generate quick daily summary."""
        return self.generate_comprehensive_report(period_days=1)
    
    def generate_weekly_summary(self) -> Dict:
        """Generate weekly summary."""
        return self.generate_comprehensive_report(period_days=7)
    
    def generate_monthly_summary(self) -> Dict:
        """Generate monthly summary."""
        return self.generate_comprehensive_report(period_days=30)
    
    def export_report_json(self, report: Dict, filename: Optional[str] = None) -> str:
        """Export report to JSON file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trading_report_{timestamp}.json"
        
        try:
            import os
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            filepath = os.path.join(base_dir, filename)
            
            with open(filepath, "w") as f:
                json.dump(report, f, indent=2)
            
            return filepath
        except Exception as e:
            return f"Export failed: {e}"


# Singleton instance
dashboard = ReportingDashboard()


def generate_comprehensive_report(period_days: int = 7) -> Dict:
    """Generate comprehensive performance report."""
    return dashboard.generate_comprehensive_report(period_days)


def generate_daily_summary() -> Dict:
    """Generate daily summary."""
    return dashboard.generate_daily_summary()


def generate_weekly_summary() -> Dict:
    """Generate weekly summary."""
    return dashboard.generate_weekly_summary()


def generate_monthly_summary() -> Dict:
    """Generate monthly summary."""
    return dashboard.generate_monthly_summary()


def export_report_json(report: Dict, filename: Optional[str] = None) -> str:
    """Export report to JSON file."""
    return dashboard.export_report_json(report, filename)