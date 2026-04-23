"""
Performance Analytics Engine - Rolling metrics and performance analysis.

Computes rolling metrics (last 50-100 trades):
- Directional performance (LONG/SHORT win rates)
- Bias alignment (WITH_TREND/COUNTER_TREND win rates)
- Session performance (London/NY/Overlap win rates)
- Quality score effectiveness
- Execution filters impact
- Trade duration analysis
"""
import json
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional
from collections import defaultdict
import statistics
from .trade_attribution import get_recent_attributions


class PerformanceAnalyticsEngine:
    def __init__(self):
        self._cache = {}
        self._cache_expiry = 0
        self._cache_duration = 300  # 5 minutes
    
    def get_performance_metrics(self, lookback_trades: int = 100) -> Dict:
        """Get comprehensive performance metrics."""
        
        # Check cache
        now = time.time()
        if now < self._cache_expiry and self._cache.get("lookback") == lookback_trades:
            return self._cache
        
        attributions = get_recent_attributions(hours=72)  # 3 days
        completed_trades = [a for a in attributions if a.get("trade_completed", False)]
        
        # Take last N completed trades
        completed_trades = sorted(completed_trades, key=lambda x: x.get("unix_time", 0))[-lookback_trades:]
        
        if len(completed_trades) < 10:
            return {"error": "Insufficient completed trades", "trade_count": len(completed_trades)}
        
        metrics = {
            "lookback_trades": lookback_trades,
            "actual_trades": len(completed_trades),
            "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
            
            # A. Directional Performance
            "directional": self._analyze_directional_performance(completed_trades),
            
            # B. Bias Alignment
            "bias_alignment": self._analyze_bias_alignment(completed_trades),
            
            # C. Session Performance
            "session": self._analyze_session_performance(completed_trades),
            
            # D. Quality Score Effectiveness
            "quality_score": self._analyze_quality_score_effectiveness(completed_trades),
            
            # E. Execution Filters Impact
            "execution_filters": self._analyze_execution_filters_impact(completed_trades),
            
            # F. Trade Duration
            "duration": self._analyze_trade_duration(completed_trades),
            
            # G. Overall Performance
            "overall": self._analyze_overall_performance(completed_trades),
        }
        
        # Cache results
        self._cache = metrics
        self._cache_expiry = now + self._cache_duration
        
        return metrics
    
    def _analyze_directional_performance(self, trades: List[Dict]) -> Dict:
        """Analyze LONG vs SHORT performance."""
        long_trades = [t for t in trades if t.get("setup_direction") == "LONG"]
        short_trades = [t for t in trades if t.get("setup_direction") == "SHORT"]
        
        return {
            "win_rate_long": self._calculate_win_rate(long_trades),
            "win_rate_short": self._calculate_win_rate(short_trades),
            "avg_pnl_long": self._calculate_avg_pnl(long_trades),
            "avg_pnl_short": self._calculate_avg_pnl(short_trades),
            "trade_count_long": len(long_trades),
            "trade_count_short": len(short_trades),
            "direction_balance": len(long_trades) / max(len(short_trades), 1) if short_trades else float('inf')
        }
    
    def _analyze_bias_alignment(self, trades: List[Dict]) -> Dict:
        """Analyze WITH_TREND vs COUNTER_TREND performance."""
        with_trend = [t for t in trades if t.get("trade_type") == "WITH_TREND"]
        counter_trend = [t for t in trades if t.get("trade_type") == "COUNTER_TREND"]
        
        return {
            "win_rate_with_trend": self._calculate_win_rate(with_trend),
            "win_rate_counter_trend": self._calculate_win_rate(counter_trend),
            "avg_pnl_with_trend": self._calculate_avg_pnl(with_trend),
            "avg_pnl_counter_trend": self._calculate_avg_pnl(counter_trend),
            "trade_count_with_trend": len(with_trend),
            "trade_count_counter_trend": len(counter_trend),
            "trend_alignment_ratio": len(with_trend) / max(len(counter_trend), 1) if counter_trend else float('inf')
        }
    
    def _analyze_session_performance(self, trades: List[Dict]) -> Dict:
        """Analyze performance by trading session."""
        sessions = defaultdict(list)
        for trade in trades:
            session = trade.get("session", "UNKNOWN")
            sessions[session].append(trade)
        
        return {
            "win_rate_london": self._calculate_win_rate(sessions.get("LONDON", [])),
            "win_rate_ny": self._calculate_win_rate(sessions.get("NY", [])),
            "win_rate_overlap": self._calculate_win_rate(sessions.get("OVERLAP", [])),
            "avg_pnl_london": self._calculate_avg_pnl(sessions.get("LONDON", [])),
            "avg_pnl_ny": self._calculate_avg_pnl(sessions.get("NY", [])),
            "avg_pnl_overlap": self._calculate_avg_pnl(sessions.get("OVERLAP", [])),
            "trade_count_london": len(sessions.get("LONDON", [])),
            "trade_count_ny": len(sessions.get("NY", [])),
            "trade_count_overlap": len(sessions.get("OVERLAP", [])),
        }
    
    def _analyze_quality_score_effectiveness(self, trades: List[Dict]) -> Dict:
        """Analyze quality score vs performance."""
        winners = [t for t in trades if t.get("outcome") == "WIN"]
        losers = [t for t in trades if t.get("outcome") == "LOSS"]
        
        winner_scores = [t.get("quality_score", 0) for t in winners]
        loser_scores = [t.get("quality_score", 0) for t in losers]
        
        # Score ranges
        high_score_trades = [t for t in trades if t.get("quality_score", 0) >= 0.75]
        med_score_trades = [t for t in trades if 0.65 <= t.get("quality_score", 0) < 0.75]
        low_score_trades = [t for t in trades if t.get("quality_score", 0) < 0.65]
        
        return {
            "avg_score_winners": statistics.mean(winner_scores) if winner_scores else 0,
            "avg_score_losers": statistics.mean(loser_scores) if loser_scores else 0,
            "score_difference": (statistics.mean(winner_scores) - statistics.mean(loser_scores)) if winner_scores and loser_scores else 0,
            "win_rate_high_score": self._calculate_win_rate(high_score_trades),
            "win_rate_med_score": self._calculate_win_rate(med_score_trades),
            "win_rate_low_score": self._calculate_win_rate(low_score_trades),
            "trade_count_high_score": len(high_score_trades),
            "trade_count_med_score": len(med_score_trades),
            "trade_count_low_score": len(low_score_trades),
        }
    
    def _analyze_execution_filters_impact(self, trades: List[Dict]) -> Dict:
        """Analyze impact of execution filters."""
        # High vs low spread
        high_spread = [t for t in trades if t.get("spread_mean", 0) > 0.25]
        low_spread = [t for t in trades if t.get("spread_mean", 0) <= 0.25]
        
        # High vs low tick ratio
        high_tick_ratio = [t for t in trades if t.get("tick_ratio", 0.5) > 0.65]
        low_tick_ratio = [t for t in trades if t.get("tick_ratio", 0.5) <= 0.65]
        
        # High vs low velocity
        high_velocity = [t for t in trades if t.get("tick_velocity", 0) > 15]
        low_velocity = [t for t in trades if t.get("tick_velocity", 0) <= 15]
        
        return {
            "win_rate_high_spread": self._calculate_win_rate(high_spread),
            "win_rate_low_spread": self._calculate_win_rate(low_spread),
            "win_rate_high_tick_ratio": self._calculate_win_rate(high_tick_ratio),
            "win_rate_low_tick_ratio": self._calculate_win_rate(low_tick_ratio),
            "win_rate_high_velocity": self._calculate_win_rate(high_velocity),
            "win_rate_low_velocity": self._calculate_win_rate(low_velocity),
            "spread_impact": self._calculate_win_rate(low_spread) - self._calculate_win_rate(high_spread),
            "tick_ratio_impact": self._calculate_win_rate(high_tick_ratio) - self._calculate_win_rate(low_tick_ratio),
            "velocity_impact": self._calculate_win_rate(high_velocity) - self._calculate_win_rate(low_velocity),
        }
    
    def _analyze_trade_duration(self, trades: List[Dict]) -> Dict:
        """Analyze trade duration patterns."""
        winners = [t for t in trades if t.get("outcome") == "WIN"]
        losers = [t for t in trades if t.get("outcome") == "LOSS"]
        
        winner_durations = [t.get("trade_duration", 0) for t in winners if t.get("trade_duration")]
        loser_durations = [t.get("trade_duration", 0) for t in losers if t.get("trade_duration")]
        
        return {
            "avg_duration_wins": statistics.mean(winner_durations) if winner_durations else 0,
            "avg_duration_losses": statistics.mean(loser_durations) if loser_durations else 0,
            "median_duration_wins": statistics.median(winner_durations) if winner_durations else 0,
            "median_duration_losses": statistics.median(loser_durations) if loser_durations else 0,
            "duration_difference": (statistics.mean(winner_durations) - statistics.mean(loser_durations)) if winner_durations and loser_durations else 0,
        }
    
    def _analyze_overall_performance(self, trades: List[Dict]) -> Dict:
        """Analyze overall performance metrics."""
        winners = [t for t in trades if t.get("outcome") == "WIN"]
        losers = [t for t in trades if t.get("outcome") == "LOSS"]
        be_trades = [t for t in trades if t.get("outcome") == "BE"]
        
        pnls = [t.get("pnl", 0) for t in trades]
        win_pnls = [t.get("pnl", 0) for t in winners]
        loss_pnls = [t.get("pnl", 0) for t in losers]
        
        # Calculate drawdown
        cumulative_pnl = []
        running_total = 0
        for trade in trades:
            running_total += trade.get("pnl", 0)
            cumulative_pnl.append(running_total)
        
        peak = cumulative_pnl[0]
        max_drawdown = 0
        for pnl in cumulative_pnl:
            if pnl > peak:
                peak = pnl
            drawdown = peak - pnl
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        return {
            "total_trades": len(trades),
            "win_rate": len(winners) / len(trades) if trades else 0,
            "total_pnl": sum(pnls),
            "avg_win": statistics.mean(win_pnls) if win_pnls else 0,
            "avg_loss": statistics.mean(loss_pnls) if loss_pnls else 0,
            "profit_factor": abs(sum(win_pnls) / sum(loss_pnls)) if loss_pnls and sum(loss_pnls) != 0 else float('inf'),
            "max_drawdown": max_drawdown,
            "win_count": len(winners),
            "loss_count": len(losers),
            "be_count": len(be_trades),
        }
    
    def _calculate_win_rate(self, trades: List[Dict]) -> float:
        """Calculate win rate for a list of trades."""
        if not trades:
            return 0.0
        winners = [t for t in trades if t.get("outcome") == "WIN"]
        return len(winners) / len(trades)
    
    def _calculate_avg_pnl(self, trades: List[Dict]) -> float:
        """Calculate average PnL for a list of trades."""
        if not trades:
            return 0.0
        pnls = [t.get("pnl", 0) for t in trades]
        return statistics.mean(pnls)
    
    def get_filter_effectiveness_report(self) -> Dict:
        """Generate report on filter effectiveness."""
        # Get all decisions (taken and skipped)
        all_attributions = get_recent_attributions(hours=72)
        
        taken = [a for a in all_attributions if a.get("decision") == "TRADE_TAKEN"]
        skipped = [a for a in all_attributions if a.get("decision") == "TRADE_SKIPPED"]
        
        # Analyze skip reasons
        skip_reasons = defaultdict(int)
        for attr in skipped:
            reason = attr.get("reason", "Unknown")
            # Extract main reason (before first |)
            main_reason = reason.split("|")[0].strip()
            skip_reasons[main_reason] += 1
        
        return {
            "total_signals": len(all_attributions),
            "trades_taken": len(taken),
            "trades_skipped": len(skipped),
            "conversion_rate": len(taken) / len(all_attributions) if all_attributions else 0,
            "skip_reasons": dict(skip_reasons),
            "most_common_skip": max(skip_reasons.items(), key=lambda x: x[1]) if skip_reasons else ("None", 0),
        }


# Singleton instance
analytics_engine = PerformanceAnalyticsEngine()


def get_performance_metrics(lookback_trades: int = 100) -> Dict:
    """Get comprehensive performance metrics."""
    return analytics_engine.get_performance_metrics(lookback_trades)


def get_filter_effectiveness_report() -> Dict:
    """Get filter effectiveness report."""
    return analytics_engine.get_filter_effectiveness_report()