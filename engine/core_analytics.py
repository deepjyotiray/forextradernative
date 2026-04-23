"""
Core Analytics Engine - Performance metrics, parameter sensitivity, and trade segmentation.

Components:
- Performance metrics calculator
- Parameter sensitivity analyzer
- Trade segmentation engine
- Statistical analysis functions
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import statistics
from collections import defaultdict
from .data_pipeline import get_completed_trades_only, get_trade_data


class PerformanceMetricsCalculator:
    """Calculate comprehensive performance metrics from trade data."""
    
    @staticmethod
    def calculate_basic_metrics(df: pd.DataFrame) -> Dict:
        """Calculate basic performance metrics."""
        if df.empty:
            return {"error": "No trade data available"}
        
        completed_trades = df.dropna(subset=['outcome'])
        if completed_trades.empty:
            return {"error": "No completed trades available"}
        
        winners = completed_trades[completed_trades['outcome'] == 'WIN']
        losers = completed_trades[completed_trades['outcome'] == 'LOSS']
        be_trades = completed_trades[completed_trades['outcome'] == 'BE']
        
        pnls = completed_trades['pnl'].values
        win_pnls = winners['pnl'].values
        loss_pnls = losers['pnl'].values
        
        # Calculate drawdown
        cumulative_pnl = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdowns = running_max - cumulative_pnl
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0
        
        return {
            "total_trades": len(completed_trades),
            "win_count": len(winners),
            "loss_count": len(losers),
            "be_count": len(be_trades),
            "win_rate": len(winners) / len(completed_trades) if len(completed_trades) > 0 else 0,
            "total_pnl": float(np.sum(pnls)),
            "avg_win": float(np.mean(win_pnls)) if len(win_pnls) > 0 else 0,
            "avg_loss": float(np.mean(loss_pnls)) if len(loss_pnls) > 0 else 0,
            "profit_factor": abs(np.sum(win_pnls) / np.sum(loss_pnls)) if len(loss_pnls) > 0 and np.sum(loss_pnls) != 0 else float('inf'),
            "max_drawdown": float(max_drawdown),
            "avg_trade": float(np.mean(pnls)),
            "std_dev": float(np.std(pnls)),
            "sharpe_ratio": float(np.mean(pnls) / np.std(pnls)) if np.std(pnls) != 0 else 0,
            "best_trade": float(np.max(pnls)) if len(pnls) > 0 else 0,
            "worst_trade": float(np.min(pnls)) if len(pnls) > 0 else 0,
        }
    
    @staticmethod
    def calculate_directional_metrics(df: pd.DataFrame) -> Dict:
        """Calculate LONG vs SHORT performance metrics."""
        completed_trades = df.dropna(subset=['outcome'])
        
        long_trades = completed_trades[completed_trades['setup_direction'] == 'LONG']
        short_trades = completed_trades[completed_trades['setup_direction'] == 'SHORT']
        
        def analyze_direction(trades_df, direction_name):
            if trades_df.empty:
                return {"trade_count": 0, "direction": direction_name}
            
            winners = trades_df[trades_df['outcome'] == 'WIN']
            pnls = trades_df['pnl'].values
            durations = trades_df['trade_duration'].dropna().values
            
            return {
                "direction": direction_name,
                "trade_count": len(trades_df),
                "win_count": len(winners),
                "win_rate": len(winners) / len(trades_df),
                "total_pnl": float(np.sum(pnls)),
                "avg_pnl": float(np.mean(pnls)),
                "avg_duration": float(np.mean(durations)) if len(durations) > 0 else 0,
                "best_trade": float(np.max(pnls)) if len(pnls) > 0 else 0,
                "worst_trade": float(np.min(pnls)) if len(pnls) > 0 else 0,
            }
        
        long_metrics = analyze_direction(long_trades, "LONG")
        short_metrics = analyze_direction(short_trades, "SHORT")
        
        return {
            "long": long_metrics,
            "short": short_metrics,
            "direction_balance": long_metrics["trade_count"] / max(short_metrics["trade_count"], 1),
            "better_direction": "LONG" if long_metrics.get("win_rate", 0) > short_metrics.get("win_rate", 0) else "SHORT"
        }
    
    @staticmethod
    def calculate_session_metrics(df: pd.DataFrame) -> Dict:
        """Calculate performance by trading session."""
        completed_trades = df.dropna(subset=['outcome'])
        sessions = ["LONDON", "OVERLAP", "NY"]
        
        session_metrics = {}
        
        for session in sessions:
            session_trades = completed_trades[completed_trades['session'] == session]
            
            if session_trades.empty:
                session_metrics[session] = {"trade_count": 0, "session": session}
                continue
            
            winners = session_trades[session_trades['outcome'] == 'WIN']
            pnls = session_trades['pnl'].values
            
            session_metrics[session] = {
                "session": session,
                "trade_count": len(session_trades),
                "win_count": len(winners),
                "win_rate": len(winners) / len(session_trades),
                "total_pnl": float(np.sum(pnls)),
                "avg_pnl": float(np.mean(pnls)),
                "best_trade": float(np.max(pnls)) if len(pnls) > 0 else 0,
                "worst_trade": float(np.min(pnls)) if len(pnls) > 0 else 0,
            }
        
        # Find best and worst sessions
        active_sessions = {k: v for k, v in session_metrics.items() if v["trade_count"] > 0}
        best_session = max(active_sessions.items(), key=lambda x: x[1]["win_rate"]) if active_sessions else None
        worst_session = min(active_sessions.items(), key=lambda x: x[1]["win_rate"]) if active_sessions else None
        
        return {
            "sessions": session_metrics,
            "best_session": best_session[0] if best_session else None,
            "worst_session": worst_session[0] if worst_session else None,
        }
    
    @staticmethod
    def calculate_bias_alignment_metrics(df: pd.DataFrame) -> Dict:
        """Calculate WITH_TREND vs COUNTER_TREND performance."""
        completed_trades = df.dropna(subset=['outcome'])
        
        with_trend = completed_trades[completed_trades['trade_type'] == 'WITH_TREND']
        counter_trend = completed_trades[completed_trades['trade_type'] == 'COUNTER_TREND']
        
        def analyze_trade_type(trades_df, type_name):
            if trades_df.empty:
                return {"trade_count": 0, "type": type_name}
            
            winners = trades_df[trades_df['outcome'] == 'WIN']
            pnls = trades_df['pnl'].values
            
            return {
                "type": type_name,
                "trade_count": len(trades_df),
                "win_count": len(winners),
                "win_rate": len(winners) / len(trades_df),
                "total_pnl": float(np.sum(pnls)),
                "avg_pnl": float(np.mean(pnls)),
            }
        
        return {
            "with_trend": analyze_trade_type(with_trend, "WITH_TREND"),
            "counter_trend": analyze_trade_type(counter_trend, "COUNTER_TREND"),
            "trend_alignment_ratio": len(with_trend) / max(len(counter_trend), 1) if len(counter_trend) > 0 else float('inf')
        }


class ParameterSensitivityAnalyzer:
    """Analyze parameter sensitivity by simulating changes on historical data."""
    
    @staticmethod
    def analyze_quality_threshold_sensitivity(df: pd.DataFrame, thresholds: List[float] = None) -> Dict:
        """Analyze sensitivity to quality score thresholds."""
        if thresholds is None:
            thresholds = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
        
        completed_trades = df.dropna(subset=['outcome'])
        if completed_trades.empty:
            return {"error": "No completed trades for analysis"}
        
        results = {}
        
        for threshold in thresholds:
            # Filter trades that would have been taken at this threshold
            filtered_trades = completed_trades[completed_trades['quality_score'] >= threshold]
            
            if filtered_trades.empty:
                results[threshold] = {
                    "threshold": threshold,
                    "trade_count": 0,
                    "win_rate": 0,
                    "avg_pnl": 0,
                    "total_pnl": 0,
                    "max_drawdown": 0
                }
                continue
            
            winners = filtered_trades[filtered_trades['outcome'] == 'WIN']
            pnls = filtered_trades['pnl'].values
            
            # Calculate drawdown for this threshold
            cumulative_pnl = np.cumsum(pnls)
            running_max = np.maximum.accumulate(cumulative_pnl)
            drawdowns = running_max - cumulative_pnl
            max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0
            
            results[threshold] = {
                "threshold": threshold,
                "trade_count": len(filtered_trades),
                "win_rate": len(winners) / len(filtered_trades),
                "avg_pnl": float(np.mean(pnls)),
                "total_pnl": float(np.sum(pnls)),
                "max_drawdown": float(max_drawdown),
                "profit_factor": abs(np.sum(filtered_trades[filtered_trades['outcome'] == 'WIN']['pnl']) / 
                                   np.sum(filtered_trades[filtered_trades['outcome'] == 'LOSS']['pnl'])) 
                                   if len(filtered_trades[filtered_trades['outcome'] == 'LOSS']) > 0 else float('inf')
            }
        
        # Find optimal threshold
        valid_results = {k: v for k, v in results.items() if v["trade_count"] >= 5}
        optimal_threshold = max(valid_results.items(), 
                              key=lambda x: x[1]["win_rate"] * np.log(x[1]["trade_count"])) if valid_results else None
        
        return {
            "analysis": results,
            "optimal_threshold": optimal_threshold[0] if optimal_threshold else None,
            "optimal_metrics": optimal_threshold[1] if optimal_threshold else None
        }
    
    @staticmethod
    def analyze_tick_ratio_sensitivity(df: pd.DataFrame, thresholds: List[float] = None) -> Dict:
        """Analyze sensitivity to tick ratio thresholds."""
        if thresholds is None:
            thresholds = [0.55, 0.60, 0.65, 0.70, 0.75]
        
        completed_trades = df.dropna(subset=['outcome', 'tick_ratio'])
        if completed_trades.empty:
            return {"error": "No completed trades with tick ratio data"}
        
        results = {}
        
        for threshold in thresholds:
            # Filter trades based on tick ratio threshold
            # For LONG: tick_ratio >= threshold, for SHORT: tick_ratio <= (1-threshold)
            long_trades = completed_trades[
                (completed_trades['setup_direction'] == 'LONG') & 
                (completed_trades['tick_ratio'] >= threshold)
            ]
            short_trades = completed_trades[
                (completed_trades['setup_direction'] == 'SHORT') & 
                (completed_trades['tick_ratio'] <= (1.0 - threshold))
            ]
            
            filtered_trades = pd.concat([long_trades, short_trades])
            
            if filtered_trades.empty:
                results[threshold] = {
                    "threshold": threshold,
                    "trade_count": 0,
                    "win_rate": 0,
                    "avg_pnl": 0
                }
                continue
            
            winners = filtered_trades[filtered_trades['outcome'] == 'WIN']
            pnls = filtered_trades['pnl'].values
            
            results[threshold] = {
                "threshold": threshold,
                "trade_count": len(filtered_trades),
                "win_rate": len(winners) / len(filtered_trades),
                "avg_pnl": float(np.mean(pnls)),
                "total_pnl": float(np.sum(pnls))
            }
        
        return {"analysis": results}
    
    @staticmethod
    def analyze_sl_tp_multiplier_sensitivity(df: pd.DataFrame, 
                                           sl_multipliers: List[float] = None,
                                           tp_multipliers: List[float] = None) -> Dict:
        """Analyze sensitivity to SL/TP multipliers."""
        if sl_multipliers is None:
            sl_multipliers = [0.5, 0.6, 0.7]
        if tp_multipliers is None:
            tp_multipliers = [0.8, 1.0, 1.2]
        
        completed_trades = df.dropna(subset=['outcome', 'atr_value', 'entry_price'])
        if completed_trades.empty:
            return {"error": "No completed trades with required data"}
        
        results = {}
        
        for sl_mult in sl_multipliers:
            for tp_mult in tp_multipliers:
                key = f"SL_{sl_mult}_TP_{tp_mult}"
                
                # Simulate trades with these multipliers
                simulated_results = []
                
                for _, trade in completed_trades.iterrows():
                    entry_price = trade['entry_price']
                    exit_price = trade['exit_price']
                    atr = trade['atr_value']
                    direction = trade['setup_direction']
                    
                    # Calculate new SL/TP levels
                    if direction == 'LONG':
                        new_sl = entry_price - (atr * sl_mult)
                        new_tp = entry_price + (atr * tp_mult)
                        
                        # Determine outcome with new levels
                        if exit_price <= new_sl:
                            outcome = 'LOSS'
                            pnl = new_sl - entry_price
                        elif exit_price >= new_tp:
                            outcome = 'WIN'
                            pnl = new_tp - entry_price
                        else:
                            outcome = trade['outcome']  # Keep original
                            pnl = exit_price - entry_price
                    else:  # SHORT
                        new_sl = entry_price + (atr * sl_mult)
                        new_tp = entry_price - (atr * tp_mult)
                        
                        if exit_price >= new_sl:
                            outcome = 'LOSS'
                            pnl = entry_price - new_sl
                        elif exit_price <= new_tp:
                            outcome = 'WIN'
                            pnl = entry_price - new_tp
                        else:
                            outcome = trade['outcome']
                            pnl = entry_price - exit_price
                    
                    simulated_results.append({
                        'outcome': outcome,
                        'pnl': pnl
                    })
                
                # Calculate metrics for this combination
                sim_df = pd.DataFrame(simulated_results)
                winners = sim_df[sim_df['outcome'] == 'WIN']
                pnls = sim_df['pnl'].values
                
                results[key] = {
                    "sl_multiplier": sl_mult,
                    "tp_multiplier": tp_mult,
                    "trade_count": len(sim_df),
                    "win_rate": len(winners) / len(sim_df) if len(sim_df) > 0 else 0,
                    "avg_pnl": float(np.mean(pnls)) if len(pnls) > 0 else 0,
                    "total_pnl": float(np.sum(pnls)) if len(pnls) > 0 else 0
                }
        
        return {"analysis": results}


class TradeSegmentationEngine:
    """Segment trades by various criteria for detailed analysis."""
    
    @staticmethod
    def segment_by_quality_score(df: pd.DataFrame, ranges: List[Tuple[float, float]] = None) -> Dict:
        """Segment trades by quality score ranges."""
        if ranges is None:
            ranges = [(0.0, 0.65), (0.65, 0.75), (0.75, 1.0)]
        
        completed_trades = df.dropna(subset=['outcome', 'quality_score'])
        segments = {}
        
        for min_score, max_score in ranges:
            range_key = f"{min_score:.2f}-{max_score:.2f}"
            segment_trades = completed_trades[
                (completed_trades['quality_score'] >= min_score) & 
                (completed_trades['quality_score'] < max_score)
            ]
            
            if segment_trades.empty:
                segments[range_key] = {"trade_count": 0, "range": range_key}
                continue
            
            winners = segment_trades[segment_trades['outcome'] == 'WIN']
            pnls = segment_trades['pnl'].values
            
            segments[range_key] = {
                "range": range_key,
                "trade_count": len(segment_trades),
                "win_count": len(winners),
                "win_rate": len(winners) / len(segment_trades),
                "avg_pnl": float(np.mean(pnls)),
                "total_pnl": float(np.sum(pnls)),
                "avg_quality_score": float(np.mean(segment_trades['quality_score']))
            }
        
        return segments
    
    @staticmethod
    def segment_by_spread_ranges(df: pd.DataFrame, ranges: List[Tuple[float, float]] = None) -> Dict:
        """Segment trades by spread ranges."""
        if ranges is None:
            ranges = [(0.0, 0.20), (0.20, 0.30), (0.30, 0.50)]
        
        completed_trades = df.dropna(subset=['outcome', 'spread_mean'])
        segments = {}
        
        for min_spread, max_spread in ranges:
            range_key = f"{min_spread:.2f}-{max_spread:.2f}"
            segment_trades = completed_trades[
                (completed_trades['spread_mean'] >= min_spread) & 
                (completed_trades['spread_mean'] < max_spread)
            ]
            
            if segment_trades.empty:
                segments[range_key] = {"trade_count": 0, "range": range_key}
                continue
            
            winners = segment_trades[segment_trades['outcome'] == 'WIN']
            pnls = segment_trades['pnl'].values
            
            segments[range_key] = {
                "range": range_key,
                "trade_count": len(segment_trades),
                "win_rate": len(winners) / len(segment_trades),
                "avg_pnl": float(np.mean(pnls)),
                "avg_spread": float(np.mean(segment_trades['spread_mean']))
            }
        
        return segments
    
    @staticmethod
    def segment_by_atr_ranges(df: pd.DataFrame, ranges: List[Tuple[float, float]] = None) -> Dict:
        """Segment trades by ATR ranges."""
        completed_trades = df.dropna(subset=['outcome', 'atr_value'])
        
        if ranges is None:
            # Auto-generate ranges based on data
            atr_values = completed_trades['atr_value'].values
            q25, q50, q75 = np.percentile(atr_values, [25, 50, 75])
            ranges = [(0, q25), (q25, q50), (q50, q75), (q75, np.max(atr_values))]
        
        segments = {}
        
        for min_atr, max_atr in ranges:
            range_key = f"{min_atr:.3f}-{max_atr:.3f}"
            segment_trades = completed_trades[
                (completed_trades['atr_value'] >= min_atr) & 
                (completed_trades['atr_value'] <= max_atr)
            ]
            
            if segment_trades.empty:
                segments[range_key] = {"trade_count": 0, "range": range_key}
                continue
            
            winners = segment_trades[segment_trades['outcome'] == 'WIN']
            pnls = segment_trades['pnl'].values
            
            segments[range_key] = {
                "range": range_key,
                "trade_count": len(segment_trades),
                "win_rate": len(winners) / len(segment_trades),
                "avg_pnl": float(np.mean(pnls)),
                "avg_atr": float(np.mean(segment_trades['atr_value']))
            }
        
        return segments


class StatisticalAnalyzer:
    """Statistical analysis functions for trade data."""
    
    @staticmethod
    def calculate_consecutive_streaks(df: pd.DataFrame) -> Dict:
        """Calculate consecutive win/loss streaks."""
        completed_trades = df.dropna(subset=['outcome']).sort_values('unix_time')
        
        if completed_trades.empty:
            return {"error": "No completed trades"}
        
        outcomes = completed_trades['outcome'].values
        
        # Calculate streaks
        current_streak = 1
        max_win_streak = 0
        max_loss_streak = 0
        current_win_streak = 0
        current_loss_streak = 0
        
        for i in range(len(outcomes)):
            if i == 0:
                if outcomes[i] == 'WIN':
                    current_win_streak = 1
                elif outcomes[i] == 'LOSS':
                    current_loss_streak = 1
            else:
                if outcomes[i] == outcomes[i-1]:
                    current_streak += 1
                else:
                    current_streak = 1
                
                if outcomes[i] == 'WIN':
                    current_win_streak += 1 if outcomes[i-1] == 'WIN' else 1
                    current_loss_streak = 0
                elif outcomes[i] == 'LOSS':
                    current_loss_streak += 1 if outcomes[i-1] == 'LOSS' else 1
                    current_win_streak = 0
                
                max_win_streak = max(max_win_streak, current_win_streak)
                max_loss_streak = max(max_loss_streak, current_loss_streak)
        
        return {
            "max_win_streak": max_win_streak,
            "max_loss_streak": max_loss_streak,
            "current_win_streak": current_win_streak if outcomes[-1] == 'WIN' else 0,
            "current_loss_streak": current_loss_streak if outcomes[-1] == 'LOSS' else 0
        }
    
    @staticmethod
    def calculate_trade_duration_stats(df: pd.DataFrame) -> Dict:
        """Calculate trade duration statistics."""
        completed_trades = df.dropna(subset=['outcome', 'trade_duration'])
        
        if completed_trades.empty:
            return {"error": "No completed trades with duration data"}
        
        winners = completed_trades[completed_trades['outcome'] == 'WIN']
        losers = completed_trades[completed_trades['outcome'] == 'LOSS']
        
        all_durations = completed_trades['trade_duration'].values
        win_durations = winners['trade_duration'].values
        loss_durations = losers['trade_duration'].values
        
        return {
            "all_trades": {
                "mean": float(np.mean(all_durations)),
                "median": float(np.median(all_durations)),
                "std": float(np.std(all_durations)),
                "min": float(np.min(all_durations)),
                "max": float(np.max(all_durations))
            },
            "winners": {
                "mean": float(np.mean(win_durations)) if len(win_durations) > 0 else 0,
                "median": float(np.median(win_durations)) if len(win_durations) > 0 else 0,
                "count": len(win_durations)
            },
            "losers": {
                "mean": float(np.mean(loss_durations)) if len(loss_durations) > 0 else 0,
                "median": float(np.median(loss_durations)) if len(loss_durations) > 0 else 0,
                "count": len(loss_durations)
            }
        }


# Main analytics interface
class CoreAnalyticsEngine:
    """Main interface for all analytics functions."""
    
    def __init__(self):
        self.metrics_calc = PerformanceMetricsCalculator()
        self.sensitivity_analyzer = ParameterSensitivityAnalyzer()
        self.segmentation_engine = TradeSegmentationEngine()
        self.statistical_analyzer = StatisticalAnalyzer()
    
    def get_comprehensive_analysis(self, days: int = 30) -> Dict:
        """Get comprehensive analysis of trade performance."""
        df = get_completed_trades_only(days)
        
        if df.empty:
            return {"error": f"No completed trades in last {days} days"}
        
        return {
            "period_days": days,
            "data_range": {
                "start": df['timestamp'].min().isoformat() if not df.empty else None,
                "end": df['timestamp'].max().isoformat() if not df.empty else None,
                "trade_count": len(df)
            },
            "basic_metrics": self.metrics_calc.calculate_basic_metrics(df),
            "directional_metrics": self.metrics_calc.calculate_directional_metrics(df),
            "session_metrics": self.metrics_calc.calculate_session_metrics(df),
            "bias_alignment_metrics": self.metrics_calc.calculate_bias_alignment_metrics(df),
            "consecutive_streaks": self.statistical_analyzer.calculate_consecutive_streaks(df),
            "duration_stats": self.statistical_analyzer.calculate_trade_duration_stats(df),
            "quality_segments": self.segmentation_engine.segment_by_quality_score(df),
            "spread_segments": self.segmentation_engine.segment_by_spread_ranges(df),
            "atr_segments": self.segmentation_engine.segment_by_atr_ranges(df)
        }
    
    def get_parameter_sensitivity_analysis(self, days: int = 30) -> Dict:
        """Get parameter sensitivity analysis."""
        df = get_completed_trades_only(days)
        
        if df.empty:
            return {"error": f"No completed trades in last {days} days"}
        
        return {
            "period_days": days,
            "trade_count": len(df),
            "quality_threshold_sensitivity": self.sensitivity_analyzer.analyze_quality_threshold_sensitivity(df),
            "tick_ratio_sensitivity": self.sensitivity_analyzer.analyze_tick_ratio_sensitivity(df),
            "sl_tp_sensitivity": self.sensitivity_analyzer.analyze_sl_tp_multiplier_sensitivity(df)
        }


# Singleton instance
analytics_engine = CoreAnalyticsEngine()


def get_comprehensive_analysis(days: int = 30) -> Dict:
    """Get comprehensive trade analysis."""
    return analytics_engine.get_comprehensive_analysis(days)


def get_parameter_sensitivity_analysis(days: int = 30) -> Dict:
    """Get parameter sensitivity analysis."""
    return analytics_engine.get_parameter_sensitivity_analysis(days)


def calculate_basic_metrics(days: int = 30) -> Dict:
    """Get basic performance metrics."""
    df = get_completed_trades_only(days)
    return PerformanceMetricsCalculator.calculate_basic_metrics(df)


def segment_trades_by_quality(days: int = 30) -> Dict:
    """Segment trades by quality score."""
    df = get_completed_trades_only(days)
    return TradeSegmentationEngine.segment_by_quality_score(df)