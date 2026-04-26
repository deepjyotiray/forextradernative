"""
Progress Optimization System for Uniform Backtesting Performance
Addresses non-uniform progress in parallel chunk processing.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timezone, timedelta
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil


class ProgressAnalyzer:
    """Analyzes and optimizes progress uniformity in backtesting."""
    
    def __init__(self):
        self.chunk_stats = {}
        self.progress_history = []
        self.tick_density_cache = {}
    
    def analyze_non_uniform_progress(self, chunk_states: List[Dict]) -> Dict:
        """Analyze why progress is non-uniform across chunks."""
        
        analysis = {
            'chunk_analysis': [],
            'bottlenecks': [],
            'recommendations': [],
            'tick_density_variation': 0.0,
            'processing_speed_variation': 0.0
        }
        
        # Analyze each chunk
        for i, chunk in enumerate(chunk_states):
            start_time = pd.to_datetime(chunk['start_utc'])
            end_time = pd.to_datetime(chunk['end_utc'])
            progress = chunk['progress_pct']
            
            # Estimate tick density for this time period
            estimated_ticks = self._estimate_tick_density(start_time, end_time)
            
            # Calculate processing speed
            if progress > 0:
                processing_speed = estimated_ticks * (progress / 100.0) / max(1, time.time() - chunk.get('start_timestamp', time.time()))
            else:
                processing_speed = 0
            
            chunk_analysis = {
                'chunk_id': i + 1,
                'time_range': f"{start_time.strftime('%Y-%m-%d %H:%M')} -> {end_time.strftime('%Y-%m-%d %H:%M')}",
                'duration_hours': (end_time - start_time).total_seconds() / 3600,
                'progress_pct': progress,
                'estimated_ticks': estimated_ticks,
                'processing_speed_ticks_per_sec': processing_speed,
                'market_session': self._identify_market_session(start_time),
                'expected_complexity': self._calculate_complexity_score(start_time, end_time)
            }
            
            analysis['chunk_analysis'].append(chunk_analysis)
        
        # Identify bottlenecks
        analysis['bottlenecks'] = self._identify_bottlenecks(analysis['chunk_analysis'])
        
        # Generate recommendations
        analysis['recommendations'] = self._generate_optimization_recommendations(analysis['chunk_analysis'])
        
        # Calculate variation metrics
        speeds = [c['processing_speed_ticks_per_sec'] for c in analysis['chunk_analysis'] if c['processing_speed_ticks_per_sec'] > 0]
        if speeds:
            analysis['processing_speed_variation'] = np.std(speeds) / np.mean(speeds) if np.mean(speeds) > 0 else 0
        
        return analysis
    
    def _estimate_tick_density(self, start_time: datetime, end_time: datetime) -> int:
        """Estimate tick density for a time period."""
        
        cache_key = f"{start_time.date()}_{end_time.date()}"
        if cache_key in self.tick_density_cache:
            return self.tick_density_cache[cache_key]
        
        # Base tick rates (ticks per hour) for different market conditions
        base_rates = {
            'high_activity': 3000,    # Major news, market open
            'normal_activity': 1500,  # Regular trading hours
            'low_activity': 500,      # Quiet periods
            'weekend': 50,            # Weekend/holiday
            'asian_session': 800,     # Asian trading hours
            'london_session': 2500,   # London trading hours
            'ny_session': 2000,       # New York trading hours
            'overlap': 3500           # Session overlaps
        }
        
        total_ticks = 0
        current_time = start_time
        
        while current_time < end_time:
            hour_of_day = current_time.hour
            day_of_week = current_time.weekday()
            
            # Determine market session and activity level
            if day_of_week >= 5:  # Weekend
                tick_rate = base_rates['weekend']
            elif 0 <= hour_of_day <= 6:  # Asian session
                tick_rate = base_rates['asian_session']
            elif 7 <= hour_of_day <= 11:  # London session
                tick_rate = base_rates['london_session']
            elif 12 <= hour_of_day <= 16:  # Overlap
                tick_rate = base_rates['overlap']
            elif 17 <= hour_of_day <= 21:  # NY session
                tick_rate = base_rates['ny_session']
            else:  # Low activity
                tick_rate = base_rates['low_activity']
            
            # Add some randomness for realism
            tick_rate *= (0.8 + 0.4 * np.random.random())
            
            total_ticks += int(tick_rate)
            current_time += timedelta(hours=1)
        
        self.tick_density_cache[cache_key] = total_ticks
        return total_ticks
    
    def _identify_market_session(self, timestamp: datetime) -> str:
        """Identify which market session a timestamp falls into."""
        
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        
        if day_of_week >= 5:  # Weekend
            return "weekend"
        elif 0 <= hour <= 6:
            return "asian"
        elif 7 <= hour <= 11:
            return "london"
        elif 12 <= hour <= 16:
            return "overlap"
        elif 17 <= hour <= 21:
            return "new_york"
        else:
            return "low_activity"
    
    def _calculate_complexity_score(self, start_time: datetime, end_time: datetime) -> float:
        """Calculate processing complexity score for a time period."""
        
        # Factors that affect processing complexity
        duration_hours = (end_time - start_time).total_seconds() / 3600
        
        # Market volatility periods (higher complexity)
        volatility_multiplier = 1.0
        
        # Check for high-volatility periods
        hour = start_time.hour
        if 8 <= hour <= 10 or 14 <= hour <= 16:  # Market open times
            volatility_multiplier = 1.5
        elif 12 <= hour <= 14:  # Lunch overlap
            volatility_multiplier = 1.3
        
        # Weekend has lower complexity
        if start_time.weekday() >= 5:
            volatility_multiplier = 0.3
        
        return duration_hours * volatility_multiplier
    
    def _identify_bottlenecks(self, chunk_analysis: List[Dict]) -> List[str]:
        """Identify performance bottlenecks."""
        
        bottlenecks = []
        
        # Find chunks with significantly slower progress
        progresses = [c['progress_pct'] for c in chunk_analysis]
        speeds = [c['processing_speed_ticks_per_sec'] for c in chunk_analysis if c['processing_speed_ticks_per_sec'] > 0]
        
        if progresses:
            avg_progress = np.mean(progresses)
            slow_chunks = [c for c in chunk_analysis if c['progress_pct'] < avg_progress * 0.5]
            
            if slow_chunks:
                bottlenecks.append(f"Slow chunks detected: {len(slow_chunks)} chunks significantly behind average")
        
        if speeds:
            avg_speed = np.mean(speeds)
            very_slow_chunks = [c for c in chunk_analysis if c['processing_speed_ticks_per_sec'] < avg_speed * 0.3]
            
            if very_slow_chunks:
                bottlenecks.append(f"Performance bottleneck: {len(very_slow_chunks)} chunks processing very slowly")
        
        # Check for market session imbalances
        session_progress = {}
        for chunk in chunk_analysis:
            session = chunk['market_session']
            if session not in session_progress:
                session_progress[session] = []
            session_progress[session].append(chunk['progress_pct'])
        
        for session, progresses in session_progress.items():
            if len(progresses) > 1:
                avg_session_progress = np.mean(progresses)
                if avg_session_progress < np.mean([np.mean(p) for p in session_progress.values()]) * 0.7:
                    bottlenecks.append(f"Market session bottleneck: {session} session processing slower than average")
        
        return bottlenecks
    
    def _generate_optimization_recommendations(self, chunk_analysis: List[Dict]) -> List[str]:
        """Generate recommendations to improve progress uniformity."""
        
        recommendations = []
        
        # Analyze tick density variation
        tick_densities = [c['estimated_ticks'] for c in chunk_analysis]
        if tick_densities:
            density_variation = np.std(tick_densities) / np.mean(tick_densities)
            
            if density_variation > 0.5:  # High variation
                recommendations.append("High tick density variation detected. Consider using adaptive chunk sizing based on estimated tick count rather than time periods.")
        
        # Check for weekend chunks
        weekend_chunks = [c for c in chunk_analysis if c['market_session'] == 'weekend']
        if weekend_chunks:
            recommendations.append(f"Weekend periods detected in {len(weekend_chunks)} chunks. Consider merging weekend periods with adjacent weekdays for better load balancing.")
        
        # Check for very small chunks
        small_chunks = [c for c in chunk_analysis if c['duration_hours'] < 24]
        if len(small_chunks) > len(chunk_analysis) * 0.5:
            recommendations.append("Many small chunks detected. Consider increasing minimum chunk size to reduce overhead.")
        
        # Check processing speed variation
        speeds = [c['processing_speed_ticks_per_sec'] for c in chunk_analysis if c['processing_speed_ticks_per_sec'] > 0]
        if speeds:
            speed_variation = np.std(speeds) / np.mean(speeds)
            if speed_variation > 0.3:
                recommendations.append("High processing speed variation. Consider dynamic load balancing or work stealing between workers.")
        
        return recommendations


class AdaptiveChunkManager:
    """Manages adaptive chunk sizing for uniform progress."""
    
    def __init__(self):
        self.tick_density_estimator = ProgressAnalyzer()
    
    def create_balanced_chunks(self, start_date: datetime, end_date: datetime, 
                             target_ticks_per_chunk: int = 100000,
                             max_workers: int = 8) -> List[Tuple[datetime, datetime]]:
        """Create balanced chunks based on estimated tick density."""
        
        print(f"Creating balanced chunks with target {target_ticks_per_chunk:,} ticks per chunk...")
        
        chunks = []
        current_start = start_date
        
        while current_start < end_date:
            # Find end time that gives us approximately target_ticks_per_chunk
            chunk_end = self._find_optimal_chunk_end(
                current_start, end_date, target_ticks_per_chunk
            )
            
            chunks.append((current_start, chunk_end))
            current_start = chunk_end
        
        # Ensure we don't have too many chunks for available workers
        if len(chunks) > max_workers * 2:
            chunks = self._merge_small_chunks(chunks, max_workers * 2)
        
        print(f"Created {len(chunks)} balanced chunks")
        
        # Print chunk analysis
        for i, (start, end) in enumerate(chunks):
            estimated_ticks = self.tick_density_estimator._estimate_tick_density(start, end)
            duration_hours = (end - start).total_seconds() / 3600
            print(f"  Chunk {i+1}: {duration_hours:.1f}h, ~{estimated_ticks:,} ticks")
        
        return chunks
    
    def _find_optimal_chunk_end(self, start_time: datetime, max_end_time: datetime,
                               target_ticks: int) -> datetime:
        """Find optimal end time for a chunk to reach target tick count."""
        
        # Binary search for optimal end time
        min_end = start_time + timedelta(hours=1)  # Minimum 1 hour
        max_end = min(max_end_time, start_time + timedelta(days=14))  # Maximum 2 weeks
        
        best_end = max_end
        
        for _ in range(10):  # Maximum 10 iterations
            mid_end = min_end + (max_end - min_end) / 2
            
            estimated_ticks = self.tick_density_estimator._estimate_tick_density(start_time, mid_end)
            
            if estimated_ticks < target_ticks * 0.8:  # Too few ticks
                min_end = mid_end
            elif estimated_ticks > target_ticks * 1.2:  # Too many ticks
                max_end = mid_end
                best_end = mid_end
            else:  # Good range
                best_end = mid_end
                break
        
        return min(best_end, max_end_time)
    
    def _merge_small_chunks(self, chunks: List[Tuple[datetime, datetime]], 
                           max_chunks: int) -> List[Tuple[datetime, datetime]]:
        """Merge small chunks to reduce total count."""
        
        if len(chunks) <= max_chunks:
            return chunks
        
        # Calculate merge ratio
        merge_ratio = len(chunks) / max_chunks
        
        merged_chunks = []
        i = 0
        
        while i < len(chunks):
            start_time = chunks[i][0]
            
            # Determine how many chunks to merge
            chunks_to_merge = max(1, int(merge_ratio))
            end_idx = min(i + chunks_to_merge, len(chunks))
            
            end_time = chunks[end_idx - 1][1]
            
            merged_chunks.append((start_time, end_time))
            i = end_idx
        
        return merged_chunks


class ProgressMonitor:
    """Real-time progress monitoring and optimization."""
    
    def __init__(self):
        self.chunk_start_times = {}
        self.chunk_progress_history = {}
        self.performance_metrics = {}
        self.lock = threading.Lock()
    
    def start_chunk_monitoring(self, chunk_id: int):
        """Start monitoring a chunk."""
        with self.lock:
            self.chunk_start_times[chunk_id] = time.time()
            self.chunk_progress_history[chunk_id] = []
    
    def update_chunk_progress(self, chunk_id: int, progress_pct: float, 
                            processed_ticks: int, sim_time: str):
        """Update chunk progress."""
        with self.lock:
            timestamp = time.time()
            
            if chunk_id not in self.chunk_progress_history:
                self.chunk_progress_history[chunk_id] = []
            
            self.chunk_progress_history[chunk_id].append({
                'timestamp': timestamp,
                'progress_pct': progress_pct,
                'processed_ticks': processed_ticks,
                'sim_time': sim_time
            })
            
            # Calculate performance metrics
            if chunk_id in self.chunk_start_times:
                elapsed_time = timestamp - self.chunk_start_times[chunk_id]
                if elapsed_time > 0 and processed_ticks > 0:
                    ticks_per_second = processed_ticks / elapsed_time
                    
                    self.performance_metrics[chunk_id] = {
                        'ticks_per_second': ticks_per_second,
                        'elapsed_time': elapsed_time,
                        'progress_rate': progress_pct / elapsed_time if elapsed_time > 0 else 0
                    }
    
    def get_progress_analysis(self) -> Dict:
        """Get comprehensive progress analysis."""
        with self.lock:
            analysis = {
                'chunk_performance': {},
                'overall_stats': {},
                'bottlenecks': [],
                'recommendations': []
            }
            
            # Analyze each chunk
            for chunk_id, metrics in self.performance_metrics.items():
                analysis['chunk_performance'][chunk_id] = {
                    'ticks_per_second': metrics['ticks_per_second'],
                    'elapsed_time': metrics['elapsed_time'],
                    'progress_rate': metrics['progress_rate'],
                    'status': self._classify_chunk_performance(metrics)
                }
            
            # Overall statistics
            if self.performance_metrics:
                speeds = [m['ticks_per_second'] for m in self.performance_metrics.values()]
                rates = [m['progress_rate'] for m in self.performance_metrics.values()]
                
                analysis['overall_stats'] = {
                    'avg_ticks_per_second': np.mean(speeds),
                    'min_ticks_per_second': np.min(speeds),
                    'max_ticks_per_second': np.max(speeds),
                    'speed_variation_coefficient': np.std(speeds) / np.mean(speeds) if np.mean(speeds) > 0 else 0,
                    'avg_progress_rate': np.mean(rates),
                    'total_chunks': len(self.performance_metrics)
                }
                
                # Identify bottlenecks
                avg_speed = np.mean(speeds)
                slow_chunks = [cid for cid, metrics in self.performance_metrics.items() 
                             if metrics['ticks_per_second'] < avg_speed * 0.5]
                
                if slow_chunks:
                    analysis['bottlenecks'].append(f"Slow chunks: {slow_chunks}")
                
                # Generate recommendations
                if analysis['overall_stats']['speed_variation_coefficient'] > 0.3:
                    analysis['recommendations'].append("High speed variation detected - consider load balancing")
                
                if len(slow_chunks) > len(self.performance_metrics) * 0.2:
                    analysis['recommendations'].append("Many slow chunks - consider reducing chunk size or increasing resources")
            
            return analysis
    
    def _classify_chunk_performance(self, metrics: Dict) -> str:
        """Classify chunk performance."""
        tps = metrics['ticks_per_second']
        
        if tps > 50000:
            return "excellent"
        elif tps > 20000:
            return "good"
        elif tps > 10000:
            return "average"
        elif tps > 5000:
            return "slow"
        else:
            return "very_slow"


def optimize_chunk_distribution(start_date: datetime, end_date: datetime,
                              parallel_workers: int = 8) -> List[Tuple[datetime, datetime]]:
    """Optimize chunk distribution for uniform progress."""
    
    print("Optimizing chunk distribution for uniform progress...")
    
    chunk_manager = AdaptiveChunkManager()
    
    # Calculate target ticks per chunk based on total estimated ticks and workers
    total_duration = (end_date - start_date).total_seconds() / 3600  # hours
    estimated_total_ticks = int(total_duration * 1500)  # rough estimate
    
    target_ticks_per_chunk = estimated_total_ticks // (parallel_workers * 2)  # 2 chunks per worker
    target_ticks_per_chunk = max(50000, min(200000, target_ticks_per_chunk))  # reasonable bounds
    
    print(f"Target ticks per chunk: {target_ticks_per_chunk:,}")
    
    # Create balanced chunks
    balanced_chunks = chunk_manager.create_balanced_chunks(
        start_date, end_date, target_ticks_per_chunk, parallel_workers
    )
    
    return balanced_chunks


def create_progress_monitor() -> ProgressMonitor:
    """Create a progress monitor for real-time tracking."""
    return ProgressMonitor()


# Example usage for debugging non-uniform progress
def debug_progress_issues(chunk_states: List[Dict]) -> Dict:
    """Debug progress uniformity issues."""
    
    analyzer = ProgressAnalyzer()
    analysis = analyzer.analyze_non_uniform_progress(chunk_states)
    
    print("PROGRESS ANALYSIS REPORT")
    print("=" * 40)
    
    print(f"\nChunk Analysis:")
    for chunk in analysis['chunk_analysis']:
        print(f"  Chunk {chunk['chunk_id']}: {chunk['progress_pct']:.1f}% "
              f"({chunk['market_session']} session, {chunk['estimated_ticks']:,} ticks)")
    
    if analysis['bottlenecks']:
        print(f"\nBottlenecks Identified:")
        for bottleneck in analysis['bottlenecks']:
            print(f"  • {bottleneck}")
    
    if analysis['recommendations']:
        print(f"\nRecommendations:")
        for rec in analysis['recommendations']:
            print(f"  • {rec}")
    
    return analysis