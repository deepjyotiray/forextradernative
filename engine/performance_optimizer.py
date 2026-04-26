"""
Performance configuration manager for optimizing backtest execution.
"""
import os
import psutil
import math
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class PerformanceProfile:
    """Performance optimization profile."""
    name: str
    max_ticks_per_batch: int
    chunk_size: int
    parallel_workers: int
    use_vectorized: bool
    use_streaming: bool
    cache_indicators: bool
    fast_mode: bool
    memory_limit_mb: int
    compression_level: int


class PerformanceOptimizer:
    """Automatic performance optimization based on system resources and data size."""
    
    def __init__(self):
        self.system_info = self._get_system_info()
        self.profiles = self._create_performance_profiles()
    
    def _get_system_info(self) -> Dict:
        """Get system resource information."""
        memory = psutil.virtual_memory()
        cpu_count = os.cpu_count() or 1
        
        return {
            'cpu_count': cpu_count,
            'memory_total_gb': memory.total / (1024**3),
            'memory_available_gb': memory.available / (1024**3),
            'memory_percent': memory.percent,
            'has_ssd': self._detect_ssd(),
            'python_64bit': self._is_64bit_python()
        }
    
    def _detect_ssd(self) -> bool:
        """Detect if running on SSD (heuristic)."""
        try:
            # Simple heuristic - check if temp directory is on fast storage
            import tempfile
            temp_dir = tempfile.gettempdir()
            
            # On Windows, check drive type
            if os.name == 'nt':
                import win32file
                drive = os.path.splitdrive(temp_dir)[0] + '\\\\'
                drive_type = win32file.GetDriveType(drive)
                return drive_type == win32file.DRIVE_FIXED  # Assume fixed drives are SSD
            
            return True  # Default assumption
        except Exception:
            return True
    
    def _is_64bit_python(self) -> bool:
        """Check if running 64-bit Python."""
        import sys
        return sys.maxsize > 2**32
    
    def _create_performance_profiles(self) -> Dict[str, PerformanceProfile]:
        """Create predefined performance profiles."""
        
        cpu_count = self.system_info['cpu_count']
        memory_gb = self.system_info['memory_available_gb']
        
        return {
            'ultra_fast': PerformanceProfile(
                name='Ultra Fast',
                max_ticks_per_batch=100000,
                chunk_size=100000,
                parallel_workers=min(cpu_count, 16),
                use_vectorized=True,
                use_streaming=True,
                cache_indicators=True,
                fast_mode=True,
                memory_limit_mb=int(memory_gb * 1024 * 0.8),
                compression_level=1
            ),
            
            'balanced': PerformanceProfile(
                name='Balanced',
                max_ticks_per_batch=50000,
                chunk_size=50000,
                parallel_workers=min(cpu_count, 8),
                use_vectorized=True,
                use_streaming=memory_gb < 8,
                cache_indicators=True,
                fast_mode=False,
                memory_limit_mb=int(memory_gb * 1024 * 0.6),
                compression_level=3
            ),
            
            'memory_efficient': PerformanceProfile(
                name='Memory Efficient',
                max_ticks_per_batch=10000,
                chunk_size=10000,
                parallel_workers=min(cpu_count, 4),
                use_vectorized=False,
                use_streaming=True,
                cache_indicators=False,
                fast_mode=True,
                memory_limit_mb=int(memory_gb * 1024 * 0.3),
                compression_level=6
            ),
            
            'high_accuracy': PerformanceProfile(
                name='High Accuracy',
                max_ticks_per_batch=5000,
                chunk_size=25000,
                parallel_workers=min(cpu_count, 4),
                use_vectorized=True,
                use_streaming=False,
                cache_indicators=True,
                fast_mode=False,
                memory_limit_mb=int(memory_gb * 1024 * 0.7),
                compression_level=3
            )
        }
    
    def estimate_dataset_size(self, start_date: datetime, end_date: datetime, 
                            symbol: str = 'XAUUSD') -> Dict:
        """Estimate dataset size and complexity."""
        
        duration_days = (end_date - start_date).days
        
        # Rough estimates based on typical forex data
        ticks_per_day = {
            'XAUUSD': 50000,  # High activity
            'EURUSD': 40000,
            'GBPUSD': 35000,
            'USDJPY': 30000,
            'default': 25000
        }
        
        estimated_ticks = duration_days * ticks_per_day.get(symbol, ticks_per_day['default'])
        
        # Estimate memory usage (bytes per tick)
        bytes_per_tick = 32  # timestamp, bid, ask, spread
        estimated_memory_mb = (estimated_ticks * bytes_per_tick) / (1024 * 1024)
        
        # Estimate processing complexity
        complexity_factors = {
            'duration_days': duration_days,
            'estimated_ticks': estimated_ticks,
            'estimated_memory_mb': estimated_memory_mb,
            'complexity_score': self._calculate_complexity_score(duration_days, estimated_ticks)
        }
        
        return complexity_factors
    
    def _calculate_complexity_score(self, duration_days: int, estimated_ticks: int) -> float:
        """Calculate processing complexity score (0-10)."""
        
        # Base score from duration
        duration_score = min(5.0, duration_days / 30.0)  # 0-5 based on months
        
        # Tick volume score
        tick_score = min(5.0, estimated_ticks / 1000000.0)  # 0-5 based on millions of ticks
        
        return duration_score + tick_score
    
    def recommend_profile(self, start_date: datetime, end_date: datetime,
                         symbol: str = 'XAUUSD', user_preference: str = 'auto') -> PerformanceProfile:
        """Recommend optimal performance profile."""
        
        dataset_info = self.estimate_dataset_size(start_date, end_date, symbol)
        complexity = dataset_info['complexity_score']
        memory_needed_mb = dataset_info['estimated_memory_mb']
        available_memory_mb = self.system_info['memory_available_gb'] * 1024
        
        if user_preference != 'auto':
            return self.profiles.get(user_preference, self.profiles['balanced'])
        
        # Auto-selection logic
        if complexity <= 2.0 and memory_needed_mb < available_memory_mb * 0.3:
            return self.profiles['high_accuracy']
        
        elif complexity <= 5.0 and memory_needed_mb < available_memory_mb * 0.6:
            return self.profiles['balanced']
        
        elif memory_needed_mb > available_memory_mb * 0.8:
            return self.profiles['memory_efficient']
        
        else:
            return self.profiles['ultra_fast']
    
    def optimize_request(self, req, user_profile: str = 'auto') -> Tuple[Dict, PerformanceProfile]:
        """Optimize backtest request parameters."""
        
        profile = self.recommend_profile(
            req.start_utc, req.end_utc, req.symbol, user_profile
        )
        
        # Apply profile settings to request
        optimized_params = {
            'max_ticks_per_batch': profile.max_ticks_per_batch,
            'parallel_workers': profile.parallel_workers,
            'fast_mode': profile.fast_mode,
            'use_vectorized_engine': profile.use_vectorized,
            'use_streaming_engine': profile.use_streaming,
            'memory_limit_mb': profile.memory_limit_mb,
            'enable_indicator_cache': profile.cache_indicators,
            'compression_level': profile.compression_level
        }
        
        # Adjust based on system constraints
        if self.system_info['memory_available_gb'] < 4:
            optimized_params['use_streaming_engine'] = True
            optimized_params['max_ticks_per_batch'] = min(optimized_params['max_ticks_per_batch'], 10000)
        
        if self.system_info['cpu_count'] <= 2:
            optimized_params['parallel_workers'] = 1
        
        return optimized_params, profile
    
    def get_optimization_report(self, req, profile: PerformanceProfile) -> Dict:
        """Generate optimization report."""
        
        dataset_info = self.estimate_dataset_size(req.start_utc, req.end_utc, req.symbol)
        
        # Estimate performance improvement
        baseline_time = dataset_info['estimated_ticks'] / 1000  # Baseline: 1000 ticks/second
        
        speedup_factors = {
            'vectorized': 5.0 if profile.use_vectorized else 1.0,
            'parallel': profile.parallel_workers * 0.8,  # Not perfect scaling
            'streaming': 1.2 if profile.use_streaming else 1.0,
            'fast_mode': 2.0 if profile.fast_mode else 1.0,
            'caching': 1.3 if profile.cache_indicators else 1.0
        }
        
        total_speedup = 1.0
        for factor in speedup_factors.values():
            total_speedup *= factor
        
        estimated_time_seconds = baseline_time / total_speedup
        
        return {
            'profile_name': profile.name,
            'dataset_info': dataset_info,
            'system_info': self.system_info,
            'speedup_factors': speedup_factors,
            'total_speedup': round(total_speedup, 2),
            'estimated_time_seconds': round(estimated_time_seconds, 1),
            'estimated_time_minutes': round(estimated_time_seconds / 60, 1),
            'memory_usage_mb': round(dataset_info['estimated_memory_mb'] / profile.parallel_workers, 1),
            'recommendations': self._generate_recommendations(dataset_info, profile)
        }
    
    def _generate_recommendations(self, dataset_info: Dict, profile: PerformanceProfile) -> List[str]:
        """Generate optimization recommendations."""
        
        recommendations = []
        
        if dataset_info['complexity_score'] > 7:
            recommendations.append("Consider splitting the backtest into smaller date ranges")
        
        if dataset_info['estimated_memory_mb'] > self.system_info['memory_available_gb'] * 1024 * 0.8:
            recommendations.append("Enable streaming mode to reduce memory usage")
        
        if self.system_info['cpu_count'] > profile.parallel_workers:
            recommendations.append(f"Consider increasing parallel workers to {self.system_info['cpu_count']}")
        
        if not profile.use_vectorized:
            recommendations.append("Enable vectorized processing for significant speedup")
        
        if dataset_info['duration_days'] > 90 and not profile.fast_mode:
            recommendations.append("Enable fast mode for long backtests")
        
        return recommendations


class AdaptivePerformanceManager:
    """Adaptive performance management during backtest execution."""
    
    def __init__(self):
        self.optimizer = PerformanceOptimizer()
        self.performance_history = []
        self.current_profile = None
    
    def monitor_performance(self, processed_ticks: int, elapsed_seconds: float,
                          memory_usage_mb: float) -> Dict:
        """Monitor current performance metrics."""
        
        ticks_per_second = processed_ticks / elapsed_seconds if elapsed_seconds > 0 else 0
        
        metrics = {
            'timestamp': datetime.now(),
            'ticks_per_second': ticks_per_second,
            'memory_usage_mb': memory_usage_mb,
            'elapsed_seconds': elapsed_seconds,
            'processed_ticks': processed_ticks
        }
        
        self.performance_history.append(metrics)
        
        # Keep only recent history
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-50:]
        
        return metrics
    
    def should_adjust_settings(self) -> Tuple[bool, Dict]:
        """Determine if settings should be adjusted based on performance."""
        
        if len(self.performance_history) < 5:
            return False, {}
        
        recent_metrics = self.performance_history[-5:]
        avg_tps = sum(m['ticks_per_second'] for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m['memory_usage_mb'] for m in recent_metrics) / len(recent_metrics)
        
        adjustments = {}
        
        # Memory pressure detection
        available_memory_mb = self.optimizer.system_info['memory_available_gb'] * 1024
        if avg_memory > available_memory_mb * 0.9:
            adjustments['reduce_batch_size'] = True
            adjustments['enable_streaming'] = True
        
        # Performance degradation detection
        if len(self.performance_history) >= 10:
            older_metrics = self.performance_history[-10:-5]
            older_avg_tps = sum(m['ticks_per_second'] for m in older_metrics) / len(older_metrics)
            
            if avg_tps < older_avg_tps * 0.8:  # 20% performance drop
                adjustments['performance_degraded'] = True
        
        return len(adjustments) > 0, adjustments
    
    def get_adaptive_recommendations(self) -> Dict:
        """Get adaptive recommendations based on runtime performance."""
        
        should_adjust, adjustments = self.should_adjust_settings()
        
        if not should_adjust:
            return {'status': 'optimal', 'adjustments': {}}
        
        recommendations = {}
        
        if adjustments.get('reduce_batch_size'):
            recommendations['max_ticks_per_batch'] = 5000
        
        if adjustments.get('enable_streaming'):
            recommendations['use_streaming'] = True
        
        if adjustments.get('performance_degraded'):
            recommendations['restart_workers'] = True
        
        return {
            'status': 'needs_adjustment',
            'adjustments': recommendations,
            'reason': adjustments
        }


# Global performance optimizer instance
performance_optimizer = PerformanceOptimizer()
adaptive_manager = AdaptivePerformanceManager()


def optimize_backtest_performance(req, user_preference: str = 'auto') -> Dict:
    """Main function to optimize backtest performance."""
    
    optimized_params, profile = performance_optimizer.optimize_request(req, user_preference)
    report = performance_optimizer.get_optimization_report(req, profile)
    
    return {
        'optimized_params': optimized_params,
        'profile': profile,
        'report': report,
        'system_info': performance_optimizer.system_info
    }