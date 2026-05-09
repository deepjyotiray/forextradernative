"""
Integration layer for optimized backtesting engines.
"""
from typing import Dict, Optional, Callable
import time
import logging
from datetime import datetime

from .backtest_sim import BacktestRequest, BacktestRunner
from .backtest_data import BacktestDataProvider, BacktestDataset
from .backtest_vectorized import VectorizedBacktestEngine, ParallelVectorizedEngine
from .backtest_streaming import StreamingBacktestEngine, create_optimized_backtest_runner
from .performance_optimizer import performance_optimizer, adaptive_manager
from .indicators_optimized import optimized_indicators


class OptimizedBacktestRunner:
    """Enhanced backtest runner with automatic optimization."""
    
    def __init__(self, auto_optimize: bool = True):
        self.auto_optimize = auto_optimize
        self.original_runner = BacktestRunner()
        self.vectorized_engine = None
        self.streaming_engine = None
        self.performance_log = []
    
    def run(self, req: BacktestRequest, dataset: BacktestDataset,
            progress_cb: Optional[Callable[[Dict], None]] = None,
            is_cancelled: Optional[Callable[[], bool]] = None) -> Dict:
        """Enhanced run method with automatic optimization."""
        
        start_time = time.time()
        
        # Get optimization recommendations
        if self.auto_optimize:
            optimization = performance_optimizer.optimize_backtest_performance(req)
            optimized_params = optimization['optimized_params']
            profile = optimization['profile']
            
            logging.info(f"Using performance profile: {profile.name}")
            logging.info(f"Estimated speedup: {optimization['report']['total_speedup']}x")
            
            # Choose optimal engine
            if optimized_params['use_streaming_engine']:
                return self._run_streaming(req, dataset, optimized_params, progress_cb, is_cancelled)
            elif optimized_params['use_vectorized_engine']:
                return self._run_vectorized(req, dataset, optimized_params, progress_cb, is_cancelled)
        
        # Fallback to original engine
        return self._run_original(req, dataset, progress_cb, is_cancelled)
    
    def _run_streaming(self, req: BacktestRequest, dataset: BacktestDataset,
                      params: Dict, progress_cb: Optional[Callable], 
                      is_cancelled: Optional[Callable]) -> Dict:
        """Run using streaming engine."""
        
        if not self.streaming_engine:
            self.streaming_engine = StreamingBacktestEngine(
                max_memory_mb=params['memory_limit_mb']
            )
        
        # Convert dataset to file-based format for streaming
        temp_files = self._prepare_streaming_data(dataset)
        
        try:
            # Enhanced progress callback with performance monitoring
            def enhanced_progress(update: Dict):
                self._monitor_performance(update)
                if progress_cb:
                    progress_cb(update)
            
            result = self.streaming_engine.run_streaming(
                tick_file=temp_files['ticks'],
                candle_files=temp_files['candles'],
                req=req
            )
            
            return self._enhance_result(result, 'streaming')
            
        finally:
            self._cleanup_temp_files(temp_files)
    
    def _run_vectorized(self, req: BacktestRequest, dataset: BacktestDataset,
                       params: Dict, progress_cb: Optional[Callable],
                       is_cancelled: Optional[Callable]) -> Dict:
        """Run using vectorized engine."""
        
        if params['parallel_workers'] > 1:
            if not self.vectorized_engine:
                self.vectorized_engine = ParallelVectorizedEngine(
                    max_workers=params['parallel_workers']
                )
            
            result = self.vectorized_engine.run_parallel(
                req, dataset.ticks, dataset.candles
            )
        else:
            engine = VectorizedBacktestEngine(
                chunk_size=params.get('chunk_size', 50000)
            )
            result = engine.run_vectorized(
                req, dataset.ticks, dataset.candles
            )
        
        return self._enhance_result(result, 'vectorized')
    
    def _run_original(self, req: BacktestRequest, dataset: BacktestDataset,
                     progress_cb: Optional[Callable],
                     is_cancelled: Optional[Callable]) -> Dict:
        """Run using original engine with optimized indicators."""
        
        # Monkey patch indicators for performance
        original_compute_indicators = None
        try:
            from . import indicators
            original_compute_indicators = indicators.compute_indicators
            indicators.compute_indicators = self._optimized_compute_indicators
            
            result = self.original_runner.run(req, dataset, progress_cb, is_cancelled)
            return self._enhance_result(result, 'original_optimized')
            
        finally:
            # Restore original function
            if original_compute_indicators:
                from . import indicators
                indicators.compute_indicators = original_compute_indicators
    
    def _optimized_compute_indicators(self, df):
        """Optimized indicator computation."""
        from .indicators_optimized import compute_indicators_fast
        return compute_indicators_fast(df)
    
    def _prepare_streaming_data(self, dataset: BacktestDataset) -> Dict[str, str]:
        """Prepare data files for streaming engine."""
        import tempfile
        import os
        
        temp_dir = tempfile.mkdtemp(prefix='backtest_streaming_')
        
        # Save tick data
        tick_file = os.path.join(temp_dir, 'ticks.parquet')
        dataset.ticks.to_parquet(tick_file, index=False)
        
        # Save candle data
        candle_files = {}
        for tf, df in dataset.candles.items():
            candle_file = os.path.join(temp_dir, f'{tf}.parquet')
            df.to_parquet(candle_file, index=False)
            candle_files[tf] = candle_file
        
        return {
            'ticks': tick_file,
            'candles': candle_files,
            'temp_dir': temp_dir
        }
    
    def _cleanup_temp_files(self, temp_files: Dict):
        """Clean up temporary files."""
        import shutil
        
        if 'temp_dir' in temp_files:
            try:
                shutil.rmtree(temp_files['temp_dir'])
            except Exception as e:
                logging.warning(f"Failed to cleanup temp files: {e}")
    
    def _monitor_performance(self, update: Dict):
        """Monitor performance during execution."""
        
        processed = update.get('processed_ticks', 0)
        if processed > 0 and len(self.performance_log) > 0:
            elapsed = time.time() - self.performance_log[0]['start_time']
            
            # Get memory usage
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)
            
            # Monitor with adaptive manager
            adaptive_manager.monitor_performance(processed, elapsed, memory_mb)
            
            # Check for adaptive recommendations
            should_adjust, adjustments = adaptive_manager.should_adjust_settings()
            if should_adjust:
                logging.info(f"Performance adjustment recommended: {adjustments}")
    
    def _enhance_result(self, result: Dict, engine_type: str) -> Dict:
        """Enhance result with performance metadata."""
        
        if 'meta' not in result:
            result['meta'] = {}
        
        result['meta']['engine_type'] = engine_type
        result['meta']['optimization_enabled'] = self.auto_optimize
        
        if self.performance_log:
            result['meta']['performance_log'] = self.performance_log[-10:]  # Last 10 entries
        
        return result


class BacktestEngineFactory:
    """Factory for creating optimized backtest engines."""
    
    @staticmethod
    def create_engine(engine_type: str = 'auto', **kwargs):
        """Create backtest engine based on type."""
        
        if engine_type == 'auto':
            return OptimizedBacktestRunner(auto_optimize=True)
        
        elif engine_type == 'vectorized':
            parallel_workers = kwargs.get('parallel_workers', 1)
            if parallel_workers > 1:
                return ParallelVectorizedEngine(max_workers=parallel_workers)
            else:
                return VectorizedBacktestEngine(
                    chunk_size=kwargs.get('chunk_size', 50000)
                )
        
        elif engine_type == 'streaming':
            return StreamingBacktestEngine(
                max_memory_mb=kwargs.get('max_memory_mb', 1000)
            )
        
        elif engine_type == 'original':
            return BacktestRunner()
        
        else:
            raise ValueError(f"Unknown engine type: {engine_type}")
    
    @staticmethod
    def get_recommended_engine(req: BacktestRequest) -> str:
        """Get recommended engine type for request."""
        
        optimization = performance_optimizer.optimize_backtest_performance(req)
        params = optimization['optimized_params']
        
        if params['use_streaming_engine']:
            return 'streaming'
        elif params['use_vectorized_engine']:
            return 'vectorized'
        else:
            return 'original'


# Monkey patch the original BacktestRunner for seamless integration
def patch_original_runner():
    """Patch the original BacktestRunner to use optimizations."""
    
    original_run = BacktestRunner.run
    
    def optimized_run(self, req, dataset, progress_cb=None, is_cancelled=None):
        """Optimized version of the original run method."""
        
        # Check if we should use optimization
        if hasattr(req, 'use_optimization') and req.use_optimization:
            optimized_runner = OptimizedBacktestRunner(auto_optimize=True)
            return optimized_runner.run(req, dataset, progress_cb, is_cancelled)
        
        # Use original method with optimized indicators
        from .indicators_optimized import compute_indicators_fast
        from . import indicators
        
        original_compute = indicators.compute_indicators
        try:
            indicators.compute_indicators = compute_indicators_fast
            return original_run(self, req, dataset, progress_cb, is_cancelled)
        finally:
            indicators.compute_indicators = original_compute
    
    BacktestRunner.run = optimized_run


# Performance monitoring decorator
def monitor_backtest_performance(func):
    """Decorator to monitor backtest performance."""
    
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = None
        
        try:
            import psutil
            process = psutil.Process()
            start_memory = process.memory_info().rss / (1024 * 1024)
        except ImportError:
            pass
        
        try:
            result = func(*args, **kwargs)
            
            # Add performance metrics to result
            end_time = time.time()
            execution_time = end_time - start_time
            
            if 'meta' not in result:
                result['meta'] = {}
            
            result['meta']['execution_time_seconds'] = round(execution_time, 2)
            result['meta']['execution_time_minutes'] = round(execution_time / 60, 2)
            
            if start_memory:
                try:
                    end_memory = process.memory_info().rss / (1024 * 1024)
                    result['meta']['memory_usage_mb'] = round(end_memory - start_memory, 2)
                    result['meta']['peak_memory_mb'] = round(end_memory, 2)
                except:
                    pass
            
            # Calculate performance metrics
            trades = result.get('trades', [])
            ticks_processed = result.get('meta', {}).get('ticks_replayed', 0)
            
            if execution_time > 0:
                result['meta']['ticks_per_second'] = round(ticks_processed / execution_time, 0)
                result['meta']['trades_per_second'] = round(len(trades) / execution_time, 2)
            
            logging.info(f"Backtest completed in {execution_time:.2f}s, "
                        f"processed {ticks_processed} ticks, "
                        f"generated {len(trades)} trades")
            
            return result
            
        except Exception as e:
            logging.error(f"Backtest failed after {time.time() - start_time:.2f}s: {e}")
            raise
    
    return wrapper


# Apply performance monitoring to key functions
BacktestRunner.run = monitor_backtest_performance(BacktestRunner.run)


def enable_optimizations():
    """Enable all performance optimizations."""
    
    # Patch original runner
    patch_original_runner()
    
    # Enable optimized indicators globally
    optimized_indicators.cache.clear()  # Start with clean cache
    
    logging.info("Performance optimizations enabled")


def get_performance_report(req: BacktestRequest) -> Dict:
    """Get detailed performance analysis report."""
    
    optimization = performance_optimizer.optimize_backtest_performance(req)
    
    return {
        'system_info': optimization['system_info'],
        'dataset_analysis': optimization['report']['dataset_info'],
        'recommended_profile': optimization['profile'].name,
        'estimated_speedup': optimization['report']['total_speedup'],
        'estimated_time_minutes': optimization['report']['estimated_time_minutes'],
        'memory_requirements_mb': optimization['report']['memory_usage_mb'],
        'recommendations': optimization['report']['recommendations'],
        'optimized_parameters': optimization['optimized_params']
    }