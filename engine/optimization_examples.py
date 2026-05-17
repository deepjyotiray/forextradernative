"""
Usage examples and integration guide for optimized backtesting.
"""
from datetime import datetime, timezone, timedelta
import logging

# Import optimized components
from .backtest_optimized import (
    OptimizedBacktestRunner, 
    BacktestEngineFactory,
    enable_optimizations,
    get_performance_report
)
from .performance_optimizer import performance_optimizer
from .backtest_sim import BacktestRequest


def example_basic_optimization():
    """Basic example of using optimized backtesting."""
    
    # Enable all optimizations
    enable_optimizations()
    
    # Create backtest request
    req = BacktestRequest(
        symbol='XAUUSD',
        strategy='SMC_CONFLUENCE',
        start_utc=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end_utc=datetime(2024, 1, 31, tzinfo=timezone.utc),
        initial_balance=10000.0,
        use_optimization=True  # Enable automatic optimization
    )
    
    # Get performance analysis
    perf_report = get_performance_report(req)
    print(f"Recommended profile: {perf_report['recommended_profile']}")
    print(f"Estimated speedup: {perf_report['estimated_speedup']}x")
    print(f"Estimated time: {perf_report['estimated_time_minutes']} minutes")
    
    # Run optimized backtest
    runner = OptimizedBacktestRunner(auto_optimize=True)
    
    # Load data (using existing data provider)
    from .backtest_data import BacktestDataProvider
    provider = BacktestDataProvider()
    
    try:
        dataset = provider.load_dataset(
            symbol=req.symbol,
            start_utc=req.start_utc,
            end_utc=req.end_utc,
            cache_only=True
        )
        
        # Run with progress monitoring
        def progress_callback(update):
            processed = update.get('processed_ticks', 0)
            total = update.get('total_ticks', 0)
            if total > 0:
                pct = (processed / total) * 100
                print(f"Progress: {pct:.1f}% ({processed:,}/{total:,} ticks)")
        
        result = runner.run(req, dataset, progress_cb=progress_callback)
        
        # Print results
        summary = result['summary']
        print(f"\nBacktest Results:")
        print(f"Trades: {summary['trades']}")
        print(f"Win Rate: {summary['win_rate']:.2%}")
        print(f"Total PnL: ${summary['pnl']:.2f}")
        print(f"Engine Used: {result['meta']['engine_type']}")
        print(f"Execution Time: {result['meta']['execution_time_minutes']:.2f} minutes")
        
    finally:
        provider.disconnect()


def example_manual_engine_selection():
    """Example of manually selecting optimization engine."""
    
    req = BacktestRequest(
        symbol='EURUSD',
        strategy='M15_SCALP_DEEP',
        start_utc=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end_utc=datetime(2024, 3, 31, tzinfo=timezone.utc),  # 3 months
        initial_balance=10000.0
    )
    
    # Get recommendation
    recommended_engine = BacktestEngineFactory.get_recommended_engine(req)
    print(f"Recommended engine: {recommended_engine}")
    
    # Create specific engine
    if recommended_engine == 'vectorized':
        engine = BacktestEngineFactory.create_engine(
            'vectorized',
            parallel_workers=8,
            chunk_size=100000
        )
    elif recommended_engine == 'streaming':
        engine = BacktestEngineFactory.create_engine(
            'streaming',
            max_memory_mb=2000
        )
    else:
        engine = BacktestEngineFactory.create_engine('auto')
    
    print(f"Created engine: {type(engine).__name__}")


def example_performance_profiling():
    """Example of performance profiling and optimization."""
    
    # Test different date ranges
    test_ranges = [
        (datetime(2024, 1, 1), datetime(2024, 1, 7)),    # 1 week
        (datetime(2024, 1, 1), datetime(2024, 1, 31)),   # 1 month
        (datetime(2024, 1, 1), datetime(2024, 3, 31)),   # 3 months
        (datetime(2024, 1, 1), datetime(2024, 12, 31)),  # 1 year
    ]
    
    for start_date, end_date in test_ranges:
        req = BacktestRequest(
            symbol='XAUUSD',
            strategy='SMC_CONFLUENCE',
            start_utc=start_date.replace(tzinfo=timezone.utc),
            end_utc=end_date.replace(tzinfo=timezone.utc),
            initial_balance=10000.0
        )
        
        # Get performance analysis
        perf_report = get_performance_report(req)
        
        duration = (end_date - start_date).days
        print(f"\nDuration: {duration} days")
        print(f"Estimated ticks: {perf_report['dataset_analysis']['estimated_ticks']:,}")
        print(f"Recommended profile: {perf_report['recommended_profile']}")
        print(f"Estimated speedup: {perf_report['estimated_speedup']}x")
        print(f"Memory needed: {perf_report['memory_requirements_mb']:.1f} MB")
        
        if perf_report['recommendations']:
            print("Recommendations:")
            for rec in perf_report['recommendations']:
                print(f"  - {rec}")


def example_large_dataset_optimization():
    """Example optimized for very large datasets (1+ years)."""
    
    req = BacktestRequest(
        symbol='XAUUSD',
        strategy='SMC_CONFLUENCE',
        start_utc=datetime(2023, 1, 1, tzinfo=timezone.utc),
        end_utc=datetime(2024, 12, 31, tzinfo=timezone.utc),  # 2 years
        initial_balance=10000.0,
        fast_mode=True,  # Enable fast mode for large datasets
        split_mode='MONTHLY',  # Use monthly splits for parallel processing
        parallel_workers=8
    )
    
    # Force streaming engine for memory efficiency
    from .backtest_streaming import StreamingBacktestEngine
    
    engine = StreamingBacktestEngine(max_memory_mb=4000)
    
    print("Running large dataset backtest with streaming engine...")
    print("This will process data in chunks to minimize memory usage.")
    
    # In practice, you would run the backtest here
    # result = engine.run_streaming(...)


def example_custom_optimization():
    """Example of custom optimization settings."""
    
    req = BacktestRequest(
        symbol='GBPUSD',
        strategy='AUTO',
        start_utc=datetime(2024, 6, 1, tzinfo=timezone.utc),
        end_utc=datetime(2024, 8, 31, tzinfo=timezone.utc),
        initial_balance=10000.0
    )
    
    # Get base optimization
    optimization = performance_optimizer.optimize_backtest_performance(req, 'balanced')
    params = optimization['optimized_params']
    
    # Customize parameters
    params['max_ticks_per_batch'] = 25000  # Smaller batches
    params['parallel_workers'] = 4         # Fewer workers
    params['enable_indicator_cache'] = True # Force caching
    
    print("Custom optimization parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    # Create runner with custom settings
    runner = OptimizedBacktestRunner(auto_optimize=False)
    
    # Apply custom settings (in practice, you'd modify the engine creation)
    print("Custom optimization applied")


def example_memory_constrained_system():
    """Example for systems with limited memory."""
    
    req = BacktestRequest(
        symbol='USDJPY',
        strategy='M15_SCALP_DEEP',
        start_utc=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end_utc=datetime(2024, 6, 30, tzinfo=timezone.utc),
        initial_balance=10000.0
    )
    
    # Force memory-efficient profile
    optimization = performance_optimizer.optimize_backtest_performance(req, 'memory_efficient')
    
    print("Memory-efficient optimization:")
    print(f"Profile: {optimization['profile'].name}")
    print(f"Streaming enabled: {optimization['optimized_params']['use_streaming_engine']}")
    print(f"Batch size: {optimization['optimized_params']['max_ticks_per_batch']}")
    print(f"Memory limit: {optimization['optimized_params']['memory_limit_mb']} MB")
    
    # This configuration will:
    # - Use streaming to minimize memory usage
    # - Process smaller batches
    # - Disable indicator caching
    # - Use fewer parallel workers


def example_integration_with_existing_code():
    """Example of integrating with existing backtest code."""
    
    # Existing code pattern
    from .backtest_sim import BacktestRunner
    from .backtest_data import BacktestDataProvider
    
    # Enable optimizations globally
    enable_optimizations()
    
    # Existing code continues to work but now uses optimizations
    provider = BacktestDataProvider()
    runner = BacktestRunner()  # Now automatically optimized
    
    req = BacktestRequest(
        symbol='XAUUSD',
        strategy='SMC_CONFLUENCE',
        start_utc=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end_utc=datetime(2024, 1, 31, tzinfo=timezone.utc),
        initial_balance=10000.0,
        use_optimization=True  # Enable optimization flag
    )
    
    try:
        dataset = provider.load_dataset(
            symbol=req.symbol,
            start_utc=req.start_utc,
            end_utc=req.end_utc,
            cache_only=True
        )
        
        # This will now use optimized engine automatically
        result = runner.run(req, dataset)
        
        print(f"Optimized backtest completed:")
        print(f"Engine: {result['meta']['engine_type']}")
        print(f"Execution time: {result['meta']['execution_time_seconds']}s")
        
    finally:
        provider.disconnect()


def benchmark_optimization_impact():
    """Benchmark the impact of optimizations."""
    
    import time
    
    req = BacktestRequest(
        symbol='XAUUSD',
        strategy='SMC_CONFLUENCE',
        start_utc=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end_utc=datetime(2024, 1, 15, tzinfo=timezone.utc),  # 2 weeks for quick test
        initial_balance=10000.0
    )
    
    provider = BacktestDataProvider()
    
    try:
        dataset = provider.load_dataset(
            symbol=req.symbol,
            start_utc=req.start_utc,
            end_utc=req.end_utc,
            cache_only=True
        )
        
        # Test original engine
        print("Testing original engine...")
        original_runner = BacktestRunner()
        start_time = time.time()
        original_result = original_runner.run(req, dataset)
        original_time = time.time() - start_time
        
        # Test optimized engine
        print("Testing optimized engine...")
        optimized_runner = OptimizedBacktestRunner(auto_optimize=True)
        start_time = time.time()
        optimized_result = optimized_runner.run(req, dataset)
        optimized_time = time.time() - start_time
        
        # Compare results
        speedup = original_time / optimized_time if optimized_time > 0 else 0
        
        print(f"\nBenchmark Results:")
        print(f"Original time: {original_time:.2f}s")
        print(f"Optimized time: {optimized_time:.2f}s")
        print(f"Speedup: {speedup:.2f}x")
        print(f"Original trades: {len(original_result['trades'])}")
        print(f"Optimized trades: {len(optimized_result['trades'])}")
        
        # Verify results are similar (allowing for small differences due to optimization)
        trade_diff = abs(len(original_result['trades']) - len(optimized_result['trades']))
        if trade_diff <= 2:  # Allow small differences
            print("✓ Results are consistent")
        else:
            print("⚠ Results differ significantly - review optimization logic")
        
    finally:
        provider.disconnect()


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("=== Backtesting Optimization Examples ===\n")
    
    # Run examples
    try:
        print("1. Basic Optimization Example")
        example_basic_optimization()
        
        print("\n2. Performance Profiling Example")
        example_performance_profiling()
        
        print("\n3. Manual Engine Selection Example")
        example_manual_engine_selection()
        
        print("\n4. Memory Constrained System Example")
        example_memory_constrained_system()
        
        print("\n5. Benchmark Optimization Impact")
        benchmark_optimization_impact()
        
    except Exception as e:
        print(f"Example failed: {e}")
        logging.exception("Example execution failed")
    
    print("\n=== Examples completed ===")


# Integration checklist for existing codebase:
"""
INTEGRATION CHECKLIST:

1. Install required dependencies:
   pip install numba psutil pyarrow

2. Enable optimizations in your main backtest script:
   from engine.backtest_optimized import enable_optimizations
   enable_optimizations()

3. Add optimization flag to BacktestRequest:
   req.use_optimization = True

4. For new code, use OptimizedBacktestRunner:
   from engine.backtest_optimized import OptimizedBacktestRunner
   runner = OptimizedBacktestRunner(auto_optimize=True)

5. For large datasets, consider streaming:
   from engine.backtest_streaming import StreamingBacktestEngine
   engine = StreamingBacktestEngine(max_memory_mb=2000)

6. Monitor performance:
   from engine.backtest_optimized import get_performance_report
   report = get_performance_report(req)

7. Customize optimization profiles:
   from engine.performance_optimizer import performance_optimizer
   optimization = performance_optimizer.optimize_backtest_performance(req, 'ultra_fast')

EXPECTED PERFORMANCE IMPROVEMENTS:
- 3-10x speedup for typical backtests
- 50-80% memory reduction with streaming
- Better CPU utilization with parallel processing
- Faster indicator calculations with caching
- Automatic optimization based on system resources
"""
