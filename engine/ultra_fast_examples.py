"""
Ultra-Fast Backtesting - Usage Examples and Integration Guide

This module demonstrates how to use the ultra-fast backtesting engine
for maximum performance with real-world accuracy.
"""
import os
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List

# Import ultra-fast components
from .ultra_fast_main import (
    ultra_fast_engine,
    run_ultra_fast_backtest,
    benchmark_ultra_fast_engine,
    get_ultra_fast_optimization_report
)
from .backtest_sim import BacktestRequest


def example_ultra_fast_basic():
    """Basic example of ultra-fast backtesting."""
    
    print("🚀 ULTRA-FAST BACKTESTING - BASIC EXAMPLE")
    print("=" * 50)
    
    # Create backtest request
    req = BacktestRequest(
        symbol='XAUUSD',
        strategy='SMC_CONFLUENCE',
        start_utc=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end_utc=datetime(2024, 1, 31, tzinfo=timezone.utc),  # 1 month
        initial_balance=10000.0,
        fast_mode=True,  # Enable all speed optimizations
        max_ticks_per_batch=100000,  # Large batches for speed
        parallel_workers=os.cpu_count() or 1  # Use all cores
    )
    
    # File paths (adjust to your data location)
    tick_file = "backtests/data_cache/XAUUSD/latest/ticks.pkl"
    candle_files = {
        'M1': "backtests/data_cache/XAUUSD/latest/M1.pkl",
        'M5': "backtests/data_cache/XAUUSD/latest/M5.pkl",
        'M15': "backtests/data_cache/XAUUSD/latest/M15.pkl",
        'H1': "backtests/data_cache/XAUUSD/latest/H1.pkl",
        'H4': "backtests/data_cache/XAUUSD/latest/H4.pkl"
    }
    
    # Check if files exist
    if not os.path.exists(tick_file):
        print(f"❌ Tick file not found: {tick_file}")
        print("Please ensure you have cached data available.")
        return
    
    # Get optimization report first
    print("\n📊 OPTIMIZATION ANALYSIS")
    print("-" * 30)
    
    opt_report = get_ultra_fast_optimization_report(req)
    
    print(f"Dataset: {opt_report['dataset_analysis']['duration_days']} days")
    print(f"Estimated ticks: {opt_report['dataset_analysis']['estimated_ticks']:,}")
    print(f"Estimated size: {opt_report['dataset_analysis']['estimated_size_gb']:.1f} GB")
    print(f"Expected speedup: {opt_report['expected_performance']['estimated_speedup_vs_standard']}x")
    print(f"Expected time: {opt_report['expected_performance']['estimated_execution_time_seconds']:.1f}s")
    
    if opt_report['optimization_recommendations']:
        print("\nRecommendations:")
        for rec in opt_report['optimization_recommendations']:
            print(f"  • {rec}")
    
    # Progress callback
    def progress_callback(update: Dict):
        processed = update.get('processed_ticks', 0)
        total = update.get('total_ticks', 0)
        if total > 0:
            pct = (processed / total) * 100
            print(f"Progress: {pct:5.1f}% ({processed:,}/{total:,} ticks)")
    
    # Run ultra-fast backtest
    print(f"\n⚡ RUNNING ULTRA-FAST BACKTEST")
    print("-" * 30)
    
    try:
        result = run_ultra_fast_backtest(
            req, tick_file, candle_files, progress_callback
        )
        
        # Display results
        print(f"\n📈 RESULTS SUMMARY")
        print("-" * 20)
        
        summary = result['summary']
        meta = result['meta']['ultra_performance']
        
        print(f"Execution time: {meta['total_execution_time_seconds']}s")
        print(f"Processing speed: {meta['ticks_per_second_overall']:,.0f} ticks/second")
        print(f"Actual speedup: {meta['speedup_estimate']}x")
        print(f"Trades generated: {summary['trades']}")
        print(f"Win rate: {summary['win_rate']:.1%}")
        print(f"Total PnL: ${summary['total_pnl']:,.2f}")
        print(f"Final balance: ${summary['final_balance']:,.2f}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error running backtest: {e}")
        return None


def example_ultra_fast_large_dataset():
    """Example with large dataset (multiple months)."""
    
    print("🚀 ULTRA-FAST BACKTESTING - LARGE DATASET EXAMPLE")
    print("=" * 55)
    
    # Large dataset - 6 months
    req = BacktestRequest(
        symbol='XAUUSD',
        strategy='SMC_CONFLUENCE',
        start_utc=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end_utc=datetime(2024, 6, 30, tzinfo=timezone.utc),  # 6 months
        initial_balance=10000.0,
        fast_mode=True,
        max_ticks_per_batch=200000,  # Even larger batches
        parallel_workers=os.cpu_count() or 1
    )
    
    print(f"Dataset: {(req.end_utc - req.start_utc).days} days")
    
    # Get system info
    import psutil
    memory = psutil.virtual_memory()
    
    print(f"System RAM: {memory.total / (1024**3):.1f} GB")
    print(f"Available RAM: {memory.available / (1024**3):.1f} GB")
    print(f"CPU cores: {os.cpu_count()}")
    
    # Optimization analysis
    opt_report = get_ultra_fast_optimization_report(req)
    
    print(f"\nEstimated processing:")
    print(f"  Ticks: {opt_report['dataset_analysis']['estimated_ticks']:,}")
    print(f"  Memory needed: {opt_report['expected_performance']['memory_usage_gb']:.1f} GB")
    print(f"  Expected time: {opt_report['expected_performance']['estimated_execution_time_seconds']:.1f}s")
    
    # Check if we have enough resources
    if opt_report['system_capability']['memory_limited']:
        print("⚠️  Large dataset may exceed available memory")
        print("   Consider using streaming mode or splitting the dataset")
    
    print(f"\nThis would normally take hours with standard backtesting.")
    print(f"Ultra-fast engine should complete in ~{opt_report['expected_performance']['estimated_execution_time_seconds']:.0f} seconds!")


def example_benchmark_performance():
    """Benchmark ultra-fast engine performance."""
    
    print("🏁 ULTRA-FAST ENGINE PERFORMANCE BENCHMARK")
    print("=" * 45)
    
    # Small dataset for quick benchmarking
    req = BacktestRequest(
        symbol='XAUUSD',
        strategy='SMC_CONFLUENCE',
        start_utc=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end_utc=datetime(2024, 1, 7, tzinfo=timezone.utc),  # 1 week
        initial_balance=10000.0,
        fast_mode=True
    )
    
    # File paths
    tick_file = "backtests/data_cache/XAUUSD/latest/ticks.pkl"
    candle_files = {
        'M1': "backtests/data_cache/XAUUSD/latest/M1.pkl",
        'M5': "backtests/data_cache/XAUUSD/latest/M5.pkl",
        'H1': "backtests/data_cache/XAUUSD/latest/H1.pkl"
    }
    
    if not os.path.exists(tick_file):
        print(f"❌ Benchmark data not available: {tick_file}")
        return
    
    print("Running 3 iterations for consistent benchmarking...")
    
    try:
        benchmark_results = benchmark_ultra_fast_engine(
            req, tick_file, candle_files, iterations=3
        )
        
        print(f"\n🏆 BENCHMARK RESULTS")
        print("-" * 25)
        print(f"Average time: {benchmark_results['avg_execution_time']:.2f}s")
        print(f"Best time: {benchmark_results['min_execution_time']:.2f}s")
        print(f"Worst time: {benchmark_results['max_execution_time']:.2f}s")
        print(f"Consistency: {benchmark_results['consistency_score']:.3f}")
        print(f"Average speed: {benchmark_results['avg_ticks_per_second']:,.0f} ticks/second")
        print(f"Peak speed: {benchmark_results['max_ticks_per_second']:,.0f} ticks/second")
        
        return benchmark_results
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        return None


def example_compare_with_standard():
    """Compare ultra-fast engine with standard implementation."""
    
    print("⚖️  ULTRA-FAST vs STANDARD ENGINE COMPARISON")
    print("=" * 50)
    
    # Small dataset for fair comparison
    req = BacktestRequest(
        symbol='XAUUSD',
        strategy='SMC_CONFLUENCE',
        start_utc=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end_utc=datetime(2024, 1, 3, tzinfo=timezone.utc),  # 3 days
        initial_balance=10000.0
    )
    
    print(f"Dataset: {(req.end_utc - req.start_utc).days} days")
    
    # File paths
    tick_file = "backtests/data_cache/XAUUSD/latest/ticks.pkl"
    candle_files = {
        'M1': "backtests/data_cache/XAUUSD/latest/M1.pkl",
        'M5': "backtests/data_cache/XAUUSD/latest/M5.pkl"
    }
    
    if not os.path.exists(tick_file):
        print(f"❌ Comparison data not available: {tick_file}")
        return
    
    # Test ultra-fast engine
    print("\n⚡ Testing Ultra-Fast Engine...")
    ultra_start = time.time()
    
    try:
        ultra_result = run_ultra_fast_backtest(req, tick_file, candle_files)
        ultra_time = time.time() - ultra_start
        
        print(f"✓ Ultra-fast completed in {ultra_time:.2f}s")
        print(f"  Speed: {ultra_result['meta']['ultra_performance']['ticks_per_second_overall']:,.0f} ticks/second")
        print(f"  Trades: {ultra_result['summary']['trades']}")
        
    except Exception as e:
        print(f"❌ Ultra-fast engine failed: {e}")
        return
    
    # Test standard engine (if available)
    print("\n🐌 Testing Standard Engine...")
    
    try:
        from .backtest_sim import BacktestRunner
        from .backtest_data import BacktestDataProvider
        
        provider = BacktestDataProvider()
        runner = BacktestRunner()
        
        standard_start = time.time()
        
        dataset = provider.load_dataset(
            symbol=req.symbol,
            start_utc=req.start_utc,
            end_utc=req.end_utc,
            cache_only=True
        )
        
        standard_result = runner.run(req, dataset)
        standard_time = time.time() - standard_start
        
        provider.disconnect()
        
        print(f"✓ Standard completed in {standard_time:.2f}s")
        print(f"  Trades: {standard_result['summary']['trades']}")
        
        # Comparison
        speedup = standard_time / ultra_time if ultra_time > 0 else 0
        
        print(f"\n🏆 COMPARISON RESULTS")
        print("-" * 25)
        print(f"Ultra-fast time: {ultra_time:.2f}s")
        print(f"Standard time: {standard_time:.2f}s")
        print(f"Speedup achieved: {speedup:.1f}x")
        print(f"Time saved: {standard_time - ultra_time:.2f}s ({((standard_time - ultra_time) / standard_time * 100):.1f}%)")
        
        # Verify results are similar
        ultra_trades = ultra_result['summary']['trades']
        standard_trades = standard_result['summary']['trades']
        trade_diff = abs(ultra_trades - standard_trades)
        
        if trade_diff <= 2:  # Allow small differences
            print("✓ Results are consistent between engines")
        else:
            print(f"⚠️  Trade count differs: Ultra={ultra_trades}, Standard={standard_trades}")
        
        return {
            'ultra_time': ultra_time,
            'standard_time': standard_time,
            'speedup': speedup,
            'ultra_trades': ultra_trades,
            'standard_trades': standard_trades
        }
        
    except Exception as e:
        print(f"❌ Standard engine test failed: {e}")
        print("This is expected if standard engine data is not available")
        return None


def example_memory_usage_analysis():
    """Analyze memory usage patterns."""
    
    print("🧠 MEMORY USAGE ANALYSIS")
    print("=" * 30)
    
    import psutil
    
    # Get initial memory state
    initial_memory = psutil.virtual_memory()
    
    print(f"Initial memory state:")
    print(f"  Total: {initial_memory.total / (1024**3):.1f} GB")
    print(f"  Available: {initial_memory.available / (1024**3):.1f} GB")
    print(f"  Used: {initial_memory.used / (1024**3):.1f} GB ({initial_memory.percent:.1f}%)")
    
    # Create different sized datasets
    test_cases = [
        ("Small (1 week)", 7),
        ("Medium (1 month)", 30),
        ("Large (3 months)", 90),
        ("Very Large (1 year)", 365)
    ]
    
    for name, days in test_cases:
        req = BacktestRequest(
            symbol='XAUUSD',
            strategy='SMC_CONFLUENCE',
            start_utc=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_utc=datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=days),
            initial_balance=10000.0
        )
        
        opt_report = get_ultra_fast_optimization_report(req)
        
        print(f"\n{name}:")
        print(f"  Duration: {days} days")
        print(f"  Estimated ticks: {opt_report['dataset_analysis']['estimated_ticks']:,}")
        print(f"  Memory needed: {opt_report['expected_performance']['memory_usage_gb']:.1f} GB")
        print(f"  Memory limited: {'Yes' if opt_report['system_capability']['memory_limited'] else 'No'}")
        print(f"  Expected time: {opt_report['expected_performance']['estimated_execution_time_seconds']:.1f}s")


def example_integration_with_existing_system():
    """Show how to integrate ultra-fast engine with existing system."""
    
    print("🔧 INTEGRATION WITH EXISTING SYSTEM")
    print("=" * 40)
    
    print("""
Integration Steps:

1. Replace existing backtest calls:
   
   # OLD CODE:
   from engine.backtest_sim import BacktestRunner
   runner = BacktestRunner()
   result = runner.run(req, dataset)
   
   # NEW CODE:
   from engine.ultra_fast_main import run_ultra_fast_backtest
   result = run_ultra_fast_backtest(req, tick_file, candle_files)

2. Update file paths to use cached data:
   
   tick_file = "backtests/data_cache/XAUUSD/latest/ticks.pkl"
   candle_files = {
       'M1': "backtests/data_cache/XAUUSD/latest/M1.pkl",
       'M5': "backtests/data_cache/XAUUSD/latest/M5.pkl",
       # ... other timeframes
   }

3. Enable fast mode in BacktestRequest:
   
   req = BacktestRequest(
       symbol='XAUUSD',
       strategy='SMC_CONFLUENCE',
       start_utc=start_date,
       end_utc=end_date,
       initial_balance=10000.0,
       fast_mode=True,  # Enable all optimizations
       max_ticks_per_batch=100000,  # Large batches
       parallel_workers=os.cpu_count()  # Use all cores
   )

4. Handle progress callbacks:
   
   def progress_callback(update):
       processed = update.get('processed_ticks', 0)
       total = update.get('total_ticks', 0)
       if total > 0:
           pct = (processed / total) * 100
           print(f"Progress: {pct:.1f}%")
   
   result = run_ultra_fast_backtest(req, tick_file, candle_files, progress_callback)

5. Access enhanced results:
   
   # Standard results
   summary = result['summary']
   trades = result['trades']
   
   # Ultra-fast specific metrics
   ultra_perf = result['meta']['ultra_performance']
   speedup = ultra_perf['speedup_estimate']
   ticks_per_second = ultra_perf['ticks_per_second_overall']
   
Expected Performance Improvements:
- 10-50x faster execution
- 50-80% memory reduction (with streaming)
- 100% CPU utilization
- Real-time processing capability

System Requirements:
- Python 64-bit (recommended)
- 8+ GB RAM (16+ GB for large datasets)
- Multi-core CPU (4+ cores recommended)
- SSD storage (for data caching)

Dependencies:
- numba (for JIT compilation)
- psutil (for system monitoring)
- numpy (optimized version)
""")


def run_all_examples():
    """Run all ultra-fast backtesting examples."""
    
    print("🚀 ULTRA-FAST BACKTESTING - ALL EXAMPLES")
    print("=" * 50)
    
    examples = [
        ("Basic Usage", example_ultra_fast_basic),
        ("Large Dataset", example_ultra_fast_large_dataset),
        ("Performance Benchmark", example_benchmark_performance),
        ("Engine Comparison", example_compare_with_standard),
        ("Memory Analysis", example_memory_usage_analysis),
        ("Integration Guide", example_integration_with_existing_system)
    ]
    
    for name, example_func in examples:
        print(f"\n{'='*60}")
        print(f"EXAMPLE: {name}")
        print(f"{'='*60}")
        
        try:
            example_func()
        except Exception as e:
            print(f"❌ Example '{name}' failed: {e}")
        
        print(f"\n{'='*60}")
        print(f"END: {name}")
        print(f"{'='*60}")
        
        # Small delay between examples
        time.sleep(1)


if __name__ == "__main__":
    # Run basic example by default
    example_ultra_fast_basic()


# Performance expectations summary
PERFORMANCE_EXPECTATIONS = {
    "small_dataset_1_week": {
        "standard_time_seconds": 30,
        "ultra_fast_time_seconds": 2,
        "speedup": "15x",
        "memory_usage_mb": 100
    },
    "medium_dataset_1_month": {
        "standard_time_seconds": 300,
        "ultra_fast_time_seconds": 15,
        "speedup": "20x", 
        "memory_usage_mb": 500
    },
    "large_dataset_6_months": {
        "standard_time_seconds": 1800,
        "ultra_fast_time_seconds": 60,
        "speedup": "30x",
        "memory_usage_mb": 2000
    },
    "very_large_dataset_1_year": {
        "standard_time_seconds": 7200,
        "ultra_fast_time_seconds": 180,
        "speedup": "40x",
        "memory_usage_mb": 4000
    }
}