"""
Quick Fix for Non-Uniform Progress in Existing Backtest System
This patch addresses the progress uniformity issue without major changes.
"""
import os
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple
import pandas as pd


def analyze_current_progress_issue(chunk_states: List[Dict]) -> Dict:
    """Analyze the current non-uniform progress issue."""
    
    print("ANALYZING NON-UNIFORM PROGRESS ISSUE")
    print("=" * 50)
    
    analysis = {
        'chunk_analysis': [],
        'root_causes': [],
        'quick_fixes': [],
        'performance_impact': {}
    }
    
    # Analyze each chunk
    for i, chunk in enumerate(chunk_states):
        start_time = pd.to_datetime(chunk['start_utc'])
        end_time = pd.to_datetime(chunk['end_utc'])
        progress = chunk['progress_pct']
        
        # Calculate time period characteristics
        duration_hours = (end_time - start_time).total_seconds() / 3600
        day_of_week_start = start_time.weekday()
        hour_of_day_start = start_time.hour
        
        # Estimate tick density based on market sessions
        estimated_tick_density = estimate_tick_density_simple(start_time, end_time)
        
        chunk_info = {
            'chunk_id': i + 1,
            'start_time': start_time.strftime('%Y-%m-%d %H:%M UTC'),
            'end_time': end_time.strftime('%Y-%m-%d %H:%M UTC'),
            'duration_hours': round(duration_hours, 1),
            'progress_pct': progress,
            'day_of_week': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][day_of_week_start],
            'start_hour': hour_of_day_start,
            'estimated_ticks': estimated_tick_density,
            'market_session': classify_market_session(start_time),
            'expected_complexity': calculate_processing_complexity(start_time, end_time)
        }
        
        analysis['chunk_analysis'].append(chunk_info)
        
        print(f"Chunk {i+1}: {progress:5.1f}% | {chunk_info['market_session']:12} | "
              f"{chunk_info['estimated_ticks']:6,} ticks | {chunk_info['duration_hours']:4.1f}h")
    
    # Identify root causes
    analysis['root_causes'] = identify_root_causes(analysis['chunk_analysis'])
    
    # Generate quick fixes
    analysis['quick_fixes'] = generate_quick_fixes(analysis['chunk_analysis'])
    
    # Calculate performance impact
    analysis['performance_impact'] = calculate_performance_impact(analysis['chunk_analysis'])
    
    print(f"\nROOT CAUSES IDENTIFIED:")
    for cause in analysis['root_causes']:
        print(f"  • {cause}")
    
    print(f"\nQUICK FIXES AVAILABLE:")
    for fix in analysis['quick_fixes']:
        print(f"  • {fix}")
    
    return analysis


def estimate_tick_density_simple(start_time: datetime, end_time: datetime) -> int:
    """Simple tick density estimation based on market sessions."""
    
    # Tick rates per hour for different market conditions
    tick_rates = {
        'weekend': 50,           # Very low activity
        'asian_quiet': 300,      # Asian session quiet hours
        'asian_active': 800,     # Asian session active hours
        'london_open': 2500,     # London market open
        'london_active': 2000,   # London session
        'overlap': 3500,         # London-NY overlap
        'ny_active': 2000,       # NY session
        'ny_close': 1200,        # NY session closing
        'after_hours': 400       # After market hours
    }
    
    total_ticks = 0
    current = start_time
    
    while current < end_time:
        hour_utc = current.hour
        day_of_week = current.weekday()
        
        # Determine market session
        if day_of_week >= 5:  # Weekend
            rate = tick_rates['weekend']
        elif 0 <= hour_utc <= 6:  # Asian session
            rate = tick_rates['asian_active'] if 2 <= hour_utc <= 5 else tick_rates['asian_quiet']
        elif 7 <= hour_utc <= 11:  # London session
            rate = tick_rates['london_open'] if hour_utc == 8 else tick_rates['london_active']
        elif 12 <= hour_utc <= 16:  # London-NY overlap
            rate = tick_rates['overlap']
        elif 17 <= hour_utc <= 21:  # NY session
            rate = tick_rates['ny_active']
        elif hour_utc == 22:  # NY close
            rate = tick_rates['ny_close']
        else:  # After hours
            rate = tick_rates['after_hours']
        
        total_ticks += rate
        current += timedelta(hours=1)
    
    return total_ticks


def classify_market_session(timestamp: datetime) -> str:
    """Classify market session for a timestamp."""
    
    hour = timestamp.hour
    day_of_week = timestamp.weekday()
    
    if day_of_week >= 5:
        return "Weekend"
    elif 0 <= hour <= 6:
        return "Asian"
    elif 7 <= hour <= 11:
        return "London"
    elif 12 <= hour <= 16:
        return "Overlap"
    elif 17 <= hour <= 21:
        return "New York"
    else:
        return "After Hours"


def calculate_processing_complexity(start_time: datetime, end_time: datetime) -> float:
    """Calculate processing complexity score."""
    
    duration_hours = (end_time - start_time).total_seconds() / 3600
    
    # Base complexity from duration
    complexity = duration_hours * 0.1
    
    # Add complexity for high-activity periods
    current = start_time
    while current < end_time:
        hour = current.hour
        
        # High complexity during market opens and overlaps
        if hour in [8, 9, 13, 14, 17, 18]:  # Market open times
            complexity += 0.5
        elif hour in [12, 15, 16]:  # Overlap periods
            complexity += 0.3
        
        current += timedelta(hours=1)
    
    return round(complexity, 2)


def identify_root_causes(chunk_analysis: List[Dict]) -> List[str]:
    """Identify root causes of non-uniform progress."""
    
    causes = []
    
    # Check for tick density variation
    tick_densities = [chunk['estimated_ticks'] for chunk in chunk_analysis]
    if tick_densities:
        max_ticks = max(tick_densities)
        min_ticks = min(tick_densities)
        variation_ratio = max_ticks / min_ticks if min_ticks > 0 else float('inf')
        
        if variation_ratio > 3.0:
            causes.append(f"High tick density variation: {variation_ratio:.1f}x difference between chunks")
    
    # Check for weekend chunks
    weekend_chunks = [c for c in chunk_analysis if c['day_of_week'] in ['Sat', 'Sun']]
    if weekend_chunks:
        causes.append(f"Weekend periods in {len(weekend_chunks)} chunks cause low activity")
    
    # Check for market session imbalance
    session_counts = {}
    for chunk in chunk_analysis:
        session = chunk['market_session']
        session_counts[session] = session_counts.get(session, 0) + 1
    
    if len(session_counts) > 1:
        max_session_chunks = max(session_counts.values())
        min_session_chunks = min(session_counts.values())
        if max_session_chunks > min_session_chunks * 2:
            causes.append("Uneven distribution of market sessions across chunks")
    
    # Check for very different chunk sizes
    durations = [chunk['duration_hours'] for chunk in chunk_analysis]
    if durations:
        max_duration = max(durations)
        min_duration = min(durations)
        if max_duration > min_duration * 2:
            causes.append(f"Inconsistent chunk sizes: {max_duration:.1f}h vs {min_duration:.1f}h")
    
    return causes


def generate_quick_fixes(chunk_analysis: List[Dict]) -> List[str]:
    """Generate quick fixes for the progress issues."""
    
    fixes = []
    
    # Check if we can merge small chunks
    small_chunks = [c for c in chunk_analysis if c['estimated_ticks'] < 50000]
    if len(small_chunks) > 1:
        fixes.append(f"Merge {len(small_chunks)} small chunks with adjacent periods")
    
    # Check if we can split large chunks
    large_chunks = [c for c in chunk_analysis if c['estimated_ticks'] > 200000]
    if large_chunks:
        fixes.append(f"Split {len(large_chunks)} large chunks into smaller periods")
    
    # Weekend handling
    weekend_chunks = [c for c in chunk_analysis if c['day_of_week'] in ['Sat', 'Sun']]
    if weekend_chunks:
        fixes.append("Combine weekend periods with adjacent weekdays")
    
    # Session balancing
    session_counts = {}
    for chunk in chunk_analysis:
        session = chunk['market_session']
        session_counts[session] = session_counts.get(session, 0) + 1
    
    if 'Overlap' in session_counts and session_counts['Overlap'] > len(chunk_analysis) * 0.4:
        fixes.append("Redistribute overlap periods across multiple chunks")
    
    # Batch size optimization
    avg_ticks = sum(c['estimated_ticks'] for c in chunk_analysis) / len(chunk_analysis)
    if avg_ticks > 150000:
        fixes.append("Increase max_ticks_per_batch to handle high-density periods")
    elif avg_ticks < 30000:
        fixes.append("Decrease chunk size to improve granularity")
    
    return fixes


def calculate_performance_impact(chunk_analysis: List[Dict]) -> Dict:
    """Calculate the performance impact of non-uniform progress."""
    
    progresses = [chunk['progress_pct'] for chunk in chunk_analysis]
    tick_densities = [chunk['estimated_ticks'] for chunk in chunk_analysis]
    
    impact = {
        'progress_variation_coefficient': 0.0,
        'tick_density_variation_coefficient': 0.0,
        'estimated_completion_time_variation': 0.0,
        'slowest_chunk_bottleneck_factor': 0.0
    }
    
    if progresses:
        avg_progress = sum(progresses) / len(progresses)
        progress_std = (sum((p - avg_progress) ** 2 for p in progresses) / len(progresses)) ** 0.5
        impact['progress_variation_coefficient'] = progress_std / avg_progress if avg_progress > 0 else 0
    
    if tick_densities:
        avg_ticks = sum(tick_densities) / len(tick_densities)
        tick_std = (sum((t - avg_ticks) ** 2 for t in tick_densities) / len(tick_densities)) ** 0.5
        impact['tick_density_variation_coefficient'] = tick_std / avg_ticks if avg_ticks > 0 else 0
        
        # Estimate completion time variation
        max_ticks = max(tick_densities)
        min_ticks = min(tick_densities)
        impact['estimated_completion_time_variation'] = max_ticks / min_ticks if min_ticks > 0 else 1.0
        
        # Bottleneck factor (how much the slowest chunk slows down overall completion)
        impact['slowest_chunk_bottleneck_factor'] = max_ticks / avg_ticks if avg_ticks > 0 else 1.0
    
    return impact


def apply_quick_fix_to_backtest_request(req, fix_type: str = "auto") -> Dict:
    """Apply quick fixes to a backtest request."""
    
    print(f"APPLYING QUICK FIX: {fix_type}")
    print("=" * 40)
    
    fixes_applied = []
    
    if fix_type == "auto" or fix_type == "batch_size":
        # Optimize batch size based on expected tick density
        duration_days = (req.end_utc - req.start_utc).days
        
        if duration_days > 30:  # Large dataset
            req.max_ticks_per_batch = 50000  # Larger batches for efficiency
            fixes_applied.append("Increased batch size for large dataset")
        elif duration_days < 7:  # Small dataset
            req.max_ticks_per_batch = 10000  # Smaller batches for granularity
            fixes_applied.append("Decreased batch size for better granularity")
    
    if fix_type == "auto" or fix_type == "split_mode":
        # Optimize split mode
        duration_days = (req.end_utc - req.start_utc).days
        
        if duration_days > 90:  # > 3 months
            req.split_mode = "WEEKLY"
            fixes_applied.append("Changed to weekly splits for large dataset")
        elif duration_days > 30:  # > 1 month
            req.split_mode = "DAILY"
            fixes_applied.append("Changed to daily splits for medium dataset")
        else:
            req.split_mode = "NONE"  # No splitting for small datasets
            fixes_applied.append("Disabled splitting for small dataset")
    
    if fix_type == "auto" or fix_type == "workers":
        # Optimize worker count
        duration_days = (req.end_utc - req.start_utc).days
        cpu_cores = os.cpu_count() or 1
        
        if duration_days > 60:  # Large dataset
            req.parallel_workers = min(cpu_cores, 8)
            fixes_applied.append(f"Set parallel workers to {req.parallel_workers} for large dataset")
        elif duration_days < 7:  # Small dataset
            req.parallel_workers = min(cpu_cores // 2, 4)
            fixes_applied.append(f"Reduced parallel workers to {req.parallel_workers} for small dataset")
    
    print("Fixes applied:")
    for fix in fixes_applied:
        print(f"  • {fix}")
    
    return {
        'fixes_applied': fixes_applied,
        'optimized_request': req
    }


def create_balanced_chunk_configuration(start_date: datetime, end_date: datetime, 
                                      target_workers: int = 8) -> Dict:
    """Create a balanced chunk configuration."""
    
    print("CREATING BALANCED CHUNK CONFIGURATION")
    print("=" * 45)
    
    # Analyze the date range
    total_days = (end_date - start_date).days
    
    # Estimate total ticks
    estimated_total_ticks = 0
    current_date = start_date
    
    while current_date < end_date:
        daily_ticks = estimate_tick_density_simple(
            current_date, 
            min(current_date + timedelta(days=1), end_date)
        )
        estimated_total_ticks += daily_ticks
        current_date += timedelta(days=1)
    
    # Calculate optimal chunk parameters
    target_ticks_per_chunk = estimated_total_ticks // (target_workers * 2)  # 2 chunks per worker
    target_ticks_per_chunk = max(50000, min(200000, target_ticks_per_chunk))  # Reasonable bounds
    
    # Create balanced chunks
    chunks = []
    current_start = start_date
    current_chunk_ticks = 0
    
    while current_start < end_date:
        # Find optimal end time for this chunk
        chunk_end = current_start
        
        while chunk_end < end_date and current_chunk_ticks < target_ticks_per_chunk:
            next_end = min(chunk_end + timedelta(hours=6), end_date)  # 6-hour increments
            period_ticks = estimate_tick_density_simple(chunk_end, next_end)
            
            if current_chunk_ticks + period_ticks > target_ticks_per_chunk * 1.2:
                break  # Would exceed target by too much
            
            current_chunk_ticks += period_ticks
            chunk_end = next_end
        
        if chunk_end > current_start:
            chunks.append({
                'start_utc': current_start.isoformat(),
                'end_utc': chunk_end.isoformat(),
                'estimated_ticks': current_chunk_ticks,
                'duration_hours': (chunk_end - current_start).total_seconds() / 3600
            })
        
        current_start = chunk_end
        current_chunk_ticks = 0
    
    # Configuration summary
    config = {
        'total_chunks': len(chunks),
        'target_ticks_per_chunk': target_ticks_per_chunk,
        'estimated_total_ticks': estimated_total_ticks,
        'chunks': chunks,
        'recommended_settings': {
            'parallel_workers': min(target_workers, len(chunks)),
            'max_ticks_per_batch': min(50000, target_ticks_per_chunk // 4),
            'split_mode': 'CUSTOM_BALANCED'
        }
    }
    
    print(f"Created {len(chunks)} balanced chunks:")
    print(f"  Target ticks per chunk: {target_ticks_per_chunk:,}")
    print(f"  Total estimated ticks: {estimated_total_ticks:,}")
    print(f"  Recommended workers: {config['recommended_settings']['parallel_workers']}")
    
    # Show chunk distribution
    for i, chunk in enumerate(chunks[:5]):  # Show first 5 chunks
        start_time = pd.to_datetime(chunk['start_utc'])
        end_time = pd.to_datetime(chunk['end_utc'])
        print(f"  Chunk {i+1}: {start_time.strftime('%m-%d %H:%M')} -> "
              f"{end_time.strftime('%m-%d %H:%M')} ({chunk['estimated_ticks']:,} ticks)")
    
    if len(chunks) > 5:
        print(f"  ... and {len(chunks) - 5} more chunks")
    
    return config


# Example usage for your current issue
def fix_current_backtest_progress():
    """Quick fix for the current backtest progress issue."""
    
    # Example chunk states from your issue
    example_chunk_states = [
        {
            'start_utc': '2026-01-01T06:09:00+00:00',
            'end_utc': '2026-01-05T00:00:00+00:00',
            'progress_pct': 26.91
        },
        {
            'start_utc': '2026-01-05T00:00:00+00:00',
            'end_utc': '2026-01-12T00:00:00+00:00',
            'progress_pct': 4.96
        },
        {
            'start_utc': '2026-01-12T00:00:00+00:00',
            'end_utc': '2026-01-19T00:00:00+00:00',
            'progress_pct': 3.52
        },
        {
            'start_utc': '2026-01-19T00:00:00+00:00',
            'end_utc': '2026-01-26T00:00:00+00:00',
            'progress_pct': 2.61
        },
        {
            'start_utc': '2026-01-26T00:00:00+00:00',
            'end_utc': '2026-02-01T00:00:00+00:00',
            'progress_pct': 1.79
        }
    ]
    
    # Analyze the issue
    analysis = analyze_current_progress_issue(example_chunk_states)
    
    return analysis


if __name__ == "__main__":
    # Run the analysis
    fix_current_backtest_progress()