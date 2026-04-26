"""
Auto-Parallelization Engine with Maximum CPU Utilization
Intelligently divides timeline based on tick density and uses all available CPU cores.
"""
import os
import psutil
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Optional
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import json
import time
from dataclasses import dataclass


@dataclass
class SystemCapabilities:
    """System hardware capabilities for optimization."""
    physical_cores: int
    logical_cores: int
    total_ram_gb: float
    available_ram_gb: float
    cpu_freq_mhz: float
    has_hyperthreading: bool
    optimal_threads: int
    max_memory_per_thread_mb: int


@dataclass
class ChunkInfo:
    """Information about a processing chunk."""
    chunk_id: int
    start_time: datetime
    end_time: datetime
    estimated_ticks: int
    market_sessions: List[str]
    complexity_score: float
    assigned_thread: int
    processing_priority: int


class TickDensityAnalyzer:
    """Advanced tick density analysis engine."""
    
    def __init__(self):
        self.tick_patterns = {}
        self.session_multipliers = {
            'weekend': 0.02,        # 2% of normal activity
            'holiday': 0.05,        # 5% of normal activity
            'asian_quiet': 0.15,    # 15% of peak activity
            'asian_active': 0.35,   # 35% of peak activity
            'london_pre': 0.45,     # 45% of peak activity
            'london_open': 1.0,     # 100% peak activity
            'london_active': 0.85,  # 85% of peak activity
            'overlap_pre': 0.95,    # 95% of peak activity
            'overlap_peak': 1.2,    # 120% peak activity (highest)
            'ny_open': 1.0,         # 100% peak activity
            'ny_active': 0.8,       # 80% of peak activity
            'ny_close': 0.6,        # 60% of peak activity
            'after_hours': 0.1      # 10% of peak activity
        }
        
        # Base tick rates (ticks per hour) for XAUUSD
        self.base_tick_rate = 2500  # Peak hour tick rate
        
    def analyze_timeline_density(self, start_date: datetime, end_date: datetime, 
                               symbol: str = 'XAUUSD') -> Dict:
        """Analyze tick density across the entire timeline."""
        
        print(f"🔍 ANALYZING TICK DENSITY: {symbol}")
        print(f"Timeline: {start_date.strftime('%Y-%m-%d')} -> {end_date.strftime('%Y-%m-%d')}")
        print("=" * 60)
        
        timeline_analysis = {
            'total_duration_hours': 0,
            'estimated_total_ticks': 0,
            'hourly_breakdown': [],
            'daily_summaries': [],
            'session_distribution': {},
            'complexity_zones': [],
            'optimization_opportunities': []
        }
        
        current_time = start_date
        daily_ticks = 0
        daily_sessions = []
        current_date = start_date.date()
        
        while current_time < end_date:
            # Analyze this hour
            hour_info = self._analyze_hour(current_time, symbol)
            timeline_analysis['hourly_breakdown'].append(hour_info)
            timeline_analysis['estimated_total_ticks'] += hour_info['estimated_ticks']
            
            # Track daily progress
            if current_time.date() != current_date:
                # Complete previous day
                timeline_analysis['daily_summaries'].append({
                    'date': current_date.isoformat(),
                    'total_ticks': daily_ticks,
                    'sessions': list(set(daily_sessions)),
                    'avg_ticks_per_hour': daily_ticks / 24 if daily_ticks > 0 else 0
                })
                
                # Reset for new day
                daily_ticks = 0
                daily_sessions = []
                current_date = current_time.date()
            
            daily_ticks += hour_info['estimated_ticks']
            daily_sessions.append(hour_info['session'])
            
            # Track session distribution
            session = hour_info['session']
            if session not in timeline_analysis['session_distribution']:
                timeline_analysis['session_distribution'][session] = {
                    'hours': 0, 'total_ticks': 0
                }
            timeline_analysis['session_distribution'][session]['hours'] += 1
            timeline_analysis['session_distribution'][session]['total_ticks'] += hour_info['estimated_ticks']
            
            current_time += timedelta(hours=1)
        
        # Complete final day
        if daily_ticks > 0:
            timeline_analysis['daily_summaries'].append({
                'date': current_date.isoformat(),
                'total_ticks': daily_ticks,
                'sessions': list(set(daily_sessions)),
                'avg_ticks_per_hour': daily_ticks / 24
            })
        
        timeline_analysis['total_duration_hours'] = len(timeline_analysis['hourly_breakdown'])
        
        # Identify complexity zones
        timeline_analysis['complexity_zones'] = self._identify_complexity_zones(
            timeline_analysis['hourly_breakdown']
        )
        
        # Find optimization opportunities
        timeline_analysis['optimization_opportunities'] = self._find_optimization_opportunities(
            timeline_analysis
        )
        
        self._print_timeline_analysis(timeline_analysis)
        
        return timeline_analysis
    
    def _analyze_hour(self, timestamp: datetime, symbol: str) -> Dict:
        """Analyze tick density for a specific hour."""
        
        hour_utc = timestamp.hour
        day_of_week = timestamp.weekday()
        
        # Determine market session
        session = self._classify_market_session(timestamp)
        
        # Get session multiplier
        multiplier = self.session_multipliers.get(session, 0.5)
        
        # Calculate base ticks for this hour
        base_ticks = self.base_tick_rate * multiplier
        
        # Apply symbol-specific adjustments
        symbol_multiplier = self._get_symbol_multiplier(symbol)
        estimated_ticks = int(base_ticks * symbol_multiplier)
        
        # Add volatility factors
        volatility_factor = self._calculate_volatility_factor(timestamp)
        estimated_ticks = int(estimated_ticks * volatility_factor)
        
        return {
            'timestamp': timestamp.isoformat(),
            'hour_utc': hour_utc,
            'day_of_week': day_of_week,
            'session': session,
            'estimated_ticks': estimated_ticks,
            'multiplier': multiplier,
            'volatility_factor': volatility_factor,
            'complexity_score': self._calculate_hour_complexity(timestamp, estimated_ticks)
        }
    
    def _classify_market_session(self, timestamp: datetime) -> str:
        """Classify market session with high precision."""
        
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        
        # Weekend detection
        if day_of_week >= 5:  # Saturday = 5, Sunday = 6
            return 'weekend'
        
        # Holiday detection (simplified - can be enhanced with holiday calendar)
        if self._is_holiday(timestamp):
            return 'holiday'
        
        # Detailed session classification
        if 0 <= hour <= 1:
            return 'asian_quiet'
        elif 2 <= hour <= 6:
            return 'asian_active'
        elif hour == 7:
            return 'london_pre'
        elif hour == 8:
            return 'london_open'
        elif 9 <= hour <= 11:
            return 'london_active'
        elif hour == 12:
            return 'overlap_pre'
        elif 13 <= hour <= 15:
            return 'overlap_peak'
        elif hour == 16:
            return 'overlap_pre'
        elif hour == 17:
            return 'ny_open'
        elif 18 <= hour <= 20:
            return 'ny_active'
        elif hour == 21:
            return 'ny_close'
        else:  # 22-23
            return 'after_hours'
    
    def _is_holiday(self, timestamp: datetime) -> bool:
        """Check if timestamp falls on a major market holiday."""
        
        # Major holidays that affect forex markets
        major_holidays = [
            (1, 1),   # New Year's Day
            (12, 25), # Christmas Day
            (12, 26), # Boxing Day
        ]
        
        month_day = (timestamp.month, timestamp.day)
        return month_day in major_holidays
    
    def _get_symbol_multiplier(self, symbol: str) -> float:
        """Get symbol-specific tick density multiplier."""
        
        symbol_multipliers = {
            'XAUUSD': 1.0,    # Gold - high activity
            'EURUSD': 0.9,    # EUR/USD - very high activity
            'GBPUSD': 0.8,    # GBP/USD - high activity
            'USDJPY': 0.7,    # USD/JPY - good activity
            'USDCHF': 0.6,    # USD/CHF - moderate activity
            'AUDUSD': 0.5,    # AUD/USD - moderate activity
            'NZDUSD': 0.4,    # NZD/USD - lower activity
        }
        
        return symbol_multipliers.get(symbol, 0.5)
    
    def _calculate_volatility_factor(self, timestamp: datetime) -> float:
        """Calculate volatility factor based on time patterns."""
        
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        
        # Base volatility
        volatility = 1.0
        
        # Higher volatility during market opens
        if hour in [8, 17]:  # London and NY opens
            volatility *= 1.3
        elif hour in [13, 14, 15]:  # Peak overlap
            volatility *= 1.2
        
        # Monday and Friday effects
        if day_of_week == 0:  # Monday
            volatility *= 1.1
        elif day_of_week == 4:  # Friday
            volatility *= 1.05
        
        # Add some randomness for realism
        volatility *= (0.9 + 0.2 * np.random.random())
        
        return volatility
    
    def _calculate_hour_complexity(self, timestamp: datetime, estimated_ticks: int) -> float:
        """Calculate processing complexity score for an hour."""
        
        # Base complexity from tick count
        complexity = estimated_ticks / 1000.0
        
        # Add complexity for high-volatility periods
        session = self._classify_market_session(timestamp)
        if session in ['london_open', 'overlap_peak', 'ny_open']:
            complexity *= 1.5
        elif session in ['overlap_pre', 'london_active', 'ny_active']:
            complexity *= 1.2
        
        return round(complexity, 2)
    
    def _identify_complexity_zones(self, hourly_breakdown: List[Dict]) -> List[Dict]:
        """Identify high and low complexity zones in the timeline."""
        
        zones = []
        current_zone = None
        
        for hour_info in hourly_breakdown:
            complexity = hour_info['complexity_score']
            
            # Classify complexity level
            if complexity > 3.0:
                level = 'very_high'
            elif complexity > 2.0:
                level = 'high'
            elif complexity > 1.0:
                level = 'medium'
            elif complexity > 0.5:
                level = 'low'
            else:
                level = 'very_low'
            
            # Group consecutive hours of similar complexity
            if current_zone is None or current_zone['level'] != level:
                if current_zone:
                    zones.append(current_zone)
                
                current_zone = {
                    'level': level,
                    'start_time': hour_info['timestamp'],
                    'end_time': hour_info['timestamp'],
                    'total_ticks': hour_info['estimated_ticks'],
                    'hour_count': 1,
                    'avg_complexity': complexity
                }
            else:
                current_zone['end_time'] = hour_info['timestamp']
                current_zone['total_ticks'] += hour_info['estimated_ticks']
                current_zone['hour_count'] += 1
                current_zone['avg_complexity'] = (
                    current_zone['avg_complexity'] + complexity
                ) / 2
        
        if current_zone:
            zones.append(current_zone)
        
        return zones
    
    def _find_optimization_opportunities(self, timeline_analysis: Dict) -> List[str]:
        """Find opportunities to optimize processing."""
        
        opportunities = []
        
        # Check for large low-activity periods
        low_activity_hours = sum(
            1 for hour in timeline_analysis['hourly_breakdown']
            if hour['estimated_ticks'] < 500
        )
        
        if low_activity_hours > timeline_analysis['total_duration_hours'] * 0.3:
            opportunities.append(
                f"Merge {low_activity_hours} low-activity hours into larger chunks"
            )
        
        # Check for high-activity concentration
        high_activity_hours = sum(
            1 for hour in timeline_analysis['hourly_breakdown']
            if hour['estimated_ticks'] > 3000
        )
        
        if high_activity_hours > 0:
            opportunities.append(
                f"Split {high_activity_hours} high-activity hours into smaller chunks"
            )
        
        # Check session distribution
        sessions = timeline_analysis['session_distribution']
        if 'weekend' in sessions and sessions['weekend']['hours'] > 48:
            opportunities.append("Combine weekend periods with adjacent weekdays")
        
        return opportunities
    
    def _print_timeline_analysis(self, analysis: Dict):
        """Print comprehensive timeline analysis."""
        
        print(f"\n📊 TIMELINE ANALYSIS RESULTS")
        print("=" * 50)
        print(f"Total Duration: {analysis['total_duration_hours']} hours")
        print(f"Estimated Total Ticks: {analysis['estimated_total_ticks']:,}")
        print(f"Average Ticks/Hour: {analysis['estimated_total_ticks'] / analysis['total_duration_hours']:,.0f}")
        
        print(f"\n📈 SESSION DISTRIBUTION:")
        for session, data in analysis['session_distribution'].items():
            avg_ticks = data['total_ticks'] / data['hours'] if data['hours'] > 0 else 0
            print(f"  {session:15}: {data['hours']:3}h | {data['total_ticks']:8,} ticks | {avg_ticks:6,.0f} avg/h")
        
        print(f"\n🎯 COMPLEXITY ZONES:")
        for zone in analysis['complexity_zones']:
            print(f"  {zone['level']:10}: {zone['hour_count']:3}h | {zone['total_ticks']:8,} ticks | {zone['avg_complexity']:4.1f} complexity")
        
        if analysis['optimization_opportunities']:
            print(f"\n💡 OPTIMIZATION OPPORTUNITIES:")
            for opp in analysis['optimization_opportunities']:
                print(f"  • {opp}")


class AutoParallelizationEngine:
    """Maximum CPU utilization engine with intelligent load balancing."""
    
    def __init__(self):
        self.system_caps = self._detect_system_capabilities()
        self.tick_analyzer = TickDensityAnalyzer()
        self.performance_history = []
        
        print(f"\n🚀 AUTO-PARALLELIZATION ENGINE INITIALIZED")
        print("=" * 50)
        self._print_system_capabilities()
    
    def _detect_system_capabilities(self) -> SystemCapabilities:
        """Detect and analyze system hardware capabilities."""
        
        # CPU information
        physical_cores = psutil.cpu_count(logical=False)
        logical_cores = psutil.cpu_count(logical=True)
        has_hyperthreading = logical_cores > physical_cores
        
        # Memory information
        memory = psutil.virtual_memory()
        total_ram_gb = memory.total / (1024**3)
        available_ram_gb = memory.available / (1024**3)
        
        # CPU frequency
        try:
            cpu_freq = psutil.cpu_freq()
            cpu_freq_mhz = cpu_freq.current if cpu_freq else 2400
        except:
            cpu_freq_mhz = 2400  # Default assumption
        
        # Calculate optimal thread count
        # Use all logical cores but leave 1-2 for system
        optimal_threads = max(1, logical_cores - 1)
        
        # Calculate memory per thread (leave 2GB for system)
        available_for_threads = max(1, available_ram_gb - 2)
        max_memory_per_thread_mb = int((available_for_threads * 1024) / optimal_threads)
        
        return SystemCapabilities(
            physical_cores=physical_cores,
            logical_cores=logical_cores,
            total_ram_gb=total_ram_gb,
            available_ram_gb=available_ram_gb,
            cpu_freq_mhz=cpu_freq_mhz,
            has_hyperthreading=has_hyperthreading,
            optimal_threads=optimal_threads,
            max_memory_per_thread_mb=max_memory_per_thread_mb
        )
    
    def _print_system_capabilities(self):
        """Print detected system capabilities."""
        
        caps = self.system_caps
        print(f"CPU Cores: {caps.physical_cores} physical, {caps.logical_cores} logical")
        print(f"Hyperthreading: {'Enabled' if caps.has_hyperthreading else 'Disabled'}")
        print(f"CPU Frequency: {caps.cpu_freq_mhz:.0f} MHz")
        print(f"Total RAM: {caps.total_ram_gb:.1f} GB")
        print(f"Available RAM: {caps.available_ram_gb:.1f} GB")
        print(f"Optimal Threads: {caps.optimal_threads}")
        print(f"Memory per Thread: {caps.max_memory_per_thread_mb} MB")
    
    def create_optimal_chunks(self, start_date: datetime, end_date: datetime,
                            symbol: str = 'XAUUSD') -> List[ChunkInfo]:
        """Create optimally balanced chunks for maximum parallelization."""
        
        print(f"\n🧩 CREATING OPTIMAL CHUNKS")
        print("=" * 40)
        
        # Analyze timeline tick density
        timeline_analysis = self.tick_analyzer.analyze_timeline_density(
            start_date, end_date, symbol
        )
        
        # Calculate target parameters
        total_ticks = timeline_analysis['estimated_total_ticks']
        optimal_threads = self.system_caps.optimal_threads
        
        # Target ticks per chunk (aim for 2-3 chunks per thread)
        chunks_per_thread = 2.5
        target_chunks = int(optimal_threads * chunks_per_thread)
        target_ticks_per_chunk = total_ticks // target_chunks
        
        # Ensure reasonable chunk sizes
        min_ticks_per_chunk = 30000   # Minimum for efficiency
        max_ticks_per_chunk = 200000  # Maximum for memory constraints
        
        target_ticks_per_chunk = max(min_ticks_per_chunk, 
                                   min(max_ticks_per_chunk, target_ticks_per_chunk))
        
        print(f"Target Chunks: {target_chunks}")
        print(f"Target Ticks per Chunk: {target_ticks_per_chunk:,}")
        print(f"Optimal Threads: {optimal_threads}")
        
        # Create balanced chunks
        chunks = self._create_balanced_chunks(
            timeline_analysis, target_ticks_per_chunk, optimal_threads
        )
        
        # Assign threads to chunks
        chunks = self._assign_threads_to_chunks(chunks, optimal_threads)
        
        self._print_chunk_distribution(chunks)
        
        return chunks
    
    def _create_balanced_chunks(self, timeline_analysis: Dict, 
                              target_ticks_per_chunk: int,
                              max_threads: int) -> List[ChunkInfo]:
        """Create balanced chunks based on tick density."""
        
        chunks = []
        hourly_data = timeline_analysis['hourly_breakdown']
        
        current_chunk_ticks = 0
        current_chunk_start = None
        current_sessions = []
        chunk_id = 0
        
        for hour_info in hourly_data:
            hour_timestamp = pd.to_datetime(hour_info['timestamp'])
            hour_ticks = hour_info['estimated_ticks']
            hour_session = hour_info['session']
            
            # Start new chunk if needed
            if current_chunk_start is None:
                current_chunk_start = hour_timestamp
                current_sessions = []
            
            # Add this hour to current chunk
            current_chunk_ticks += hour_ticks
            if hour_session not in current_sessions:
                current_sessions.append(hour_session)
            
            # Check if we should close this chunk
            should_close_chunk = (
                current_chunk_ticks >= target_ticks_per_chunk * 0.8 or  # Reached target
                current_chunk_ticks >= target_ticks_per_chunk * 1.5 or  # Exceeded target significantly
                hour_info == hourly_data[-1]  # Last hour
            )
            
            if should_close_chunk:
                # Calculate complexity score
                complexity_score = self._calculate_chunk_complexity(
                    current_chunk_start, hour_timestamp, current_chunk_ticks, current_sessions
                )
                
                chunk = ChunkInfo(
                    chunk_id=chunk_id,
                    start_time=current_chunk_start,
                    end_time=hour_timestamp + timedelta(hours=1),
                    estimated_ticks=current_chunk_ticks,
                    market_sessions=current_sessions.copy(),
                    complexity_score=complexity_score,
                    assigned_thread=-1,  # Will be assigned later
                    processing_priority=self._calculate_processing_priority(complexity_score)
                )
                
                chunks.append(chunk)
                
                # Reset for next chunk
                chunk_id += 1
                current_chunk_start = None
                current_chunk_ticks = 0
                current_sessions = []
        
        return chunks
    
    def _calculate_chunk_complexity(self, start_time: datetime, end_time: datetime,
                                  total_ticks: int, sessions: List[str]) -> float:
        """Calculate complexity score for a chunk."""
        
        # Base complexity from tick count
        base_complexity = total_ticks / 50000.0  # Normalize to 50K ticks = 1.0
        
        # Session complexity multipliers
        session_multipliers = {
            'weekend': 0.5,
            'holiday': 0.5,
            'asian_quiet': 0.7,
            'asian_active': 0.9,
            'london_open': 1.5,
            'overlap_peak': 1.8,
            'ny_open': 1.4,
            'after_hours': 0.6
        }
        
        # Calculate session complexity
        session_complexity = 1.0
        for session in sessions:
            multiplier = session_multipliers.get(session, 1.0)
            session_complexity = max(session_complexity, multiplier)
        
        # Duration complexity (longer chunks are slightly more complex)
        duration_hours = (end_time - start_time).total_seconds() / 3600
        duration_complexity = 1.0 + (duration_hours - 24) * 0.01  # +1% per hour over 24h
        
        total_complexity = base_complexity * session_complexity * duration_complexity
        
        return round(total_complexity, 2)
    
    def _calculate_processing_priority(self, complexity_score: float) -> int:
        """Calculate processing priority (1=highest, 5=lowest)."""
        
        if complexity_score > 3.0:
            return 1  # Highest priority
        elif complexity_score > 2.0:
            return 2  # High priority
        elif complexity_score > 1.0:
            return 3  # Medium priority
        elif complexity_score > 0.5:
            return 4  # Low priority
        else:
            return 5  # Lowest priority
    
    def _assign_threads_to_chunks(self, chunks: List[ChunkInfo], 
                                max_threads: int) -> List[ChunkInfo]:
        """Assign threads to chunks for optimal load balancing."""
        
        # Sort chunks by complexity (highest first)
        sorted_chunks = sorted(chunks, key=lambda c: c.complexity_score, reverse=True)
        
        # Initialize thread loads
        thread_loads = [0] * max_threads
        thread_assignments = [[] for _ in range(max_threads)]
        
        # Assign chunks using a greedy algorithm (assign to least loaded thread)
        for chunk in sorted_chunks:
            # Find thread with minimum load
            min_load_thread = thread_loads.index(min(thread_loads))
            
            # Assign chunk to this thread
            chunk.assigned_thread = min_load_thread
            thread_loads[min_load_thread] += chunk.estimated_ticks
            thread_assignments[min_load_thread].append(chunk)
        
        # Print thread assignments
        print(f"\n🧵 THREAD ASSIGNMENTS:")
        for thread_id in range(max_threads):
            assigned_chunks = thread_assignments[thread_id]
            total_ticks = sum(c.estimated_ticks for c in assigned_chunks)
            print(f"  Thread {thread_id}: {len(assigned_chunks)} chunks, {total_ticks:,} ticks")
        
        return chunks
    
    def _print_chunk_distribution(self, chunks: List[ChunkInfo]):
        """Print detailed chunk distribution."""
        
        print(f"\n📋 CHUNK DISTRIBUTION ({len(chunks)} chunks)")
        print("=" * 80)
        print(f"{'ID':3} {'Start Time':16} {'End Time':16} {'Ticks':8} {'Sessions':20} {'Thread':6} {'Priority':8}")
        print("-" * 80)
        
        for chunk in chunks:
            sessions_str = ', '.join(chunk.market_sessions[:3])  # Show first 3 sessions
            if len(chunk.market_sessions) > 3:
                sessions_str += '...'
            
            print(f"{chunk.chunk_id:3} "
                  f"{chunk.start_time.strftime('%m-%d %H:%M'):16} "
                  f"{chunk.end_time.strftime('%m-%d %H:%M'):16} "
                  f"{chunk.estimated_ticks:8,} "
                  f"{sessions_str:20} "
                  f"{chunk.assigned_thread:6} "
                  f"{chunk.processing_priority:8}")
        
        # Summary statistics
        total_ticks = sum(c.estimated_ticks for c in chunks)
        avg_ticks = total_ticks / len(chunks)
        min_ticks = min(c.estimated_ticks for c in chunks)
        max_ticks = max(c.estimated_ticks for c in chunks)
        
        print("-" * 80)
        print(f"Total Ticks: {total_ticks:,}")
        print(f"Average per Chunk: {avg_ticks:,.0f}")
        print(f"Range: {min_ticks:,} - {max_ticks:,}")
        print(f"Load Balance Ratio: {max_ticks/min_ticks:.1f}x")


# Global auto-parallelization engine
auto_parallel_engine = AutoParallelizationEngine()