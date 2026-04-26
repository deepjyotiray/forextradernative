"""
Ultra-High Performance Parallel Processing Framework
Maximizes CPU utilization for fastest possible backtesting.
"""
import numpy as np
import numba
from numba import jit, prange, cuda
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import threading
import queue
import time
import os
import psutil
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import ctypes


# System configuration
CPU_CORES = os.cpu_count() or 1
LOGICAL_CORES = psutil.cpu_count(logical=True)
PHYSICAL_CORES = psutil.cpu_count(logical=False)
MAX_WORKERS = LOGICAL_CORES  # Use all logical cores
CHUNK_SIZE_PER_CORE = 50000  # Optimal chunk size per core


@dataclass
class ProcessingTask:
    """Task definition for parallel processing."""
    task_id: int
    data_chunk: Dict
    start_idx: int
    end_idx: int
    parameters: Dict


@jit(nopython=True, cache=True, parallel=True)
def ultra_parallel_indicator_computation(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    chunk_starts: np.ndarray,
    chunk_ends: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute indicators in parallel chunks for maximum CPU utilization."""
    
    n_chunks = len(chunk_starts)
    total_length = len(closes)
    
    # Pre-allocate result arrays
    ema9_results = np.zeros(total_length, dtype=np.float64)
    ema20_results = np.zeros(total_length, dtype=np.float64)
    atr_results = np.zeros(total_length, dtype=np.float64)
    rsi_results = np.zeros(total_length, dtype=np.float64)
    
    # Process chunks in parallel
    for chunk_idx in prange(n_chunks):
        start = chunk_starts[chunk_idx]
        end = chunk_ends[chunk_idx]
        
        if end > start:
            # EMA calculations for this chunk
            alpha9 = 2.0 / 10.0
            alpha20 = 2.0 / 21.0
            
            # Initialize with first value or previous chunk's last value
            if start == 0:
                ema9_results[start] = closes[start]
                ema20_results[start] = closes[start]
            else:
                ema9_results[start] = ema9_results[start-1]
                ema20_results[start] = ema20_results[start-1]
            
            # Compute EMA for chunk
            for i in range(start + 1, end):
                ema9_results[i] = alpha9 * closes[i] + (1.0 - alpha9) * ema9_results[i-1]
                ema20_results[i] = alpha20 * closes[i] + (1.0 - alpha20) * ema20_results[i-1]
            
            # ATR calculation for chunk
            atr_alpha = 2.0 / 15.0
            if start == 0:
                atr_results[start] = highs[start] - lows[start]
            else:
                atr_results[start] = atr_results[start-1]
            
            for i in range(start + 1, end):
                tr1 = highs[i] - lows[i]
                tr2 = abs(highs[i] - closes[i-1])
                tr3 = abs(lows[i] - closes[i-1])
                tr = max(tr1, max(tr2, tr3))
                atr_results[i] = atr_alpha * tr + (1.0 - atr_alpha) * atr_results[i-1]
            
            # RSI calculation for chunk (simplified)
            rsi_period = 14
            for i in range(max(start, rsi_period), end):
                gains = 0.0
                losses = 0.0
                
                for j in range(i - rsi_period + 1, i + 1):
                    delta = closes[j] - closes[j-1]
                    if delta > 0:
                        gains += delta
                    else:
                        losses -= delta
                
                if losses == 0:
                    rsi_results[i] = 100.0
                else:
                    rs = gains / losses
                    rsi_results[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return ema9_results, ema20_results, atr_results, rsi_results


@jit(nopython=True, cache=True, parallel=True)
def ultra_parallel_signal_generation(
    ema9: np.ndarray,
    ema20: np.ndarray,
    ema50: np.ndarray,
    atr: np.ndarray,
    rsi: np.ndarray,
    closes: np.ndarray,
    bids: np.ndarray,
    asks: np.ndarray,
    chunk_starts: np.ndarray,
    chunk_ends: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate trading signals in parallel for maximum speed."""
    
    n_total = len(closes)
    n_chunks = len(chunk_starts)
    
    # Pre-allocate result arrays
    signals = np.zeros(n_total, dtype=np.int8)
    entry_prices = np.zeros(n_total, dtype=np.float64)
    stop_losses = np.zeros(n_total, dtype=np.float64)
    take_profits = np.zeros(n_total, dtype=np.float64)
    
    # Process each chunk in parallel
    for chunk_idx in prange(n_chunks):
        start = chunk_starts[chunk_idx]
        end = chunk_ends[chunk_idx]
        
        for i in range(max(start, 50), end):  # Need history for signals
            
            # Trend analysis
            trend_up = (ema9[i] > ema20[i] and 
                       ema20[i] > ema50[i] and
                       ema9[i] > ema9[i-1] and
                       ema20[i] > ema20[i-1])
            
            trend_down = (ema9[i] < ema20[i] and 
                         ema20[i] < ema50[i] and
                         ema9[i] < ema9[i-1] and
                         ema20[i] < ema20[i-1])
            
            # Momentum conditions
            momentum_up = (rsi[i] > 40 and rsi[i] < 70 and
                          closes[i] > closes[i-1] and
                          closes[i] > ema9[i])
            
            momentum_down = (rsi[i] < 60 and rsi[i] > 30 and
                            closes[i] < closes[i-1] and
                            closes[i] < ema9[i])
            
            # Volatility filter
            vol_expansion = atr[i] > atr[i-5] * 1.1
            vol_reasonable = atr[i] < atr[i-20] * 3.0  # Not too volatile
            
            # Generate buy signal
            if (trend_up and momentum_up and vol_expansion and vol_reasonable):
                signals[i] = 1
                entry_prices[i] = asks[i]
                stop_losses[i] = entry_prices[i] - atr[i] * 2.0
                take_profits[i] = entry_prices[i] + atr[i] * 3.0
            
            # Generate sell signal
            elif (trend_down and momentum_down and vol_expansion and vol_reasonable):
                signals[i] = -1
                entry_prices[i] = bids[i]
                stop_losses[i] = entry_prices[i] + atr[i] * 2.0
                take_profits[i] = entry_prices[i] - atr[i] * 3.0
    
    return signals, entry_prices, stop_losses, take_profits


@jit(nopython=True, cache=True)
def ultra_fast_trade_execution_chunk(
    signals: np.ndarray,
    entry_prices: np.ndarray,
    stop_losses: np.ndarray,
    take_profits: np.ndarray,
    bids: np.ndarray,
    asks: np.ndarray,
    timestamps: np.ndarray,
    start_idx: int,
    end_idx: int,
    max_positions: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Execute trades for a specific chunk with ultra-fast processing."""
    
    chunk_size = end_idx - start_idx
    max_trades = chunk_size // 5  # Estimate max trades in chunk
    
    # Pre-allocate trade arrays
    trade_entries = np.zeros(max_trades, dtype=np.int32)
    trade_exits = np.zeros(max_trades, dtype=np.int32)
    trade_pnls = np.zeros(max_trades, dtype=np.float64)
    trade_directions = np.zeros(max_trades, dtype=np.int8)
    trade_durations = np.zeros(max_trades, dtype=np.int32)
    
    # Position tracking
    open_positions = 0
    trade_count = 0
    
    # Position arrays
    pos_entries = np.zeros(max_positions, dtype=np.int32)
    pos_directions = np.zeros(max_positions, dtype=np.int8)
    pos_entry_prices = np.zeros(max_positions, dtype=np.float64)
    pos_stops = np.zeros(max_positions, dtype=np.float64)
    pos_targets = np.zeros(max_positions, dtype=np.float64)
    
    for i in range(start_idx, end_idx):
        current_bid = bids[i]
        current_ask = asks[i]
        
        # Check exits for open positions
        for pos_idx in range(open_positions):
            if pos_entries[pos_idx] == 0:
                continue
            
            direction = pos_directions[pos_idx]
            entry_price = pos_entry_prices[pos_idx]
            stop_loss = pos_stops[pos_idx]
            take_profit = pos_targets[pos_idx]
            
            exit_triggered = False
            exit_price = 0.0
            
            if direction == 1:  # Long position
                if current_bid <= stop_loss:
                    exit_price = stop_loss
                    exit_triggered = True
                elif current_bid >= take_profit:
                    exit_price = take_profit
                    exit_triggered = True
            else:  # Short position
                if current_ask >= stop_loss:
                    exit_price = stop_loss
                    exit_triggered = True
                elif current_ask <= take_profit:
                    exit_price = take_profit
                    exit_triggered = True
            
            if exit_triggered and trade_count < max_trades:
                # Calculate PnL
                if direction == 1:
                    pnl = (exit_price - entry_price) * 100000
                else:
                    pnl = (entry_price - exit_price) * 100000
                
                # Record trade
                trade_entries[trade_count] = pos_entries[pos_idx]
                trade_exits[trade_count] = i
                trade_pnls[trade_count] = pnl
                trade_directions[trade_count] = direction
                trade_durations[trade_count] = i - pos_entries[pos_idx]
                trade_count += 1
                
                # Close position
                pos_entries[pos_idx] = 0
        
        # Compact positions array
        new_open_count = 0
        for pos_idx in range(open_positions):
            if pos_entries[pos_idx] != 0:
                if new_open_count != pos_idx:
                    pos_entries[new_open_count] = pos_entries[pos_idx]
                    pos_directions[new_open_count] = pos_directions[pos_idx]
                    pos_entry_prices[new_open_count] = pos_entry_prices[pos_idx]
                    pos_stops[new_open_count] = pos_stops[pos_idx]
                    pos_targets[new_open_count] = pos_targets[pos_idx]
                new_open_count += 1
        open_positions = new_open_count
        
        # Check for new signals
        if signals[i] != 0 and open_positions < max_positions:
            pos_entries[open_positions] = i
            pos_directions[open_positions] = signals[i]
            pos_entry_prices[open_positions] = entry_prices[i]
            pos_stops[open_positions] = stop_losses[i]
            pos_targets[open_positions] = take_profits[i]
            open_positions += 1
    
    return (trade_entries[:trade_count], 
            trade_exits[:trade_count],
            trade_pnls[:trade_count], 
            trade_directions[:trade_count],
            trade_durations[:trade_count])


class UltraParallelProcessor:
    """Ultra-high performance parallel processor using all system resources."""
    
    def __init__(self):
        self.cpu_cores = CPU_CORES
        self.logical_cores = LOGICAL_CORES
        self.physical_cores = PHYSICAL_CORES
        self.max_workers = MAX_WORKERS
        
        # Set CPU affinity for maximum performance
        self._optimize_cpu_settings()
        
        print(f"Ultra-Parallel Processor initialized:")
        print(f"  Physical cores: {self.physical_cores}")
        print(f"  Logical cores: {self.logical_cores}")
        print(f"  Max workers: {self.max_workers}")
        print(f"  Chunk size per core: {CHUNK_SIZE_PER_CORE:,}")
    
    def _optimize_cpu_settings(self):
        """Optimize CPU settings for maximum performance."""
        
        try:
            # Set process priority to high
            import psutil
            p = psutil.Process()
            if hasattr(p, 'nice'):
                p.nice(-10)  # High priority on Unix
            elif hasattr(p, 'priority'):
                p.priority(psutil.HIGH_PRIORITY_CLASS)  # High priority on Windows
        except Exception as e:
            print(f"Warning: Could not set high priority: {e}")
        
        # Set environment variables for optimal performance
        os.environ['NUMBA_NUM_THREADS'] = str(self.max_workers)
        os.environ['OMP_NUM_THREADS'] = str(self.max_workers)
        os.environ['MKL_NUM_THREADS'] = str(self.max_workers)
    
    def create_optimal_chunks(self, data_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create optimal data chunks for parallel processing."""
        
        # Calculate optimal chunk size
        total_chunks = max(self.max_workers, data_length // CHUNK_SIZE_PER_CORE)
        chunk_size = max(1000, data_length // total_chunks)  # Minimum 1000 elements per chunk
        
        # Create chunk boundaries
        chunk_starts = []
        chunk_ends = []
        
        for i in range(0, data_length, chunk_size):
            chunk_starts.append(i)
            chunk_ends.append(min(i + chunk_size, data_length))
        
        print(f"Created {len(chunk_starts)} chunks of ~{chunk_size:,} elements each")
        
        return np.array(chunk_starts, dtype=np.int32), np.array(chunk_ends, dtype=np.int32)
    
    def parallel_indicator_computation(self, candle_data: Dict) -> Dict:
        """Compute all indicators using maximum parallelization."""
        
        print("Computing indicators with maximum parallelization...")
        start_time = time.time()
        
        closes = candle_data['closes']
        highs = candle_data['highs']
        lows = candle_data['lows']
        
        # Create optimal chunks
        chunk_starts, chunk_ends = self.create_optimal_chunks(len(closes))
        
        # Compute indicators in parallel
        ema9, ema20, atr, rsi = ultra_parallel_indicator_computation(
            closes, highs, lows, chunk_starts, chunk_ends
        )
        
        # Compute EMA50 separately (needs more history)
        ema50 = self._compute_ema50_parallel(closes, chunk_starts, chunk_ends)
        
        computation_time = time.time() - start_time
        elements_per_second = len(closes) / computation_time if computation_time > 0 else 0
        
        print(f"Indicator computation completed in {computation_time:.3f}s")
        print(f"Processing speed: {elements_per_second:,.0f} elements/second")
        
        return {
            'ema9': ema9,
            'ema20': ema20,
            'ema50': ema50,
            'atr': atr,
            'rsi': rsi,
            'closes': closes,
            'highs': highs,
            'lows': lows
        }
    
    @jit(nopython=True, cache=True, parallel=True)
    def _compute_ema50_parallel(self, closes: np.ndarray, 
                               chunk_starts: np.ndarray, 
                               chunk_ends: np.ndarray) -> np.ndarray:
        """Compute EMA50 with parallel processing."""
        
        n = len(closes)
        ema50 = np.zeros(n, dtype=np.float64)
        alpha = 2.0 / 51.0
        
        # Initialize
        ema50[0] = closes[0]
        
        # Sequential computation (EMA requires previous values)
        for i in range(1, n):
            ema50[i] = alpha * closes[i] + (1.0 - alpha) * ema50[i-1]
        
        return ema50
    
    def parallel_signal_generation(self, indicators: Dict, tick_data: Dict) -> Dict:
        """Generate trading signals using maximum parallelization."""
        
        print("Generating signals with maximum parallelization...")
        start_time = time.time()
        
        # Align tick data with indicators (simplified for speed)
        n_ticks = len(tick_data['bids'])
        n_candles = len(indicators['closes'])
        
        # Create alignment (every N ticks = 1 candle)
        alignment_ratio = max(1, n_ticks // n_candles)
        
        # Expand indicators to tick resolution
        expanded_indicators = self._expand_indicators_to_ticks(indicators, n_ticks, alignment_ratio)
        
        # Create chunks for parallel processing
        chunk_starts, chunk_ends = self.create_optimal_chunks(n_ticks)
        
        # Generate signals in parallel
        signals, entry_prices, stop_losses, take_profits = ultra_parallel_signal_generation(
            expanded_indicators['ema9'],
            expanded_indicators['ema20'],
            expanded_indicators['ema50'],
            expanded_indicators['atr'],
            expanded_indicators['rsi'],
            expanded_indicators['closes'],
            tick_data['bids'],
            tick_data['asks'],
            chunk_starts,
            chunk_ends
        )
        
        signal_time = time.time() - start_time
        signals_per_second = n_ticks / signal_time if signal_time > 0 else 0
        
        print(f"Signal generation completed in {signal_time:.3f}s")
        print(f"Processing speed: {signals_per_second:,.0f} signals/second")
        
        return {
            'signals': signals,
            'entry_prices': entry_prices,
            'stop_losses': stop_losses,
            'take_profits': take_profits
        }
    
    def _expand_indicators_to_ticks(self, indicators: Dict, n_ticks: int, ratio: int) -> Dict:
        """Expand candle indicators to tick resolution."""
        
        expanded = {}
        
        for key, values in indicators.items():
            if isinstance(values, np.ndarray):
                # Repeat each value 'ratio' times and trim to exact tick count
                expanded_values = np.repeat(values, ratio)
                if len(expanded_values) > n_ticks:
                    expanded[key] = expanded_values[:n_ticks]
                elif len(expanded_values) < n_ticks:
                    # Pad with last value
                    padding = np.full(n_ticks - len(expanded_values), values[-1])
                    expanded[key] = np.concatenate([expanded_values, padding])
                else:
                    expanded[key] = expanded_values
        
        return expanded
    
    def parallel_trade_execution(self, signals: Dict, tick_data: Dict, 
                               initial_balance: float) -> Dict:
        """Execute trades using maximum parallelization."""
        
        print("Executing trades with maximum parallelization...")
        start_time = time.time()
        
        n_ticks = len(tick_data['bids'])
        
        # Create chunks for parallel execution
        chunk_starts, chunk_ends = self.create_optimal_chunks(n_ticks)
        
        # Execute trades in parallel chunks
        all_trades = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            
            for i in range(len(chunk_starts)):
                future = executor.submit(
                    self._execute_chunk,
                    signals, tick_data, chunk_starts[i], chunk_ends[i]
                )
                futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                chunk_trades = future.result()
                all_trades.extend(chunk_trades)
        
        # Sort trades by entry time
        all_trades.sort(key=lambda x: x['entry_idx'])
        
        execution_time = time.time() - start_time
        trades_per_second = len(all_trades) / execution_time if execution_time > 0 else 0
        
        print(f"Trade execution completed in {execution_time:.3f}s")
        print(f"Generated {len(all_trades)} trades")
        print(f"Processing speed: {trades_per_second:.1f} trades/second")
        
        return {
            'trades': all_trades,
            'execution_time': execution_time
        }
    
    def _execute_chunk(self, signals: Dict, tick_data: Dict, 
                      start_idx: int, end_idx: int) -> List[Dict]:
        """Execute trades for a single chunk."""
        
        # Use ultra-fast trade execution
        trade_entries, trade_exits, trade_pnls, trade_directions, trade_durations = \
            ultra_fast_trade_execution_chunk(
                signals['signals'],
                signals['entry_prices'],
                signals['stop_losses'],
                signals['take_profits'],
                tick_data['bids'],
                tick_data['asks'],
                tick_data['timestamps'],
                start_idx,
                end_idx,
                max_positions=3  # Limit positions per chunk
            )
        
        # Convert to trade dictionaries
        trades = []
        for i in range(len(trade_pnls)):
            trades.append({
                'entry_idx': int(trade_entries[i]),
                'exit_idx': int(trade_exits[i]),
                'direction': 'BUY' if trade_directions[i] == 1 else 'SELL',
                'pnl': float(trade_pnls[i]),
                'duration_ticks': int(trade_durations[i]),
                'won': trade_pnls[i] > 0
            })
        
        return trades
    
    def run_ultra_parallel_backtest(self, data: Dict, req) -> Dict:
        """Run complete backtest using maximum parallelization."""
        
        print("Starting ultra-parallel backtest...")
        total_start_time = time.time()
        
        # Get primary data
        m1_candles = data['candles']['M1']
        tick_data = data['ticks']
        
        # Step 1: Parallel indicator computation
        indicators = self.parallel_indicator_computation(m1_candles)
        
        # Step 2: Parallel signal generation
        signals = self.parallel_signal_generation(indicators, tick_data)
        
        # Step 3: Parallel trade execution
        execution_result = self.parallel_trade_execution(signals, tick_data, req.initial_balance)
        
        # Build final result
        total_time = time.time() - total_start_time
        
        trades = execution_result['trades']
        total_pnl = sum(t['pnl'] for t in trades)
        wins = sum(1 for t in trades if t['won'])
        
        result = {
            'summary': {
                'initial_balance': req.initial_balance,
                'final_balance': req.initial_balance + total_pnl,
                'total_pnl': total_pnl,
                'trades': len(trades),
                'wins': wins,
                'losses': len(trades) - wins,
                'win_rate': wins / len(trades) if trades else 0.0
            },
            'trades': trades,
            'meta': {
                'engine_type': 'ultra_parallel',
                'total_execution_time': total_time,
                'ticks_processed': len(tick_data['bids']),
                'ticks_per_second': len(tick_data['bids']) / total_time if total_time > 0 else 0,
                'cpu_cores_used': self.max_workers,
                'parallel_efficiency': f"{(len(tick_data['bids']) / total_time / self.max_workers):,.0f} ticks/core/second"
            }
        }
        
        print(f"Ultra-parallel backtest completed in {total_time:.2f}s")
        print(f"Overall speed: {len(tick_data['bids']) / total_time:,.0f} ticks/second")
        print(f"Parallel efficiency: {result['meta']['parallel_efficiency']}")
        
        return result


# Global ultra-parallel processor
ultra_parallel_processor = UltraParallelProcessor()