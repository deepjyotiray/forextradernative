"""
Vectorized backtesting engine for high-performance execution.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone
import numba
from concurrent.futures import ThreadPoolExecutor

from .backtest_sim import BacktestRequest, SimExecutionEngine
from .indicators import ema, atr, rsi


@numba.jit(nopython=True, cache=True)
def fast_tick_metrics(bids: np.ndarray, asks: np.ndarray, timestamps: np.ndarray, window: int = 30) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized tick metrics calculation."""
    n = len(bids)
    velocity = np.zeros(n)
    momentum = np.zeros(n)
    direction = np.zeros(n)
    
    for i in range(window, n):
        start_idx = i - window
        time_span = timestamps[i] - timestamps[start_idx]
        if time_span > 0:
            velocity[i] = window / time_span
            momentum[i] = (bids[i] - bids[start_idx]) / time_span
        
        # Direction calculation
        up_ticks = 0
        down_ticks = 0
        for j in range(start_idx + 1, i + 1):
            if bids[j] > bids[j-1]:
                up_ticks += 1
            elif bids[j] < bids[j-1]:
                down_ticks += 1
        
        total_dir = up_ticks + down_ticks
        if total_dir > 0:
            direction[i] = up_ticks / total_dir
    
    return velocity, momentum, direction


@numba.jit(nopython=True, cache=True)
def fast_signal_detection(closes: np.ndarray, ema9: np.ndarray, ema20: np.ndarray, 
                         rsi_vals: np.ndarray, atr_vals: np.ndarray) -> np.ndarray:
    """Vectorized signal detection logic."""
    n = len(closes)
    signals = np.zeros(n, dtype=np.int8)  # 0=no signal, 1=buy, -1=sell
    
    for i in range(20, n):  # Need history for calculations
        # Simple trend following logic
        if (ema9[i] > ema20[i] and 
            ema9[i] > ema9[i-1] and 
            rsi_vals[i] > 30 and rsi_vals[i] < 70 and
            atr_vals[i] > atr_vals[i-5]):  # Volatility expansion
            signals[i] = 1
        elif (ema9[i] < ema20[i] and 
              ema9[i] < ema9[i-1] and 
              rsi_vals[i] > 30 and rsi_vals[i] < 70 and
              atr_vals[i] > atr_vals[i-5]):
            signals[i] = -1
    
    return signals


class VectorizedBacktestEngine:
    """High-performance vectorized backtesting engine."""
    
    def __init__(self, chunk_size: int = 50000):
        self.chunk_size = chunk_size
        self._indicator_cache = {}
    
    def run_vectorized(self, req: BacktestRequest, ticks_df: pd.DataFrame, 
                      candles: Dict[str, pd.DataFrame]) -> Dict:
        """Run backtest using vectorized operations."""
        
        # Pre-compute all indicators
        indicators = self._precompute_indicators(candles)
        
        # Convert to numpy arrays for speed
        tick_data = self._prepare_tick_arrays(ticks_df)
        
        # Process in chunks
        results = []
        for chunk_start in range(0, len(ticks_df), self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, len(ticks_df))
            chunk_result = self._process_chunk(
                tick_data, indicators, chunk_start, chunk_end, req
            )
            results.append(chunk_result)
        
        # Aggregate results
        return self._aggregate_results(results, req)
    
    def _precompute_indicators(self, candles: Dict[str, pd.DataFrame]) -> Dict:
        """Pre-compute all indicators for all timeframes."""
        indicators = {}
        
        for tf, df in candles.items():
            if df.empty or len(df) < 50:
                continue
                
            cache_key = f"{tf}_{len(df)}"
            if cache_key in self._indicator_cache:
                indicators[tf] = self._indicator_cache[cache_key]
                continue
            
            c = df['close'].values.astype(np.float64)
            h = df['high'].values.astype(np.float64)
            l = df['low'].values.astype(np.float64)
            
            tf_indicators = {
                'ema9': ema(c, 9),
                'ema20': ema(c, 20),
                'ema50': ema(c, 50) if len(c) >= 50 else ema(c, 20),
                'atr': atr(h, l, c, 14),
                'rsi': rsi(c, 14),
                'datetime': df['datetime'].values
            }
            
            indicators[tf] = tf_indicators
            self._indicator_cache[cache_key] = tf_indicators
        
        return indicators
    
    def _prepare_tick_arrays(self, ticks_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Convert tick data to numpy arrays."""
        return {
            'bid': ticks_df['bid'].values.astype(np.float64),
            'ask': ticks_df['ask'].values.astype(np.float64),
            'timestamp': ticks_df['datetime'].astype(np.int64) // 10**9,  # Convert to seconds
            'datetime': ticks_df['datetime'].values
        }
    
    def _process_chunk(self, tick_data: Dict, indicators: Dict, 
                      start_idx: int, end_idx: int, req: BacktestRequest) -> Dict:
        """Process a chunk of ticks vectorized."""
        
        # Extract chunk data
        chunk_bids = tick_data['bid'][start_idx:end_idx]
        chunk_asks = tick_data['ask'][start_idx:end_idx]
        chunk_timestamps = tick_data['timestamp'][start_idx:end_idx]
        
        # Calculate tick metrics
        velocity, momentum, direction = fast_tick_metrics(
            chunk_bids, chunk_asks, chunk_timestamps
        )
        
        # Get corresponding M1 indicators
        m1_indicators = indicators.get('M1', {})
        if not m1_indicators:
            return {'trades': [], 'equity_curve': []}
        
        # Align tick data with M1 candles
        aligned_signals = self._align_with_candles(
            tick_data['datetime'][start_idx:end_idx],
            m1_indicators,
            velocity, momentum, direction
        )
        
        # Simulate trades
        trades = self._simulate_trades_vectorized(
            chunk_bids, chunk_asks, aligned_signals, req
        )
        
        return {
            'trades': trades,
            'chunk_start': start_idx,
            'chunk_end': end_idx
        }
    
    def _align_with_candles(self, tick_times: np.ndarray, m1_indicators: Dict,
                           velocity: np.ndarray, momentum: np.ndarray, 
                           direction: np.ndarray) -> np.ndarray:
        """Align tick data with M1 candle indicators."""
        
        if not m1_indicators or 'datetime' not in m1_indicators:
            return np.zeros(len(tick_times), dtype=np.int8)
        
        candle_times = pd.to_datetime(m1_indicators['datetime'])
        
        # Use searchsorted for efficient alignment
        indices = np.searchsorted(candle_times, tick_times, side='right') - 1
        indices = np.clip(indices, 0, len(m1_indicators['ema9']) - 1)
        
        # Generate signals using vectorized logic
        signals = fast_signal_detection(
            m1_indicators['ema9'][indices],
            m1_indicators['ema9'][indices],
            m1_indicators['ema20'][indices],
            m1_indicators['rsi'][indices],
            m1_indicators['atr'][indices]
        )
        
        return signals
    
    def _simulate_trades_vectorized(self, bids: np.ndarray, asks: np.ndarray,
                                   signals: np.ndarray, req: BacktestRequest) -> List[Dict]:
        """Simulate trades using vectorized operations."""
        trades = []
        
        # Find signal points
        signal_indices = np.where(signals != 0)[0]
        
        for idx in signal_indices:
            if idx >= len(bids) - 1:
                continue
                
            signal_type = signals[idx]
            entry_price = asks[idx] if signal_type == 1 else bids[idx]
            
            # Simple exit logic - find next opposite signal or timeout
            exit_idx = self._find_exit_point(signals, idx, signal_type)
            if exit_idx is None or exit_idx >= len(bids):
                continue
            
            exit_price = bids[exit_idx] if signal_type == 1 else asks[exit_idx]
            
            # Calculate PnL
            if signal_type == 1:  # Buy
                pnl = (exit_price - entry_price) * 100000  # Assuming standard lot
            else:  # Sell
                pnl = (entry_price - exit_price) * 100000
            
            trades.append({
                'entry_idx': idx,
                'exit_idx': exit_idx,
                'signal_type': 'BUY' if signal_type == 1 else 'SELL',
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl,
                'won': pnl > 0
            })
        
        return trades
    
    def _find_exit_point(self, signals: np.ndarray, entry_idx: int, 
                        signal_type: int, max_hold: int = 100) -> Optional[int]:
        """Find exit point for a trade."""
        search_end = min(entry_idx + max_hold, len(signals))
        
        for i in range(entry_idx + 1, search_end):
            # Exit on opposite signal or timeout
            if signals[i] == -signal_type or i == search_end - 1:
                return i
        
        return None
    
    def _aggregate_results(self, chunk_results: List[Dict], req: BacktestRequest) -> Dict:
        """Aggregate results from all chunks."""
        all_trades = []
        
        for chunk in chunk_results:
            all_trades.extend(chunk['trades'])
        
        # Calculate summary statistics
        if not all_trades:
            return {
                'summary': {
                    'trades': 0,
                    'wins': 0,
                    'losses': 0,
                    'win_rate': 0.0,
                    'total_pnl': 0.0
                },
                'trades': [],
                'equity_curve': []
            }
        
        wins = sum(1 for t in all_trades if t['won'])
        losses = len(all_trades) - wins
        total_pnl = sum(t['pnl'] for t in all_trades)
        win_rate = wins / len(all_trades) if all_trades else 0.0
        
        return {
            'summary': {
                'trades': len(all_trades),
                'wins': wins,
                'losses': losses,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'initial_balance': req.initial_balance,
                'final_balance': req.initial_balance + total_pnl
            },
            'trades': all_trades,
            'equity_curve': self._build_equity_curve(all_trades, req.initial_balance)
        }
    
    def _build_equity_curve(self, trades: List[Dict], initial_balance: float) -> List[Dict]:
        """Build equity curve from trades."""
        equity_curve = [{'balance': initial_balance, 'trade_idx': 0}]
        
        running_balance = initial_balance
        for i, trade in enumerate(trades):
            running_balance += trade['pnl']
            equity_curve.append({
                'balance': running_balance,
                'trade_idx': i + 1
            })
        
        return equity_curve


class ParallelVectorizedEngine:
    """Parallel processing wrapper for vectorized engine."""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or min(8, (os.cpu_count() or 1))
        self.engines = [VectorizedBacktestEngine() for _ in range(self.max_workers)]
    
    def run_parallel(self, req: BacktestRequest, ticks_df: pd.DataFrame,
                    candles: Dict[str, pd.DataFrame]) -> Dict:
        """Run backtest using parallel vectorized processing."""
        
        # Split data into chunks for parallel processing
        chunk_size = len(ticks_df) // self.max_workers
        chunks = []
        
        for i in range(self.max_workers):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size if i < self.max_workers - 1 else len(ticks_df)
            
            if start_idx < len(ticks_df):
                chunks.append((start_idx, end_idx))
        
        # Process chunks in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            
            for i, (start_idx, end_idx) in enumerate(chunks):
                chunk_ticks = ticks_df.iloc[start_idx:end_idx].copy()
                future = executor.submit(
                    self.engines[i].run_vectorized,
                    req, chunk_ticks, candles
                )
                futures.append(future)
            
            # Collect results
            results = [future.result() for future in futures]
        
        # Merge results
        return self._merge_parallel_results(results, req)
    
    def _merge_parallel_results(self, results: List[Dict], req: BacktestRequest) -> Dict:
        """Merge results from parallel processing."""
        all_trades = []
        total_pnl = 0.0
        
        for result in results:
            all_trades.extend(result['trades'])
            total_pnl += result['summary']['total_pnl']
        
        # Sort trades by entry time
        all_trades.sort(key=lambda x: x['entry_idx'])
        
        wins = sum(1 for t in all_trades if t['won'])
        losses = len(all_trades) - wins
        win_rate = wins / len(all_trades) if all_trades else 0.0
        
        return {
            'summary': {
                'trades': len(all_trades),
                'wins': wins,
                'losses': losses,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'initial_balance': req.initial_balance,
                'final_balance': req.initial_balance + total_pnl
            },
            'trades': all_trades,
            'equity_curve': self._build_merged_equity_curve(all_trades, req.initial_balance)
        }
    
    def _build_merged_equity_curve(self, trades: List[Dict], initial_balance: float) -> List[Dict]:
        """Build equity curve from merged trades."""
        equity_curve = [{'balance': initial_balance, 'trade_idx': 0}]
        
        running_balance = initial_balance
        for i, trade in enumerate(trades):
            running_balance += trade['pnl']
            equity_curve.append({
                'balance': running_balance,
                'trade_idx': i + 1
            })
        
        return equity_curve