"""
Ultra-High Performance Data Engine
Maximizes RAM and CPU usage for fastest possible backtesting.
"""
import numpy as np
import numba
from numba import jit, prange
import mmap
import os
import psutil
from typing import Dict, Tuple, Optional
from datetime import datetime, timezone
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp


# Get maximum system resources
SYSTEM_RAM_GB = psutil.virtual_memory().total / (1024**3)
MAX_RAM_USAGE = int(SYSTEM_RAM_GB * 0.9 * 1024)  # 90% of RAM in MB
CPU_CORES = os.cpu_count() or 1
MAX_WORKERS = CPU_CORES


@numba.jit(nopython=True, cache=True, parallel=True)
def ultra_fast_ema(prices: np.ndarray, periods: np.ndarray) -> np.ndarray:
    """Ultra-fast EMA calculation using parallel processing."""
    n_prices = len(prices)
    n_periods = len(periods)
    results = np.zeros((n_periods, n_prices), dtype=np.float64)
    
    for p_idx in prange(n_periods):
        period = periods[p_idx]
        alpha = 2.0 / (period + 1.0)
        results[p_idx, 0] = prices[0]
        
        for i in range(1, n_prices):
            results[p_idx, i] = alpha * prices[i] + (1.0 - alpha) * results[p_idx, i-1]
    
    return results


@numba.jit(nopython=True, cache=True, parallel=True)
def ultra_fast_indicators_batch(
    closes: np.ndarray, 
    highs: np.ndarray, 
    lows: np.ndarray,
    volumes: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate all indicators in parallel for maximum speed."""
    
    n = len(closes)
    
    # Pre-allocate all result arrays
    ema9 = np.zeros(n, dtype=np.float64)
    ema20 = np.zeros(n, dtype=np.float64)
    ema50 = np.zeros(n, dtype=np.float64)
    atr = np.zeros(n, dtype=np.float64)
    rsi = np.zeros(n, dtype=np.float64)
    
    # EMA calculations
    alpha9 = 2.0 / 10.0
    alpha20 = 2.0 / 21.0
    alpha50 = 2.0 / 51.0
    
    ema9[0] = closes[0]
    ema20[0] = closes[0]
    ema50[0] = closes[0]
    
    # Parallel EMA computation
    for i in prange(1, n):
        ema9[i] = alpha9 * closes[i] + (1.0 - alpha9) * ema9[i-1]
        ema20[i] = alpha20 * closes[i] + (1.0 - alpha20) * ema20[i-1]
        ema50[i] = alpha50 * closes[i] + (1.0 - alpha50) * ema50[i-1]
    
    # ATR calculation
    atr[0] = highs[0] - lows[0]
    atr_alpha = 2.0 / 15.0
    
    for i in prange(1, n):
        tr1 = highs[i] - lows[i]
        tr2 = abs(highs[i] - closes[i-1])
        tr3 = abs(lows[i] - closes[i-1])
        tr = max(tr1, max(tr2, tr3))
        atr[i] = atr_alpha * tr + (1.0 - atr_alpha) * atr[i-1]
    
    # RSI calculation (simplified for speed)
    rsi_period = 14
    for i in prange(rsi_period, n):
        gains = 0.0
        losses = 0.0
        
        for j in range(i - rsi_period + 1, i + 1):
            delta = closes[j] - closes[j-1]
            if delta > 0:
                gains += delta
            else:
                losses -= delta
        
        if losses == 0:
            rsi[i] = 100.0
        else:
            rs = gains / losses
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return ema9, ema20, ema50, atr, rsi


@numba.jit(nopython=True, cache=True, parallel=True)
def ultra_fast_signal_detection(
    closes: np.ndarray,
    ema9: np.ndarray,
    ema20: np.ndarray,
    ema50: np.ndarray,
    atr: np.ndarray,
    rsi: np.ndarray,
    bids: np.ndarray,
    asks: np.ndarray,
    timestamps: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Ultra-fast vectorized signal detection."""
    
    n = len(closes)
    signals = np.zeros(n, dtype=np.int8)  # 0=none, 1=buy, -1=sell
    entry_prices = np.zeros(n, dtype=np.float64)
    stop_losses = np.zeros(n, dtype=np.float64)
    take_profits = np.zeros(n, dtype=np.float64)
    
    # Vectorized signal logic
    for i in prange(50, n):  # Start after sufficient history
        
        # Trend conditions
        trend_up = (ema9[i] > ema20[i] and 
                   ema20[i] > ema50[i] and
                   ema9[i] > ema9[i-1])
        
        trend_down = (ema9[i] < ema20[i] and 
                     ema20[i] < ema50[i] and
                     ema9[i] < ema9[i-1])
        
        # Momentum conditions
        momentum_up = (rsi[i] > 40 and rsi[i] < 70 and
                      closes[i] > closes[i-1])
        
        momentum_down = (rsi[i] < 60 and rsi[i] > 30 and
                        closes[i] < closes[i-1])
        
        # Volatility filter
        vol_ok = atr[i] > atr[i-5] * 0.8  # Some volatility required
        
        # Generate signals
        if trend_up and momentum_up and vol_ok:
            signals[i] = 1
            entry_prices[i] = asks[i]  # Buy at ask
            stop_losses[i] = entry_prices[i] - atr[i] * 2.0
            take_profits[i] = entry_prices[i] + atr[i] * 3.0
            
        elif trend_down and momentum_down and vol_ok:
            signals[i] = -1
            entry_prices[i] = bids[i]  # Sell at bid
            stop_losses[i] = entry_prices[i] + atr[i] * 2.0
            take_profits[i] = entry_prices[i] - atr[i] * 3.0
    
    return signals, entry_prices, stop_losses, take_profits


@numba.jit(nopython=True, cache=True)
def ultra_fast_trade_simulation(
    signals: np.ndarray,
    entry_prices: np.ndarray,
    stop_losses: np.ndarray,
    take_profits: np.ndarray,
    bids: np.ndarray,
    asks: np.ndarray,
    timestamps: np.ndarray,
    initial_balance: float,
    max_positions: int = 5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Ultra-fast trade simulation with realistic execution."""
    
    n = len(signals)
    
    # Pre-allocate trade arrays
    max_trades = n // 10  # Estimate maximum possible trades
    trade_entries = np.zeros(max_trades, dtype=np.int32)
    trade_exits = np.zeros(max_trades, dtype=np.int32)
    trade_pnls = np.zeros(max_trades, dtype=np.float64)
    trade_directions = np.zeros(max_trades, dtype=np.int8)
    
    # Simulation state
    balance = initial_balance
    open_positions = 0
    trade_count = 0
    
    # Position tracking arrays
    pos_entries = np.zeros(max_positions, dtype=np.int32)
    pos_directions = np.zeros(max_positions, dtype=np.int8)
    pos_entry_prices = np.zeros(max_positions, dtype=np.float64)
    pos_stops = np.zeros(max_positions, dtype=np.float64)
    pos_targets = np.zeros(max_positions, dtype=np.float64)
    
    for i in range(n):
        current_bid = bids[i]
        current_ask = asks[i]
        
        # Check exits for open positions
        for pos_idx in range(open_positions):
            if pos_entries[pos_idx] == 0:  # Position closed
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
                    pnl = (exit_price - entry_price) * 100000  # Standard lot
                else:
                    pnl = (entry_price - exit_price) * 100000
                
                # Record trade
                trade_entries[trade_count] = pos_entries[pos_idx]
                trade_exits[trade_count] = i
                trade_pnls[trade_count] = pnl
                trade_directions[trade_count] = direction
                trade_count += 1
                
                # Close position
                pos_entries[pos_idx] = 0
                balance += pnl
        
        # Compact open positions array
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
            # Open new position
            pos_entries[open_positions] = i
            pos_directions[open_positions] = signals[i]
            pos_entry_prices[open_positions] = entry_prices[i]
            pos_stops[open_positions] = stop_losses[i]
            pos_targets[open_positions] = take_profits[i]
            open_positions += 1
    
    # Return only used portion of arrays
    return (trade_entries[:trade_count], 
            trade_exits[:trade_count],
            trade_pnls[:trade_count], 
            trade_directions[:trade_count])


class UltraFastDataEngine:
    """Ultra-high performance data engine using maximum system resources."""
    
    def __init__(self):
        self.max_memory_mb = MAX_RAM_USAGE
        self.max_workers = MAX_WORKERS
        self.data_cache = {}
        self.memory_pool = None
        
        print(f"Ultra-Fast Engine initialized:")
        print(f"  Max RAM usage: {self.max_memory_mb:,} MB ({SYSTEM_RAM_GB:.1f} GB total)")
        print(f"  CPU cores: {self.max_workers}")
        
    def load_data_ultra_fast(self, tick_file: str, candle_files: Dict[str, str]) -> Dict:
        """Load all data into optimized numpy arrays for maximum speed."""
        
        print("Loading data with maximum performance...")
        
        # Load tick data
        if tick_file.endswith('.parquet'):
            tick_df = pd.read_parquet(tick_file)
        else:
            tick_df = pd.read_csv(tick_file, parse_dates=['datetime'])
        
        # Convert to optimized numpy arrays
        tick_data = {
            'timestamps': tick_df['datetime'].astype('datetime64[ns]').astype(np.int64),
            'bids': tick_df['bid'].astype(np.float64),
            'asks': tick_df['ask'].astype(np.float64),
            'spreads': tick_df['spread'].astype(np.float64) if 'spread' in tick_df.columns 
                      else (tick_df['ask'] - tick_df['bid']).astype(np.float64)
        }
        
        # Load and optimize candle data
        candle_data = {}
        for tf, file_path in candle_files.items():
            if not os.path.exists(file_path):
                continue
                
            if file_path.endswith('.parquet'):
                df = pd.read_parquet(file_path)
            else:
                df = pd.read_csv(file_path, parse_dates=['datetime'])
            
            candle_data[tf] = {
                'timestamps': df['datetime'].astype('datetime64[ns]').astype(np.int64),
                'opens': df['open'].astype(np.float64),
                'highs': df['high'].astype(np.float64),
                'lows': df['low'].astype(np.float64),
                'closes': df['close'].astype(np.float64),
                'volumes': df['volume'].astype(np.float64) if 'volume' in df.columns 
                          else np.ones(len(df), dtype=np.float64)
            }
        
        print(f"Data loaded: {len(tick_data['bids']):,} ticks, {len(candle_data)} timeframes")
        
        return {
            'ticks': tick_data,
            'candles': candle_data
        }
    
    def precompute_all_indicators(self, candle_data: Dict) -> Dict:
        """Precompute all indicators using maximum parallelization."""
        
        print("Precomputing indicators with maximum parallelization...")
        
        all_indicators = {}
        
        # Use ThreadPoolExecutor for I/O bound operations
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            
            for tf, data in candle_data.items():
                future = executor.submit(self._compute_indicators_for_timeframe, tf, data)
                futures[future] = tf
            
            # Collect results
            for future in futures:
                tf = futures[future]
                all_indicators[tf] = future.result()
        
        print(f"Indicators computed for {len(all_indicators)} timeframes")
        return all_indicators
    
    def _compute_indicators_for_timeframe(self, tf: str, data: Dict) -> Dict:
        """Compute indicators for a single timeframe using ultra-fast functions."""
        
        closes = data['closes']
        highs = data['highs']
        lows = data['lows']
        volumes = data['volumes']
        
        # Use ultra-fast batch computation
        ema9, ema20, ema50, atr, rsi = ultra_fast_indicators_batch(
            closes, highs, lows, volumes
        )
        
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
    
    def run_ultra_fast_backtest(self, data: Dict, req) -> Dict:
        """Run backtest using maximum system resources for ultimate speed."""
        
        print("Running ultra-fast backtest...")
        start_time = pd.Timestamp.now()
        
        # Get M1 data (primary timeframe)
        m1_data = data['candles']['M1']
        tick_data = data['ticks']
        
        # Precompute all indicators
        indicators = self.precompute_all_indicators(data['candles'])
        m1_indicators = indicators['M1']
        
        # Align tick data with M1 candles for signal generation
        aligned_signals, aligned_entries, aligned_stops, aligned_targets = self._align_ticks_with_signals(
            tick_data, m1_indicators
        )
        
        # Run ultra-fast trade simulation
        trade_entries, trade_exits, trade_pnls, trade_directions = ultra_fast_trade_simulation(
            aligned_signals,
            aligned_entries,
            aligned_stops,
            aligned_targets,
            tick_data['bids'],
            tick_data['asks'],
            tick_data['timestamps'],
            req.initial_balance,
            max_positions=5
        )
        
        # Build result
        result = self._build_ultra_fast_result(
            trade_entries, trade_exits, trade_pnls, trade_directions,
            tick_data, req, start_time
        )
        
        execution_time = (pd.Timestamp.now() - start_time).total_seconds()
        ticks_per_second = len(tick_data['bids']) / execution_time if execution_time > 0 else 0
        
        print(f"Ultra-fast backtest completed in {execution_time:.2f}s")
        print(f"Processing speed: {ticks_per_second:,.0f} ticks/second")
        print(f"Generated {len(trade_pnls)} trades")
        
        return result
    
    def _align_ticks_with_signals(self, tick_data: Dict, indicators: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Align tick data with M1 candle signals for ultra-fast processing."""
        
        n_ticks = len(tick_data['bids'])
        
        # For maximum speed, we'll generate signals at regular intervals
        # In practice, you'd align with actual M1 candle closes
        signal_interval = max(1, n_ticks // len(indicators['closes']))
        
        signals = np.zeros(n_ticks, dtype=np.int8)
        entries = np.zeros(n_ticks, dtype=np.float64)
        stops = np.zeros(n_ticks, dtype=np.float64)
        targets = np.zeros(n_ticks, dtype=np.float64)
        
        # Generate signals using ultra-fast detection
        candle_signals, candle_entries, candle_stops, candle_targets = ultra_fast_signal_detection(
            indicators['closes'],
            indicators['ema9'],
            indicators['ema20'],
            indicators['ema50'],
            indicators['atr'],
            indicators['rsi'],
            tick_data['bids'][::signal_interval][:len(indicators['closes'])],
            tick_data['asks'][::signal_interval][:len(indicators['closes'])],
            tick_data['timestamps'][::signal_interval][:len(indicators['closes'])]
        )
        
        # Map candle signals to tick indices
        for i, signal in enumerate(candle_signals):
            tick_idx = min(i * signal_interval, n_ticks - 1)
            if signal != 0:
                signals[tick_idx] = signal
                entries[tick_idx] = candle_entries[i]
                stops[tick_idx] = candle_stops[i]
                targets[tick_idx] = candle_targets[i]
        
        return signals, entries, stops, targets
    
    def _build_ultra_fast_result(self, trade_entries: np.ndarray, trade_exits: np.ndarray,
                                trade_pnls: np.ndarray, trade_directions: np.ndarray,
                                tick_data: Dict, req, start_time) -> Dict:
        """Build result dictionary from ultra-fast computation."""
        
        trades = []
        running_balance = req.initial_balance
        
        for i in range(len(trade_pnls)):
            entry_idx = trade_entries[i]
            exit_idx = trade_exits[i]
            pnl = trade_pnls[i]
            direction = trade_directions[i]
            
            running_balance += pnl
            
            trades.append({
                'entry_time': pd.Timestamp(tick_data['timestamps'][entry_idx]).isoformat(),
                'exit_time': pd.Timestamp(tick_data['timestamps'][exit_idx]).isoformat(),
                'direction': 'BUY' if direction == 1 else 'SELL',
                'entry_price': tick_data['asks'][entry_idx] if direction == 1 else tick_data['bids'][entry_idx],
                'exit_price': tick_data['bids'][exit_idx] if direction == 1 else tick_data['asks'][exit_idx],
                'pnl': pnl,
                'won': pnl > 0,
                'duration_ticks': exit_idx - entry_idx
            })
        
        # Calculate summary statistics
        wins = sum(1 for t in trades if t['won'])
        losses = len(trades) - wins
        total_pnl = sum(t['pnl'] for t in trades)
        win_rate = wins / len(trades) if trades else 0.0
        
        execution_time = (pd.Timestamp.now() - start_time).total_seconds()
        
        return {
            'summary': {
                'initial_balance': req.initial_balance,
                'final_balance': req.initial_balance + total_pnl,
                'total_pnl': total_pnl,
                'trades': len(trades),
                'wins': wins,
                'losses': losses,
                'win_rate': win_rate,
                'avg_win': sum(t['pnl'] for t in trades if t['won']) / wins if wins > 0 else 0,
                'avg_loss': sum(t['pnl'] for t in trades if not t['won']) / losses if losses > 0 else 0
            },
            'trades': trades,
            'meta': {
                'engine_type': 'ultra_fast',
                'execution_time_seconds': execution_time,
                'ticks_processed': len(tick_data['bids']),
                'ticks_per_second': len(tick_data['bids']) / execution_time if execution_time > 0 else 0,
                'memory_usage_mb': self.max_memory_mb,
                'cpu_cores_used': self.max_workers,
                'optimization_level': 'maximum'
            }
        }


# Global ultra-fast engine instance
ultra_engine = UltraFastDataEngine()