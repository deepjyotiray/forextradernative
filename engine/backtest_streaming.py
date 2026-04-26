"""
Memory-efficient streaming data processor for large backtests.
"""
import os
import mmap
import struct
from typing import Iterator, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timezone


class TickDataStreamer:
    """Memory-efficient tick data streaming."""
    
    def __init__(self, chunk_size: int = 10000):
        self.chunk_size = chunk_size
        self._file_handle = None
        self._mmap = None
    
    def stream_from_file(self, file_path: str) -> Iterator[pd.DataFrame]:
        """Stream tick data from file in chunks."""
        
        if file_path.endswith('.pkl'):
            yield from self._stream_pickle_chunks(file_path)
        elif file_path.endswith('.csv'):
            yield from self._stream_csv_chunks(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
    
    def _stream_pickle_chunks(self, file_path: str) -> Iterator[pd.DataFrame]:
        """Stream pickle file in chunks."""
        # Load full pickle file (optimize this for very large files)
        df = pd.read_pickle(file_path)
        
        for start_idx in range(0, len(df), self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, len(df))
            yield df.iloc[start_idx:end_idx].copy()
    
    def _stream_csv_chunks(self, file_path: str) -> Iterator[pd.DataFrame]:
        """Stream CSV file in chunks."""
        chunk_iter = pd.read_csv(
            file_path,
            chunksize=self.chunk_size,
            parse_dates=['datetime'],
            dtype={'bid': 'float64', 'ask': 'float64', 'spread': 'float64'}
        )
        
        for chunk in chunk_iter:
            yield chunk


class BinaryTickStreamer:
    """Ultra-fast binary tick data streaming."""
    
    TICK_STRUCT = struct.Struct('<Qddd')  # timestamp, bid, ask, spread
    TICK_SIZE = TICK_STRUCT.size
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self._file = None
        self._mmap = None
        self._size = 0
    
    def __enter__(self):
        self._file = open(self.file_path, 'rb')
        self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        self._size = len(self._mmap) // self.TICK_SIZE
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._mmap:
            self._mmap.close()
        if self._file:
            self._file.close()
    
    def stream_chunks(self, chunk_size: int = 50000) -> Iterator[np.ndarray]:
        """Stream binary tick data as numpy arrays."""
        
        for start_idx in range(0, self._size, chunk_size):
            end_idx = min(start_idx + chunk_size, self._size)
            chunk_bytes = end_idx - start_idx
            
            # Read binary data
            offset = start_idx * self.TICK_SIZE
            data = self._mmap[offset:offset + chunk_bytes * self.TICK_SIZE]
            
            # Convert to structured numpy array
            ticks = np.frombuffer(data, dtype=[
                ('timestamp', 'u8'),
                ('bid', 'f8'),
                ('ask', 'f8'),
                ('spread', 'f8')
            ])
            
            yield ticks
    
    @staticmethod
    def convert_csv_to_binary(csv_path: str, binary_path: str):
        """Convert CSV tick data to binary format for faster streaming."""
        
        with open(binary_path, 'wb') as f:
            for chunk in pd.read_csv(csv_path, chunksize=10000, parse_dates=['datetime']):
                for _, row in chunk.iterrows():
                    timestamp = int(row['datetime'].timestamp() * 1000000)  # microseconds
                    tick_data = BinaryTickStreamer.TICK_STRUCT.pack(
                        timestamp, row['bid'], row['ask'], row['spread']
                    )
                    f.write(tick_data)


class StreamingBacktestEngine:
    """Streaming backtest engine for memory efficiency."""
    
    def __init__(self, max_memory_mb: int = 1000):
        self.max_memory_mb = max_memory_mb
        self.chunk_size = self._calculate_chunk_size()
        self._indicator_buffer = {}
        self._trade_buffer = []
    
    def _calculate_chunk_size(self) -> int:
        """Calculate optimal chunk size based on memory limit."""
        # Estimate memory per tick (8 bytes * 4 fields = 32 bytes)
        bytes_per_tick = 32
        max_bytes = self.max_memory_mb * 1024 * 1024
        return max_bytes // bytes_per_tick
    
    def run_streaming(self, tick_file: str, candle_files: Dict[str, str],
                     req) -> Dict:
        """Run backtest using streaming data processing."""
        
        # Pre-load candle data (usually much smaller than ticks)
        candles = self._load_candle_data(candle_files)
        
        # Initialize streaming
        streamer = TickDataStreamer(chunk_size=self.chunk_size)
        
        # Process tick chunks
        total_trades = []
        equity_points = []
        running_balance = req.initial_balance
        
        for chunk_idx, tick_chunk in enumerate(streamer.stream_from_file(tick_file)):
            
            # Process chunk
            chunk_trades = self._process_tick_chunk(
                tick_chunk, candles, running_balance, req
            )
            
            # Update running balance
            for trade in chunk_trades:
                running_balance += trade['pnl']
                equity_points.append({
                    'time': trade['close_time'],
                    'balance': running_balance,
                    'equity': running_balance
                })
            
            total_trades.extend(chunk_trades)
            
            # Memory management - clear old data
            if chunk_idx % 10 == 0:
                self._cleanup_buffers()
        
        return self._build_final_result(total_trades, equity_points, req)
    
    def _load_candle_data(self, candle_files: Dict[str, str]) -> Dict[str, pd.DataFrame]:
        """Load candle data (smaller datasets)."""
        candles = {}
        
        for tf, file_path in candle_files.items():
            if os.path.exists(file_path):
                candles[tf] = pd.read_pickle(file_path)
        
        return candles
    
    def _process_tick_chunk(self, tick_chunk: pd.DataFrame, candles: Dict,
                           current_balance: float, req) -> List[Dict]:
        """Process a single chunk of tick data."""
        
        if tick_chunk.empty:
            return []
        
        # Get relevant candle data for this time period
        start_time = tick_chunk['datetime'].iloc[0]
        end_time = tick_chunk['datetime'].iloc[-1]
        
        chunk_candles = self._filter_candles_by_time(candles, start_time, end_time)
        
        # Run mini-backtest on chunk
        from .backtest_vectorized import VectorizedBacktestEngine
        
        engine = VectorizedBacktestEngine(chunk_size=len(tick_chunk))
        result = engine.run_vectorized(req, tick_chunk, chunk_candles)
        
        return result.get('trades', [])
    
    def _filter_candles_by_time(self, candles: Dict, start_time: datetime,
                               end_time: datetime) -> Dict[str, pd.DataFrame]:
        """Filter candle data to relevant time period."""
        
        filtered = {}
        
        for tf, df in candles.items():
            if df.empty:
                filtered[tf] = df
                continue
            
            # Add buffer for indicators
            buffer_start = start_time - pd.Timedelta(hours=24)
            
            mask = (df['datetime'] >= buffer_start) & (df['datetime'] <= end_time)
            filtered[tf] = df.loc[mask].copy()
        
        return filtered
    
    def _cleanup_buffers(self):
        """Clean up memory buffers."""
        # Keep only recent indicator data
        max_buffer_size = 1000
        
        for key in list(self._indicator_buffer.keys()):
            if len(self._indicator_buffer[key]) > max_buffer_size:
                # Keep only recent half
                keep_size = max_buffer_size // 2
                self._indicator_buffer[key] = self._indicator_buffer[key][-keep_size:]
    
    def _build_final_result(self, trades: List[Dict], equity_curve: List[Dict],
                           req) -> Dict:
        """Build final backtest result."""
        
        if not trades:
            return {
                'summary': {
                    'trades': 0,
                    'wins': 0,
                    'losses': 0,
                    'win_rate': 0.0,
                    'total_pnl': 0.0,
                    'initial_balance': req.initial_balance,
                    'final_balance': req.initial_balance
                },
                'trades': [],
                'equity_curve': []
            }
        
        wins = sum(1 for t in trades if t.get('won', False))
        losses = len(trades) - wins
        total_pnl = sum(t.get('pnl', 0) for t in trades)
        win_rate = wins / len(trades)
        
        return {
            'summary': {
                'trades': len(trades),
                'wins': wins,
                'losses': losses,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'initial_balance': req.initial_balance,
                'final_balance': req.initial_balance + total_pnl
            },
            'trades': trades,
            'equity_curve': equity_curve
        }


class CompressedDataManager:
    """Manage compressed data storage for faster I/O."""
    
    def __init__(self, compression: str = 'lz4'):
        self.compression = compression
    
    def save_compressed(self, data: pd.DataFrame, file_path: str):
        """Save DataFrame with compression."""
        
        if self.compression == 'lz4':
            data.to_pickle(file_path, compression='lz4')
        elif self.compression == 'gzip':
            data.to_pickle(file_path, compression='gzip')
        else:
            data.to_pickle(file_path)
    
    def load_compressed(self, file_path: str) -> pd.DataFrame:
        """Load compressed DataFrame."""
        return pd.read_pickle(file_path)
    
    def convert_to_parquet(self, pickle_file: str, parquet_file: str):
        """Convert pickle to parquet for better performance."""
        df = pd.read_pickle(pickle_file)
        df.to_parquet(parquet_file, compression='snappy', index=False)
    
    def stream_parquet(self, file_path: str, chunk_size: int = 50000) -> Iterator[pd.DataFrame]:
        """Stream parquet file efficiently."""
        import pyarrow.parquet as pq
        
        parquet_file = pq.ParquetFile(file_path)
        
        for batch in parquet_file.iter_batches(batch_size=chunk_size):
            yield batch.to_pandas()


# Usage example integration
def create_optimized_backtest_runner(req, use_streaming: bool = True,
                                   use_vectorized: bool = True) -> Dict:
    """Factory function to create optimized backtest runner."""
    
    if use_streaming and use_vectorized:
        # Best performance for large datasets
        engine = StreamingBacktestEngine(max_memory_mb=2000)
        return engine.run_streaming(
            tick_file=f"data/{req.symbol}_ticks.pkl",
            candle_files={
                'M1': f"data/{req.symbol}_M1.pkl",
                'M5': f"data/{req.symbol}_M5.pkl",
                'H1': f"data/{req.symbol}_H1.pkl"
            },
            req=req
        )
    
    elif use_vectorized:
        # Good performance for medium datasets
        from .backtest_vectorized import ParallelVectorizedEngine
        
        engine = ParallelVectorizedEngine()
        # Load data (implement data loading here)
        ticks_df = pd.read_pickle(f"data/{req.symbol}_ticks.pkl")
        candles = {
            'M1': pd.read_pickle(f"data/{req.symbol}_M1.pkl"),
            'M5': pd.read_pickle(f"data/{req.symbol}_M5.pkl")
        }
        
        return engine.run_parallel(req, ticks_df, candles)
    
    else:
        # Fallback to original engine
        from .backtest_sim import BacktestRunner
        from .backtest_data import BacktestDataProvider
        
        provider = BacktestDataProvider()
        runner = BacktestRunner()
        
        dataset = provider.load_dataset(
            symbol=req.symbol,
            start_utc=req.start_utc,
            end_utc=req.end_utc,
            cache_only=req.cache_only
        )
        
        return runner.run(req, dataset)