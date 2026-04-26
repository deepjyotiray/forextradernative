"""
Memory-Mapped Ultra-Fast Data Access System
Zero-copy operations for maximum performance.
"""
import numpy as np
import mmap
import os
import struct
from typing import Dict, Tuple, Optional
import psutil
from concurrent.futures import ThreadPoolExecutor
import numba
from numba import jit, prange


# Memory layout constants for binary data format
TICK_STRUCT_FORMAT = '<Qddd'  # timestamp(uint64), bid(float64), ask(float64), spread(float64)
TICK_SIZE = struct.calcsize(TICK_STRUCT_FORMAT)
CANDLE_STRUCT_FORMAT = '<Qdddddd'  # timestamp, open, high, low, close, volume, count
CANDLE_SIZE = struct.calcsize(CANDLE_STRUCT_FORMAT)

# System memory info
TOTAL_RAM_GB = psutil.virtual_memory().total / (1024**3)
MAX_MMAP_SIZE = int(TOTAL_RAM_GB * 0.8 * 1024 * 1024 * 1024)  # 80% of RAM in bytes


@jit(nopython=True, cache=True)
def fast_binary_search(timestamps: np.ndarray, target: np.int64) -> np.int32:
    """Ultra-fast binary search for timestamp alignment."""
    left = 0
    right = len(timestamps) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if timestamps[mid] == target:
            return mid
        elif timestamps[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return right  # Return closest lower index


@jit(nopython=True, cache=True, parallel=True)
def align_tick_to_candle_indices(tick_timestamps: np.ndarray, 
                                candle_timestamps: np.ndarray) -> np.ndarray:
    """Ultra-fast alignment of tick data to candle indices."""
    n_ticks = len(tick_timestamps)
    candle_indices = np.zeros(n_ticks, dtype=np.int32)
    
    for i in prange(n_ticks):
        candle_indices[i] = fast_binary_search(candle_timestamps, tick_timestamps[i])
    
    return candle_indices


class MemoryMappedDataManager:
    """Ultra-high performance memory-mapped data manager."""
    
    def __init__(self, cache_dir: str = "ultra_cache"):
        self.cache_dir = cache_dir
        self.mmap_files = {}
        self.data_arrays = {}
        self.total_mapped_size = 0
        
        os.makedirs(cache_dir, exist_ok=True)
        
        print(f"Memory-mapped manager initialized:")
        print(f"  Cache directory: {cache_dir}")
        print(f"  Max mmap size: {MAX_MMAP_SIZE / (1024**3):.1f} GB")
    
    def convert_to_binary_format(self, symbol: str, tick_df, candle_dfs: Dict) -> Dict[str, str]:
        """Convert data to ultra-fast binary format."""
        
        print(f"Converting {symbol} data to binary format...")
        
        file_paths = {}
        
        # Convert tick data
        tick_file = os.path.join(self.cache_dir, f"{symbol}_ticks.bin")
        self._write_tick_binary(tick_df, tick_file)
        file_paths['ticks'] = tick_file
        
        # Convert candle data
        file_paths['candles'] = {}
        for tf, df in candle_dfs.items():
            candle_file = os.path.join(self.cache_dir, f"{symbol}_{tf}.bin")
            self._write_candle_binary(df, candle_file)
            file_paths['candles'][tf] = candle_file
        
        print(f"Binary conversion completed for {symbol}")
        return file_paths
    
    def _write_tick_binary(self, df, file_path: str):
        """Write tick data in optimized binary format."""
        
        with open(file_path, 'wb') as f:
            for _, row in df.iterrows():
                timestamp = int(row['datetime'].timestamp() * 1_000_000)  # microseconds
                bid = float(row['bid'])
                ask = float(row['ask'])
                spread = float(row.get('spread', ask - bid))
                
                data = struct.pack(TICK_STRUCT_FORMAT, timestamp, bid, ask, spread)
                f.write(data)
    
    def _write_candle_binary(self, df, file_path: str):
        """Write candle data in optimized binary format."""
        
        with open(file_path, 'wb') as f:
            for _, row in df.iterrows():
                timestamp = int(row['datetime'].timestamp() * 1_000_000)  # microseconds
                open_price = float(row['open'])
                high = float(row['high'])
                low = float(row['low'])
                close = float(row['close'])
                volume = float(row.get('volume', 1.0))
                count = float(row.get('tick_count', 1.0))
                
                data = struct.pack(CANDLE_STRUCT_FORMAT, timestamp, open_price, high, low, close, volume, count)
                f.write(data)
    
    def load_memory_mapped_data(self, file_paths: Dict) -> Dict:
        """Load data using memory mapping for zero-copy access."""
        
        print("Loading data with memory mapping...")
        
        data = {}
        
        # Load tick data
        tick_file = file_paths['ticks']
        tick_data = self._load_mmap_ticks(tick_file)
        data['ticks'] = tick_data
        
        # Load candle data
        data['candles'] = {}
        for tf, candle_file in file_paths['candles'].items():
            candle_data = self._load_mmap_candles(candle_file)
            data['candles'][tf] = candle_data
        
        print(f"Memory-mapped data loaded:")
        print(f"  Ticks: {len(tick_data['timestamps']):,}")
        print(f"  Candles: {sum(len(cd['timestamps']) for cd in data['candles'].values()):,}")
        print(f"  Total mapped: {self.total_mapped_size / (1024**2):.1f} MB")
        
        return data
    
    def _load_mmap_ticks(self, file_path: str) -> Dict[str, np.ndarray]:
        """Load tick data using memory mapping."""
        
        file_size = os.path.getsize(file_path)
        n_ticks = file_size // TICK_SIZE
        
        # Open file and create memory map
        f = open(file_path, 'rb')
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        
        # Create numpy arrays that directly reference the memory-mapped data
        dtype = np.dtype([
            ('timestamp', 'u8'),
            ('bid', 'f8'),
            ('ask', 'f8'),
            ('spread', 'f8')
        ])
        
        # Create structured array from memory map
        raw_data = np.frombuffer(mm, dtype=dtype, count=n_ticks)
        
        # Extract individual arrays (zero-copy views)
        tick_data = {
            'timestamps': raw_data['timestamp'].copy(),  # Copy to ensure we own the data
            'bids': raw_data['bid'].copy(),
            'asks': raw_data['ask'].copy(),
            'spreads': raw_data['spread'].copy()
        }
        
        # Store references to keep memory map alive
        self.mmap_files[file_path] = (f, mm)
        self.total_mapped_size += file_size
        
        return tick_data
    
    def _load_mmap_candles(self, file_path: str) -> Dict[str, np.ndarray]:
        """Load candle data using memory mapping."""
        
        file_size = os.path.getsize(file_path)
        n_candles = file_size // CANDLE_SIZE
        
        # Open file and create memory map
        f = open(file_path, 'rb')
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        
        # Create numpy arrays that directly reference the memory-mapped data
        dtype = np.dtype([
            ('timestamp', 'u8'),
            ('open', 'f8'),
            ('high', 'f8'),
            ('low', 'f8'),
            ('close', 'f8'),
            ('volume', 'f8'),
            ('count', 'f8')
        ])
        
        # Create structured array from memory map
        raw_data = np.frombuffer(mm, dtype=dtype, count=n_candles)
        
        # Extract individual arrays (zero-copy views)
        candle_data = {
            'timestamps': raw_data['timestamp'].copy(),
            'opens': raw_data['open'].copy(),
            'highs': raw_data['high'].copy(),
            'lows': raw_data['low'].copy(),
            'closes': raw_data['close'].copy(),
            'volumes': raw_data['volume'].copy()
        }
        
        # Store references to keep memory map alive
        self.mmap_files[file_path] = (f, mm)
        self.total_mapped_size += file_size
        
        return candle_data
    
    def preload_all_to_memory(self, data: Dict) -> Dict:
        """Preload all data arrays to system memory for maximum speed."""
        
        print("Preloading all data to system memory...")
        
        # Calculate total memory needed
        total_memory_needed = 0
        for tick_array in data['ticks'].values():
            total_memory_needed += tick_array.nbytes
        
        for candle_data in data['candles'].values():
            for candle_array in candle_data.values():
                total_memory_needed += candle_array.nbytes
        
        available_memory = psutil.virtual_memory().available
        
        if total_memory_needed > available_memory * 0.8:
            print(f"Warning: Data size ({total_memory_needed / (1024**3):.1f} GB) "
                  f"exceeds 80% of available memory ({available_memory / (1024**3):.1f} GB)")
        
        # Force all arrays to be loaded into memory
        preloaded_data = {}
        
        # Preload tick data
        preloaded_data['ticks'] = {}
        for key, array in data['ticks'].items():
            # Force memory allocation by accessing all elements
            preloaded_data['ticks'][key] = np.array(array, copy=True)
            _ = preloaded_data['ticks'][key].sum()  # Force memory access
        
        # Preload candle data
        preloaded_data['candles'] = {}
        for tf, candle_data in data['candles'].items():
            preloaded_data['candles'][tf] = {}
            for key, array in candle_data.items():
                preloaded_data['candles'][tf][key] = np.array(array, copy=True)
                _ = preloaded_data['candles'][tf][key].sum()  # Force memory access
        
        print(f"All data preloaded to memory: {total_memory_needed / (1024**2):.1f} MB")
        
        return preloaded_data
    
    def create_aligned_datasets(self, data: Dict) -> Dict:
        """Create perfectly aligned datasets for ultra-fast processing."""
        
        print("Creating aligned datasets...")
        
        # Get M1 candle timestamps as reference
        m1_timestamps = data['candles']['M1']['timestamps']
        tick_timestamps = data['ticks']['timestamps']
        
        # Create alignment indices
        alignment_indices = align_tick_to_candle_indices(tick_timestamps, m1_timestamps)
        
        # Create aligned tick data
        aligned_data = {
            'ticks': data['ticks'].copy(),
            'candles': data['candles'].copy(),
            'alignment': {
                'tick_to_m1_indices': alignment_indices,
                'm1_timestamps': m1_timestamps,
                'tick_timestamps': tick_timestamps
            }
        }
        
        print(f"Alignment created: {len(alignment_indices):,} tick-to-candle mappings")
        
        return aligned_data
    
    def cleanup(self):
        """Clean up memory-mapped files."""
        
        for file_path, (f, mm) in self.mmap_files.items():
            try:
                mm.close()
                f.close()
            except Exception as e:
                print(f"Warning: Failed to close {file_path}: {e}")
        
        self.mmap_files.clear()
        self.total_mapped_size = 0
        print("Memory-mapped files cleaned up")


class UltraFastDataLoader:
    """Ultra-fast data loading with automatic optimization."""
    
    def __init__(self, cache_dir: str = "ultra_cache"):
        self.mmap_manager = MemoryMappedDataManager(cache_dir)
        self.cpu_cores = os.cpu_count() or 1
    
    def load_for_backtest(self, symbol: str, tick_file: str, candle_files: Dict[str, str]) -> Dict:
        """Load and optimize data for ultra-fast backtesting."""
        
        print(f"Loading {symbol} data for ultra-fast backtesting...")
        
        # Check if binary cache exists
        binary_files = self._check_binary_cache(symbol)
        
        if not binary_files:
            # Convert to binary format first
            print("Converting to binary format...")
            
            # Load original data
            import pandas as pd
            
            if tick_file.endswith('.pkl'):
                tick_df = pd.read_pickle(tick_file)
            else:
                tick_df = pd.read_csv(tick_file, parse_dates=['datetime'])
            
            candle_dfs = {}
            for tf, file_path in candle_files.items():
                if os.path.exists(file_path):
                    if file_path.endswith('.pkl'):
                        candle_dfs[tf] = pd.read_pickle(file_path)
                    else:
                        candle_dfs[tf] = pd.read_csv(file_path, parse_dates=['datetime'])
            
            # Convert to binary
            binary_files = self.mmap_manager.convert_to_binary_format(symbol, tick_df, candle_dfs)
        
        # Load using memory mapping
        data = self.mmap_manager.load_memory_mapped_data(binary_files)
        
        # Preload to memory for maximum speed
        data = self.mmap_manager.preload_all_to_memory(data)
        
        # Create aligned datasets
        data = self.mmap_manager.create_aligned_datasets(data)
        
        print(f"Ultra-fast data loading completed for {symbol}")
        
        return data
    
    def _check_binary_cache(self, symbol: str) -> Optional[Dict[str, str]]:
        """Check if binary cache files exist."""
        
        tick_file = os.path.join(self.mmap_manager.cache_dir, f"{symbol}_ticks.bin")
        
        if not os.path.exists(tick_file):
            return None
        
        # Check for candle files
        candle_files = {}
        for tf in ['M1', 'M5', 'M15', 'H1', 'H4']:
            candle_file = os.path.join(self.mmap_manager.cache_dir, f"{symbol}_{tf}.bin")
            if os.path.exists(candle_file):
                candle_files[tf] = candle_file
        
        if not candle_files:
            return None
        
        return {
            'ticks': tick_file,
            'candles': candle_files
        }
    
    def get_memory_usage_info(self) -> Dict:
        """Get detailed memory usage information."""
        
        memory = psutil.virtual_memory()
        
        return {
            'total_ram_gb': memory.total / (1024**3),
            'available_ram_gb': memory.available / (1024**3),
            'used_ram_gb': memory.used / (1024**3),
            'memory_percent': memory.percent,
            'mapped_size_mb': self.mmap_manager.total_mapped_size / (1024**2),
            'max_mmap_size_gb': MAX_MMAP_SIZE / (1024**3),
            'cpu_cores': self.cpu_cores
        }
    
    def cleanup(self):
        """Clean up all resources."""
        self.mmap_manager.cleanup()


# Global ultra-fast data loader
ultra_data_loader = UltraFastDataLoader()


@jit(nopython=True, cache=True, parallel=True)
def ultra_fast_data_validation(timestamps: np.ndarray, prices: np.ndarray) -> Tuple[bool, np.int32]:
    """Ultra-fast data validation using parallel processing."""
    
    n = len(timestamps)
    is_valid = True
    error_count = 0
    
    # Check for monotonic timestamps
    for i in prange(1, n):
        if timestamps[i] <= timestamps[i-1]:
            is_valid = False
            error_count += 1
    
    # Check for valid prices
    for i in prange(n):
        if prices[i] <= 0 or not np.isfinite(prices[i]):
            is_valid = False
            error_count += 1
    
    return is_valid, error_count


def validate_ultra_fast_data(data: Dict) -> Dict:
    """Validate data integrity for ultra-fast processing."""
    
    print("Validating data integrity...")
    
    validation_results = {}
    
    # Validate tick data
    tick_valid, tick_errors = ultra_fast_data_validation(
        data['ticks']['timestamps'], 
        data['ticks']['bids']
    )
    
    validation_results['ticks'] = {
        'valid': tick_valid,
        'error_count': int(tick_errors),
        'total_records': len(data['ticks']['timestamps'])
    }
    
    # Validate candle data
    validation_results['candles'] = {}
    for tf, candle_data in data['candles'].items():
        candle_valid, candle_errors = ultra_fast_data_validation(
            candle_data['timestamps'],
            candle_data['closes']
        )
        
        validation_results['candles'][tf] = {
            'valid': candle_valid,
            'error_count': int(candle_errors),
            'total_records': len(candle_data['timestamps'])
        }
    
    print(f"Data validation completed:")
    print(f"  Ticks: {'✓' if validation_results['ticks']['valid'] else '✗'}")
    for tf, result in validation_results['candles'].items():
        print(f"  {tf}: {'✓' if result['valid'] else '✗'}")
    
    return validation_results