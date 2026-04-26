"""
High-performance indicator calculations with caching and vectorization.
"""
import numpy as np
import pandas as pd
import numba
from typing import Dict, Optional, Tuple, List
from functools import lru_cache
import hashlib


@numba.jit(nopython=True, cache=True)
def fast_ema(prices: np.ndarray, period: int) -> np.ndarray:
    """Ultra-fast EMA calculation using numba."""
    alpha = 2.0 / (period + 1)
    result = np.empty_like(prices)
    result[0] = prices[0]
    
    for i in range(1, len(prices)):
        result[i] = alpha * prices[i] + (1 - alpha) * result[i - 1]
    
    return result


@numba.jit(nopython=True, cache=True)
def fast_sma(prices: np.ndarray, period: int) -> np.ndarray:
    """Ultra-fast SMA calculation using numba."""
    result = np.empty_like(prices)
    result[:period-1] = np.nan
    
    # Calculate first SMA
    result[period-1] = np.mean(prices[:period])
    
    # Rolling calculation
    for i in range(period, len(prices)):
        result[i] = result[i-1] + (prices[i] - prices[i-period]) / period
    
    return result


@numba.jit(nopython=True, cache=True)
def fast_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> np.ndarray:
    """Ultra-fast ATR calculation using numba."""
    n = len(highs)
    tr = np.empty(n)
    
    # First TR
    tr[0] = highs[0] - lows[0]
    
    # Calculate True Range
    for i in range(1, n):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        tr[i] = max(hl, max(hc, lc))
    
    # Calculate ATR using EMA
    return fast_ema(tr, period)


@numba.jit(nopython=True, cache=True)
def fast_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Ultra-fast RSI calculation using numba."""
    n = len(prices)
    if n < period + 1:
        return np.full(n, 50.0)
    
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    
    # Calculate initial averages
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    
    rsi = np.empty(n)
    rsi[:period] = 50.0  # Default for insufficient data
    
    # Calculate RSI
    alpha = 1.0 / period
    for i in range(period, n - 1):
        avg_gain = (1 - alpha) * avg_gain + alpha * gains[i]
        avg_loss = (1 - alpha) * avg_loss + alpha * losses[i]
        
        if avg_loss == 0:
            rsi[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i + 1] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi


@numba.jit(nopython=True, cache=True)
def fast_bollinger_bands(prices: np.ndarray, period: int = 20, std_dev: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ultra-fast Bollinger Bands calculation."""
    sma = fast_sma(prices, period)
    
    # Calculate rolling standard deviation
    std = np.empty_like(prices)
    std[:period-1] = np.nan
    
    for i in range(period - 1, len(prices)):
        window = prices[i - period + 1:i + 1]
        std[i] = np.std(window)
    
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    
    return upper, sma, lower


@numba.jit(nopython=True, cache=True)
def fast_macd(prices: np.ndarray, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ultra-fast MACD calculation."""
    ema_fast = fast_ema(prices, fast_period)
    ema_slow = fast_ema(prices, slow_period)
    
    macd_line = ema_fast - ema_slow
    signal_line = fast_ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


@numba.jit(nopython=True, cache=True)
def fast_stochastic(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, k_period: int = 14, d_period: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Ultra-fast Stochastic oscillator calculation."""
    n = len(closes)
    k_percent = np.empty(n)
    k_percent[:k_period-1] = 50.0
    
    for i in range(k_period - 1, n):
        highest_high = np.max(highs[i - k_period + 1:i + 1])
        lowest_low = np.min(lows[i - k_period + 1:i + 1])
        
        if highest_high == lowest_low:
            k_percent[i] = 50.0
        else:
            k_percent[i] = ((closes[i] - lowest_low) / (highest_high - lowest_low)) * 100.0
    
    d_percent = fast_sma(k_percent, d_period)
    
    return k_percent, d_percent


class IndicatorCache:
    """High-performance indicator caching system."""
    
    def __init__(self, max_cache_size: int = 1000):
        self.max_cache_size = max_cache_size
        self._cache = {}
        self._access_order = []
    
    def _generate_key(self, data: np.ndarray, indicator: str, **params) -> str:
        """Generate cache key for data and parameters."""
        data_hash = hashlib.md5(data.tobytes()).hexdigest()[:16]
        param_str = "_".join(f"{k}={v}" for k, v in sorted(params.items()))
        return f"{indicator}_{data_hash}_{param_str}"
    
    def get_or_compute(self, data: np.ndarray, indicator: str, compute_func, **params):
        """Get cached result or compute and cache."""
        key = self._generate_key(data, indicator, **params)
        
        if key in self._cache:
            # Move to end (most recently used)
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        
        # Compute new result
        result = compute_func(data, **params)
        
        # Cache management
        if len(self._cache) >= self.max_cache_size:
            # Remove least recently used
            oldest_key = self._access_order.pop(0)
            del self._cache[oldest_key]
        
        # Add to cache
        self._cache[key] = result
        self._access_order.append(key)
        
        return result
    
    def clear(self):
        """Clear all cached data."""
        self._cache.clear()
        self._access_order.clear()


class OptimizedIndicatorEngine:
    """High-performance indicator calculation engine."""
    
    def __init__(self, enable_cache: bool = True, cache_size: int = 1000):
        self.enable_cache = enable_cache
        self.cache = IndicatorCache(cache_size) if enable_cache else None
        self._batch_size = 10000
    
    def compute_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Compute EMA with caching."""
        if self.enable_cache:
            return self.cache.get_or_compute(prices, "ema", fast_ema, period=period)
        return fast_ema(prices, period)
    
    def compute_sma(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Compute SMA with caching."""
        if self.enable_cache:
            return self.cache.get_or_compute(prices, "sma", fast_sma, period=period)
        return fast_sma(prices, period)
    
    def compute_atr(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> np.ndarray:
        """Compute ATR with caching."""
        combined_data = np.column_stack([highs, lows, closes]).flatten()
        
        if self.enable_cache:
            return self.cache.get_or_compute(combined_data, "atr", 
                                           lambda data, **kwargs: fast_atr(
                                               data.reshape(-1, 3)[:, 0],
                                               data.reshape(-1, 3)[:, 1], 
                                               data.reshape(-1, 3)[:, 2],
                                               kwargs['period']
                                           ), period=period)
        return fast_atr(highs, lows, closes, period)
    
    def compute_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Compute RSI with caching."""
        if self.enable_cache:
            return self.cache.get_or_compute(prices, "rsi", fast_rsi, period=period)
        return fast_rsi(prices, period)
    
    def compute_bollinger_bands(self, prices: np.ndarray, period: int = 20, std_dev: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute Bollinger Bands with caching."""
        if self.enable_cache:
            return self.cache.get_or_compute(prices, "bb", fast_bollinger_bands, period=period, std_dev=std_dev)
        return fast_bollinger_bands(prices, period, std_dev)
    
    def compute_macd(self, prices: np.ndarray, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute MACD with caching."""
        if self.enable_cache:
            return self.cache.get_or_compute(prices, "macd", fast_macd, 
                                           fast_period=fast_period, slow_period=slow_period, signal_period=signal_period)
        return fast_macd(prices, fast_period, slow_period, signal_period)
    
    def compute_stochastic(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, k_period: int = 14, d_period: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Stochastic with caching."""
        combined_data = np.column_stack([highs, lows, closes]).flatten()
        
        if self.enable_cache:
            return self.cache.get_or_compute(combined_data, "stoch",
                                           lambda data, **kwargs: fast_stochastic(
                                               data.reshape(-1, 3)[:, 0],
                                               data.reshape(-1, 3)[:, 1],
                                               data.reshape(-1, 3)[:, 2],
                                               kwargs['k_period'], kwargs['d_period']
                                           ), k_period=k_period, d_period=d_period)
        return fast_stochastic(highs, lows, closes, k_period, d_period)
    
    def compute_all_indicators(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Compute all common indicators for a DataFrame."""
        if len(df) < 50:
            return {}
        
        closes = df['close'].values.astype(np.float64)
        highs = df['high'].values.astype(np.float64)
        lows = df['low'].values.astype(np.float64)
        
        indicators = {}
        
        # EMAs
        indicators['ema9'] = self.compute_ema(closes, 9)
        indicators['ema20'] = self.compute_ema(closes, 20)
        indicators['ema50'] = self.compute_ema(closes, 50) if len(closes) >= 50 else indicators['ema20']
        
        # ATR
        indicators['atr'] = self.compute_atr(highs, lows, closes, 14)
        indicators['atr7'] = self.compute_atr(highs, lows, closes, 7)
        
        # RSI
        indicators['rsi'] = self.compute_rsi(closes, 14)
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self.compute_bollinger_bands(closes, 20, 2.0)
        indicators['bb_upper'] = bb_upper
        indicators['bb_middle'] = bb_middle
        indicators['bb_lower'] = bb_lower
        
        # MACD
        macd_line, signal_line, histogram = self.compute_macd(closes)
        indicators['macd'] = macd_line
        indicators['macd_signal'] = signal_line
        indicators['macd_histogram'] = histogram
        
        # Stochastic
        stoch_k, stoch_d = self.compute_stochastic(highs, lows, closes)
        indicators['stoch_k'] = stoch_k
        indicators['stoch_d'] = stoch_d
        
        return indicators
    
    def compute_batch_indicators(self, dataframes: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, np.ndarray]]:
        """Compute indicators for multiple timeframes efficiently."""
        all_indicators = {}
        
        for timeframe, df in dataframes.items():
            if df.empty or len(df) < 20:
                all_indicators[timeframe] = {}
                continue
            
            all_indicators[timeframe] = self.compute_all_indicators(df)
        
        return all_indicators
    
    def update_incremental(self, existing_indicators: Dict[str, np.ndarray], 
                          new_data: pd.DataFrame, lookback: int = 100) -> Dict[str, np.ndarray]:
        """Update indicators incrementally with new data."""
        
        if not existing_indicators or len(new_data) == 0:
            return self.compute_all_indicators(new_data)
        
        # For incremental updates, we need to recalculate with some lookback
        # This is a simplified version - full implementation would be more complex
        
        updated_indicators = {}
        
        closes = new_data['close'].values.astype(np.float64)
        highs = new_data['high'].values.astype(np.float64)
        lows = new_data['low'].values.astype(np.float64)
        
        # Take last N points from existing + new data for recalculation
        if len(closes) > lookback:
            start_idx = len(closes) - lookback
            closes = closes[start_idx:]
            highs = highs[start_idx:]
            lows = lows[start_idx:]
        
        # Recalculate indicators
        updated_indicators = self.compute_all_indicators(
            pd.DataFrame({
                'close': closes,
                'high': highs,
                'low': lows,
                'open': new_data['open'].values[-len(closes):] if 'open' in new_data.columns else closes
            })
        )
        
        return updated_indicators


# Global optimized indicator engine instance
optimized_indicators = OptimizedIndicatorEngine(enable_cache=True, cache_size=2000)


def compute_indicators_fast(df: pd.DataFrame) -> Dict:
    """Drop-in replacement for original compute_indicators function."""
    
    if len(df) < 21:
        return {}
    
    indicators_arrays = optimized_indicators.compute_all_indicators(df)
    
    # Convert to scalar values (last element) for compatibility
    result = {}
    last_idx = len(df) - 1
    
    for key, array in indicators_arrays.items():
        if len(array) > last_idx:
            result[key] = float(array[last_idx])
    
    # Add additional computed fields for compatibility
    if 'ema9' in result and 'ema20' in result:
        result['ema_gap'] = abs(result['ema9'] - result['ema20'])
    
    if 'atr' in result and 'atr7' in result:
        result['atr_ratio'] = result['atr7'] / result['atr'] if result['atr'] > 0 else 1.0
    
    # Add candle analysis
    if len(df) > 0:
        last_candle = df.iloc[-1]
        candle_range = last_candle['high'] - last_candle['low']
        body = abs(last_candle['close'] - last_candle['open'])
        
        result['body_ratio'] = body / candle_range if candle_range > 0 else 0
        result['candle_range'] = candle_range
        result['is_bullish'] = last_candle['close'] > last_candle['open']
    
    return result