"""
Correlation Engine — DXY and US10Y yield proxy as gold directional filters.

Gold correlations:
  - DXY (US Dollar Index): INVERSE — DXY up = gold down
  - US10Y (10-year yield): INVERSE — yields up = gold down (usually)

Reads from MT5 symbols. Common broker names:
  DXY:   "USDX", "DXY", "DX", "USDIndex"
  US10Y: "US10Y", "TNX", "US10YR", "USTBOND"

If symbols not available, falls back to EURUSD as inverse DXY proxy.
"""
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from typing import Dict, Optional
from .indicators import ema


# Try these symbol names in order
_DXY_NAMES = ["USDX", "DXY", "DX", "USDIndex", "DXY.f", "DOLLAR"]
_US10Y_NAMES = ["US10Y.f", "US10Y", "TNX", "US10YR", "USTBOND"]
_EURUSD = "EURUSD"  # fallback inverse DXY proxy


class CorrelationEngine:
    def __init__(self):
        self._dxy_symbol: Optional[str] = None
        self._us10y_symbol: Optional[str] = None
        self._eurusd_available = False
        self._initialized = False

    def init_symbols(self):
        """Detect available correlation symbols on this broker."""
        if self._initialized:
            return

        # Find DXY
        for name in _DXY_NAMES:
            info = mt5.symbol_info(name)
            if info is not None:
                mt5.symbol_select(name, True)
                self._dxy_symbol = name
                break

        # Find US10Y
        for name in _US10Y_NAMES:
            info = mt5.symbol_info(name)
            if info is not None:
                mt5.symbol_select(name, True)
                self._us10y_symbol = name
                break

        # EURUSD fallback
        info = mt5.symbol_info(_EURUSD)
        if info is not None:
            mt5.symbol_select(_EURUSD, True)
            self._eurusd_available = True

        self._initialized = True

    def compute(self) -> Dict:
        """
        Compute correlation-based gold bias.
        Returns: {
            bias: "LONG" / "SHORT" / "NEUTRAL",
            confidence: 0-1,
            dxy: {symbol, trend, slope, available},
            us10y: {symbol, trend, slope, available},
            reasons: [...]
        }
        """
        self.init_symbols()

        reasons = []
        scores = {"LONG": 0.0, "SHORT": 0.0}

        # === DXY analysis ===
        dxy_info = self._analyze_dxy()
        if dxy_info["available"]:
            trend = dxy_info["trend"]
            # DXY up = gold down (inverse)
            if trend == "UP":
                scores["SHORT"] += 0.3
                reasons.append(f"DXY ({dxy_info['symbol']}) trending UP -> gold bearish")
            elif trend == "DOWN":
                scores["LONG"] += 0.3
                reasons.append(f"DXY ({dxy_info['symbol']}) trending DOWN -> gold bullish")

        # === US10Y analysis ===
        us10y_info = self._analyze_us10y()
        if us10y_info["available"]:
            trend = us10y_info["trend"]
            # Yields up = gold down (inverse, usually)
            if trend == "UP":
                scores["SHORT"] += 0.2
                reasons.append(f"US10Y ({us10y_info['symbol']}) trending UP -> gold bearish")
            elif trend == "DOWN":
                scores["LONG"] += 0.2
                reasons.append(f"US10Y ({us10y_info['symbol']}) trending DOWN -> gold bullish")

        # === EURUSD fallback (if no DXY) ===
        if not dxy_info["available"] and self._eurusd_available:
            eu_info = self._analyze_eurusd()
            if eu_info["available"]:
                trend = eu_info["trend"]
                # EURUSD up = DXY down = gold up (same direction)
                if trend == "UP":
                    scores["LONG"] += 0.2
                    reasons.append("EURUSD trending UP (weak USD) -> gold bullish")
                elif trend == "DOWN":
                    scores["SHORT"] += 0.2
                    reasons.append("EURUSD trending DOWN (strong USD) -> gold bearish")

        # === Compute final bias ===
        long_s, short_s = scores["LONG"], scores["SHORT"]
        diff = abs(long_s - short_s)
        if diff < 0.1:
            bias = "NEUTRAL"
            confidence = 0.0
        elif long_s > short_s:
            bias = "LONG"
            confidence = round(min(1.0, diff * 2), 3)
        else:
            bias = "SHORT"
            confidence = round(min(1.0, diff * 2), 3)

        return {
            "bias": bias,
            "confidence": confidence,
            "dxy": dxy_info,
            "us10y": us10y_info,
            "scores": scores,
            "reasons": reasons,
        }

    def _analyze_dxy(self) -> Dict:
        if not self._dxy_symbol:
            return {"available": False, "symbol": None, "trend": None, "slope": 0}
        return self._trend_from_symbol(self._dxy_symbol)

    def _analyze_us10y(self) -> Dict:
        if not self._us10y_symbol:
            return {"available": False, "symbol": None, "trend": None, "slope": 0}
        return self._trend_from_symbol(self._us10y_symbol)

    def _analyze_eurusd(self) -> Dict:
        if not self._eurusd_available:
            return {"available": False, "symbol": None, "trend": None, "slope": 0}
        return self._trend_from_symbol(_EURUSD)

    def _trend_from_symbol(self, symbol: str) -> Dict:
        """Get H4 trend direction from a symbol using EMA20/50."""
        try:
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H4, 0, 100)
            if rates is None or len(rates) < 55:
                return {"available": False, "symbol": symbol, "trend": None, "slope": 0}

            c = np.array([r[4] for r in rates], dtype=float)  # close prices
            ema20 = ema(c, 20)
            ema50 = ema(c, 50)

            slope = ema20[-1] - ema20[-5] if len(ema20) >= 5 else 0
            # Normalize slope relative to price
            norm_slope = slope / c[-1] * 10000 if c[-1] > 0 else 0

            if ema20[-1] > ema50[-1] and norm_slope > 0.5:
                trend = "UP"
            elif ema20[-1] < ema50[-1] and norm_slope < -0.5:
                trend = "DOWN"
            else:
                trend = "FLAT"

            tick = mt5.symbol_info_tick(symbol)
            price = tick.bid if tick else c[-1]

            return {
                "available": True,
                "symbol": symbol,
                "trend": trend,
                "slope": round(norm_slope, 3),
                "price": round(price, 4),
                "ema20": round(ema20[-1], 4),
                "ema50": round(ema50[-1], 4),
            }
        except Exception:
            return {"available": False, "symbol": symbol, "trend": None, "slope": 0}


# Singleton
correlation = CorrelationEngine()
