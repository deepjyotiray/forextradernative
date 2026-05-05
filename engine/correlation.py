"""
Correlation Engine — Multi-market gold directional filters.

Gold correlations:
  - DXY (US Dollar Index):  INVERSE  — DXY up   = gold down
  - US10Y (10-yr yield):    INVERSE  — yield up  = gold down
  - Oil (WTI/Brent):        DIRECT   — oil up    = gold up  (inflation proxy)
  - US Stocks (SPX/NQ/DJI): INVERSE  — stocks up = gold down (risk-on)
                             DIRECT   — stocks crash = gold up (safe haven)

Reads from MT5 symbols. Common broker names:
  DXY:    "USDX", "DXY", "DX", "USDIndex"
  US10Y:  "US10Y", "TNX", "US10YR", "USTBOND"
  Oil:    "USOIL", "WTI", "XTIUSD", "OIL", "BRENT", "XBRUSD"
  SP500:  "US500", "SPX500", "SP500", "US500.f", "SPX"
  Nasdaq: "US100", "NAS100", "NASDAQ", "US100.f", "NQ"
  Dow:    "US30", "DJ30", "DOW30", "US30.f", "DJIA"

Bias-building framework (confluence = higher confidence):
  DXY      falling  -> bullish gold
  US10Y    falling  -> bullish gold
  Oil      rising   -> bullish gold (inflation demand)
  Stocks   falling  -> bullish gold (safe haven flow)
  3+ markets aligned = high-confidence gold bias

If DXY not available, falls back to EURUSD as inverse DXY proxy.
"""
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from typing import Dict, Optional
from .indicators import ema


# Try these symbol names in order
_DXY_NAMES   = ["USDX", "DXY", "DX", "USDIndex", "DXY.f", "DOLLAR"]
_US10Y_NAMES = ["US10Y.f", "US10Y", "TNX", "US10YR", "USTBOND"]
_OIL_NAMES   = ["USOIL", "WTI", "XTIUSD", "OIL", "BRENT", "XBRUSD", "OIL.f"]
_SP500_NAMES = ["US500", "SPX500", "SP500", "US500.f", "SPX", "S&P500"]
_NAS_NAMES   = ["US100", "NAS100", "NASDAQ", "US100.f", "NQ", "USTEC"]
_DOW_NAMES   = ["US30", "DJ30", "DOW30", "US30.f", "DJIA", "DJI"]
_EURUSD = "EURUSD"  # fallback inverse DXY proxy


class CorrelationEngine:
    def __init__(self):
        self._dxy_symbol: Optional[str] = None
        self._us10y_symbol: Optional[str] = None
        self._oil_symbol: Optional[str] = None
        self._sp500_symbol: Optional[str] = None
        self._nas_symbol: Optional[str] = None
        self._dow_symbol: Optional[str] = None
        self._eurusd_available = False
        self._initialized = False

    def _find_symbol(self, candidates):
        for name in candidates:
            info = mt5.symbol_info(name)
            if info is not None:
                mt5.symbol_select(name, True)
                return name
        return None

    def init_symbols(self):
        """Detect available correlation symbols on this broker."""
        if self._initialized:
            return
        self._dxy_symbol   = self._find_symbol(_DXY_NAMES)
        self._us10y_symbol = self._find_symbol(_US10Y_NAMES)
        self._oil_symbol   = self._find_symbol(_OIL_NAMES)
        self._sp500_symbol = self._find_symbol(_SP500_NAMES)
        self._nas_symbol   = self._find_symbol(_NAS_NAMES)
        self._dow_symbol   = self._find_symbol(_DOW_NAMES)
        info = mt5.symbol_info(_EURUSD)
        if info is not None:
            mt5.symbol_select(_EURUSD, True)
            self._eurusd_available = True
        self._initialized = True

    def compute(self) -> Dict:
        """
        Compute correlation-based gold bias using 6 markets:
          DXY, US10Y, Oil, SP500, Nasdaq, Dow

        Returns: {
            bias: "LONG" / "SHORT" / "NEUTRAL",
            confidence: 0-1,
            confluence: int (number of markets agreeing),
            dxy, us10y, oil, sp500, nas, dow: {symbol, trend, slope, available},
            reasons: [...]
        }

        Correlation rules:
          DXY      UP   -> SHORT gold (inverse)         weight 0.30
          US10Y    UP   -> SHORT gold (inverse)         weight 0.20
          Oil      UP   -> LONG  gold (inflation proxy) weight 0.15
          Stocks   DOWN -> LONG  gold (safe haven)      weight 0.10 each
          Stocks   UP   -> SHORT gold (risk-on)         weight 0.10 each
        """
        self.init_symbols()

        reasons = []
        scores = {"LONG": 0.0, "SHORT": 0.0}
        confluence = 0

        # === DXY — most direct, highest weight ===
        dxy_info = self._analyze_dxy()
        if dxy_info["available"]:
            if dxy_info["trend"] == "UP":
                scores["SHORT"] += 0.30
                confluence += 1
                reasons.append(f"DXY ({dxy_info['symbol']}) UP -> gold bearish")
            elif dxy_info["trend"] == "DOWN":
                scores["LONG"] += 0.30
                confluence += 1
                reasons.append(f"DXY ({dxy_info['symbol']}) DOWN -> gold bullish")

        # === US10Y Bond Yields ===
        us10y_info = self._analyze_us10y()
        if us10y_info["available"]:
            if us10y_info["trend"] == "UP":
                scores["SHORT"] += 0.20
                confluence += 1
                reasons.append(f"US10Y ({us10y_info['symbol']}) UP -> gold bearish")
            elif us10y_info["trend"] == "DOWN":
                scores["LONG"] += 0.20
                confluence += 1
                reasons.append(f"US10Y ({us10y_info['symbol']}) DOWN -> gold bullish")

        # === Oil — inflation proxy, same direction as gold ===
        oil_info = self._analyze_oil()
        if oil_info["available"]:
            if oil_info["trend"] == "UP":
                scores["LONG"] += 0.15
                confluence += 1
                reasons.append(f"Oil ({oil_info['symbol']}) UP -> inflation demand -> gold bullish")
            elif oil_info["trend"] == "DOWN":
                scores["SHORT"] += 0.15
                confluence += 1
                reasons.append(f"Oil ({oil_info['symbol']}) DOWN -> deflation risk -> gold bearish")

        # === US Stocks — risk-on/off proxy ===
        # SP500
        sp500_info = self._analyze_sp500()
        if sp500_info["available"]:
            if sp500_info["trend"] == "DOWN":
                scores["LONG"] += 0.10
                confluence += 1
                reasons.append(f"SP500 ({sp500_info['symbol']}) DOWN -> risk-off -> gold bullish")
            elif sp500_info["trend"] == "UP":
                scores["SHORT"] += 0.10
                confluence += 1
                reasons.append(f"SP500 ({sp500_info['symbol']}) UP -> risk-on -> gold bearish")

        # Nasdaq
        nas_info = self._analyze_nas()
        if nas_info["available"]:
            if nas_info["trend"] == "DOWN":
                scores["LONG"] += 0.10
                confluence += 1
                reasons.append(f"Nasdaq ({nas_info['symbol']}) DOWN -> risk-off -> gold bullish")
            elif nas_info["trend"] == "UP":
                scores["SHORT"] += 0.10
                confluence += 1
                reasons.append(f"Nasdaq ({nas_info['symbol']}) UP -> risk-on -> gold bearish")

        # Dow Jones
        dow_info = self._analyze_dow()
        if dow_info["available"]:
            if dow_info["trend"] == "DOWN":
                scores["LONG"] += 0.10
                confluence += 1
                reasons.append(f"Dow ({dow_info['symbol']}) DOWN -> risk-off -> gold bullish")
            elif dow_info["trend"] == "UP":
                scores["SHORT"] += 0.10
                confluence += 1
                reasons.append(f"Dow ({dow_info['symbol']}) UP -> risk-on -> gold bearish")

        # === EURUSD fallback (if no DXY) ===
        if not dxy_info["available"] and self._eurusd_available:
            eu_info = self._analyze_eurusd()
            if eu_info["available"]:
                if eu_info["trend"] == "UP":
                    scores["LONG"] += 0.20
                    reasons.append("EURUSD UP (weak USD) -> gold bullish")
                elif eu_info["trend"] == "DOWN":
                    scores["SHORT"] += 0.20
                    reasons.append("EURUSD DOWN (strong USD) -> gold bearish")

        # === Final bias ===
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
            "confluence": confluence,
            "dxy": dxy_info,
            "us10y": us10y_info,
            "oil": oil_info,
            "sp500": sp500_info,
            "nas": nas_info,
            "dow": dow_info,
            "scores": scores,
            "reasons": reasons,
        }

    def _analyze_dxy(self)    -> Dict: return self._trend_from_symbol(self._dxy_symbol)
    def _analyze_us10y(self)  -> Dict: return self._trend_from_symbol(self._us10y_symbol)
    def _analyze_oil(self)    -> Dict: return self._trend_from_symbol(self._oil_symbol)
    def _analyze_sp500(self)  -> Dict: return self._trend_from_symbol(self._sp500_symbol)
    def _analyze_nas(self)    -> Dict: return self._trend_from_symbol(self._nas_symbol)
    def _analyze_dow(self)    -> Dict: return self._trend_from_symbol(self._dow_symbol)
    def _analyze_eurusd(self) -> Dict:
        if not self._eurusd_available:
            return {"available": False, "symbol": None, "trend": None, "slope": 0}
        return self._trend_from_symbol(_EURUSD)

    def _trend_from_symbol(self, symbol: Optional[str]) -> Dict:
        """Get H4 trend direction from a symbol using EMA20/50."""
        if not symbol:
            return {"available": False, "symbol": None, "trend": None, "slope": 0}
        try:
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H4, 0, 100)
            if rates is None or len(rates) < 55:  # type: ignore[arg-type]
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
