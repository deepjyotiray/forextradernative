import MetaTrader5 as mt5
import time
mt5.initialize()
mt5.symbol_select('XAUUSD', True)
time.sleep(1)

import pandas as pd
from engine.indicators import compute_indicators
from engine.zones import ZoneDetector
from engine.regime import classify_regime
from engine.mtf_bias import compute_bias
from engine.liquidity import compute_liquidity
from engine.sweep_scalper import SweepScalper
from engine.smc_strategy import SMCStrategy
from engine.tick_processor import TickProcessor

# Fetch data
candles = {}
for tf, mtf in [("M1", mt5.TIMEFRAME_M1), ("M5", mt5.TIMEFRAME_M5),
                ("M15", mt5.TIMEFRAME_M15), ("H1", mt5.TIMEFRAME_H1),
                ("H4", mt5.TIMEFRAME_H4), ("D1", mt5.TIMEFRAME_D1)]:
    rates = mt5.copy_rates_from_pos('XAUUSD', mtf, 0, 1000)
    if rates is not None:
        df = pd.DataFrame(rates)
        df["datetime"] = pd.to_datetime(df["time"], unit="s")
        df = df.rename(columns={"tick_volume": "volume"})
        candles[tf] = df

t = mt5.symbol_info_tick('XAUUSD')
tick = {"bid": t.bid, "ask": t.ask, "spread": round(t.ask - t.bid, 2)}
ind = compute_indicators(candles["M1"])
zones = ZoneDetector().detect(candles["M5"])

print(f"Price: {tick['bid']}  Spread: {tick['spread']}")
print(f"ATR: {ind.get('atr')}  RSI: {ind.get('rsi')}")
print()

data = {
    "m1_df": candles.get("M1"), "m5_df": candles.get("M5"),
    "m15_df": candles.get("M15"), "h1_df": candles.get("H1"),
    "h4_df": candles.get("H4"), "d1_df": candles.get("D1"),
    "tick": tick, "zones": zones, "indicators": ind,
    "account": {}, "positions": [], "correlation": {}, "calendar": {},
}

print("=== SWEEP SCALPER ===")
sc = SweepScalper()
# Feed ticks to build buffer
for i in range(10):
    sc.tick_proc.feed(tick)
    time.sleep(0.01)
sig = sc.generate_signal(data)
print(f"Signal: {sig.get('signal')}")
print(f"Reason: {sig.get('reason')}")
print()

print("=== SMC CONFLUENCE ===")
smc = SMCStrategy()
for i in range(10):
    smc.tick_proc.feed(tick)
    time.sleep(0.01)
sig2 = smc.generate_signal(data)
print(f"Signal: {sig2.get('signal')}")
print(f"Reason: {sig2.get('reason')}")
if sig2.get('score'):
    print(f"Score: {sig2.get('score')}")
if sig2.get('reasons'):
    print(f"Reasons: {sig2.get('reasons')}")
print()

mt5.shutdown()
