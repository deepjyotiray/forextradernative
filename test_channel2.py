import sys, glob
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from engine.trendline import detect_trendlines
from engine.strategies.trend_channel_strategy import TrendChannelStrategy

files = glob.glob('backtests/data_cache/XAUUSD/*/M15.parquet')
df = pd.read_parquet(files[-1]).tail(200).reset_index(drop=True)

tl = detect_trendlines(df, 'XAUUSD', 'M15')
trendlines = tl.get('trendlines', [])
channels   = tl.get('channels', [])
c = df['close'].values.astype(float)
price = float(c[-1])

print(f"Price: {price:.2f}")
print(f"Trendlines: {len(trendlines)}")
for t in trendlines:
    proj = t['slope'] * (len(c)-1) + t['intercept']
    print(f"  {t['type']:12s} slope={t['slope']:+.4f} valid={t['valid']} proj@now={proj:.2f}")
print(f"Channels: {len(channels)}")

strat = TrendChannelStrategy()
channel = strat._find_channel(tl, c, price, 1)
print(f"\n_find_channel (st_dir=+1 LONG): {channel}")
channel2 = strat._find_channel(tl, c, price, -1)
print(f"_find_channel (st_dir=-1 SHORT): {channel2}")

# Check swing highs above price
swing_highs = tl.get('swing_highs', [])
above = [s['price'] for s in swing_highs[-20:] if s['price'] > price]
print(f"\nRecent swing highs above {price:.2f}: {sorted(above)[:5]}")
swing_lows = tl.get('swing_lows', [])
below = [s['price'] for s in swing_lows[-20:] if s['price'] < price]
print(f"Recent swing lows below {price:.2f}: {sorted(below, reverse=True)[:5]}")
