import sys
sys.path.insert(0, '.')
import config as cfg
from engine.trendline import detect_trendlines
import pandas as pd, json

# Load cached M15 data
import os, glob
cache_files = glob.glob('backtests/data_cache/XAUUSD/*/M15.parquet')
if not cache_files:
    print("No cached M15 data found")
    sys.exit()

df = pd.read_parquet(cache_files[-1])

if hasattr(df, 'columns'):
    print(f"Loaded M15 df: {len(df)} rows, cols={list(df.columns)}")
    # rename if needed
    if 'Open' in df.columns:
        df = df.rename(columns={'Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'})
    df = df.tail(200).reset_index(drop=True)
    result = detect_trendlines(df, symbol='XAUUSD', timeframe='M15')
    tls = result.get('trendlines', [])
    channels = result.get('channels', [])
    print(f"Trendlines: {len(tls)}")
    for tl in tls:
        print(f"  {tl['type']:12s} slope={tl['slope']:+.4f} valid={tl['valid']} touches={tl['touches']} strength={tl['strength_score']}")
    print(f"Channels: {len(channels)}")
    for ch in channels:
        print(f"  {ch}")
    print(f"Trend: {result.get('trend')}")
    print(f"Swing highs: {len(result.get('swing_highs',[]))}  Swing lows: {len(result.get('swing_lows',[]))}")
else:
    print("Unexpected data format:", type(df))
