import sys, glob
sys.path.insert(0, '.')
import numpy as np
import pandas as pd

files = glob.glob('backtests/data_cache/XAUUSD/*/M15.parquet')
df = pd.read_parquet(files[-1]).tail(200).reset_index(drop=True)
h = df['high'].values.astype(float)
l = df['low'].values.astype(float)
c = df['close'].values.astype(float)

# Test both lines
lines = [
    ('support',    0.457911,  4664.33 - 0.457911*103, 103),
    ('resistance', -1.480532, 4722.32 - (-1.480532)*104, 103),
]

for ltype, slope, intercept, first_anchor in lines:
    viol = 0
    total = 0
    for i in range(first_anchor, 200):
        lp = slope * i + intercept
        tol = lp * 0.003
        total += 1
        if ltype == 'support' and c[i] < lp - tol:
            viol += 1
        elif ltype == 'resistance' and c[i] > lp + tol:
            viol += 1
    max_v = max(2, int(total * 0.20))
    print(f"{ltype}: slope={slope:.4f} viol={viol}/{total} max={max_v} valid={viol<=max_v}")
    # Show line projection at key points
    print(f"  line@{first_anchor}={slope*first_anchor+intercept:.2f} price={c[first_anchor]:.2f}")
    print(f"  line@199={slope*199+intercept:.2f} price={c[199]:.2f}")
