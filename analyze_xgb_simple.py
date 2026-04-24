import json
import os
from collections import Counter

# Load trade history
trade_file = "trade_history.json.bak"
if os.path.exists(trade_file):
    with open(trade_file) as f:
        trades = json.load(f)
else:
    print("No trade history file found")
    trades = []

print(f"=== XGBoost Analysis ({len(trades)} trades) ===")

# Filter trades with features (XGBoost training data)
featured_trades = [t for t in trades if t.get("features")]
print(f"Trades with features: {len(featured_trades)}")

if not featured_trades:
    print("No trades with features found - XGBoost has no training data!")
    exit()

# Analyze win rates by direction
buy_trades = [t for t in featured_trades if t.get("direction") == "BUY"]
sell_trades = [t for t in featured_trades if t.get("direction") == "SELL"]

print(f"\nBUY trades: {len(buy_trades)}")
print(f"SELL trades: {len(sell_trades)}")

if buy_trades:
    buy_wins = sum(1 for t in buy_trades if t.get("won", False))
    buy_wr = buy_wins / len(buy_trades)
    print(f"BUY Win Rate: {buy_wr:.0%} ({buy_wins}/{len(buy_trades)})")

if sell_trades:
    sell_wins = sum(1 for t in sell_trades if t.get("won", False))
    sell_wr = sell_wins / len(sell_trades)
    print(f"SELL Win Rate: {sell_wr:.0%} ({sell_wins}/{len(sell_trades)})")

# Check if all features are identical (suspicious)
if featured_trades:
    first_features = featured_trades[0].get("features", {})
    identical_features = True
    for trade in featured_trades[1:]:
        if trade.get("features", {}) != first_features:
            identical_features = False
            break
    
    if identical_features:
        print(f"\n⚠️  WARNING: All {len(featured_trades)} trades have IDENTICAL features!")
        print("This suggests feature extraction is broken or frozen.")
        print(f"Features: {first_features}")
    else:
        print(f"\n✓ Features vary across trades (good)")

# Analyze feature importance simulation
feature_names = ["bias_conf", "atr_ratio", "regime", "ema_slope", "rsi", "atr", "body_ratio", "spread", "session"]
if featured_trades:
    # Count unique values per feature to see if they're actually varying
    feature_stats = {}
    for fname in feature_names:
        values = []
        for trade in featured_trades:
            feat = trade.get("features", {})
            if fname in feat:
                values.append(feat[fname])
        
        if values:
            unique_vals = len(set(str(v) for v in values))
            feature_stats[fname] = {
                "unique_values": unique_vals,
                "sample_values": values[:5]
            }
    
    print(f"\nFeature Variation Analysis:")
    for fname, stats in feature_stats.items():
        print(f"  {fname}: {stats['unique_values']} unique values, samples: {stats['sample_values']}")

# Check market sentiment
total_wins = sum(1 for t in featured_trades if t.get("won", False))
total_wr = total_wins / len(featured_trades) if featured_trades else 0

if total_wr > 0.6:
    sentiment = "BULLISH"
elif total_wr < 0.4:
    sentiment = "BEARISH"
else:
    sentiment = "NEUTRAL"

print(f"\nOverall Win Rate: {total_wr:.0%} ({total_wins}/{len(featured_trades)})")
print(f"Market Sentiment: {sentiment}")

# Check for data quality issues
issues = []
if len(featured_trades) < 15:
    issues.append(f"Insufficient training data ({len(featured_trades)} < 15 minimum)")

if buy_trades and sell_trades:
    if abs(buy_wr - sell_wr) < 0.01:
        issues.append("Identical win rates for BUY/SELL (suspicious)")

if identical_features:
    issues.append("All trades have identical features (broken feature extraction)")

if issues:
    print(f"\n🚨 DATA QUALITY ISSUES:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print(f"\n✅ Data quality looks reasonable")

# Simulate top features (based on common XGBoost patterns)
print(f"\nTOP FEATURES (simulated):")
print(f"  bias_conf: 12%")
print(f"  atr_ratio: 12%") 
print(f"  regime: 11%")
print(f"  ema_slope: 11%")