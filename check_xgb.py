import json
import os

# Check what the bot's closed trades look like (features populated?)
state_file = "open_trades.json"
perf_file = "trade_history.json"

print("=== PERFORMANCE TRACKER (trade_history.json) ===")
if os.path.isfile(perf_file):
    with open(perf_file) as f:
        trades = json.load(f)
    print(f"{len(trades)} trades")
    if trades:
        last = trades[-1]
        print(f"Last trade keys: {list(last.keys())}")
        print(f"Has features: {'features' in last}")
        if 'features' in last:
            print(f"Features: {last['features']}")
        else:
            print("NO FEATURES - XGBoost training on defaults!")
        print()
        # Count how many have real features
        with_feat = sum(1 for t in trades if t.get("features") and any(v != 0 for v in t["features"].values() if isinstance(v, (int, float))))
        print(f"Trades with real features: {with_feat}/{len(trades)}")
else:
    print("File not found")

print()

# Check CSV closed trades
from engine import csv_reader
import MetaTrader5 as mt5
mt5.initialize()
ti = mt5.terminal_info()
csv_reader.set_files_dir(ti.data_path)

csv_trades = csv_reader.get_closed_trades()
print(f"=== CSV DEALS ({len(csv_trades)} trades) ===")
if csv_trades:
    last = csv_trades[-1]
    print(f"Keys: {list(last.keys())}")
    print(f"Has features: {'features' in last}")
    print(f"Has atr: {'atr' in last}")
    print(f"Has session: {last.get('session')}")
    print(f"Has reason: {last.get('reason')}")
    print()

# Check what XGBoost _extract_features gets from each source
from engine.xgb_model import _extract_features

print("=== FEATURE EXTRACTION TEST ===")
if csv_trades:
    feat = _extract_features(csv_trades[-1])
    print(f"From CSV trade: {feat}")
    print(f"All defaults? {feat == [0, 0, 12, 1.5, 1.0, 50, 0, 0.5, 0.3, 0, 0.5, 0.5, 0.01] if feat else 'N/A'}")

print()
if os.path.isfile(perf_file):
    with open(perf_file) as f:
        trades = json.load(f)
    if trades:
        feat2 = _extract_features(trades[-1])
        print(f"From perf trade: {feat2}")
        if trades[-1].get("features"):
            print(f"Stored features: {trades[-1]['features']}")

mt5.shutdown()

# Check XGBoost model status
print()
from engine.xgb_model import xgb_model
print(f"=== MODEL STATUS ===")
print(f"Trained: {xgb_model.is_trained}")
print(f"Trades at last train: {xgb_model._trades_at_last_train}")
print(f"Feature importance: {xgb_model.get_feature_importance()}")
