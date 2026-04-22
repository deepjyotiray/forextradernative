from engine import csv_reader
from engine.xgb_model import xgb_model
import MetaTrader5 as mt5

mt5.initialize()
ti = mt5.terminal_info()
csv_reader.set_files_dir(ti.data_path)
print("CSV available:", csv_reader.available())

pnl = csv_reader.get_today_pnl()
print("Today PnL:", pnl)

trades = csv_reader.get_closed_trades()
print(f"\n{len(trades)} closed trades:")
for t in trades:
    print(f"  #{t['ticket']} {t['direction']} {t['volume']}lot "
          f"{t['entry_price']}->{t['exit_price']} pnl:${t['pnl']:+.2f} "
          f"{'W' if t['won'] else 'L'} reason:{t['reason']} "
          f"session:{t['session']} entry:{t['entry_comment']}")

print(f"\nXGB trained: {xgb_model.is_trained}")
print(f"Should retrain: {xgb_model.should_retrain(len(trades))}")
if len(trades) >= 15:
    xgb_model.train(trades)
    print("Feature importance:", xgb_model.get_feature_importance())
else:
    print(f"Need {15 - len(trades)} more trades to train")

mt5.shutdown()
