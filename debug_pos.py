import MetaTrader5 as mt5
import time
mt5.initialize()
mt5.symbol_select('XAUUSD', True)
time.sleep(1)
pos = mt5.positions_get(symbol='XAUUSD')
if pos:
    p = pos[0]
    print("Position fields:")
    for attr in dir(p):
        if not attr.startswith('_'):
            try:
                print(f"  {attr} = {getattr(p, attr)}")
            except:
                pass
else:
    print("No positions")
mt5.shutdown()
