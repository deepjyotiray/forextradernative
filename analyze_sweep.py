import sqlite3, json
from datetime import datetime
from collections import Counter

conn = sqlite3.connect('orders.db')
conn.row_factory = sqlite3.Row
rows = conn.execute(
    "SELECT * FROM orders WHERE strategy='SWEEP_SCALPER' AND status='CLOSED' ORDER BY open_time"
).fetchall()

orders = []
for r in rows:
    o = dict(r)
    o['features'] = json.loads(o.get('features') or '{}')
    ep = float(o['entry_price'] or 0)
    sl = float(o['sl'] or 0)
    tp = float(o['tp'] or 0)
    xp = float(o['exit_price'] or 0)
    pnl = float(o['final_pnl'] or 0)
    sl_dist = abs(ep - sl) if ep and sl else 0
    tp_dist = abs(tp - ep) if tp and ep else 0
    o['_sl_dist']    = round(sl_dist, 2)
    o['_tp_dist']    = round(tp_dist, 2)
    o['_planned_rr'] = round(tp_dist / sl_dist, 2) if sl_dist > 0 else 0
    o['_won']        = pnl > 0
    try:
        dt = datetime.fromisoformat(o['open_time'])
        h  = dt.hour
        o['_session'] = 'LONDON' if 7<=h<12 else 'OVERLAP' if 12<=h<14 else 'NY' if 14<=h<21 else 'ASIAN'
    except:
        o['_session'] = 'UNKNOWN'
    orders.append(o)

wins   = [o for o in orders if o['_won']]
losses = [o for o in orders if float(o['final_pnl'] or 0) < 0]
pnls   = [float(o['final_pnl'] or 0) for o in orders]
win_pnls  = [float(o['final_pnl']) for o in wins]
loss_pnls = [abs(float(o['final_pnl'])) for o in losses]
wr        = len(wins) / len(orders) if orders else 0
avg_win   = sum(win_pnls)  / len(win_pnls)  if win_pnls  else 0
avg_loss  = sum(loss_pnls) / len(loss_pnls) if loss_pnls else 0
expectancy = wr * avg_win - (1 - wr) * avg_loss
avg_rr     = sum(o['_planned_rr'] for o in orders) / len(orders) if orders else 0

# Max drawdown
cum, peak, max_dd = 0, 0, 0
for p in pnls:
    cum += p
    if cum > peak: peak = cum
    dd = peak - cum
    if dd > max_dd: max_dd = dd

print("=== CORE METRICS ===")
print(f"Total: {len(orders)} | Wins: {len(wins)} | Losses: {len(losses)}")
print(f"Win rate: {wr*100:.1f}%")
print(f"Avg win: +{avg_win:.2f} | Avg loss: -{avg_loss:.2f}")
print(f"Expectancy: {expectancy:.2f}")
print(f"Total PnL: {sum(pnls):.2f}")
print(f"Avg planned RR: {avg_rr:.2f}")
print(f"Max drawdown: {max_dd:.2f}")

print("\n=== SESSION ===")
for sess in ['LONDON','OVERLAP','NY','ASIAN']:
    s  = [o for o in orders if o['_session'] == sess]
    if not s: continue
    sw = [o for o in s if o['_won']]
    print(f"  {sess}: {len(s)} trades | WR {len(sw)/len(s)*100:.0f}% | PnL {sum(float(o['final_pnl'] or 0) for o in s):.2f}")

print("\n=== DIRECTION ===")
for d in ['BUY','SELL']:
    s  = [o for o in orders if o['direction'] == d]
    if not s: continue
    sw = [o for o in s if o['_won']]
    print(f"  {d}: {len(s)} trades | WR {len(sw)/len(s)*100:.0f}% | PnL {sum(float(o['final_pnl'] or 0) for o in s):.2f}")

print("\n=== SL TIGHTNESS ===")
for label, lo, hi in [('<0.5',0,0.5),('0.5-1.0',0.5,1.0),('1.0-1.5',1.0,1.5),('>1.5',1.5,999)]:
    s  = [o for o in orders if lo <= o['_sl_dist'] < hi]
    if not s: continue
    sw = [o for o in s if o['_won']]
    print(f"  SL {label}pt: {len(s)} trades | WR {len(sw)/len(s)*100:.0f}%")

print("\n=== PLANNED RR ===")
for label, lo, hi in [('<1.0',0,1.0),('1.0-1.5',1.0,1.5),('1.5-2.0',1.5,2.0),('>2.0',2.0,999)]:
    s  = [o for o in orders if lo <= o['_planned_rr'] < hi]
    if not s: continue
    sw = [o for o in s if o['_won']]
    print(f"  RR {label}: {len(s)} trades | WR {len(sw)/len(s)*100:.0f}%")

print("\n=== CLOSE REASONS (losses) ===")
print(Counter(o['close_reason'] or 'unknown' for o in losses))

print("\n=== HELD TIME ===")
held   = [float(o['held_seconds']) for o in orders if o['held_seconds']]
held_w = [float(o['held_seconds']) for o in wins   if o['held_seconds']]
held_l = [float(o['held_seconds']) for o in losses if o['held_seconds']]
if held:   print(f"  All  n={len(held)}  avg={sum(held)/len(held):.0f}s  min={min(held):.0f}s  max={max(held):.0f}s")
if held_w: print(f"  Wins avg held: {sum(held_w)/len(held_w):.0f}s")
if held_l: print(f"  Losses avg held: {sum(held_l)/len(held_l):.0f}s")

print("\n=== PEAK R ANALYSIS ===")
peak_r_data = [o for o in orders if o['peak_r'] is not None]
if peak_r_data:
    pw = [float(o['peak_r']) for o in peak_r_data if o['_won']]
    pl = [float(o['peak_r']) for o in peak_r_data if not o['_won']]
    print(f"  Wins  avg peak_r: {sum(pw)/len(pw):.2f}" if pw else "")
    print(f"  Losses avg peak_r: {sum(pl)/len(pl):.2f}" if pl else "")
    # Losses that peaked > 0.5R before reversing = entry was right, exit was wrong
    near_miss = [o for o in peak_r_data if not o['_won'] and float(o['peak_r']) >= 0.5]
    print(f"  Near-miss losses (peak_r>=0.5): {len(near_miss)}")
else:
    print("  No peak_r data")

print("\n=== FEATURES ANALYSIS ===")
ft = [o for o in orders if o['features'] and o['features'].get('atr')]
print(f"  Trades with features: {len(ft)}")
if ft:
    # ATR buckets
    for label, lo, hi in [('<1.0',0,1.0),('1.0-2.0',1.0,2.0),('2.0-3.0',2.0,3.0),('>3.0',3.0,999)]:
        s  = [o for o in ft if lo <= float(o['features'].get('atr',0)) < hi]
        if not s: continue
        sw = [o for o in s if o['_won']]
        print(f"  ATR {label}: {len(s)} trades | WR {len(sw)/len(s)*100:.0f}%")
    # Body ratio
    for label, lo, hi in [('<0.5',0,0.5),('0.5-0.7',0.5,0.7),('>0.7',0.7,999)]:
        s  = [o for o in ft if lo <= float(o['features'].get('body_ratio',0)) < hi]
        if not s: continue
        sw = [o for o in s if o['_won']]
        print(f"  Body {label}: {len(s)} trades | WR {len(sw)/len(s)*100:.0f}%")
    # Regime
    for reg in ['TRENDING','RANGING','EXPANSION']:
        s  = [o for o in ft if o['features'].get('regime') == reg]
        if not s: continue
        sw = [o for o in s if o['_won']]
        print(f"  Regime {reg}: {len(s)} trades | WR {len(sw)/len(s)*100:.0f}%")
    # Velocity
    for label, lo, hi in [('<5',0,5),('5-10',5,10),('>10',10,999)]:
        s  = [o for o in ft if lo <= float(o['features'].get('entry_tick_velocity',0)) < hi]
        if not s: continue
        sw = [o for o in s if o['_won']]
        print(f"  Velocity {label}/s: {len(s)} trades | WR {len(sw)/len(s)*100:.0f}%")
    # Volume ratio
    for label, lo, hi in [('<1.0',0,1.0),('1.0-1.5',1.0,1.5),('>1.5',1.5,999)]:
        s  = [o for o in ft if lo <= float(o['features'].get('entry_volume_ratio',0) or 0) < hi]
        if not s: continue
        sw = [o for o in s if o['_won']]
        print(f"  Vol ratio {label}x: {len(s)} trades | WR {len(sw)/len(s)*100:.0f}%")
    print("\n  Per-trade detail:")
    for o in ft:
        f = o['features']
        print(f"  #{o['ticket']} {o['direction']} pnl={float(o['final_pnl']):+.2f} sl_dist={o['_sl_dist']} rr={o['_planned_rr']} | "
              f"atr={f.get('atr')} body={f.get('body_ratio')} vel={f.get('entry_tick_velocity')} "
              f"press={f.get('entry_tick_pressure_score')} regime={f.get('regime')} "
              f"sess={f.get('session')} conf={f.get('signal_confidence')} vol={f.get('entry_volume_ratio')}")
