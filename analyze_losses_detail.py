import sqlite3, json
from datetime import datetime, timezone, timedelta

_IST = timedelta(hours=5, minutes=30)
tickets = [8405172784, 8405141094, 8404829882, 8404821907, 8404815340, 8404804856]

conn = sqlite3.connect('orders.db')
conn.row_factory = sqlite3.Row
rows = conn.execute(
    f"SELECT * FROM orders WHERE ticket IN ({','.join('?'*len(tickets))})",
    tickets
).fetchall()

orders = {r['ticket']: dict(r) for r in rows}

for t in tickets:
    o = orders.get(t)
    if not o:
        print(f"#{t} NOT FOUND"); continue

    f = json.loads(o.get('features') or '{}')
    ep  = float(o['entry_price'] or 0)
    sl  = float(o['sl'] or 0)
    tp  = float(o['tp'] or 0)
    xp  = float(o['exit_price'] or 0)
    pnl = float(o['final_pnl'] or 0)
    sl_dist = round(abs(ep - sl), 2)
    tp_dist = round(abs(tp - ep), 2)
    rr_planned = round(tp_dist / sl_dist, 2) if sl_dist > 0 else 0
    actual_move = round(xp - ep if o['direction']=='BUY' else ep - xp, 2)
    peak_r = o.get('peak_r')
    final_r = o.get('final_r')

    print(f"\n{'='*60}")
    print(f"#{t} {o['direction']} | PnL: {pnl:+.2f} | {'WIN' if pnl>0 else 'LOSS' if pnl<0 else 'BE'}")
    print(f"  Entry: {ep}  SL: {sl}  TP: {tp}  Exit: {xp}")
    print(f"  SL dist: {sl_dist}  TP dist: {tp_dist}  Planned RR: {rr_planned}")
    print(f"  Actual move: {actual_move:+.2f}  Peak R: {peak_r}  Final R: {final_r}")
    print(f"  Held: {o.get('held_seconds')}s")
    print(f"  Close reason: {o.get('close_reason')}  Category: {o.get('close_reason_category')}")
    print(f"  MT5 close reason: {o.get('mt5_close_reason')}  Comment: {o.get('mt5_close_comment')}")
    print(f"  Open: {o.get('open_time')}  Close: {o.get('close_time')}")

    if f:
        print(f"  --- FEATURES ---")
        print(f"  ATR: {f.get('atr')}  ATR ratio: {f.get('atr_ratio')}")
        print(f"  Body ratio: {f.get('body_ratio')}  RSI: {f.get('rsi')}")
        print(f"  EMA slope: {f.get('ema_slope')}")
        print(f"  Velocity: {f.get('entry_tick_velocity')}  Pressure: {f.get('entry_tick_pressure_score')}  Bias: {f.get('entry_tick_pressure_bias')}")
        print(f"  Volume ratio: {f.get('entry_volume_ratio')}  Strong: {f.get('entry_volume_strong')}")
        print(f"  Regime: {f.get('regime')}  Session: {f.get('session')}  Confidence: {f.get('signal_confidence')}")
        print(f"  Spread: {f.get('spread')}")
        gc = f.get('gate_context') or {}
        print(f"  Gate score: {gc.get('score')}  Confirmations: {gc.get('confirmation_count')}")
        print(f"  Passed: {gc.get('passed_gates')}")
        print(f"  Failed: {gc.get('failed_gates')}")
        m15 = gc.get('m15_context') or {}
        print(f"  M15 context: allowed={m15.get('allowed')} reason={m15.get('reason')} dist_atr={m15.get('distance_atr')}")
        ar = gc.get('auto_relax') or {}
        print(f"  Auto relax: {ar.get('active')} reason={ar.get('reason')}")
    else:
        print(f"  No features stored")

    # Diagnose
    print(f"  --- DIAGNOSIS ---")
    reasons = []

    if sl_dist < 0.5:
        reasons.append("SL too tight (<0.5pt) — noise hit SL")
    if sl_dist > 1.5:
        reasons.append(f"SL wide ({sl_dist}pt) — large loss when hit")
    if rr_planned < 1.0 and pnl < 0:
        reasons.append(f"RR {rr_planned} < 1.0 — negative expectancy setup")
    if f.get('body_ratio') and float(f.get('body_ratio',0)) < 0.5:
        reasons.append(f"Weak displacement body={f.get('body_ratio'):.2f} < 0.5")
    if f.get('entry_volume_ratio') and float(f.get('entry_volume_ratio') or 0) < 1.0:
        reasons.append(f"Below-avg volume ratio={f.get('entry_volume_ratio'):.2f}")
    if f.get('regime') == 'EXPANSION':
        reasons.append("EXPANSION regime — volatile, no edge")
    if f.get('regime') == 'RANGING' and pnl < 0:
        reasons.append("RANGING regime — sweep may be noise not liquidity grab")
    if f.get('atr') and float(f.get('atr',0)) < 1.0:
        reasons.append(f"Low ATR={f.get('atr'):.3f} — insufficient volatility for sweep")
    if o['direction'] == 'BUY' and f.get('entry_tick_pressure_score') and float(f.get('entry_tick_pressure_score',0)) < 0:
        reasons.append("Tick pressure negative on BUY entry")
    if o['direction'] == 'SELL' and f.get('entry_tick_pressure_score') and float(f.get('entry_tick_pressure_score',0)) > 0.1:
        reasons.append(f"Tick pressure positive ({f.get('entry_tick_pressure_score')}) on SELL entry")
    if o.get('mt5_close_reason') == 'sl':
        reasons.append("Hit SL directly — no favorable move after entry")
    if peak_r and float(peak_r or 0) < 0.3 and pnl < 0:
        reasons.append(f"Price never moved favorably (peak_r={peak_r}) — entry timing wrong")
    if peak_r and float(peak_r or 0) >= 0.5 and pnl < 0:
        reasons.append(f"Near-miss: peaked at {peak_r}R then reversed — exit/trail issue")
    held = o.get('held_seconds')
    if held and float(held) < 10 and pnl < 0:
        reasons.append(f"Closed in {held}s — immediate reversal, entry was wrong side")

    for r in reasons:
        print(f"  >> {r}")
    if not reasons:
        print(f"  >> No clear single cause — market moved against setup")
