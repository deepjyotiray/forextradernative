from engine import csv_reader
import MetaTrader5 as mt5

mt5.initialize()
ti = mt5.terminal_info()
csv_reader.set_files_dir(ti.data_path)
trades = csv_reader.get_closed_trades()
mt5.shutdown()

print(f"{'='*60}")
print(f"  TRADE ANALYSIS — {len(trades)} trades")
print(f"{'='*60}")

# By hour
by_hour = {}
for t in trades:
    h = 0
    ct = t.get("open_time", "")
    parts = ct.split(" ")
    if len(parts) >= 2:
        h = int(parts[1].split(":")[0])
    by_hour.setdefault(h, []).append(t)

print("\n=== BY HOUR (UTC) ===")
print(f"{'Hour':>5} {'Trades':>7} {'Wins':>5} {'WR':>6} {'PnL':>8} {'Avg':>7}")
for h in sorted(by_hour):
    tt = by_hour[h]
    wins = sum(1 for t in tt if t["won"])
    pnl = sum(t["pnl"] for t in tt)
    wr = wins / len(tt) if tt else 0
    avg = pnl / len(tt) if tt else 0
    flag = " ***" if wr >= 0.7 and pnl > 0 else " !!!" if wr < 0.4 else ""
    print(f"  {h:02d}   {len(tt):>5}   {wins:>3}  {wr:>5.0%}  ${pnl:>+7.2f}  ${avg:>+6.2f}{flag}")

# By session
by_session = {}
for t in trades:
    by_session.setdefault(t.get("session", "?"), []).append(t)

print("\n=== BY SESSION ===")
for s in ["LONDON", "NEW_YORK", "ASIAN", "UNKNOWN"]:
    tt = by_session.get(s, [])
    if not tt:
        continue
    wins = sum(1 for t in tt if t["won"])
    pnl = sum(t["pnl"] for t in tt)
    wr = wins / len(tt) if tt else 0
    avg = pnl / len(tt) if tt else 0
    print(f"  {s:<12} {len(tt):>3} trades  {wins}W/{len(tt)-wins}L  WR:{wr:.0%}  PnL:${pnl:+.2f}  Avg:${avg:+.2f}")

# By strategy
by_strat = {}
for t in trades:
    comment = t.get("entry_comment", "")
    if "SWEEP" in comment:
        s = "SWEEP_SCALPER"
    elif "SMC" in comment:
        s = "SMC_CONFLUENCE"
    else:
        s = "UNKNOWN"
    by_strat.setdefault(s, []).append(t)

print("\n=== BY STRATEGY ===")
for s, tt in by_strat.items():
    wins = sum(1 for t in tt if t["won"])
    pnl = sum(t["pnl"] for t in tt)
    wr = wins / len(tt) if tt else 0
    avg = pnl / len(tt) if tt else 0
    print(f"  {s:<16} {len(tt):>3} trades  {wins}W/{len(tt)-wins}L  WR:{wr:.0%}  PnL:${pnl:+.2f}  Avg:${avg:+.2f}")

# By close reason
by_reason = {}
for t in trades:
    by_reason.setdefault(t.get("reason", "?"), []).append(t)

print("\n=== BY CLOSE REASON ===")
for r, tt in sorted(by_reason.items(), key=lambda x: -sum(t["pnl"] for t in x[1])):
    wins = sum(1 for t in tt if t["won"])
    pnl = sum(t["pnl"] for t in tt)
    wr = wins / len(tt) if tt else 0
    print(f"  {r:<10} {len(tt):>3} trades  {wins}W/{len(tt)-wins}L  WR:{wr:.0%}  PnL:${pnl:+.2f}")

# Window analysis: 07-12,13-17 vs outside
print("\n=== WINDOW ANALYSIS ===")
inside = []
outside = []
for t in trades:
    h = 0
    ct = t.get("open_time", "")
    parts = ct.split(" ")
    if len(parts) >= 2:
        h = int(parts[1].split(":")[0])
    if (7 <= h < 12) or (13 <= h < 17):
        inside.append(t)
    else:
        outside.append(t)

for label, tt in [("INSIDE 07-12,13-17", inside), ("OUTSIDE", outside)]:
    if not tt:
        print(f"  {label}: no trades")
        continue
    wins = sum(1 for t in tt if t["won"])
    pnl = sum(t["pnl"] for t in tt)
    wr = wins / len(tt)
    avg = pnl / len(tt)
    print(f"  {label}: {len(tt)} trades  {wins}W/{len(tt)-wins}L  WR:{wr:.0%}  PnL:${pnl:+.2f}  Avg:${avg:+.2f}")

# Extended window: 07-19 vs outside
inside2 = []
outside2 = []
for t in trades:
    h = 0
    ct = t.get("open_time", "")
    parts = ct.split(" ")
    if len(parts) >= 2:
        h = int(parts[1].split(":")[0])
    if 7 <= h < 20:
        inside2.append(t)
    else:
        outside2.append(t)

print(f"\n  EXTENDED 07-20: {len(inside2)} trades  PnL:${sum(t['pnl'] for t in inside2):+.2f}")
print(f"  OUTSIDE 20-07: {len(outside2)} trades  PnL:${sum(t['pnl'] for t in outside2):+.2f}")

# Duration analysis
print("\n=== TRADE DURATION ===")
for t in trades:
    try:
        parts_open = t["open_time"].split(" ")
        parts_close = t["close_time"].split(" ")
        oh, om, os_ = [int(x) for x in parts_open[1].split(":")]
        ch, cm, cs_ = [int(x) for x in parts_close[1].split(":")]
        dur = (ch * 3600 + cm * 60 + cs_) - (oh * 3600 + om * 60 + os_)
        t["duration_s"] = dur
    except:
        t["duration_s"] = 0

winners = [t for t in trades if t["won"]]
losers = [t for t in trades if not t["won"]]
avg_win_dur = sum(t["duration_s"] for t in winners) / len(winners) if winners else 0
avg_loss_dur = sum(t["duration_s"] for t in losers) / len(losers) if losers else 0
print(f"  Avg winner duration: {avg_win_dur:.0f}s ({avg_win_dur/60:.1f}min)")
print(f"  Avg loser duration:  {avg_loss_dur:.0f}s ({avg_loss_dur/60:.1f}min)")
