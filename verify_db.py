import sqlite3
conn = sqlite3.connect('orders.db')
conn.row_factory = sqlite3.Row

# Get all closed trades in close_time order (ASC for drawdown calc)
c = conn.execute("SELECT final_pnl FROM orders WHERE status='CLOSED' ORDER BY close_time ASC")
pnls = [r['final_pnl'] for r in c.fetchall()]
wins = [p for p in pnls if p > 0]
losses = [p for p in pnls if p <= 0]

avg_win = round(sum(wins) / len(wins), 2) if wins else 0
avg_loss_vals = [abs(p) for p in losses]
avg_loss = round(sum(avg_loss_vals) / len(avg_loss_vals), 2) if avg_loss_vals else 0
win_rate = round(len(wins) / len(pnls) * 100, 1) if pnls else 0
expectancy = round((win_rate/100 * avg_win) - ((1 - win_rate/100) * avg_loss), 2)

cum = 0
peak = 0
max_dd = 0
for p in pnls:
    cum += p
    if cum > peak:
        peak = cum
    dd = peak - cum
    if dd > max_dd:
        max_dd = dd

print("=== Performance Stats (from DB) ===")
print(f"  Trades: {len(pnls)}")
print(f"  Win Rate: {win_rate}%")
print(f"  Total P&L: +{sum(pnls):.2f}")
print(f"  Avg Win: +{avg_win}")
print(f"  Avg Loss: -{avg_loss}")
print(f"  Expectancy: +{expectancy}")
print(f"  Max DD: {round(max_dd, 2)}")

# Daily PNL
print("\n=== Daily PNL (IST) ===")
c = conn.execute("""
    SELECT substr(close_time_ist,1,10) as d, COUNT(*) as t,
           SUM(CASE WHEN final_pnl>0 THEN 1 ELSE 0 END) as w,
           SUM(CASE WHEN final_pnl<0 THEN 1 ELSE 0 END) as l,
           ROUND(SUM(final_pnl),2) as p
    FROM orders WHERE status='CLOSED' GROUP BY d ORDER BY d
""")
for r in c.fetchall():
    print(f"  {r['d']}: ${r['p']:+.2f} | {r['t']} trades ({r['w']}W/{r['l']}L)")

# IST time check
print("\n=== IST Time Samples ===")
c = conn.execute("SELECT ticket,open_time,open_time_ist,close_time,close_time_ist FROM orders ORDER BY close_time LIMIT 3")
for r in c.fetchall():
    print(f"  #{r['ticket']}: open_ist={r['open_time_ist'][:19]} close_ist={r['close_time_ist'][:19]}")
c = conn.execute("SELECT ticket,open_time,open_time_ist,close_time,close_time_ist FROM orders ORDER BY close_time DESC LIMIT 3")
for r in c.fetchall():
    print(f"  #{r['ticket']}: open_ist={r['open_time_ist'][:19]} close_ist={r['close_time_ist'][:19]}")
