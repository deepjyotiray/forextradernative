"""
One-time sync: update orders.db from snapshot2.xlsx (MT5 export = source of truth).
Updates exit_price, final_pnl, swap, commission for all closed orders.
"""
import sqlite3
import openpyxl

DB_PATH = "orders.db"
XLSX_PATH = "snapshot2.xlsx"


def parse_xlsx():
    wb = openpyxl.load_workbook(XLSX_PATH, read_only=True)
    ws = wb["Sheet1"]
    rows = list(ws.iter_rows(values_only=True))

    # Positions section: header at row 6, data rows 7-136
    positions = {}
    for i in range(7, len(rows)):
        r = rows[i]
        # Stop at next section
        if r and r[0] and isinstance(r[0], str) and r[0].strip() in ("Orders", "Deals", "Total Net Profit:"):
            break
        if not (r and r[1] and isinstance(r[1], (int, float))):
            continue
        ticket = int(r[1])
        positions[ticket] = {
            "exit_price": float(r[9]) if r[9] else 0,
            "commission": float(r[10]) if r[10] else 0,
            "swap": float(r[11]) if r[11] else 0,
            "profit": float(r[12]) if r[12] else 0,
        }
    return positions


def sync():
    xlsx = parse_xlsx()
    print(f"Parsed {len(xlsx)} positions from {XLSX_PATH}")

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.execute("SELECT ticket, exit_price, final_pnl, swap, commission FROM orders WHERE status='CLOSED'")
    db_rows = {row["ticket"]: dict(row) for row in cursor.fetchall()}

    updated = 0
    for ticket, x in xlsx.items():
        if ticket not in db_rows:
            print(f"  SKIP #{ticket}: not in DB")
            continue

        d = db_rows[ticket]
        pnl_diff = abs((d["final_pnl"] or 0) - x["profit"])
        exit_diff = abs((d["exit_price"] or 0) - x["exit_price"]) if x["exit_price"] else 0

        if pnl_diff > 0.005 or exit_diff > 0.005:
            conn.execute("""
                UPDATE orders SET
                    exit_price = ?,
                    final_pnl = ?,
                    swap = ?,
                    commission = ?,
                    updated_at = strftime('%s', 'now')
                WHERE ticket = ?
            """, (x["exit_price"], x["profit"], x["swap"], x["commission"], ticket))
            updated += 1
            print(f"  FIX #{ticket}: pnl {d['final_pnl']} -> {x['profit']}, "
                  f"exit {d['exit_price']} -> {x['exit_price']}")

    conn.commit()
    conn.close()
    print(f"\nDone. Updated {updated} orders.")


if __name__ == "__main__":
    sync()
