from pathlib import Path
import re
from datetime import datetime, timezone, timedelta

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from engine.analytics_api import router as analytics_router
from engine.order_api import router as order_router
from engine.trader_api import router as trader_router
import config as cfg

app = FastAPI(title="Trading System API", version=cfg.APP_VERSION)
_BASE_DIR = Path(__file__).resolve().parent
_ANALYTICS_DIR = _BASE_DIR / "analytics_outputs"
_ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)

app.include_router(order_router)
app.include_router(analytics_router)
app.include_router(trader_router)
app.mount("/analytics-assets", StaticFiles(directory=str(_ANALYTICS_DIR)), name="analytics-assets")


@app.get("/")
async def root():
    return {
        "message": "Trading System API",
        "version": cfg.APP_VERSION,
        "analytics_dashboard": "/analytics",
        "trading_dashboard": "/dashboard",
        "status": "/status",
        "health": "/health",
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "trading_system", "app_version": cfg.APP_VERSION}


_LOG_PATTERN = re.compile(
    r'^\[(\d{2}:\d{2}:\d{2}\.\d{3}) UTC \| (\d{2}:\d{2}:\d{2}\.\d{3}) IST(?:\s*\|\s*(\d{4}-\d{2}-\d{2}))?\]'
    r'\[([A-Z_]+)\](?:\[([\d.]+)\])? (.*)$'
)
_IST = timezone(timedelta(hours=5, minutes=30))


@app.get("/logs/today")
async def logs_today():
    """Return TRADE+RESULT entries for today's IST date by cross-referencing the orders DB."""
    import sqlite3
    log_path = _BASE_DIR / "trader.log"
    db_path = _BASE_DIR / "orders.db"
    if not log_path.exists():
        return {"logs": [], "count": 0}

    now_ist = datetime.now(_IST)
    today_ist_date = now_ist.strftime("%Y-%m-%d")

    # Get today's tickets AND open UTC times from DB
    today_tickets: set = set()
    today_open_utc_times: set = set()  # HH:MM strings of open times
    if db_path.exists():
        try:
            con = sqlite3.connect(str(db_path))
            cur = con.execute(
                "SELECT ticket, open_time FROM orders WHERE open_time_ist LIKE ? OR close_time_ist LIKE ?",
                (today_ist_date + "%", today_ist_date + "%")
            )
            for row in cur.fetchall():
                today_tickets.add(str(row[0]))
                # open_time is ISO UTC e.g. "2026-05-04T00:01:56.412630+00:00"
                try:
                    t = str(row[1] or "")
                    hhmm = t[11:16]  # "00:01"
                    if hhmm:
                        today_open_utc_times.add(hhmm)
                except Exception:
                    pass
            con.close()
        except Exception:
            pass

    entries = []
    try:
        raw = log_path.read_bytes().decode("utf-8", errors="ignore")
        for line in raw.splitlines():
            m = _LOG_PATTERN.match(line.strip())
            if not m:
                continue
            utc_t, ist_t, ist_date, tag, pr, msg = m.groups()
            if tag not in ("TRADE", "RESULT"):
                continue
            # New format: use embedded IST date
            if ist_date:
                if ist_date != today_ist_date:
                    continue
            elif today_tickets:
                if tag == "RESULT":
                    # Match by ticket number in message
                    if not any(t in msg for t in today_tickets):
                        continue
                else:  # TRADE
                    # Match by open UTC time HH:MM
                    entry_hhmm = utc_t[:5]  # "00:01"
                    if entry_hhmm not in today_open_utc_times:
                        continue
            else:
                continue
            entries.append({
                "time": utc_t,
                "time_ist": ist_t,
                "tag": tag,
                "msg": msg,
                "price": float(pr) if pr else None
            })
    except Exception as e:
        return {"logs": [], "count": 0, "error": str(e)}

    return {"logs": entries, "count": len(entries)}
