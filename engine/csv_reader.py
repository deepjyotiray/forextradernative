"""
CSV Reader — reads EA-exported files. Source of truth for P&L and trade history.
"""
import os
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timezone
import config as cfg

_files_dir = ""
_cache: Dict[str, tuple] = {}


def set_files_dir(mt5_data_path: str):
    global _files_dir
    _files_dir = os.path.join(mt5_data_path, "MQL5", "Files")


def _path(name: str) -> str:
    return os.path.join(_files_dir, name)


def _read_csv(name: str) -> Optional[pd.DataFrame]:
    path = _path(name)
    if not os.path.isfile(path):
        return None
    try:
        mtime = os.path.getmtime(path)
        cached = _cache.get(name)
        if cached and cached[0] == mtime:
            return cached[1]
        df = pd.read_csv(path)
        _cache[name] = (mtime, df)
        return df
    except Exception:
        return _cache.get(name, (0, None))[1]


def available() -> bool:
    return bool(_files_dir) and os.path.isfile(_path("ft_deals.csv"))


def read_account() -> Dict:
    df = _read_csv("ft_account.csv")
    if df is None or df.empty:
        return {}
    result = {}
    for _, row in df.iterrows():
        key = str(row["field"])
        val = str(row["value"])
        try:
            result[key] = float(val)
        except ValueError:
            result[key] = val
    return result


def read_positions() -> List[Dict]:
    df = _read_csv("ft_positions.csv")
    if df is None or df.empty:
        return []
    positions = []
    for _, r in df.iterrows():
        positions.append({
            "ticket": int(r["ticket"]),
            "symbol": str(r["symbol"]),
            "type": str(r["type"]),
            "volume": float(r["volume"]),
            "open_price": float(r["open_price"]),
            "sl": float(r["sl"]) if float(r["sl"]) != 0 else None,
            "tp": float(r["tp"]) if float(r["tp"]) != 0 else None,
            "profit": float(r["profit"]),
            "swap": float(r["swap"]),
            "commission": float(r.get("commission", 0)),
            "net_profit": round(float(r["profit"]) + float(r["swap"]) + float(r.get("commission", 0)), 2),
            "open_time": str(r["open_time"]),
            "magic": int(r["magic"]),
            "comment": str(r.get("comment", "")),
        })
    return positions


def _session_from_time(time_str: str) -> str:
    """Determine session from deal time string like '2026.04.21 16:05:38'."""
    try:
        parts = time_str.strip().split(" ")
        if len(parts) >= 2:
            h = int(parts[1].split(":")[0])
            if h < 7:
                return "ASIAN"
            if h < 13:
                return "LONDON"
            if h < 22:
                return "NEW_YORK"
            return "ASIAN"
    except Exception:
        pass
    return "UNKNOWN"


def get_closed_trades() -> List[Dict]:
    """Pair IN/OUT deals by position_id. Full trade details."""
    df = _read_csv("ft_deals.csv")
    if df is None or df.empty:
        return []
    entries = {}
    closed = []
    for _, r in df.iterrows():
        typ = str(r.get("type", ""))
        if typ in ("BALANCE", "OTHER_2"):
            continue
        entry = str(r.get("entry", ""))
        pid = int(r["position_id"])
        if entry == "IN":
            entries[pid] = r
        elif entry == "OUT":
            e = entries.get(pid)
            entry_price = float(e["price"]) if e is not None else 0
            entry_time = str(e["time"]) if e is not None else ""
            exit_price = float(r["price"])
            exit_time = str(r["time"])
            pnl = round(float(r["profit"]) + float(r["swap"]) + float(r["commission"]), 2)
            direction = str(e["type"]) if e is not None else ("BUY" if typ == "SELL" else "SELL")
            closed.append({
                "ticket": pid,
                "symbol": str(r["symbol"]),
                "direction": direction,
                "volume": float(r["volume"]),
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": pnl,
                "won": pnl > 0,
                "open_time": entry_time,
                "close_time": exit_time,
                "reason": str(r.get("reason", "")),
                "magic": int(r.get("magic", 0)),
                "comment": str(r.get("comment", "")),
                "entry_comment": str(e["comment"]) if e is not None else "",
                "session": _session_from_time(entry_time),
                "sl_distance": round(abs(exit_price - entry_price), 2) if entry_price else 0,
            })
    return closed


def get_today_pnl() -> Dict:
    """Today's P&L from deal history."""
    closed = get_closed_trades()
    if not closed:
        return {"pnl": 0.0, "trades": 0, "wins": 0, "losses": 0, "source": "csv_empty"}

    today = datetime.now(timezone.utc).strftime("%Y.%m.%d")
    pnl = 0.0
    trades = 0
    wins = 0
    losses = 0
    for t in closed:
        ct = t.get("close_time", "")
        if today in ct:
            pnl += t["pnl"]
            trades += 1
            if t["won"]:
                wins += 1
            else:
                losses += 1

    # If no today filter match, sum all (might be different date format)
    if trades == 0:
        for t in closed:
            pnl += t["pnl"]
            trades += 1
            if t["won"]:
                wins += 1
            else:
                losses += 1

    return {"pnl": round(pnl, 2), "trades": trades, "wins": wins, "losses": losses, "source": "csv_deals"}


def get_all_pnl() -> Dict:
    """All-time P&L from deal history."""
    closed = get_closed_trades()
    pnl = sum(t["pnl"] for t in closed)
    wins = sum(1 for t in closed if t["won"])
    losses = sum(1 for t in closed if not t["won"])
    return {"pnl": round(pnl, 2), "trades": len(closed), "wins": wins, "losses": losses}
