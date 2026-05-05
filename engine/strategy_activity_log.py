import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List

_IST = timezone(timedelta(hours=5, minutes=30))

_BASE_DIR = Path(__file__).resolve().parent.parent
_STRATEGY_LOG_FILES = {
    "SWING_ENGINE": _BASE_DIR / "swing_trades.json",
    "INTRADAY_ENGINE": _BASE_DIR / "intraday_trades.json",
}


def log_path_for_strategy(strategy: str) -> Path | None:
    return _STRATEGY_LOG_FILES.get(str(strategy or "").strip().upper())


def read_strategy_events(strategy: str) -> List[Dict[str, Any]]:
    path = log_path_for_strategy(strategy)
    if path is None or not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    return payload if isinstance(payload, list) else []


def append_strategy_event(strategy: str, event: Dict[str, Any]) -> None:
    path = log_path_for_strategy(strategy)
    if path is None:
        return
    rows = read_strategy_events(strategy)
    row = dict(event or {})
    row.setdefault("timestamp", datetime.now(_IST).isoformat())
    rows.append(row)
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def count_strategy_opens_today(strategy: str, now_utc: datetime | None = None) -> int:
    now = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)
    total = 0
    for row in read_strategy_events(strategy):
        if str(row.get("event") or "").upper() != "OPEN":
            continue
        timestamp = str(row.get("timestamp") or "").strip()
        try:
            opened_at = datetime.fromisoformat(timestamp.replace("Z", "+00:00")).astimezone(timezone.utc)
        except Exception:
            continue
        if opened_at.date() == now.date():
            total += 1
    return total
