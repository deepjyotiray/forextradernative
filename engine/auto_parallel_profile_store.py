"""
Persistent learning store for AUTO parallelization profiles.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional
import json
import os


_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BACKTEST_DIR = os.path.join(_BASE_DIR, "backtests")
_PROFILE_PATH = os.path.join(_BACKTEST_DIR, "auto_parallel_profiles.json")


class AutoParallelProfileStore:
    def __init__(self, path: Optional[str] = None):
        self.path = path or _PROFILE_PATH
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def load_symbol_profile(self, symbol: str) -> Dict:
        payload = self._load()
        profiles = payload.get("profiles", {}) or {}
        profile = profiles.get(str(symbol).upper().strip(), {}) or {}
        return dict(profile)

    def update_symbol_profile(self, symbol: str, learning_rows: List[Dict]) -> Dict:
        rows = [row for row in (learning_rows or []) if float(row.get("processing_seconds", 0) or 0) > 0]
        if not rows:
            return self.load_symbol_profile(symbol)

        payload = self._load()
        profiles = payload.setdefault("profiles", {})
        key = str(symbol).upper().strip()
        current = dict(profiles.get(key, {}) or {})

        count = int(current.get("sample_count", 0) or 0)
        avg_ticks_per_second = float(current.get("avg_ticks_per_second", 0.0) or 0.0)
        avg_chunk_seconds = float(current.get("avg_chunk_seconds", 0.0) or 0.0)
        avg_chunk_ticks = float(current.get("avg_chunk_ticks", 0.0) or 0.0)
        avg_load_balance_ratio = float(current.get("avg_load_balance_ratio", 0.0) or 0.0)
        preferred_queue_factor = int(current.get("preferred_queue_factor", 0) or 0)

        observed_tps = sum(float(r.get("ticks_per_second", 0.0) or 0.0) for r in rows) / len(rows)
        observed_chunk_seconds = sum(float(r.get("processing_seconds", 0.0) or 0.0) for r in rows) / len(rows)
        observed_chunk_ticks = sum(float(r.get("actual_ticks", 0.0) or 0.0) for r in rows) / len(rows)

        actual_ticks = [max(1.0, float(r.get("actual_ticks", 0.0) or 0.0)) for r in rows]
        max_ticks = max(actual_ticks)
        min_ticks = min(actual_ticks)
        observed_balance_ratio = max_ticks / max(1.0, min_ticks)

        predicted_queue_factor = self._recommend_queue_factor(
            observed_chunk_seconds=observed_chunk_seconds,
            observed_balance_ratio=observed_balance_ratio,
            current_queue_factor=preferred_queue_factor or 2,
        )

        next_count = count + 1
        merged = {
            "sample_count": next_count,
            "avg_ticks_per_second": self._blend(avg_ticks_per_second, observed_tps, count),
            "avg_chunk_seconds": self._blend(avg_chunk_seconds, observed_chunk_seconds, count),
            "avg_chunk_ticks": self._blend(avg_chunk_ticks, observed_chunk_ticks, count),
            "avg_load_balance_ratio": self._blend(avg_load_balance_ratio, observed_balance_ratio, count),
            "preferred_queue_factor": predicted_queue_factor,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        profiles[key] = merged
        self._save(payload)
        return merged

    @staticmethod
    def _blend(current_value: float, new_value: float, sample_count: int) -> float:
        if sample_count <= 0:
            return float(new_value)
        alpha = min(0.35, 1.0 / float(sample_count + 1))
        return (1.0 - alpha) * float(current_value) + alpha * float(new_value)

    @staticmethod
    def _recommend_queue_factor(
        observed_chunk_seconds: float,
        observed_balance_ratio: float,
        current_queue_factor: int,
    ) -> int:
        queue_factor = max(2, int(current_queue_factor or 2))
        if observed_balance_ratio > 2.0 or observed_chunk_seconds > 90:
            queue_factor += 1
        elif observed_balance_ratio < 1.2 and observed_chunk_seconds < 25:
            queue_factor -= 1
        return max(2, min(6, queue_factor))

    def _load(self) -> Dict:
        if not os.path.exists(self.path):
            return {"profiles": {}}
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return {"profiles": {}}
            data.setdefault("profiles", {})
            return data
        except Exception:
            return {"profiles": {}}

    def _save(self, payload: Dict) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
