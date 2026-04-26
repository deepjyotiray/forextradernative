"""
Deterministic auto-parallel planner backed by cached historical density data.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
import math
import os

import pandas as pd
from .auto_parallel_profile_store import AutoParallelProfileStore
try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None


@dataclass
class PlannedChunk:
    index: int
    start_utc: datetime
    end_utc: datetime
    estimated_ticks: int


@dataclass
class AutoParallelPlan:
    symbol: str
    split_mode: str
    density_source: str
    requested_workers: int
    allocated_workers: int
    total_estimated_ticks: int
    chunk_target_ticks: int
    system_profile: Dict[str, object]
    learned_profile: Dict[str, object]
    timeline_density: List[Dict[str, object]]
    chunks: List[PlannedChunk]


@dataclass
class SystemProfile:
    physical_cores: int
    logical_cores: int
    hyperthreading: bool
    available_ram_gb: float
    reserved_ram_gb: float
    memory_budget_gb: float
    estimated_ram_per_worker_gb: float
    cpu_recommended_workers: int
    memory_limited_workers: int
    recommended_workers: int
    chunk_queue_factor: int

    def as_dict(self) -> Dict[str, object]:
        return {
            "physical_cores": int(self.physical_cores),
            "logical_cores": int(self.logical_cores),
            "hyperthreading": bool(self.hyperthreading),
            "available_ram_gb": round(float(self.available_ram_gb), 2),
            "reserved_ram_gb": round(float(self.reserved_ram_gb), 2),
            "memory_budget_gb": round(float(self.memory_budget_gb), 2),
            "estimated_ram_per_worker_gb": round(float(self.estimated_ram_per_worker_gb), 2),
            "cpu_recommended_workers": int(self.cpu_recommended_workers),
            "memory_limited_workers": int(self.memory_limited_workers),
            "recommended_workers": int(self.recommended_workers),
            "chunk_queue_factor": int(self.chunk_queue_factor),
        }


class AutoParallelPlanner:
    """Builds chunk plans that try to equalize estimated ticks per chunk."""

    def __init__(self, provider, profile_store: Optional[AutoParallelProfileStore] = None):
        self.provider = provider
        self.profile_store = profile_store or AutoParallelProfileStore()

    def create_plan(
        self,
        symbol: str,
        start_utc: datetime,
        end_utc: datetime,
        requested_workers: int,
        worker_cap: Optional[int] = None,
    ) -> AutoParallelPlan:
        start_utc = self._ensure_utc(start_utc)
        end_utc = self._ensure_utc(end_utc)
        if end_utc <= start_utc:
            raise ValueError("end_utc must be after start_utc")

        cpu_count = max(1, os.cpu_count() or 1)
        hard_cap = max(1, worker_cap or cpu_count)
        requested = max(1, int(requested_workers or 1))

        density = self.provider.load_tick_density_profile(symbol, start_utc, end_utc)
        density_source = str(density.attrs.get("density_source", "unavailable"))
        total_minutes = max(1, len(density)) if not density.empty else max(
            1, int((end_utc - start_utc).total_seconds() // 60)
        )
        total_estimated_ticks = int(density["estimated_ticks"].sum()) if not density.empty else 0
        learned_profile = self.profile_store.load_symbol_profile(symbol)
        system_profile = detect_system_profile(
            requested_workers=requested,
            worker_cap=hard_cap,
            total_minutes=total_minutes,
            total_estimated_ticks=total_estimated_ticks,
            learned_profile=learned_profile,
        )
        allocated = int(system_profile.recommended_workers)

        if density.empty:
            chunks = self._build_even_time_chunks(start_utc, end_utc, allocated)
            return AutoParallelPlan(
                symbol=symbol,
                split_mode="AUTO",
                density_source=density_source,
                requested_workers=requested,
                allocated_workers=min(allocated, len(chunks)),
                total_estimated_ticks=0,
                chunk_target_ticks=0,
                system_profile=system_profile.as_dict(),
                learned_profile=learned_profile,
                timeline_density=self._build_chunk_density_fallback(chunks),
                chunks=chunks,
            )

        desired_chunks = self._desired_chunk_count(total_minutes, allocated, system_profile.chunk_queue_factor)
        chunk_target_ticks = self._select_chunk_target_ticks(
            total_estimated_ticks=total_estimated_ticks,
            desired_chunks=desired_chunks,
            learned_profile=learned_profile,
        )

        chunks = self._build_density_balanced_chunks(
            density=density,
            start_utc=start_utc,
            end_utc=end_utc,
            desired_chunks=desired_chunks,
            chunk_target_ticks=chunk_target_ticks,
        )
        chunks = self._merge_tiny_tail(chunks, chunk_target_ticks)
        timeline_density = self._compress_density_profile(density)

        return AutoParallelPlan(
            symbol=symbol,
            split_mode="AUTO",
            density_source=density_source,
            requested_workers=requested,
            allocated_workers=min(allocated, len(chunks)),
            total_estimated_ticks=total_estimated_ticks,
            chunk_target_ticks=chunk_target_ticks,
            system_profile=system_profile.as_dict(),
            learned_profile=learned_profile,
            timeline_density=timeline_density,
            chunks=chunks,
        )

    @staticmethod
    def _ensure_utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    @staticmethod
    def _desired_chunk_count(total_minutes: int, allocated_workers: int, chunk_queue_factor: int) -> int:
        if total_minutes <= 180:
            return 1
        max_by_granularity = max(1, total_minutes // 30)
        desired = max(allocated_workers, allocated_workers * max(1, chunk_queue_factor))
        return max(1, min(desired, max_by_granularity))

    @staticmethod
    def _select_chunk_target_ticks(
        total_estimated_ticks: int,
        desired_chunks: int,
        learned_profile: Dict[str, object],
    ) -> int:
        baseline_target = max(1, math.ceil(total_estimated_ticks / max(1, desired_chunks)))
        avg_tps = float(learned_profile.get("avg_ticks_per_second", 0.0) or 0.0)
        avg_chunk_seconds = float(learned_profile.get("avg_chunk_seconds", 0.0) or 0.0)
        sample_count = int(learned_profile.get("sample_count", 0) or 0)
        if sample_count < 1 or avg_tps <= 0:
            return baseline_target
        desired_runtime_sec = avg_chunk_seconds if avg_chunk_seconds > 0 else 45.0
        desired_runtime_sec = min(90.0, max(25.0, desired_runtime_sec))
        learned_target = max(1, int(avg_tps * desired_runtime_sec))
        return max(1, int((baseline_target + learned_target) / 2.0))

    def _build_even_time_chunks(
        self,
        start_utc: datetime,
        end_utc: datetime,
        allocated_workers: int,
    ) -> List[PlannedChunk]:
        duration_seconds = max(1.0, (end_utc - start_utc).total_seconds())
        chunk_count = max(1, min(allocated_workers, int(duration_seconds // 1800) or 1))
        out: List[PlannedChunk] = []
        cursor = start_utc
        remaining = chunk_count
        for index in range(chunk_count):
            if index == chunk_count - 1:
                next_boundary = end_utc
            else:
                seconds_left = (end_utc - cursor).total_seconds()
                next_boundary = cursor + timedelta(seconds=seconds_left / remaining)
            out.append(
                PlannedChunk(
                    index=index,
                    start_utc=cursor,
                    end_utc=next_boundary,
                    estimated_ticks=0,
                )
            )
            cursor = next_boundary
            remaining -= 1
        return out

    def _build_density_balanced_chunks(
        self,
        density: pd.DataFrame,
        start_utc: datetime,
        end_utc: datetime,
        desired_chunks: int,
        chunk_target_ticks: int,
    ) -> List[PlannedChunk]:
        minutes = density.to_dict("records")
        total_ticks = int(density["estimated_ticks"].sum())
        ticks_remaining = total_ticks
        chunks_remaining = max(1, desired_chunks)
        chunk_start = start_utc
        chunk_ticks = 0
        chunk_minutes = 0
        min_chunk_minutes = 30
        out: List[PlannedChunk] = []

        for idx, row in enumerate(minutes):
            bucket_start = self._ensure_utc(pd.Timestamp(row["bucket_start"]).to_pydatetime())
            bucket_end = min(end_utc, bucket_start + timedelta(minutes=1))
            minute_ticks = int(row.get("estimated_ticks", 0) or 0)
            chunk_ticks += minute_ticks
            chunk_minutes += 1
            ticks_remaining -= minute_ticks

            is_last_minute = idx == len(minutes) - 1
            next_target = math.ceil(ticks_remaining / max(1, chunks_remaining - 1)) if chunks_remaining > 1 else 0
            enough_ticks = chunk_ticks >= chunk_target_ticks
            enough_time = chunk_minutes >= min_chunk_minutes
            keep_balance = chunk_ticks >= max(1, next_target)

            should_close = is_last_minute or (
                chunks_remaining > 1 and enough_time and (enough_ticks or keep_balance)
            )
            if not should_close:
                continue

            out.append(
                PlannedChunk(
                    index=len(out),
                    start_utc=chunk_start,
                    end_utc=bucket_end,
                    estimated_ticks=max(0, chunk_ticks),
                )
            )
            chunk_start = bucket_end
            chunk_ticks = 0
            chunk_minutes = 0
            chunks_remaining -= 1

        if not out:
            out.append(
                PlannedChunk(
                    index=0,
                    start_utc=start_utc,
                    end_utc=end_utc,
                    estimated_ticks=max(0, total_ticks),
                )
            )

        if out[-1].end_utc < end_utc:
            last = out[-1]
            out[-1] = PlannedChunk(
                index=last.index,
                start_utc=last.start_utc,
                end_utc=end_utc,
                estimated_ticks=last.estimated_ticks,
            )

        return out

    @staticmethod
    def _merge_tiny_tail(chunks: List[PlannedChunk], chunk_target_ticks: int) -> List[PlannedChunk]:
        if len(chunks) < 2:
            return chunks
        tail = chunks[-1]
        if tail.estimated_ticks > max(1, int(chunk_target_ticks * 0.2)):
            return chunks

        prev = chunks[-2]
        merged = PlannedChunk(
            index=prev.index,
            start_utc=prev.start_utc,
            end_utc=tail.end_utc,
            estimated_ticks=prev.estimated_ticks + tail.estimated_ticks,
        )
        merged_chunks = chunks[:-2] + [merged]
        return [
            PlannedChunk(
                index=i,
                start_utc=chunk.start_utc,
                end_utc=chunk.end_utc,
                estimated_ticks=chunk.estimated_ticks,
            )
            for i, chunk in enumerate(merged_chunks)
        ]

    @staticmethod
    def _compress_density_profile(
        density: pd.DataFrame,
        max_points: int = 240,
    ) -> List[Dict[str, object]]:
        if density.empty:
            return []
        point_count = len(density)
        if point_count <= max_points:
            return [
                {
                    "bucket_start": pd.Timestamp(row["bucket_start"]).isoformat(),
                    "estimated_ticks": int(row.get("estimated_ticks", 0) or 0),
                }
                for row in density.to_dict("records")
            ]
        bucket_size = max(1, math.ceil(point_count / max_points))
        rows: List[Dict[str, object]] = []
        for start_idx in range(0, point_count, bucket_size):
            window = density.iloc[start_idx:start_idx + bucket_size]
            if window.empty:
                continue
            rows.append(
                {
                    "bucket_start": pd.Timestamp(window.iloc[0]["bucket_start"]).isoformat(),
                    "estimated_ticks": int(window["estimated_ticks"].sum()),
                }
            )
        return rows

    @staticmethod
    def _build_chunk_density_fallback(chunks: List[PlannedChunk]) -> List[Dict[str, object]]:
        return [
            {
                "bucket_start": chunk.start_utc.isoformat(),
                "estimated_ticks": int(chunk.estimated_ticks),
            }
            for chunk in chunks
        ]


def detect_system_profile(
    requested_workers: int,
    worker_cap: Optional[int] = None,
    total_minutes: int = 0,
    total_estimated_ticks: int = 0,
    respect_duration_cap: bool = True,
    learned_profile: Optional[Dict[str, object]] = None,
) -> SystemProfile:
    learned_profile = dict(learned_profile or {})
    logical_cores = max(1, os.cpu_count() or 1)
    physical_cores = logical_cores
    available_ram_gb = 8.0

    if psutil is not None:
        try:
            logical_cores = max(1, int(psutil.cpu_count(logical=True) or logical_cores))
        except Exception:
            logical_cores = max(1, os.cpu_count() or 1)
        try:
            physical_cores = max(1, int(psutil.cpu_count(logical=False) or logical_cores))
        except Exception:
            physical_cores = logical_cores
        try:
            available_ram_gb = float(psutil.virtual_memory().available) / float(1024 ** 3)
        except Exception:
            available_ram_gb = 8.0

    hyperthreading = logical_cores > physical_cores
    hard_cap = max(1, int(worker_cap or logical_cores))
    requested = max(1, int(requested_workers or 1))

    if available_ram_gb >= 24:
        reserved_ram_gb = 3.0
    elif available_ram_gb >= 12:
        reserved_ram_gb = 2.0
    else:
        reserved_ram_gb = 1.0
    memory_budget_gb = max(0.5, available_ram_gb - reserved_ram_gb)

    estimated_ram_per_worker_gb = 0.6
    if total_minutes >= 24 * 60:
        estimated_ram_per_worker_gb += 0.15
    if total_minutes >= 7 * 24 * 60:
        estimated_ram_per_worker_gb += 0.1
    if total_estimated_ticks >= 2_000_000:
        estimated_ram_per_worker_gb += 0.1
    if total_estimated_ticks >= 8_000_000:
        estimated_ram_per_worker_gb += 0.1
    memory_limited_workers = max(1, int(memory_budget_gb // max(0.25, estimated_ram_per_worker_gb)))

    cpu_recommended_workers = logical_cores
    if logical_cores >= 4:
        cpu_recommended_workers = logical_cores - 1
    if hyperthreading and logical_cores >= 8:
        cpu_recommended_workers = max(physical_cores, logical_cores - 1)

    if not respect_duration_cap:
        duration_cap = cpu_recommended_workers
    elif total_minutes <= 180:
        duration_cap = 1
    elif total_minutes <= 6 * 60:
        duration_cap = 2
    elif total_minutes <= 12 * 60:
        duration_cap = min(4, cpu_recommended_workers)
    else:
        duration_cap = cpu_recommended_workers

    recommended_workers = min(requested, hard_cap, cpu_recommended_workers, memory_limited_workers, duration_cap)
    recommended_workers = max(1, recommended_workers)

    learned_queue_factor = int(learned_profile.get("preferred_queue_factor", 0) or 0)
    if learned_queue_factor > 0:
        chunk_queue_factor = max(2, min(6, learned_queue_factor))
    elif total_minutes >= 7 * 24 * 60 or recommended_workers >= 12:
        chunk_queue_factor = 4
    elif total_minutes >= 24 * 60 or recommended_workers >= 6:
        chunk_queue_factor = 3
    else:
        chunk_queue_factor = 2

    return SystemProfile(
        physical_cores=physical_cores,
        logical_cores=logical_cores,
        hyperthreading=hyperthreading,
        available_ram_gb=available_ram_gb,
        reserved_ram_gb=reserved_ram_gb,
        memory_budget_gb=memory_budget_gb,
        estimated_ram_per_worker_gb=estimated_ram_per_worker_gb,
        cpu_recommended_workers=cpu_recommended_workers,
        memory_limited_workers=memory_limited_workers,
        recommended_workers=recommended_workers,
        chunk_queue_factor=chunk_queue_factor,
    )
