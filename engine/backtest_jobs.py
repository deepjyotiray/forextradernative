"""
Asynchronous backtest job manager.
"""
from __future__ import annotations

import os
import shutil
import threading
import traceback
import uuid
import json
import math
import time
import multiprocessing as mp
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import config as cfg
import pandas as pd
from .auto_parallel_planner import AutoParallelPlanner, detect_system_profile
from .auto_parallel_profile_store import AutoParallelProfileStore
from .backtest_data import BacktestDataProvider
from .backtest_sim import BacktestRequest, BacktestRunner, persist_backtest_artifacts


_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BACKTEST_DIR = os.path.join(_BASE_DIR, "backtests")
_JOBS_STATE_PATH = os.path.join(_BACKTEST_DIR, "jobs_state.json")
_TIME_SPLITS = {"DAILY", "WEEKLY", "MONTHLY", "QUARTERLY"}
_AUTO_SPLITS = {"AUTO"}
_PARALLEL_SPLITS = _TIME_SPLITS | _AUTO_SPLITS
_ALLOWED_SPLITS = {"NONE"} | _PARALLEL_SPLITS


def _next_split_boundary(current: datetime, mode: str) -> datetime:
    cur = current.astimezone(timezone.utc)
    if mode == "DAILY":
        return (cur + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    if mode == "WEEKLY":
        days_to_next_monday = 7 - cur.weekday()
        return (cur + timedelta(days=days_to_next_monday)).replace(hour=0, minute=0, second=0, microsecond=0)
    if mode == "MONTHLY":
        year = cur.year + (1 if cur.month == 12 else 0)
        month = 1 if cur.month == 12 else cur.month + 1
        return cur.replace(year=year, month=month, day=1, hour=0, minute=0, second=0, microsecond=0)
    if mode == "QUARTERLY":
        quarter_start_month = ((cur.month - 1) // 3) * 3 + 1
        next_q_month = quarter_start_month + 3
        year = cur.year
        if next_q_month > 12:
            next_q_month -= 12
            year += 1
        return cur.replace(year=year, month=next_q_month, day=1, hour=0, minute=0, second=0, microsecond=0)
    return cur + timedelta(days=1)


def _build_split_ranges(start_utc: datetime, end_utc: datetime, mode: str) -> List[Tuple[datetime, datetime]]:
    if end_utc <= start_utc:
        return []
    ranges: List[Tuple[datetime, datetime]] = []
    cur = start_utc.astimezone(timezone.utc)
    end = end_utc.astimezone(timezone.utc)
    while cur < end:
        nxt = _next_split_boundary(cur, mode)
        if nxt <= cur:
            nxt = cur + timedelta(days=1)
        if nxt > end:
            nxt = end
        ranges.append((cur, nxt))
        cur = nxt
    return ranges


def _run_chunk_worker(req_payload: Dict) -> Dict:
    import config as cfg
    provider = BacktestDataProvider()
    runner = BacktestRunner()
    try:
        req = BacktestRequest.from_payload(req_payload)
        if req.use_runtime_config:
            cfg.apply_runtime_profile(req.config_profile or cfg.get_active_profile("backtest"))
        chunk_index = int(req_payload.get("chunk_index", -1))
        progress_queue = req_payload.get("progress_queue")
        last_emit = 0.0

        def _progress(update: Dict):
            nonlocal last_emit
            if progress_queue is None:
                return
            now = time.monotonic()
            processed = int(update.get("processed_ticks", 0) or 0)
            total = int(update.get("total_ticks", 0) or 0)
            # Throttle cross-process progress chatter.
            if now - last_emit < 0.5 and not (total > 0 and processed >= total):
                return
            last_emit = now
            try:
                progress_queue.put_nowait({
                    "chunk_index": chunk_index,
                    "processed_ticks": processed,
                    "total_ticks": total,
                    "sim_time_utc": update.get("sim_time_utc"),
                })
            except Exception:
                return

        dataset = provider.load_dataset(
            symbol=req.symbol,
            start_utc=req.start_utc,
            end_utc=req.end_utc,
            cache_only=req.cache_only,
        )
        result = runner.run(
            req=req,
            dataset=dataset,
            progress_cb=_progress if progress_queue is not None else None,
            is_cancelled=None,
        )
        return {"ok": True, "result": result}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    finally:
        provider.disconnect()


@dataclass
class BacktestJob:
    job_id: str
    request: BacktestRequest
    status: str = "queued"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    processed_ticks: int = 0
    total_ticks: int = 0
    sim_time_utc: Optional[str] = None
    progress_pct: float = 0.0
    allocated_workers: int = 1
    chunk_states: List[Dict] = field(default_factory=list)
    error: Optional[str] = None
    result: Optional[Dict] = None
    artifacts: Optional[Dict] = None
    planning_meta: Dict = field(default_factory=dict)
    runtime_meta: Dict = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)
    _cancelled: bool = False
    _future: Optional[Future] = None

    def as_status_payload(self) -> Dict:
        split_mode = getattr(self.request, "split_mode", "NONE")
        chunk_mode = split_mode in _PARALLEL_SPLITS
        chunk_estimates = [
            int(c.get("estimated_ticks", 0) or 0)
            for c in self.chunk_states
            if int(c.get("estimated_ticks", 0) or 0) > 0
        ]
        status_counts: Dict[str, int] = {}
        for chunk in self.chunk_states:
            status = str(chunk.get("status", "unknown"))
            status_counts[status] = status_counts.get(status, 0) + 1
        runtime_meta = dict(self.runtime_meta or {})
        if self.chunk_states:
            runtime_meta["chunk_status_counts"] = status_counts
            runtime_meta.setdefault("chunk_count", len(self.chunk_states))
            if chunk_estimates:
                runtime_meta.setdefault("estimated_total_ticks", sum(chunk_estimates))
                runtime_meta.setdefault("min_estimated_ticks", min(chunk_estimates))
                runtime_meta.setdefault("max_estimated_ticks", max(chunk_estimates))
                runtime_meta.setdefault(
                    "load_balance_ratio",
                    round(max(chunk_estimates) / max(1, min(chunk_estimates)), 4),
                )
        return {
            "job_id": self.job_id,
            "status": self.status,
            "request": {
                "symbol": self.request.symbol,
                "strategy": self.request.strategy,
                "start_utc": self.request.start_utc.isoformat(),
                "end_utc": self.request.end_utc.isoformat(),
                "initial_balance": self.request.initial_balance,
                "config_profile": self.request.config_profile,
                "use_runtime_config": self.request.use_runtime_config,
                "max_ticks_per_batch": self.request.max_ticks_per_batch,
                "replay_max_points": self.request.replay_max_points,
                "fast_mode": self.request.fast_mode,
                "split_mode": self.request.split_mode,
                "parallel_workers": self.request.parallel_workers,
                "cache_only": self.request.cache_only,
            },
            "progress_pct": round(self.progress_pct, 2),
            "allocated_workers": int(self.allocated_workers or 1),
            "chunks": self.chunk_states,
            "processed_ticks": self.processed_ticks,
            "total_ticks": self.total_ticks,
            "processed_units": self.processed_ticks,
            "total_units": self.total_ticks,
            "progress_unit": "chunks" if chunk_mode else "ticks",
            "split_mode": split_mode,
            "sim_time_utc": self.sim_time_utc,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "error": self.error,
            "planning_meta": self.planning_meta or {},
            "runtime_meta": runtime_meta,
            "log_count": len(self.logs),
            "logs_tail": self.logs[-200:],
        }

    def cancel(self):
        self._cancelled = True
        if self.status in ("queued", "running"):
            self.status = "cancelled"

    def is_cancelled(self) -> bool:
        return self._cancelled


class BacktestJobManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._jobs: Dict[str, BacktestJob] = {}
        _job_slots = max(2, (os.cpu_count() or 2) // 2)
        self._executor = ThreadPoolExecutor(max_workers=_job_slots, thread_name_prefix="BacktestJob")
        os.makedirs(_BACKTEST_DIR, exist_ok=True)
        self._load_state()

    def create_job(self, req: BacktestRequest) -> BacktestJob:
        if req.symbol not in cfg.AVAILABLE_SYMBOLS:
            raise ValueError(f"Unsupported symbol: {req.symbol}")
        if req.strategy not in (
            "AUTO",
            "SMC_CONFLUENCE",
            "M15_SCALP_DEEP",
            "M15_ZONE_SCALP",
            "TREND_CHANNEL",
            "SWING_ENGINE",
            "HTF_LONG",
            "HTF_SHORT",
        ):
            raise ValueError(f"Unsupported strategy: {req.strategy}")
        if req.split_mode not in _ALLOWED_SPLITS:
            raise ValueError(f"Unsupported split_mode: {req.split_mode}")
        job_id = uuid.uuid4().hex[:12]
        job = BacktestJob(job_id=job_id, request=req)
        with self._lock:
            self._jobs[job_id] = job
            self._save_state_locked()
        fut = self._executor.submit(self._run_job, job_id)
        job._future = fut
        return job

    def get_job(self, job_id: str) -> Optional[BacktestJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self) -> Dict[str, Dict]:
        with self._lock:
            return {jid: job.as_status_payload() for jid, job in self._jobs.items()}

    def cancel_job(self, job_id: str) -> Optional[BacktestJob]:
        job = self.get_job(job_id)
        if not job:
            return None
        job.cancel()
        with self._lock:
            self._save_state_locked()
        return job

    def delete_job(self, job_id: str, delete_artifacts: bool = True, force: bool = False) -> bool:
        future = None
        with self._lock:
            job = self._jobs.get(job_id)
            if job and job.status in ("queued", "running"):
                if not force:
                    raise ValueError("Cannot delete a running/queued job. Cancel it first.")
                job.cancel()
                future = job._future
            removed = self._jobs.pop(job_id, None) is not None
            self._save_state_locked()
        if future is not None:
            try:
                future.cancel()
            except Exception:
                pass
        if delete_artifacts:
            job_dir = os.path.join(_BACKTEST_DIR, job_id)
            if os.path.isdir(job_dir):
                shutil.rmtree(job_dir, ignore_errors=True)
                removed = True
        return removed

    def _run_job(self, job_id: str):
        job = self.get_job(job_id)
        if not job:
            return
        runner = BacktestRunner()
        provider = BacktestDataProvider()
        try:
            job.status = "running"
            job.started_at = datetime.now(timezone.utc).isoformat()
            self._append_log(job, f"Job started | symbol={job.request.symbol} strategy={job.request.strategy}")
            self._append_log(
                job,
                f"Mode | split={job.request.split_mode} workers={job.request.parallel_workers} "
                f"fast_mode={job.request.fast_mode} cache_only={job.request.cache_only}",
            )
            with self._lock:
                self._save_state_locked()
            if job.request.split_mode in _PARALLEL_SPLITS:
                self._append_log(job, "Starting parallel split execution")
                result = self._run_job_parallel_chunks(job)
            else:
                job.allocated_workers = 1
                self._append_log(job, "Loading dataset")
                if job.request.use_runtime_config:
                    cfg.apply_runtime_profile(job.request.config_profile or cfg.get_active_profile("backtest"))
                dataset = provider.load_dataset(
                    symbol=job.request.symbol,
                    start_utc=job.request.start_utc,
                    end_utc=job.request.end_utc,
                    cache_only=job.request.cache_only,
                )
                self._append_log(
                    job,
                    f"Dataset loaded from {dataset.data_source} | ticks={len(dataset.ticks)} "
                    f"effective={dataset.effective_start_utc.isoformat()}->{dataset.effective_end_utc.isoformat()}",
                )
                job.total_ticks = len(dataset.ticks)
                if job.total_ticks <= 0:
                    raise RuntimeError("No ticks loaded for backtest range")

                def _progress(update: Dict):
                    job.processed_ticks = int(update.get("processed_ticks", job.processed_ticks))
                    job.total_ticks = int(update.get("total_ticks", job.total_ticks))
                    job.sim_time_utc = update.get("sim_time_utc")
                    if job.total_ticks > 0:
                        job.progress_pct = (job.processed_ticks / job.total_ticks) * 100.0
                    self._append_log(
                        job,
                        f"Progress | {job.processed_ticks}/{job.total_ticks} "
                        f"({job.progress_pct:.2f}%) sim_time={job.sim_time_utc}",
                    )

                result = runner.run(
                    req=job.request,
                    dataset=dataset,
                    progress_cb=_progress,
                    is_cancelled=job.is_cancelled,
                )
            if job.is_cancelled():
                job.status = "cancelled"
                job.ended_at = datetime.now(timezone.utc).isoformat()
                self._append_log(job, "Job cancelled")
                with self._lock:
                    self._save_state_locked()
                return
            artifacts = persist_backtest_artifacts(job.job_id, result, _BACKTEST_DIR)
            job.result = result
            job.artifacts = artifacts
            job.status = "completed"
            job.progress_pct = 100.0
            job.processed_ticks = job.total_ticks
            job.sim_time_utc = job.request.end_utc.isoformat()
            job.ended_at = datetime.now(timezone.utc).isoformat()
            self._append_log(job, "Job completed")
            with self._lock:
                self._save_state_locked()
        except Exception as e:
            if job.is_cancelled():
                job.status = "cancelled"
            else:
                job.status = "failed"
                job.error = f"{type(e).__name__}: {e}"
            job.ended_at = datetime.now(timezone.utc).isoformat()
            if not job.error and job.status == "failed":
                job.error = traceback.format_exc(limit=1)
            self._append_log(job, f"Job failed: {job.error}")
            with self._lock:
                self._save_state_locked()
        finally:
            provider.disconnect()

    def _run_job_parallel_chunks(self, job: BacktestJob) -> Dict:
        req = job.request
        self._append_log(
            job,
            f"Split request range: {req.start_utc.isoformat()} -> {req.end_utc.isoformat()} ({req.split_mode})",
        )
        worker_count = max(1, int(req.parallel_workers))
        cpu_count = max(1, os.cpu_count() or 1)
        env_override = int(os.getenv("BACKTEST_MAX_WORKERS", "0") or 0)
        worker_cap = max(1, env_override) if env_override > 0 else cpu_count
        plan_meta: Dict[str, object] = {}
        provider = BacktestDataProvider()
        try:
            if req.split_mode == "AUTO":
                planner = AutoParallelPlanner(provider, AutoParallelProfileStore())
                plan = planner.create_plan(
                    symbol=req.symbol,
                    start_utc=req.start_utc,
                    end_utc=req.end_utc,
                    requested_workers=worker_count,
                    worker_cap=worker_cap,
                )
                ranges = [(chunk.start_utc, chunk.end_utc) for chunk in plan.chunks]
                estimated_ticks_by_chunk = {
                    chunk.index: int(chunk.estimated_ticks) for chunk in plan.chunks
                }
                max_workers = int(plan.allocated_workers)
                plan_meta = {
                    "planning_mode": "auto_tick_density",
                    "density_source": plan.density_source,
                    "requested_workers": int(plan.requested_workers),
                    "allocated_workers": int(plan.allocated_workers),
                    "chunk_target_ticks": int(plan.chunk_target_ticks),
                    "estimated_total_ticks": int(plan.total_estimated_ticks),
                    "system_profile": dict(plan.system_profile),
                    "learned_profile": dict(plan.learned_profile or {}),
                    "timeline_density": list(plan.timeline_density),
                }
                self._append_log(
                    job,
                    f"Auto plan ready: chunks={len(ranges)} density_source={plan.density_source} "
                    f"target_ticks={plan.chunk_target_ticks} estimated_total_ticks={plan.total_estimated_ticks}",
                )
            else:
                ranges = _build_split_ranges(req.start_utc, req.end_utc, req.split_mode)
                density = provider.load_tick_density_profile(req.symbol, req.start_utc, req.end_utc)
                estimated_ticks_by_chunk = self._estimate_chunk_ticks_from_density(density, ranges)
                max_workers = min(worker_count, worker_cap)
                phase2_profile = detect_system_profile(
                    requested_workers=worker_count,
                    worker_cap=worker_cap,
                    total_minutes=max(1, int((req.end_utc - req.start_utc).total_seconds() // 60)),
                    total_estimated_ticks=int(density["estimated_ticks"].sum()) if not density.empty else 0,
                )
                plan_meta = {
                    "planning_mode": "time_split",
                    "density_source": str(density.attrs.get("density_source", "unavailable")),
                    "estimated_total_ticks": int(density["estimated_ticks"].sum()) if not density.empty else 0,
                    "system_profile": phase2_profile.as_dict(),
                    "timeline_density": AutoParallelPlanner._compress_density_profile(density),
                }
        finally:
            provider.disconnect()
        if not ranges:
            raise RuntimeError("No split ranges generated for requested period")

        job.allocated_workers = max_workers
        job.planning_meta = dict(plan_meta)
        job.runtime_meta = {
            "worker_cap": int(worker_cap),
            "cpu_count": int(cpu_count),
            "env_override": int(env_override),
            "requested_workers": int(worker_count),
            "allocated_workers": int(max_workers),
        }
        self._append_log(
            job,
            f"Split ranges generated: {len(ranges)} chunks, max_workers={max_workers} "
            f"(requested={worker_count}, cpu_count={cpu_count}, "
            f"worker_cap={worker_cap}, env_override={env_override})",
        )
        if len(ranges) == 1:
            self._append_log(
                job,
                "Split note: only one chunk generated for selected date range. "
                "To parallelize, increase the date window.",
            )
        job.total_ticks = len(ranges)
        job.processed_ticks = 0
        job.progress_pct = 0.0
        job.chunk_states = [
            {
                "index": i,
                "status": "queued",
                "progress_pct": 0.0,
                "sim_time_utc": None,
                "start_utc": s.isoformat(),
                "end_utc": e.isoformat(),
                "estimated_ticks": int(estimated_ticks_by_chunk.get(i, 0)),
            }
            for i, (s, e) in enumerate(ranges)
        ]
        with self._lock:
            self._save_state_locked()

        if req.cache_only:
            for idx, (chunk_start, chunk_end) in enumerate(ranges):
                coverage = provider.describe_cache_coverage(
                    symbol=req.symbol,
                    start_utc=chunk_start,
                    end_utc=chunk_end,
                )
                if coverage.get("has_tick_overlap"):
                    continue
                effective_start = coverage.get("effective_start_utc")
                effective_end = coverage.get("effective_end_utc")
                if coverage.get("cache_entry_found"):
                    raise RuntimeError(
                        "Cache-only mode: cached window exists but has no tick overlap for "
                        f"chunk {idx+1}/{len(ranges)} ({chunk_start.isoformat()} -> {chunk_end.isoformat()}). "
                        f"Effective cache coverage is {effective_start} -> {effective_end}."
                    )
                raise RuntimeError(
                    "Cache-only mode: no cached dataset covers "
                    f"chunk {idx+1}/{len(ranges)} ({chunk_start.isoformat()} -> {chunk_end.isoformat()}) "
                    f"for {req.symbol}."
                )

        futures_map = {}
        chunk_outputs: List[Dict] = []
        chunk_progress: Dict[int, float] = {}
        completed_chunks = 0
        last_state_save = time.monotonic()
        with mp.Manager() as mgr:
            progress_queue = mgr.Queue()
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                submit_order = sorted(
                    range(len(ranges)),
                    key=lambda idx: int(estimated_ticks_by_chunk.get(idx, 0)),
                    reverse=True,
                )
                for idx in submit_order:
                    chunk_start, chunk_end = ranges[idx]
                    payload = {
                        "symbol": req.symbol,
                        "strategy": req.strategy,
                        "start_utc": chunk_start.isoformat(),
                        "end_utc": chunk_end.isoformat(),
                        "initial_balance": req.initial_balance,
                        "config_profile": req.config_profile,
                        "use_runtime_config": req.use_runtime_config,
                        "max_ticks_per_batch": req.max_ticks_per_batch,
                        "replay_max_points": req.replay_max_points,
                        "fast_mode": req.fast_mode,
                        "split_mode": "NONE",
                        "parallel_workers": 1,
                        "cache_only": req.cache_only,
                        "chunk_index": idx,
                        "progress_queue": progress_queue,
                    }
                    self._append_log(
                        job,
                        f"Submitting chunk {idx+1}/{len(ranges)} "
                        f"{chunk_start.isoformat()} -> {chunk_end.isoformat()} "
                        f"est_ticks={int(estimated_ticks_by_chunk.get(idx, 0))}",
                    )
                    job.chunk_states[idx]["status"] = "running"
                    job.chunk_states[idx]["submitted_at"] = datetime.now(timezone.utc).isoformat()
                    fut = pool.submit(_run_chunk_worker, payload)
                    futures_map[fut] = (idx, chunk_start, chunk_end)
                pending = set(futures_map.keys())
                while pending:
                    if job.is_cancelled():
                        self._append_log(job, "Cancellation requested: terminating chunk workers")
                        for c in job.chunk_states:
                            if c.get("status") in ("queued", "running"):
                                c["status"] = "cancelled"
                        for p in pending:
                            p.cancel()
                        try:
                            if hasattr(pool, "terminate_workers"):
                                pool.terminate_workers()  # type: ignore[attr-defined]
                            else:
                                pool.shutdown(wait=False, cancel_futures=True)
                        except Exception:
                            pass
                        raise RuntimeError("Backtest cancelled")

                    # Drain worker progress updates.
                    got_progress = False
                    while True:
                        try:
                            upd = progress_queue.get_nowait()
                        except Exception:
                            break
                        got_progress = True
                        cidx = int(upd.get("chunk_index", -1))
                        p_ticks = int(upd.get("processed_ticks", 0) or 0)
                        t_ticks = int(upd.get("total_ticks", 0) or 0)
                        if cidx >= 0 and t_ticks > 0:
                            chunk_progress[cidx] = max(0.0, min(1.0, p_ticks / t_ticks))
                            job.chunk_states[cidx]["progress_pct"] = round(chunk_progress[cidx] * 100.0, 2)
                        sim_t = upd.get("sim_time_utc")
                        if sim_t and cidx >= 0:
                            job.chunk_states[cidx]["sim_time_utc"] = sim_t
                        times = [c.get("sim_time_utc") for c in job.chunk_states if c.get("sim_time_utc")]
                        if times:
                            job.sim_time_utc = max(times)

                    done_now = [f for f in list(pending) if f.done()]
                    for fut in done_now:
                        pending.remove(fut)
                        idx, chunk_start, chunk_end = futures_map[fut]
                        out = fut.result()
                        if not out.get("ok"):
                            job.chunk_states[idx]["status"] = "failed"
                            job.chunk_states[idx]["failed_at"] = datetime.now(timezone.utc).isoformat()
                            for other_idx, chunk in enumerate(job.chunk_states):
                                if other_idx == idx:
                                    continue
                                if chunk.get("status") in ("queued", "running"):
                                    chunk["status"] = "cancelled"
                            try:
                                if hasattr(pool, "terminate_workers"):
                                    pool.terminate_workers()  # type: ignore[attr-defined]
                                else:
                                    pool.shutdown(wait=False, cancel_futures=True)
                            except Exception:
                                pass
                            raise RuntimeError(
                                f"Chunk {idx+1}/{len(ranges)} failed "
                                f"({chunk_start.isoformat()}->{chunk_end.isoformat()}): {out.get('error', 'unknown error')}"
                            )
                        chunk_outputs.append(
                            {
                                "index": idx,
                                "start_utc": chunk_start,
                                "end_utc": chunk_end,
                                "result": out["result"],
                            }
                        )
                        self._append_log(
                            job,
                            f"Chunk done {idx+1}/{len(ranges)} "
                            f"{chunk_start.isoformat()} -> {chunk_end.isoformat()}",
                        )
                        completed_chunks += 1
                        chunk_progress[idx] = 1.0
                        job.chunk_states[idx]["status"] = "completed"
                        job.chunk_states[idx]["progress_pct"] = 100.0
                        job.chunk_states[idx]["sim_time_utc"] = chunk_end.isoformat()
                        job.chunk_states[idx]["completed_at"] = datetime.now(timezone.utc).isoformat()
                        actual_ticks = int((((out.get("result") or {}).get("meta", {}) or {}).get("ticks_replayed", 0)) or 0)
                        job.chunk_states[idx]["actual_ticks"] = actual_ticks
                        started_iso = job.chunk_states[idx].get("submitted_at")
                        finished_iso = job.chunk_states[idx].get("completed_at")
                        processing_seconds = 0.0
                        if started_iso and finished_iso:
                            try:
                                processing_seconds = max(
                                    0.0,
                                    (
                                        datetime.fromisoformat(str(finished_iso)) -
                                        datetime.fromisoformat(str(started_iso))
                                    ).total_seconds(),
                                )
                            except Exception:
                                processing_seconds = 0.0
                        job.chunk_states[idx]["processing_seconds"] = round(processing_seconds, 3)
                        if processing_seconds > 0 and actual_ticks > 0:
                            job.chunk_states[idx]["ticks_per_second"] = round(actual_ticks / processing_seconds, 3)
                        job.processed_ticks = completed_chunks
                        times = [c.get("sim_time_utc") for c in job.chunk_states if c.get("sim_time_utc")]
                        if times:
                            job.sim_time_utc = max(times)
                        got_progress = True

                    if job.total_ticks > 0:
                        partial = sum(chunk_progress.values())
                        job.progress_pct = min(100.0, (partial / job.total_ticks) * 100.0)

                    now = time.monotonic()
                    if got_progress or (now - last_state_save) >= 1.0:
                        with self._lock:
                            self._save_state_locked()
                        last_state_save = now

                    if pending:
                        time.sleep(0.2)

        chunk_outputs.sort(key=lambda x: x["index"])
        self._append_log(job, "Aggregating chunk results")
        result = self._aggregate_chunk_outputs(req, chunk_outputs, max_workers)
        if req.split_mode == "AUTO":
            learning_rows = [
                {
                    "estimated_ticks": int(chunk.get("estimated_ticks", 0) or 0),
                    "actual_ticks": int(chunk.get("actual_ticks", 0) or 0),
                    "processing_seconds": float(chunk.get("processing_seconds", 0.0) or 0.0),
                    "ticks_per_second": float(chunk.get("ticks_per_second", 0.0) or 0.0),
                }
                for chunk in job.chunk_states
                if float(chunk.get("processing_seconds", 0.0) or 0.0) > 0
            ]
            learned_profile = AutoParallelProfileStore().update_symbol_profile(req.symbol, learning_rows)
            job.planning_meta["learned_profile"] = dict(learned_profile or {})
            result.setdefault("meta", {})["learned_profile"] = dict(learned_profile or {})
            self._append_log(
                job,
                f"Auto profile updated: samples={int((learned_profile or {}).get('sample_count', 0) or 0)} "
                f"avg_tps={round(float((learned_profile or {}).get('avg_ticks_per_second', 0.0) or 0.0), 2)} "
                f"queue_factor={int((learned_profile or {}).get('preferred_queue_factor', 0) or 0)}",
            )
        if plan_meta:
            result.setdefault("meta", {}).update(plan_meta)
        result.setdefault("meta", {}).update({
            "runtime_meta": dict(job.runtime_meta or {}),
        })
        return result

    @staticmethod
    def _estimate_chunk_ticks_from_density(
        density: pd.DataFrame,
        ranges: List[Tuple[datetime, datetime]],
    ) -> Dict[int, int]:
        if density.empty or not ranges:
            return {}
        working = density.copy()
        working["bucket_start"] = pd.to_datetime(working["bucket_start"], utc=True)
        estimates: Dict[int, int] = {}
        for idx, (start_utc, end_utc) in enumerate(ranges):
            start_ts = pd.Timestamp(start_utc).tz_convert("UTC")
            end_ts = pd.Timestamp(end_utc).tz_convert("UTC")
            mask = (working["bucket_start"] >= start_ts) & (working["bucket_start"] < end_ts)
            estimates[idx] = int(working.loc[mask, "estimated_ticks"].sum())
        return estimates

    def _aggregate_chunk_outputs(self, req: BacktestRequest, outputs: List[Dict], workers_used: int) -> Dict:
        if not outputs:
            raise RuntimeError("No chunk results produced")
        init_balance = float(req.initial_balance)
        compounded = 1.0
        equity_curve: List[Dict] = []
        trades: List[Dict] = []
        wins = losses = total_trades = 0
        sum_win = 0.0
        sum_loss = 0.0
        skip_counter: Dict[str, int] = {}
        strategy_breakdown: Dict[str, Dict[str, int]] = {}
        chunk_rows: List[Dict] = []
        ticks_replayed = 0
        returns: List[float] = []

        for row in outputs:
            res = row["result"] or {}
            sm = res.get("summary", {}) or {}
            chunk_init = float(sm.get("initial_balance", init_balance) or init_balance)
            chunk_final = float(sm.get("final_balance", chunk_init) or chunk_init)
            chunk_ret = (chunk_final - chunk_init) / chunk_init if chunk_init > 0 else 0.0
            returns.append(chunk_ret)
            compounded *= (1.0 + chunk_ret)
            virt_eq = init_balance * compounded
            equity_curve.append(
                {
                    "time": row["end_utc"].isoformat(),
                    "equity": round(virt_eq, 2),
                    "balance": round(virt_eq, 2),
                }
            )

            chunk_trades = res.get("trades", []) or []
            for t in chunk_trades:
                item = dict(t)
                item["chunk_index"] = row["index"]
                item["chunk_start_utc"] = row["start_utc"].isoformat()
                item["chunk_end_utc"] = row["end_utc"].isoformat()
                trades.append(item)

            total_trades += int(sm.get("trades", 0) or 0)
            wins += int(sm.get("wins", 0) or 0)
            losses += int(sm.get("losses", 0) or 0)
            sum_win += float(sm.get("avg_win", 0.0) or 0.0) * int(sm.get("wins", 0) or 0)
            sum_loss += float(sm.get("avg_loss", 0.0) or 0.0) * int(sm.get("losses", 0) or 0)

            ds = res.get("decision_stats", {}) or {}
            for ent in ds.get("top_skip_reasons", []) or []:
                reason = str(ent.get("reason", "Unknown"))
                skip_counter[reason] = skip_counter.get(reason, 0) + int(ent.get("count", 0) or 0)

            sb = res.get("strategy_breakdown", {}) or {}
            for sname, vals in sb.items():
                if sname not in strategy_breakdown:
                    strategy_breakdown[sname] = {"taken": 0, "skipped": 0}
                strategy_breakdown[sname]["taken"] += int((vals or {}).get("taken", 0) or 0)
                strategy_breakdown[sname]["skipped"] += int((vals or {}).get("skipped", 0) or 0)

            ticks_replayed += int(((res.get("meta", {}) or {}).get("ticks_replayed", 0)) or 0)
            chunk_rows.append(
                {
                    "index": row["index"],
                    "start_utc": row["start_utc"].isoformat(),
                    "end_utc": row["end_utc"].isoformat(),
                    "return_pct": round(chunk_ret * 100.0, 4),
                    "summary": sm,
                }
            )

        final_balance = init_balance * compounded
        pnl = final_balance - init_balance
        win_rate = (wins / total_trades) if total_trades > 0 else 0.0
        avg_win = (sum_win / wins) if wins > 0 else 0.0
        avg_loss = (sum_loss / losses) if losses > 0 else 0.0
        expectancy = (pnl / total_trades) if total_trades > 0 else 0.0
        max_drawdown = self._max_drawdown_from_curve(equity_curve)

        sharpe_like = 0.0
        if returns:
            mean_r = sum(returns) / len(returns)
            var = sum((r - mean_r) ** 2 for r in returns) / len(returns)
            std = math.sqrt(var)
            sharpe_like = (mean_r / std) if std > 1e-12 else 0.0

        top_skip_reasons = [
            {"reason": k, "count": v}
            for k, v in sorted(skip_counter.items(), key=lambda x: x[1], reverse=True)[:20]
        ]
        total_decisions = sum(v for v in skip_counter.values()) + sum(v["taken"] for v in strategy_breakdown.values())
        trades_taken = sum(v["taken"] for v in strategy_breakdown.values())
        trades_skipped = sum(v["skipped"] for v in strategy_breakdown.values())

        return {
            "summary": {
                "initial_balance": round(init_balance, 2),
                "final_balance": round(final_balance, 2),
                "pnl": round(pnl, 2),
                "trades": int(total_trades),
                "wins": int(wins),
                "losses": int(losses),
                "win_rate": round(win_rate, 6),
                "avg_win": round(avg_win, 2),
                "avg_loss": round(avg_loss, 2),
                "expectancy": round(expectancy, 6),
                "max_drawdown": round(max_drawdown, 6),
                "sharpe_like": round(sharpe_like, 6),
            },
            "equity_curve": equity_curve,
            "trades": trades,
            "decision_stats": {
                "total_decisions": int(total_decisions),
                "trades_taken": int(trades_taken),
                "trades_skipped": int(trades_skipped),
                "top_skip_reasons": top_skip_reasons,
            },
            "strategy_breakdown": strategy_breakdown,
            "replay": {"capture_every": 1, "frames": [], "thoughts": []},
            "chunks": chunk_rows,
            "meta": {
                "symbol": req.symbol,
                "strategy": req.strategy,
                "requested_start_utc": req.start_utc.isoformat(),
                "requested_end_utc": req.end_utc.isoformat(),
                "initial_balance": req.initial_balance,
                "use_runtime_config": req.use_runtime_config,
                "max_ticks_per_batch": req.max_ticks_per_batch,
                "replay_max_points": req.replay_max_points,
                "split_mode": req.split_mode,
                "parallel_workers": req.parallel_workers,
                "cache_only": req.cache_only,
                "effective_start_utc": outputs[0]["start_utc"].isoformat(),
                "effective_end_utc": outputs[-1]["end_utc"].isoformat(),
                "config_profile": req.config_profile,
                "coverage_warnings": [],
                "config_snapshot_hash": outputs[0].get("result", {}).get("meta", {}).get("config_snapshot_hash", "parallel-split"),
                "config_snapshot": outputs[0].get("result", {}).get("meta", {}).get("config_snapshot", {}),
                "ticks_replayed": int(ticks_replayed),
                "fast_mode": req.fast_mode,
                "split_mode": req.split_mode,
                "parallel_workers": workers_used,
                "chunk_count": len(outputs),
                "growth_pct": round((compounded - 1.0) * 100.0, 6),
            },
        }

    @staticmethod
    def _max_drawdown_from_curve(curve: List[Dict]) -> float:
        if not curve:
            return 0.0
        peak = float(curve[0].get("equity", 0.0) or 0.0)
        max_dd = 0.0
        for pt in curve:
            eq = float(pt.get("equity", 0.0) or 0.0)
            if eq > peak:
                peak = eq
            if peak > 0:
                dd = (peak - eq) / peak
                if dd > max_dd:
                    max_dd = dd
        return max_dd

    def _request_to_dict(self, req: BacktestRequest) -> Dict:
        return {
            "symbol": req.symbol,
            "strategy": req.strategy,
            "start_utc": req.start_utc.isoformat(),
            "end_utc": req.end_utc.isoformat(),
            "initial_balance": req.initial_balance,
            "config_profile": req.config_profile,
            "use_runtime_config": req.use_runtime_config,
            "max_ticks_per_batch": req.max_ticks_per_batch,
            "replay_max_points": req.replay_max_points,
            "fast_mode": req.fast_mode,
            "split_mode": req.split_mode,
            "parallel_workers": req.parallel_workers,
            "cache_only": req.cache_only,
        }

    def _request_from_dict(self, payload: Dict) -> BacktestRequest:
        return BacktestRequest.from_payload(payload)

    def _job_to_state_dict(self, job: BacktestJob) -> Dict:
        return {
            "job_id": job.job_id,
            "request": self._request_to_dict(job.request),
            "status": job.status,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "ended_at": job.ended_at,
            "processed_ticks": job.processed_ticks,
            "total_ticks": job.total_ticks,
            "sim_time_utc": job.sim_time_utc,
            "progress_pct": job.progress_pct,
            "allocated_workers": int(job.allocated_workers or 1),
            "chunk_states": job.chunk_states,
            "error": job.error,
            "artifacts": job.artifacts,
            "planning_meta": job.planning_meta,
            "runtime_meta": job.runtime_meta,
            "cancelled": job._cancelled,
            "logs": job.logs[-2000:],
        }

    def _save_state_locked(self):
        payload = {"jobs": [self._job_to_state_dict(j) for j in self._jobs.values()]}
        with open(_JOBS_STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _load_state(self):
        if not os.path.exists(_JOBS_STATE_PATH):
            return
        try:
            with open(_JOBS_STATE_PATH, "r", encoding="utf-8") as f:
                payload = json.load(f)
            jobs = payload.get("jobs", [])
            recovered = False
            for row in jobs:
                req_payload = row.get("request") or {}
                try:
                    req = self._request_from_dict(req_payload)
                except Exception:
                    continue
                job = BacktestJob(
                    job_id=str(row.get("job_id")),
                    request=req,
                    status=str(row.get("status") or "queued"),
                    created_at=str(row.get("created_at") or datetime.now(timezone.utc).isoformat()),
                    started_at=row.get("started_at"),
                    ended_at=row.get("ended_at"),
                    processed_ticks=int(row.get("processed_ticks") or 0),
                    total_ticks=int(row.get("total_ticks") or 0),
                    sim_time_utc=row.get("sim_time_utc"),
                    progress_pct=float(row.get("progress_pct") or 0.0),
                    allocated_workers=int(row.get("allocated_workers") or 1),
                    chunk_states=list(row.get("chunk_states") or []),
                    error=row.get("error"),
                    result=None,
                    artifacts=row.get("artifacts"),
                    planning_meta=dict(row.get("planning_meta") or {}),
                    runtime_meta=dict(row.get("runtime_meta") or {}),
                    logs=list(row.get("logs") or []),
                    _cancelled=bool(row.get("cancelled", False)),
                )
                if job.status in ("queued", "running"):
                    job.status = "cancelled" if job._cancelled else "failed"
                    job.ended_at = datetime.now(timezone.utc).isoformat()
                    if not job.error:
                        job.error = "Recovered stale in-progress job after process restart"
                    self._append_log(job, "Recovered stale in-progress job from persisted state")
                    for chunk in job.chunk_states:
                        if chunk.get("status") in ("queued", "running"):
                            chunk["status"] = "cancelled" if job._cancelled else "failed"
                    recovered = True
                job._future = None
                self._jobs[job.job_id] = job
            if recovered:
                self._save_state_locked()
        except Exception:
            # Ignore corrupted state and continue with fresh in-memory registry.
            return

    def _append_log(self, job: BacktestJob, message: str):
        ts = datetime.now(timezone.utc).isoformat()
        line = f"{ts} | {message}"
        job.logs.append(line)
        if len(job.logs) > 4000:
            job.logs = job.logs[-2000:]


backtest_jobs = BacktestJobManager()
