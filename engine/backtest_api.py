"""
Backtest API routes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import json
import os

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse

import config as cfg
from .auto_parallel_planner import detect_system_profile
from .backtest_data import BacktestDataProvider
from .backtest_jobs import backtest_jobs
from .backtest_sim import BacktestRequest


router = APIRouter(prefix="/backtest", tags=["backtest"])
_BASE_DIR = Path(__file__).resolve().parent.parent
_BACKTEST_DIR = _BASE_DIR / "backtests"


@router.get("", response_class=HTMLResponse)
@router.get("/", response_class=HTMLResponse)
async def get_backtest_page():
    base_dir = Path(__file__).resolve().parent.parent
    page = base_dir / "backtest.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail="Backtest page not found")
    return HTMLResponse(page.read_text(encoding="utf-8"))


@router.get("/presets")
async def get_presets():
    cpu_count = max(1, (os.cpu_count() or 1))
    system_profile = detect_system_profile(
        requested_workers=cpu_count,
        worker_cap=cpu_count,
        respect_duration_cap=False,
    )
    return {
        "symbols": cfg.AVAILABLE_SYMBOLS,
        "strategies": [
            "AUTO",
            "SMC_CONFLUENCE",
            "M15_SCALP_DEEP",
            "M15_ZONE_SCALP",
            "TREND_CHANNEL",
            "SWING_ENGINE",
            "HTF_LONG",
            "HTF_SHORT",
        ],
        "profiles": cfg.list_runtime_profiles("backtest"),
        "defaults": {
            "symbol": cfg.SYMBOL,
            "strategy": cfg.DEFAULT_STRATEGY if cfg.DEFAULT_STRATEGY in (
                "AUTO",
                "SMC_CONFLUENCE",
                "M15_SCALP_DEEP",
                "M15_ZONE_SCALP",
                "TREND_CHANNEL",
                "SWING_ENGINE",
                "HTF_LONG",
                "HTF_SHORT",
            ) else "AUTO",
            "initial_balance": 10000.0,
            "config_profile": cfg.get_active_profile("backtest"),
            "use_runtime_config": True,
            "max_ticks_per_batch": 2000,
            "replay_max_points": 12000,
            "fast_mode": False,
            "split_mode": "AUTO",
            "parallel_workers": system_profile.recommended_workers,
            "cache_only": True,
        },
        "fast_mode_defaults": {
            "max_ticks_per_batch": 50000,
            "replay_max_points": 1500,
        },
        "resource_defaults": {
            "cpu_count": cpu_count,
            "recommended_workers": system_profile.recommended_workers,
            "max_workers": cpu_count,
            "system_profile": system_profile.as_dict(),
        },
        "split_modes": ["AUTO", "NONE", "DAILY", "WEEKLY", "MONTHLY", "QUARTERLY"],
        "risk_config": {
            "MAX_POSITIONS": cfg.MAX_POSITIONS,
            "MAX_RISK_PCT": cfg.MAX_RISK_PCT,
            "MIN_TRADE_COOLDOWN": cfg.MIN_TRADE_COOLDOWN,
        },
    }


@router.get("/data-availability")
async def get_data_availability(symbol: Optional[str] = Query(default=None)):
    provider = BacktestDataProvider()
    return provider.get_cache_availability(symbol=symbol)


@router.post("/run")
async def run_backtest(payload: Dict[str, Any]):
    try:
        req = BacktestRequest.from_payload(payload)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid payload: {e}")
    try:
        job = backtest_jobs.create_job(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"job_id": job.job_id, **job.as_status_payload()}


@router.get("/jobs")
async def list_jobs():
    return {"jobs": list(backtest_jobs.list_jobs().values())}


@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    job = backtest_jobs.get_job(job_id)
    if not job:
        persisted = _load_persisted_result(job_id)
        if persisted is not None:
            meta = persisted.get("meta", {})
            return {
                "job_id": job_id,
                "status": "completed",
                "request": {
                    "symbol": meta.get("symbol"),
                    "strategy": meta.get("strategy"),
                    "start_utc": meta.get("requested_start_utc"),
                    "end_utc": meta.get("requested_end_utc"),
                    "initial_balance": meta.get("initial_balance"),
                    "config_profile": meta.get("config_profile"),
                    "use_runtime_config": meta.get("use_runtime_config", True),
                    "max_ticks_per_batch": meta.get("max_ticks_per_batch"),
                    "replay_max_points": meta.get("replay_max_points"),
                    "fast_mode": meta.get("fast_mode", False),
                    "split_mode": meta.get("split_mode", "NONE"),
                    "parallel_workers": meta.get("parallel_workers", 1),
                    "cache_only": meta.get("cache_only", True),
                },
                "progress_pct": 100.0,
                "allocated_workers": int(meta.get("parallel_workers", 1) or 1),
                "processed_ticks": int(meta.get("ticks_replayed", 0) or 0),
                "total_ticks": int(meta.get("ticks_replayed", 0) or 0),
                "sim_time_utc": meta.get("effective_end_utc"),
                "started_at": None,
                "ended_at": None,
                "error": None,
            }
        return {
            "job_id": job_id,
            "status": "deleted",
            "progress_pct": 0.0,
            "allocated_workers": 0,
            "processed_ticks": 0,
            "total_ticks": 0,
            "sim_time_utc": None,
            "started_at": None,
            "ended_at": None,
            "error": "Job not found (deleted or expired)",
        }
    return job.as_status_payload()


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    job = backtest_jobs.cancel_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.as_status_payload()


@router.delete("/jobs/{job_id}")
async def delete_job(job_id: str, force: bool = Query(default=True)):
    try:
        removed = backtest_jobs.delete_job(job_id, delete_artifacts=True, force=force)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not removed:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id, "deleted": True}


@router.post("/jobs/{job_id}/delete")
async def delete_job_post(job_id: str, force: bool = Query(default=True)):
    # Fallback for environments where DELETE may be blocked by proxy/security policy.
    return await delete_job(job_id=job_id, force=force)


@router.get("/jobs/{job_id}/results")
async def get_job_results(job_id: str):
    job = backtest_jobs.get_job(job_id)
    if not job:
        persisted = _load_persisted_result(job_id)
        if persisted is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return {
            "status": "completed",
            "summary": persisted.get("summary", {}),
            "equity_curve": persisted.get("equity_curve", []),
            "trades": persisted.get("trades", []),
            "replay": persisted.get("replay", {}),
            "decision_stats": persisted.get("decision_stats", {}),
            "strategy_breakdown": persisted.get("strategy_breakdown", {}),
            "meta": persisted.get("meta", {}),
            "artifacts": {
                "result_json": str((_BACKTEST_DIR / job_id / "result.json").resolve()),
                "trades_csv": str((_BACKTEST_DIR / job_id / "trades.csv").resolve()),
                "equity_curve_csv": str((_BACKTEST_DIR / job_id / "equity_curve.csv").resolve()),
            },
        }
    if job.status == "failed":
        raise HTTPException(status_code=400, detail=job.error or "Job failed")
    if job.status != "completed":
        return {
            "status": job.status,
            "message": "Results not available yet",
            **job.as_status_payload(),
        }
    if not job.result:
        persisted = _load_persisted_result(job_id)
        if persisted is not None:
            job.result = persisted
    if not job.result:
        return {
            "status": job.status,
            "message": "Results file not available",
            **job.as_status_payload(),
        }
    return {
        "status": job.status,
        "summary": job.result.get("summary", {}),
        "equity_curve": job.result.get("equity_curve", []),
        "trades": job.result.get("trades", []),
        "replay": job.result.get("replay", {}),
        "decision_stats": job.result.get("decision_stats", {}),
        "strategy_breakdown": job.result.get("strategy_breakdown", {}),
        "meta": job.result.get("meta", {}),
        "artifacts": job.artifacts or {},
    }


@router.get("/jobs/{job_id}/replay")
async def get_job_replay(
    job_id: str,
    start: int = Query(0, ge=0),
    limit: int = Query(1000, ge=1, le=5000),
):
    job = backtest_jobs.get_job(job_id)
    if not job:
        persisted = _load_persisted_result(job_id)
        if persisted is None:
            raise HTTPException(status_code=404, detail="Job not found")
        replay = persisted.get("replay", {}) or {}
        frames = replay.get("frames", []) or []
        thoughts = replay.get("thoughts", []) or []
        end = min(len(frames), start + limit)
        start_th = min(len(thoughts), start + limit)
        return {
            "status": "completed",
            "capture_every": replay.get("capture_every", 1),
            "start": start,
            "end": end,
            "total_frames": len(frames),
            "total_thoughts": len(thoughts),
            "frames": frames[start:end],
            "thoughts": thoughts[start:start_th],
        }
    if job.status != "completed" or not job.result:
        return {"status": job.status, "frames": [], "thoughts": [], "total_frames": 0, "total_thoughts": 0}

    replay = job.result.get("replay", {}) or {}
    frames = replay.get("frames", []) or []
    thoughts = replay.get("thoughts", []) or []
    end = min(len(frames), start + limit)
    start_th = min(len(thoughts), start + limit)
    return {
        "status": job.status,
        "capture_every": replay.get("capture_every", 1),
        "start": start,
        "end": end,
        "total_frames": len(frames),
        "total_thoughts": len(thoughts),
        "frames": frames[start:end],
        "thoughts": thoughts[start:start_th],
    }


@router.get("/jobs/{job_id}/logs")
async def get_job_logs(
    job_id: str,
    start: int = Query(0, ge=0),
    limit: int = Query(500, ge=1, le=5000),
):
    job = backtest_jobs.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    logs = job.logs or []
    end = min(len(logs), start + limit)
    return {
        "job_id": job_id,
        "status": job.status,
        "start": start,
        "end": end,
        "total_logs": len(logs),
        "logs": logs[start:end],
    }


def _load_persisted_result(job_id: str) -> Optional[Dict[str, Any]]:
    path = _BACKTEST_DIR / job_id / "result.json"
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None
