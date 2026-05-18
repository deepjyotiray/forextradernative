from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List, Optional

_IST = timezone(timedelta(hours=5, minutes=30))
_GIT_TIMEOUT_SECONDS = 2.0
_VERSION_RECORD_FILE = "deployment_version_record.json"
_VERSION_HISTORY_FILE = "deployment_version_history.jsonl"
_UI_SCOPE_PATHS = [
    "dashboard.html",
    "strategies.html",
    "_dashboard_script.js",
]
_STRATEGY_SCOPE_PATHS = [
    "auto_trader.py",
    "engine/strategy_manager.py",
    "engine/smc_strategy.py",
    "engine/liquidity_sweep_ob_strategy.py",
    "engine/m15_scalp_deep_strategy.py",
    "engine/m15_zone_scalp_strategy.py",
    "engine/m15_zone_scalp_inverse_strategy.py",
    "engine/swing_engine_strategy.py",
    "engine/strategies",
    "engine/strategy_configs/__init__.py",
    "engine/strategy_configs/base.py",
    "engine/strategy_configs/trend_channel.py",
    "engine/strategy_configs/swing_engine.py",
    "engine/strategy_configs/smc_confluence.py",
    "engine/strategy_configs/m15_scalp_deep.py",
    "engine/strategy_configs/m15_zone_scalp.py",
    "engine/strategy_configs/m15_zone_scalp_inverse.py",
    "engine/strategy_configs/htf_short.py",
    "engine/strategy_configs/htf_long.py",
]


def _run_git(base_dir: Path, args: List[str]) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(base_dir),
            check=False,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return (result.stdout or "").strip()


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _fmt_ist(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    return dt.astimezone(_IST).strftime("%Y-%m-%d %H:%M:%S IST")


def _version_token(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    return dt.astimezone(_IST).strftime("%Y%m%d-%H%M%S")


def _latest_dt(*values: Optional[datetime]) -> Optional[datetime]:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return max(present)


def _rel(base_dir: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(base_dir.resolve()).as_posix()
    except Exception:
        return path.name


def _iter_scope_files(base_dir: Path, scope_paths: Iterable[str]) -> List[Path]:
    files: List[Path] = []
    seen = set()
    for scope_path in scope_paths:
        target = (base_dir / scope_path).resolve()
        if not target.exists():
            continue
        if target.is_file():
            if target not in seen:
                files.append(target)
                seen.add(target)
            continue
        for child in sorted(target.rglob("*")):
            if not child.is_file():
                continue
            if child in seen:
                continue
            files.append(child)
            seen.add(child)
    return files


def _latest_file_info(base_dir: Path, scope_paths: Iterable[str]) -> dict:
    files = _iter_scope_files(base_dir, scope_paths)
    latest_file = None
    latest_dt = None
    for file_path in files:
        try:
            modified = datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc)
        except OSError:
            continue
        if latest_dt is None or modified > latest_dt:
            latest_dt = modified
            latest_file = file_path
    return {
        "file_count": len(files),
        "latest_file": _rel(base_dir, latest_file) if latest_file else None,
        "latest_file_modified_utc": latest_dt.isoformat() if latest_dt else None,
        "latest_file_modified_ist": _fmt_ist(latest_dt),
    }


def _git_head_info(base_dir: Path, display_dt: Optional[datetime] = None) -> dict:
    raw = _run_git(base_dir, ["log", "-1", "--date=iso-strict", "--format=%H%x1f%h%x1f%cI%x1f%s"])
    branch = _run_git(base_dir, ["rev-parse", "--abbrev-ref", "HEAD"])
    dirty_raw = _run_git(base_dir, ["status", "--porcelain"])
    dirty = bool(dirty_raw)
    if not raw:
        return {
            "branch": None,
            "commit": None,
            "commit_short": None,
            "committed_at_utc": None,
            "committed_at_ist": None,
            "display_version": _fmt_ist(display_dt),
            "message": None,
            "dirty": dirty,
            "version": _version_token(display_dt),
        }
    parts = raw.split("\x1f", 3)
    parts += [""] * max(0, 4 - len(parts))
    commit, commit_short, committed_at_raw, message = parts[:4]
    committed_at = _parse_iso(committed_at_raw)
    version = commit_short or None
    if version and dirty:
        version = f"{version} + local"
    return {
        "branch": branch or None,
        "commit": commit or None,
        "commit_short": commit_short or None,
        "committed_at_utc": committed_at.isoformat() if committed_at else None,
        "committed_at_ist": _fmt_ist(committed_at),
        "display_version": _fmt_ist(display_dt or committed_at),
        "message": message or None,
        "dirty": dirty,
        "version": version,
    }


def _git_scope_info(base_dir: Path, label: str, scope_paths: Iterable[str], display_dt: Optional[datetime] = None) -> dict:
    path_args = list(scope_paths)
    raw = _run_git(
        base_dir,
        ["log", "-1", "--date=iso-strict", "--format=%H%x1f%h%x1f%cI%x1f%s", "--", *path_args],
    )
    dirty_raw = _run_git(base_dir, ["status", "--porcelain", "--", *path_args])
    latest_file_info = _latest_file_info(base_dir, scope_paths)
    dirty = bool(dirty_raw)
    latest_file_dt = _parse_iso(latest_file_info.get("latest_file_modified_utc"))
    if raw:
        parts = raw.split("\x1f", 3)
        parts += [""] * max(0, 4 - len(parts))
        commit, commit_short, committed_at_raw, message = parts[:4]
        committed_at = _parse_iso(committed_at_raw)
    else:
        commit = commit_short = message = None
        committed_at = None
    last_modified = latest_file_dt if dirty and latest_file_dt else committed_at or latest_file_dt
    if commit_short:
        version = commit_short
        if dirty:
            version = f"{version} + local"
    elif last_modified:
        version = last_modified.strftime("%Y%m%d-%H%M%S")
    else:
        version = None
    return {
        "label": label,
        "commit": commit or None,
        "commit_short": commit_short or None,
        "committed_at_utc": committed_at.isoformat() if committed_at else None,
        "committed_at_ist": _fmt_ist(committed_at),
        "last_modified_utc": last_modified.isoformat() if last_modified else None,
        "last_modified_ist": _fmt_ist(last_modified),
        "display_version": _fmt_ist(display_dt or last_modified or committed_at),
        "message": message or None,
        "dirty": dirty,
        "version": version,
        **latest_file_info,
    }


def _record_paths(base_dir: Path) -> dict:
    return {
        "current": str((base_dir / _VERSION_RECORD_FILE).resolve()),
        "history": str((base_dir / _VERSION_HISTORY_FILE).resolve()),
    }


def _record_payload(snapshot: dict, reason: str) -> dict:
    return {
        "recorded_at_utc": snapshot.get("captured_at_utc"),
        "recorded_at_ist": snapshot.get("captured_at_ist"),
        "reason": reason,
        "app_version": snapshot.get("app_version_label"),
        "ui_version": ((snapshot.get("ui") or {}).get("display_version")),
        "strategies_version": ((snapshot.get("strategies") or {}).get("display_version")),
        "git_commit": ((snapshot.get("git") or {}).get("commit")),
        "git_commit_short": ((snapshot.get("git") or {}).get("commit_short")),
        "git_branch": ((snapshot.get("git") or {}).get("branch")),
        "ui_latest_file": ((snapshot.get("ui") or {}).get("latest_file")),
        "strategies_latest_file": ((snapshot.get("strategies") or {}).get("latest_file")),
    }


def _persist_version_record(base_dir: Path, snapshot: dict, reason: str) -> None:
    record = _record_payload(snapshot, reason)
    record_paths = _record_paths(base_dir)
    current_path = Path(record_paths["current"])
    history_path = Path(record_paths["history"])
    current_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    should_append = True
    if history_path.exists():
        try:
            last_line = ""
            with history_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        last_line = line.strip()
            if last_line:
                previous = json.loads(last_line)
                should_append = previous != record
        except Exception:
            should_append = True
    if should_append:
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")


def capture_code_snapshot(base_dir: Path, as_runtime: bool = False, record_reason: str = "runtime_snapshot") -> dict:
    base_dir = Path(base_dir).resolve()
    captured_at = datetime.now(timezone.utc)
    version_dt = captured_at if as_runtime else None
    ui = _git_scope_info(base_dir, "UI", _UI_SCOPE_PATHS, display_dt=version_dt)
    strategies = _git_scope_info(base_dir, "Strategies", _STRATEGY_SCOPE_PATHS, display_dt=version_dt)
    current_version_dt = _latest_dt(
        _parse_iso(ui.get("last_modified_utc")),
        _parse_iso(strategies.get("last_modified_utc")),
    )
    app_version_dt = version_dt or current_version_dt or captured_at
    snapshot = {
        "captured_at_utc": captured_at.isoformat(),
        "captured_at_ist": _fmt_ist(captured_at),
        "app_version_label": _fmt_ist(app_version_dt),
        "git": _git_head_info(base_dir, display_dt=app_version_dt),
        "ui": ui,
        "strategies": strategies,
        "record_paths": _record_paths(base_dir),
    }
    if as_runtime:
        _persist_version_record(base_dir, snapshot, record_reason)
    return snapshot


def _scope_fingerprint(scope: Optional[dict]) -> tuple:
    scope = scope or {}
    return (
        scope.get("commit"),
        bool(scope.get("dirty")),
        scope.get("last_modified_utc"),
        scope.get("latest_file"),
    )


def compare_snapshots(runtime_snapshot: Optional[dict], current_snapshot: Optional[dict]) -> dict:
    runtime_snapshot = runtime_snapshot or {}
    current_snapshot = current_snapshot or {}
    differences: List[str] = []

    runtime_git = (runtime_snapshot.get("git") or {}).get("commit")
    current_git = (current_snapshot.get("git") or {}).get("commit")
    if runtime_git != current_git:
        differences.append("code")
    if _scope_fingerprint(runtime_snapshot.get("ui")) != _scope_fingerprint(current_snapshot.get("ui")):
        differences.append("ui")
    if _scope_fingerprint(runtime_snapshot.get("strategies")) != _scope_fingerprint(current_snapshot.get("strategies")):
        differences.append("strategies")

    in_sync = not differences
    status = "up_to_date" if in_sync else "restart_required"
    if in_sync:
        message = "Running code matches the latest files on disk."
    else:
        changed = ", ".join(differences)
        message = f"Newer {changed} changes are present on disk. A full service restart is needed to deploy them."
    return {
        "status": status,
        "in_sync": in_sync,
        "restart_required": not in_sync,
        "differences": differences,
        "message": message,
        "runtime": runtime_snapshot,
        "current": current_snapshot,
        "record_paths": runtime_snapshot.get("record_paths") or current_snapshot.get("record_paths") or {},
    }
