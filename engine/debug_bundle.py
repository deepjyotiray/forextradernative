from __future__ import annotations

import json
import os
import shutil
import sqlite3
import tempfile
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .deployment_metadata import capture_code_snapshot

_IST = timezone(timedelta(hours=5, minutes=30))

_DEBUG_EXPORT_DIR = "debug_exports"
_RECENT_LOGS_DIR = "recent_logs_since_restart"
_TEXT_TAIL_BYTES = 4 * 1024 * 1024
_JSONL_TAIL_LINES = 20000
_RECENT_DAYS_DEFAULT = 7
_FULL_COPY_LOG_MAX_BYTES = 16 * 1024 * 1024

_FULL_COPY_ROOT_FILES = [
    "deployment_version_record.json",
    "deployment_version_history.jsonl",
    "runtime_config.json",
    "config_profiles.json",
    "anti_starvation_state.json",
    "risk_control_state.json",
    "trade_pacing_state.json",
    "open_trades.json",
    "xgb_model.json",
    "xgb_model.json.meta",
    "closed_trades.jsonl",
    "trade_history.json",
    "deploy_watcher.log",
    "startup_stdout.log",
    "startup_stderr.log",
]

_TAIL_COPY_ROOT_FILES = [
    "trader.log",
    "trade_attribution.jsonl",
    "trade_decisions.jsonl",
    "launcher_auto_deploy.log",
    "pnl_validation.log",
    "cf_log.txt",
]

_SQLITE_FILES = [
    "orders.db",
    "analytics.db",
]


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _bundle_token(now_utc: datetime) -> str:
    return now_utc.astimezone(_IST).strftime("%Y%m%d_%H%M%S")


def _safe_stat(path: Path) -> Optional[os.stat_result]:
    try:
        return path.stat()
    except OSError:
        return None


def _rel(base_dir: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(base_dir.resolve()).as_posix()
    except Exception:
        return path.name


def _recent_cutoff(recent_days: int) -> datetime:
    return _now_utc() - timedelta(days=max(1, int(recent_days)))


def _file_info(base_dir: Path, path: Path) -> Dict[str, Any]:
    stat = _safe_stat(path)
    exists = stat is not None
    modified = (
        datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
        if stat is not None
        else None
    )
    return {
        "path": _rel(base_dir, path),
        "exists": exists,
        "size_bytes": int(stat.st_size) if stat is not None else 0,
        "modified_at_utc": modified,
    }


def _iter_recent_files(directory: Path, recent_days: int, patterns: Iterable[str]) -> List[Path]:
    if not directory.exists():
        return []
    cutoff = _recent_cutoff(recent_days)
    files: List[Path] = []
    seen = set()
    for pattern in patterns:
        for candidate in sorted(directory.glob(pattern)):
            if not candidate.is_file():
                continue
            if candidate in seen:
                continue
            stat = _safe_stat(candidate)
            if stat is None:
                continue
            modified = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
            if modified < cutoff:
                continue
            files.append(candidate)
            seen.add(candidate)
    files.sort(key=lambda item: item.stat().st_mtime)
    return files


def build_debug_manifest(base_dir: Path | str, recent_days: int = _RECENT_DAYS_DEFAULT) -> Dict[str, Any]:
    base_dir = Path(base_dir).resolve()
    recent_days = max(1, int(recent_days))
    logs_dir = base_dir / "logs"
    backup_dir = base_dir / "backup_logs"
    recent_dir = base_dir / _RECENT_LOGS_DIR
    export_dir = base_dir / _DEBUG_EXPORT_DIR

    categories = {
        "root_full_copy": [_file_info(base_dir, base_dir / name) for name in _FULL_COPY_ROOT_FILES],
        "root_tail_copy": [_file_info(base_dir, base_dir / name) for name in _TAIL_COPY_ROOT_FILES],
        "sqlite_snapshots": [_file_info(base_dir, base_dir / name) for name in _SQLITE_FILES],
        "decision_logs_recent": [
            _file_info(base_dir, path)
            for path in _iter_recent_files(logs_dir, recent_days, ["decision_log_*.jsonl"])
        ],
        "backup_logs_recent": [
            _file_info(base_dir, path)
            for path in _iter_recent_files(
                backup_dir,
                recent_days,
                ["decision_log_*.jsonl", "trader_*.log"],
            )
        ],
        "recent_since_restart": [
            _file_info(base_dir, path)
            for path in sorted(recent_dir.glob("*"))
            if path.is_file()
        ],
    }

    exported_bundles = []
    if export_dir.exists():
        for bundle in sorted(export_dir.glob("debug_bundle_*.zip"))[-5:]:
            exported_bundles.append(_file_info(base_dir, bundle))

    return {
        "generated_at_utc": _now_utc().isoformat(),
        "recent_days": recent_days,
        "code_snapshot": capture_code_snapshot(base_dir),
        "categories": categories,
        "recent_exports": exported_bundles,
        "notes": [
            "Large text/jsonl logs are exported as recent tails to keep bundles portable.",
            "SQLite databases are copied using sqlite backup when possible for a consistent snapshot.",
            "Only recent daily backup logs are included by default.",
        ],
    }


def _append_entry(entries: List[Dict[str, Any]], **kwargs) -> None:
    entries.append(kwargs)


def _copy_full_file(base_dir: Path, src: Path, dest: Path, entries: List[Dict[str, Any]]) -> None:
    if not src.exists():
        _append_entry(
            entries,
            source=_rel(base_dir, src),
            bundle_path=dest.as_posix(),
            included=False,
            mode="missing",
            source_size_bytes=0,
            included_size_bytes=0,
        )
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    stat = src.stat()
    _append_entry(
        entries,
        source=_rel(base_dir, src),
        bundle_path=dest.as_posix(),
        included=True,
        mode="full_copy",
        source_size_bytes=int(stat.st_size),
        included_size_bytes=int(stat.st_size),
    )


def _tail_text_file(
    base_dir: Path,
    src: Path,
    dest: Path,
    entries: List[Dict[str, Any]],
    *,
    max_bytes: int = _TEXT_TAIL_BYTES,
    max_lines: Optional[int] = None,
) -> None:
    if not src.exists():
        _append_entry(
            entries,
            source=_rel(base_dir, src),
            bundle_path=dest.as_posix(),
            included=False,
            mode="missing",
            source_size_bytes=0,
            included_size_bytes=0,
        )
        return
    stat = src.stat()
    dest.parent.mkdir(parents=True, exist_ok=True)
    raw = b""
    with src.open("rb") as handle:
        if stat.st_size > max_bytes:
            handle.seek(stat.st_size - max_bytes)
        raw = handle.read()
    text = raw.decode("utf-8", errors="ignore")
    if max_lines is not None:
        lines = text.splitlines()
        text = "\n".join(lines[-max_lines:])
        if text and not text.endswith("\n"):
            text += "\n"
    dest.write_text(text, encoding="utf-8")
    included_size = dest.stat().st_size
    _append_entry(
        entries,
        source=_rel(base_dir, src),
        bundle_path=dest.as_posix(),
        included=True,
        mode="tail_copy",
        source_size_bytes=int(stat.st_size),
        included_size_bytes=int(included_size),
        max_bytes=int(max_bytes),
        max_lines=int(max_lines) if max_lines is not None else None,
    )


def _snapshot_sqlite(base_dir: Path, src: Path, dest: Path, entries: List[Dict[str, Any]]) -> None:
    if not src.exists():
        _append_entry(
            entries,
            source=_rel(base_dir, src),
            bundle_path=dest.as_posix(),
            included=False,
            mode="missing",
            source_size_bytes=0,
            included_size_bytes=0,
        )
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    copied = False
    try:
        src_conn = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
        dest_conn = sqlite3.connect(str(dest))
        try:
            src_conn.backup(dest_conn)
        finally:
            dest_conn.close()
            src_conn.close()
        copied = True
    except Exception:
        try:
            shutil.copy2(src, dest)
            copied = True
        except Exception:
            copied = False
    stat = _safe_stat(src)
    dest_stat = _safe_stat(dest)
    _append_entry(
        entries,
        source=_rel(base_dir, src),
        bundle_path=dest.as_posix(),
        included=bool(copied and dest_stat is not None),
        mode="sqlite_backup" if copied else "copy_failed",
        source_size_bytes=int(stat.st_size) if stat is not None else 0,
        included_size_bytes=int(dest_stat.st_size) if dest_stat is not None else 0,
    )


def _write_json(dest: Path, payload: Dict[str, Any]) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _copy_log_artifact(base_dir: Path, src: Path, dest: Path, entries: List[Dict[str, Any]]) -> None:
    stat = _safe_stat(src)
    if stat is None:
        _append_entry(
            entries,
            source=_rel(base_dir, src),
            bundle_path=dest.as_posix(),
            included=False,
            mode="missing",
            source_size_bytes=0,
            included_size_bytes=0,
        )
        return
    is_jsonl = src.suffix.lower() == ".jsonl"
    if stat.st_size <= _FULL_COPY_LOG_MAX_BYTES:
        _copy_full_file(base_dir, src, dest, entries)
        return
    _tail_text_file(
        base_dir,
        src,
        dest,
        entries,
        max_bytes=_TEXT_TAIL_BYTES,
        max_lines=_JSONL_TAIL_LINES if is_jsonl else None,
    )


def _zip_directory(source_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for file_path in sorted(source_dir.rglob("*")):
            if not file_path.is_file():
                continue
            arcname = file_path.relative_to(source_dir).as_posix()
            bundle.write(file_path, arcname)


def build_debug_bundle(base_dir: Path | str, recent_days: int = _RECENT_DAYS_DEFAULT) -> Dict[str, Any]:
    base_dir = Path(base_dir).resolve()
    recent_days = max(1, int(recent_days))
    export_dir = base_dir / _DEBUG_EXPORT_DIR
    export_dir.mkdir(parents=True, exist_ok=True)

    now_utc = _now_utc()
    token = _bundle_token(now_utc)
    zip_path = export_dir / f"debug_bundle_{token}.zip"
    entries: List[Dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix=f"debug_bundle_{token}_", dir=str(export_dir)) as temp_dir:
        temp_root = Path(temp_dir)
        bundle_root = temp_root / "bundle"
        bundle_root.mkdir(parents=True, exist_ok=True)

        manifest = build_debug_manifest(base_dir, recent_days=recent_days)

        for name in _FULL_COPY_ROOT_FILES:
            _copy_full_file(
                base_dir,
                base_dir / name,
                bundle_root / "root" / name,
                entries,
            )

        for name in _TAIL_COPY_ROOT_FILES:
            max_lines = _JSONL_TAIL_LINES if name.endswith(".jsonl") else None
            _tail_text_file(
                base_dir,
                base_dir / name,
                bundle_root / "root" / name,
                entries,
                max_bytes=_TEXT_TAIL_BYTES,
                max_lines=max_lines,
            )

        for name in _SQLITE_FILES:
            _snapshot_sqlite(
                base_dir,
                base_dir / name,
                bundle_root / "sqlite" / name,
                entries,
            )

        for path in _iter_recent_files(base_dir / "logs", recent_days, ["decision_log_*.jsonl"]):
            _copy_log_artifact(base_dir, path, bundle_root / "logs" / path.name, entries)

        for path in _iter_recent_files(
            base_dir / "backup_logs",
            recent_days,
            ["decision_log_*.jsonl", "trader_*.log"],
        ):
            _copy_log_artifact(base_dir, path, bundle_root / "backup_logs" / path.name, entries)

        recent_dir = base_dir / _RECENT_LOGS_DIR
        if recent_dir.exists():
            for path in sorted(recent_dir.glob("*")):
                if path.is_file():
                    _copy_full_file(base_dir, path, bundle_root / _RECENT_LOGS_DIR / path.name, entries)

        readme = {
            "generated_at_utc": now_utc.isoformat(),
            "purpose": "Portable runtime debug bundle for remote analysis.",
            "how_to_use": [
                "Unzip the bundle beside a clone of the repository.",
                "Inspect manifest.json first to see what was included fully versus tailed.",
                "Use the SQLite snapshot and recent logs together when reproducing or explaining live behavior.",
            ],
        }
        _write_json(bundle_root / "bundle_readme.json", readme)

        manifest["bundle"] = {
            "generated_at_utc": now_utc.isoformat(),
            "token": token,
            "path": _rel(base_dir, zip_path),
            "entries": entries,
        }
        _write_json(bundle_root / "manifest.json", manifest)
        _zip_directory(bundle_root, zip_path)

    zip_stat = zip_path.stat()
    return {
        "bundle_path": str(zip_path),
        "bundle_name": zip_path.name,
        "bundle_size_bytes": int(zip_stat.st_size),
        "generated_at_utc": now_utc.isoformat(),
        "recent_days": recent_days,
        "entries": entries,
        "manifest_path_inside_bundle": "manifest.json",
    }
