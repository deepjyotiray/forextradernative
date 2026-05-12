import json
from pathlib import Path

import engine.trade_attribution as ta


def test_attribution_writer_rotates_and_writes(tmp_path: Path, monkeypatch):
    primary = tmp_path / "trade_attribution.jsonl"
    backup_dir = tmp_path / "backup_logs"
    fallback = tmp_path / "trade_attribution_fallback.jsonl"
    backup_dir.mkdir()
    primary.write_text('{"old":true}\n', encoding="utf-8")

    monkeypatch.setattr(ta, "_ATTRIBUTION_FILE", str(primary))
    monkeypatch.setattr(ta, "_BACKUP_DIR", str(backup_dir))
    monkeypatch.setattr(ta, "_ATTRIBUTION_FALLBACK_FILE", str(fallback))
    monkeypatch.setattr(ta, "_ATTRIBUTION_MAX_BYTES", 1)
    monkeypatch.setattr(ta, "_ATTRIBUTION_LINE_MAX_BYTES", 2048)

    ta.attribution_engine._write_attribution({"timestamp": "t", "unix_time": 1, "strategy": "S", "decision": "TRADE_SKIPPED", "reason": "x", "price": 1.0})

    rotated = list(backup_dir.glob("trade_attribution_*.jsonl"))
    assert rotated
    lines = primary.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["strategy"] == "S"


def test_attribution_writer_falls_back_when_primary_write_fails(tmp_path: Path, monkeypatch):
    primary = tmp_path / "trade_attribution.jsonl"
    backup_dir = tmp_path / "backup_logs"
    fallback = tmp_path / "trade_attribution_fallback.jsonl"
    backup_dir.mkdir()

    monkeypatch.setattr(ta, "_ATTRIBUTION_FILE", str(primary))
    monkeypatch.setattr(ta, "_BACKUP_DIR", str(backup_dir))
    monkeypatch.setattr(ta, "_ATTRIBUTION_FALLBACK_FILE", str(fallback))
    monkeypatch.setattr(ta, "_ATTRIBUTION_MAX_BYTES", 1024 * 1024)
    monkeypatch.setattr(ta, "_ATTRIBUTION_LINE_MAX_BYTES", 256)

    original_append = ta._append_bytes
    state = {"calls": 0}

    def flaky_append(path: str, data: bytes) -> None:
        state["calls"] += 1
        if state["calls"] == 1 and path == str(primary):
            raise OSError(22, "Invalid argument")
        return original_append(path, data)

    monkeypatch.setattr(ta, "_append_bytes", flaky_append)

    ta.attribution_engine._write_attribution(
        {
            "timestamp": "t",
            "unix_time": 1,
            "strategy": "S",
            "decision": "TRADE_SKIPPED",
            "reason": "x" * 5000,
            "price": 1.0,
        }
    )

    assert fallback.exists()
    row = json.loads(fallback.read_text(encoding="utf-8").splitlines()[0])
    assert row["strategy"] == "S"
    assert row["payload_truncated"] is True
