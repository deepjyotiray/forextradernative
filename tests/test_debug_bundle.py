import json
import sqlite3
import zipfile
from pathlib import Path

from engine.debug_bundle import build_debug_bundle, build_debug_manifest


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_build_debug_manifest_lists_recent_artifacts(tmp_path: Path):
    _write_text(tmp_path / "trader.log", "line-1\nline-2\n")
    _write_text(tmp_path / "runtime_config.json", "{}\n")
    _write_text(tmp_path / "logs" / "decision_log_2026-05-12.jsonl", '{"x":1}\n')
    _write_text(tmp_path / "recent_logs_since_restart" / "summary.txt", "ok\n")

    manifest = build_debug_manifest(tmp_path, recent_days=7)

    assert manifest["recent_days"] == 7
    categories = manifest["categories"]
    assert any(item["path"] == "trader.log" and item["exists"] for item in categories["root_tail_copy"])
    assert any(item["path"] == "runtime_config.json" and item["exists"] for item in categories["root_full_copy"])
    assert any(item["path"].startswith("logs/decision_log_") for item in categories["decision_logs_recent"])
    assert any(item["path"] == "recent_logs_since_restart/summary.txt" for item in categories["recent_since_restart"])


def test_build_debug_bundle_creates_zip_with_manifest_and_sqlite_snapshot(tmp_path: Path):
    _write_text(tmp_path / "trader.log", "\n".join(f"log-{i}" for i in range(5000)))
    _write_text(tmp_path / "trade_attribution.jsonl", "\n".join(f'{{"n":{i}}}' for i in range(5000)))
    _write_text(tmp_path / "runtime_config.json", '{"mode":"live"}\n')
    _write_text(tmp_path / "logs" / "decision_log_2026-05-12.jsonl", '{"decision":"TRADE_TAKEN"}\n')

    db_path = tmp_path / "orders.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE sample (id INTEGER PRIMARY KEY, value TEXT)")
        conn.execute("INSERT INTO sample (value) VALUES ('hello')")

    bundle = build_debug_bundle(tmp_path, recent_days=7)

    bundle_path = Path(bundle["bundle_path"])
    assert bundle_path.exists()
    assert bundle_path.suffix == ".zip"
    assert bundle["entries"]

    with zipfile.ZipFile(bundle_path) as zf:
        names = set(zf.namelist())
        assert "manifest.json" in names
        assert "root/trader.log" in names
        assert "sqlite/orders.db" in names

        manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
        modes = {entry["source"]: entry["mode"] for entry in manifest["bundle"]["entries"]}
        assert modes["trader.log"] == "tail_copy"
        assert modes["orders.db"] == "sqlite_backup"

        with zf.open("sqlite/orders.db") as handle:
            copied_db = tmp_path / "copied_orders.db"
            copied_db.write_bytes(handle.read())

    with sqlite3.connect(tmp_path / "copied_orders.db") as conn:
        row = conn.execute("SELECT value FROM sample").fetchone()
    assert row[0] == "hello"
