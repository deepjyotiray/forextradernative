from datetime import datetime, timezone

from engine import deployment_metadata as dm


def test_compare_snapshots_flags_restart_when_code_moves_ahead():
    runtime = {
        "git": {"commit": "abc123"},
        "ui": {"commit": "ui1", "dirty": False, "last_modified_utc": "2026-05-11T10:00:00+00:00", "latest_file": "dashboard.html"},
        "strategies": {"commit": "st1", "dirty": False, "last_modified_utc": "2026-05-11T10:00:00+00:00", "latest_file": "auto_trader.py"},
    }
    current = {
        "git": {"commit": "def456"},
        "ui": {"commit": "ui2", "dirty": False, "last_modified_utc": "2026-05-11T11:00:00+00:00", "latest_file": "dashboard.html"},
        "strategies": {"commit": "st1", "dirty": False, "last_modified_utc": "2026-05-11T10:00:00+00:00", "latest_file": "auto_trader.py"},
    }

    result = dm.compare_snapshots(runtime, current)

    assert result["restart_required"] is True
    assert result["in_sync"] is False
    assert result["status"] == "restart_required"
    assert result["differences"] == ["code", "ui"]


def test_capture_code_snapshot_falls_back_to_filesystem_when_git_is_unavailable(tmp_path, monkeypatch):
    engine_dir = tmp_path / "engine"
    strategies_dir = engine_dir / "strategies"
    strategy_cfg_dir = engine_dir / "strategy_configs"
    strategies_dir.mkdir(parents=True)
    strategy_cfg_dir.mkdir(parents=True)

    files = {
        tmp_path / "dashboard.html": 100,
        tmp_path / "strategies.html": 200,
        tmp_path / "_dashboard_script.js": 300,
        tmp_path / "auto_trader.py": 400,
        engine_dir / "strategy_manager.py": 500,
        engine_dir / "smc_strategy.py": 600,
        engine_dir / "sweep_scalper.py": 700,
        engine_dir / "m15_sr_strategy.py": 800,
        engine_dir / "swing_engine_strategy.py": 900,
        engine_dir / "intraday_engine_strategy.py": 1000,
        strategies_dir / "base_strategy.py": 1100,
        strategy_cfg_dir / "__init__.py": 1200,
        strategy_cfg_dir / "base.py": 1300,
        strategy_cfg_dir / "trend_channel.py": 1400,
        strategy_cfg_dir / "swing_engine.py": 1500,
        strategy_cfg_dir / "sweep_scalper.py": 1600,
        strategy_cfg_dir / "smc_confluence.py": 1700,
        strategy_cfg_dir / "m15_sr.py": 1800,
        strategy_cfg_dir / "intraday_engine.py": 1900,
        strategy_cfg_dir / "htf_short.py": 2000,
        strategy_cfg_dir / "htf_long.py": 2100,
    }
    for path, ts in files.items():
        path.write_text("x", encoding="utf-8")
        path.touch()
        path_ts = float(ts)
        path.chmod(0o666)
        import os
        os.utime(path, (path_ts, path_ts))

    monkeypatch.setattr(dm, "_run_git", lambda *_args, **_kwargs: None)

    snapshot = dm.capture_code_snapshot(tmp_path)

    assert snapshot["git"]["commit"] is None
    assert snapshot["ui"]["version"] == "19700101-000500"
    assert snapshot["ui"]["latest_file"] == "_dashboard_script.js"
    assert snapshot["strategies"]["latest_file"] == "engine/strategy_configs/htf_long.py"
    assert snapshot["strategies"]["last_modified_utc"] == datetime.fromtimestamp(2100, tz=timezone.utc).isoformat()


def test_capture_code_snapshot_marks_dirty_scope_and_uses_file_mtime(tmp_path, monkeypatch):
    dashboard_file = tmp_path / "dashboard.html"
    dashboard_file.write_text("x", encoding="utf-8")
    strategies_page = tmp_path / "strategies.html"
    strategies_page.write_text("x", encoding="utf-8")
    ui_script = tmp_path / "_dashboard_script.js"
    ui_script.write_text("x", encoding="utf-8")
    strategy_file = tmp_path / "auto_trader.py"
    strategy_file.write_text("x", encoding="utf-8")
    engine_dir = tmp_path / "engine"
    (engine_dir / "strategies").mkdir(parents=True)
    (engine_dir / "strategy_configs").mkdir(parents=True)
    for rel in [
        "strategy_manager.py",
        "smc_strategy.py",
        "sweep_scalper.py",
        "m15_sr_strategy.py",
        "swing_engine_strategy.py",
        "intraday_engine_strategy.py",
        "strategies/base_strategy.py",
        "strategy_configs/__init__.py",
        "strategy_configs/base.py",
        "strategy_configs/trend_channel.py",
        "strategy_configs/swing_engine.py",
        "strategy_configs/sweep_scalper.py",
        "strategy_configs/smc_confluence.py",
        "strategy_configs/m15_sr.py",
        "strategy_configs/intraday_engine.py",
        "strategy_configs/htf_short.py",
        "strategy_configs/htf_long.py",
    ]:
        target = engine_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("x", encoding="utf-8")

    import os
    os.utime(dashboard_file, (1000, 1000))
    os.utime(strategies_page, (2000, 2000))
    os.utime(ui_script, (5000, 5000))

    def fake_run_git(_base_dir, args):
        joined = " ".join(args)
        if "status --porcelain -- dashboard.html strategies.html _dashboard_script.js" in joined:
            return " M _dashboard_script.js"
        if "status --porcelain -- auto_trader.py" in joined:
            return ""
        if "status --porcelain" == joined:
            return " M _dashboard_script.js"
        if "log -1" in joined and "-- dashboard.html strategies.html _dashboard_script.js" in joined:
            return "abcd1234\x1fabc1234\x1f2026-05-10T00:00:00+00:00\x1fUI change"
        if "log -1" in joined and "-- auto_trader.py" in joined:
            return "ffff9999\x1fffff999\x1f2026-05-09T00:00:00+00:00\x1fStrategy change"
        if "log -1" in joined:
            return "head9999\x1fhead999\x1f2026-05-11T00:00:00+00:00\x1fHead change"
        if "rev-parse --abbrev-ref HEAD" == joined:
            return "main"
        return None

    monkeypatch.setattr(dm, "_run_git", fake_run_git)

    snapshot = dm.capture_code_snapshot(tmp_path)

    assert snapshot["git"]["version"] == "head999 + local"
    assert snapshot["ui"]["dirty"] is True
    assert snapshot["ui"]["version"] == "abc1234 + local"
    assert snapshot["ui"]["last_modified_utc"] == datetime.fromtimestamp(5000, tz=timezone.utc).isoformat()
