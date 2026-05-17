import json

from engine.strategy_configs import base as config_store


def test_load_uses_m15_scalp_deep_defaults_when_file_missing_keys(tmp_path):
    stale_payload = {
        "enabled": True,
        "risk_pct": 0.5,
        "max_active_trades": 5,
        "max_trades_day": 5,
        "spread_max": 0.5,
        "target_rr": 2.2,
        "min_rr": 1.4,
        "sessions": ["LONDON", "NEW_YORK"],
        "news_block": True,
        "notes": "",
    }
    path = tmp_path / "M15_SCALP_DEEP.json"
    path.write_text(json.dumps(stale_payload), encoding="utf-8")

    original_dir = config_store._DATA_DIR
    original_cache = dict(config_store._cache)
    try:
        config_store._DATA_DIR = tmp_path
        config_store._cache.clear()
        loaded = config_store.load("M15_SCALP_DEEP")
    finally:
        config_store._DATA_DIR = original_dir
        config_store._cache.clear()
        config_store._cache.update(original_cache)

    assert loaded["target_rr"] == 2.2
    assert loaded["min_rr"] == 1.4
    assert loaded["min_h1_bars"] == 40
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["min_m5_bars"] == 20


def test_save_ignores_unknown_keys_for_m15_scalp_deep(tmp_path):
    original_dir = config_store._DATA_DIR
    original_cache = dict(config_store._cache)
    try:
        config_store._DATA_DIR = tmp_path
        config_store._cache.clear()
        saved = config_store.save("M15_SCALP_DEEP", {"unknown_key": 9.9, "min_rr": 2.2})
    finally:
        config_store._DATA_DIR = original_dir
        config_store._cache.clear()
        config_store._cache.update(original_cache)

    assert "unknown_key" not in saved
    assert saved["min_rr"] == 2.2
    persisted = json.loads((tmp_path / "M15_SCALP_DEEP.json").read_text(encoding="utf-8"))
    assert "unknown_key" not in persisted
    assert persisted["min_rr"] == 2.2
