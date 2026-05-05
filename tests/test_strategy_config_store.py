import json

from engine.strategy_configs import base as config_store


def test_load_normalizes_disabled_intraday_tp3_field(tmp_path):
    stale_payload = {
        "enabled": True,
        "risk_pct": 0.5,
        "max_active_trades": 5,
        "max_trades_day": 5,
        "spread_max": 0.5,
        "sl_min": 1.5,
        "sl_max": 3.0,
        "tp1_r": 1.5,
        "tp2_r": 2.0,
        "tp3_r": 3.5,
        "sessions": ["LONDON", "NEW_YORK"],
        "news_block": True,
        "cooldown_seconds": 120,
        "consecutive_loss_limit": 3,
        "loss_pause_seconds": 900,
        "notes": "",
    }
    path = tmp_path / "INTRADAY_ENGINE.json"
    path.write_text(json.dumps(stale_payload), encoding="utf-8")

    original_dir = config_store._DATA_DIR
    original_cache = dict(config_store._cache)
    try:
        config_store._DATA_DIR = tmp_path
        config_store._cache.clear()
        loaded = config_store.load("INTRADAY_ENGINE")
    finally:
        config_store._DATA_DIR = original_dir
        config_store._cache.clear()
        config_store._cache.update(original_cache)

    assert loaded["tp3_r"] is None
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["tp3_r"] is None


def test_save_rejects_manual_override_for_disabled_intraday_tp3(tmp_path):
    original_dir = config_store._DATA_DIR
    original_cache = dict(config_store._cache)
    try:
        config_store._DATA_DIR = tmp_path
        config_store._cache.clear()
        saved = config_store.save("INTRADAY_ENGINE", {"tp3_r": 9.9, "tp2_r": 2.2})
    finally:
        config_store._DATA_DIR = original_dir
        config_store._cache.clear()
        config_store._cache.update(original_cache)

    assert saved["tp3_r"] is None
    assert saved["tp2_r"] == 2.2
    persisted = json.loads((tmp_path / "INTRADAY_ENGINE.json").read_text(encoding="utf-8"))
    assert persisted["tp3_r"] is None
    assert persisted["tp2_r"] == 2.2
