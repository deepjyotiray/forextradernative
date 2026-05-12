import json
from pathlib import Path

import config as cfg


def test_trading_enabled_persists_in_runtime_config(tmp_path: Path):
    original_state_file = cfg._CONFIG_STATE_FILE
    original_profiles_file = cfg._CONFIG_PROFILES_FILE
    original_cache = dict(cfg._PROFILE_CACHE)
    original_active = dict(cfg._PROFILE_ACTIVE)
    original_flag = cfg.TRADING_ENABLED
    original_default = cfg._DEFAULT_RUNTIME_CONFIG.get("TRADING_ENABLED")

    try:
        cfg._CONFIG_STATE_FILE = tmp_path / "runtime_config.json"
        cfg._CONFIG_PROFILES_FILE = tmp_path / "config_profiles.json"
        cfg._PROFILE_CACHE = {}
        cfg._PROFILE_ACTIVE = {"live": "live", "backtest": "live"}
        cfg._DEFAULT_RUNTIME_CONFIG["TRADING_ENABLED"] = False

        cfg.TRADING_ENABLED = True
        saved = cfg.save_runtime_config()

        cfg.TRADING_ENABLED = False
        cfg.load_runtime_config()

        assert saved["TRADING_ENABLED"] is True
        assert cfg.TRADING_ENABLED is True

        state_payload = json.loads(cfg._CONFIG_STATE_FILE.read_text(encoding="utf-8"))
        assert state_payload["TRADING_ENABLED"] is True
    finally:
        cfg._CONFIG_STATE_FILE = original_state_file
        cfg._CONFIG_PROFILES_FILE = original_profiles_file
        cfg._PROFILE_CACHE = original_cache
        cfg._PROFILE_ACTIVE = original_active
        cfg.TRADING_ENABLED = original_flag
        if original_default is None:
            cfg._DEFAULT_RUNTIME_CONFIG.pop("TRADING_ENABLED", None)
        else:
            cfg._DEFAULT_RUNTIME_CONFIG["TRADING_ENABLED"] = original_default
