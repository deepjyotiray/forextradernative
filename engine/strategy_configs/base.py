"""
Strategy config store — load, save, reset, apply to cfg.

Each strategy has its own JSON file in strategy_configs/data/.
On first run the file is created from DEFAULTS.
On save, CFG_MAP keys are applied live to the cfg module.
"""
from __future__ import annotations

import json
import copy
from pathlib import Path
from typing import Any, Dict

import config as cfg

from . import swing_engine
from . import intraday_engine
from . import smc_confluence
from . import sweep_scalper
from . import m15_sr
from . import trend_channel
from . import htf_long
from . import htf_short

# Registry: strategy name → definition module
_REGISTRY = {
    swing_engine.NAME: swing_engine,
    intraday_engine.NAME: intraday_engine,
    smc_confluence.NAME: smc_confluence,
    sweep_scalper.NAME: sweep_scalper,
    m15_sr.NAME: m15_sr,
    trend_channel.NAME: trend_channel,
    htf_long.NAME: htf_long,
    htf_short.NAME: htf_short,
}

_DATA_DIR = Path(__file__).parent / "data"
_DATA_DIR.mkdir(exist_ok=True)

# In-memory cache: name → config dict
_cache: Dict[str, Dict[str, Any]] = {}


def _path(name: str) -> Path:
    return _DATA_DIR / f"{name}.json"


def _merge(defaults: Dict, saved: Dict) -> Dict:
    """Merge saved values onto defaults — new keys in defaults are added, unknown keys in saved are dropped."""
    result = copy.deepcopy(defaults)
    for k, v in saved.items():
        if k in result:
            result[k] = v
    return result


def _normalize_disabled_fields(mod: Any, values: Dict[str, Any]) -> Dict[str, Any]:
    """Force disabled strategy config fields back to their default values."""
    normalized = copy.deepcopy(values)
    defaults = getattr(mod, "DEFAULTS", {})
    for key in getattr(mod, "DISABLED", {}):
        if key in defaults:
            normalized[key] = copy.deepcopy(defaults[key])
    return normalized


def load(name: str) -> Dict[str, Any]:
    """Load config for a strategy. Creates from defaults if file missing."""
    if name in _cache:
        return dict(_cache[name])
    mod = _REGISTRY.get(name)
    if mod is None:
        return {}
    defaults = copy.deepcopy(mod.DEFAULTS)
    p = _path(name)
    if p.exists():
        try:
            saved = json.loads(p.read_text(encoding="utf-8"))
            merged = _normalize_disabled_fields(mod, _merge(defaults, saved))
            if merged != saved:
                p.write_text(json.dumps(merged, indent=2), encoding="utf-8")
        except Exception:
            merged = defaults
    else:
        merged = _normalize_disabled_fields(mod, defaults)
        p.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    _cache[name] = merged
    return dict(merged)


def save(name: str, values: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate, persist, update cache, and apply CFG_MAP keys to cfg module live.
    Only keys present in DEFAULTS are accepted — unknown keys are ignored.
    """
    mod = _REGISTRY.get(name)
    if mod is None:
        raise ValueError(f"Unknown strategy: {name}")
    current = load(name)
    for k, v in values.items():
        if k in current:
            current[k] = v
    current = _normalize_disabled_fields(mod, current)
    _path(name).write_text(json.dumps(current, indent=2), encoding="utf-8")
    _cache[name] = current
    _apply_to_cfg(name, current)
    return dict(current)


def reset(name: str) -> Dict[str, Any]:
    """Reset a strategy config to defaults, persist, and re-apply to cfg."""
    mod = _REGISTRY.get(name)
    if mod is None:
        raise ValueError(f"Unknown strategy: {name}")
    defaults = _normalize_disabled_fields(mod, copy.deepcopy(mod.DEFAULTS))
    _path(name).write_text(json.dumps(defaults, indent=2), encoding="utf-8")
    _cache[name] = defaults
    _apply_to_cfg(name, defaults)
    return dict(defaults)


def get(name: str) -> Dict[str, Any]:
    """Alias for load — used by strategy_manager and risk_manager."""
    return load(name)


def get_all() -> Dict[str, Any]:
    """Return configs for all registered strategies."""
    return {name: load(name) for name in _REGISTRY}


def get_meta(name: str) -> Dict[str, Any]:
    """Return LOCKED and DISABLED metadata for a strategy (used by UI)."""
    mod = _REGISTRY.get(name)
    if mod is None:
        return {"locked": {}, "disabled": {}}
    return {
        "locked": dict(getattr(mod, "LOCKED", {})),
        "disabled": dict(getattr(mod, "DISABLED", {})),
        "cfg_map": dict(getattr(mod, "CFG_MAP", {})),
    }


def get_all_meta() -> Dict[str, Any]:
    return {name: get_meta(name) for name in _REGISTRY}


def _apply_to_cfg(name: str, values: Dict[str, Any]) -> None:
    """Write CFG_MAP keys from values into the live cfg module."""
    mod = _REGISTRY.get(name)
    if mod is None:
        return
    cfg_map = getattr(mod, "CFG_MAP", {})
    for local_key, cfg_key in cfg_map.items():
        if local_key in values and values[local_key] is not None:
            if hasattr(cfg, cfg_key):
                setattr(cfg, cfg_key, values[local_key])


def apply_all_to_cfg() -> None:
    """Apply all strategy configs to cfg on startup."""
    for name in _REGISTRY:
        _apply_to_cfg(name, load(name))


def list_strategies() -> list:
    return list(_REGISTRY.keys())
