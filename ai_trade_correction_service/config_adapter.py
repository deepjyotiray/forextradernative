from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class ServiceRuntimeConfig:
    enabled: bool
    dry_run: bool
    fail_open: bool
    model: str
    timeout_seconds: float
    api_base: str
    api_key_present: bool


def _load_local_env_file() -> None:
    root_dir = Path(__file__).resolve().parent.parent
    env_path = root_dir / ".ai_trade_correction.local.env"
    if not env_path.exists():
        return
    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                continue
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            os.environ[key] = value
    except Exception:
        # Fail open: keep existing environment behavior if the local secrets file is malformed.
        return


def load_service_runtime_config() -> ServiceRuntimeConfig:
    _load_local_env_file()
    enabled_text = os.getenv("AI_TRADE_CORRECTION_ENABLED", "1").strip().lower()
    dry_run_text = os.getenv("AI_TRADE_CORRECTION_DRY_RUN", "0").strip().lower()
    fail_open_text = os.getenv("AI_TRADE_CORRECTION_FAIL_OPEN", "1").strip().lower()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    return ServiceRuntimeConfig(
        enabled=enabled_text not in {"0", "false", "off", "no"},
        dry_run=dry_run_text in {"1", "true", "on", "yes"},
        fail_open=fail_open_text not in {"0", "false", "off", "no"},
        model=os.getenv("AI_TRADE_CORRECTION_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini",
        timeout_seconds=float(os.getenv("AI_TRADE_CORRECTION_TIMEOUT_SECONDS", "20").strip() or "20"),
        api_base=(os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1").strip() or "https://api.openai.com/v1").rstrip("/"),
        api_key_present=bool(api_key),
    )


def resolve_enabled_strategy(cfg_module, strategy_manager=None) -> Dict[str, Any]:
    active_name = ""
    selected = []
    if strategy_manager is not None:
        active_name = str(getattr(strategy_manager, "active_name", "") or "").strip().upper()
        selected = [str(name or "").strip().upper() for name in getattr(strategy_manager, "selected", []) or [] if str(name or "").strip()]

    default_strategy = getattr(cfg_module, "DEFAULT_STRATEGY", "AUTO")
    if isinstance(default_strategy, (list, tuple, set)):
        configured = [str(name or "").strip().upper() for name in default_strategy if str(name or "").strip()]
    else:
        configured = [part.strip().upper() for part in str(default_strategy or "AUTO").split(",") if part.strip()]

    runtime_selected = selected or configured
    unique_runtime = [name for name in runtime_selected if name and name != "AUTO"]

    result = {
        "active_name": active_name or ",".join(unique_runtime) or "AUTO",
        "configured": configured,
        "selected": selected,
        "enabled_strategy": unique_runtime[0] if len(unique_runtime) == 1 else "",
        "single_strategy_mode": len(unique_runtime) == 1,
        "reason": "",
    }
    if not unique_runtime:
        result["reason"] = "No concrete strategy is selected; current selection resolves to AUTO."
    elif len(unique_runtime) > 1:
        result["reason"] = "More than one strategy is selected; AI trade correction only supports one enabled strategy."
    return result


def session_is_active(session_label: str) -> bool:
    return str(session_label or "").upper() not in {"", "CLOSED", "ROLLOVER"}


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)
