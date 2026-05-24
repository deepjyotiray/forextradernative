from __future__ import annotations

import os
from pathlib import Path


DEFAULT_LOCAL_ENV_FILE = ".ai_trade_correction.local.env"


def load_local_env_file(base_dir: str | Path | None = None, env_filename: str = DEFAULT_LOCAL_ENV_FILE) -> Path:
    root_dir = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parent
    env_path = root_dir / env_filename
    if not env_path.exists():
        return env_path
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
            if key not in os.environ or not str(os.environ.get(key) or "").strip():
                os.environ[key] = value
    except Exception:
        # Fail open: keep existing environment behavior if the local secrets file is malformed.
        return env_path
    return env_path
