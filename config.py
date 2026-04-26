"""
Shared configuration.
"""
import copy
import json
from pathlib import Path

# MT5 connection
MT5_PATH = None
MT5_LOGIN = None
MT5_PASSWORD = None
MT5_SERVER = None

# Trading
APP_VERSION = "2026.04.24.2"
SYMBOL = "XAUUSD"
AVAILABLE_SYMBOLS = ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD", "BTCUSD", "ETHUSD"]
MAGIC_NUMBER = 234000
DEVIATION = 20
PIP_VALUE_PER_LOT = 100

# Risk defaults
MAX_POSITIONS = 4
MAX_RISK_PCT = 0.5
MAX_DRAWDOWN_PCT = 5.0
MAX_LOT = 1.0
MIN_LOT = 0.01
DAILY_TARGET_DOLLARS = 100.0
DAILY_TARGET_ENABLED = True
DAILY_LOSS_LIMIT_PCT = 2.0
MAX_CONSECUTIVE_LOSSES = 4
MIN_TRADE_COOLDOWN = 10.0
LOSS_STREAK_PAUSE = 900
SESSION_MAX_TRADES = 12

# Session (UTC)
ASIAN_START = 0
LONDON_START = 7
NY_START = 13
SESSION_BLOCK_MINUTES = 3

# Strategy trade windows (UTC)
TRADE_WINDOW_LONDON_START = 7
TRADE_WINDOW_LONDON_END = 9
TRADE_WINDOW_OVERLAP_START = 12
TRADE_WINDOW_OVERLAP_END = 17

# Data
MAX_CANDLES = 1000
CANDLE_LIMITS = {
    "M1": 300,
    "M5": 200,
    "M15": 200,
    "H1": 200,
    "H4": 200,
}

# API
API_HOST = "127.0.0.1"
API_PORT = 8899

# Live price feed
MARKET_TICK_POLL_INTERVAL = 0.02
DASHBOARD_WS_PUSH_INTERVAL = 0.03

# Default strategy on startup
DEFAULT_STRATEGY = "AUTO"

# Tier 1 upgrades toggle
TIER1_ENABLED = True

# Manual override for strategy session gate
SESSION_OVERRIDE_ENABLED = False

# Manual overrides for hard strategy gates. Keep disabled for normal live trading.
ALL_GATES_OVERRIDE_ENABLED = False
SPREAD_GATE_OVERRIDE_ENABLED = False
COMPRESSION_GATE_OVERRIDE_ENABLED = False
EXECUTION_GATE_OVERRIDE_ENABLED = False

# Fine-grained strategy gate overrides.
TIME_GATE_OVERRIDE_ENABLED = False
SPREAD_MEAN_GATE_OVERRIDE_ENABLED = False
SPREAD_VOLATILITY_GATE_OVERRIDE_ENABLED = False
SPREAD_PERCENTILE_GATE_OVERRIDE_ENABLED = False
SPREAD_DELTA_GATE_OVERRIDE_ENABLED = False
COMPRESSION_RANGE_GATE_OVERRIDE_ENABLED = False
ATR_RISING_GATE_OVERRIDE_ENABLED = False
EXECUTION_QUALITY_GATE_OVERRIDE_ENABLED = False
TICK_DIRECTION_GATE_OVERRIDE_ENABLED = False
POST_SIGNAL_SPREAD_GATE_OVERRIDE_ENABLED = False

# Strategy spread-quality gates
SMC_SPREAD_MEAN_MAX = 0.50
SMC_SPREAD_STD_MAX = 0.05
SMC_SPREAD_PERCENTILE_MAX = 0.50
SMC_CURRENT_SPREAD_DELTA_MAX = 0.03

SCALPER_SPREAD_MEAN_MAX = 0.50
SCALPER_SPREAD_STD_MAX = 0.04
SCALPER_SPREAD_PERCENTILE_MAX = 0.50
SCALPER_CURRENT_SPREAD_DELTA_MAX = 0.03

# Compression / Range10 gates
COMPRESSION_RANGE_LOOKBACK = 10
COMPRESSION_ATR_MULTIPLIER = 1.80

# SMC gates and thresholds
SMC_THRESHOLD_WITH_TREND = 0.65
SMC_THRESHOLD_COUNTER = 0.70
SMC_THRESHOLD_COUNTER_MAX = 0.75
SMC_MIN_RR = 1.20
SMC_TIMEFRAME_EMA_SLOPE_MIN = 0.10
SMC_LTF_TICK_CONFLICT_LONG_MAX = 0.40
SMC_LTF_TICK_CONFLICT_SHORT_MIN = 0.60

# SMC regime-aware threshold offsets
SMC_RANGING_THRESHOLD_OFFSET = 0.05   # added to both thresholds when regime=RANGING
SMC_TRENDING_THRESHOLD_OFFSET = 0.00  # offset when regime=TRENDING (0 = use base values)

# SMC early-fail protection (Tier1)
SMC_EARLY_FAIL_POINTS = 0.20          # default early-fail distance in points
SMC_EARLY_FAIL_RANGING = 0.15         # tighter early-fail in ranging markets
SMC_EARLY_FAIL_TRENDING = 0.25        # wider tolerance in trending markets
SMC_EARLY_FAIL_CONF_HIGH_MULT = 1.20  # multiply early-fail by this when confidence >= HIGH_CONF_THRESHOLD (more room)
SMC_EARLY_FAIL_CONF_LOW_MULT  = 0.85  # multiply early-fail by this when confidence < 0.65 (cut faster)

# Tier1 tick window for early-fail
SMC_TIER1_MIN_TICKS = 3               # do not trigger early-fail before this many ticks (noise filter)
SMC_TIER1_MAX_TICKS = 15              # stop checking early-fail after this many ticks

# Tier1 spread re-check delta (pre-send, separate from strategy spread gate)
SMC_TIER1_SPREAD_DELTA_MAX = 0.02     # max spread widening allowed between signal and order send

# SMC high-confidence scaling
SMC_HIGH_CONF_THRESHOLD = 0.75        # confidence >= this triggers high-conf mode
SMC_HIGH_CONF_RR_MULTIPLIER = 1.25    # TP distance multiplied by this in high-conf mode
SMC_HIGH_CONF_RISK_MULTIPLIER = 1.50  # lot size multiplied by this in high-conf mode (capped at MAX_LOT)

# Adaptive parameter engine — hard bounds
# These define the allowed range for live adaptive adjustments.
# The engine will never move a parameter outside these bounds.
ADAPT_THRESHOLD_WTR_MIN  = 0.65
ADAPT_THRESHOLD_WTR_MAX  = 0.77
ADAPT_THRESHOLD_CTR_MIN  = 0.70
ADAPT_THRESHOLD_CTR_MAX  = 0.82
ADAPT_MIN_RR_MIN         = 1.50
ADAPT_MIN_RR_MAX         = 2.20
ADAPT_SESSION_TRADES_MIN = 3
ADAPT_SESSION_TRADES_MAX = 8
ADAPT_COOLDOWN_MIN       = 600
ADAPT_COOLDOWN_MAX       = 1800

# Sweep Scalper gates and thresholds
SCALPER_QUALITY_THRESHOLD = 0.65
SCALPER_ATR_MIN = 0.30
SCALPER_ATR_MAX = 5.00
SCALPER_EMA20_SLOPE_MIN = 0.05
SCALPER_BODY_RATIO_MIN = 0.40
SCALPER_TICK_DIR_THRESHOLD = 0.65
SCALPER_SWEEP_LOOKBACK = 15
SCALPER_SWEEP_TOLERANCE = 0.30
SCALPER_MAX_TRADES_SESSION = 5
SCALPER_LEVEL_COOLDOWN = 1200


_CONFIG_STATE_FILE = Path(__file__).with_name("runtime_config.json")
_CONFIG_PROFILES_FILE = Path(__file__).with_name("config_profiles.json")
_PERSISTED_KEYS = {
    "MAX_POSITIONS", "MAX_RISK_PCT", "MAX_DRAWDOWN_PCT", "MAX_LOT", "MIN_LOT",
    "DAILY_TARGET_DOLLARS", "DAILY_TARGET_ENABLED", "DAILY_LOSS_LIMIT_PCT",
    "MAX_CONSECUTIVE_LOSSES", "MIN_TRADE_COOLDOWN", "LOSS_STREAK_PAUSE",
    "SESSION_MAX_TRADES",
    "MARKET_TICK_POLL_INTERVAL", "DASHBOARD_WS_PUSH_INTERVAL",
    "DEFAULT_STRATEGY", "TIER1_ENABLED", "SESSION_OVERRIDE_ENABLED",
    "ALL_GATES_OVERRIDE_ENABLED", "SPREAD_GATE_OVERRIDE_ENABLED",
    "COMPRESSION_GATE_OVERRIDE_ENABLED", "EXECUTION_GATE_OVERRIDE_ENABLED",
    "TIME_GATE_OVERRIDE_ENABLED", "SPREAD_MEAN_GATE_OVERRIDE_ENABLED",
    "SPREAD_VOLATILITY_GATE_OVERRIDE_ENABLED", "SPREAD_PERCENTILE_GATE_OVERRIDE_ENABLED",
    "SPREAD_DELTA_GATE_OVERRIDE_ENABLED", "COMPRESSION_RANGE_GATE_OVERRIDE_ENABLED",
    "ATR_RISING_GATE_OVERRIDE_ENABLED", "EXECUTION_QUALITY_GATE_OVERRIDE_ENABLED",
    "TICK_DIRECTION_GATE_OVERRIDE_ENABLED", "POST_SIGNAL_SPREAD_GATE_OVERRIDE_ENABLED",
    "SMC_SPREAD_MEAN_MAX", "SMC_SPREAD_STD_MAX", "SMC_SPREAD_PERCENTILE_MAX",
    "SMC_CURRENT_SPREAD_DELTA_MAX", "SCALPER_SPREAD_MEAN_MAX",
    "SCALPER_SPREAD_STD_MAX", "SCALPER_SPREAD_PERCENTILE_MAX",
    "SCALPER_CURRENT_SPREAD_DELTA_MAX", "COMPRESSION_RANGE_LOOKBACK",
    "COMPRESSION_ATR_MULTIPLIER", "SMC_THRESHOLD_WITH_TREND",
    "SMC_THRESHOLD_COUNTER", "SMC_THRESHOLD_COUNTER_MAX", "SMC_MIN_RR",
    "SMC_TIMEFRAME_EMA_SLOPE_MIN", "SMC_LTF_TICK_CONFLICT_LONG_MAX",
    "SMC_LTF_TICK_CONFLICT_SHORT_MIN",
    "SMC_RANGING_THRESHOLD_OFFSET", "SMC_TRENDING_THRESHOLD_OFFSET",
    "SMC_EARLY_FAIL_POINTS", "SMC_EARLY_FAIL_RANGING", "SMC_EARLY_FAIL_TRENDING",
    "SMC_EARLY_FAIL_CONF_HIGH_MULT", "SMC_EARLY_FAIL_CONF_LOW_MULT",
    "SMC_TIER1_MIN_TICKS", "SMC_TIER1_MAX_TICKS", "SMC_TIER1_SPREAD_DELTA_MAX",
    "SMC_HIGH_CONF_THRESHOLD", "SMC_HIGH_CONF_RR_MULTIPLIER", "SMC_HIGH_CONF_RISK_MULTIPLIER",
    "ADAPT_THRESHOLD_WTR_MIN", "ADAPT_THRESHOLD_WTR_MAX",
    "ADAPT_THRESHOLD_CTR_MIN", "ADAPT_THRESHOLD_CTR_MAX",
    "ADAPT_MIN_RR_MIN", "ADAPT_MIN_RR_MAX",
    "ADAPT_SESSION_TRADES_MIN", "ADAPT_SESSION_TRADES_MAX",
    "ADAPT_COOLDOWN_MIN", "ADAPT_COOLDOWN_MAX",
    "SCALPER_QUALITY_THRESHOLD",
    "SCALPER_ATR_MIN", "SCALPER_ATR_MAX", "SCALPER_EMA20_SLOPE_MIN",
    "SCALPER_BODY_RATIO_MIN", "SCALPER_TICK_DIR_THRESHOLD",
    "SCALPER_SWEEP_LOOKBACK", "SCALPER_SWEEP_TOLERANCE",
    "SCALPER_MAX_TRADES_SESSION", "SCALPER_LEVEL_COOLDOWN",
    "TRADE_WINDOW_LONDON_START", "TRADE_WINDOW_LONDON_END",
    "TRADE_WINDOW_OVERLAP_START", "TRADE_WINDOW_OVERLAP_END",
}
_DEFAULT_RUNTIME_CONFIG = {
    key: globals()[key]
    for key in sorted(_PERSISTED_KEYS)
    if key in globals()
}

_PROFILE_ROLES = {"live", "backtest"}
_PROFILE_SCOPES = {"shared", "live", "backtest"}
_PROFILE_ACTIVE = {"live": "live", "backtest": "live"}
_PROFILE_CACHE = {}


def get_runtime_defaults() -> dict:
    """Return code defaults before runtime_config.json overrides are applied."""
    return dict(_DEFAULT_RUNTIME_CONFIG)


def _sanitize_profile_name(name: str) -> str:
    text = str(name or "").strip()
    if not text:
        raise ValueError("Profile name is required")
    safe = []
    for ch in text:
        safe.append(ch if ch.isalnum() or ch in ("_", "-", ".") else "_")
    normalized = "".join(safe).strip("._-")
    if not normalized:
        raise ValueError("Profile name must contain letters or numbers")
    return normalized[:64]


def _normalize_profile_values(values: dict) -> dict:
    normalized = {}
    for key, value in (values or {}).items():
        if key in _PERSISTED_KEYS and key in globals():
            normalized[key] = value
    return normalized


def _default_profiles_payload() -> dict:
    defaults = get_runtime_defaults()
    return {
        "version": 1,
        "active_profiles": {"live": "live", "backtest": "live"},
        "profiles": {
            "live": {
                "name": "live",
                "description": "Current live trading profile",
                "scope": "live",
                "values": defaults,
            }
        },
    }


def _normalize_profile_scope(scope: str | None, fallback_name: str | None = None) -> str:
    text = str(scope or "").strip().lower()
    if text in _PROFILE_SCOPES:
        return text
    name = str(fallback_name or "").strip().lower()
    if name == "live":
        return "live"
    if name.startswith("backtest"):
        return "backtest"
    return "shared"


def _profile_matches_role(scope: str, role: str) -> bool:
    return scope == "shared" or scope == role


def _migrate_legacy_runtime_config() -> dict:
    payload = _default_profiles_payload()
    if not _CONFIG_STATE_FILE.exists():
        return payload
    try:
        legacy = json.loads(_CONFIG_STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return payload
    live_values = dict(payload["profiles"]["live"]["values"])
    for key, value in legacy.items():
        if key in _PERSISTED_KEYS and key in live_values:
            live_values[key] = value
    payload["profiles"]["live"]["values"] = live_values
    return payload


def _load_profiles_payload() -> dict:
    global _PROFILE_CACHE, _PROFILE_ACTIVE
    payload = None
    if _CONFIG_PROFILES_FILE.exists():
        try:
            payload = json.loads(_CONFIG_PROFILES_FILE.read_text(encoding="utf-8"))
        except Exception:
            payload = None
    if not isinstance(payload, dict):
        payload = _migrate_legacy_runtime_config()

    profiles = payload.get("profiles", {}) if isinstance(payload, dict) else {}
    if not isinstance(profiles, dict) or not profiles:
        profiles = _default_profiles_payload()["profiles"]

    normalized_profiles = {}
    for raw_name, raw_profile in profiles.items():
        try:
            name = _sanitize_profile_name(raw_name)
        except ValueError:
            continue
        values = _normalize_profile_values((raw_profile or {}).get("values", {}))
        if not values:
            values = get_runtime_defaults()
        normalized_profiles[name] = {
            "name": name,
            "description": str((raw_profile or {}).get("description", "")).strip(),
            "scope": _normalize_profile_scope((raw_profile or {}).get("scope"), fallback_name=name),
            "values": values,
        }
    if "live" not in normalized_profiles:
        normalized_profiles["live"] = {
            "name": "live",
            "description": "Current live trading profile",
            "scope": "live",
            "values": get_runtime_defaults(),
        }

    active_profiles = payload.get("active_profiles", {}) if isinstance(payload, dict) else {}
    normalized_active = {}
    for role in _PROFILE_ROLES:
        candidate = str(active_profiles.get(role, "")).strip() or ("live" if role == "live" else "live")
        if candidate not in normalized_profiles:
            candidate = "live"
        normalized_active[role] = candidate

    _PROFILE_CACHE = normalized_profiles
    _PROFILE_ACTIVE = normalized_active
    return {
        "version": 1,
        "active_profiles": dict(_PROFILE_ACTIVE),
        "profiles": copy.deepcopy(_PROFILE_CACHE),
    }


def _save_profiles_payload(payload: dict) -> dict:
    global _PROFILE_CACHE, _PROFILE_ACTIVE
    _CONFIG_PROFILES_FILE.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _PROFILE_CACHE = copy.deepcopy(payload.get("profiles", {}))
    _PROFILE_ACTIVE = dict(payload.get("active_profiles", {}))
    return payload


def list_runtime_profiles(role: str | None = None) -> dict:
    """Return persisted runtime profiles and active assignments."""
    payload = _load_profiles_payload()
    role_name = str(role or "").strip().lower()
    if role_name and role_name not in _PROFILE_ROLES:
        raise ValueError(f"Unknown profile role: {role_name}")
    profiles = copy.deepcopy(payload["profiles"])
    if role_name:
        profiles = {
            name: profile
            for name, profile in profiles.items()
            if _profile_matches_role(str(profile.get("scope", "shared")), role_name)
        }
    return {
        "active_profiles": dict(payload["active_profiles"]),
        "profiles": profiles,
        "profiles_by_role": {
            scoped_role: {
                name: copy.deepcopy(profile)
                for name, profile in payload["profiles"].items()
                if _profile_matches_role(str(profile.get("scope", "shared")), scoped_role)
            }
            for scoped_role in sorted(_PROFILE_ROLES)
        },
    }


def get_active_profile(role: str = "live") -> str:
    """Return the active profile name for the requested role."""
    role = str(role or "live").strip().lower()
    if role not in _PROFILE_ROLES:
        raise ValueError(f"Unknown profile role: {role}")
    _load_profiles_payload()
    return str(_PROFILE_ACTIVE.get(role, "live"))


def get_profile_values(name: str, role: str | None = None) -> dict:
    """Return persisted values for a named profile."""
    profile_name = _sanitize_profile_name(name)
    payload = _load_profiles_payload()
    profile = payload["profiles"].get(profile_name)
    if not profile:
        raise ValueError(f"Unknown profile: {profile_name}")
    if role:
        role_name = str(role).strip().lower()
        if role_name not in _PROFILE_ROLES:
            raise ValueError(f"Unknown profile role: {role_name}")
        if not _profile_matches_role(str(profile.get("scope", "shared")), role_name):
            raise ValueError(f"Profile {profile_name} is not available for role: {role_name}")
    return dict(profile.get("values", {}))


def _apply_runtime_values(values: dict) -> dict:
    applied = {}
    for key, value in (values or {}).items():
        if key in _PERSISTED_KEYS and key in globals():
            globals()[key] = value
            applied[key] = value
    return applied


def apply_runtime_profile(name: str, persist_role: str | None = None) -> dict:
    """Apply a named profile to in-memory config, optionally persisting a role selection."""
    profile_name = _sanitize_profile_name(name)
    payload = _load_profiles_payload()
    profile = payload["profiles"].get(profile_name)
    if not profile:
        raise ValueError(f"Unknown profile: {profile_name}")
    applied = _apply_runtime_values(profile.get("values", {}))
    if persist_role:
        role = str(persist_role).strip().lower()
        if role not in _PROFILE_ROLES:
            raise ValueError(f"Unknown profile role: {role}")
        if not _profile_matches_role(str(profile.get("scope", "shared")), role):
            raise ValueError(f"Profile {profile_name} is not available for role: {role}")
        payload["active_profiles"][role] = profile_name
        _save_profiles_payload(payload)
    return applied


def set_active_profile(role: str, name: str, apply_now: bool = False) -> dict:
    """Assign a profile to a role; optionally apply immediately to in-memory globals."""
    role_name = str(role or "").strip().lower()
    if role_name not in _PROFILE_ROLES:
        raise ValueError(f"Unknown profile role: {role_name}")
    profile_name = _sanitize_profile_name(name)
    payload = _load_profiles_payload()
    if profile_name not in payload["profiles"]:
        raise ValueError(f"Unknown profile: {profile_name}")
    profile = payload["profiles"][profile_name]
    if not _profile_matches_role(str(profile.get("scope", "shared")), role_name):
        raise ValueError(f"Profile {profile_name} is not available for role: {role_name}")
    payload["active_profiles"][role_name] = profile_name
    _save_profiles_payload(payload)
    if apply_now:
        return apply_runtime_profile(profile_name)
    return dict(payload["active_profiles"])


def save_runtime_profile(
    name: str | None = None,
    description: str | None = None,
    role_scope: str | None = None,
) -> dict:
    """Save current in-memory config into a named profile."""
    payload = _load_profiles_payload()
    profile_name = _sanitize_profile_name(name or get_active_profile("live"))
    current_values = {
        key: globals()[key]
        for key in sorted(_PERSISTED_KEYS)
        if key in globals() and key not in _ADAPTIVE_MANAGED_KEYS
    }
    existing = payload["profiles"].get(profile_name, {})
    scope = _normalize_profile_scope(
        role_scope if role_scope is not None else existing.get("scope"),
        fallback_name=profile_name,
    )
    payload["profiles"][profile_name] = {
        "name": profile_name,
        "description": str(description if description is not None else existing.get("description", "")).strip(),
        "scope": scope,
        "values": current_values,
    }
    _save_profiles_payload(payload)
    if profile_name == get_active_profile("live"):
        _CONFIG_STATE_FILE.write_text(
            json.dumps(current_values, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return copy.deepcopy(payload["profiles"][profile_name])


def create_runtime_profile(
    name: str,
    base_profile: str | None = None,
    description: str = "",
    role_scope: str | None = None,
) -> dict:
    """Create a new profile from defaults or from another profile."""
    payload = _load_profiles_payload()
    profile_name = _sanitize_profile_name(name)
    if profile_name in payload["profiles"]:
        raise ValueError(f"Profile already exists: {profile_name}")
    if base_profile:
        values = get_profile_values(base_profile)
    else:
        values = get_runtime_defaults()
    payload["profiles"][profile_name] = {
        "name": profile_name,
        "description": str(description or "").strip(),
        "scope": _normalize_profile_scope(role_scope, fallback_name=profile_name),
        "values": values,
    }
    _save_profiles_payload(payload)
    return copy.deepcopy(payload["profiles"][profile_name])


def delete_runtime_profile(name: str) -> dict:
    """Delete a saved profile if it is not active."""
    payload = _load_profiles_payload()
    profile_name = _sanitize_profile_name(name)
    if profile_name == "live":
        raise ValueError("The live profile cannot be deleted")
    if profile_name not in payload["profiles"]:
        raise ValueError(f"Unknown profile: {profile_name}")
    if profile_name in payload["active_profiles"].values():
        raise ValueError(f"Profile is active and cannot be deleted: {profile_name}")
    del payload["profiles"][profile_name]
    _save_profiles_payload(payload)
    return {"deleted": profile_name}


def update_runtime_profile(
    name: str,
    values: dict | None = None,
    description: str | None = None,
    role_scope: str | None = None,
) -> dict:
    """Update a profile's persisted values and/or description."""
    payload = _load_profiles_payload()
    profile_name = _sanitize_profile_name(name)
    if profile_name not in payload["profiles"]:
        raise ValueError(f"Unknown profile: {profile_name}")
    profile = payload["profiles"][profile_name]
    merged_values = dict(profile.get("values", {}))
    merged_values.update(_normalize_profile_values(values or {}))
    profile["values"] = merged_values
    if description is not None:
        profile["description"] = str(description).strip()
    if role_scope is not None:
        profile["scope"] = _normalize_profile_scope(role_scope, fallback_name=profile_name)
    else:
        profile["scope"] = _normalize_profile_scope(profile.get("scope"), fallback_name=profile_name)
    payload["profiles"][profile_name] = profile
    _save_profiles_payload(payload)
    return copy.deepcopy(profile)


def copy_profile_to_profile(
    source_name: str,
    target_name: str,
    target_scope: str | None = None,
    apply_if_active_live: bool = False,
) -> dict:
    """Copy all persisted values from one profile into another."""
    payload = _load_profiles_payload()
    source_profile_name = _sanitize_profile_name(source_name)
    target_profile_name = _sanitize_profile_name(target_name)
    source = payload["profiles"].get(source_profile_name)
    if not source:
        raise ValueError(f"Unknown source profile: {source_profile_name}")
    target = payload["profiles"].get(target_profile_name)
    if not target:
        raise ValueError(f"Unknown target profile: {target_profile_name}")
    target["values"] = dict(source.get("values", {}))
    target["scope"] = _normalize_profile_scope(
        target_scope if target_scope is not None else target.get("scope"),
        fallback_name=target_profile_name,
    )
    payload["profiles"][target_profile_name] = target
    _save_profiles_payload(payload)
    if apply_if_active_live and get_active_profile("live") == target_profile_name:
        apply_runtime_profile(target_profile_name)
    return copy.deepcopy(target)


def load_runtime_config(profile_name: str | None = None) -> dict:
    """Load a saved profile into in-memory globals."""
    target = profile_name or get_active_profile("live")
    return apply_runtime_profile(target)


# Keys managed by adaptive_params.py — written to cfg live each cycle but
# must NOT be persisted by save_runtime_config(), otherwise the adapted values
# overwrite the intended baseline in runtime_config.json.
_ADAPTIVE_MANAGED_KEYS = {
    "SMC_THRESHOLD_WITH_TREND", "SMC_THRESHOLD_COUNTER",
    "SMC_MIN_RR", "SESSION_MAX_TRADES", "MIN_TRADE_COOLDOWN",
    "SMC_SPREAD_MEAN_MAX", "SMC_EARLY_FAIL_RANGING", "SMC_EARLY_FAIL_TRENDING",
}


def save_runtime_config() -> dict:
    """Persist current config into the active live profile for the next restart."""
    profile = save_runtime_profile(get_active_profile("live"))
    return dict(profile.get("values", {}))


load_runtime_config()
