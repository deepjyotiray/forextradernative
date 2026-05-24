from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import os

import config as cfg
from engine.ai_market_bias import AIMarketBiasService, _coerce_utc_datetime
from engine import trader_api


def _service(tmp_path: Path, *, enabled=True, alpha_key="alpha", openai_key="openai"):
    cfg.AI_AUTOMATION_ENABLED = True
    return AIMarketBiasService(
        tmp_path,
        cfg,
        enabled=enabled,
        alpha_api_key=alpha_key,
        openai_api_key=openai_key,
        model="gpt-test",
        endpoint="https://api.openai.com/v1",
        timeout_seconds=1.0,
    )


def test_disabled_service_fails_open_without_credentials(tmp_path: Path):
    service = _service(tmp_path, enabled=False, alpha_key="", openai_key="")

    snapshot = service.refresh_if_due({"symbol": "XAUUSD", "now_utc": "2026-05-21T12:00:00+00:00"}, force=True)
    result = service.apply_to_signal(
        "SMC_CONFLUENCE",
        {"signal": "BUY", "sl": 99.0, "tp": 101.0},
        {"symbol": "XAUUSD"},
    )

    assert snapshot["status"] == "disabled"
    assert result["allowed"] is True
    assert result["adjusted_signal"]["signal"] == "BUY"


@patch("engine.ai_market_bias.requests.get")
def test_alpha_vantage_failure_returns_fail_open_behavior(get_mock, tmp_path: Path):
    get_mock.side_effect = RuntimeError("timeout")
    service = _service(tmp_path)

    snapshot = service.refresh_if_due({"symbol": "XAUUSD", "now_utc": "2026-05-21T12:00:00+00:00"}, force=True)
    result = service.apply_to_signal(
        "SMC_CONFLUENCE",
        {"signal": "SELL", "sl": 101.0, "tp": 99.0},
        {"symbol": "XAUUSD"},
    )

    assert snapshot["status"] in {"error", "stale"}
    assert result["allowed"] is True


def test_long_snapshot_blocks_sell_only_above_threshold(tmp_path: Path):
    service = _service(tmp_path)
    service._snapshot = service._default_snapshot(status="ready")
    service._snapshot.update(
        {
            "enabled": True,
            "status": "ready",
            "generated_at": "2099-05-24T10:00:00+00:00",
            "expires_at": "2099-05-24T10:30:00+00:00",
            "directional_bias": "LONG",
            "confidence": 0.80,
            "risk_mode": "NORMAL",
            "summary": "Gold macro/news bias is strongly long.",
        }
    )

    sell = service.apply_to_signal("SMC_CONFLUENCE", {"signal": "SELL"}, {"symbol": "XAUUSD"})
    buy = service.apply_to_signal("SMC_CONFLUENCE", {"signal": "BUY"}, {"symbol": "XAUUSD"})

    assert sell["allowed"] is False
    assert "blocked SHORT trades" in sell["reason"]
    assert buy["allowed"] is True


def test_expired_snapshot_is_ignored(tmp_path: Path):
    service = _service(tmp_path)
    service._snapshot = service._default_snapshot(status="ready")
    service._snapshot.update(
        {
            "enabled": True,
            "status": "ready",
            "generated_at": "2020-05-24T09:00:00+00:00",
            "expires_at": "2020-05-24T09:05:00+00:00",
            "directional_bias": "SHORT",
            "confidence": 0.99,
            "risk_mode": "BLOCK_ALL",
        }
    )

    result = service.apply_to_signal("SMC_CONFLUENCE", {"signal": "BUY"}, {"symbol": "XAUUSD"})

    assert result["allowed"] is True
    assert result["adjusted_signal"]["signal"] == "BUY"


@patch("engine.ai_market_bias.requests.post")
@patch("engine.ai_market_bias.requests.get")
def test_malformed_openai_response_is_handled_safely(get_mock, post_mock, tmp_path: Path):
    ok_response = Mock()
    ok_response.raise_for_status = Mock()
    ok_response.json.return_value = {"data": [], "feed": [], "markets": []}
    get_mock.return_value = ok_response

    bad_response = Mock()
    bad_response.raise_for_status = Mock()
    bad_response.json.return_value = {"output_text": "not json"}
    post_mock.return_value = bad_response

    service = _service(tmp_path)
    snapshot = service.refresh_if_due({"symbol": "XAUUSD", "now_utc": "2026-05-21T12:00:00+00:00"}, force=True)

    assert snapshot["status"] in {"error", "stale"}
    assert snapshot["directional_bias"] == "NEUTRAL"


def test_trader_api_status_builder_returns_market_bias_payload(tmp_path: Path):
    service = _service(tmp_path)
    service._snapshot = service._default_snapshot(status="ready")
    service._snapshot.update(
        {
            "enabled": True,
            "status": "ready",
            "generated_at": "2026-05-24T10:00:00+00:00",
            "expires_at": "2026-05-24T10:15:00+00:00",
            "directional_bias": "LONG",
            "confidence": 0.77,
            "risk_mode": "REDUCED",
            "risk_multiplier": 0.7,
            "summary": "Reduce risk while long bias remains intact.",
        }
    )
    previous = trader_api._auto_trader_instance
    trader_api._auto_trader_instance = SimpleNamespace(ai_market_bias_service=service)
    try:
        payload = trader_api._build_ai_market_bias_status()
    finally:
        trader_api._auto_trader_instance = previous

    assert payload["status"] == "ready"
    assert payload["directional_bias"] == "LONG"
    assert payload["risk_mode"] == "REDUCED"
    assert payload["risk_multiplier"] == 0.7


@patch("engine.ai_market_bias.requests.post")
@patch("engine.ai_market_bias.requests.get")
def test_market_bias_does_not_call_external_services_when_market_closed(get_mock, post_mock, tmp_path: Path):
    service = _service(tmp_path)
    snapshot = service.refresh_if_due(
        {"symbol": "XAUUSD", "now_utc": "2026-05-23T22:30:00+00:00"},
        force=True,
    )

    assert snapshot["last_refresh_reason"] == "market_closed"
    assert get_mock.call_count == 0
    assert post_mock.call_count == 0


@patch.dict(os.environ, {"OPENAI_API_KEY": "", "ALPHAVANTAGE_API_KEY": ""}, clear=False)
def test_service_loads_local_env_file_on_init(tmp_path: Path):
    env_path = tmp_path / ".ai_trade_correction.local.env"
    env_path.write_text(
        "OPENAI_API_KEY=test-openai-key\nALPHAVANTAGE_API_KEY=test-alpha-key\n",
        encoding="utf-8",
    )

    service = AIMarketBiasService(
        tmp_path,
        cfg,
        enabled=True,
        model="gpt-test",
        endpoint="https://api.openai.com/v1",
        timeout_seconds=1.0,
    )

    assert service.openai_api_key == "test-openai-key"
    assert service.alpha_api_key == "test-alpha-key"


@patch.dict(os.environ, {"AI_AUTOMATION_ENABLED": "0"}, clear=False)
def test_market_bias_respects_global_automation_bypass(tmp_path: Path):
    service = _service(tmp_path)

    snapshot = service.refresh_if_due({"symbol": "XAUUSD", "now_utc": "2026-05-21T12:00:00+00:00"}, force=True)
    result = service.apply_to_signal(
        "SMC_CONFLUENCE",
        {"signal": "BUY", "sl": 99.0, "tp": 101.0},
        {"symbol": "XAUUSD"},
    )

    assert snapshot["status"] == "disabled"
    assert snapshot["last_refresh_reason"] == "automation_bypassed"
    assert result["allowed"] is True


@patch.dict(os.environ, {"AI_AUTOMATION_ENABLED": "1"}, clear=False)
def test_market_bias_respects_source_toggles(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(cfg, "AI_MARKET_BIAS_ENABLED", True, raising=False)
    monkeypatch.setattr(cfg, "AI_MARKET_BIAS_ALPHA_ENABLED", False, raising=False)
    monkeypatch.setattr(cfg, "AI_MARKET_BIAS_OPENAI_ENABLED", True, raising=False)
    service = _service(tmp_path)

    snapshot = service.refresh_if_due({"symbol": "XAUUSD", "now_utc": "2026-05-21T12:00:00+00:00"}, force=True)

    assert snapshot["status"] == "disabled"
    assert snapshot["last_refresh_reason"] == "source_disabled"
    assert "Alpha Vantage" in snapshot["last_error"]


def test_collect_context_includes_compact_weekend_intel(tmp_path: Path):
    service = _service(tmp_path)
    service._alpha_get = Mock(
        side_effect=[
            {"data": {"symbol": "GOLD", "price": "2330.1"}},
            {"data": [{"date": "2026-05-22", "value": "2331.2"}]},
            {"feed": []},
            {"feed": []},
            {"markets": []},
        ]
    )

    context = service._collect_context(
        {
            "symbol": "XAUUSD",
            "session": "LONDON",
            "tick": {"bid": 2330.0, "ask": 2330.2, "spread": 0.2},
            "weekend_intel": {
                "status": "ready",
                "summary": "Collected weekend intel.",
                "sample_count": 3,
                "preopen_window": True,
                "latest_x_counts": [{"label": "gold_core", "total_count": 42}],
                "latest_x_posts": [{"query_label": "gold_core", "text": "Gold chatter rising.", "like_count": 5}],
                "latest_alpha_news": [{"title": "Gold steady before open", "source": "Desk"}],
                "latest_gold_history": [{"date": "2026-05-22", "close": 2331.2}],
            },
        },
        _coerce_utc_datetime("2026-05-25T01:00:00+00:00"),
    )

    weekend_intel = context["external_market"]["weekend_intel"]
    assert weekend_intel["status"] == "ready"
    assert weekend_intel["sample_count"] == 3
    assert weekend_intel["preopen_window"] is True
    assert weekend_intel["latest_x_counts"][0]["total_count"] == 42
