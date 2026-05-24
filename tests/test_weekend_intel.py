from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import config as cfg
from engine import trader_api
from engine.weekend_intel import WeekendIntelService


def _service(tmp_path: Path, *, enabled=True, alpha_key="alpha", x_token="x-token"):
    return WeekendIntelService(
        tmp_path,
        cfg,
        enabled=enabled,
        alpha_api_key=alpha_key,
        x_bearer_token=x_token,
        timeout_seconds=1.0,
    )


def _resp(payload):
    response = Mock()
    response.raise_for_status = Mock()
    response.json.return_value = payload
    return response


@patch("engine.weekend_intel.requests.get")
def test_weekend_intel_skips_external_calls_when_market_open(get_mock, tmp_path: Path):
    service = _service(tmp_path)

    snapshot = service.refresh_if_due({"symbol": "XAUUSD", "now_utc": "2026-05-24T22:30:00+00:00"}, force=True)

    assert snapshot["status"] == "idle"
    assert snapshot["last_refresh_reason"] == "market_open"
    assert get_mock.call_count == 0


@patch("engine.weekend_intel.requests.get")
def test_weekend_intel_collects_counts_and_skips_posts_without_spike(get_mock, tmp_path: Path):
    responses = [
        _resp({"markets": [{"market_type": "forex", "current_status": "closed"}]}),
        _resp({"feed": [{"title": "Macro headline", "summary": "summary", "source": "Desk", "time_published": "20260524T1000"}]}),
        _resp({"feed": [{"title": "USD headline", "summary": "summary", "source": "Desk", "time_published": "20260524T1015"}]}),
        _resp({"data": [{"date": "2026-05-23", "value": "2331.2"}]}),
        _resp({"data": [{"start": "2026-05-24T08:00:00Z", "end": "2026-05-24T09:00:00Z", "tweet_count": 8}], "meta": {"total_tweet_count": 8}}),
        _resp({"data": [{"start": "2026-05-24T08:00:00Z", "end": "2026-05-24T09:00:00Z", "tweet_count": 5}], "meta": {"total_tweet_count": 5}}),
        _resp({"data": [{"start": "2026-05-24T08:00:00Z", "end": "2026-05-24T09:00:00Z", "tweet_count": 6}], "meta": {"total_tweet_count": 6}}),
    ]
    get_mock.side_effect = responses
    service = _service(tmp_path)

    snapshot = service.refresh_if_due({"symbol": "XAUUSD", "now_utc": "2026-05-24T12:00:00+00:00"}, force=True)

    assert snapshot["status"] == "ready"
    assert len(snapshot["latest_x_counts"]) == 3
    assert snapshot["latest_x_posts"] == []
    assert snapshot["sample_count"] == 1
    called_urls = [call.args[0] for call in get_mock.call_args_list]
    assert "https://api.x.com/2/tweets/search/recent" not in called_urls


@patch("engine.weekend_intel.requests.get")
def test_weekend_intel_fetches_posts_in_preopen_window(get_mock, tmp_path: Path):
    responses = [
        _resp({"markets": [{"market_type": "forex", "current_status": "closed"}]}),
        _resp({"feed": []}),
        _resp({"feed": []}),
        _resp({"data": [{"date": "2026-05-23", "value": "2331.2"}]}),
        _resp({"data": [], "meta": {"total_tweet_count": 12}}),
        _resp({"data": [], "meta": {"total_tweet_count": 9}}),
        _resp({"data": [], "meta": {"total_tweet_count": 7}}),
        _resp({"data": [{"id": "1", "text": "Gold opening risk", "created_at": "2026-05-25T17:00:00Z", "public_metrics": {"like_count": 3}}]}),
        _resp({"data": [{"id": "2", "text": "Fed chatter before open", "created_at": "2026-05-25T17:05:00Z", "public_metrics": {"like_count": 2}}]}),
        _resp({"data": [{"id": "3", "text": "USD and yields moving", "created_at": "2026-05-25T17:10:00Z", "public_metrics": {"like_count": 1}}]}),
    ]
    get_mock.side_effect = responses
    service = _service(tmp_path)

    snapshot = service.refresh_if_due({"symbol": "XAUUSD", "now_utc": "2026-05-24T18:30:00+00:00"}, force=True)

    assert snapshot["status"] == "ready"
    assert snapshot["preopen_window"] is True
    assert len(snapshot["latest_x_posts"]) == 3
    called_urls = [call.args[0] for call in get_mock.call_args_list]
    assert called_urls.count("https://api.x.com/2/tweets/search/recent") == 3


def test_trader_api_status_builder_returns_weekend_intel_payload(tmp_path: Path):
    service = _service(tmp_path)
    service._snapshot = service._default_snapshot(status="ready")
    service._snapshot.update(
        {
            "enabled": True,
            "status": "ready",
            "generated_at": "2026-05-24T12:00:00+00:00",
            "sample_count": 2,
            "summary": "Collected weekend intel.",
            "latest_x_counts": [{"label": "gold_core", "total_count": 18}],
        }
    )
    previous = trader_api._auto_trader_instance
    trader_api._auto_trader_instance = SimpleNamespace(weekend_intel_service=service)
    try:
        payload = trader_api._build_weekend_intel_status()
    finally:
        trader_api._auto_trader_instance = previous

    assert payload["status"] == "ready"
    assert payload["sample_count"] == 2
    assert payload["latest_x_counts"][0]["total_count"] == 18


@patch.dict("os.environ", {"AI_AUTOMATION_ENABLED": "0"}, clear=False)
def test_weekend_intel_respects_global_automation_bypass(tmp_path: Path):
    service = _service(tmp_path)

    snapshot = service.refresh_if_due({"symbol": "XAUUSD", "now_utc": "2026-05-24T12:00:00+00:00"}, force=True)

    assert snapshot["status"] == "disabled"
    assert snapshot["last_refresh_reason"] == "automation_bypassed"


def test_weekend_intel_disables_when_all_sources_are_toggled_off(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(cfg, "WEEKEND_INTEL_ENABLED", True, raising=False)
    monkeypatch.setattr(cfg, "WEEKEND_INTEL_ALPHA_ENABLED", False, raising=False)
    monkeypatch.setattr(cfg, "WEEKEND_INTEL_X_ENABLED", False, raising=False)
    service = _service(tmp_path)

    snapshot = service.refresh_if_due({"symbol": "XAUUSD", "now_utc": "2026-05-24T12:00:00+00:00"}, force=True)

    assert snapshot["status"] == "disabled"
    assert snapshot["last_refresh_reason"] == "source_disabled"
