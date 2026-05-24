from unittest.mock import Mock, patch

import config as cfg
from engine.ai_manual_trade_ideas import AIManualTradeIdeaService


def _service():
    return AIManualTradeIdeaService(
        api_key="openai-test",
        enabled=True,
        model="gpt-test",
        endpoint="https://api.openai.com/v1",
        timeout_seconds=1.0,
    )


@patch("engine.ai_manual_trade_ideas.requests.post")
def test_manual_trade_idea_uses_openai_responses_api(post_mock):
    response = Mock()
    response.raise_for_status = Mock()
    response.json.return_value = {
        "output_text": """{
          "action": "BUY",
          "confidence": 0.82,
          "entry": 2330.5,
          "sl": 2328.9,
          "tp": 2334.1,
          "rr": 2.25,
          "setup_type": "LIMIT",
          "reasoning": "Demand reaction and positive pressure align.",
          "invalidation": "Lose the local demand.",
          "checklist": ["Watch spread", "Confirm pressure"]
        }"""
    }
    post_mock.return_value = response
    service = _service()

    result = service.generate_trade_idea({"symbol": "XAUUSD", "timeframe": "M5"}, {"bid": 2330.3, "ask": 2330.5})

    assert result["status"] == "ready"
    assert result["provider"] == "openai"
    assert result["action"] == "BUY"
    assert result["pending_order_type"] in {"BUY_LIMIT", "BUY_STOP"}
    assert post_mock.call_args.args[0] == "https://api.openai.com/v1/responses"


def test_manual_trade_idea_dashboard_toggle_disables_service(monkeypatch):
    monkeypatch.setattr(cfg, "AI_MANUAL_TRADE_IDEAS_ENABLED", False, raising=False)
    service = _service()

    result = service.generate_trade_idea({}, {"bid": 2300.0, "ask": 2300.2})

    assert result["status"] == "disabled"
    assert result["provider"] == "openai"
