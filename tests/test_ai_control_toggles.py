from engine.ai_analysis import ai_analysis_service
from engine.ai_manual_trade_ideas import ai_manual_trade_idea_service
import config as cfg


def test_ai_analysis_respects_dashboard_toggle(monkeypatch):
    monkeypatch.setattr(cfg, "AI_ANALYSIS_ENABLED", False, raising=False)

    payload = ai_analysis_service.generate_summary(days=1, refresh=False)

    assert payload["status"] == "disabled"
    assert payload["enabled"] is False


def test_manual_ai_trade_ideas_respect_dashboard_toggle(monkeypatch):
    monkeypatch.setattr(cfg, "AI_MANUAL_TRADE_IDEAS_ENABLED", False, raising=False)

    payload = ai_manual_trade_idea_service.generate_trade_idea({}, {"bid": 2300.0, "ask": 2300.2})

    assert payload["status"] == "disabled"
    assert payload["enabled"] is False
