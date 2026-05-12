from datetime import datetime, timezone
from unittest.mock import patch

from engine.anti_starvation import AntiStarvationManager


def test_starvation_rechecks_within_same_session_after_30_minutes():
    manager = AntiStarvationManager()
    manager._session_trades = {"2026-05-12_LONDON": 0}
    manager._current_date = "2026-05-12"
    manager._current_session = "LONDON"
    manager._relaxation_active = False
    manager._relaxation_type = None

    now_utc = datetime(2026, 5, 12, 7, 31, tzinfo=timezone.utc)

    with patch("engine.anti_starvation.is_backtest_mode", return_value=True):
        with patch("engine.anti_starvation.is_market_open", return_value=True):
            with patch("engine.anti_starvation.get_session", return_value="LONDON"):
                with patch("engine.anti_starvation._now_utc", return_value=now_utc):
                    manager.update_session()

    assert manager._relaxation_active is True
    assert manager._relaxation_type in {"tick_ratio", "spread_tolerance"}
