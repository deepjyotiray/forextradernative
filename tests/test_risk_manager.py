from datetime import datetime, timedelta, timezone

from engine.risk_manager import RiskManager


def _today_trade(pnl: float, *, seconds_ago: int = 0) -> dict:
    close_time = datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)
    return {
        "pnl": pnl,
        "close_time": close_time.isoformat(),
    }


def test_hard_reset_blocks_mt5_reseed_from_restoring_old_loss_streak_and_daily_pnl():
    risk = RiskManager()
    closed_trades = [
        _today_trade(-0.8, seconds_ago=120),
        _today_trade(-1.6, seconds_ago=60),
    ]

    risk.seed_from_mt5({"pnl": -2.4, "trades": 2}, closed_trades)
    assert risk.daily_status["consecutive_losses"] == 2
    assert risk.daily_status["pnl"] == -2.4
    assert risk.daily_status["trades"] == 2

    risk.reset_gates(hard=True)
    risk.seed_from_mt5({"pnl": -2.4, "trades": 2}, closed_trades)

    assert risk.daily_status["consecutive_losses"] == 0
    assert risk.daily_status["pnl"] == 0.0
    assert risk.daily_status["trades"] == 0

    allowed, reason = risk.can_trade({"balance": 1000.0, "equity": 1000.0}, 0)
    assert allowed is True, reason


def test_soft_reset_keeps_daily_pnl_but_ignores_old_losses_for_streak_rebuild():
    risk = RiskManager()
    closed_trades = [
        _today_trade(-0.8, seconds_ago=120),
        _today_trade(-1.6, seconds_ago=60),
    ]

    risk.seed_from_mt5({"pnl": -2.4, "trades": 2}, closed_trades)
    assert risk.daily_status["consecutive_losses"] == 2

    risk.reset_gates(hard=False)
    risk.seed_from_mt5({"pnl": -2.4, "trades": 2}, closed_trades)

    assert risk.daily_status["consecutive_losses"] == 0
    assert risk.daily_status["pnl"] == -2.4
    assert risk.daily_status["trades"] == 2

    allowed, reason = risk.can_trade({"balance": 1000.0, "equity": 1000.0}, 0)
    assert allowed is True, reason
