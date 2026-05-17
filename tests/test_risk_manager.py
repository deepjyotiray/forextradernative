from datetime import datetime, timedelta, timezone

import config as cfg

from engine.risk_manager import RiskManager


def _today_trade(pnl: float, *, seconds_ago: int = 0) -> dict:
    close_time = datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)
    return {
        "pnl": pnl,
        "close_time": close_time.isoformat(),
    }


def test_hard_reset_blocks_mt5_reseed_from_restoring_old_loss_streak_and_daily_pnl():
    risk = RiskManager()
    original_temp_disable = cfg.TEMP_DISABLE_GLOBAL_BLOCKS
    cfg.TEMP_DISABLE_GLOBAL_BLOCKS = False
    closed_trades = [
        _today_trade(-0.8, seconds_ago=120),
        _today_trade(-1.6, seconds_ago=60),
    ]

    try:
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
    finally:
        cfg.TEMP_DISABLE_GLOBAL_BLOCKS = original_temp_disable


def test_soft_reset_keeps_daily_pnl_but_ignores_old_losses_for_streak_rebuild():
    risk = RiskManager()
    original_temp_disable = cfg.TEMP_DISABLE_GLOBAL_BLOCKS
    cfg.TEMP_DISABLE_GLOBAL_BLOCKS = False
    closed_trades = [
        _today_trade(-0.8, seconds_ago=120),
        _today_trade(-1.6, seconds_ago=60),
    ]

    try:
        risk.seed_from_mt5({"pnl": -2.4, "trades": 2}, closed_trades)
        assert risk.daily_status["consecutive_losses"] == 2

        risk.reset_gates(hard=False)
        risk.seed_from_mt5({"pnl": -2.4, "trades": 2}, closed_trades)

        assert risk.daily_status["consecutive_losses"] == 0
        assert risk.daily_status["pnl"] == -2.4
        assert risk.daily_status["trades"] == 2

        allowed, reason = risk.can_trade({"balance": 1000.0, "equity": 1000.0}, 0)
        assert allowed is True, reason
    finally:
        cfg.TEMP_DISABLE_GLOBAL_BLOCKS = original_temp_disable


def test_global_open_trade_cap_does_not_block_different_strategy_when_independent_slots_enabled():
    risk = RiskManager()
    original_allow = cfg.ALLOW_CONCURRENT_STRATEGY_POSITIONS
    original_max_open = cfg.MAX_OPEN_TRADES
    original_max_positions = cfg.MAX_POSITIONS
    original_temp_disable = cfg.TEMP_DISABLE_GLOBAL_BLOCKS
    try:
        cfg.TEMP_DISABLE_GLOBAL_BLOCKS = False
        cfg.ALLOW_CONCURRENT_STRATEGY_POSITIONS = True
        cfg.MAX_OPEN_TRADES = 1
        cfg.MAX_POSITIONS = 1

        allowed, reason = risk.can_trade(
            {"balance": 1000.0, "equity": 1000.0},
            1,
            strategy_name="M15_SCALP_DEEP",
            strategy_trade_counts={"SMC_CONFLUENCE": 1, "M15_SCALP_DEEP": 0},
        )

        assert allowed is True, reason
    finally:
        cfg.TEMP_DISABLE_GLOBAL_BLOCKS = original_temp_disable
        cfg.ALLOW_CONCURRENT_STRATEGY_POSITIONS = original_allow
        cfg.MAX_OPEN_TRADES = original_max_open
        cfg.MAX_POSITIONS = original_max_positions


def test_temp_disable_global_blocks_bypasses_shared_risk_checks():
    risk = RiskManager()
    original_temp_disable = cfg.TEMP_DISABLE_GLOBAL_BLOCKS
    original_max_open = cfg.MAX_OPEN_TRADES
    try:
        cfg.TEMP_DISABLE_GLOBAL_BLOCKS = True
        cfg.MAX_OPEN_TRADES = 0

        allowed, reason = risk.can_trade({"balance": 1000.0, "equity": 1000.0}, 99, strategy_name="SMC_CONFLUENCE")

        assert allowed is True
        assert reason == "GLOBAL_BLOCKS_DISABLED"
    finally:
        cfg.TEMP_DISABLE_GLOBAL_BLOCKS = original_temp_disable
        cfg.MAX_OPEN_TRADES = original_max_open
