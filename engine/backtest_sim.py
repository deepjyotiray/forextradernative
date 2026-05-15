"""
Tick + candle backtest simulator.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

import config as cfg
from .backtest_context import backtest_context
from .backtest_context import set_backtest_now
from .backtest_data import BacktestDataset
from .indicators import compute_indicators, compute_timeframe_context
from .liquidity import compute_liquidity
from .market_state import compute_market_state
from .mtf_bias import compute_bias
from .regime import classify_regime
from .session_filter import is_market_open, is_session_open_blocked
from .smc_strategy import SMCStrategy
from .m15_scalp_deep_strategy import M15ScalpDeepStrategy
from .m15_sr_strategy import M15SupportResistanceStrategy
from .m15_zone_scalp_strategy import M15ZoneScalpStrategy
from .strategy_manager import StrategyManager
from .sweep_scalper import SweepScalper
from .strategies.trend_channel_strategy import TrendChannelStrategy
from .tick_processor import TickProcessor, analyze_tick_pressure
from .zones import ZoneDetector
from .xgb_model import xgb_model


@dataclass
class BacktestRequest:
    symbol: str
    strategy: str
    start_utc: datetime
    end_utc: datetime
    initial_balance: float
    config_profile: str = ""
    use_runtime_config: bool = True
    max_ticks_per_batch: int = 2000
    replay_max_points: int = 12000
    fast_mode: bool = False
    split_mode: str = "NONE"
    parallel_workers: int = 1
    cache_only: bool = True

    @staticmethod
    def from_payload(payload: Dict) -> "BacktestRequest":
        def _dt(v: str) -> datetime:
            dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)

        return BacktestRequest(
            symbol=str(payload["symbol"]).upper(),
            strategy=str(payload["strategy"]).upper(),
            start_utc=_dt(payload["start_utc"]),
            end_utc=_dt(payload["end_utc"]),
            initial_balance=float(payload["initial_balance"]),
            config_profile=str(payload.get("config_profile") or cfg.get_active_profile("backtest")).strip(),
            use_runtime_config=bool(payload.get("use_runtime_config", True)),
            max_ticks_per_batch=max(200, int(payload.get("max_ticks_per_batch", 2000))),
            replay_max_points=max(1000, int(payload.get("replay_max_points", 12000))),
            fast_mode=bool(payload.get("fast_mode", False)),
            split_mode=str(payload.get("split_mode", "NONE")).upper(),
            parallel_workers=max(1, int(payload.get("parallel_workers", 1))),
            cache_only=bool(payload.get("cache_only", True)),
        )


@dataclass
class SimTrade:
    ticket: int
    symbol: str
    strategy: str
    direction: str  # BUY / SELL
    volume: float
    entry_price: float
    sl: float
    tp: float
    sl_distance: float
    open_time: datetime
    confidence: float = 0.0
    reason: str = ""
    scalp: bool = False
    be_trigger_price: float = 0.0
    timeout_seconds: int = 60
    early_fail_points: float = 0.0
    tier1_min_ticks: int = 3
    tier1_max_ticks: int = 15
    entry_tick_velocity: float = 0.0
    high_conf: bool = False
    peak_pnl: float = 0.0
    live_pnl: float = 0.0
    sl_breakeven: bool = False
    partial_closed: bool = False
    trail_active: bool = False
    close_reason: str = ""
    close_time: Optional[datetime] = None
    exit_price: float = 0.0
    manage_updates: int = 0
    entry_tick_count: int = 0
    realized_partial_pnl: float = 0.0
    remaining_volume: float = 0.0
    current_price: float = 0.0

    def __post_init__(self):
        if self.remaining_volume <= 0:
            self.remaining_volume = self.volume
        if self.current_price <= 0:
            self.current_price = self.entry_price


class BacktestClock:
    def __init__(self, start_utc: datetime):
        self.current_utc = start_utc.astimezone(timezone.utc)

    def set(self, dt: datetime):
        self.current_utc = dt.astimezone(timezone.utc)


class SimExecutionEngine:
    def __init__(self, initial_balance: float):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.equity = initial_balance
        self.peak_equity = initial_balance
        self.max_drawdown = 0.0
        self._ticket_counter = 1000000
        self.open_trades: Dict[int, SimTrade] = {}
        self.closed_trades: List[Dict] = []
        self.equity_curve: List[Dict] = []
        self._last_equity_ts: Optional[datetime] = None
        self._risk_pct = float(cfg.MAX_RISK_PCT)
        self._max_positions = int(cfg.MAX_POSITIONS)
        self._cooldown_seconds = float(cfg.MIN_TRADE_COOLDOWN)
        self._last_open_ts = 0.0

    def can_open(self, now_utc: datetime) -> Tuple[bool, str]:
        if len(self.open_trades) >= self._max_positions:
            return False, "Max positions reached"
        now_ts = now_utc.timestamp()
        if (now_ts - self._last_open_ts) < self._cooldown_seconds:
            return False, "Cooldown"
        return True, "OK"

    def account_snapshot(self) -> Dict:
        return {
            "balance": round(self.balance, 2),
            "equity": round(self.equity, 2),
            "currency": "USD",
        }

    def open_positions_payload(self) -> List[Dict]:
        out = []
        for t in self.open_trades.values():
            out.append(
                {
                    "ticket": t.ticket,
                    "symbol": t.symbol,
                    "type": t.direction,
                    "volume": round(t.remaining_volume, 2),
                    "open_price": t.entry_price,
                    "current_price": t.current_price,
                    "sl": t.sl,
                    "tp": t.tp,
                    "net_profit": round(t.live_pnl, 2),
                    "magic": cfg.MAGIC_NUMBER,
                    "open_time": t.open_time.isoformat(),
                }
            )
        return out

    def calculate_lot(self, sl_distance: float, high_conf: bool = False, strategy: str = "") -> float:
        if sl_distance <= 0:
            return cfg.MIN_LOT
        strategy_name = str(strategy or "").upper()
        if strategy_name in {"SWEEP_SCALPER", "M15_SCALP_DEEP"}:
            base_risk_pct = float(getattr(cfg, "INTRADAY_RISK_PCT", self._risk_pct))
        elif strategy_name in {"SMC_CONFLUENCE", "M15_SUPPORT_RESISTANCE_REJECTION_V1", "TREND_CHANNEL"}:
            base_risk_pct = float(getattr(cfg, "SWING_RISK_PCT", self._risk_pct))
        else:
            base_risk_pct = self._risk_pct
        risk_pct = base_risk_pct * (cfg.SMC_HIGH_CONF_RISK_MULTIPLIER if high_conf else 1.0)
        risk_amount = self.balance * (risk_pct / 100.0)
        lot = risk_amount / (sl_distance * cfg.PIP_VALUE_PER_LOT)
        lot = max(cfg.MIN_LOT, min(cfg.MAX_LOT, lot))
        return round(lot, 2)

    def open_from_signal(
        self,
        symbol: str,
        strategy: str,
        signal: Dict,
        now_utc: datetime,
        tick: Dict,
        tick_count: int,
    ) -> Optional[int]:
        direction = signal.get("signal")
        if direction not in ("BUY", "SELL"):
            return None
        sl = float(signal.get("sl") or 0.0)
        tp = float(signal.get("tp") or 0.0)
        entry = float(tick["ask"] if direction == "BUY" else tick["bid"])
        sl_distance = abs(entry - sl)
        if sl_distance <= 0:
            return None
        vol = self.calculate_lot(sl_distance, high_conf=bool(signal.get("_high_conf", False)), strategy=strategy)
        self._ticket_counter += 1
        t = SimTrade(
            ticket=self._ticket_counter,
            symbol=symbol,
            strategy=strategy,
            direction=direction,
            volume=vol,
            entry_price=entry,
            sl=sl,
            tp=tp,
            sl_distance=sl_distance,
            open_time=now_utc,
            confidence=float(signal.get("confidence", 0.0)),
            reason=str(signal.get("reason", "")),
            scalp=bool(signal.get("_scalp", False)),
            be_trigger_price=float(signal.get("_be_trigger", 0.0)),
            timeout_seconds=int(signal.get("_timeout", 60)),
            early_fail_points=float(signal.get("_early_fail", cfg.SMC_EARLY_FAIL_POINTS)),
            tier1_min_ticks=int(signal.get("_tier1_min_ticks", cfg.SMC_TIER1_MIN_TICKS)),
            tier1_max_ticks=int(signal.get("_tier1_max_ticks", cfg.SMC_TIER1_MAX_TICKS)),
            entry_tick_velocity=float(signal.get("_entry_tick_velocity", 0.0)),
            high_conf=bool(signal.get("_high_conf", False)),
            entry_tick_count=tick_count,
        )
        self.open_trades[t.ticket] = t
        self._last_open_ts = now_utc.timestamp()
        return t.ticket

    def on_tick(self, now_utc: datetime, tick: Dict, tick_count: int, tick_metrics: Dict):
        to_close: List[Tuple[int, float, str]] = []
        bid = float(tick["bid"])
        ask = float(tick["ask"])
        for ticket, t in list(self.open_trades.items()):
            current_px = bid if t.direction == "BUY" else ask
            t.current_price = current_px
            points_move = (current_px - t.entry_price) if t.direction == "BUY" else (t.entry_price - current_px)
            t.live_pnl = round(points_move * t.remaining_volume * cfg.PIP_VALUE_PER_LOT, 2)
            if t.live_pnl > t.peak_pnl:
                t.peak_pnl = t.live_pnl

            # Hard exits first (tick-realistic)
            if t.direction == "BUY":
                if bid <= t.sl:
                    to_close.append((ticket, t.sl, "SL"))
                    continue
                if bid >= t.tp:
                    to_close.append((ticket, t.tp, "TP"))
                    continue
            else:
                if ask >= t.sl:
                    to_close.append((ticket, t.sl, "SL"))
                    continue
                if ask <= t.tp:
                    to_close.append((ticket, t.tp, "TP"))
                    continue

            t.manage_updates += 1
            age = max(0.0, now_utc.timestamp() - t.open_time.timestamp())
            ticks_since_entry = max(0, tick_count - t.entry_tick_count)
            early_fail = t.early_fail_points or cfg.SMC_EARLY_FAIL_POINTS

            if (
                cfg.TIER1_ENABLED
                and t.tier1_min_ticks <= ticks_since_entry <= t.tier1_max_ticks
                and points_move <= -early_fail
            ):
                to_close.append((ticket, current_px, f"EARLY_FAIL_{ticks_since_entry}"))
                continue

            if t.be_trigger_price > 0 and not t.sl_breakeven and t.live_pnl >= t.be_trigger_price:
                t.sl = round(t.entry_price, 2)
                t.sl_breakeven = True

            if age >= t.timeout_seconds and points_move < 0.30:
                to_close.append((ticket, current_px, "TIMEOUT"))
                continue

            current_velocity = float(tick_metrics.get("velocity", 0) or 0)
            if t.entry_tick_velocity > 0 and current_velocity > 0 and age >= 10:
                if current_velocity <= max(1.0, t.entry_tick_velocity * 0.5) and points_move < 0.30:
                    to_close.append((ticket, current_px, "VELOCITY_DROP"))
                    continue

            if t.scalp:
                if t.peak_pnl > 2.0 and t.live_pnl < 0.5:
                    to_close.append((ticket, current_px, "SCALP_REVERSAL"))
                    continue
            else:
                self._manage_swing_partial_and_trail(t, current_px, age, now_utc)

        for ticket, px, reason in to_close:
            self.close_trade(ticket, now_utc, px, reason)

        floating = sum(t.live_pnl for t in self.open_trades.values())
        self.equity = round(self.balance + floating, 2)
        self.peak_equity = max(self.peak_equity, self.equity)
        dd = self.peak_equity - self.equity
        if dd > self.max_drawdown:
            self.max_drawdown = dd

        self._append_equity(now_utc)

    def _manage_swing_partial_and_trail(self, t: SimTrade, current_px: float, age: float, now_utc: datetime):
        r = max(1.0, t.sl_distance * t.volume * cfg.PIP_VALUE_PER_LOT)
        pnl = t.live_pnl
        peak = t.peak_pnl

        if pnl >= 0.5 * r and not t.sl_breakeven:
            buf = max(t.sl_distance * 0.15, 0.30)
            if t.direction == "BUY":
                t.sl = round(max(t.sl, t.entry_price + buf), 2)
            else:
                t.sl = round(min(t.sl, t.entry_price - buf), 2)
            t.sl_breakeven = True

        if pnl >= 1.0 * r and not t.partial_closed:
            close_vol = round(t.volume * 0.5, 2)
            if close_vol >= cfg.MIN_LOT and (t.remaining_volume - close_vol) >= cfg.MIN_LOT:
                self._realize_partial(t, close_vol, current_px)
            t.partial_closed = True
            rr = 2.0
            if t.direction == "BUY":
                t.tp = round(t.entry_price + t.sl_distance * rr, 2)
                t.sl = round(max(t.sl, t.entry_price + t.sl_distance * 0.5), 2)
            else:
                t.tp = round(t.entry_price - t.sl_distance * rr, 2)
                t.sl = round(min(t.sl, t.entry_price - t.sl_distance * 0.5), 2)
            t.trail_active = True

        if t.trail_active and pnl >= 1.5 * r:
            if t.direction == "BUY":
                t.sl = round(max(t.sl, t.entry_price + t.sl_distance * 1.0), 2)
            else:
                t.sl = round(min(t.sl, t.entry_price - t.sl_distance * 1.0), 2)

        if age >= 600 and pnl < 0.2 * r:
            self.close_trade(t.ticket, now_utc, current_px, "TIME_EXIT")
            return
        if peak >= 1.2 * r and pnl < 0.3 * r:
            self.close_trade(t.ticket, now_utc, current_px, "REVERSAL")

    def _realize_partial(self, t: SimTrade, close_vol: float, current_px: float):
        if close_vol <= 0 or close_vol >= t.remaining_volume:
            return
        points_move = (current_px - t.entry_price) if t.direction == "BUY" else (t.entry_price - current_px)
        pnl = round(points_move * close_vol * cfg.PIP_VALUE_PER_LOT, 2)
        self.balance = round(self.balance + pnl, 2)
        t.realized_partial_pnl += pnl
        t.remaining_volume = round(t.remaining_volume - close_vol, 2)

    def close_trade(self, ticket: int, now_utc: datetime, exit_price: float, reason: str):
        t = self.open_trades.get(ticket)
        if not t:
            return
        points_move = (exit_price - t.entry_price) if t.direction == "BUY" else (t.entry_price - exit_price)
        pnl_rest = round(points_move * t.remaining_volume * cfg.PIP_VALUE_PER_LOT, 2)
        pnl = round(t.realized_partial_pnl + pnl_rest, 2)
        self.balance = round(self.balance + pnl_rest, 2)
        t.close_reason = reason
        t.close_time = now_utc
        t.exit_price = float(exit_price)
        duration = int(max(0, (now_utc - t.open_time).total_seconds()))
        self.closed_trades.append(
            {
                "ticket": t.ticket,
                "symbol": t.symbol,
                "strategy": t.strategy,
                "direction": t.direction,
                "volume": t.volume,
                "entry_price": round(t.entry_price, 5),
                "exit_price": round(t.exit_price, 5),
                "sl": round(t.sl, 5),
                "tp": round(t.tp, 5),
                "open_time": t.open_time.isoformat(),
                "close_time": t.close_time.isoformat(),
                "duration_s": duration,
                "pnl": pnl,
                "won": pnl > 0,
                "close_reason": reason,
                "confidence": round(t.confidence, 4),
            }
        )
        del self.open_trades[ticket]

    def close_all(self, now_utc: datetime, tick: Dict, reason: str = "FORCED_END"):
        for ticket in list(self.open_trades.keys()):
            t = self.open_trades[ticket]
            px = float(tick["bid"] if t.direction == "BUY" else tick["ask"])
            self.close_trade(ticket, now_utc, px, reason)

    def _append_equity(self, now_utc: datetime):
        if self._last_equity_ts and (now_utc - self._last_equity_ts).total_seconds() < 1:
            return
        self._last_equity_ts = now_utc
        self.equity_curve.append(
            {
                "time": now_utc.isoformat(),
                "equity": round(self.equity, 2),
                "balance": round(self.balance, 2),
            }
        )

    def summary(self) -> Dict:
        pnls = [t["pnl"] for t in self.closed_trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        total = len(pnls)
        win_rate = (len(wins) / total) if total else 0.0
        avg_win = (sum(wins) / len(wins)) if wins else 0.0
        avg_loss = abs(sum(losses) / len(losses)) if losses else 0.0
        expectancy = (win_rate * avg_win) - ((1.0 - win_rate) * avg_loss)
        sharpe_like = 0.0
        if total > 1:
            mu = sum(pnls) / total
            variance = sum((p - mu) ** 2 for p in pnls) / (total - 1)
            sigma = math.sqrt(variance)
            if sigma > 0:
                sharpe_like = (mu / sigma) * math.sqrt(total)
        return {
            "initial_balance": round(self.initial_balance, 2),
            "final_balance": round(self.balance, 2),
            "pnl": round(self.balance - self.initial_balance, 2),
            "trades": total,
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(win_rate, 4),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "expectancy": round(expectancy, 2),
            "max_drawdown": round(self.max_drawdown, 2),
            "sharpe_like": round(sharpe_like, 4),
        }


class BacktestRunner:
    def __init__(self):
        self.zone_detector = ZoneDetector()
        self.smc = SMCStrategy()
        self.scalper = SweepScalper()
        self.m15_sr = M15SupportResistanceStrategy()
        self.m15_scalp_deep = M15ScalpDeepStrategy()
        self.m15_zone_scalp = M15ZoneScalpStrategy()
        self.trend_channel = TrendChannelStrategy()
        self.strategy_manager = StrategyManager()
        self.strategy_manager.register(self.smc)
        self.strategy_manager.register(self.scalper)
        self.strategy_manager.register(self.m15_sr)
        self.strategy_manager.register(self.m15_scalp_deep)
        self.strategy_manager.register(self.m15_zone_scalp)
        self.strategy_manager.register(self.trend_channel)
        self._decision_entries: List[Dict] = []

    def run(
        self,
        req: BacktestRequest,
        dataset: BacktestDataset,
        progress_cb: Optional[Callable[[Dict], None]] = None,
        is_cancelled: Optional[Callable[[], bool]] = None,
    ) -> Dict:
        self._decision_entries = []
        clock = BacktestClock(req.start_utc)
        sim = SimExecutionEngine(initial_balance=req.initial_balance)
        candles = dataset.candles
        ticks = dataset.ticks
        if ticks.empty:
            raise RuntimeError("No ticks to replay")

        tf_indices = {tf: 0 for tf in candles}
        tf_sizes = {tf: len(candles[tf]) for tf in candles}
        current_frames = {tf: candles[tf].iloc[:0].copy() for tf in candles}
        last_m1_bar_count = 0
        indicators: Dict = {}
        zones: Dict = {}
        regime: Dict = {}
        bias: Dict = {}
        liquidity: Dict = {}
        tf_context: Dict = {}
        strat_counts = defaultdict(lambda: {"taken": 0, "skipped": 0})
        top_skip_reasons = Counter()
        tick_count = 0
        last_tick = None
        total_ticks = len(ticks)
        if req.fast_mode:
            req.max_ticks_per_batch = max(req.max_ticks_per_batch, 20000)
            req.replay_max_points = min(req.replay_max_points, 1500)
        capture_every = max(1, total_ticks // max(1, req.replay_max_points))
        replay_frames: List[Dict] = []
        replay_thoughts: List[Dict] = []
        tick_proc = TickProcessor(buffer_size=2000)
        self._reset_control_state_for_backtest()

        def _on_decision(entry: Dict):
            self._decision_entries.append(entry)

        try:
            for row in ticks.itertuples(index=False):
                now_utc = row.datetime.to_pydatetime().astimezone(timezone.utc)
                set_backtest_now(now_utc)
                clock.set(now_utc)
                tick_count += 1
                if tick_count % req.max_ticks_per_batch == 0:
                    if is_cancelled and is_cancelled():
                        raise RuntimeError("Backtest cancelled")
                    if progress_cb:
                        progress_cb(
                            {
                                "processed_ticks": tick_count,
                                "total_ticks": total_ticks,
                                "sim_time_utc": now_utc.isoformat(),
                            }
                        )
                last_tick = {
                    "bid": float(row.bid),
                    "ask": float(row.ask),
                    "spread": float(row.spread),
                    "time": now_utc.isoformat(),
                    "volume": float(getattr(row, "volume", 0.0) or 0.0),
                    "ts": now_utc.timestamp(),
                }
                tick_proc.feed(last_tick)
                tick_metrics = tick_proc.snapshot()
                tick_pressure = analyze_tick_pressure(list(getattr(tick_proc, "_ticks", [])))
                thought_message = ""
                thought_kind = "TICK"
                event_tag = ""

                for tf, df in candles.items():
                    idx = tf_indices[tf]
                    size = tf_sizes[tf]
                    while idx < size and df.iloc[idx]["datetime"].to_pydatetime() <= now_utc:
                        idx += 1
                    if idx != tf_indices[tf]:
                        tf_indices[tf] = idx
                        current_frames[tf] = df.iloc[:idx].reset_index(drop=True)

                m1 = current_frames["M1"]
                if len(m1) != last_m1_bar_count:
                    last_m1_bar_count = len(m1)
                    if len(m1) >= 60 and len(current_frames["M5"]) >= 60:
                        try:
                            indicators = compute_indicators(m1)
                        except Exception:
                            indicators = {}
                        try:
                            tf_context = compute_timeframe_context(
                                current_frames["M1"],
                                current_frames["M5"],
                                current_frames["H1"],
                            )
                            indicators.update(tf_context)
                        except Exception:
                            tf_context = {}
                        try:
                            zones = self.zone_detector.detect(current_frames["M5"]) if len(current_frames["M5"]) >= 50 else {}
                        except Exception:
                            zones = {}
                        try:
                            regime = classify_regime(
                                current_frames["H4"],
                                current_frames["H1"],
                                current_frames["M15"],
                                tick_metrics,
                            ) or {}
                        except Exception:
                            regime = {}
                        try:
                            bias = compute_bias(
                                current_frames["H4"],
                                current_frames["H1"],
                                current_frames["M15"],
                            ) or {}
                        except Exception:
                            bias = {}
                        try:
                            liquidity = compute_liquidity(
                                current_frames["M5"],
                                current_frames["M15"],
                                current_frames["H1"],
                                None,
                            ) or {}
                        except Exception:
                            liquidity = {}

                sim.on_tick(now_utc, last_tick, tick_count, tick_metrics)

                # Mirror live gate flow as closely as possible.
                if len(current_frames["M1"]) < 60 or len(current_frames["M5"]) < 60:
                    continue
                # Drop closed-market intervals from decision/skip accounting so
                # results focus on tradable periods only.
                if not is_market_open():
                    continue
                blocked, block_reason = is_session_open_blocked()
                if blocked:
                    reason = f"Session blocked: {block_reason}"
                    top_skip_reasons[reason] += 1
                    thought_kind = "GATE"
                    thought_message = "" if req.fast_mode else reason
                    event_tag = "GATE"
                    self._capture_replay_point(
                        replay_frames, replay_thoughts, tick_count, capture_every, now_utc, last_tick,
                        sim, current_frames, thought_kind, thought_message, event_tag, force=False
                    )
                    continue

                market_data = {
                    "m1_df": current_frames["M1"],
                    "m5_df": current_frames["M5"],
                    "m15_df": current_frames["M15"],
                    "h1_df": current_frames["H1"],
                    "h4_df": current_frames["H4"],
                    "tick": last_tick,
                    "zones": zones,
                    "indicators": indicators,
                    "tf_context": tf_context,
                    "account": sim.account_snapshot(),
                    "positions": sim.open_positions_payload(),
                    "regime": regime,
                    "bias": bias,
                    "liquidity": liquidity,
                    "correlation": {},
                    "calendar": {},
                    "tick_snapshot": tick_metrics,
                    "tick_pressure": tick_pressure,
                    "market_state": compute_market_state(current_frames, now_utc),
                    "now_utc": now_utc,
                }

                sig, strat_name = self._generate_signal(req.strategy, market_data, _on_decision)
                action = sig.get("signal")
                if action not in ("BUY", "SELL"):
                    strat_counts[strat_name]["skipped"] += 1
                    reason = str(sig.get("reason", "No signal"))
                    top_skip_reasons[reason] += 1
                    thought_kind = "NO_TRADE"
                    thought_message = "" if req.fast_mode else f"[{strat_name}] {reason[:240]}"
                    event_tag = "NO_TRADE"
                    self._capture_replay_point(
                        replay_frames, replay_thoughts, tick_count, capture_every, now_utc, last_tick,
                        sim, current_frames, thought_kind, thought_message, event_tag, force=False
                    )
                    continue

                # Live-equivalent XGB guard.
                if not getattr(cfg, "XGB_BYPASS_ENABLED", False):
                    xgb_prob = xgb_model.predict_win_prob(sig, indicators, regime, bias, last_tick)
                    sig["xgb_prob"] = xgb_prob if xgb_model.is_trained else None
                    sig["_xgb_trained"] = bool(xgb_model.is_trained)
                    if bool(getattr(cfg, "XGB_BLOCKING_ENABLED", False)) and xgb_model.is_trained and xgb_prob < float(getattr(cfg, "XGB_BLOCK_THRESHOLD", 0.35) or 0.35):
                        strat_counts[strat_name]["skipped"] += 1
                        reason = f"XGB blocked ({xgb_prob:.0%})"
                        top_skip_reasons[reason] += 1
                        thought_kind = "XGB_BLOCK"
                        thought_message = f"[{strat_name}] {reason}"
                        event_tag = "XGB_BLOCK"
                        self._capture_replay_point(
                            replay_frames, replay_thoughts, tick_count, capture_every, now_utc, last_tick,
                            sim, current_frames, thought_kind, thought_message, event_tag, force=True
                        )
                        continue
                    if bool(getattr(cfg, "XGB_BLEND_CONFIDENCE_ENABLED", False)) and xgb_model.is_trained:
                        sig["confidence"] = round(float(sig.get("confidence", 0)) * 0.7 + xgb_prob * 0.3, 3)

                can_open, why = sim.can_open(now_utc)
                if not can_open:
                    strat_counts[strat_name]["skipped"] += 1
                    reason = f"Execution: {why}"
                    top_skip_reasons[reason] += 1
                    thought_kind = "EXEC_BLOCK"
                    thought_message = "" if req.fast_mode else f"[{strat_name}] {reason}"
                    event_tag = "EXEC_BLOCK"
                    self._capture_replay_point(
                        replay_frames, replay_thoughts, tick_count, capture_every, now_utc, last_tick,
                        sim, current_frames, thought_kind, thought_message, event_tag, force=False
                    )
                    continue
                opened = sim.open_from_signal(
                    symbol=req.symbol,
                    strategy=strat_name,
                    signal=sig,
                    now_utc=now_utc,
                    tick=last_tick,
                    tick_count=tick_count,
                )
                if opened:
                    strat_counts[strat_name]["taken"] += 1
                    thought_kind = "TRADE_OPEN"
                    thought_message = (
                        f"[{strat_name}] {action} ticket={opened} conf={float(sig.get('confidence', 0)):.2f} "
                        f"reason={str(sig.get('reason', ''))[:160]}"
                    )
                    event_tag = "TRADE_OPEN"
                    self._capture_replay_point(
                        replay_frames, replay_thoughts, tick_count, capture_every, now_utc, last_tick,
                        sim, current_frames, thought_kind, thought_message, event_tag, force=True
                    )
                else:
                    strat_counts[strat_name]["skipped"] += 1
                    reason = "Execution: invalid signal params"
                    top_skip_reasons[reason] += 1
                    thought_kind = "EXEC_BLOCK"
                    thought_message = f"[{strat_name}] {reason}"
                    event_tag = "EXEC_BLOCK"
                    self._capture_replay_point(
                        replay_frames, replay_thoughts, tick_count, capture_every, now_utc, last_tick,
                        sim, current_frames, thought_kind, thought_message, event_tag, force=True
                    )

        finally:
            set_backtest_now(None)

        if last_tick:
            sim.close_all(clock.current_utc, last_tick, reason="END_OF_RANGE")

        summary = sim.summary()
        decision_stats = self._decision_stats(top_skip_reasons)
        strategy_breakdown = dict(strat_counts)
        cfg_snapshot = self._runtime_config_snapshot(req.use_runtime_config)
        result = {
            "summary": summary,
            "equity_curve": sim.equity_curve,
            "trades": sim.closed_trades,
            "decision_stats": decision_stats,
            "strategy_breakdown": strategy_breakdown,
            "replay": {
                "capture_every": capture_every,
                "frames": replay_frames,
                "thoughts": replay_thoughts,
            },
            "meta": {
                "symbol": req.symbol,
                "strategy": req.strategy,
                "requested_start_utc": req.start_utc.isoformat(),
                "requested_end_utc": req.end_utc.isoformat(),
                "initial_balance": req.initial_balance,
                "use_runtime_config": req.use_runtime_config,
                "max_ticks_per_batch": req.max_ticks_per_batch,
                "replay_max_points": req.replay_max_points,
                "split_mode": req.split_mode,
                "parallel_workers": req.parallel_workers,
                "cache_only": req.cache_only,
                "effective_start_utc": dataset.effective_start_utc.isoformat(),
                "effective_end_utc": dataset.effective_end_utc.isoformat(),
                "config_profile": req.config_profile,
                "coverage_warnings": dataset.coverage_warnings,
                "config_snapshot_hash": cfg_snapshot["hash"],
                "config_snapshot": cfg_snapshot["values"],
                "ticks_replayed": total_ticks,
                "fast_mode": req.fast_mode,
            },
        }
        return result

    @staticmethod
    def _capture_replay_point(
        replay_frames: List[Dict],
        replay_thoughts: List[Dict],
        tick_count: int,
        capture_every: int,
        now_utc: datetime,
        tick: Dict,
        sim: SimExecutionEngine,
        frames: Dict[str, pd.DataFrame],
        thought_kind: str,
        thought_message: str,
        event_tag: str,
        force: bool = False,
    ):
        should_capture = force or (tick_count % capture_every == 0)
        if not should_capture:
            return
        m1 = frames.get("M1", pd.DataFrame())
        m5 = frames.get("M5", pd.DataFrame())
        m1_last = BacktestRunner._latest_candle_payload(m1)
        m5_last = BacktestRunner._latest_candle_payload(m5)
        replay_frames.append(
            {
                "tick_index": tick_count,
                "time": now_utc.isoformat(),
                "bid": round(float(tick.get("bid", 0.0)), 5),
                "ask": round(float(tick.get("ask", 0.0)), 5),
                "spread": round(float(tick.get("spread", 0.0)), 5),
                "equity": round(float(sim.equity), 2),
                "balance": round(float(sim.balance), 2),
                "open_positions": len(sim.open_trades),
                "event": event_tag,
                "m1": m1_last,
                "m5": m5_last,
            }
        )
        if thought_message:
            replay_thoughts.append(
                {
                    "tick_index": tick_count,
                    "time": now_utc.isoformat(),
                    "kind": thought_kind,
                    "message": thought_message[:400],
                }
            )

    @staticmethod
    def _latest_candle_payload(df: pd.DataFrame) -> Dict:
        if df is None or df.empty:
            return {}
        row = df.iloc[-1]
        dt = row["datetime"]
        if hasattr(dt, "to_pydatetime"):
            dt = dt.to_pydatetime()
        if isinstance(dt, datetime):
            iso = dt.astimezone(timezone.utc).isoformat() if dt.tzinfo else dt.replace(tzinfo=timezone.utc).isoformat()
        else:
            iso = str(dt)
        return {
            "time": iso,
            "open": round(float(row["open"]), 5),
            "high": round(float(row["high"]), 5),
            "low": round(float(row["low"]), 5),
            "close": round(float(row["close"]), 5),
        }

    @staticmethod
    def _reset_control_state_for_backtest():
        """Ensure global singleton controls start clean for each replay run."""
        try:
            from .session_risk_control import risk_controller
            risk_controller._current_session = ""
            risk_controller._current_date = ""
            risk_controller._session_trades = 0
            risk_controller._consecutive_losses = 0
            risk_controller._daily_pnl = 0.0
            risk_controller._risk_blocks = {
                "consecutive_losses": False,
                "daily_loss": False,
                "session_trades": False,
            }
        except Exception:
            pass
        try:
            from .trade_pacing import pacing_controller
            pacing_controller._last_trade_time = 0
            pacing_controller._trade_history = []
        except Exception:
            pass
        try:
            from .anti_starvation import anti_starvation
            anti_starvation._session_trades = {}
            anti_starvation._current_session = ""
            anti_starvation._current_date = ""
            anti_starvation._relaxation_active = False
            anti_starvation._relaxation_type = None
        except Exception:
            pass

    def _generate_signal(
        self,
        selected_strategy: str,
        market_data: Dict,
        decision_sink: Callable[[Dict], None],
    ) -> Tuple[Dict, str]:
        self.strategy_manager.set_active(selected_strategy if selected_strategy != "AUTO" else "AUTO")
        with backtest_context(decision_sink=decision_sink):
            chosen = self.strategy_manager.generate_signal(market_data)
        chosen = chosen or {"signal": {"signal": "NO_TRADE", "reason": "Empty signal"}, "strategy": selected_strategy}
        sig = chosen.get("signal") or {"signal": "NO_TRADE", "reason": "Empty signal"}
        strat_name = chosen.get("strategy") or selected_strategy
        return sig, (strat_name or selected_strategy)

    def _decision_stats(self, top_skip_reasons: Counter) -> Dict:
        taken = sum(1 for e in self._decision_entries if e.get("decision") == "TRADE_TAKEN")
        skipped = sum(1 for e in self._decision_entries if e.get("decision") == "TRADE_SKIPPED")
        top = [{"reason": k, "count": v} for k, v in top_skip_reasons.most_common(20)]
        return {
            "total_decisions": taken + skipped,
            "trades_taken": taken,
            "trades_skipped": skipped,
            "top_skip_reasons": top,
        }

    @staticmethod
    def _runtime_config_snapshot(use_runtime_config: bool) -> Dict:
        keys = [
            "MAX_RISK_PCT",
            "INTRADAY_RISK_PCT",
            "SWING_RISK_PCT",
            "MAX_POSITIONS",
            "MIN_TRADE_COOLDOWN",
            "SESSION_MAX_TRADES",
            "MAX_TRADES_PER_DAY",
            "SMC_THRESHOLD_WITH_TREND",
            "SMC_THRESHOLD_COUNTER",
            "SMC_MIN_RR",
            "SMC_SPREAD_STD_MAX",
            "SMC_SPREAD_PERCENTILE_MAX",
            "SCALPER_QUALITY_THRESHOLD",
            "SCALPER_SPREAD_STD_MAX",
            "SCALPER_SPREAD_PERCENTILE_MAX",
            "SCALPER_TICK_DIR_THRESHOLD",
            "COMPRESSION_ATR_MULTIPLIER",
            "CALENDAR_BLOCKING_ENABLED",
            "CALENDAR_BLOCK_BEFORE_MINUTES",
            "CALENDAR_BLOCK_AFTER_MINUTES",
            "TRADE_WINDOW_LONDON_START",
            "TRADE_WINDOW_LONDON_END",
            "TRADE_WINDOW_OVERLAP_START",
            "TRADE_WINDOW_OVERLAP_END",
        ]
        values = {}
        defaults = cfg.get_runtime_defaults()
        for k in keys:
            if use_runtime_config:
                values[k] = getattr(cfg, k)
            else:
                values[k] = defaults.get(k, getattr(cfg, k))
        payload = json.dumps(values, sort_keys=True).encode("utf-8")
        return {"values": values, "hash": hashlib.sha256(payload).hexdigest()[:16]}


def persist_backtest_artifacts(run_id: str, result: Dict, out_dir: str) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    run_dir = os.path.join(out_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    result_path = os.path.join(run_dir, "result.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    trades_csv = os.path.join(run_dir, "trades.csv")
    pd.DataFrame(result.get("trades", [])).to_csv(trades_csv, index=False)
    equity_csv = os.path.join(run_dir, "equity_curve.csv")
    pd.DataFrame(result.get("equity_curve", [])).to_csv(equity_csv, index=False)
    return {
        "run_dir": run_dir,
        "result_json": result_path,
        "trades_csv": trades_csv,
        "equity_csv": equity_csv,
    }
