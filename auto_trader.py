"""
XAUUSD Auto Trader — Background Service with selectable strategies.

Strategies:
  AUTO                                   — choose the best eligible strategy
  SMC_CONFLUENCE                         — swing / SMC confluence
  M15_SUPPORT_RESISTANCE_REJECTION_V1    — M15 structure rejection
  SWEEP_SCALPER                          — intraday sweep scalper
  TREND_CHANNEL                          — trend-following channel continuation
"""
import time
import sys
import threading
import json
import os
import mimetypes
from collections import deque
from pathlib import Path
from typing import Dict
from datetime import datetime, timezone, timedelta
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, unquote

_IST = timezone(timedelta(hours=5, minutes=30))

import config as cfg
from engine.analytics_api import _build_analytics_page, _LATEST_DIR, _OUTPUT_ROOT
from engine.mt5_bridge import MT5Bridge
from engine.indicators import compute_indicators, compute_timeframe_context
from engine.zones import ZoneDetector
from engine.session_filter import get_session, is_session_open_blocked, is_market_open
from engine.risk_manager import RiskManager
from engine.trade_manager import TradeManager
from engine.tick_processor import TickProcessor, analyze_tick_pressure
from engine.regime import classify_regime
from engine.mtf_bias import compute_bias
from engine.liquidity import compute_liquidity
from engine.performance import PerformanceTracker
from engine.strategy_manager import StrategyManager
from engine.smc_strategy import SMCStrategy
from engine.sweep_scalper import SweepScalper
from engine.m15_sr_strategy import M15SupportResistanceStrategy
from engine.strategies.trend_channel_strategy import TrendChannelStrategy
from engine.swing_engine_strategy import SwingEngineStrategy
from engine.intraday_engine_strategy import IntradayEngineStrategy
from engine.strategies.htf_long_strategy import HTFLongStrategy
from engine.strategies.htf_short_strategy import HTFShortStrategy
from engine.calendar import calendar as eco_calendar
from engine.correlation import correlation as corr_engine
from engine.order_database import OrderDatabase
from engine.visual_analytics import generate_visual_suite
from engine.xgb_model import (
    xgb_model,
    xgb_bypass_enabled,
    xgb_training_enabled,
    xgb_blocking_enabled,
    xgb_blending_enabled,
)
from engine.ai_trade_advisor import ai_trade_advisor_service
from engine.anti_starvation import record_trade_taken
from engine.trade_pacing import record_trade_taken as record_pacing_trade
from engine.session_risk_control import record_trade_taken as record_session_trade
from engine.decision_logger import get_live_blockers
from engine.master_control import log_trade_decision_comprehensive
from engine.master_trade_gate import set_gate_log_fn as _set_gate_log_fn
from engine import adaptive_params
from engine.market_state import compute_market_state
from engine.strategy_configs import apply_all_to_cfg as _apply_strategy_configs
from engine.sl_streak_guard import sl_streak_guard
from engine.deployment_metadata import capture_code_snapshot

_TF_REFRESH = {"M1": 1, "M5": 2, "M15": 10, "M30": 20, "H1": 60, "H4": 120, "D1": 720}
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _safe_dict(value):
    return value if isinstance(value, dict) else {}


def _safe_float(value, default=0.0):
    """Safely convert value to float, return default if conversion fails."""
    try:
        return float(value) if value is not None else default
    except (ValueError, TypeError):
        return default


class AutoTrader:
    def __init__(self):
        self._started_at_utc = datetime.now(timezone.utc)
        self._deployment_snapshot = capture_code_snapshot(Path(_BASE_DIR))
        self.bridge = MT5Bridge()
        self.zone_detector = ZoneDetector()
        self.risk = RiskManager()
        self.trades: TradeManager = None
        self.tick_proc = TickProcessor()
        self.perf = PerformanceTracker(os.path.join(_BASE_DIR, "trade_history.json"))

        # Strategy manager
        self.strat_mgr = StrategyManager()
        self._register_strategies()
        self._apply_startup_strategy()
        _apply_strategy_configs()  # apply per-strategy config to cfg on startup

        self._candles: Dict = {}
        self._candle_counts: Dict = {}
        self._indicators: Dict = {}
        self._tf_context: Dict = {}
        self._zones: Dict = {}
        self._regime: Dict = {}
        self._bias: Dict = {}
        self._liquidity: Dict = {}
        self._calendar: Dict = {}
        self._correlation: Dict = {}
        self._market_state: Dict = {}

        self._running = False
        self.enabled = False
        self._mt5_connected = False
        self._cycle = 0
        self._last_log_date_ist: str = ""
        self._stats = {"cycles": 0, "signals": 0, "trades": 0, "errors": 0}
        self._last_log_time = 0
        self._log: deque = deque(maxlen=500)
        self._last_tick: Dict = {}
        self._tick_seq = 0
        self._last_tick_signature = None
        self._tick_pump_started = False
        self._last_account: Dict = {}
        self._tick_pressure: Dict = {"ready": False}
        self._last_tick_pressure_poll = 0.0

        # MT5 history cache (polled every 5s in engine loop)
        self._mt5_closed_history: list = []
        self._mt5_today_pnl: Dict = {}
        self._last_history_poll = 0.0

        # Cached positions for API/WS layer (updated after manage_all so closed trades vanish immediately)
        self._cached_positions: list = []
        self._cached_floating_pnl: Dict = {"total": 0, "count": 0}
        self._positions_version: int = 0
        self._last_strategy_skip_log: Dict[str, float] = {}
        self._last_signal: Dict = {}
        self._last_strategy_results: Dict[str, Dict] = {}

    def _register_strategies(self):
        self.strat_mgr._strategies.clear()
        self.strat_mgr._selected = []
        self.strat_mgr._active = ""
        self.strat_mgr._auto = True
        _strategy_classes = [
            ("SMC_CONFLUENCE", SMCStrategy),
            ("M15_SR", M15SupportResistanceStrategy),
            ("SWEEP_SCALPER", SweepScalper),
            ("TREND_CHANNEL", TrendChannelStrategy),
            ("SWING_ENGINE", SwingEngineStrategy),
            ("INTRADAY_ENGINE", IntradayEngineStrategy),
            ("HTF_LONG", HTFLongStrategy),
            ("HTF_SHORT", HTFShortStrategy),
        ]
        for label, cls in _strategy_classes:
            try:
                self.strat_mgr.register(cls())
            except Exception as e:
                import traceback
                traceback.print_exc()
                try:
                    self.log("ERR", f"Failed to register strategy {label}: {e}")
                except Exception:
                    print(f"[STRATEGY_ERR] Failed to register {label}: {e}")

    def _apply_startup_strategy(self):
        default_strategy = getattr(cfg, "DEFAULT_STRATEGY", "AUTO")
        if isinstance(default_strategy, (list, tuple, set)):
            if not self.strat_mgr.set_selection(default_strategy):
                self.strat_mgr.set_active("AUTO")
            return
        text = str(default_strategy or "AUTO").strip()
        if "," in text:
            names = [part.strip().upper() for part in text.split(",") if part.strip()]
            if not self.strat_mgr.set_selection(names):
                self.strat_mgr.set_active("AUTO")
            return
        if not self.strat_mgr.set_active(text.upper() or "AUTO"):
            self.strat_mgr.set_active("AUTO")

    def _backup_log_if_new_day(self):
        """At midnight IST, move yesterday's trader.log to backup_logs/."""
        today = datetime.now(_IST).strftime("%Y-%m-%d")
        if self._last_log_date_ist == today:
            return
        if self._last_log_date_ist:  # not first run
            log_path = os.path.join(_BASE_DIR, "trader.log")
            if os.path.isfile(log_path):
                backup_dir = os.path.join(_BASE_DIR, "backup_logs")
                os.makedirs(backup_dir, exist_ok=True)
                dest = os.path.join(backup_dir, f"trader_{self._last_log_date_ist}.log")
                try:
                    import shutil
                    shutil.move(log_path, dest)
                except Exception:
                    pass
            # Reset performance tracker for the new IST day
            try:
                self.perf.reset_for_new_day()
            except Exception:
                pass
        self._last_log_date_ist = today

    def mark_restart_time(self, started_at_utc=None):
        """Update the engine restart timestamp used by status surfaces."""
        if started_at_utc is None:
            started_at_utc = datetime.now(timezone.utc)
        if started_at_utc.tzinfo is None:
            started_at_utc = started_at_utc.replace(tzinfo=timezone.utc)
        self._started_at_utc = started_at_utc.astimezone(timezone.utc)

    def _restart_status(self) -> Dict:
        started_at_utc = self._started_at_utc
        started_at_ist = started_at_utc.astimezone(_IST)
        uptime_seconds = max(0, int((datetime.now(timezone.utc) - started_at_utc).total_seconds()))
        return {
            "last_restart_utc": started_at_utc.isoformat(),
            "last_restart_ist": started_at_ist.strftime("%Y-%m-%d %H:%M:%S IST"),
            "uptime_seconds": uptime_seconds,
        }

    def log(self, tag: str, msg: str):
        now = datetime.now(timezone.utc)
        utc_str = now.strftime("%H:%M:%S.%f")[:-3]
        ist_now = now.astimezone(_IST)
        ist_str = ist_now.strftime("%H:%M:%S.%f")[:-3]
        ist_date = ist_now.strftime("%Y-%m-%d")
        raw_price = (self._last_tick or {}).get("bid")
        price = None
        if isinstance(raw_price, (int, float)):
            price = round(float(raw_price), 2)
        entry = {"time": ist_str, "time_ist": ist_str, "ist_date": ist_date, "tag": tag, "msg": msg, "price": price}
        self._log.append(entry)
        try:
            with open(os.path.join(_BASE_DIR, "trader.log"), "a") as f:
                price_part = f"[{price:.2f}]" if price is not None else ""
                f.write(f"[{utc_str} UTC | {ist_str} IST | {ist_date}][{tag}]{price_part} {msg}\n")
        except Exception:
            pass

    def run_service(self):
        self.log("INIT", f"Starting {cfg.SYMBOL} Auto Trader service...")
        if self.bridge.connect():
            self._mt5_connected = True
            acct = self.bridge.get_account()
            if acct:
                self.risk.set_start_balance(acct.get("balance", 0))
            self.risk.set_risk_multiplier(self.perf.risk_multiplier())
            # Seed daily P&L from MT5 deal history (single source of truth)
            # Initial MT5 history poll
            self._poll_mt5_history()
            today_pnl = self._mt5_today_pnl
            if today_pnl.get("pnl", 0) != 0 or today_pnl.get("trades", 0) > 0:
                self.log("INIT", f"Today MT5 P&L: ${today_pnl['pnl']:+.2f} ({today_pnl['trades']} trades, {today_pnl['wins']}W/{today_pnl['losses']}L) [{today_pnl.get('source','')}]")
            self.trades = TradeManager(self.bridge)
            corr_engine.init_symbols()
            # XGBoost: train from performance tracker (has features), not CSV
            perf_trades = self.perf.trades
            featured = [t for t in perf_trades if t.get("features") and
                        any(isinstance(v, (int, float)) and v != 0
                            for v in t["features"].values())]
            if xgb_bypass_enabled():
                self.log("INIT", "XGBoost bypass enabled")
            elif not xgb_training_enabled():
                self.log("INIT", "XGBoost training disabled")
            elif not xgb_model.is_trained and len(featured) >= 15:
                # Block briefly on first train so dashboard shows trained state immediately
                xgb_model.train(featured)
                self.log("INIT", f"XGBoost trained on {len(featured)} trades (with features)")
            elif xgb_model.is_trained and xgb_model.should_retrain(len(featured)):
                threading.Thread(target=xgb_model.train, args=(featured,), daemon=True).start()
                self.log("INIT", f"XGBoost retraining on {len(featured)} trades")
            elif xgb_model.is_trained:
                self.log("INIT", f"XGBoost loaded ({xgb_model._trades_at_last_train} trades)")
            else:
                self.log("INIT", f"XGBoost: {len(featured)} featured trades < 15 min, skipping")
            self.log("INIT", "Loading candle data...")
            for tf in _TF_REFRESH:
                df = self.bridge.fetch_candles(tf)
                if not df.empty:
                    self._candles[tf] = df
                    self._candle_counts[tf] = len(df)
            self._recompute_all()
            loaded = ", ".join(f"{tf}:{len(df)}" for tf, df in self._candles.items())
            self.log("INIT", f"MT5 connected. Data: {loaded}")
            self.log("INIT", f"Strategies: {', '.join(self.strat_mgr.available)} | Active: {self.strat_mgr.active_name}")
        else:
            self.log("WARN", "MT5 not connected. Will retry.")
            self.trades = TradeManager(self.bridge)

        self._running = True
        self.mark_restart_time()
        _set_gate_log_fn(self.log)
        self._backup_log_if_new_day()
        threading.Thread(target=self._engine_loop, daemon=True, name="Engine").start()
        self.log("INIT", f"Dashboard: http://{cfg.API_HOST}:{cfg.API_PORT} | Strategy: {self.strat_mgr.active_name}")
        self._run_http_server()

    def _engine_loop(self):
        self.log("ENGINE", "Trading engine started")
        self._start_tick_pump()
        while self._running:
            t0 = time.time()
            try:
                # Proactive connection health check every cycle
                if self._mt5_connected and not self.bridge.ping():
                    self._mt5_connected = False
                    self.log("ENGINE", "MT5 connection lost (ping failed)")

                if not self._mt5_connected:
                    if self.bridge.connect():
                        self._mt5_connected = True
                        corr_engine.init_symbols()
                        acct = self.bridge.get_account()
                        if acct:
                            self.risk.set_start_balance(acct.get("balance", 0))
                        self.log("ENGINE", "MT5 reconnected")
                    else:
                        time.sleep(5)
                        continue
                self._cycle_once()
            except Exception as e:
                self.log("ERR", str(e))
                self._stats["errors"] += 1
                if "initialize" in str(e).lower() or "terminal" in str(e).lower() or not self.bridge.ping():
                    self._mt5_connected = False
                    self.log("ENGINE", "MT5 disconnected, will retry...")
            elapsed = time.time() - t0
            if elapsed < 0.01:
                time.sleep(0.01)

    def _start_tick_pump(self):
        if self._tick_pump_started:
            return
        self._tick_pump_started = True
        threading.Thread(target=self._tick_pump_loop, daemon=True, name="MarketTickPump").start()

    def _tick_pump_loop(self):
        _pos_refresh_interval = 1.0  # full MT5 positions_get every 1s (structure/SL/TP sync)
        _last_pos_refresh = 0.0
        _last_pos_count = -1
        while self._running:
            try:
                if self._mt5_connected:
                    tick = self.bridge.get_tick()
                    if tick:
                        signature = (tick.get("bid"), tick.get("ask"), tick.get("time"), tick.get("volume"))
                        self._last_tick = tick
                        if signature != self._last_tick_signature:
                            self._last_tick_signature = signature
                            self._tick_seq += 1
                            self.tick_proc.feed(tick)
                            # Recompute floating PnL from tick price on every new tick
                            # — same speed as the price display, no extra MT5 call needed
                            self._recompute_floating_pnl_from_tick(tick)
                    # Full positions_get at 1s cadence for structure sync (SL/TP/volume)
                    now = time.monotonic()
                    if now - _last_pos_refresh >= _pos_refresh_interval:
                        _last_pos_refresh = now
                        all_pos = self.bridge.get_positions()
                        magic_pos = [p for p in (all_pos or []) if p.get("magic") == cfg.MAGIC_NUMBER]
                        self._cached_positions = magic_pos
                        self._cached_floating_pnl = {
                            "total": round(sum(p.get("net_profit", 0) for p in magic_pos), 2),
                            "count": len(magic_pos),
                        }
                        # Bump version if a position disappeared (trade closed)
                        if _last_pos_count != -1 and len(magic_pos) < _last_pos_count:
                            self._positions_version += 1
                        _last_pos_count = len(magic_pos)
            except Exception:
                pass
            time.sleep(max(0.005, float(getattr(cfg, "MARKET_TICK_POLL_INTERVAL", 0.02))))

    def _recompute_floating_pnl_from_tick(self, tick: Dict):
        """Recompute floating PnL from the live tick price without calling MT5.
        Uses cached open trade records (entry price, volume, direction) so PnL
        updates at the same frequency as the bid/ask price display."""
        if not self.trades or not self.trades.open_trades:
            return
        bid = float(tick.get("bid") or 0.0)
        ask = float(tick.get("ask") or 0.0)
        if not bid and not ask:
            return
        total = 0.0
        updated_positions = []
        for pos in self._cached_positions:
            ticket = pos.get("ticket")
            trade = self.trades.open_trades.get(ticket) if ticket else None
            if trade is None:
                updated_positions.append(pos)
                total += pos.get("net_profit", 0.0)
                continue
            close_price = bid if trade.direction == "BUY" else ask
            if trade.direction == "BUY":
                raw_pnl = (close_price - trade.entry) * trade.volume * cfg.PIP_VALUE_PER_LOT
            else:
                raw_pnl = (trade.entry - close_price) * trade.volume * cfg.PIP_VALUE_PER_LOT
            pnl = round(raw_pnl, 2)
            updated = dict(pos)
            updated["current_price"] = close_price
            updated["net_profit"] = pnl
            updated_positions.append(updated)
            total += pnl
        self._cached_positions = updated_positions
        self._cached_floating_pnl = {
            "total": round(total, 2),
            "count": len(updated_positions),
        }

    def _cycle_once(self):
        self._cycle += 1
        self._stats["cycles"] += 1

        tick = self._last_tick or self.bridge.get_tick()
        if not tick:
            return
        self._last_tick = tick
        self._refresh_tick_pressure()

        account = self.bridge.get_account()
        if not account:
            return
        self._last_account = account
        positions = self.bridge.get_my_positions()
        all_positions = self.bridge.get_positions()  # ALL for P&L tracking

        # Candle refresh
        zones_dirty = regime_dirty = False
        for tf, interval in _TF_REFRESH.items():
            if self._cycle % interval != 0:
                continue
            df = self.bridge.fetch_candles(tf)
            if df.empty:
                continue
            new_count = len(df)
            old_count = self._candle_counts.get(tf, 0)
            self._candles[tf] = df
            if new_count != old_count:
                self._candle_counts[tf] = new_count
                if tf == "M1":
                    try:
                        self._indicators = compute_indicators(df)
                    except Exception as e:
                        self.log("ERR", f"Indicators computation failed for {tf}: {e}")
                        self._indicators = {}
                    self._refresh_timeframe_context()
                if tf == "M5":
                    zones_dirty = True
                    self._refresh_timeframe_context()
                if tf in ("M15", "M30", "H1", "H4"):
                    regime_dirty = True
                    if tf == "H1":
                        self._refresh_timeframe_context()

        if zones_dirty:
            m5 = self._candles.get("M5")
            if m5 is not None and len(m5) >= 50:
                try:
                    self._zones = self.zone_detector.detect(m5)
                except Exception as e:
                    self.log("ERR", f"Zone detection failed: {e}")
                    self._zones = {}
        if regime_dirty or self._cycle % 30 == 0:
            try:
                self._recompute_regime_bias()
            except Exception as e:
                self.log("ERR", f"Regime/bias computation failed: {e}")
                self._regime = {}
                self._bias = {}
                self._liquidity = {}
            # Adaptive parameter adjustment — runs every ~30 cycles
            try:
                changes = adaptive_params.apply(self._regime, self.risk, self.perf)
                if changes and changes.get("regime"):
                    self.log("ADAPT", f"[{changes['regime']}] "
                             f"WTR:{changes['SMC_THRESHOLD_WITH_TREND']:.0%} "
                             f"CTR:{changes['SMC_THRESHOLD_COUNTER']:.0%} "
                             f"RR:{changes['SMC_MIN_RR']:.1f} "
                             f"Trades:{changes['SESSION_MAX_TRADES']} "
                             f"Cool:{changes['MIN_TRADE_COOLDOWN']:.0f}s "
                             f"Spread:{changes['SMC_SPREAD_MEAN_MAX']:.2f} "
                             f"streak:{changes.get('streak',0)} "
                             f"dd:{changes.get('daily_dd_pct',0):.2f}%")
            except Exception as e:
                self.log("ERR", f"Adaptive params failed: {e}")
        if self._cycle % 60 == 0:
            eco_calendar.poll()
            self._calendar = eco_calendar.check()
        if self._cycle % 300 == 0:
            self._correlation = corr_engine.compute()

        # Feed floating P&L to risk manager for combined target check
        floating = sum(p.get("net_profit", 0) for p in (all_positions or []) if p.get("magic") == cfg.MAGIC_NUMBER)
        self.risk.update_floating_pnl(floating)

        # Close all open trades if daily target/loss limit breached (realized+floating)
        if self.trades and self.risk.should_close_all() and self.trades.open_count > 0:
            reason = self.risk.should_close_all_reason()
            self.log("RISK", f"{reason}. Closing all.")
            self.trades.close_all()

        # Close all if equity drawdown exceeds limit
        if self.trades and account and self.risk.should_close_drawdown(account) and self.trades.open_count > 0:
            bal = account.get('balance', 0)
            eq = account.get('equity', bal)
            dd = (bal - eq) / bal * 100 if bal > 0 else 0
            self.log("RISK", f"Drawdown {dd:.1f}% >= {cfg.MAX_DRAWDOWN_PCT}%. Closing all.")
            self.trades.close_all()

        # Manage open trades (ALWAYS) — pass ALL positions for P&L matching
        closed = self.trades.manage_all(
            all_positions or [],
            tick_metrics=self.tick_proc.snapshot(),
            market_context={
                "m1_df": self._candles.get("M1"),
                "m5_df": self._candles.get("M5"),
                "m15_df": self._candles.get("M15"),
                "h1_df": self._candles.get("H1"),
                "h4_df": self._candles.get("H4"),
                "d1_df": self._candles.get("D1"),
                "tick_pressure": self._tick_pressure,
            },
        )

        # Bump version on close so WS pushes immediately (tick pump handles live P&L refresh)
        if closed:
            self._positions_version += 1

        # Re-entry check: if a swing trade was closed on recovery exit, immediately
        # re-evaluate the swing strategy and open a new trade if conditions allow.
        reentry_strategy = self.trades._pending_reentry_check
        if reentry_strategy:
            self.trades._pending_reentry_check = None
            self._try_reentry(reentry_strategy, tick, account, positions)

        for ticket, pnl, won in closed:
            # Record to performance tracker with MT5-sourced P&L + features from DB
            record = {"ticket": ticket, "pnl": pnl, "won": won,
                      "time": datetime.now(timezone.utc).isoformat()}
            db_order = self.trades.order_db.get_order(ticket)
            if db_order and db_order.get("features"):
                record["features"] = db_order["features"]
            # Use MT5 close_time if available so the time field reflects actual trade time
            mt5_trade = next((t for t in self._mt5_closed_history if t.get("ticket") == ticket), None)
            if mt5_trade and mt5_trade.get("close_time"):
                record["time"] = mt5_trade["close_time"]
            self.perf.record(record)
            self.risk.record_trade_result(pnl, won)
            self.risk.set_risk_multiplier(self.perf.risk_multiplier())
            self.log("RESULT", f"#{ticket} {'WIN' if won else 'LOSS'} ${pnl:+.2f} (MT5 sourced)")
            # Feed SL streak guard — determine if this was an SL hit
            if db_order:
                strategy_name = db_order.get("strategy", "")
                close_cat = db_order.get("close_reason_category", "")
                if strategy_name:
                    if won:
                        sl_streak_guard.record_win(strategy_name)
                    elif close_cat in ("sl", "mt5_external", "breakeven_stop") or not won:
                        sl_streak_guard.record_sl(strategy_name)
                        guard_status = sl_streak_guard.status(strategy_name)
                        if guard_status["consecutive_sl_hits"] >= 2:
                            self.log(
                                "GUARD",
                                f"[{strategy_name}] {guard_status['consecutive_sl_hits']} consecutive SL hits — "
                                f"{'cooldown ' + str(guard_status['cooldown_remaining_seconds']) + 's' if guard_status['in_cooldown'] else 'strict gate active'}",
                            )
            # XGBoost: retrain using perf trades with real features (not raw MT5)
            featured = [t for t in self.perf.trades if t.get("features") and
                        any(isinstance(v, (int, float)) and v != 0
                            for v in t["features"].values())]
            if len(featured) >= 15 and xgb_model.should_retrain(len(featured)):
                threading.Thread(target=xgb_model.train, args=(featured,), daemon=True).start()

        # Poll MT5 history every 5s — single source of truth
        now_ts = time.time()
        if now_ts - self._last_history_poll >= 5.0:
            self._last_history_poll = now_ts
            self._poll_mt5_history()

        # Daily log backup check
        self._backup_log_if_new_day()

        # Periodic status (moved before gates so it always fires)
        now = now_ts
        if now - self._last_log_time > 30:
            self._last_log_time = now
            self._log_status(tick)

        if not self.enabled:
            return
        if not is_market_open():
            if now - getattr(self, '_last_gate_log', 0) > 600:
                self._last_gate_log = now
                h = datetime.now(timezone.utc).hour
                msg = "Rollover hour (21:00-22:00 UTC)" if h == 21 else "Market closed"
                self.log("GATE", msg)
            return

        strat_data = {
            "m1_df": self._candles.get("M1"), "m5_df": self._candles.get("M5"),
            "m15_df": self._candles.get("M15"), "m30_df": self._candles.get("M30"),
            "h1_df": self._candles.get("H1"),
            "h4_df": self._candles.get("H4"),
            "d1_df": self._candles.get("D1"),
            "symbol": self.bridge.current_symbol if self.bridge else cfg.SYMBOL,
            "tick": tick, "zones": self._zones, "indicators": self._indicators,
            "tf_context": self._tf_context,
            "account": account, "positions": positions or [],
            "correlation": self._correlation, "calendar": self._calendar,
            "bias": self._bias, "regime": self._regime,
            "liquidity": self._liquidity,
            "tick_snapshot": self.tick_proc.snapshot() or {},
            "tick_pressure": self._tick_pressure,
            "market_state": self._market_state,
            "_risk_manager": self.risk,
            "now_utc": datetime.now(timezone.utc),
            "strategy_trade_counts": {
                name: sum(1 for t in (self.trades.open_trades.values() if self.trades else [])
                          if getattr(t, "strategy", "") == name)
                for name in ["SWING_ENGINE", "INTRADAY_ENGINE"]
            },
        }
        if self.strat_mgr.is_auto:
            item = self.strat_mgr.generate_signal(strat_data)
            signals_to_execute = [item]
            self._last_strategy_results = dict(item.get("all_results") or {})
        else:
            # Non-AUTO: all enabled strategies evaluated independently
            signals_to_execute = self.strat_mgr.generate_signals(strat_data)
            # Build results summary for dashboard
            self._last_strategy_results = {r["strategy"]: {"signal": r["signal"].get("signal", "NO_TRADE"), "reason": str(r["signal"].get("reason", ""))[:120]} for r in signals_to_execute}

        for item in signals_to_execute:
            strat_name = item["strategy"]
            sig = item["signal"]
            action = sig.get("signal", "NO_TRADE")

            if action not in ("BUY", "SELL"):
                self._log_strategy_skip(strat_name, str(sig.get("reason") or "No signal")[:120])
                continue

            if self._strategy_open_count(strat_name) > 0:
                self._log_strategy_skip(strat_name, "Open trade already exists for this strategy")
                continue

            self._stats["signals"] += 1
            sl, tp = sig.get("sl"), sig.get("tp")
            if not sl or not tp:
                self._log_strategy_skip(strat_name, "Execution blocked: missing SL/TP")
                continue

            entry = sig.get("entry", tick["bid"] if action == "SELL" else tick["ask"])
            sl_distance = sig.get("sl_distance", abs(entry - sl))
            if sl_distance <= 0:
                self._log_strategy_skip(strat_name, "Execution blocked: invalid stop distance")
                continue

            lot = _safe_float(sig.get("lot", sig.get("lot_size")), 0.0)
            if lot <= 0:
                lot = self.risk.calculate_lot(
                    account,
                    sl_distance,
                    high_conf=bool(sig.get("_high_conf", False)),
                    strategy=strat_name,
                )
            if lot <= 0:
                self._log_strategy_skip(strat_name, "Execution blocked: invalid lot")
                continue
            lot = max(cfg.MIN_LOT, min(cfg.MAX_LOT, lot))

            strategy_obj = self.strat_mgr.get(strat_name)
            if strategy_obj is not None and hasattr(strategy_obj, "pre_send_revalidate"):
                pre_send = strategy_obj.pre_send_revalidate(sig, strat_data)
                if not pre_send.get("allowed", True):
                    self._log_strategy_skip(strat_name, f"Execution blocked: {pre_send.get('reason', 'PRE_SEND_BLOCK')}")
                    continue

            comment = sig.get("_comment") or f"FT_{strat_name[:8]}"
            result = None
            for _ in range(3):
                result = self.bridge.open_trade(action, lot, sl, tp, comment=comment)
                if result.get("success"):
                    break
                time.sleep(0.1)

            if result and result.get("success"):
                t = result["ticket"]
                fp = result["price"]
                live_tick_metrics = self.tick_proc.snapshot() or {}
                self.trades.register_trade(
                    t, action, lot, fp, sl, tp, sl_distance,
                    strategy=strat_name, confidence=sig.get("confidence", 0),
                    reason=sig.get("reason", ""),
                    scalp=bool(sig.get("_scalp", False)),
                    features={
                        "atr": self._indicators.get("atr", 0),
                        "atr_ratio": self._indicators.get("atr_ratio", 1),
                        "rsi": self._indicators.get("rsi", 50),
                        "ema_slope": self._indicators.get("ema9_slope", 0),
                        "body_ratio": self._indicators.get("body_ratio", 0),
                        "spread": (tick or {}).get("spread", 0),
                        "signal_confidence": sig.get("confidence", 0),
                        "entry_tick_velocity": live_tick_metrics.get("velocity", 0),
                        "entry_tick_count": live_tick_metrics.get("tick_count", 0),
                        "entry_tick_pressure_score": self._tick_pressure.get("pressure_score"),
                        "entry_tick_pressure_bias": self._tick_pressure.get("directional_bias"),
                        "entry_tick_burst_rate": self._tick_pressure.get("burst_rate"),
                        "regime": self._regime.get("state", ""),
                        "bias_conf": self._bias.get("confidence", 0),
                        "session": get_session(),
                        "profile_name": sig.get("_exit_profile"),
                        "tp_levels": sig.get("tp_levels") or sig.get("_tp_levels") or [],
                        "macro_bias": sig.get("_macro_bias"),
                        "news": sig.get("_news"),
                        "entry_volume_ratio": sig.get("_entry_volume_ratio"),
                        "signal_family": sig.get("_signal_family"),
                        "context_hash": sig.get("_context_hash"),
                        "signal_id": sig.get("_signal_id"),
                        "ttl_seconds": sig.get("_ttl_seconds"),
                    },
                )
                record_trade_taken()
                record_pacing_trade()
                record_session_trade()
                self.risk.record_trade_opened()
                self._confirm_strategy_execution(strat_name, sig)
                self._stats["trades"] += 1
                self.log("TRADE", f"[{strat_name}] {action} {lot:.2f} lot @ {fp} | "
                          f"SL:{sl} TP:{tp} | {sig.get('reason','')[:80]}")
            elif result:
                self.log("TRADE", f"[{strat_name}] FAILED: {result.get('error')}")

        self._last_signal = {"strategy": signals_to_execute[0]["strategy"] if signals_to_execute else "AUTO", **(signals_to_execute[0]["signal"] if signals_to_execute else {})}
        if self._last_strategy_results:
            self._last_signal["_strategy_results"] = dict(self._last_strategy_results)


    def _try_reentry(self, strategy_name: str, tick: Dict, account: Dict, positions: list):
        """Re-evaluate a strategy immediately after a recovery exit and open a new trade if valid."""
        if not self.enabled or not is_market_open():
            return
        if self._strategy_open_count(strategy_name) > 0:
            return
        strat_data = {
            "m1_df": self._candles.get("M1"), "m5_df": self._candles.get("M5"),
            "m15_df": self._candles.get("M15"), "m30_df": self._candles.get("M30"),
            "h1_df": self._candles.get("H1"), "h4_df": self._candles.get("H4"),
            "d1_df": self._candles.get("D1"),
            "symbol": self.bridge.current_symbol if self.bridge else cfg.SYMBOL,
            "tick": tick, "zones": self._zones, "indicators": self._indicators,
            "tf_context": self._tf_context, "account": account,
            "positions": positions or [],
            "correlation": self._correlation, "calendar": self._calendar,
            "bias": self._bias, "regime": self._regime, "liquidity": self._liquidity,
            "tick_snapshot": self.tick_proc.snapshot() or {},
            "tick_pressure": self._tick_pressure,
            "market_state": self._market_state,
            "_risk_manager": self.risk,
            "now_utc": datetime.now(timezone.utc),
            "strategy_trade_counts": {
                name: sum(1 for t in (self.trades.open_trades.values() if self.trades else [])
                          if getattr(t, "strategy", "") == name)
                for name in ["SWING_ENGINE", "INTRADAY_ENGINE"]
            },
        }
        strategy = self.strat_mgr.get(strategy_name)
        if strategy is None:
            return
        try:
            sig = strategy.generate_signal(strat_data)
        except Exception:
            return
        action = sig.get("signal", "NO_TRADE")
        if action not in ("BUY", "SELL"):
            self.log("REENTRY", f"[{strategy_name}] No re-entry signal after recovery exit: {sig.get('reason', '')}")
            return
        sl, tp = sig.get("sl"), sig.get("tp")
        if not sl or not tp:
            return
        entry = sig.get("entry", tick["bid"] if action == "SELL" else tick["ask"])
        sl_distance = sig.get("sl_distance", abs(entry - sl))
        if sl_distance <= 0:
            return
        lot = _safe_float(sig.get("lot", sig.get("lot_size")), 0.0)
        if lot <= 0:
            lot = self.risk.calculate_lot(account, sl_distance, strategy=strategy_name)
        if lot <= 0:
            return
        lot = max(cfg.MIN_LOT, min(cfg.MAX_LOT, lot))
        if hasattr(strategy, "pre_send_revalidate"):
            pre_send = strategy.pre_send_revalidate(sig, strat_data)
            if not pre_send.get("allowed", True):
                self.log("REENTRY", f"[{strategy_name}] Re-entry blocked: {pre_send.get('reason', 'PRE_SEND_BLOCK')}")
                return
        comment = sig.get("_comment") or f"FT_{strategy_name[:8]}_RE"
        result = self.bridge.open_trade(action, lot, sl, tp, comment=comment)
        if result and result.get("success"):
            t = result["ticket"]
            fp = result["price"]
            live_tick_metrics = self.tick_proc.snapshot() or {}
            self.trades.register_trade(
                t, action, lot, fp, sl, tp, sl_distance,
                strategy=strategy_name, confidence=sig.get("confidence", 0),
                reason=f"[re-entry] {sig.get('reason', '')}",
                features={
                    "atr": self._indicators.get("atr", 0),
                    "signal_confidence": sig.get("confidence", 0),
                    "entry_tick_velocity": live_tick_metrics.get("velocity", 0),
                    "entry_tick_count": live_tick_metrics.get("tick_count", 0),
                    "regime": self._regime.get("state", ""),
                    "session": get_session(),
                    "profile_name": sig.get("_exit_profile"),
                },
            )
            record_trade_taken()
            record_pacing_trade()
            record_session_trade()
            self.risk.record_trade_opened()
            self._stats["trades"] += 1
            self.log("REENTRY", f"[{strategy_name}] {action} {lot:.2f} lot @ {fp} | SL:{sl} TP:{tp}")
        else:
            self.log("REENTRY", f"[{strategy_name}] Re-entry order failed: {(result or {}).get('error', '')}")

    def _recompute_all(self):
        m1 = self._candles.get("M1")
        if m1 is not None and len(m1) >= 21:
            self._indicators = compute_indicators(m1)
        self._refresh_timeframe_context()
        m5 = self._candles.get("M5")
        if m5 is not None and len(m5) >= 50:
            self._zones = self.zone_detector.detect(m5)
        self._recompute_regime_bias()
        self._market_state = _safe_dict(compute_market_state(self._candles, datetime.now(timezone.utc)))

    def _refresh_timeframe_context(self):
        self._tf_context = compute_timeframe_context(
            self._candles.get("M1"),
            self._candles.get("M5"),
            self._candles.get("H1"),
        )
        if self._indicators is None:
            self._indicators = {}
        self._indicators.update(self._tf_context)

    def _poll_mt5_history(self):
        """Poll MT5 deal history - SINGLE SOURCE OF TRUTH for closed trades."""
        try:
            self._mt5_today_pnl = self.bridge.get_today_pnl()
            self._mt5_closed_history = self.bridge.get_closed_trades(30)
            # Sync risk manager — pass full list, seed_from_mt5 filters by trading day
            self.risk.seed_from_mt5(self._mt5_today_pnl, self._mt5_closed_history)
            if self._mt5_today_pnl.get("pnl", 0) >= cfg.DAILY_TARGET_DOLLARS:
                self.log("SEED", f"MT5 today_pnl=${self._mt5_today_pnl.get('pnl',0):.2f} trades={self._mt5_today_pnl.get('trades',0)} src={self._mt5_today_pnl.get('source','')} closed_count={len(self._mt5_closed_history)}")
            # Update performance tracker with MT5 data if needed
            self._sync_performance_with_mt5()
            # Reconcile order DB with MT5 deal history
            self._reconcile_db_with_mt5()
        except Exception as e:
            self.log("ERR", f"History poll failed: {e}")

    def _refresh_tick_pressure(self):
        now = time.time()
        if now - self._last_tick_pressure_poll < 0.5:
            return
        self._last_tick_pressure_poll = now
        try:
            raw_ticks = self.bridge.get_recent_ticks()
            self._tick_pressure = analyze_tick_pressure(raw_ticks)
        except Exception:
            self._tick_pressure = {"ready": False}

    def _confirm_strategy_execution(self, strat_name: str, sig: Dict):
        strategy = self.strat_mgr.get(strat_name)
        if strategy and hasattr(strategy, "confirm_trade_executed"):
            if strat_name == "SWEEP_SCALPER":
                try:
                    candle_ts = float(self._candles.get("M1", {}).iloc[-1]["datetime"].timestamp()) if self._candles.get("M1") is not None and len(self._candles.get("M1")) else 0.0
                except Exception:
                    candle_ts = 0.0
                strategy.confirm_trade_executed(float(sig.get("sweep_level") or 0.0), candle_ts)
            else:
                strategy.confirm_trade_executed(sig)

    def _strategy_open_count(self, strategy_name: str) -> int:
        if not self.trades:
            return 0
        target = str(strategy_name or "").upper()
        return sum(1 for trade in self.trades.open_trades.values() if str(trade.strategy or "").upper() == target)

    def _log_strategy_skip(self, strategy_name: str, message: str, *, cooldown_seconds: float = 30.0):
        now = time.time()
        key = f"{strategy_name}:{message}"
        if now - self._last_strategy_skip_log.get(key, 0.0) < cooldown_seconds:
            return
        self._last_strategy_skip_log[key] = now
        self.log("NO_SIG", f"[{strategy_name}] {message}")

    def _sync_performance_with_mt5(self):
        """Sync performance tracker with MT5 deal history (authoritative)."""
        try:
            self.perf.sync_from_mt5(self._mt5_closed_history, self.trades.order_db if self.trades else None)
        except Exception as e:
            self.log("ERR", f"Performance sync failed: {e}")

    def _reconcile_db_with_mt5(self):
        """Reconcile closed orders in DB against MT5 deal history.
        Fixes exit_price, final_pnl, swap, commission if they differ from MT5."""
        if not self._mt5_closed_history or not self.trades:
            return
        try:
            mt5_map = {t["ticket"]: t for t in self._mt5_closed_history}
            db = self.trades.order_db
            stale = db.get_stale_closed_orders(mt5_map)
            if not stale:
                return
            patched = 0
            for ticket, db_row in stale.items():
                mt5t = mt5_map[ticket]
                db.reconcile_order(
                    ticket,
                    final_pnl=mt5t["pnl"],
                    exit_price=mt5t.get("exit_price", 0),
                    swap=mt5t.get("swap", 0),
                    commission=mt5t.get("commission", 0),
                )
                patched += 1
            if patched:
                self.log("RECONCILE", f"Fixed {patched} orders from MT5 history")
        except Exception as e:
            self.log("ERR", f"Reconciliation failed: {e}")



    def _recompute_regime_bias(self):
        ts = self.tick_proc.snapshot()
        if ts is None:
            ts = {"ready": False}
        self._regime = _safe_dict(classify_regime(self._candles.get("H4"), self._candles.get("H1"), self._candles.get("M15"), ts))
        self._bias = _safe_dict(compute_bias(self._candles.get("H4"), self._candles.get("H1"), self._candles.get("M15")))
        self._liquidity = _safe_dict(compute_liquidity(
            self._candles.get("M5"),
            self._candles.get("M15"),
            self._candles.get("H1"),
            self._candles.get("D1"),
        ))
        self._market_state = _safe_dict(compute_market_state(self._candles, datetime.now(timezone.utc)))

    def _log_status(self, tick):
        s, p = get_session(), (tick or {}).get("bid", 0)
        regime = _safe_dict(self._regime)
        bias = _safe_dict(self._bias)
        r, b = regime.get("state", "?"), bias.get("direction", "?")
        daily_pnl = 0
        if self.trades and self.trades.order_db:
            today_stats = self.trades.order_db.get_today_stats()
            if today_stats:
                daily_pnl = today_stats.get('total_pnl', 0)
        self.log("STATUS", f"[{self.strat_mgr.active_name}] {s} | {p:.2f} | {r} | Bias:{b} | "
                 f"Open:{self.trades and self.trades.open_count or 0} | Daily:${daily_pnl:+.2f}")

    def _get_xgb_live_prediction(self) -> Dict:
        """Get live XGBoost prediction for current market conditions."""
        if xgb_bypass_enabled():
            return {"available": False, "reason": "XGBoost bypass enabled"}
        if not xgb_training_enabled():
            return {"available": False, "reason": "XGBoost training disabled"}
        if not xgb_model.is_trained or not self._last_tick:
            return {"available": False, "reason": "Model not trained or no tick data"}
        
        try:
            indicators = _safe_dict(self._indicators)
            regime = _safe_dict(self._regime)
            bias = _safe_dict(self._bias)
            # Create a mock signal for prediction
            mock_signal = {
                "signal": "SELL",  # Default direction for prediction
                "sl_distance": 0.5,
                "volume": 0.01
            }
            
            # Get prediction for SELL signal
            sell_prob = xgb_model.predict_win_prob(
                {**mock_signal, "signal": "SELL"}, 
                indicators, 
                regime, 
                bias, 
                self._last_tick
            )
            
            # Get prediction for BUY signal
            buy_prob = xgb_model.predict_win_prob(
                {**mock_signal, "signal": "BUY"}, 
                indicators, 
                regime, 
                bias, 
                self._last_tick
            )
            
            # Determine market sentiment
            if sell_prob > 0.6 or buy_prob > 0.6:
                sentiment = "FAVORABLE"
                sentiment_color = "green"
            elif sell_prob < 0.4 and buy_prob < 0.4:
                sentiment = "UNFAVORABLE"
                sentiment_color = "red"
            else:
                sentiment = "NEUTRAL"
                sentiment_color = "yellow"
            
            return {
                "available": True,
                "buy_probability": round(buy_prob, 3),
                "sell_probability": round(sell_prob, 3),
                "sentiment": sentiment,
                "sentiment_color": sentiment_color,
                "model_confidence": "HIGH" if abs(buy_prob - sell_prob) > 0.2 else "LOW",
                "features_used": {
                    "session": get_session(),
                    "atr": indicators.get("atr", 0),
                    "atr_ratio": indicators.get("atr_ratio", 1),
                    "rsi": indicators.get("rsi", 50),
                    "ema_slope": indicators.get("ema9_slope", 0),
                    "body_ratio": indicators.get("body_ratio", 0),
                    "spread": (self._last_tick or {}).get("spread", 0),
                    "regime": regime.get("state", ""),
                    "bias_conf": bias.get("confidence", 0)
                }
            }
        except Exception as e:
            return {"available": False, "reason": f"Prediction error: {str(e)}"}

    def get_full_status(
        self,
        include_closed_history: bool = True,
        include_xgb: bool = True,
        include_performance: bool = True,
    ) -> Dict:
        # Debug: Log current symbol state
        current_sym = self.bridge.current_symbol
        if hasattr(self, '_last_logged_symbol') and self._last_logged_symbol != current_sym:
            self.log("DEBUG", f"Symbol changed from {getattr(self, '_last_logged_symbol', 'unknown')} to {current_sym}")
        self._last_logged_symbol = current_sym
        indicators = _safe_dict(self._indicators)
        zones = _safe_dict(self._zones)
        regime = _safe_dict(self._regime)
        bias = _safe_dict(self._bias)
        liquidity = _safe_dict(self._liquidity)
        calendar_state = _safe_dict(self._calendar)
        if calendar_state.get("blocked") and not cfg.CALENDAR_BLOCKING_ENABLED:
            calendar_state = {
                **calendar_state,
                "blocked": False,
                "advisory": True,
                "blocking_enabled": False,
            }
        else:
            calendar_state = {
                **calendar_state,
                "blocking_enabled": bool(cfg.CALENDAR_BLOCKING_ENABLED),
            }
        correlation = _safe_dict(self._correlation)
        # Daily PNL from DB (single source of truth, IST-based)
        db_today = {}
        if self.trades and self.trades.order_db:
            db_today = self.trades.order_db.get_today_stats() or {}
        db_daily_pnl = db_today.get('total_pnl', 0)
        db_trades = db_today.get('trades', 0)
        db_wins = db_today.get('wins', 0)
        db_losses = db_today.get('losses', 0)

        closed_for_dash = []
        if include_closed_history or include_performance:
            # Closed history from DB for dashboard, enriched with MT5 data for missing fields
            db_closed = self.trades.order_db.get_closed_orders(30) if self.trades and self.trades.order_db else []
            mt5_map = {t['ticket']: t for t in (self._mt5_closed_history or [])}
            db_tickets = set()
            for o in db_closed:
                mt5_t = mt5_map.get(o['ticket'], {})
                exit_price = o.get('exit_price') or mt5_t.get('exit_price') or 0
                close_time = o.get('close_time') or mt5_t.get('close_time') or ''
                features = o.get('features') or {}
                closed_for_dash.append({
                    'ticket': o['ticket'],
                    'symbol': o['symbol'],
                    'direction': o['direction'],
                    'volume': o['volume'],
                    'entry_price': o['entry_price'],
                    'exit_price': exit_price,
                    'pnl': o['final_pnl'] or 0,
                    'won': (o['final_pnl'] or 0) > 0,
                    'open_time': o['open_time'],
                    'close_time': close_time,
                    'close_reason': o.get('close_reason') or o.get('mt5_close_reason') or o.get('strategy', ''),
                    'close_reason_category': o.get('close_reason_category') or '',
                    'comment': o.get('close_reason') or o.get('strategy', ''),
                    'strategy': o.get('strategy', ''),
                    'setup_type': features.get('_scalper_setup_type') or features.get('ai_manual_setup_type') or '',
                })
                db_tickets.add(o['ticket'])
            for t in (self._mt5_closed_history or []):
                if t['ticket'] not in db_tickets:
                    closed_for_dash.append({
                        'ticket': t['ticket'],
                        'symbol': t.get('symbol', self.bridge.current_symbol),
                        'direction': t.get('direction', ''),
                        'volume': t.get('volume', 0),
                        'entry_price': t.get('entry_price', 0),
                        'exit_price': t.get('exit_price', 0),
                        'pnl': t.get('pnl', 0),
                        'won': t.get('pnl', 0) > 0,
                        'open_time': t.get('open_time', ''),
                        'close_time': t.get('close_time', ''),
                        'close_reason': t.get('comment', 'MT5'),
                        'close_reason_category': '',
                        'comment': t.get('comment', 'MT5'),
                    })
            closed_for_dash.sort(key=lambda x: x.get('close_time') or '', reverse=True)

        performance = {}
        if include_performance:
            pnls = [t['pnl'] for t in closed_for_dash]
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p <= 0]
            total_pnl = round(sum(pnls), 2) if pnls else 0

            # pnls is in DESC order from DB; reverse for chronological drawdown calc
            pnls_chrono = list(reversed(pnls))
            avg_win = round(sum(wins) / len(wins), 2) if wins else 0
            avg_loss_vals = [abs(p) for p in losses]
            avg_loss = round(sum(avg_loss_vals) / len(avg_loss_vals), 2) if avg_loss_vals else 0
            win_rate = round(len(wins) / len(pnls), 3) if pnls else 0
            expectancy = round((win_rate * avg_win) - ((1 - win_rate) * avg_loss), 2)
            cum = 0
            peak = 0
            max_dd = 0
            for p in pnls_chrono:
                cum += p
                if cum > peak:
                    peak = cum
                dd = peak - cum
                if dd > max_dd:
                    max_dd = dd
            performance = {
                "total": len(pnls),
                "wins": len(wins),
                "losses": len(losses),
                "win_rate": win_rate,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "expectancy": expectancy,
                "total_pnl": total_pnl,
                "max_drawdown": round(max_dd, 2),
                "avg_rr": round(avg_win / avg_loss, 2) if avg_loss > 0 else 0,
            }

        return {
            "app_version": cfg.APP_VERSION,
            "enabled": self.enabled,
            "mt5_connected": self._mt5_connected,
            "strategy": self.strat_mgr.active_name,
            "strategies": self.strat_mgr.status(),
            "symbol": self.bridge.current_symbol,
            "available_symbols": cfg.AVAILABLE_SYMBOLS,
            "session": get_session(),
            "market_open": is_market_open(),
            "tick": self._last_tick,
            "account": self._last_account,
            "indicators": indicators,
            "zones": zones,
            "regime": regime,
            "bias": bias,
            "liquidity": {
                "sweeps": len(liquidity.get("sweeps", [])),
                "order_blocks": len(liquidity.get("order_blocks", [])),
                "fvg": len(liquidity.get("fvg", [])),
                "key_levels": liquidity.get("key_levels", {}),
            },
            "calendar": calendar_state,
            "correlation": correlation,
            "candles": {tf: len(df) for tf, df in self._candles.items()},
            "market_state": self._market_state,
            **self._restart_status(),
            "risk": self.risk.daily_status,
            "daily_target": cfg.DAILY_TARGET_DOLLARS,
            "daily_target_enabled": cfg.DAILY_TARGET_ENABLED,
            "risk_config": {
                "MAX_POSITIONS": cfg.MAX_POSITIONS,
                "MAX_RISK_PCT": cfg.MAX_RISK_PCT,
                "MAX_DRAWDOWN_PCT": cfg.MAX_DRAWDOWN_PCT,
                "MAX_LOT": cfg.MAX_LOT,
                "MIN_LOT": cfg.MIN_LOT,
                "DAILY_TARGET_DOLLARS": cfg.DAILY_TARGET_DOLLARS,
                "DAILY_LOSS_LIMIT_PCT": cfg.DAILY_LOSS_LIMIT_PCT,
                "MAX_CONSECUTIVE_LOSSES": cfg.MAX_CONSECUTIVE_LOSSES,
                "MIN_TRADE_COOLDOWN": cfg.MIN_TRADE_COOLDOWN,
                "LOSS_STREAK_PAUSE": cfg.LOSS_STREAK_PAUSE,
                "SESSION_MAX_TRADES": cfg.SESSION_MAX_TRADES,
            },
            "gate_config": {
                "ALL_GATES_OVERRIDE_ENABLED": cfg.ALL_GATES_OVERRIDE_ENABLED,
                "SPREAD_GATE_OVERRIDE_ENABLED": cfg.SPREAD_GATE_OVERRIDE_ENABLED,
                "COMPRESSION_GATE_OVERRIDE_ENABLED": cfg.COMPRESSION_GATE_OVERRIDE_ENABLED,
                "EXECUTION_GATE_OVERRIDE_ENABLED": cfg.EXECUTION_GATE_OVERRIDE_ENABLED,
                "TIME_GATE_OVERRIDE_ENABLED": cfg.TIME_GATE_OVERRIDE_ENABLED,
                "SPREAD_MEAN_GATE_OVERRIDE_ENABLED": cfg.SPREAD_MEAN_GATE_OVERRIDE_ENABLED,
                "SPREAD_VOLATILITY_GATE_OVERRIDE_ENABLED": cfg.SPREAD_VOLATILITY_GATE_OVERRIDE_ENABLED,
                "SPREAD_PERCENTILE_GATE_OVERRIDE_ENABLED": cfg.SPREAD_PERCENTILE_GATE_OVERRIDE_ENABLED,
                "SPREAD_DELTA_GATE_OVERRIDE_ENABLED": cfg.SPREAD_DELTA_GATE_OVERRIDE_ENABLED,
                "COMPRESSION_RANGE_GATE_OVERRIDE_ENABLED": cfg.COMPRESSION_RANGE_GATE_OVERRIDE_ENABLED,
                "ATR_RISING_GATE_OVERRIDE_ENABLED": cfg.ATR_RISING_GATE_OVERRIDE_ENABLED,
                "EXECUTION_QUALITY_GATE_OVERRIDE_ENABLED": cfg.EXECUTION_QUALITY_GATE_OVERRIDE_ENABLED,
                "TICK_DIRECTION_GATE_OVERRIDE_ENABLED": cfg.TICK_DIRECTION_GATE_OVERRIDE_ENABLED,
                "POST_SIGNAL_SPREAD_GATE_OVERRIDE_ENABLED": cfg.POST_SIGNAL_SPREAD_GATE_OVERRIDE_ENABLED,
                "SMC_SPREAD_MEAN_MAX": cfg.SMC_SPREAD_MEAN_MAX,
                "SMC_SPREAD_STD_MAX": cfg.SMC_SPREAD_STD_MAX,
                "SMC_SPREAD_PERCENTILE_MAX": cfg.SMC_SPREAD_PERCENTILE_MAX,
                "SMC_CURRENT_SPREAD_DELTA_MAX": cfg.SMC_CURRENT_SPREAD_DELTA_MAX,
                "SCALPER_SPREAD_MEAN_MAX": cfg.SCALPER_SPREAD_MEAN_MAX,
                "SCALPER_SPREAD_STD_MAX": cfg.SCALPER_SPREAD_STD_MAX,
                "SCALPER_SPREAD_PERCENTILE_MAX": cfg.SCALPER_SPREAD_PERCENTILE_MAX,
                "SCALPER_CURRENT_SPREAD_DELTA_MAX": cfg.SCALPER_CURRENT_SPREAD_DELTA_MAX,
                "COMPRESSION_RANGE_LOOKBACK": cfg.COMPRESSION_RANGE_LOOKBACK,
                "COMPRESSION_ATR_MULTIPLIER": cfg.COMPRESSION_ATR_MULTIPLIER,
                "SMC_THRESHOLD_WITH_TREND": cfg.SMC_THRESHOLD_WITH_TREND,
                "SMC_THRESHOLD_COUNTER": cfg.SMC_THRESHOLD_COUNTER,
                "SMC_THRESHOLD_COUNTER_MAX": cfg.SMC_THRESHOLD_COUNTER_MAX,
                "SMC_MIN_RR": cfg.SMC_MIN_RR,
                "SMC_TIMEFRAME_EMA_SLOPE_MIN": cfg.SMC_TIMEFRAME_EMA_SLOPE_MIN,
                "SMC_LTF_TICK_CONFLICT_LONG_MAX": cfg.SMC_LTF_TICK_CONFLICT_LONG_MAX,
                "SMC_LTF_TICK_CONFLICT_SHORT_MIN": cfg.SMC_LTF_TICK_CONFLICT_SHORT_MIN,
                "SCALPER_QUALITY_THRESHOLD": cfg.SCALPER_QUALITY_THRESHOLD,
                "SCALPER_ATR_MIN": cfg.SCALPER_ATR_MIN,
                "SCALPER_ATR_MAX": cfg.SCALPER_ATR_MAX,
                "SCALPER_EMA20_SLOPE_MIN": cfg.SCALPER_EMA20_SLOPE_MIN,
                "SCALPER_BODY_RATIO_MIN": cfg.SCALPER_BODY_RATIO_MIN,
                "SCALPER_TICK_DIR_THRESHOLD": cfg.SCALPER_TICK_DIR_THRESHOLD,
                "SCALPER_SWEEP_LOOKBACK": cfg.SCALPER_SWEEP_LOOKBACK,
                "SCALPER_SWEEP_TOLERANCE": cfg.SCALPER_SWEEP_TOLERANCE,
                "SCALPER_MAX_TRADES_SESSION": cfg.SCALPER_MAX_TRADES_SESSION,
                "SCALPER_LEVEL_COOLDOWN": cfg.SCALPER_LEVEL_COOLDOWN,
                "MARKET_TICK_POLL_INTERVAL": cfg.MARKET_TICK_POLL_INTERVAL,
                "DASHBOARD_WS_PUSH_INTERVAL": cfg.DASHBOARD_WS_PUSH_INTERVAL,
            },
            "tier1_enabled": cfg.TIER1_ENABLED,
            "session_override_enabled": cfg.SESSION_OVERRIDE_ENABLED,
            "mt5_today_pnl": {
                'pnl': db_daily_pnl,
                'trades': db_trades,
                'wins': db_wins,
                'losses': db_losses,
                'source': 'orders_db',
            },
            "mt5_positions": self._cached_positions,
            "mt5_floating_pnl": self._cached_floating_pnl,
            "closed_history": closed_for_dash if include_closed_history else [],
            "xgb": xgb_model.get_feature_importance() if include_xgb else {},
            "xgb_live_prediction": {},  # overwritten by trader_api with cached version
            "performance": performance,
            "trades": self.trades.status if self.trades else {"open_trades": [], "open_count": 0, "today_stats": {"trades": 0, "wins": 0, "losses": 0, "total_pnl": 0, "avg_pnl": 0, "best_trade": 0, "worst_trade": 0, "open_count": 0, "open_pnl": 0}},
            "stats": self._stats,
            "live_blockers": {},  # overwritten by trader_api with cached version
            "log": list(self._log)[-50:],
        }

    def _run_http_server(self):
        dash_path = os.path.join(_BASE_DIR, "dashboard.html")
        with open(dash_path, encoding="utf-8") as f:
            dashboard_html = f.read()
        trader = self
        fallback_order_db = OrderDatabase()

        class H(BaseHTTPRequestHandler):
            def log_message(self, *_): pass
            def _j(s, d, c=200):
                b = json.dumps(d, default=str).encode()
                s.send_response(c); s.send_header("Content-Type","application/json")
                s.send_header("Access-Control-Allow-Origin","*"); s.end_headers(); s.wfile.write(b)
            def _h(s, c):
                s.send_response(200); s.send_header("Content-Type","text/html; charset=utf-8")
                s.end_headers(); s.wfile.write(c.encode("utf-8"))
            def _bytes(s, payload, content_type="application/octet-stream", c=200):
                s.send_response(c)
                s.send_header("Content-Type", content_type)
                s.send_header("Access-Control-Allow-Origin","*")
                s.end_headers()
                s.wfile.write(payload)
            def _order_db(s):
                if trader.trades and trader.trades.order_db:
                    return trader.trades.order_db
                return fallback_order_db

            def do_GET(s):
                parsed = urlparse(s.path)
                p = parsed.path
                q = parse_qs(parsed.query)
                if p == "/status": s._j(trader.get_full_status())
                elif p == "/logs": s._j({"logs": list(trader._log)[-200:]})
                elif p == "/logs/today":
                    import re as _re
                    log_path = os.path.join(_BASE_DIR, "trader.log")
                    pattern = _re.compile(
                        r'^\[(\d{2}:\d{2}:\d{2}\.\d{3}) UTC \| (\d{2}:\d{2}:\d{2}\.\d{3}) IST\]'
                        r'\[([A-Z_]+)\](?:\[([\d.]+)\])? (.*)$'
                    )
                    entries = []
                    try:
                        with open(log_path, "rb") as _f:
                            raw = _f.read().decode("utf-8", errors="ignore")
                        for _line in raw.splitlines():
                            _m = pattern.match(_line.strip())
                            if not _m: continue
                            _ut, _it, _tag, _pr, _msg = _m.groups()
                            entries.append({"time": _it, "time_ist": _it, "tag": _tag, "msg": _msg, "price": float(_pr) if _pr else None})
                    except Exception as _e:
                        entries = []
                    s._j({"logs": entries, "count": len(entries)})
                elif p == "/trades": s._j(trader.trades.status if trader.trades else {})
                elif p == "/performance": s._j(trader.perf.get_stats())
                elif p == "/health": s._j({"status":"healthy","service":"auto_trader","app_version":cfg.APP_VERSION})
                elif p == "/config":
                    from engine import strategy_configs as _scfg
                    _all_cfg = {k: getattr(cfg, k) for k in cfg._PERSISTED_KEYS if hasattr(cfg, k) and isinstance(getattr(cfg, k), (bool, int, float, str, list))}
                    s._j({"app_version":cfg.APP_VERSION,"symbol":cfg.SYMBOL,"strategy":trader.strat_mgr.active_name,
                        "strategies":trader.strat_mgr.status(),
                        "config": _all_cfg,
                        "strategy_configs": _scfg.get_all(),
                    })
                elif p == "/analytics":
                    days = int((q.get("days") or ["30"])[0])
                    refresh = int((q.get("refresh") or ["0"])[0])
                    manifest_path = _LATEST_DIR / "manifest.json"
                    if refresh or not _LATEST_DIR.exists():
                        result = generate_visual_suite(days=days, output_dir=str(_LATEST_DIR))
                    else:
                        if manifest_path.exists():
                            result = json.loads(manifest_path.read_text(encoding="utf-8"))
                            if result.get("period_days") != days:
                                result = generate_visual_suite(days=days, output_dir=str(_LATEST_DIR))
                        else:
                            result = generate_visual_suite(days=days, output_dir=str(_LATEST_DIR))
                    manifest_path.parent.mkdir(parents=True, exist_ok=True)
                    manifest_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
                    s._h(_build_analytics_page(result, days))
                elif p == "/analytics/api/generate":
                    days = int((q.get("days") or ["30"])[0])
                    result = generate_visual_suite(days=days, output_dir=str(_LATEST_DIR))
                    manifest_path = _LATEST_DIR / "manifest.json"
                    manifest_path.parent.mkdir(parents=True, exist_ok=True)
                    manifest_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
                    s._j(result)
                elif p.startswith("/analytics-assets/"):
                    rel_path = unquote(p[len("/analytics-assets/"):]).lstrip("/")
                    target = (_OUTPUT_ROOT / rel_path).resolve()
                    if not str(target).startswith(str(_OUTPUT_ROOT.resolve())) or not target.is_file():
                        s._j({"error":"Asset not found"},404)
                    else:
                        mime_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
                        s._bytes(target.read_bytes(), mime_type)
                elif p == "/orders/open":
                    s._j(s._order_db().get_open_orders())
                elif p == "/orders/closed":
                    days = int((q.get("days") or ["30"])[0])
                    s._j(s._order_db().get_closed_orders(days))
                elif p == "/orders/stats/today":
                    s._j(s._order_db().get_today_stats() or {})
                elif p == "/orders/stats/strategy":
                    days = int((q.get("days") or ["30"])[0])
                    s._j(s._order_db().get_strategy_performance(days))
                elif p == "/orders/daily-pnl":
                    days = int((q.get("days") or ["30"])[0])
                    s._j(s._order_db().get_daily_pnl(days))
                elif p == "/orders/summary":
                    order_db = s._order_db()
                    today_stats = order_db.get_today_stats() or {}
                    open_orders = order_db.get_open_orders()
                    s._j({
                        "today": today_stats,
                        "strategies_30d": order_db.get_strategy_performance(30),
                        "open_orders_count": len(open_orders),
                        "open_orders": open_orders[:5],
                        "summary_generated_at": datetime.now().isoformat(),
                    })
                elif p == "/orders/review":
                    days = int((q.get("days") or ["30"])[0])
                    s._j(s._order_db().get_trade_outcome_review(days))
                elif p in ("/","/dashboard"): s._h(dashboard_html)
                else: s._j({"error":"Not found"},404)

            def do_POST(s):
                p = s.path
                if p == "/start":
                    trader.enabled = True; trader.log("API","Trading ENABLED"); s._j({"enabled":True,"strategy":trader.strat_mgr.active_name if trader.strat_mgr else "AUTO"})
                elif p == "/stop":
                    trader.enabled = False; trader.log("API","Trading DISABLED"); s._j({"enabled":False})
                elif p == "/emergency":
                    trader.enabled = False
                    if trader.trades: trader.trades.close_all()
                    trader.log("API","EMERGENCY STOP"); s._j({"enabled":False})
                elif p == "/strategies/select":
                    import json as _json
                    length = int(s.headers.get('Content-Length', 0))
                    body = _json.loads(s.rfile.read(length)) if length else {}
                    raw_selected = body.get("selected") or []
                    if isinstance(raw_selected, str):
                        raw_selected = [raw_selected]
                    selected = [str(item or "").strip().upper() for item in raw_selected if str(item or "").strip()]
                    if trader.strat_mgr.set_selection(selected):
                        cfg.DEFAULT_STRATEGY = "AUTO" if trader.strat_mgr.is_auto else trader.strat_mgr.enabled_strategies()
                        try:
                            cfg.save_runtime_config()
                        except Exception:
                            pass
                        trader.log("API", f"Strategy selection -> {trader.strat_mgr.active_name}")
                        s._j(trader.strat_mgr.status())
                    else:
                        s._j({"error":f"Unknown selection: {selected}","available":trader.strat_mgr.available},400)
                elif p.startswith("/strategy/"):
                    name = p.split("/strategy/")[1].upper()
                    if trader.strat_mgr.set_active(name):
                        cfg.DEFAULT_STRATEGY = name
                        try:
                            cfg.save_runtime_config()
                        except Exception:
                            pass
                        trader.log("API", f"Strategy -> {name}")
                        s._j(trader.strat_mgr.status())
                    else:
                        s._j({"error":f"Unknown: {name}","available":trader.strat_mgr.available},400)
                elif p.startswith("/symbol/"):
                    symbol = p.split("/symbol/")[1].upper()
                    if symbol in cfg.AVAILABLE_SYMBOLS:
                        if trader.bridge.set_symbol(symbol):
                            # Clear all cached data when symbol changes
                            trader._candles.clear()
                            trader._candle_counts.clear()
                            trader._indicators.clear()
                            trader._zones.clear()
                            trader._regime.clear()
                            trader._bias.clear()
                            trader._liquidity.clear()
                            trader._last_tick = {}
                            # Force immediate data refresh for new symbol
                            try:
                                for tf in ["M1", "M5", "M15", "H1", "H4", "D1"]:
                                    df = trader.bridge.fetch_candles(tf)
                                    if not df.empty:
                                        trader._candles[tf] = df
                                        trader._candle_counts[tf] = len(df)
                                trader._recompute_all()
                            except Exception as e:
                                trader.log("API", f"Symbol data refresh failed: {e}")
                            trader.log("API", f"Symbol -> {symbol} (cache cleared, data refreshed)")
                            s._j({"symbol":symbol,"available":cfg.AVAILABLE_SYMBOLS})
                        else:
                            s._j({"error":f"Failed to set {symbol}"},500)
                    else:
                        s._j({"error":f"Unknown symbol: {symbol}","available":cfg.AVAILABLE_SYMBOLS},400)
                elif p == "/config":
                    import json as _json
                    length = int(s.headers.get('Content-Length', 0))
                    body = _json.loads(s.rfile.read(length)) if length else {}
                    updated = {}
                    for k, v in body.items():
                        if not hasattr(cfg, k):
                            continue
                        cur = getattr(cfg, k)
                        try:
                            if isinstance(cur, bool) or k.endswith('_ENABLED'):
                                val = bool(v)
                            elif isinstance(cur, int):
                                val = int(v)
                            elif isinstance(cur, float):
                                val = float(v)
                            elif isinstance(cur, list):
                                val = list(v) if isinstance(v, list) else [v]
                            elif isinstance(cur, str):
                                val = str(v)
                            else:
                                val = v
                        except (TypeError, ValueError):
                            continue
                        setattr(cfg, k, val)
                        updated[k] = val
                    if updated:
                        try:
                            cfg.save_runtime_config()
                        except Exception as e:
                            trader.log("API", f"Config persistence failed: {e}")
                        try:
                            adaptive_params._capture_base()
                        except Exception:
                            pass
                        if any(k in updated for k in ('LOSS_STREAK_PAUSE', 'MAX_CONSECUTIVE_LOSSES')):
                            trader.risk._loss_streak_pause_until = 0.0
                            trader.risk._consecutive_losses = 0
                        trader.log("API", f"Config updated: {updated}")
                    s._j({"updated":updated, "config_version": getattr(cfg, '_config_version', 0)})
                elif p == "/shutdown":
                    trader.enabled = False; trader._running = False
                    if trader.trades: trader.trades.close_all()
                    trader.log("API","SHUTDOWN"); s._j({"message":"Shutting down"})
                    threading.Thread(target=lambda:(time.sleep(1),os._exit(0)),daemon=True).start()
                else: s._j({"error":"Unknown"},404)

        server = HTTPServer((cfg.API_HOST, cfg.API_PORT), H)
        server.serve_forever()


if __name__ == "__main__":
    AutoTrader().run_service()
