"""
XAUUSD Auto Trader — Background Service with selectable strategies.

Strategies:
  SMC_CONFLUENCE  — multi-TF Smart Money swing trades (RR 1.5+)
  SWEEP_SCALPER   — liquidity sweep scalps ($3-$6, London/NY open)

API:
  POST /strategy/SMC_CONFLUENCE   — switch strategy
  POST /strategy/SWEEP_SCALPER    — switch strategy
  POST /start | /stop | /emergency | /shutdown
  GET  /status | /logs | /trades | /performance | /config | /
"""
import time
import sys
import threading
import json
import os
import mimetypes
from collections import deque
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
from engine.tick_processor import TickProcessor
from engine.regime import classify_regime
from engine.mtf_bias import compute_bias
from engine.liquidity import compute_liquidity
from engine.performance import PerformanceTracker
from engine.strategy_manager import StrategyManager
from engine.smc_strategy import SMCStrategy
from engine.sweep_scalper import SweepScalper
from engine.calendar import calendar as eco_calendar
from engine.correlation import correlation as corr_engine
from engine.order_database import OrderDatabase
from engine.visual_analytics import generate_visual_suite
from engine.xgb_model import xgb_model
from engine.anti_starvation import record_trade_taken
from engine.trade_pacing import record_trade_taken as record_pacing_trade
from engine.session_risk_control import record_trade_taken as record_session_trade
from engine.decision_logger import get_live_blockers
from engine import adaptive_params

_TF_REFRESH = {"M1": 1, "M5": 2, "M15": 10, "H1": 60, "H4": 120}
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _safe_dict(value):
    return value if isinstance(value, dict) else {}


class AutoTrader:
    def __init__(self):
        self.bridge = MT5Bridge()
        self.zone_detector = ZoneDetector()
        self.risk = RiskManager()
        self.trades: TradeManager = None
        self.tick_proc = TickProcessor()
        self.perf = PerformanceTracker(os.path.join(_BASE_DIR, "trade_history.json"))

        # Strategy manager
        self.strat_mgr = StrategyManager()
        self.strat_mgr.register(SMCStrategy())
        self.strat_mgr.register(SweepScalper())
        if cfg.DEFAULT_STRATEGY in self.strat_mgr.available:
            self.strat_mgr.set_active(cfg.DEFAULT_STRATEGY)

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

        self._running = False
        self.enabled = False
        self._mt5_connected = False
        self._cycle = 0
        self._stats = {"cycles": 0, "signals": 0, "trades": 0, "errors": 0}
        self._last_log_time = 0
        self._log: deque = deque(maxlen=500)
        self._last_tick: Dict = {}
        self._tick_seq = 0
        self._last_tick_signature = None
        self._tick_pump_started = False
        self._last_account: Dict = {}

        # MT5 history cache (polled every 5s in engine loop)
        self._mt5_closed_history: list = []
        self._mt5_today_pnl: Dict = {}
        self._last_history_poll = 0.0

        # Cached positions for API/WS layer (updated after manage_all so closed trades vanish immediately)
        self._cached_positions: list = []
        self._cached_floating_pnl: Dict = {"total": 0, "count": 0}
        self._positions_version: int = 0

    def log(self, tag: str, msg: str):
        now = datetime.now(timezone.utc)
        utc_str = now.strftime("%H:%M:%S.%f")[:-3]
        ist_str = now.astimezone(_IST).strftime("%H:%M:%S.%f")[:-3]
        entry = {"time": utc_str, "time_ist": ist_str, "tag": tag, "msg": msg}
        self._log.append(entry)
        try:
            with open(os.path.join(_BASE_DIR, "trader.log"), "a") as f:
                f.write(f"[{utc_str} UTC | {ist_str} IST][{tag}] {msg}\n")
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
            if not xgb_model.is_trained and len(featured) >= 15:
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
        _pos_refresh_interval = 0.1  # refresh positions every 100ms independently of engine cycle
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
                    # Refresh positions at 100ms cadence so live P&L stays smooth
                    # regardless of how long the engine cycle takes
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

    def _cycle_once(self):
        self._cycle += 1
        self._stats["cycles"] += 1

        tick = self._last_tick or self.bridge.get_tick()
        if not tick:
            return
        self._last_tick = tick

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
                if tf in ("M15", "H1", "H4"):
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
        closed = self.trades.manage_all(all_positions or [], tick_metrics=self.tick_proc.snapshot())

        # Bump version on close so WS pushes immediately (tick pump handles live P&L refresh)
        if closed:
            self._positions_version += 1

        for ticket, pnl, won in closed:
            # Record to performance tracker with MT5-sourced P&L + features from DB
            record = {"ticket": ticket, "pnl": pnl, "won": won,
                      "time": datetime.now(timezone.utc).isoformat()}
            db_order = self.trades.order_db.get_order(ticket)
            if db_order and db_order.get("features"):
                record["features"] = db_order["features"]
            self.perf.record(record)
            self.risk.record_trade_result(pnl, won)
            self.risk.set_risk_multiplier(self.perf.risk_multiplier())
            self.log("RESULT", f"#{ticket} {'WIN' if won else 'LOSS'} ${pnl:+.2f} (MT5 sourced)")
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

        # Periodic status (moved before gates so it always fires)
        now = now_ts
        if now - self._last_log_time > 30:
            self._last_log_time = now
            self._log_status(tick)

        if not self.enabled:
            return
        if not is_market_open():
            if now - getattr(self, '_last_gate_log', 0) > 60:
                self._last_gate_log = now
                self.log("GATE", "Market closed")
            return
        blocked, block_reason = is_session_open_blocked()
        if blocked:
            if now - getattr(self, '_last_gate_log', 0) > 60:
                self._last_gate_log = now
                self.log("GATE", f"Session blocked: {block_reason}")
            return
        calendar_state = _safe_dict(self._calendar)
        if calendar_state.get("blocked"):
            if now - getattr(self, '_last_gate_log', 0) > 60:
                self._last_gate_log = now
                self.log("GATE", f"Calendar blocked: {calendar_state.get('reason', '')}")
            return
        allowed, risk_reason = self.risk.can_trade(account, len(positions or []))  # count only our positions
        if not allowed:
            if now - getattr(self, '_last_gate_log', 0) > 60:
                self._last_gate_log = now
                self.log("GATE", f"Risk blocked: {risk_reason}")
            return

        # === Strategy signal ===
        strat_data = {
            "m1_df": self._candles.get("M1"), "m5_df": self._candles.get("M5"),
            "m15_df": self._candles.get("M15"), "h1_df": self._candles.get("H1"),
            "h4_df": self._candles.get("H4"),
            "tick": tick, "zones": self._zones, "indicators": self._indicators,
            "tf_context": self._tf_context,
            "account": account, "positions": positions or [],
            "correlation": self._correlation, "calendar": self._calendar,
        }
        sig, strat_name, trade_id = self.strat_mgr.generate_signal(strat_data)
        action = sig.get("signal", "NO_TRADE")

        if action not in ("BUY", "SELL"):
            if now - getattr(self, '_last_sig_log', 0) > 30:
                self._last_sig_log = now
                reason = sig.get("reason", "unknown")[:120]
                score = sig.get("score", 0)
                # Log arbitration context if present
                arb = sig.get("_arb_log")
                if arb:
                    self.log("ARB", f"[{strat_name}] setup:{arb.get('setup_dir','?')} "
                             f"bias:{arb.get('bias_dir','?')} Q:{arb.get('quality',0):.0%} "
                             f"T:{arb.get('threshold',0):.0%} "
                             f"{'CTR' if arb.get('counter') else 'WTR'} "
                             f"PB:{arb.get('pullback',False)} | {reason[:80]}")
                # In AUTO mode, show all strategy results
                auto_results = sig.get("_auto_results", {})
                if auto_results:
                    parts = [f"{n}: {r.get('reason','-')[:60]}" for n, r in auto_results.items()]
                    self.log("NO_SIG", " | ".join(parts))
                else:
                    self.log("NO_SIG", f"[{strat_name}] {reason} (score:{score:.2f})")
            return

        self._stats["signals"] += 1

        # XGBoost: adjust confidence with win probability prediction
        xgb_prob = xgb_model.predict_win_prob(sig, self._indicators, self._regime, self._bias, tick)
        orig_conf = sig.get("confidence", 0)
        if xgb_model.is_trained and xgb_prob < 0.35:
            self.log("XGB", f"[{strat_name}] Blocked: win_prob {xgb_prob:.0%} < 35%")
            return
        if xgb_model.is_trained:
            blended = round(orig_conf * 0.7 + xgb_prob * 0.3, 3)
            sig["confidence"] = blended
            sig["xgb_prob"] = xgb_prob

        sl, tp = sig.get("sl"), sig.get("tp")
        if not sl or not tp:
            return
        entry = sig.get("entry", tick["bid"] if action == "SELL" else tick["ask"])
        sl_distance = sig.get("sl_distance", abs(entry - sl))
        if sl_distance <= 0 or sig.get("confidence", 0) < 0.55:
            return

        # Lot sizing — scalper caps at 0.05, SMC uses full risk calc
        is_scalp = sig.get("_scalp", False)
        high_conf = sig.get("_high_conf", False)
        lot = self.risk.calculate_lot(account or {}, sl_distance, high_conf=high_conf)
        if is_scalp:
            lot = min(0.05, lot)
        elif get_session() == "ASIAN":
            lot = round(lot * 0.7, 2)
        lot = max(cfg.MIN_LOT, min(cfg.MAX_LOT, lot))

        # Execute with retry + spread re-check
        signal_spread = sig.get("_entry_spread", (tick or {}).get("spread", 0))
        if cfg.TIER1_ENABLED and self.tick_proc.spread_changed(signal_spread, max_delta=cfg.SMC_TIER1_SPREAD_DELTA_MAX):
            self.log("BLOCKED", f"[{strat_name}] Spread widened since signal")
            return
        comment = f"FT_{strat_name[:8]}"
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
                scalp=is_scalp,
                be_trigger=sig.get("_be_trigger", 0),
                timeout=sig.get("_timeout", 0),
                early_fail=sig.get("_early_fail", 0),
                tier1_min_ticks=sig.get("_tier1_min_ticks", cfg.SMC_TIER1_MIN_TICKS),
                tier1_max_ticks=sig.get("_tier1_max_ticks", cfg.SMC_TIER1_MAX_TICKS),
                features={
                    "atr": self._indicators.get("atr", 0),
                    "atr_ratio": self._indicators.get("atr_ratio", 1),
                    "rsi": self._indicators.get("rsi", 50),
                    "ema_slope": self._indicators.get("ema9_slope", 0),
                    "body_ratio": self._indicators.get("body_ratio", 0),
                    "spread": (tick or {}).get("spread", 0),
                    "entry_tick_velocity": live_tick_metrics.get("velocity", 0),
                    "entry_tick_count": live_tick_metrics.get("tick_count", 0),
                    "regime": self._regime.get("state", ""),
                    "bias_conf": self._bias.get("confidence", 0),
                    "session": get_session(),
                },
            )
            record_trade_taken()
            record_pacing_trade()
            record_session_trade()
            self.risk.record_trade_opened()
            # Confirm sweep scalper level lock after successful execution
            if is_scalp and hasattr(sig.get('_original_signal', sig), 'get'):
                sweep_lvl = sig.get('sweep_level') or sig.get('_original_signal', {}).get('sweep_level')
            else:
                sweep_lvl = sig.get('sweep_level')
            if is_scalp and sweep_lvl:
                scalper = self.strat_mgr.get('SWEEP_SCALPER')
                if scalper:
                    scalper.confirm_trade_executed(sweep_lvl)
            self._stats["trades"] += 1
            self.log("TRADE", f"[{strat_name}] {action} {lot} lot @ {fp} | "
                     f"SL:{sl} TP:{tp} RR:{sig.get('rr',0):.1f} | {sig.get('reason','')[:80]}")
        elif result:
            self.log("TRADE", f"FAILED: {result.get('error')}")



    def _recompute_all(self):
        m1 = self._candles.get("M1")
        if m1 is not None and len(m1) >= 21:
            self._indicators = compute_indicators(m1)
        self._refresh_timeframe_context()
        m5 = self._candles.get("M5")
        if m5 is not None and len(m5) >= 50:
            self._zones = self.zone_detector.detect(m5)
        self._recompute_regime_bias()

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
            # Sync risk manager daily P&L from MT5 deals (overwrite incremental tracking)
            # Pass today's closed trades so consecutive losses is computed from actual sequence
            today_str = datetime.now(_IST).strftime("%Y-%m-%d")
            ist_midnight_utc = datetime.now(_IST).replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc).isoformat()
            today_closed = [t for t in self._mt5_closed_history if t.get("close_time", "") >= ist_midnight_utc]
            self.risk.seed_from_mt5(self._mt5_today_pnl, today_closed)
            if self._mt5_today_pnl.get("pnl", 0) >= cfg.DAILY_TARGET_DOLLARS:
                self.log("SEED", f"MT5 today_pnl=${self._mt5_today_pnl.get('pnl',0):.2f} trades={self._mt5_today_pnl.get('trades',0)} src={self._mt5_today_pnl.get('source','')} today_closed={len(today_closed)}")
            # Update performance tracker with MT5 data if needed
            self._sync_performance_with_mt5()
            # Reconcile order DB with MT5 deal history
            self._reconcile_db_with_mt5()
        except Exception as e:
            self.log("ERR", f"History poll failed: {e}")

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
        # Ensure ts is not None before passing to classify_regime
        if ts is None:
            ts = {"ready": False}
        self._regime = _safe_dict(classify_regime(self._candles.get("H4"), self._candles.get("H1"), self._candles.get("M15"), ts))
        self._bias = _safe_dict(compute_bias(self._candles.get("H4"), self._candles.get("H1"), self._candles.get("M15")))
        self._liquidity = _safe_dict(compute_liquidity(
            self._candles.get("M5"),
            self._candles.get("M15"),
            self._candles.get("H1"),
            None,
        ))

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

    def get_full_status(self) -> Dict:
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
        correlation = _safe_dict(self._correlation)
        # Daily PNL from DB (single source of truth, IST-based)
        db_today = {}
        if self.trades and self.trades.order_db:
            db_today = self.trades.order_db.get_today_stats() or {}
        db_daily_pnl = db_today.get('total_pnl', 0)
        db_trades = db_today.get('trades', 0)
        db_wins = db_today.get('wins', 0)
        db_losses = db_today.get('losses', 0)

        # Closed history from DB for dashboard, enriched with MT5 data for missing fields
        db_closed = self.trades.order_db.get_closed_orders(30) if self.trades and self.trades.order_db else []
        mt5_map = {t['ticket']: t for t in (self._mt5_closed_history or [])}
        closed_for_dash = []
        db_tickets = set()
        for o in db_closed:
            mt5_t = mt5_map.get(o['ticket'], {})
            exit_price = o.get('exit_price') or mt5_t.get('exit_price') or 0
            close_time = o.get('close_time') or mt5_t.get('close_time') or ''
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
                'comment': o.get('close_reason') or o.get('strategy', ''),
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
                    'comment': t.get('comment', 'MT5'),
                })
        closed_for_dash.sort(key=lambda x: x.get('close_time') or '', reverse=True)

        # Performance stats from DB closed history
        pnls = [t['pnl'] for t in closed_for_dash]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        total_pnl = round(sum(pnls), 2) if pnls else 0

        # Performance stats computed entirely from DB
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
            "closed_history": closed_for_dash,
            "xgb": xgb_model.get_feature_importance(),
            "xgb_live_prediction": {},  # overwritten by trader_api with cached version
            "performance": {
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
            },
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
                elif p == "/trades": s._j(trader.trades.status if trader.trades else {})
                elif p == "/performance": s._j(trader.perf.get_stats())
                elif p == "/health": s._j({"status":"healthy","service":"auto_trader","app_version":cfg.APP_VERSION})
                elif p == "/config": s._j({"app_version":cfg.APP_VERSION,"symbol":cfg.SYMBOL,"strategy":trader.strat_mgr.active_name,
                    "strategies":trader.strat_mgr.status(),"risk_pct":cfg.MAX_RISK_PCT,
                    "daily_target":cfg.DAILY_TARGET_DOLLARS,
                    "tier1_enabled":cfg.TIER1_ENABLED,
                    "session_override_enabled":cfg.SESSION_OVERRIDE_ENABLED,
                    "gate_config":{
                        "ALL_GATES_OVERRIDE_ENABLED":cfg.ALL_GATES_OVERRIDE_ENABLED,
                        "SPREAD_GATE_OVERRIDE_ENABLED":cfg.SPREAD_GATE_OVERRIDE_ENABLED,
                        "COMPRESSION_GATE_OVERRIDE_ENABLED":cfg.COMPRESSION_GATE_OVERRIDE_ENABLED,
                        "EXECUTION_GATE_OVERRIDE_ENABLED":cfg.EXECUTION_GATE_OVERRIDE_ENABLED,
                        "TIME_GATE_OVERRIDE_ENABLED":cfg.TIME_GATE_OVERRIDE_ENABLED,
                        "SPREAD_MEAN_GATE_OVERRIDE_ENABLED":cfg.SPREAD_MEAN_GATE_OVERRIDE_ENABLED,
                        "SPREAD_VOLATILITY_GATE_OVERRIDE_ENABLED":cfg.SPREAD_VOLATILITY_GATE_OVERRIDE_ENABLED,
                        "SPREAD_PERCENTILE_GATE_OVERRIDE_ENABLED":cfg.SPREAD_PERCENTILE_GATE_OVERRIDE_ENABLED,
                        "SPREAD_DELTA_GATE_OVERRIDE_ENABLED":cfg.SPREAD_DELTA_GATE_OVERRIDE_ENABLED,
                        "COMPRESSION_RANGE_GATE_OVERRIDE_ENABLED":cfg.COMPRESSION_RANGE_GATE_OVERRIDE_ENABLED,
                        "ATR_RISING_GATE_OVERRIDE_ENABLED":cfg.ATR_RISING_GATE_OVERRIDE_ENABLED,
                        "EXECUTION_QUALITY_GATE_OVERRIDE_ENABLED":cfg.EXECUTION_QUALITY_GATE_OVERRIDE_ENABLED,
                        "TICK_DIRECTION_GATE_OVERRIDE_ENABLED":cfg.TICK_DIRECTION_GATE_OVERRIDE_ENABLED,
                        "POST_SIGNAL_SPREAD_GATE_OVERRIDE_ENABLED":cfg.POST_SIGNAL_SPREAD_GATE_OVERRIDE_ENABLED,
                        "SMC_SPREAD_MEAN_MAX":cfg.SMC_SPREAD_MEAN_MAX,
                        "SMC_SPREAD_STD_MAX":cfg.SMC_SPREAD_STD_MAX,
                        "SMC_SPREAD_PERCENTILE_MAX":cfg.SMC_SPREAD_PERCENTILE_MAX,
                        "SMC_CURRENT_SPREAD_DELTA_MAX":cfg.SMC_CURRENT_SPREAD_DELTA_MAX,
                        "SCALPER_SPREAD_MEAN_MAX":cfg.SCALPER_SPREAD_MEAN_MAX,
                        "SCALPER_SPREAD_STD_MAX":cfg.SCALPER_SPREAD_STD_MAX,
                        "SCALPER_SPREAD_PERCENTILE_MAX":cfg.SCALPER_SPREAD_PERCENTILE_MAX,
                        "SCALPER_CURRENT_SPREAD_DELTA_MAX":cfg.SCALPER_CURRENT_SPREAD_DELTA_MAX,
                        "COMPRESSION_RANGE_LOOKBACK":cfg.COMPRESSION_RANGE_LOOKBACK,
                        "COMPRESSION_ATR_MULTIPLIER":cfg.COMPRESSION_ATR_MULTIPLIER,
                        "SMC_THRESHOLD_WITH_TREND":cfg.SMC_THRESHOLD_WITH_TREND,
                        "SMC_THRESHOLD_COUNTER":cfg.SMC_THRESHOLD_COUNTER,
                        "SMC_THRESHOLD_COUNTER_MAX":cfg.SMC_THRESHOLD_COUNTER_MAX,
                        "SMC_MIN_RR":cfg.SMC_MIN_RR,
                        "SMC_TIMEFRAME_EMA_SLOPE_MIN":cfg.SMC_TIMEFRAME_EMA_SLOPE_MIN,
                        "SMC_LTF_TICK_CONFLICT_LONG_MAX":cfg.SMC_LTF_TICK_CONFLICT_LONG_MAX,
                        "SMC_LTF_TICK_CONFLICT_SHORT_MIN":cfg.SMC_LTF_TICK_CONFLICT_SHORT_MIN,
                        "SCALPER_QUALITY_THRESHOLD":cfg.SCALPER_QUALITY_THRESHOLD,
                        "SCALPER_ATR_MIN":cfg.SCALPER_ATR_MIN,
                        "SCALPER_ATR_MAX":cfg.SCALPER_ATR_MAX,
                        "SCALPER_EMA20_SLOPE_MIN":cfg.SCALPER_EMA20_SLOPE_MIN,
                        "SCALPER_BODY_RATIO_MIN":cfg.SCALPER_BODY_RATIO_MIN,
                        "SCALPER_TICK_DIR_THRESHOLD":cfg.SCALPER_TICK_DIR_THRESHOLD,
                        "SCALPER_SWEEP_LOOKBACK":cfg.SCALPER_SWEEP_LOOKBACK,
                        "SCALPER_SWEEP_TOLERANCE":cfg.SCALPER_SWEEP_TOLERANCE,
                        "SCALPER_MAX_TRADES_SESSION":cfg.SCALPER_MAX_TRADES_SESSION,
                        "SCALPER_LEVEL_COOLDOWN":cfg.SCALPER_LEVEL_COOLDOWN,
                        "MARKET_TICK_POLL_INTERVAL":cfg.MARKET_TICK_POLL_INTERVAL,
                        "DASHBOARD_WS_PUSH_INTERVAL":cfg.DASHBOARD_WS_PUSH_INTERVAL,
                    }})
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
                elif p in ("/","/dashboard"): s._h(dashboard_html)
                else: s._j({"error":"Not found"},404)

            def do_POST(s):
                p = s.path
                if p == "/start":
                    trader.enabled = True; trader.log("API","Trading ENABLED"); s._j({"enabled":True})
                elif p == "/stop":
                    trader.enabled = False; trader.log("API","Trading DISABLED"); s._j({"enabled":False})
                elif p == "/emergency":
                    trader.enabled = False
                    if trader.trades: trader.trades.close_all()
                    trader.log("API","EMERGENCY STOP"); s._j({"enabled":False})
                elif p.startswith("/strategy/"):
                    name = p.split("/strategy/")[1].upper()
                    if trader.strat_mgr.set_active(name):
                        trader.log("API", f"Strategy -> {name}")
                        s._j({"active":name,"available":trader.strat_mgr.available})
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
                                for tf in ["M1", "M5", "M15", "H1", "H4"]:
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
                    _RISK_KEYS = {'MAX_POSITIONS':int,'MAX_RISK_PCT':float,'MAX_DRAWDOWN_PCT':float,
                        'MAX_LOT':float,'MIN_LOT':float,'DAILY_TARGET_DOLLARS':float,
                        'DAILY_LOSS_LIMIT_PCT':float,'MAX_CONSECUTIVE_LOSSES':int,
                        'MIN_TRADE_COOLDOWN':float,'LOSS_STREAK_PAUSE':int,
                        'SESSION_MAX_TRADES':int}
                    _GATE_FLOAT_KEYS = {'SMC_SPREAD_MEAN_MAX','SMC_SPREAD_STD_MAX','SMC_SPREAD_PERCENTILE_MAX',
                        'SMC_CURRENT_SPREAD_DELTA_MAX','SCALPER_SPREAD_MEAN_MAX','SCALPER_SPREAD_STD_MAX',
                        'SCALPER_SPREAD_PERCENTILE_MAX','SCALPER_CURRENT_SPREAD_DELTA_MAX',
                        'COMPRESSION_ATR_MULTIPLIER','SMC_THRESHOLD_WITH_TREND','SMC_THRESHOLD_COUNTER',
                        'SMC_THRESHOLD_COUNTER_MAX','SMC_MIN_RR','SMC_TIMEFRAME_EMA_SLOPE_MIN',
                        'SMC_LTF_TICK_CONFLICT_LONG_MAX','SMC_LTF_TICK_CONFLICT_SHORT_MIN',
                        'SCALPER_QUALITY_THRESHOLD','SCALPER_ATR_MIN','SCALPER_ATR_MAX',
                        'SCALPER_EMA20_SLOPE_MIN','SCALPER_BODY_RATIO_MIN',
                        'SCALPER_TICK_DIR_THRESHOLD','SCALPER_SWEEP_TOLERANCE',
                        'MARKET_TICK_POLL_INTERVAL','DASHBOARD_WS_PUSH_INTERVAL',
                        'SMC_RANGING_THRESHOLD_OFFSET','SMC_TRENDING_THRESHOLD_OFFSET',
                        'SMC_EARLY_FAIL_POINTS','SMC_EARLY_FAIL_RANGING','SMC_EARLY_FAIL_TRENDING',
                        'SMC_HIGH_CONF_THRESHOLD','SMC_HIGH_CONF_RR_MULTIPLIER','SMC_HIGH_CONF_RISK_MULTIPLIER',
                        'ADAPT_THRESHOLD_WTR_MIN','ADAPT_THRESHOLD_WTR_MAX',
                        'ADAPT_THRESHOLD_CTR_MIN','ADAPT_THRESHOLD_CTR_MAX',
                        'ADAPT_MIN_RR_MIN','ADAPT_MIN_RR_MAX',
                        'SMC_EARLY_FAIL_CONF_HIGH_MULT','SMC_EARLY_FAIL_CONF_LOW_MULT',
                        'SMC_TIER1_SPREAD_DELTA_MAX'}
                    _GATE_INT_KEYS = {'COMPRESSION_RANGE_LOOKBACK','SCALPER_SWEEP_LOOKBACK',
                        'SCALPER_MAX_TRADES_SESSION','SCALPER_LEVEL_COOLDOWN',
                        'ADAPT_SESSION_TRADES_MIN','ADAPT_SESSION_TRADES_MAX',
                        'ADAPT_COOLDOWN_MIN','ADAPT_COOLDOWN_MAX',
                        'SMC_TIER1_MIN_TICKS','SMC_TIER1_MAX_TICKS'}
                    _GATE_BOOL_KEYS = {'ALL_GATES_OVERRIDE_ENABLED','SPREAD_GATE_OVERRIDE_ENABLED',
                        'COMPRESSION_GATE_OVERRIDE_ENABLED','EXECUTION_GATE_OVERRIDE_ENABLED',
                        'TIME_GATE_OVERRIDE_ENABLED','SPREAD_MEAN_GATE_OVERRIDE_ENABLED',
                        'SPREAD_VOLATILITY_GATE_OVERRIDE_ENABLED','SPREAD_PERCENTILE_GATE_OVERRIDE_ENABLED',
                        'SPREAD_DELTA_GATE_OVERRIDE_ENABLED','COMPRESSION_RANGE_GATE_OVERRIDE_ENABLED',
                        'ATR_RISING_GATE_OVERRIDE_ENABLED','EXECUTION_QUALITY_GATE_OVERRIDE_ENABLED',
                        'TICK_DIRECTION_GATE_OVERRIDE_ENABLED','POST_SIGNAL_SPREAD_GATE_OVERRIDE_ENABLED'}
                    updated = {}
                    for k,v in body.items():
                        if k in _RISK_KEYS:
                            val = _RISK_KEYS[k](v)
                            setattr(cfg, k, val)
                            updated[k] = val
                        elif k in _GATE_FLOAT_KEYS:
                            val = max(0.0, float(v))
                            if k.endswith('_PERCENTILE_MAX'):
                                val = min(1.0, val)
                            setattr(cfg, k, val)
                            updated[k] = val
                        elif k in _GATE_INT_KEYS:
                            val = max(1, int(v))
                            setattr(cfg, k, val)
                            updated[k] = val
                        elif k in _GATE_BOOL_KEYS:
                            val = bool(v)
                            setattr(cfg, k, val)
                            updated[k] = val
                    if 'TIER1_ENABLED' in body:
                        cfg.TIER1_ENABLED = bool(body['TIER1_ENABLED'])
                        updated['TIER1_ENABLED'] = cfg.TIER1_ENABLED
                    if 'SESSION_OVERRIDE_ENABLED' in body:
                        cfg.SESSION_OVERRIDE_ENABLED = bool(body['SESSION_OVERRIDE_ENABLED'])
                        updated['SESSION_OVERRIDE_ENABLED'] = cfg.SESSION_OVERRIDE_ENABLED
                    if 'DAILY_TARGET_ENABLED' in body:
                        cfg.DAILY_TARGET_ENABLED = bool(body['DAILY_TARGET_ENABLED'])
                        updated['DAILY_TARGET_ENABLED'] = cfg.DAILY_TARGET_ENABLED
                    if updated:
                        try:
                            cfg.save_runtime_config()
                        except Exception as e:
                            trader.log("API", f"Config persistence failed: {e}")
                        # Refresh adaptive engine baseline so new saved values become the base
                        try:
                            adaptive_params._capture_base()
                        except Exception:
                            pass
                        if any(k in updated for k in ('LOSS_STREAK_PAUSE', 'MAX_CONSECUTIVE_LOSSES')):
                            trader.risk._loss_streak_pause_until = 0.0
                            trader.risk._consecutive_losses = 0
                        trader.log("API", f"Config updated: {updated}")
                    s._j({"updated":updated})
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
