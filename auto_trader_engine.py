"""
XAUUSD Auto Trader — Background Service (No HTTP Server)
Works with unified_startup.py to consolidate all endpoints on port 8000.
"""
import time
import threading
import os
from collections import deque
from typing import Dict
from datetime import datetime, timezone, timedelta

_IST = timezone(timedelta(hours=5, minutes=30))

import config as cfg
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
from engine.strategies.trend_channel_strategy import TrendChannelStrategy
from engine.strategies.htf_long_strategy import HTFLongStrategy
from engine.strategies.htf_short_strategy import HTFShortStrategy
from engine.swing_engine_strategy import SwingEngineStrategy
from engine.m15_scalp_deep_strategy import M15ScalpDeepStrategy
from engine.m15_zone_scalp_strategy import M15ZoneScalpStrategy
from engine.m15_zone_scalp_inverse_strategy import M15ZoneScalpInverseStrategy
from engine.calendar import calendar as eco_calendar
from engine.correlation import correlation as corr_engine
from engine.xgb_model import xgb_model, xgb_bypass_enabled, xgb_training_enabled
from engine.anti_starvation import record_trade_taken
from engine.decision_logger import get_live_blockers

_TF_REFRESH = {"M1": 1, "M5": 2, "M15": 10, "M30": 20, "H1": 60, "H4": 120, "D1": 720}
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _safe_dict(value):
    return value if isinstance(value, dict) else {}


class AutoTrader:
    def __init__(self):
        self._started_at_utc = datetime.now(timezone.utc)
        self.bridge = MT5Bridge()
        self.zone_detector = ZoneDetector()
        self.risk = RiskManager()
        self.trades: TradeManager = None
        self.tick_proc = TickProcessor()
        self.perf = PerformanceTracker(os.path.join(_BASE_DIR, "trade_history.json"))

        # Strategy manager
        self.strat_mgr = StrategyManager()
        _strategy_classes = [
            ("SMC_CONFLUENCE", SMCStrategy),
            ("TREND_CHANNEL", TrendChannelStrategy),
            ("HTF_LONG", HTFLongStrategy),
            ("HTF_SHORT", HTFShortStrategy),
            ("SWING_ENGINE", SwingEngineStrategy),
            ("M15_SCALP_DEEP", M15ScalpDeepStrategy),
            ("M15_ZONE_SCALP", M15ZoneScalpStrategy),
            ("M15_ZONE_SCALP_INVERSE", M15ZoneScalpInverseStrategy),
        ]
        for label, cls in _strategy_classes:
            try:
                self.strat_mgr.register(cls())
            except Exception as e:
                print(f"[STRATEGY_ERR] Failed to register {label}: {e}")
        startup_strategy = "AUTO"
        if startup_strategy in self.strat_mgr.available:
            self.strat_mgr.set_active(startup_strategy)

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

        # Cached positions for API (updated every engine cycle, avoids MT5 IPC in API)
        self._cached_positions: list = []
        self._cached_floating_pnl: Dict = {"total": 0, "count": 0}

    def mark_restart_time(self, started_at_utc=None):
        if started_at_utc is None:
            started_at_utc = datetime.now(timezone.utc)
        if started_at_utc.tzinfo is None:
            started_at_utc = started_at_utc.replace(tzinfo=timezone.utc)
        self._started_at_utc = started_at_utc.astimezone(timezone.utc)

    def log(self, tag: str, msg: str):
        now = datetime.now(timezone.utc)
        utc_str = now.strftime("%H:%M:%S.%f")[:-3]
        ist_str = now.astimezone(_IST).strftime("%H:%M:%S.%f")[:-3]
        raw_price = (self._last_tick or {}).get("bid")
        price = None
        if isinstance(raw_price, (int, float)):
            price = round(float(raw_price), 2)
        entry = {"time": utc_str, "time_ist": ist_str, "tag": tag, "msg": msg, "price": price}
        self._log.append(entry)
        try:
            with open(os.path.join(_BASE_DIR, "trader.log"), "a") as f:
                price_part = f"[{price:.2f}]" if price is not None else ""
                f.write(f"[{utc_str} UTC | {ist_str} IST][{tag}]{price_part} {msg}\n")
        except Exception:
            pass

    def start_engine(self):
        """Start the trading engine without HTTP server."""
        self.log("INIT", f"Starting {cfg.SYMBOL} Auto Trader engine...")
        
        if self.bridge.connect():
            self._mt5_connected = True
            acct = self.bridge.get_account()
            if acct:
                self.risk.set_start_balance(acct.get("balance", 0))
            self.risk.set_risk_multiplier(self.perf.risk_multiplier())
            
            # Initial MT5 history poll
            self._poll_mt5_history()
            today_pnl = self._mt5_today_pnl
            if today_pnl.get("pnl", 0) != 0 or today_pnl.get("trades", 0) > 0:
                self.log("INIT", f"Today MT5 P&L: ${today_pnl['pnl']:+.2f} ({today_pnl['trades']} trades, {today_pnl['wins']}W/{today_pnl['losses']}L) [{today_pnl.get('source','')}]")
            
            self.trades = TradeManager(self.bridge)
            corr_engine.init_symbols()
            
            # XGBoost training — synchronous on first train, background on retrain
            perf_trades = self.perf.trades
            featured = [t for t in perf_trades if t.get("features") and
                        any(isinstance(v, (int, float)) and v != 0
                            for v in t["features"].values())]
            if xgb_bypass_enabled():
                self.log("INIT", "XGBoost bypass enabled")
            elif not xgb_training_enabled():
                self.log("INIT", "XGBoost training disabled")
            elif not xgb_model.is_trained and len(featured) >= 15:
                xgb_model.train(featured)
                self.log("INIT", f"XGBoost trained on {len(featured)} trades")
            elif xgb_model.is_trained and xgb_model.should_retrain(len(featured)):
                threading.Thread(target=xgb_model.train, args=(featured,), daemon=True).start()
                self.log("INIT", f"XGBoost retraining on {len(featured)} trades")
            elif xgb_model.is_trained:
                self.log("INIT", f"XGBoost loaded ({xgb_model._trades_at_last_train} trades)")
            else:
                self.log("INIT", f"XGBoost: {len(featured)} featured trades < 15, skipping")
            
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
        self.log("ENGINE", "Trading engine started")

    def _engine_loop(self):
        """Main trading engine loop."""
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
            except Exception:
                pass
            time.sleep(max(0.005, float(getattr(cfg, "MARKET_TICK_POLL_INTERVAL", 0.02))))

    def _cycle_once(self):
        """Single trading cycle."""
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

        # Cache positions for the API layer (avoids MT5 IPC on every /tick call)
        self._cached_positions = positions or []
        magic_positions = [p for p in (all_positions or []) if p.get("magic") == cfg.MAGIC_NUMBER]
        self._cached_floating_pnl = {
            "total": round(sum(p.get("net_profit", 0) for p in magic_positions), 2),
            "count": len(magic_positions),
        }

        # Candle refresh logic
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
        
        if self._cycle % 60 == 0:
            eco_calendar.poll()
            self._calendar = eco_calendar.check()
        
        if self._cycle % 300 == 0:
            self._correlation = corr_engine.compute()

        # Feed floating P&L to risk manager for combined target check
        floating = sum(p.get("net_profit", 0) for p in (all_positions or []) if p.get("magic") == cfg.MAGIC_NUMBER)
        self.risk.update_floating_pnl(floating)

        # Risk management and trade management logic continues...
        # (Rest of the _cycle_once method from original auto_trader.py)
        
        # Poll MT5 history every 5s
        now_ts = time.time()
        if now_ts - self._last_history_poll >= 5.0:
            self._last_history_poll = now_ts
            self._poll_mt5_history()

        # Periodic status logging
        if now_ts - self._last_log_time > 30:
            self._last_log_time = now_ts
            self._log_status(tick)

        # Trading gates and signal processing...
        # (Continue with rest of trading logic)

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
            None,
        ))

    def _poll_mt5_history(self):
        """Poll MT5 deal history - SINGLE SOURCE OF TRUTH for closed trades."""
        try:
            self._mt5_today_pnl = self.bridge.get_today_pnl()
            self._mt5_closed_history = self.bridge.get_closed_trades(30)
            # Sync risk manager — pass full list, seed_from_mt5 filters by trading day
            self.risk.seed_from_mt5(self._mt5_today_pnl, self._mt5_closed_history)
        except Exception as e:
            self.log("ERR", f"History poll failed: {e}")

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

    def get_full_status(self) -> Dict:
        """Get complete trading system status."""
        # Implementation continues with all the status gathering logic...
        return {
            "enabled": self.enabled,
            "mt5_connected": self._mt5_connected,
            "strategy": self.strat_mgr.active_name,
            "symbol": self.bridge.current_symbol,
            "session": get_session(),
            "market_open": is_market_open(),
            "tick": self._last_tick,
            "account": self._last_account,
            "indicators": _safe_dict(self._indicators),
            "zones": _safe_dict(self._zones),
            "regime": _safe_dict(self._regime),
            "bias": _safe_dict(self._bias),
            "last_restart_utc": self._started_at_utc.isoformat(),
            "last_restart_ist": self._started_at_utc.astimezone(_IST).strftime("%Y-%m-%d %H:%M:%S IST"),
            "stats": self._stats,
            "log": list(self._log)[-50:],
        }


if __name__ == "__main__":
    # For backward compatibility, still allow direct execution
    trader = AutoTrader()
    trader.start_engine()
    trader._engine_loop()
