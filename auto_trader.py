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
from collections import deque
from typing import Dict
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler

import config as cfg
from engine.mt5_bridge import MT5Bridge
from engine.indicators import compute_indicators
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
from engine import csv_reader
from engine.xgb_model import xgb_model

_TF_REFRESH = {"M1": 1, "M5": 2, "M15": 10, "H1": 60, "H4": 120, "D1": 300}
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


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
        self._last_account: Dict = {}

    def log(self, tag: str, msg: str):
        entry = {"time": datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3], "tag": tag, "msg": msg}
        self._log.append(entry)
        try:
            with open(os.path.join(_BASE_DIR, "trader.log"), "a") as f:
                f.write(f"[{entry['time']}][{tag}] {msg}\n")
        except Exception:
            pass

    def run_service(self):
        self.log("INIT", "Starting XAUUSD Auto Trader service...")
        if self.bridge.connect():
            self._mt5_connected = True
            acct = self.bridge.get_account()
            self.risk.set_start_balance(acct.get("balance", 0))
            self.risk.set_risk_multiplier(self.perf.risk_multiplier())
            # Seed daily P&L from MT5 history
            today_pnl = self.bridge.get_today_pnl()
            self.risk.seed_from_mt5(today_pnl)
            if today_pnl["trades"] > 0:
                self.log("INIT", f"Today's MT5 P&L: ${today_pnl['pnl']:+.2f} ({today_pnl['trades']} trades, {today_pnl['wins']}W/{today_pnl['losses']}L)")
            self.trades = TradeManager(self.bridge)
            corr_engine.init_symbols()
            # Init CSV reader for EA-exported files
            import MetaTrader5 as mt5
            ti = mt5.terminal_info()
            if ti:
                csv_reader.set_files_dir(ti.data_path)
                self.log("INIT", f"CSV reader: {ti.data_path}/MQL5/Files")
            # Seed daily P&L
            today_pnl = self._get_today_pnl()
            self.risk.seed_from_mt5(today_pnl)
            if today_pnl.get("pnl", 0) != 0 or today_pnl.get("trades", 0) > 0:
                self.log("INIT", f"Today P&L: ${today_pnl['pnl']:+.2f} ({today_pnl.get('trades',0)} trades) [{today_pnl.get('source','')}]")
            # XGBoost: train from CSV history
            csv_trades = csv_reader.get_closed_trades() if csv_reader.available() else []
            if csv_trades and xgb_model.should_retrain(len(csv_trades)):
                threading.Thread(target=xgb_model.train, args=(csv_trades,), daemon=True).start()
                self.log("INIT", f"XGBoost training on {len(csv_trades)} trades")
            elif xgb_model.is_trained:
                self.log("INIT", f"XGBoost loaded ({xgb_model._trades_at_last_train} trades)")
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
        while self._running:
            t0 = time.time()
            try:
                if not self._mt5_connected:
                    if self.bridge.connect():
                        self._mt5_connected = True
                        corr_engine.init_symbols()
                        self.risk.set_start_balance(self.bridge.get_account().get("balance", 0))
                        self.log("ENGINE", "MT5 reconnected")
                    else:
                        time.sleep(5)
                        continue
                self._cycle_once()
            except Exception as e:
                self.log("ERR", str(e))
                self._stats["errors"] += 1
                if "initialize" in str(e).lower():
                    self._mt5_connected = False
            elapsed = time.time() - t0
            if elapsed < 0.01:
                time.sleep(0.01)

    def _cycle_once(self):
        self._cycle += 1
        self._stats["cycles"] += 1

        tick = self.bridge.get_tick()
        if not tick:
            return
        self._last_tick = tick
        self.tick_proc.feed(tick)

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
                    self._indicators = compute_indicators(df)
                if tf == "M5":
                    zones_dirty = True
                if tf in ("M15", "H1", "H4"):
                    regime_dirty = True

        if zones_dirty:
            m5 = self._candles.get("M5")
            if m5 is not None and len(m5) >= 50:
                self._zones = self.zone_detector.detect(m5)
        if regime_dirty or self._cycle % 30 == 0:
            self._recompute_regime_bias()
        if self._cycle % 60 == 0:
            eco_calendar.poll()
            self._calendar = eco_calendar.check()
        if self._cycle % 300 == 0:
            self._correlation = corr_engine.compute()

        # Manage open trades (ALWAYS) — pass ALL positions for P&L matching
        closed = self.trades.manage_all(all_positions)
        for ticket, pnl, won in closed:
            self.risk.record_trade_result(pnl, won)
            self.perf.record({"ticket": ticket, "pnl": pnl, "won": won,
                              "time": datetime.now(timezone.utc).isoformat(),
                              **({"features": self.trades.closed_trades[-1].get("features", {})} if self.trades.closed_trades else {})})
            self.risk.set_risk_multiplier(self.perf.risk_multiplier())
            self.log("RESULT", f"#{ticket} {'WIN' if won else 'LOSS'} ${pnl:+.2f} | Daily: ${self.risk._daily_pnl:+.2f}")
            # XGBoost: retrain periodically — use bot's own trades (have features)
            all_closed = self.trades.closed_trades
            if all_closed and xgb_model.should_retrain(len(all_closed)):
                threading.Thread(target=xgb_model.train, args=(all_closed,), daemon=True).start()

        if not self.enabled:
            return
        if not is_market_open():
            return
        blocked, _ = is_session_open_blocked()
        if blocked:
            return
        if self._calendar.get("blocked"):
            return
        allowed, _ = self.risk.can_trade(account, len(positions))  # count only our positions
        if not allowed:
            return

        # === Strategy signal ===
        strat_data = {
            "m1_df": self._candles.get("M1"), "m5_df": self._candles.get("M5"),
            "m15_df": self._candles.get("M15"), "h1_df": self._candles.get("H1"),
            "h4_df": self._candles.get("H4"), "d1_df": self._candles.get("D1"),
            "tick": tick, "zones": self._zones, "indicators": self._indicators,
            "account": account, "positions": positions,
            "correlation": self._correlation, "calendar": self._calendar,
        }
        sig, strat_name = self.strat_mgr.generate_signal(strat_data)
        action = sig.get("signal", "NO_TRADE")

        if action == "NO_TRADE" and sig.get("score", 0) > 0.3:
            self.log("FILTERED", f"[{strat_name}] {sig.get('reason','')[:120]}")
        if action not in ("BUY", "SELL"):
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
        lot = self.risk.calculate_lot(account, sl_distance)
        if is_scalp:
            lot = min(0.05, lot)
        elif get_session() == "ASIAN":
            lot = round(lot * 0.7, 2)
        lot = max(cfg.MIN_LOT, min(cfg.MAX_LOT, lot))

        # Execute with retry + spread re-check
        signal_spread = tick.get("spread", 0)
        if self.tick_proc.spread_changed(signal_spread, max_delta=0.03):
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
            self.trades.register_trade(
                t, action, lot, fp, sl, tp, sl_distance,
                strategy=strat_name, confidence=sig.get("confidence", 0),
                reason=sig.get("reason", ""),
                scalp=is_scalp,
                be_trigger=sig.get("_be_trigger", 0),
                timeout=sig.get("_timeout", 0),
                features={
                    "atr": self._indicators.get("atr", 0),
                    "atr_ratio": self._indicators.get("atr_ratio", 1),
                    "rsi": self._indicators.get("rsi", 50),
                    "ema_slope": self._indicators.get("ema9_slope", 0),
                    "body_ratio": self._indicators.get("body_ratio", 0),
                    "spread": tick.get("spread", 0),
                    "regime": self._regime.get("state", ""),
                    "bias_conf": self._bias.get("confidence", 0),
                    "session": get_session(),
                },
            )
            self.risk.record_trade_opened()
            self._stats["trades"] += 1
            self.log("TRADE", f"[{strat_name}] {action} {lot} lot @ {fp} | "
                     f"SL:{sl} TP:{tp} RR:{sig.get('rr',0):.1f} | {sig.get('reason','')[:80]}")
        elif result:
            self.log("TRADE", f"FAILED: {result.get('error')}")

        # Periodic status
        now = time.time()
        if now - self._last_log_time > 30:
            self._last_log_time = now
            self._log_status(tick)

    def _recompute_all(self):
        m1 = self._candles.get("M1")
        if m1 is not None and len(m1) >= 21:
            self._indicators = compute_indicators(m1)
        m5 = self._candles.get("M5")
        if m5 is not None and len(m5) >= 50:
            self._zones = self.zone_detector.detect(m5)
        self._recompute_regime_bias()

    def _get_today_pnl(self) -> Dict:
        """Best source for today's P&L: CSV deals > MT5 API > balance delta."""
        if csv_reader.available():
            return csv_reader.get_today_pnl()
        return self.bridge.get_today_pnl()

    def _recompute_regime_bias(self):
        ts = self.tick_proc.snapshot()
        self._regime = classify_regime(self._candles.get("H4"), self._candles.get("H1"), self._candles.get("M15"), ts)
        self._bias = compute_bias(self._candles.get("H4"), self._candles.get("H1"), self._candles.get("M15"))
        self._liquidity = compute_liquidity(self._candles.get("M5"), self._candles.get("M15"),
                                             self._candles.get("H1"), self._candles.get("D1"))

    def _log_status(self, tick):
        s, p = get_session(), tick["bid"]
        r, b = self._regime.get("state", "?"), self._bias.get("direction", "?")
        self.log("STATUS", f"[{self.strat_mgr.active_name}] {s} | {p:.2f} | {r} | Bias:{b} | "
                 f"Open:{self.trades.open_count} | Daily:${self.risk._daily_pnl:+.2f}")

    def get_full_status(self) -> Dict:
        return {
            "enabled": self.enabled,
            "mt5_connected": self._mt5_connected,
            "strategy": self.strat_mgr.active_name,
            "strategies": self.strat_mgr.status(),
            "symbol": cfg.SYMBOL,
            "session": get_session(),
            "market_open": is_market_open(),
            "tick": self._last_tick,
            "account": self._last_account,
            "indicators": self._indicators,
            "zones": self._zones,
            "regime": self._regime,
            "bias": self._bias,
            "liquidity": {
                "sweeps": len(self._liquidity.get("sweeps", [])),
                "order_blocks": len(self._liquidity.get("order_blocks", [])),
                "fvg": len(self._liquidity.get("fvg", [])),
                "key_levels": self._liquidity.get("key_levels", {}),
            },
            "calendar": self._calendar,
            "correlation": self._correlation,
            "candles": {tf: len(df) for tf, df in self._candles.items()},
            "risk": self.risk.daily_status,
            "daily_target": cfg.DAILY_TARGET_DOLLARS,
            "mt5_today_pnl": self._get_today_pnl() if self._mt5_connected else {},
            "closed_history": csv_reader.get_closed_trades()[-20:] if csv_reader.available() else [],
            "xgb": xgb_model.get_feature_importance(),
            "performance": self.perf.get_stats(),
            "trades": self.trades.status if self.trades else {},
            "stats": self._stats,
            "log": list(self._log)[-50:],
        }

    def _run_http_server(self):
        dash_path = os.path.join(_BASE_DIR, "dashboard.html")
        with open(dash_path, encoding="utf-8") as f:
            dashboard_html = f.read()
        trader = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *_): pass
            def _j(s, d, c=200):
                b = json.dumps(d, default=str).encode()
                s.send_response(c); s.send_header("Content-Type","application/json")
                s.send_header("Access-Control-Allow-Origin","*"); s.end_headers(); s.wfile.write(b)
            def _h(s, c):
                s.send_response(200); s.send_header("Content-Type","text/html; charset=utf-8")
                s.end_headers(); s.wfile.write(c.encode("utf-8"))

            def do_GET(s):
                p = s.path
                if p == "/status": s._j(trader.get_full_status())
                elif p == "/logs": s._j({"logs": list(trader._log)[-200:]})
                elif p == "/trades": s._j(trader.trades.status if trader.trades else {})
                elif p == "/performance": s._j(trader.perf.get_stats())
                elif p == "/config": s._j({"symbol":cfg.SYMBOL,"strategy":trader.strat_mgr.active_name,
                    "strategies":trader.strat_mgr.status(),"risk_pct":cfg.MAX_RISK_PCT,
                    "daily_target":cfg.DAILY_TARGET_DOLLARS})
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
