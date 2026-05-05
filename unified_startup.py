"""
Unified Trading System Startup
Runs auto_trader engine and FastAPI server together on port 8000.
"""
import threading
import time
import subprocess
import os
import sys
import uvicorn
from auto_trader import AutoTrader
from engine.trader_api import set_auto_trader_instance

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_CF_EXE = os.path.join(_BASE_DIR, "cloudflared.exe")
_CF_CONFIG = os.path.join(_BASE_DIR, "cloudflared-config.yml")

def start_auto_trader():
    """Start the auto trader engine in a separate thread."""
    trader = AutoTrader()
    
    # Start the trading engine (without HTTP server)
    trader.log("INIT", f"Starting unified trading system...")
    
    if trader.bridge.connect():
        trader._mt5_connected = True
        acct = trader.bridge.get_account()
        if acct:
            trader.risk.set_start_balance(acct.get("balance", 0))
        trader.risk.set_risk_multiplier(trader.perf.risk_multiplier())
        
        # Initial MT5 history poll
        trader._poll_mt5_history()
        today_pnl = trader._mt5_today_pnl
        if today_pnl.get("pnl", 0) != 0 or today_pnl.get("trades", 0) > 0:
            trader.log("INIT", f"Today MT5 P&L: ${today_pnl['pnl']:+.2f} ({today_pnl['trades']} trades, {today_pnl['wins']}W/{today_pnl['losses']}L) [{today_pnl.get('source','')}]")
        
        from engine.trade_manager import TradeManager
        from engine.correlation import correlation as corr_engine
        trader.trades = TradeManager(trader.bridge)
        corr_engine.init_symbols()

        # XGBoost: train synchronously so dashboard shows trained state immediately
        from engine.xgb_model import xgb_model, xgb_bypass_enabled, xgb_training_enabled
        perf_trades = trader.perf.trades
        featured = [t for t in perf_trades if t.get("features") and
                    any(isinstance(v, (int, float)) and v != 0
                        for v in t["features"].values())]
        if xgb_bypass_enabled():
            trader.log("INIT", "XGBoost bypass enabled")
        elif not xgb_training_enabled():
            trader.log("INIT", "XGBoost training disabled")
        elif not xgb_model.is_trained and len(featured) >= 15:
            xgb_model.train(featured)
            trader.log("INIT", f"XGBoost trained on {len(featured)} trades")
        elif xgb_model.is_trained and xgb_model.should_retrain(len(featured)):
            import threading
            threading.Thread(target=xgb_model.train, args=(featured,), daemon=True).start()
            trader.log("INIT", f"XGBoost retraining on {len(featured)} trades")
        elif xgb_model.is_trained:
            trader.log("INIT", f"XGBoost loaded ({xgb_model._trades_at_last_train} trades)")
        else:
            trader.log("INIT", f"XGBoost: {len(featured)} featured trades < 15, skipping")

        # Load candle data
        trader.log("INIT", "Loading candle data...")
        for tf in ["M1", "M5", "M15", "M30", "H1", "H4", "D1"]:
            df = trader.bridge.fetch_candles(tf)
            if not df.empty:
                trader._candles[tf] = df
                trader._candle_counts[tf] = len(df)

        trader._recompute_all()
        loaded = ", ".join(f"{tf}:{len(df)}" for tf, df in trader._candles.items())
        trader.log("INIT", f"MT5 connected. Data: {loaded}")
        trader.log("INIT", f"Strategies: {', '.join(trader.strat_mgr.available)} | Active: {trader.strat_mgr.active_name}")
    else:
        trader.log("WARN", "MT5 not connected. Will retry.")
        from engine.trade_manager import TradeManager
        trader.trades = TradeManager(trader.bridge)
    
    trader._running = True
    if hasattr(trader, "mark_restart_time"):
        trader.mark_restart_time()
    trader.log("ENGINE", "Trading engine started")
    
    # Set the trader instance for the API AFTER initialization
    set_auto_trader_instance(trader)
    trader.log("API", "Trader instance connected to API layer")
    
    # Start the engine loop
    trader._engine_loop()

def main():
    """Main startup function."""
    print("Starting Unified Trading System...")
    
    # Start auto trader in background thread
    trader_thread = threading.Thread(target=start_auto_trader, daemon=True, name="AutoTrader")
    trader_thread.start()
    
    # Give the trader a moment to initialize
    time.sleep(2)

    # Start Cloudflare tunnel
    if os.path.exists(_CF_EXE) and os.path.exists(_CF_CONFIG):
        result = subprocess.run("tasklist /FI \"IMAGENAME eq cloudflared.exe\" /NH",
                                shell=True, capture_output=True, text=True)
        if "cloudflared.exe" in result.stdout:
            print("Cloudflare tunnel already running, skipping restart.")
        else:
            print("Starting Cloudflare tunnel (trader.healthymealspot.com)...")
            subprocess.Popen(
                [_CF_EXE, "tunnel", "--config", _CF_CONFIG, "run"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=0x08000000,  # CREATE_NO_WINDOW
            )
    else:
        print("Cloudflare tunnel skipped (cloudflared.exe or config not found)")

    # Start FastAPI server
    print("Starting FastAPI server on http://localhost:8000")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
        ws_ping_interval=None,
        ws_ping_timeout=None,
    )

if __name__ == "__main__":
    main()
