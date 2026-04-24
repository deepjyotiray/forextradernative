"""
Auto Trader API routes for FastAPI integration.
Consolidates all trading system endpoints onto a single port.
"""
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from typing import Dict, Any
import json
import os
import numpy as np
from pathlib import Path

# This will be set by the main application when the auto_trader instance is available
_auto_trader_instance = None

router = APIRouter(tags=["trading"])

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def set_auto_trader_instance(instance):
    """Set the auto trader instance for the API to use."""
    global _auto_trader_instance
    _auto_trader_instance = instance

def get_auto_trader():
    """Get the auto trader instance, raise error if not available."""
    if _auto_trader_instance is None:
        raise HTTPException(status_code=503, detail="Auto trader not initialized")
    return _auto_trader_instance

@router.get("/status")
async def get_status():
    """Get full trading system status."""
    trader = get_auto_trader()
    import config as cfg
    from engine.xgb_model import xgb_model
    
    status = trader.get_full_status()
    
    # Add missing data that dashboard expects
    status.update({
        "tier1_enabled": cfg.TIER1_ENABLED,
        "session_override_enabled": cfg.SESSION_OVERRIDE_ENABLED,
        "daily_target_enabled": cfg.DAILY_TARGET_ENABLED,
        "daily_target": cfg.DAILY_TARGET_DOLLARS,
        "available_symbols": cfg.AVAILABLE_SYMBOLS,
        "symbol": trader.bridge.current_symbol if trader.bridge else cfg.SYMBOL,
        # Add strategy data for UI
        "strategies": trader.strat_mgr.status() if trader.strat_mgr else {"available": ["AUTO"], "active": "AUTO"},
        "strategy": trader.strat_mgr.active_name if trader.strat_mgr else "AUTO",
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
            "LOSS_STREAK_PAUSE": cfg.LOSS_STREAK_PAUSE
        }
    })
    
    # Add MT5 data if available
    if trader.bridge:
        try:
            status["mt5_positions"] = trader.bridge.get_my_positions() or []
            all_positions = trader.bridge.get_positions() or []
            magic_positions = [p for p in all_positions if p.get("magic") == cfg.MAGIC_NUMBER]
            status["mt5_floating_pnl"] = {
                "total": sum(p.get("net_profit", 0) for p in magic_positions),
                "count": len(magic_positions)
            }
        except Exception:
            status["mt5_positions"] = []
            status["mt5_floating_pnl"] = {"total": 0, "count": 0}
    
    # Add closed history and today's P&L
    status["mt5_today_pnl"] = getattr(trader, '_mt5_today_pnl', {})
    status["closed_history"] = getattr(trader, '_mt5_closed_history', [])
    
    # Add performance data
    if trader.perf:
        try:
            status["performance"] = trader.perf.get_stats()
        except Exception:
            status["performance"] = {}
    else:
        status["performance"] = {}
    
    # Add XGBoost data with live predictions
    xgb_info = xgb_model.get_feature_importance()
    status["xgb"] = xgb_info
    
    # Add live XGBoost predictions if model is trained
    if xgb_model.is_trained:
        try:
            # Create a dummy signal for prediction
            dummy_signal = {"signal": "BUY", "volume": 0.01, "sl_distance": 0.5}
            indicators = status.get("indicators", {})
            regime = status.get("regime", {})
            bias = status.get("bias", {})
            tick = status.get("tick", {})
            
            # Get predictions for both BUY and SELL
            buy_signal = {**dummy_signal, "signal": "BUY"}
            sell_signal = {**dummy_signal, "signal": "SELL"}
            
            buy_prob = xgb_model.predict_win_prob(buy_signal, indicators, regime, bias, tick)
            sell_prob = xgb_model.predict_win_prob(sell_signal, indicators, regime, bias, tick)
            
            # Determine market sentiment
            if buy_prob > 0.6 and buy_prob > sell_prob:
                sentiment = "BULLISH"
                sentiment_color = "green"
            elif sell_prob > 0.6 and sell_prob > buy_prob:
                sentiment = "BEARISH"
                sentiment_color = "red"
            else:
                sentiment = "NEUTRAL"
                sentiment_color = "yellow"
            
            status["xgb_live_prediction"] = {
                "available": True,
                "buy_probability": buy_prob,
                "sell_probability": sell_prob,
                "sentiment": sentiment,
                "sentiment_color": sentiment_color
            }
        except Exception as e:
            status["xgb_live_prediction"] = {
                "available": False,
                "reason": f"Prediction error: {str(e)}"
            }
    else:
        status["xgb_live_prediction"] = {
            "available": False,
            "reason": "Model not trained yet (need 15+ trades with features)"
        }
    
    # Add risk manager data
    if trader.risk:
        status["risk"] = {
            "pnl": getattr(trader.risk, 'daily_pnl', 0),
            "trades": getattr(trader.risk, 'daily_trades', 0),
            "consecutive_losses": getattr(trader.risk, 'consecutive_losses', 0),
            "risk_multiplier": getattr(trader.risk, 'risk_multiplier', 1.0),
            "target_hit": getattr(trader.risk, 'target_hit', False),
            "loss_limit_hit": getattr(trader.risk, 'loss_limit_hit', False)
        }
    else:
        status["risk"] = {
            "pnl": 0, "trades": 0, "consecutive_losses": 0,
            "risk_multiplier": 1.0, "target_hit": False, "loss_limit_hit": False
        }
    
    # Add live blockers from decision logger
    from engine.decision_logger import get_live_blockers
    try:
        status["live_blockers"] = get_live_blockers(50)  # Get last 50 decisions
    except Exception:
        status["live_blockers"] = {
            "latest_by_strategy": {},
            "top_reasons": [],
            "recent_skipped_count": 0
        }
    
    # Convert numpy types to JSON-serializable types
    return convert_numpy_types(status)

@router.get("/logs")
async def get_logs():
    """Get recent trading logs."""
    trader = get_auto_trader()
    return {"logs": list(trader._log)[-200:]}

@router.get("/trades")
async def get_trades():
    """Get current trades status."""
    trader = get_auto_trader()
    trades_status = trader.trades.status if trader.trades else {}
    return convert_numpy_types(trades_status)

@router.get("/performance")
async def get_performance():
    """Get performance statistics."""
    trader = get_auto_trader()
    perf_stats = trader.perf.get_stats()
    return convert_numpy_types(perf_stats)

@router.get("/config")
async def get_config():
    """Get current configuration."""
    trader = get_auto_trader()
    import config as cfg
    return {
        "symbol": cfg.SYMBOL,
        "strategy": trader.strat_mgr.active_name,
        "strategies": trader.strat_mgr.status(),
        "risk_pct": cfg.MAX_RISK_PCT,
        "daily_target": cfg.DAILY_TARGET_DOLLARS,
        "tier1_enabled": cfg.TIER1_ENABLED,
        "session_override_enabled": cfg.SESSION_OVERRIDE_ENABLED,
        "daily_target_enabled": cfg.DAILY_TARGET_ENABLED,
        # Risk configuration parameters - nested under risk_config for dashboard
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
            "LOSS_STREAK_PAUSE": cfg.LOSS_STREAK_PAUSE
        }
    }

@router.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard():
    """Get trading dashboard HTML."""
    base_dir = Path(__file__).resolve().parent.parent
    dash_path = base_dir / "dashboard.html"
    if dash_path.exists():
        return HTMLResponse(dash_path.read_text(encoding="utf-8"))
    else:
        raise HTTPException(status_code=404, detail="Dashboard not found")

@router.post("/start")
async def start_trading():
    """Enable trading."""
    trader = get_auto_trader()
    trader.enabled = True
    trader.log("API", "Trading ENABLED")
    return {"enabled": True}

@router.post("/stop")
async def stop_trading():
    """Disable trading."""
    trader = get_auto_trader()
    trader.enabled = False
    trader.log("API", "Trading DISABLED")
    return {"enabled": False}

@router.post("/emergency")
async def emergency_stop():
    """Emergency stop - disable trading and close all positions."""
    trader = get_auto_trader()
    trader.enabled = False
    if trader.trades:
        trader.trades.close_all()
    trader.log("API", "EMERGENCY STOP")
    return {"enabled": False}

@router.post("/strategy/{strategy_name}")
async def set_strategy(strategy_name: str):
    """Set active trading strategy."""
    trader = get_auto_trader()
    name = strategy_name.upper()
    if trader.strat_mgr.set_active(name):
        trader.log("API", f"Strategy -> {name}")
        return {"active": name, "available": trader.strat_mgr.available}
    else:
        raise HTTPException(
            status_code=400, 
            detail={"error": f"Unknown: {name}", "available": trader.strat_mgr.available}
        )

@router.post("/symbol/{symbol}")
async def set_symbol(symbol: str):
    """Set trading symbol."""
    trader = get_auto_trader()
    import config as cfg
    
    symbol = symbol.upper()
    if symbol not in cfg.AVAILABLE_SYMBOLS:
        raise HTTPException(
            status_code=400,
            detail={"error": f"Unknown symbol: {symbol}", "available": cfg.AVAILABLE_SYMBOLS}
        )
    
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
        return {"symbol": symbol, "available": cfg.AVAILABLE_SYMBOLS}
    else:
        raise HTTPException(status_code=500, detail=f"Failed to set {symbol}")

@router.post("/config")
async def update_config(request: Request):
    """Update trading configuration."""
    trader = get_auto_trader()
    import config as cfg
    
    body = await request.json()
    
    _RISK_KEYS = {
        'MAX_POSITIONS': int, 'MAX_RISK_PCT': float, 'MAX_DRAWDOWN_PCT': float,
        'MAX_LOT': float, 'MIN_LOT': float, 'DAILY_TARGET_DOLLARS': float,
        'DAILY_LOSS_LIMIT_PCT': float, 'MAX_CONSECUTIVE_LOSSES': int,
        'MIN_TRADE_COOLDOWN': float, 'LOSS_STREAK_PAUSE': int
    }
    
    updated = {}
    for k, v in body.items():
        if k in _RISK_KEYS:
            val = _RISK_KEYS[k](v)
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
        if any(k in updated for k in ('LOSS_STREAK_PAUSE', 'MAX_CONSECUTIVE_LOSSES')):
            trader.risk._loss_streak_pause_until = 0.0
            trader.risk._consecutive_losses = 0
        trader.log("API", f"Config updated: {updated}")
    
    return {"updated": updated}

@router.post("/shutdown")
async def shutdown():
    """Shutdown the trading system."""
    trader = get_auto_trader()
    trader.enabled = False
    trader._running = False
    if trader.trades:
        trader.trades.close_all()
    trader.log("API", "SHUTDOWN")
    
    # Schedule shutdown
    import threading
    import time
    import os
    threading.Thread(target=lambda: (time.sleep(1), os._exit(0)), daemon=True).start()
    
    return {"message": "Shutting down"}

@router.post("/reload")
async def hot_reload():
    """Hot reload the trading system - restart background service."""
    trader = get_auto_trader()
    trader.log("API", "HOT RELOAD initiated")
    
    # Stop current trading
    trader.enabled = False
    
    # Close any open positions (optional - comment out if you want to keep positions)
    # if trader.trades:
    #     trader.trades.close_all()
    
    # Restart the engine in a separate thread
    import threading
    import time
    
    def restart_engine():
        time.sleep(0.5)  # Brief pause
        trader.log("API", "Restarting engine...")
        
        # Reinitialize components
        try:
            # Reconnect MT5
            if trader.bridge:
                trader.bridge.connect()
            
            # Reload strategies
            if trader.strat_mgr:
                # Re-register strategies (they will reload their modules)
                from engine.smc_strategy import SMCStrategy
                from engine.sweep_scalper import SweepScalper
                trader.strat_mgr._strategies.clear()
                trader.strat_mgr.register(SMCStrategy())
                trader.strat_mgr.register(SweepScalper())
                
                # Restore active strategy
                import config as cfg
                if cfg.DEFAULT_STRATEGY in trader.strat_mgr.available:
                    trader.strat_mgr.set_active(cfg.DEFAULT_STRATEGY)
            
            # Clear caches to force fresh data
            trader._candles.clear()
            trader._candle_counts.clear()
            trader._indicators.clear()
            trader._zones.clear()
            trader._regime.clear()
            trader._bias.clear()
            trader._liquidity.clear()
            trader._last_tick = {}
            
            # Reload candle data
            for tf in ["M1", "M5", "M15", "H1", "H4"]:
                try:
                    df = trader.bridge.fetch_candles(tf)
                    if not df.empty:
                        trader._candles[tf] = df
                        trader._candle_counts[tf] = len(df)
                except Exception as e:
                    trader.log("RELOAD", f"Failed to load {tf} data: {e}")
            
            # Recompute all indicators
            trader._recompute_all()
            
            trader.log("API", "HOT RELOAD completed - engine restarted")
            
        except Exception as e:
            trader.log("ERROR", f"Hot reload failed: {e}")
    
    threading.Thread(target=restart_engine, daemon=True).start()
    
    return {"message": "Hot reload initiated - engine restarting"}