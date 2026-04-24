"""
Auto Trader API routes for FastAPI integration.
Consolidates all trading system endpoints onto a single port.
"""
from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.responses import Response as _RawResponse
from typing import Dict, Any, Set
import json
import os
import time
import asyncio
import numpy as np
from pathlib import Path

try:
    import orjson
    def _fast_json(obj):
        return orjson.dumps(obj, option=orjson.OPT_SERIALIZE_NUMPY).decode()
except ImportError:
    import json as _json
    class _NumpyEncoder(_json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.bool_, np.integer, np.floating)):
                return o.item()
            if isinstance(o, np.ndarray):
                return o.tolist()
            return super().default(o)
    def _fast_json(obj):
        return _json.dumps(obj, cls=_NumpyEncoder, separators=(',', ':'))

# --- WebSocket broadcast infrastructure ---
_ws_clients: Set[WebSocket] = set()
_ws_latest_tick: str = '{}'
_ws_latest_cycle: int = -1

# This will be set by the main application when the auto_trader instance is available
_auto_trader_instance = None
_status_cache = None
_status_cache_time = 0.0
_STATUS_CACHE_TTL = 4.0  # seconds — full status is heavy, cache aggressively

# Separate caches for slow-changing data
_blockers_cache = None
_blockers_cache_time = 0.0
_BLOCKERS_CACHE_TTL = 10.0

_xgb_cache = None
_xgb_cache_time = 0.0
_XGB_CACHE_TTL = 15.0

_perf_cache = None
_perf_cache_time = 0.0
_PERF_CACHE_TTL = 10.0

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

def _get_cached_blockers() -> dict:
    """Get live blockers with caching (reads JSONL file — expensive)."""
    global _blockers_cache, _blockers_cache_time
    now = time.monotonic()
    if _blockers_cache is not None and (now - _blockers_cache_time) < _BLOCKERS_CACHE_TTL:
        return _blockers_cache
    from engine.decision_logger import get_live_blockers
    try:
        _blockers_cache = get_live_blockers(50)
    except Exception:
        _blockers_cache = {"latest_by_strategy": {}, "top_reasons": [], "recent_skipped_count": 0}
    _blockers_cache_time = now
    return _blockers_cache


def _get_cached_xgb(indicators: dict, regime: dict, bias: dict, tick: dict) -> tuple:
    """Get XGB predictions with caching (model inference — expensive)."""
    global _xgb_cache, _xgb_cache_time
    now = time.monotonic()
    if _xgb_cache is not None and (now - _xgb_cache_time) < _XGB_CACHE_TTL:
        return _xgb_cache
    from engine.xgb_model import xgb_model
    xgb_info = xgb_model.get_feature_importance()
    xgb_pred = {"available": False, "reason": "Model not trained yet (need 15+ trades with features)"}
    if xgb_model.is_trained:
        try:
            dummy = {"volume": 0.01, "sl_distance": 0.5}
            buy_prob = xgb_model.predict_win_prob({**dummy, "signal": "BUY"}, indicators, regime, bias, tick)
            sell_prob = xgb_model.predict_win_prob({**dummy, "signal": "SELL"}, indicators, regime, bias, tick)
            if buy_prob > 0.6 and buy_prob > sell_prob:
                sentiment, sentiment_color = "BULLISH", "green"
            elif sell_prob > 0.6 and sell_prob > buy_prob:
                sentiment, sentiment_color = "BEARISH", "red"
            else:
                sentiment, sentiment_color = "NEUTRAL", "yellow"
            xgb_pred = {"available": True, "buy_probability": buy_prob, "sell_probability": sell_prob,
                        "sentiment": sentiment, "sentiment_color": sentiment_color}
        except Exception as e:
            xgb_pred = {"available": False, "reason": f"Prediction error: {str(e)}"}
    _xgb_cache = (xgb_info, xgb_pred)
    _xgb_cache_time = now
    return _xgb_cache


def _get_cached_perf(trader) -> dict:
    """Get performance stats with caching."""
    global _perf_cache, _perf_cache_time
    now = time.monotonic()
    if _perf_cache is not None and (now - _perf_cache_time) < _PERF_CACHE_TTL:
        return _perf_cache
    if trader.perf:
        try:
            _perf_cache = trader.perf.get_stats()
        except Exception:
            _perf_cache = {}
    else:
        _perf_cache = {}
    _perf_cache_time = now
    return _perf_cache


def _build_risk_dict(trader) -> dict:
    if trader.risk:
        return {
            "pnl": getattr(trader.risk, 'daily_pnl', 0),
            "trades": getattr(trader.risk, 'daily_trades', 0),
            "consecutive_losses": getattr(trader.risk, 'consecutive_losses', 0),
            "risk_multiplier": getattr(trader.risk, 'risk_multiplier', 1.0),
            "target_hit": getattr(trader.risk, 'target_hit', False),
            "loss_limit_hit": getattr(trader.risk, 'loss_limit_hit', False),
        }
    return {"pnl": 0, "trades": 0, "consecutive_losses": 0,
            "risk_multiplier": 1.0, "target_hit": False, "loss_limit_hit": False}


def _build_config_dict() -> dict:
    import config as cfg
    return {
        "MAX_POSITIONS": cfg.MAX_POSITIONS, "MAX_RISK_PCT": cfg.MAX_RISK_PCT,
        "MAX_DRAWDOWN_PCT": cfg.MAX_DRAWDOWN_PCT, "MAX_LOT": cfg.MAX_LOT,
        "MIN_LOT": cfg.MIN_LOT, "DAILY_TARGET_DOLLARS": cfg.DAILY_TARGET_DOLLARS,
        "DAILY_LOSS_LIMIT_PCT": cfg.DAILY_LOSS_LIMIT_PCT,
        "MAX_CONSECUTIVE_LOSSES": cfg.MAX_CONSECUTIVE_LOSSES,
        "MIN_TRADE_COOLDOWN": cfg.MIN_TRADE_COOLDOWN,
        "LOSS_STREAK_PAUSE": cfg.LOSS_STREAK_PAUSE,
    }


def _build_tick_data(trader) -> dict:
    """Build the tick payload dict from trader state. Pure memory reads."""
    import config as cfg
    from engine.session_filter import get_session, is_market_open
    return {
        "enabled": trader.enabled,
        "mt5_connected": trader._mt5_connected,
        "tick": trader._last_tick,
        "account": trader._last_account,
        "session": get_session(),
        "market_open": is_market_open(),
        "symbol": trader.bridge.current_symbol if trader.bridge else cfg.SYMBOL,
        "strategy": trader.strat_mgr.active_name if trader.strat_mgr else "AUTO",
        "mt5_positions": getattr(trader, '_cached_positions', None) or [],
        "mt5_floating_pnl": getattr(trader, '_cached_floating_pnl', None) or {"total": 0, "count": 0},
        "mt5_today_pnl": getattr(trader, '_mt5_today_pnl', None) or {},
        "risk": _build_risk_dict(trader),
        "indicators": trader._indicators or {},
        "log": list(trader._log)[-50:],
        "stats": trader._stats,
    }


def _refresh_tick_cache(trader) -> str:
    """Serialize tick data once per engine cycle. Returns cached JSON string."""
    global _ws_latest_tick, _ws_latest_cycle
    cycle = trader._stats.get("cycles", 0)
    if cycle == _ws_latest_cycle:
        return _ws_latest_tick
    _ws_latest_tick = _fast_json(_build_tick_data(trader))
    _ws_latest_cycle = cycle
    return _ws_latest_tick


# --- WebSocket: server pushes tick data every engine cycle ---
@router.websocket("/ws")
async def ws_tick(ws: WebSocket):
    await ws.accept()
    _ws_clients.add(ws)

    async def _reader():
        """Drain incoming frames so we detect disconnect."""
        try:
            while True:
                await ws.receive_text()
        except (WebSocketDisconnect, Exception):
            pass

    reader_task = asyncio.create_task(_reader())
    last_pushed_cycle = -1
    try:
        while not reader_task.done():
            trader = _auto_trader_instance
            if trader is not None:
                cycle = trader._stats.get("cycles", 0)
                if cycle != last_pushed_cycle:
                    msg = _refresh_tick_cache(trader)
                    await ws.send_text(msg)
                    last_pushed_cycle = cycle
            await asyncio.sleep(0.15)  # ~6-7 pushes/sec, only sends on new cycle
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        reader_task.cancel()
        _ws_clients.discard(ws)


# Keep /tick as HTTP fallback (e.g. curl, other clients)
@router.get("/tick")
async def get_tick():
    """HTTP fallback for tick data. Prefer /ws WebSocket for real-time."""
    trader = get_auto_trader()
    return _RawResponse(content=_refresh_tick_cache(trader), media_type="application/json")


def _build_full_status_sync() -> dict:
    """Build full status dict — runs in thread pool to avoid blocking event loop."""
    trader = get_auto_trader()
    import config as cfg

    status = trader.get_full_status()
    status.update({
        "tier1_enabled": cfg.TIER1_ENABLED,
        "session_override_enabled": cfg.SESSION_OVERRIDE_ENABLED,
        "daily_target_enabled": cfg.DAILY_TARGET_ENABLED,
        "daily_target": cfg.DAILY_TARGET_DOLLARS,
        "available_symbols": cfg.AVAILABLE_SYMBOLS,
        "symbol": trader.bridge.current_symbol if trader.bridge else cfg.SYMBOL,
        "strategies": trader.strat_mgr.status() if trader.strat_mgr else {"available": ["AUTO"], "active": "AUTO"},
        "strategy": trader.strat_mgr.active_name if trader.strat_mgr else "AUTO",
        "risk_config": _build_config_dict(),
    })
    status["mt5_positions"] = getattr(trader, '_cached_positions', []) or []
    status["mt5_floating_pnl"] = getattr(trader, '_cached_floating_pnl', {"total": 0, "count": 0})
    status["mt5_today_pnl"] = getattr(trader, '_mt5_today_pnl', {})
    status["closed_history"] = getattr(trader, '_mt5_closed_history', [])
    status["performance"] = _get_cached_perf(trader)
    status["risk"] = _build_risk_dict(trader)
    xgb_info, xgb_pred = _get_cached_xgb(
        status.get("indicators", {}), status.get("regime", {}),
        status.get("bias", {}), status.get("tick", {}),
    )
    status["xgb"] = xgb_info
    status["xgb_live_prediction"] = xgb_pred
    status["live_blockers"] = _get_cached_blockers()
    return convert_numpy_types(status)


@router.get("/status")
async def get_status():
    """Full status — heavy, runs in thread pool."""
    global _status_cache, _status_cache_time
    now = time.monotonic()
    if _status_cache is not None and (now - _status_cache_time) < _STATUS_CACHE_TTL:
        return _status_cache
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _build_full_status_sync)
    _status_cache = result
    _status_cache_time = time.monotonic()
    return result

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