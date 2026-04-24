"""
Order API - FastAPI endpoints for order database access.
Provides endpoints for viewing order history, stats, and exporting snapshots.
"""
from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Optional
from datetime import datetime
from engine.order_database import OrderDatabase

router = APIRouter(prefix="/orders", tags=["orders"])
order_db = OrderDatabase()


@router.get("/open", response_model=List[Dict])
async def get_open_orders():
    """Get all currently open orders."""
    return order_db.get_open_orders()


@router.get("/closed", response_model=List[Dict])
async def get_closed_orders(days: int = Query(30, ge=1, le=365)):
    """Get closed orders from last N days."""
    return order_db.get_closed_orders(days)


@router.get("/history/{ticket}", response_model=Dict)
async def get_order_details(ticket: int):
    """Get specific order details by ticket."""
    order = order_db.get_order(ticket)
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    return order


@router.get("/stats/today", response_model=Dict)
async def get_today_stats():
    """Get today's trading statistics."""
    stats = order_db.get_today_stats()
    return stats or {}


@router.get("/stats/strategy", response_model=List[Dict])
async def get_strategy_performance(days: int = Query(30, ge=1, le=365)):
    """Get performance breakdown by strategy."""
    return order_db.get_strategy_performance(days)


@router.post("/export", response_model=Dict)
async def export_snapshot(filename: Optional[str] = None):
    """Export complete order database snapshot to JSON file."""
    try:
        filepath = order_db.export_snapshot(filename)
        return {
            "success": True,
            "filepath": filepath,
            "exported_at": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/cleanup")
async def cleanup_old_data(days: int = Query(90, ge=30, le=365)):
    """Remove orders older than specified days."""
    try:
        deleted_count = order_db.cleanup_old_data(days)
        return {
            "success": True,
            "deleted_count": deleted_count,
            "cutoff_days": days
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/daily-pnl", response_model=List[Dict])
async def get_daily_pnl(days: int = Query(30, ge=1, le=365)):
    """Get daily PNL breakdown using IST dates."""
    return order_db.get_daily_pnl(days)


@router.get("/summary", response_model=Dict)
async def get_order_summary():
    """Get comprehensive order summary."""
    today_stats = order_db.get_today_stats() or {}
    strategy_perf = order_db.get_strategy_performance(30)
    open_orders = order_db.get_open_orders()
    
    return {
        "today": today_stats,
        "strategies_30d": strategy_perf,
        "open_orders_count": len(open_orders),
        "open_orders": open_orders[:5],  # Show first 5 open orders
        "summary_generated_at": datetime.now().isoformat()
    }


@router.get("/validate/{ticket}", response_model=Dict)
async def validate_order_pnl(ticket: int):
    """Validate PnL accuracy for specific order."""
    from engine.mt5_bridge import MT5Bridge
    from engine.pnl_validator import PnLValidator
    
    try:
        mt5_bridge = MT5Bridge()
        if not mt5_bridge.connect():
            raise HTTPException(status_code=503, detail="MT5 connection failed")
        
        validator = PnLValidator(mt5_bridge)
        
        # Check if position is open or closed
        order = order_db.get_order(ticket)
        if not order:
            raise HTTPException(status_code=404, detail="Order not found")
        
        if order['status'] == 'OPEN':
            validation = validator.get_accurate_live_pnl(ticket)
        else:
            validation = validator.validate_closed_position_pnl(ticket)
        
        return {
            "ticket": ticket,
            "order_status": order['status'],
            "database_pnl": order.get('live_pnl') or order.get('final_pnl'),
            "validation": validation,
            "validated_at": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))