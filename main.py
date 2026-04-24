from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from engine.analytics_api import router as analytics_router
from engine.order_api import router as order_router
from engine.trader_api import router as trader_router

app = FastAPI(title="Trading System API", version="1.0.0")
_BASE_DIR = Path(__file__).resolve().parent
_ANALYTICS_DIR = _BASE_DIR / "analytics_outputs"
_ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)

# Include all API routes
app.include_router(order_router)
app.include_router(analytics_router)
app.include_router(trader_router)
app.mount("/analytics-assets", StaticFiles(directory=str(_ANALYTICS_DIR)), name="analytics-assets")


@app.get("/")
async def root():
    return {
        "message": "Trading System API",
        "version": "1.0.0",
        "analytics_dashboard": "/analytics",
        "trading_dashboard": "/dashboard",
        "status": "/status",
        "health": "/health",
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "trading_system"}