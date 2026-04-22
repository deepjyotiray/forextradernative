from fastapi import FastAPI
from engine.order_api import router as order_router

app = FastAPI(title="Trading System API", version="1.0.0")

# Include order management routes
app.include_router(order_router)


@app.get("/")
async def root():
    return {"message": "Trading System API", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "trading_system"}
