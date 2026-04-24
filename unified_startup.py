"""
Unified Trading System Startup
Runs auto_trader engine and FastAPI server together on port 8000.
"""
import threading
import time
import uvicorn
from auto_trader_engine import AutoTrader
from engine.trader_api import set_auto_trader_instance

def start_auto_trader():
    """Start the auto trader engine in a separate thread."""
    trader = AutoTrader()
    
    # Set the trader instance for the API
    set_auto_trader_instance(trader)
    
    # Start the trading engine (without HTTP server)
    trader.start_engine()
    
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
    
    # Start FastAPI server
    print("Starting FastAPI server on http://localhost:8000")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )

if __name__ == "__main__":
    main()