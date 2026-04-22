# Order Database System

Complete local storage system for all trading orders with comprehensive tracking and session context.

## Features

- **Complete Order Storage**: All order data stored locally in SQLite database
- **Session Tracking**: Automatic detection of trading sessions (Asian, London, NY, Quiet)
- **Live PnL Updates**: Real-time PnL tracking from MT5 while maintaining local records
- **Management Flags**: Track breakeven, partial closes, trailing stops
- **Performance Analytics**: Strategy performance breakdown and statistics
- **Data Export**: JSON snapshots for backup and analysis
- **API Access**: RESTful endpoints for data access

## Database Schema

The `orders` table stores:
- **Basic Info**: ticket, symbol, direction, volume, prices
- **Strategy Data**: strategy name, confidence, reason, features
- **Timing**: open/close times in UTC and IST, session info
- **PnL Tracking**: live PnL, peak PnL, final PnL, swap, commission
- **Management**: breakeven flags, partial close status, trailing
- **Metadata**: magic number, creation/update timestamps

## Usage

### 1. Basic Order Storage

```python
from engine.order_database import OrderDatabase
from engine.session_detector import SessionDetector

order_db = OrderDatabase()
session, phase = SessionDetector.get_current_session()

# Store new order
order_db.store_order(
    ticket=123456,
    direction="BUY",
    volume=0.1,
    entry_price=2650.50,
    sl=2648.00,
    tp=2655.00,
    sl_distance=2.50,
    strategy="SMC_STRATEGY",
    confidence=85.0,
    reason="Bullish breakout",
    session_type=session,
    market_phase=phase
)
```

### 2. Live PnL Updates

```python
# Update from MT5 live data
order_db.update_live_pnl(ticket=123456, live_pnl=12.50)

# Update management flags
order_db.update_management_flags(ticket=123456, sl_breakeven=True)

# Close order
order_db.close_order(ticket=123456, final_pnl=18.75, close_reason="TP_HIT")
```

### 3. Analytics and Reports

```python
# Today's statistics
stats = order_db.get_today_stats()
print(f"Today: {stats['trades']} trades, ${stats['total_pnl']} PnL")

# Strategy performance
performance = order_db.get_strategy_performance(30)  # Last 30 days

# Export snapshot
filepath = order_db.export_snapshot()
```

### 4. API Endpoints

Start the FastAPI server and access:

- `GET /orders/open` - Current open orders
- `GET /orders/closed?days=30` - Closed orders (last 30 days)
- `GET /orders/stats/today` - Today's statistics
- `GET /orders/stats/strategy` - Strategy performance
- `POST /orders/export` - Export data snapshot
- `GET /orders/summary` - Comprehensive summary

### 5. Integration with TradeManager

The system automatically integrates with the existing TradeManager:

```python
# TradeManager now includes order database
trade_manager = TradeManager(mt5_bridge)

# Register trade (now stores in both memory and database)
trade_manager.register_trade(
    ticket, direction, volume, entry, sl, tp, sl_distance,
    strategy, confidence, reason, scalp, be_trigger, timeout,
    features, session_type, market_phase
)

# Get enhanced status with database stats
status = trade_manager.status  # Includes today_stats
```

## Session Detection

Automatic session detection provides context:

- **ASIAN**: 00:00-07:00 UTC
- **LONDON**: 07:00-13:00 UTC  
- **NY**: 13:00-20:00 UTC
- **QUIET**: 20:00-00:00 UTC

Market phases: OPEN, ACTIVE, CLOSE, OVERLAP, QUIET

## Data Persistence

- **SQLite Database**: `orders.db` in project root
- **Automatic Backups**: Export snapshots regularly
- **Data Cleanup**: Remove old records to manage size
- **Migration Safe**: Database schema handles updates

## Benefits

1. **Complete Record**: Every order stored with full context
2. **No MT5 Dependency**: Local storage independent of MT5 connection
3. **Rich Analytics**: Detailed performance tracking by strategy/session
4. **Easy Export**: JSON snapshots for external analysis
5. **API Access**: RESTful interface for web dashboards
6. **Session Context**: Orders tagged with trading session information

## Example Usage

Run the example:

```bash
python order_example.py
```

This demonstrates:
- Order creation with session context
- Live PnL updates simulation
- Management flag updates
- Order closure
- Statistics display
- Data export

## API Server

Start the FastAPI server:

```bash
uvicorn main:app --host 127.0.0.1 --port 8899
```

Access the API documentation at: http://localhost:8899/docs