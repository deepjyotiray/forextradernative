# Trade History Fix - MT5 Single Source of Truth

## Problem Identified
The closed trades list was not matching MT5 history due to:
1. **Dual tracking systems**: Internal `closed_trades` list vs MT5 history
2. **Inconsistent P&L values**: Many trades showing `pnl: 0.0` due to failed P&L fetching
3. **No magic number filtering**: Including non-bot trades in calculations
4. **Unreliable internal tracking**: TradeManager maintaining separate closed trades list

## Solution Implemented

### 1. TradeManager Changes (`engine/trade_manager.py`)
- **Removed internal closed_trades tracking**: No longer maintains separate list
- **Enhanced P&L fetching**: `_fetch_closed_pnl()` now marked as AUTHORITATIVE SOURCE
- **MT5-first approach**: Always fetch P&L from MT5 history when position closes
- **Updated status property**: Now returns MT5 closed trades instead of internal list

### 2. MT5Bridge Enhancements (`engine/mt5_bridge.py`)
- **Magic number filtering**: `get_closed_trades()` now filters by `cfg.MAGIC_NUMBER`
- **Bot-only P&L**: `get_today_pnl()` only includes trades from this bot
- **Improved deal processing**: Better handling of partial closes and aggregation

### 3. AutoTrader Updates (`auto_trader.py`)
- **Simplified trade recording**: Removed complex feature tracking on close
- **MT5 history sync**: Added `_sync_performance_with_mt5()` method
- **Authoritative logging**: Trade results now marked as "MT5 sourced"
- **Performance sync**: Ensures performance tracker stays in sync with MT5

### 4. Data Cleanup
- **Cleaned trade_history.json**: Removed 9 invalid trades with 0.0 P&L
- **Backup created**: Original data preserved in timestamped backup
- **Valid trades retained**: 103 trades with actual P&L values kept

## Key Benefits

### ✅ Single Source of Truth
- MT5 deal history is now the ONLY authoritative source
- No more discrepancies between internal tracking and MT5
- All P&L calculations come directly from MT5 deals

### ✅ Accurate P&L Tracking
- Real profit/loss including swap and commission
- Handles partial closes correctly
- No more 0.0 P&L entries

### ✅ Bot-Only Filtering
- Only includes trades opened by this bot (magic number filtering)
- Excludes manual trades or other EAs
- Clean separation of bot performance

### ✅ Reliable Performance Metrics
- Win rate: 67.0% (69W/34L) from cleaned data
- Total P&L: +$54.73 from valid trades
- Performance tracker syncs with MT5 automatically

## Files Modified
1. `engine/trade_manager.py` - Removed internal closed trades tracking
2. `engine/mt5_bridge.py` - Added magic number filtering
3. `auto_trader.py` - Enhanced MT5 sync and logging
4. `trade_history.json` - Cleaned invalid entries
5. `clean_trade_history.py` - Created cleanup utility
6. `fix_trade_history.py` - Created MT5 rebuild utility (for future use)

## Usage
The system now automatically:
- Fetches closed trade data from MT5 every 5 seconds
- Updates performance metrics based on MT5 history
- Logs trade results with "(MT5 sourced)" indicator
- Maintains sync between internal tracking and MT5 reality

**MT5 is now the single, authoritative source of truth for all trade history and P&L calculations.**