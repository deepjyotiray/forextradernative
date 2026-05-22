# AI Trade Correction Architecture

## Enabled Strategy

- Enabled strategy name: `M15_ZONE_SCALP`
- Source of truth: `runtime_config.json` sets `DEFAULT_STRATEGY` to `["M15_ZONE_SCALP"]`
- Important note: the root `enabled` file is present in the repo but is not referenced by the live Python execution path

## Key Source Files Found

- Trader runtime and execution loop: [auto_trader.py](/c:/Users/deepjyotiray/source/repos/CTF/forextradernative/auto_trader.py)
- Strategy selection and signal generation: [engine/strategy_manager.py](/c:/Users/deepjyotiray/source/repos/CTF/forextradernative/engine/strategy_manager.py)
- Enabled strategy logic: [engine/m15_zone_scalp_strategy.py](/c:/Users/deepjyotiray/source/repos/CTF/forextradernative/engine/m15_zone_scalp_strategy.py)
- Market regime / bias / liquidity context:
  [engine/market_state.py](/c:/Users/deepjyotiray/source/repos/CTF/forextradernative/engine/market_state.py)
  [engine/mtf_bias.py](/c:/Users/deepjyotiray/source/repos/CTF/forextradernative/engine/mtf_bias.py)
  [engine/liquidity.py](/c:/Users/deepjyotiray/source/repos/CTF/forextradernative/engine/liquidity.py)
  [engine/indicators.py](/c:/Users/deepjyotiray/source/repos/CTF/forextradernative/engine/indicators.py)
  [engine/tick_processor.py](/c:/Users/deepjyotiray/source/repos/CTF/forextradernative/engine/tick_processor.py)
- Live order send / MT5 bridge: [engine/mt5_bridge.py](/c:/Users/deepjyotiray/source/repos/CTF/forextradernative/engine/mt5_bridge.py)
- Order persistence and trade lifecycle: [engine/trade_manager.py](/c:/Users/deepjyotiray/source/repos/CTF/forextradernative/engine/trade_manager.py)
- Permanent order history store: [engine/order_database.py](/c:/Users/deepjyotiray/source/repos/CTF/forextradernative/engine/order_database.py)
- Decision and closed-trade logs:
  [engine/master_control.py](/c:/Users/deepjyotiray/source/repos/CTF/forextradernative/engine/master_control.py)
  [engine/trade_attribution.py](/c:/Users/deepjyotiray/source/repos/CTF/forextradernative/engine/trade_attribution.py)
  [engine/decision_logger.py](/c:/Users/deepjyotiray/source/repos/CTF/forextradernative/engine/decision_logger.py)

## Existing Config Mechanism

- Shared runtime config is loaded through `config.py -> load_runtime_config()` from `runtime_config.json`
- The active runtime profile is managed through `config_profiles.json`
- Strategy-specific config values are loaded from `engine/strategy_configs/data/*.json` and applied live into `config.py`
- Safe live modification point for this feature:
  use a separate runtime override layer instead of rewriting strategy source or mutating unrelated config globally during trading

## Current Signal / Execution Flow

1. `AutoTrader._cycle_once()` builds `strat_data`
2. `StrategyManager.generate_signals()` asks `M15_ZONE_SCALP` for a signal
3. `AutoTrader._prepare_execution_order()` performs execution approval / blocking checks
4. `AutoTrader._execute_prepared_order()` sends the order through `MT5Bridge.open_trade(...)`
5. `TradeManager.register_trade()` persists the live trade and its feature snapshot into `orders.db`
6. `TradeManager.manage_all()` tracks live management and detects closures
7. `AutoTrader` reconciles closed trades from MT5 + `orders.db`

## Where The New Service Hooks In

- Pre-execution hook:
  `AutoTrader._prepare_execution_order()`
  The new service captures the signal snapshot, checks active AI overrides, compares similar historical patterns, and can `SKIP`, `WAIT`, or modify SL/TP/exit behavior before MT5 order send
- Pending confirmation release:
  `AutoTrader._cycle_once()`
  The new service can release previously delayed signals when confirmation conditions become true
- Trade-open snapshot attach:
  `AutoTrader._execute_prepared_order()`
  The new service binds the live MT5 ticket to the earlier signal snapshot
- Runtime trade monitoring:
  `AutoTrader._cycle_once()`
  The new service updates open-trade MFE/MAE and market-change fields
- Post-close forensics trigger:
  `AutoTrader._cycle_once()` after `TradeManager.manage_all()` returns closed trades
- Periodic AI triggers:
  driven from the new service during cycle updates and post-close events

## Data Already Available

- Strategy name, direction, entry, SL, TP, RR, confidence, zone type, route type
- Spread, ATR, RSI, EMA slopes, body ratio
- Session label
- M1/M5/M15/H1/H4/D1 candle data in memory
- Regime, HTF/LTF trend, structure, market memory
- Tick pressure / microflow statistics
- Liquidity sweeps, order blocks, FVGs, key levels
- Support / resistance zones from `ZoneDetector`
- `orders.db` with open/close times, PnL, close reason, peak/final R, held seconds, JSON feature blob
- `closed_trades.jsonl`, `trade_attribution.jsonl`, `trade_decisions.jsonl`, `trader.log`

## New Data Recorded By This Feature

- New SQLite store: `ai_trade_correction.db`
- Per-signal / per-trade snapshots with:
  signal snapshot
  open-trade live metrics
  closed-trade forensic snapshot
  counterfactual results
  applied analysis payload
- New pending confirmation queue in the same SQLite store
- New live override store: `ai_trade_correction_live_rules.json`
- New audit outputs:
  `ai_decisions.jsonl`
  `live_modifications.jsonl`
  `trade_forensics.jsonl`
  `ai_request_response_meta.jsonl`
  `daily_ai_report.md`

## How Live Modifications Are Applied

- The AI does not place trades and does not rewrite strategy source during runtime
- The AI returns strict JSON validated against `ai_trade_correction_service/decision_schema.py`
- Validated modifications are written into the live override store
- Before a new `M15_ZONE_SCALP` trade is sent, the override controller can:
  skip the setup
  convert the setup into a pending confirmation flow
  widen / relocate SL for that setup pattern
  reduce / reshape TP
  enable trailing / partial-exit behavior through per-trade feature overrides consumed by `TradeManager`
- Exit-behavior changes are injected per trade through the existing `TradeManager` exit-profile feature override path, so the strategy remains the same strategy

## Missing Data / TODOs

- Exact post-exit path tracking after a bot restart is not fully reconstructable from current persisted data alone
- Nearest safe integration point implemented:
  the new service updates open-trade lifecycle data in-memory each cycle and writes approximated post-exit counterfactuals at close time
- VWAP was not already a first-class engine field
- Nearest safe integration point implemented:
  the new service computes a local VWAP approximation from recent M1/M5 candle volume when volume exists, otherwise marks VWAP relation as `UNKNOWN`

