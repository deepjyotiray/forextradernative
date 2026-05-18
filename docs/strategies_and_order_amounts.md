# Live Strategies And Order Amounts

Global state:
- `TEMP_DISABLE_GLOBAL_BLOCKS = True` in [config.py](/c:/Users/deepjyotiray/source/repos/CTF/forextradernative/config.py:31)
- `ALLOW_CONCURRENT_STRATEGY_POSITIONS = True` in [config.py](/c:/Users/deepjyotiray/source/repos/CTF/forextradernative/config.py:31)

What that means:
- Shared software blocks are temporarily bypassed.
- A trade from one strategy does not block a different strategy from opening.
- Same-strategy duplicate prevention still remains through each strategy's own `max_active_trades`.
- MT5 / broker-side rejections can still happen.

Final live order-size behavior:
- The live engine now sends the strategy-provided `lot` unchanged.
- If a strategy does not provide `lot`, the fallback is `RiskManager.calculate_lot()`.
- MT5 submission still rounds `volume` to 2 decimals in [engine/mt5_bridge.py](/c:/Users/deepjyotiray/source/repos/CTF/forextradernative/engine/mt5_bridge.py:253), but there is no extra live clamp in `auto_trader.py`.

## Strategy List

### `SMC_CONFLUENCE`
Code:
- [engine/smc_strategy.py](/c:/Users/deepjyotiray/source/repos/CTF/forextradernative/engine/smc_strategy.py:10)
- [engine/liquidity_sweep_ob_strategy.py](/c:/Users/deepjyotiray/source/repos/CTF/forextradernative/engine/liquidity_sweep_ob_strategy.py:77)

Setup:
- HTF liquidity sweep
- HTF break of structure
- Retest of the originating order block
- LTF IFVG confirmation

Current order amount basis:
- Config `risk_pct = 1.0`
- Config `fixed_lot = null`
- Live execution lot comes from `RiskManager.calculate_lot()`
- No extra live clamp in `auto_trader.py`

Current config:
- [engine/strategy_configs/data/SMC_CONFLUENCE.json](/c:/Users/deepjyotiray/source/repos/CTF/forextradernative/engine/strategy_configs/data/SMC_CONFLUENCE.json:1)

### `M15_SCALP_DEEP`
Code:
- [engine/m15_scalp_deep_strategy.py](/c:/Users/deepjyotiray/source/repos/CTF/forextradernative/engine/m15_scalp_deep_strategy.py:112)

Setup:
- H1 trend with at least two BOS
- Unmitigated H1 supply / demand origin
- M5 structure shift
- Breaker block retrace
- Micro execution pressure confirmation

Current order amount basis:
- Config `risk_pct = 0.35`
- Config `fixed_lot = null`
- If `fixed_lot` stays null, the strategy computes:
- `risk_amount = balance * 0.35%`
- `lot = max(0.01, min(0.05, risk_amount / max(1.0, sl_dist * 100.0)))`
- If `fixed_lot` is set, internal cap is `0.10`
- Live execution uses the strategy-computed `lot` as-is

Current config:
- [engine/strategy_configs/data/M15_SCALP_DEEP.json](/c:/Users/deepjyotiray/source/repos/CTF/forextradernative/engine/strategy_configs/data/M15_SCALP_DEEP.json:1)

### `M15_ZONE_SCALP`
Code:
- [engine/m15_zone_scalp_strategy.py](/c:/Users/deepjyotiray/source/repos/CTF/forextradernative/engine/m15_zone_scalp_strategy.py:171)

Setup:
- M15 zone touch
- HTF bias alignment
- Rejection / displacement candle on the last closed M15 bar

Current order amount basis:
- Config `risk_pct = 0.5`
- Config `fixed_lot = null`
- If `fixed_lot` stays null, the strategy computes:
- `risk_amount = balance * 0.5%`
- `lot = max(0.01, min(0.05, risk_amount / max(1.0, sl_dist * 100.0)))`
- If `fixed_lot` is set, internal cap is `0.10`
- Live execution uses the strategy-computed `lot` as-is

Current config:
- [engine/strategy_configs/data/M15_ZONE_SCALP.json](/c:/Users/deepjyotiray/source/repos/CTF/forextradernative/engine/strategy_configs/data/M15_ZONE_SCALP.json:1)

### `M15_ZONE_SCALP_INVERSE`
Code:
- [engine/m15_zone_scalp_inverse_strategy.py](/c:/Users/deepjyotiray/source/repos/CTF/forextradernative/engine/m15_zone_scalp_inverse_strategy.py:1)

Setup:
- Uses the same trigger as `M15_ZONE_SCALP`
- Flips the trade direction
- Mirrors SL and TP from the base setup

Current order amount basis:
- Config `risk_pct = 0.5`
- Config `fixed_lot = null`
- If `fixed_lot` stays null, the strategy computes:
- `risk_amount = balance * 0.5%`
- `lot = max(0.01, min(0.05, risk_amount / max(1.0, sl_dist * 100.0)))`
- If `fixed_lot` is set, internal cap is `0.10`
- Live execution uses the strategy-computed `lot` as-is

Current config:
- [engine/strategy_configs/data/M15_ZONE_SCALP_INVERSE.json](/c:/Users/deepjyotiray/source/repos/CTF/forextradernative/engine/strategy_configs/data/M15_ZONE_SCALP_INVERSE.json:1)

### `TREND_CHANNEL`
Code:
- [engine/strategies/trend_channel_strategy.py](/c:/Users/deepjyotiray/source/repos/CTF/forextradernative/engine/strategies/trend_channel_strategy.py:27)

Setup:
- Supertrend-aligned channel continuation
- Buy support in uptrend or sell resistance in downtrend
- Optional H1 EMA alignment

Current order amount basis:
- Config `risk_pct = 1.0`
- Config `fixed_lot = null`
- Live execution lot comes from `RiskManager.calculate_lot()`
- No extra live clamp in `auto_trader.py`

Current config:
- [engine/strategy_configs/data/TREND_CHANNEL.json](/c:/Users/deepjyotiray/source/repos/CTF/forextradernative/engine/strategy_configs/data/TREND_CHANNEL.json:1)

### `SWING_ENGINE`
Code:
- [engine/swing_engine_strategy.py](/c:/Users/deepjyotiray/source/repos/CTF/forextradernative/engine/swing_engine_strategy.py:79)

Setup:
- D1 and H4 alignment
- Key swing location or pullback location
- Confirmed H4 rejection or directional candle
- Macro filter support

Current order amount basis:
- Config `risk_pct = 1.0`
- Config `fixed_lot = null`
- The strategy computes its own lot:
- `risk_amount = balance * 1.0%`
- `lot = max(0.01, min(0.03, risk_amount / max(1.0, sl_dist * 100.0)))`
- Live execution uses the strategy-computed `lot` as-is

Current config:
- [engine/strategy_configs/data/SWING_ENGINE.json](/c:/Users/deepjyotiray/source/repos/CTF/forextradernative/engine/strategy_configs/data/SWING_ENGINE.json:1)

### `HTF_LONG`
Code:
- [engine/strategies/htf_long_strategy.py](/c:/Users/deepjyotiray/source/repos/CTF/forextradernative/engine/strategies/htf_long_strategy.py:32)
- [engine/htf_bias_engine.py](/c:/Users/deepjyotiray/source/repos/CTF/forextradernative/engine/htf_bias_engine.py:1)

Setup:
- Macro bullish for gold
- Weekly or daily continuation long
- Pullback completion and H1 structure support

Current order amount basis:
- Config `risk_pct = 1.0`
- Config `fixed_lot = null`
- Live execution lot comes from `RiskManager.calculate_lot()`
- No extra live clamp in `auto_trader.py`

Current config:
- [engine/strategy_configs/data/HTF_LONG.json](/c:/Users/deepjyotiray/source/repos/CTF/forextradernative/engine/strategy_configs/data/HTF_LONG.json:1)

### `HTF_SHORT`
Code:
- [engine/strategies/htf_short_strategy.py](/c:/Users/deepjyotiray/source/repos/CTF/forextradernative/engine/strategies/htf_short_strategy.py:27)
- [engine/htf_short_bias_engine.py](/c:/Users/deepjyotiray/source/repos/CTF/forextradernative/engine/htf_short_bias_engine.py:1)

Setup:
- Macro bearish for gold
- Weekly or daily continuation short
- Pullback completion and H1 structure support

Current order amount basis:
- Config `risk_pct = 1.0`
- Config `fixed_lot = null`
- Live execution lot comes from `RiskManager.calculate_lot()`
- No extra live clamp in `auto_trader.py`

Current config:
- [engine/strategy_configs/data/HTF_SHORT.json](/c:/Users/deepjyotiray/source/repos/CTF/forextradernative/engine/strategy_configs/data/HTF_SHORT.json:1)

## Current Strategy-Local Hard Requirements

`SMC_CONFLUENCE`
- Tick, spread, session, optional news pass
- HTF and LTF data present
- HTF sweep plus BOS
- LTF IFVG confirmation
- Valid stop and minimum RR

`M15_SCALP_DEEP`
- Gold symbol only
- H1 and M5 data present
- Spread, session, optional news pass
- Not ranging unless allowed
- H1 zone, tap, M5 breaker, micro confirmation
- Valid stop and minimum RR

`M15_ZONE_SCALP`
- Gold symbol only
- M15 data present
- Spread, session, optional news pass
- Valid M15 ATR and detected zones
- HTF bias alignment
- Rejection / displacement candle

`M15_ZONE_SCALP_INVERSE`
- Gold symbol only
- Needs a valid `M15_ZONE_SCALP` setup first
- Same session, spread, ATR, zone, and candle requirements as base strategy
- Flips direction and mirrors stop/target

`TREND_CHANNEL`
- Stable supertrend
- Available trend channel
- Valid ATR
- Boundary touch
- Optional H1 alignment
- Valid RR

`SWING_ENGINE`
- H4 and D1 data present
- Spread pass
- D1 / H4 alignment
- Macro not opposing
- Key level or pullback location
- H4 confirmation candle

`HTF_LONG`
- Valid price and spread
- HTF long decision from bias engine
- Valid SL distance
- Minimum RR

`HTF_SHORT`
- Valid price and spread
- HTF short decision from bias engine
- Valid SL distance
- Minimum RR
