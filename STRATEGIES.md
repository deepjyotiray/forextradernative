# Strategies Reference

## Overview

Three modes: **AUTO** (evaluates all, picks highest confidence), **SMC_CONFLUENCE**, **SWEEP_SCALPER**.

Default on startup: `AUTO` (configurable via `config.DEFAULT_STRATEGY`).

---

## 1. SMC_CONFLUENCE — Smart Money Concepts Swing Strategy

**Goal:** High-confluence swing trades using multi-timeframe analysis, liquidity sweeps, order blocks, and rejection candles.

### Signal Pipeline (in order)

| # | Check | Parameters | Pass Condition | Fail Action |
|---|-------|-----------|----------------|-------------|
| 1 | **Session Filter** | `get_session()` | Session = LONDON or NEW_YORK | Reject if ASIAN |
| 2 | **Market Regime** | H4, H1, M15 candles + tick snapshot | `regime.trade_allowed == True` | Reject if NEWS_VOLATILITY or ATR ratio > 3.0 |
| 3 | **MTF Bias** | H4 EMA50/200, H1 structure (HH/HL/LH/LL + BOS), M15 pullback | `bias.direction != "NEUTRAL"` | Reject if no directional bias |
| 4 | **Tick Chaos** | `tick_processor.is_chaotic()` | velocity ≤ 50 OR spread stable | Reject if chaotic (news-like) |
| 5 | **Spread Model** | Configurable strategy spread-quality gate | spread_mean < `SMC_SPREAD_MEAN_MAX`, spread_std ≤ `SMC_SPREAD_STD_MAX`, percentile ≤ `SMC_SPREAD_PERCENTILE_MAX` | Reject unless the matching individual spread override is enabled |
| 6 | **Compression** | `COMPRESSION_RANGE_LOOKBACK`, `COMPRESSION_ATR_MULTIPLIER` | RangeN < multiplier × ATR and ATR rising | Reject unless matching override enabled |
| 7 | **Confluence Score** | See scoring table below | score >= configured SMC threshold | Reject with score + reasons |
| 8 | **SL/TP Computation** | ATR, zones, liquidity sweeps, key levels | Valid SL and TP computed, SL distance ≥ 0.50 | Reject |
| 9 | **Risk:Reward** | `tp_dist / sl_dist` | `RR >= SMC_MIN_RR` | Reject |

### Confluence Scoring Breakdown

| Component | Max Score | How It's Calculated |
|-----------|-----------|-------------------|
| **Bias alignment** | 0.25 | `min(0.25, bias.confidence * 0.3)` |
| **Regime alignment** | 0.15 | +0.15 if TRENDING in same direction, +0.10 if EXPANSION |
| **Price at key level** | 0.25 | At support zone (LONG), resistance zone (SHORT), order block, or FVG |
| **Liquidity sweep** | 0.20 | Recent buy sweep (LONG) or sell sweep (SHORT) on M5 |
| **Rejection candle** | 0.15 | M5 candle with >55% wick ratio in trade direction |
| **RSI confirmation** | 0.05 | RSI 30-50 for LONG, RSI 50-70 for SHORT |
| **Tick momentum** | 0.05 | Momentum aligned with direction |
| **Correlation (DXY/US10Y)** | 0.10 | DXY/US10Y bias matches direction. -0.05 penalty if conflicting |
| **Total possible** | **1.20** | Capped at 1.0 |

### SL/TP Logic

- **SL base:** ATR × 1.5 from entry
- **SL refinement:** Nearest support/resistance zone or sweep wick level
- **SL max:** ATR × 3
- **SL min distance:** $0.50
- **TP base:** SL distance × 2.0
- **TP refinement:** Nearest opposing zone or key level (session high/low, prev day H/L)
- **TP max:** ATR × 5

### Lot Sizing

- Standard risk calc: `balance × (risk_pct × risk_multiplier) / (sl_distance × pip_value_per_lot)`
- Asian session: lot × 0.70
- Clamped to `[MIN_LOT, MAX_LOT]`

---

## 2. SWEEP_SCALPER — Liquidity Sweep Scalp Strategy

**Goal:** Quick $3-$6 scalps on liquidity sweeps at London/NY open.

### Signal Pipeline (in order)

| # | Check | Parameters | Pass Condition | Fail Action |
|---|-------|-----------|----------------|-------------|
| 1 | **Time Window** | UTC hour | 07-09 (London), 12-13 (Overlap), 13-15 (NY) | Reject outside windows |
| 2 | **Session Trade Limit** | `_session_trades` | < 5 trades this session | Reject |
| 3 | **Spread Model** | Configurable strategy spread-quality gate | spread_mean < `SCALPER_SPREAD_MEAN_MAX`, spread_std ≤ `SCALPER_SPREAD_STD_MAX`, percentile ≤ `SCALPER_SPREAD_PERCENTILE_MAX` | Reject unless the matching individual spread override is enabled |
| 4 | **ATR Filter** | M1 ATR(14) | `SCALPER_ATR_MIN ≤ ATR ≤ SCALPER_ATR_MAX` | Reject if too quiet or too volatile |
| 5 | **EMA20 Bias** | M1 EMA(20) slope | `abs(slope) >= SCALPER_EMA20_SLOPE_MIN` | Reject if flat |
| 6 | **M5 Bias** | M5 close vs EMA(20) | Must not conflict with sweep direction | Reject if conflicting |
| 7 | **Sweep Detection** | `SCALPER_SWEEP_LOOKBACK`, `SCALPER_SWEEP_TOLERANCE` | Equal highs/lows swept + price reclaimed | No signal if no sweep |
| 8 | **First-Sweep-Only** | `_traded_levels` dict | Level not traded in last 20 min (±$0.50) | Reject duplicate level |
| 9 | **EMA Direction Agreement** | EMA20 slope vs sweep direction | Slope positive for LONG, negative for SHORT | Reject if conflicting |
| 10 | **Displacement Candle** | Last M1 candle | body_ratio ≥ `SCALPER_BODY_RATIO_MIN`, body > avg(last 10), bullish for LONG / bearish for SHORT | Reject |
| 11 | **Price vs EMA20** | M1 close vs EMA20 | Price above EMA20 for LONG, below for SHORT | Reject |
| 12 | **Execution Quality** | `tick_processor.check_execution_quality()` | velocity ≥ 3 (or increasing), direction stable, spread stable | Reject |
| 13 | **Tick Direction** | `tick_snapshot.dir_pct` | Meets `SCALPER_TICK_DIR_THRESHOLD` in trade direction | Reject unless tick-direction override enabled |
| 14 | **Spread Re-check** | `tick_processor.spread_changed(...)` | Spread has not widened beyond `SCALPER_CURRENT_SPREAD_DELTA_MAX` since signal | Reject unless post-signal spread override enabled |

### SL/TP Logic

- **SL:** `clamp(ATR × 0.6, $0.80, $1.50)`, anchored to sweep level - $0.10
- **TP:** `max(ATR × 0.8, spread_mean × 2.2)`, capped at `ATR × 1.5`, floor at `spread_mean × 2.2`
- **Confidence:** Fixed 0.80

### Trade Management

| Parameter | Value | Description |
|-----------|-------|-------------|
| `_be_trigger` | $0.30 | Move SL to breakeven after $0.30 profit |
| `_timeout` | 60s | Speed exit if trade stalls after 60 seconds |
| `_early_fail` | $0.20 | Exit if $0.20 adverse in first 15 ticks |
| Max lot | 0.05 | Hard cap for scalps |

### Sweep Detection Algorithm

1. Scan last 15 M1 candles for **equal lows** (within $0.30 tolerance)
2. Look for a candle that **wicks below** the equal low level
3. Confirm price **reclaims** above the level with a bullish close → **BUY sweep**
4. Mirror logic for equal highs → **SELL sweep**

---

## 3. AUTO Mode

Runs **all** registered strategies every cycle. Picks the signal with the highest `confidence` score. Tags the signal with `_auto_selected: True` and includes results from all strategies.

---

## Shared Pre-Trade Filters (applied in auto_trader.py after strategy signal)

These run **after** a strategy returns BUY/SELL, before execution:

| # | Filter | Parameters | Condition |
|---|--------|-----------|-----------|
| 1 | **Market open** | `is_market_open()` | Not Saturday, not Friday after 22:00 UTC, not 21:00 UTC rollover |
| 2 | **Session open block** | `is_session_open_blocked()` | First 3 min of each session blocked |
| 3 | **Calendar blackout** | `calendar.check()` | 30 min before / 15 min after high-impact USD events (NFP, CPI, FOMC, etc.) |
| 4 | **Risk manager** | `risk.can_trade(account, positions)` | See risk checks below |
| 5 | **XGBoost filter** | `xgb_model.predict_win_prob()` | Block if win probability < 35% (when trained) |
| 6 | **XGBoost blending** | Blend confidence | `final_conf = original × 0.7 + xgb_prob × 0.3` |
| 7 | **SL/TP validation** | Signal SL, TP | Both must be non-null |
| 8 | **SL distance** | `sl_distance` | Must be > 0 |
| 9 | **Confidence floor** | `confidence` | Must be ≥ 0.55 |
| 10 | **Spread re-check** | `tick_processor.spread_changed(spread, 0.03)` | Spread hasn't widened since signal |
| 11 | **Execution retry** | 3 attempts, 100ms apart | `bridge.open_trade()` |

### Risk Manager Checks

| Check | Parameter | Threshold |
|-------|-----------|-----------|
| Daily target | `DAILY_TARGET_DOLLARS` | $100 — stop trading when hit |
| Daily loss limit | `DAILY_LOSS_LIMIT_PCT` | 2% of balance |
| Max positions | `MAX_POSITIONS` | 2 concurrent |
| Max drawdown | `MAX_DRAWDOWN_PCT` | 5% |
| Consecutive losses | `MAX_CONSECUTIVE_LOSSES` | 4 — then pause 900s |
| Trade cooldown | `MIN_TRADE_COOLDOWN` | 10s between trades |
| Risk per trade | `MAX_RISK_PCT` | 0.5% of balance |
| Lot range | `MIN_LOT` / `MAX_LOT` | 0.01 — 1.00 |
| Risk multiplier | Performance tracker | 0.50x — 1.25x (adaptive) |

---

## Data Pipeline

| Timeframe | Refresh Interval (cycles) | Used By |
|-----------|--------------------------|---------|
| M1 | Every 1 cycle | Indicators, Sweep Scalper |
| M5 | Every 2 cycles | Zones, Liquidity sweeps, SMC |
| M15 | Every 10 cycles | Regime, Bias, Order blocks, FVG |
| H1 | Every 60 cycles | Regime, Bias, Key levels |
| H4 | Every 120 cycles | Regime, Bias, Correlation |
| D1 | Every 300 cycles | Key levels (prev day H/L) |

### Indicator Values (computed from M1)

| Indicator | Period | Usage |
|-----------|--------|-------|
| EMA 9 | 9 | Slope for momentum |
| EMA 15 | 15 | Gap from EMA9 |
| EMA 50 | 50 | Trend filter |
| EMA 200 | 200 | Macro trend (if data available) |
| ATR 7 | 7 | Primary volatility (short-term) |
| ATR 14 | 14 | Standard volatility |
| ATR 20 | 20 | ATR ratio denominator |
| RSI 14 | 14 | Overbought/oversold |

---

## XGBoost Model

**Features (13):** session, direction, hour, ATR, ATR ratio, RSI, EMA slope, body ratio, spread, regime, bias confidence, SL distance, volume

**Training:** Auto-retrains every 10 new trades (min 15 trades with features). Persisted to `xgb_model.json`.

**Inference:** Predicts win probability on every signal. Blocks if < 35%. Blends into confidence at 30% weight.

---

## Correlation Engine

| Instrument | Relationship to Gold | Weight |
|------------|---------------------|--------|
| DXY (USD Index) | Inverse — DXY up = gold down | 0.30 |
| US10Y (10yr yield) | Inverse — yields up = gold down | 0.20 |
| EURUSD (fallback) | Same direction — EUR up = gold up | 0.20 |

Trend detection: H4 EMA20 vs EMA50 + normalized slope. Threshold: ±0.5 normalized slope.
