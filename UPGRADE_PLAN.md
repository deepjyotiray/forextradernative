# XAUUSD Auto Trader — Upgrade Plan
# Last updated: 2026-04-22

========================================
SYSTEM STATUS
========================================

Account: MetaQuotes-Demo | Balance: ~$560
Strategies: SMC_CONFLUENCE + SWEEP_SCALPER (AUTO mode)
Performance: 110 trades, 77W/33L (70% WR), +$87 P&L
XGBoost: Trained on 61 featured trades, spread = #1 predictor
Branch: tier1-upgrades (committed)
Baseline: main branch (rollback target)

========================================
TIER 1 — DONE ✓
========================================

1.1 SPREAD ENGINE ✓
    - Rolling spread model (mean/std/percentile over 100 ticks + 10min window)
    - Scalper max: $0.25 mean, percentile <= 50%
    - SMC max: $0.30 mean, percentile <= 60%
    - Pre-execution re-check: abort if spread widened > $0.03 since signal
    - Files: tick_processor.py, sweep_scalper.py, smc_strategy.py, auto_trader.py

1.2 DYNAMIC SL + EARLY FAILURE EXIT ✓
    - SL = clamp(ATR * 0.6, $0.80, $1.50) — was fixed $0.60-$1.00
    - Early failure: exit if $0.20 adverse in first 15 seconds
    - Speed exit: exit if no +$0.30 move in 60 seconds — was 150s
    - Files: sweep_scalper.py, trade_manager.py

1.3 FIRST-SWEEP-ONLY ✓
    - Tracks swept levels in dict (rounded to $0.50)
    - Blocks re-entry at same level for 20 minutes
    - Prevents rapid-fire losses at same price zone
    - Files: sweep_scalper.py

1.4 XGBOOST DATA FIX ✓
    - Trains from performance tracker (has real features) not CSV deals (defaults)
    - Filters to trades with non-zero feature values
    - Retrains only on featured trades after closes
    - Files: auto_trader.py

1.5 PARTIAL CLOSE FIX ✓
    - Skips partial close when position is at minimum lot (0.01)
    - Instead tightens SL to lock +0.5R and sets runner TP at 2R
    - Prevents closing entire position when trying to close half
    - Files: trade_manager.py

========================================
TIER 2 — PENDING
========================================

2.1 SESSION WINDOW ENFORCEMENT
    Priority: HIGH (XGBoost: hour = 29.8% importance)
    
    Current:
      Scalper: 07-09, 12-13, 13-15 UTC (already tightened in tier 1)
      SMC: blocks Asian only, trades all of London + NY
    
    Change:
      SMC should also respect optimal windows:
        - London: 07:00-12:00 (keep as-is)
        - NY: 13:00-19:00 (extend from current 13:00-22:00)
        - Block 19:00-22:00 (low volume NY afternoon)
      
    Why:
      XGBoost says hour is 29.8% of win prediction.
      Trades after 19:00 UTC have lower follow-through.
      The SMC trade at 17:xx that lost $9.85 was in this dead zone.
    
    Files: smc_strategy.py
    Risk: May miss occasional late NY setups

2.2 TRADE CLUSTERING
    Priority: HIGH (prevented 5 rapid losses)
    
    Current: No clustering control
    
    Change:
      - Max 2 consecutive trades in same direction
      - Second trade ONLY if:
        a) Previous trade was profitable
        b) Previous trade duration < 60 seconds (fast win = momentum still alive)
        c) Tick momentum conditions still valid (re-check execution quality)
      - Reset counter on direction change or after 10 minutes
    
    Implementation:
      Add to auto_trader._cycle_once():
        Track: _last_trade_direction, _last_trade_won, _last_trade_duration, _last_trade_time
        Before execution: check clustering rules
    
    Files: auto_trader.py
    Risk: May block valid follow-up trades in strong trends

2.3 COST-AWARE DYNAMIC TP
    Priority: MEDIUM
    
    Current:
      Scalper: dynamic TP = max(ATR*0.8, spread*2.2) capped at ATR*1.5 (done in tier 1)
      SMC: TP from zones, clamped to 5x ATR
    
    Change for SMC:
      TP = max(zone_target, ATR * 1.0)
      TP capped at ATR * 3.0 (was 5x — too far)
      TP floor = spread * 2.5 (ensure net positive after costs)
    
    Why:
      Some SMC trades had TP at $13+ which never got hit.
      ATR * 3.0 cap keeps TP realistic.
    
    Files: smc_strategy.py
    Risk: Smaller wins on big moves

2.4 EXECUTION QUALITY FILTER
    Priority: MEDIUM
    
    Current:
      Scalper: uses check_execution_quality() (tier 1)
      SMC: only checks chaos + spread
    
    Change for SMC:
      Before execution, validate:
        - Tick velocity > 2 (not dead)
        - Direction stable (not oscillating)
        - Spread stable
      If any fails, skip trade
    
    Files: smc_strategy.py, auto_trader.py
    Risk: May filter valid slow-developing setups

2.5 ANTI-STARVATION FALLBACK
    Priority: MEDIUM
    
    Current: No fallback — if filters are too tight, zero trades
    
    Change:
      If no trades occur during a valid session (2+ hours):
        Relax ONE parameter:
          Option A: Tick direction threshold 65% -> 60%
          Option B: Spread tolerance +$0.02
        Do NOT relax multiple simultaneously
        Log which parameter was relaxed
        Reset relaxation at next session
    
    Implementation:
      Track: _session_trade_count, _session_start_time
      At each cycle: if session active > 2 hours and 0 trades, apply fallback
    
    Files: auto_trader.py, sweep_scalper.py
    Risk: Slightly lower quality trades during fallback

========================================
TIER 3 — FUTURE
========================================

3.1 COMPRESSION GATE REFINEMENT
    Priority: LOW-MEDIUM
    
    Current: Regime engine detects EXPANSION (range < 3x ATR + ATR rising)
    
    Change for scalper:
      Only allow sweep trades if:
        range(last 10 M1 candles) < 1.8 * ATR
        AND ATR is rising (current ATR > ATR 5 candles ago)
      This filters sweeps in choppy conditions
    
    Files: sweep_scalper.py
    Risk: Fewer scalp trades

3.2 MOMENTUM DECAY EXIT
    Priority: LOW-MEDIUM
    
    Current: Speed exit at 60s, early failure at 15s
    
    Change:
      After entry, monitor tick velocity:
        If velocity drops > 50% from entry velocity within 30s, exit
      This catches fading displacement moves before they reverse
    
    Implementation:
      Store entry_velocity in TradeRecord
      In _manage_scalp: compare current velocity to entry velocity
    
    Files: trade_manager.py, tick_processor.py
    Risk: May exit trades that pause then continue

3.3 LIGHT REGIME FILTER
    Priority: LOW
    
    Current: Regime blocks NEWS_VOLATILITY only
    
    Change:
      Also block if:
        ATR is falling (current < 5-candle-ago) AND
        EMA slope is flat (abs < 0.05)
      This catches dying/sideways markets
    
    Files: regime.py or auto_trader.py
    Risk: May block trades at start of new moves

3.4 STRUCTURED LOGGING
    Priority: LOW
    
    Current: Free-text log with tag + message
    
    Change:
      Log per trade as structured JSON:
        {
          spread_mean, spread_std, atr, sl, tp,
          tick_velocity, dir_pct, entry_validation,
          exit_reason, duration_s, pnl
        }
      Store in trade_log.json for analysis
      XGBoost can train directly from this
    
    Files: auto_trader.py, trade_manager.py
    Risk: None — pure improvement

3.5 XGBOOST FEATURE EXPANSION
    Priority: LOW (needs more data)
    
    Current features (13):
      session, direction, hour, atr, atr_ratio, rsi,
      ema_slope, body_ratio, spread, regime, bias_conf,
      sl_distance, volume
    
    Add:
      - strategy (SMC vs SCALPER encoded)
      - sweep_level_distance (how far price is from sweep)
      - time_since_session_open (minutes)
      - consecutive_wins/losses
      - spread_percentile
      - compression_ratio (range / ATR)
    
    Needs: 200+ featured trades for meaningful learning
    Files: xgb_model.py, auto_trader.py

========================================
INFRASTRUCTURE — PENDING
========================================

I1. REMOTE ORIGIN
    Push to GitHub/GitLab for backup
    Current: local git only, no remote

I2. AUTO-START ON BOOT
    Windows Task Scheduler to run start_trader.bat on login
    Current: manual double-click

I3. ALERTING
    Send notifications on:
      - Trade opened/closed
      - Daily P&L threshold hit
      - Error count > 5
      - MT5 disconnected
    Options: Telegram bot, email, Windows toast

I4. BACKTESTING
    Replay historical M1 data through strategies
    Measure: win rate, drawdown, expectancy per strategy
    Validate tier 2/3 changes before live deployment

I5. MULTI-SYMBOL
    Add: EURUSD, GBPUSD as separate strategy instances
    Shared: risk manager (cross-symbol position limits)
    Separate: strategy params per symbol

========================================
ROLLBACK PROCEDURE
========================================

If performance degrades after any tier:

1. Stop the bot:
   curl -X POST http://localhost:8899/shutdown

2. Switch to baseline:
   "C:\Program Files\Git\cmd\git.exe" checkout main

3. Restart:
   Double-click start_trader.bat

4. To go back to tier 1:
   "C:\Program Files\Git\cmd\git.exe" checkout tier1-upgrades

========================================
METRICS TO TRACK
========================================

After each tier, compare 50-trade windows:

  Win rate: target 75%+ (baseline 70%)
  Avg win: target $5+ (baseline $4.50)
  Avg loss: target < $2 (baseline $3.50)
  Max drawdown: target < $15 (baseline $10)
  Trades/day: target 5-8 (baseline 10-15)
  Expectancy: target $2+/trade (baseline $0.80)
  Scalper instant SL rate: target < 10% (baseline 40%)

========================================
END
========================================
