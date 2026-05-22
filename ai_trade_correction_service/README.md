# AI Trade Correction Service

This module adds a trade-forensics and adaptive-correction layer for the one enabled live strategy. It does not place orders, does not switch strategies, and does not act like a generic loss-based risk throttle.

## What it does

- Captures a full signal snapshot before execution
- Tracks open-trade behaviour and closed-trade outcomes
- Runs forensic review triggers after closes and on periodic session checkpoints
- Stores validated live overrides for the single enabled strategy
- Applies those overrides before future executions as `SKIP`, `WAIT`, SL changes, TP changes, trailing changes, or partial-exit changes

## Files created

- `ai_trade_correction.db`
- `ai_trade_correction_live_rules.json`
- `ai_decisions.jsonl`
- `live_modifications.jsonl`
- `trade_forensics.jsonl`
- `ai_request_response_meta.jsonl`
- `daily_ai_report.md`

## Enable / Disable

- Enable service: set `AI_TRADE_CORRECTION_ENABLED=1`
- Disable service: set `AI_TRADE_CORRECTION_ENABLED=0`
- Dry-run mode without API calls: set `AI_TRADE_CORRECTION_DRY_RUN=1`
- Live OpenAI mode: set `OPENAI_API_KEY` and keep `AI_TRADE_CORRECTION_DRY_RUN=0`

## OpenAI settings

- `AI_TRADE_CORRECTION_MODEL`
- `AI_TRADE_CORRECTION_TIMEOUT_SECONDS`
- `OPENAI_API_BASE`

## Safety model

- Only the currently enabled strategy is corrected
- AI output must validate against the strict schema in `decision_schema.py`
- Invalid AI payloads are rejected and no live change is applied
- API failures fail open: existing strategy behaviour stays unchanged
- All modifications have expiry and rollback conditions

## Main runtime flow

1. Strategy generates a signal
2. Service stores the signal snapshot
3. Active overrides and similar trade patterns are checked
4. Result is one of:
   `TAKE_NOW`, `WAIT_FOR_CONFIRMATION`, `WAIT_FOR_RETEST`, `MODIFY_SL`, `MODIFY_TP`, `USE_TRAILING`, `SKIP`
5. If a trade opens, per-trade exit overrides are stored in the trade feature blob
6. When the trade closes, counterfactuals and forensic classification are updated
7. AI review may apply new validated overrides for later similar setups

## Dry-run verification

The unit tests cover:

- invalid structured output rejection
- losing trade classification
- similar setup skip/delay behaviour
- SL/TP override application
- modification expiry
- rollback after repeated corrected losses

