"""
Liquidity Sweep Scalper with execution-quality filters.
"""
import time
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .anti_starvation import anti_starvation
from .decision_logger import log_scalper_decision
from .indicators import atr, ema
from .session_filter import get_session_at
from .signal_quality import quality_score
from .strategies.base_strategy import BaseStrategy
from .tick_processor import TickProcessor, entry_pressure_block_reason
import config as cfg

_BE_TRIGGER = 0.30
_EARLY_FAIL_PTS = 0.20


def _relative_volume_ratio(df: pd.DataFrame, lookback: int = 8) -> float:
    if df is None or "volume" not in df or len(df) < max(3, lookback + 1):
        return 0.0
    recent = df["volume"].tail(lookback + 1).astype(float)
    baseline = float(recent.iloc[:-1].mean() or 0.0)
    if baseline <= 0:
        return 0.0
    return round(float(recent.iloc[-1] or 0.0) / baseline, 3)


def _marginal_setup_has_directional_support(
    direction: str,
    quality_score_value: float,
    ema20_slope: float,
    pressure_score: float,
) -> bool:
    marginal_quality_max = float(getattr(cfg, "SCALPER_MARGINAL_QUALITY_MAX", 0.60) or 0.0)
    if marginal_quality_max <= 0 or quality_score_value > marginal_quality_max:
        return True
    direction = str(direction or "").upper()
    if direction == "LONG":
        return pressure_score >= 0 or ema20_slope >= 0
    if direction == "SHORT":
        return pressure_score <= 0 or ema20_slope <= 0
    return True


def _ema_slope_alignment_ok(direction: str, ema15_slope: float, ema20_slope: float) -> bool:
    direction = str(direction or "").upper()
    ema15_min = float(getattr(cfg, "SCALPER_EMA15_SLOPE_MIN", 0.0) or 0.0)
    ema20_min = float(getattr(cfg, "SCALPER_EMA20_SLOPE_MIN", 0.0) or 0.0)
    if direction == "LONG":
        return ema15_slope >= ema15_min and ema20_slope > 0 and abs(ema20_slope) >= ema20_min
    if direction == "SHORT":
        return ema15_slope <= -ema15_min and ema20_slope < 0 and abs(ema20_slope) >= ema20_min
    return True


class SweepScalper(BaseStrategy):
    name = "SWEEP_SCALPER"

    def __init__(self):
        self.tick_proc = TickProcessor(buffer_size=500)
        self._session_trades = 0
        self._session_date = ""
        self._last_session = ""
        self._traded_levels: Dict[float, float] = {}
        self._last_candle_ts: float = 0.0  # timestamp of last traded M1 candle

    def generate_signal(self, data: Dict) -> Dict:
        tick = data.get("tick")
        if not tick:
            return _no("No tick")

        self.tick_proc.feed(tick)
        tick_snap = self.tick_proc.snapshot()

        m1 = data.get("m1_df")
        m5 = data.get("m5_df")
        m15 = data.get("m15_df")
        m30 = data.get("m30_df")
        regime = data.get("regime") or {}
        bias = data.get("bias") or {}
        tick_pressure = data.get("tick_pressure") or {}
        risk_manager = data.get("_risk_manager")
        strict_spec_mode = bool(getattr(cfg, "SCALPER_SPEC_STRICT_MODE", True))

        price = tick["bid"]
        spread = tick.get("spread", 0)
        spread_mean = tick_snap.get("spread_mean", spread)
        spread_std = tick_snap.get("spread_std", 0)

        def skip(reason: str,
                 setup_direction: Optional[str] = None,
                 quality_score_value: float = 0.0,
                 compression_ok: bool = False,
                 sweep_level: Optional[float] = None) -> Dict:
            log_scalper_decision(
                setup_direction=setup_direction,
                quality_score=quality_score_value,
                spread_mean=spread_mean,
                spread_std=spread_std,
                compression_ok=compression_ok,
                decision="TRADE_SKIPPED",
                reason=reason,
                price=price,
                sweep_level=sweep_level,
            )
            return {"signal": "NO_TRADE", "reason": reason, "score": quality_score_value}

        if m1 is None or len(m1) < 30:
            return skip("Insufficient M1 data")
        if m5 is None or len(m5) < 25:
            return skip("Insufficient M5 data")
        ms = data.get("market_state") or {}
        if bool(getattr(cfg, "SCALPER_M15_TREND_REQUIRED", True)):
            m15_filter = self._resolve_m15_trade_direction(m15, bias, ms)
            m15_trade_dir = str(m15_filter.get("direction") or "NEUTRAL").upper()
            if m15_trade_dir not in {"LONG", "SHORT"}:
                return skip(f"M15 trend unclear ({m15_filter.get('reason', 'no M15 edge')})")
        else:
            m15_filter = {"direction": "NEUTRAL", "reason": "disabled"}
            m15_trade_dir = "NEUTRAL"
        regime_is_ranging = str((regime or {}).get("state") or "").upper() == "RANGING"
        if risk_manager is not None:
            loss_cap = int(getattr(cfg, "SCALPER_MAX_CONSECUTIVE_LOSSES", 3) or 3)
            if int(getattr(risk_manager, "_consecutive_losses", 0) or 0) >= loss_cap:
                return skip(f"Paused after {loss_cap} consecutive losses")
        hard_spread_max = float(getattr(cfg, "SCALPER_HARD_SPREAD_MAX", 0.50) or 0.50)
        if hard_spread_max > 0 and spread > hard_spread_max:
            return skip(f"Spread {spread:.2f} > {hard_spread_max:.2f}")
        now = data.get("now_utc")
        if not isinstance(now, datetime):
            now = datetime.now(timezone.utc)
        elif now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        else:
            now = now.astimezone(timezone.utc)
        hour = now.hour
        session = get_session_at(now)
        volume_source = m5 if strict_spec_mode else m1
        volume_ratio = _relative_volume_ratio(volume_source)
        _vol_min = float(getattr(cfg, "SCALPER_MIN_ENTRY_VOLUME_RATIO", 0.0))
        if session == "ASIAN":
            _vol_min = max(_vol_min, float(getattr(cfg, "SCALPER_ASIAN_MIN_ENTRY_VOLUME_RATIO", _vol_min) or _vol_min))
        in_window = (
            cfg.TRADE_WINDOW_LONDON_START <= hour < cfg.TRADE_WINDOW_LONDON_END
            or cfg.TRADE_WINDOW_OVERLAP_START <= hour < cfg.TRADE_WINDOW_OVERLAP_END
        )
        if not in_window and not (cfg.SESSION_OVERRIDE_ENABLED or cfg.TIME_GATE_OVERRIDE_ENABLED or cfg.ALL_GATES_OVERRIDE_ENABLED):
            return skip(f"Outside trade window (UTC {hour}:xx)")

        today = now.strftime("%Y-%m-%d")
        if session != self._last_session or today != self._session_date:
            self._session_trades = 0
            self._last_session = session
            self._session_date = today

        if session == "ASIAN" and not bool(getattr(cfg, "SCALPER_ALLOW_ASIAN_SESSION", False)) and not cfg.ALL_GATES_OVERRIDE_ENABLED:
            return skip("ASIAN_SESSION_DISABLED")

        # If Asian trading is explicitly enabled, still keep the bar materially higher.
        if session == "ASIAN":
            closes = m1["close"].values.astype(float)
            highs  = m1["high"].values.astype(float)
            lows   = m1["low"].values.astype(float)
            _atr_check = atr(highs, lows, closes, 14)[-1]
            _asian_atr_min = float(getattr(cfg, "SCALPER_ASIAN_ATR_MIN", 1.5))
            if _atr_check < _asian_atr_min:
                return skip(f"ASIAN_LOW_LIQUIDITY: ATR {_atr_check:.3f} < {_asian_atr_min}")
            # Also block if EMA slope is too flat — drifting market, no momentum
            _ema = ema(closes, 20)
            _slope = abs(_ema[-1] - _ema[-3]) if len(_ema) >= 3 else 0
            _asian_slope_min = float(getattr(cfg, "SCALPER_ASIAN_EMA_SLOPE_MIN", 0.15))
            if _slope < _asian_slope_min:
                return skip(f"ASIAN_FLAT_MARKET: EMA slope {_slope:.3f} < {_asian_slope_min}")

        if int(getattr(cfg, "SCALPER_MAX_TRADES_SESSION", 0) or 0) > 0 and self._session_trades >= cfg.SCALPER_MAX_TRADES_SESSION:
            return skip(f"Max {cfg.SCALPER_MAX_TRADES_SESSION} trades this session")

        spread_ok, spread_reason = self._check_spread_quality(spread, tick_snap)
        if not spread_ok:
            return skip(f"Spread: {spread_reason}")

        closes = m1["close"].values.astype(float)
        highs = m1["high"].values.astype(float)
        lows = m1["low"].values.astype(float)
        opens = m1["open"].values.astype(float)
        atr_val = float((ms.get("atr") or {}).get("M1") or 0.0)
        if atr_val <= 0:
            atr_val = atr(highs, lows, closes, 14)[-1]
        if atr_val < cfg.SCALPER_ATR_MIN:
            return skip(f"ATR {atr_val:.3f} < {cfg.SCALPER_ATR_MIN}")
        if atr_val > cfg.SCALPER_ATR_MAX:
            return skip(f"ATR {atr_val:.3f} > {cfg.SCALPER_ATR_MAX}")

        # Block entries during RSI extremes
        _rsi_val = float((ms.get("rsi") or {}).get("M1") or 50.0)
        _rsi_oversold = float(getattr(cfg, "SCALPER_RSI_OVERSOLD_BLOCK", 25.0))
        _rsi_overbought = float(getattr(cfg, "SCALPER_RSI_OVERBOUGHT_BLOCK", 75.0))
        if _rsi_val <= _rsi_oversold:
            return skip(f"RSI oversold ({_rsi_val:.1f}) - no SHORT entries")
        if _rsi_val >= _rsi_overbought:
            return skip(f"RSI overbought ({_rsi_val:.1f}) - no LONG entries")

        # Block ATR spike entries
        _atr_ratio_max = float(getattr(cfg, "SCALPER_ATR_RATIO_MAX", 1.5))
        _atr_baseline = atr(highs, lows, closes, 50)[-1] if len(closes) >= 50 else atr_val
        _atr_ratio = atr_val / _atr_baseline if _atr_baseline > 0 else 1.0
        if _atr_ratio > _atr_ratio_max:
            return skip(f"ATR spike ({_atr_ratio:.2f}x baseline) > {_atr_ratio_max}x")

        compression_ok, comp_reason = self._check_compression_gate(m1, atr_val)
        if not compression_ok:
            return skip(f"Compression: {comp_reason}")

        ema_m1 = (ms.get("ema") or {}).get("M1_20") or {}
        ema20_val   = float(ema_m1.get("value") or 0.0) or float(ema(closes, 20)[-1])
        ema20_slope = float(ema_m1.get("slope") or 0.0)
        ema15 = ema(closes, 15)
        ema15_val   = float(ema15[-1])
        ema15_slope = float(ema15[-1] - ema15[-3]) if len(ema15) >= 3 else 0.0
        if abs(ema15_slope) < cfg.SCALPER_EMA15_SLOPE_MIN:
            return skip(f"EMA15 flat ({ema15_slope:.3f})")
        if abs(ema20_slope) < cfg.SCALPER_EMA20_SLOPE_MIN:
            return skip(f"EMA20 flat ({ema20_slope:.3f})")

        m5_candle = self._ltf_candle_stats(m5)

        setup_type = "SWEEP"
        setup_anchor_level = None
        if strict_spec_mode and not regime_is_ranging:
            sweep_df = m5
            sweep = self._detect_sweep(
                sweep_df["high"].values.astype(float),
                sweep_df["low"].values.astype(float),
                sweep_df["close"].values.astype(float),
                sweep_df["open"].values.astype(float),
            )
        else:
            sweep = self._detect_sweep(highs, lows, closes, opens)
        consolidation = range_edge = None
        if sweep:
            direction = sweep["direction"]
            sweep_level = sweep["level"]
            setup_anchor_level = sweep_level
        else:
            consolidation = self._detect_bias_consolidation_setup(
                m15_df=m15,
                m30_df=m30,
                bias=bias,
                regime=regime,
                price=price,
                spread=spread,
            )
            if consolidation:
                direction = consolidation["direction"]
                sweep_level = consolidation["anchor_level"]
                setup_anchor_level = sweep_level
                setup_type = "BIAS_CONSOLIDATION"
            else:
                range_edge = self._detect_range_edge_scalp_setup(
                    m15_df=m15,
                    m30_df=m30,
                    bias=bias,
                    regime=regime,
                    price=price,
                    spread=spread,
                )
                if range_edge:
                    direction = range_edge["direction"]
                    sweep_level = range_edge["anchor_level"]
                    setup_anchor_level = sweep_level
                    setup_type = "RANGE_EDGE"
                elif strict_spec_mode and regime_is_ranging:
                    return skip("No short-range or big-range scalp setup", compression_ok=True)
                elif strict_spec_mode:
                    return skip("No liquidity sweep detected", compression_ok=True)
                else:
                    return skip("No sweep, bias-consolidation, or range-edge setup", compression_ok=True)

        entry_vol_min = _vol_min
        if setup_type in {"BIAS_CONSOLIDATION", "RANGE_EDGE"}:
            entry_vol_min = min(entry_vol_min, float(getattr(cfg, "SCALPER_RANGE_MIN_ENTRY_VOLUME_RATIO", entry_vol_min) or entry_vol_min))
        if entry_vol_min > 0 and volume_ratio > 0 and volume_ratio < entry_vol_min:
            return skip(f"Volume ratio {volume_ratio:.2f} < {entry_vol_min:.2f}")

        if m15_trade_dir in {"LONG", "SHORT"} and direction != m15_trade_dir:
            return skip(
                f"M15 {m15_trade_dir} blocks {direction} setup ({m15_filter.get('reason', 'M15 first')})",
                setup_direction=direction,
                compression_ok=True,
                sweep_level=sweep_level,
            )

        sideways_range = None
        if not strict_spec_mode:
            sideways_range = self._validate_sideways_range_entry(
                direction=direction,
                price=price,
                spread=spread,
                regime=regime,
                m15_df=m15,
                m30_df=m30,
            )
            if sideways_range and not sideways_range.get("allowed", False):
                return skip(
                    str(sideways_range.get("reason") or "Sideways range entry blocked"),
                    setup_direction=direction,
                    compression_ok=True,
                    sweep_level=sweep_level,
                )

        if not _ema_slope_alignment_ok(direction, ema15_slope, ema20_slope):
            return skip(
                f"EMA slope misaligned for {direction} (EMA15 {ema15_slope:+.3f}, EMA20 {ema20_slope:+.3f})",
                setup_direction=direction,
                compression_ok=True,
                sweep_level=sweep_level,
            )

        # HTF bias alignment — only trade with the prevailing H4/H1 bias
        htf_bias = bias
        bias_dir = str(htf_bias.get("direction") or "NEUTRAL").upper()
        if bias_dir not in ("", "NEUTRAL"):
            if direction == "LONG" and bias_dir == "SHORT":
                return skip("HTF bias SHORT — no BUY sweep", setup_direction=direction,
                            compression_ok=True, sweep_level=sweep_level)
            if direction == "SHORT" and bias_dir == "LONG":
                return skip("HTF bias LONG — no SELL sweep", setup_direction=direction,
                            compression_ok=True, sweep_level=sweep_level)

        level_key = round(setup_anchor_level or sweep_level, 1)

        # One trade per M1 candle — prevents cluster entries on the same candle
        try:
            current_candle_ts = float(m1.iloc[-1]["datetime"].timestamp())
        except Exception:
            current_candle_ts = 0.0
        if current_candle_ts > 0 and current_candle_ts == self._last_candle_ts:
            return skip("ALREADY_TRADED_THIS_CANDLE", setup_direction=direction,
                        compression_ok=True, sweep_level=sweep_level)

        self._cleanup_levels()
        if level_key in self._traded_levels:
            return skip(f"Level {sweep_level:.2f} already traded", setup_direction=direction,
                        compression_ok=True, sweep_level=sweep_level)

        m1_aligned = self._check_timeframe_alignment(m1, direction)
        m5_aligned = self._check_timeframe_alignment(m5, direction) if m5 is not None else False

        last = len(closes) - 1
        body = abs(closes[last] - opens[last])
        candle_range = highs[last] - lows[last]
        body_ratio = body / candle_range if candle_range > 0 else 0
        ltf_body_ratio = float(m5_candle["body_ratio"]) if strict_spec_mode else body_ratio
        ltf_candle_range = float(m5_candle["range"]) if strict_spec_mode else candle_range
        ltf_close_position = str(m5_candle["close_position"]) if strict_spec_mode else ("HIGH" if closes[last] >= highs[last] - candle_range * 0.3 else "LOW")
        avg_body = np.abs(closes[last - 10:last] - opens[last - 10:last]).mean()
        body_vs_avg_min = float(getattr(cfg, "SCALPER_BODY_VS_AVG_MIN", 1.0) or 0.0)
        if setup_type == "SWEEP" and body < avg_body * body_vs_avg_min:
            return skip(f"Body {body:.3f} < avg*{body_vs_avg_min:.2f} ({avg_body * body_vs_avg_min:.3f})", setup_direction=direction,
                        compression_ok=True, sweep_level=sweep_level)
        if setup_type == "SWEEP" and direction == "LONG" and (closes[last] <= opens[last] or price < ema20_val):
            return skip("LONG displacement invalid", setup_direction=direction,
                        compression_ok=True, sweep_level=sweep_level)
        if setup_type == "SWEEP" and direction == "SHORT" and (closes[last] >= opens[last] or price > ema20_val):
            return skip("SHORT displacement invalid", setup_direction=direction,
                        compression_ok=True, sweep_level=sweep_level)
        required_body_ratio = cfg.SCALPER_BODY_RATIO_MIN
        if setup_type in {"BIAS_CONSOLIDATION", "RANGE_EDGE"}:
            required_body_ratio = float(
                getattr(cfg, "SCALPER_RANGE_BODY_RATIO_MIN", required_body_ratio) or required_body_ratio
            )
        if ltf_body_ratio < required_body_ratio:
            return skip(f"Body ratio {ltf_body_ratio:.2f} < {required_body_ratio:.2f}", setup_direction=direction,
                        compression_ok=True, sweep_level=sweep_level)

        candle_ok, candle_reason = self._setup_candle_confirmation_ok(
            setup_type=setup_type,
            direction=direction,
            close_position=ltf_close_position,
            last_open=float(m5.iloc[-1]["open"]) if strict_spec_mode and m5 is not None and len(m5) else float(opens[last]),
            last_close=float(m5.iloc[-1]["close"]) if strict_spec_mode and m5 is not None and len(m5) else float(closes[last]),
        )
        if strict_spec_mode and not candle_ok:
            return skip(candle_reason, setup_direction=direction,
                        compression_ok=True, sweep_level=sweep_level)

        if strict_spec_mode and setup_type == "SWEEP":
            m15_struct = bias.get("m15_structure") or {}
            if bool(m15_struct.get("bos")):
                if m15_trade_dir == "LONG" and direction != "LONG":
                    return skip("Possible exhaustion at highs", setup_direction=direction,
                                compression_ok=True, sweep_level=sweep_level)
                if m15_trade_dir == "SHORT" and direction != "SHORT":
                    return skip("Possible exhaustion at lows", setup_direction=direction,
                                compression_ok=True, sweep_level=sweep_level)

        if not tick_snap.get("ready"):
            return skip("Tick buffer not ready", setup_direction=direction,
                        compression_ok=True, sweep_level=sweep_level)
        pressure_score = float(tick_pressure.get("pressure_score", 0.0) or 0.0)
        pressure_bias = str(tick_pressure.get("directional_bias", "NEUTRAL") or "NEUTRAL")
        pressure_reason = entry_pressure_block_reason(
            direction,
            tick_pressure,
            strong_threshold=float(getattr(cfg, "TICK_PRESSURE_ENTRY_BLOCK_THRESHOLD", 0.35) or 0.35),
            short_positive_veto_threshold=float(getattr(cfg, "TICK_PRESSURE_SHORT_ENTRY_VETO_THRESHOLD", 0.05) or 0.05),
        )
        if pressure_reason:
            return skip(pressure_reason, setup_direction=direction, compression_ok=True, sweep_level=sweep_level)
        if tick_pressure.get("ready"):
            # Block neutral pressure - no directional conviction from order flow
            _require_pressure_alignment = bool(getattr(cfg, "SCALPER_REQUIRE_PRESSURE_ALIGNMENT", False))
            if _require_pressure_alignment and pressure_bias == "NEUTRAL":
                return skip("Tick pressure NEUTRAL - no order flow conviction", setup_direction=direction,
                            compression_ok=True, sweep_level=sweep_level)
            if session == "ASIAN" and bool(getattr(cfg, "SCALPER_ASIAN_REQUIRE_PRESSURE_ALIGNMENT", True)):
                if direction == "LONG" and pressure_score <= 0:
                    return skip("Asian LONG setup needs positive tick pressure", setup_direction=direction,
                                compression_ok=True, sweep_level=sweep_level)
                if direction == "SHORT" and pressure_score >= 0:
                    return skip("Asian SHORT setup needs negative tick pressure", setup_direction=direction,
                                compression_ok=True, sweep_level=sweep_level)
        eq_ok, eq_reason = self.tick_proc.check_execution_quality()
        if not eq_ok and not (
            cfg.ALL_GATES_OVERRIDE_ENABLED
            or cfg.EXECUTION_GATE_OVERRIDE_ENABLED
            or cfg.EXECUTION_QUALITY_GATE_OVERRIDE_ENABLED
        ):
            return skip(f"Exec quality: {eq_reason}", setup_direction=direction,
                        compression_ok=True, sweep_level=sweep_level)

        relaxed_params = anti_starvation.get_relaxed_params()
        tick_threshold = cfg.SCALPER_TICK_DIR_THRESHOLD
        if relaxed_params["active"] and relaxed_params["type"] == "tick_ratio":
            tick_threshold = relaxed_params["tick_ratio_threshold"]
        if session == "ASIAN":
            tick_threshold = max(
                tick_threshold,
                float(getattr(cfg, "SCALPER_ASIAN_TICK_DIR_THRESHOLD", tick_threshold) or tick_threshold),
            )

        dir_pct = tick_snap.get("dir_pct", 0.5)
        directional_ok = (
            (direction == "LONG" and dir_pct >= tick_threshold)
            or (direction == "SHORT" and dir_pct <= (1.0 - tick_threshold))
        )
        if not directional_ok and not (
            cfg.ALL_GATES_OVERRIDE_ENABLED
            or cfg.EXECUTION_GATE_OVERRIDE_ENABLED
            or cfg.TICK_DIRECTION_GATE_OVERRIDE_ENABLED
        ):
            target = tick_threshold if direction == "LONG" else (1.0 - tick_threshold)
            return skip(
                f"Tick direction {dir_pct:.0%} not aligned with {direction} threshold {target:.0%}",
                setup_direction=direction,
                compression_ok=True,
                sweep_level=sweep_level,
            )

        q_score, q_reasons = self._quality_score(
            direction=direction,
            body_ratio=ltf_body_ratio,
            dir_pct=dir_pct,
            m1_aligned=m1_aligned,
            m5_aligned=m5_aligned,
            compression_ok=compression_ok,
            tick_snap=tick_snap,
        )
        if q_score < cfg.SCALPER_QUALITY_THRESHOLD:
            return skip(
                f"Quality {q_score:.0%} < {cfg.SCALPER_QUALITY_THRESHOLD:.0%}",
                setup_direction=direction,
                quality_score_value=q_score,
                compression_ok=compression_ok,
                sweep_level=sweep_level,
            )
        if session == "ASIAN":
            asian_quality_threshold = max(
                cfg.SCALPER_QUALITY_THRESHOLD,
                float(getattr(cfg, "SCALPER_ASIAN_QUALITY_THRESHOLD", cfg.SCALPER_QUALITY_THRESHOLD) or cfg.SCALPER_QUALITY_THRESHOLD),
            )
            if q_score < asian_quality_threshold:
                return skip(
                    f"Asian quality {q_score:.0%} < {asian_quality_threshold:.0%}",
                    setup_direction=direction,
                    quality_score_value=q_score,
                    compression_ok=compression_ok,
                    sweep_level=sweep_level,
                )
        if not _marginal_setup_has_directional_support(direction, q_score, ema20_slope, pressure_score):
            return skip(
                f"Marginal {direction} lacks directional support (pressure {pressure_score:+.2f}, EMA20 slope {ema20_slope:+.3f})",
                setup_direction=direction,
                quality_score_value=q_score,
                compression_ok=compression_ok,
                sweep_level=sweep_level,
            )

        signal = "BUY" if direction == "LONG" else "SELL"
        if setup_type == "BIAS_CONSOLIDATION":
            sl, tp, sl_dist = self._compute_bias_consolidation_sl_tp(
                price=price,
                signal=signal,
                anchor_level=float(consolidation["anchor_level"]),
                range_low=float(consolidation["range_low"]),
                range_high=float(consolidation["range_high"]),
                spread=spread,
                atr_val=float(consolidation["atr15"]),
            )
        elif setup_type == "RANGE_EDGE":
            sl, tp, sl_dist = self._compute_range_edge_sl_tp(
                price=price,
                signal=signal,
                anchor_level=float(range_edge["anchor_level"]),
                range_low=float(range_edge["range_low"]),
                range_high=float(range_edge["range_high"]),
                spread=spread,
                atr_val=float(range_edge["atr_ref"]),
            )
        elif strict_spec_mode:
            sl, tp, sl_dist = self._compute_micro_trade_plan(
                price=price,
                direction=direction,
                body_ratio=ltf_body_ratio,
                volume_ratio=volume_ratio,
                candle_range=ltf_candle_range,
                spread=spread,
            )
        else:
            sl, tp, sl_dist = self._compute_cost_aware_sl_tp(price, signal, atr_val, spread, sweep_level)
        if sl is None or tp is None:
            return skip("Cannot compute valid SL/TP", setup_direction=direction,
                        quality_score_value=q_score, compression_ok=compression_ok,
                        sweep_level=sweep_level)

        tp_dist = abs(tp - price)
        rr = tp_dist / sl_dist if sl_dist > 0 else 0
        if (
            self.tick_proc.spread_changed(spread, max_delta=cfg.SCALPER_CURRENT_SPREAD_DELTA_MAX)
            and not (
                cfg.ALL_GATES_OVERRIDE_ENABLED
                or cfg.SPREAD_GATE_OVERRIDE_ENABLED
                or cfg.SPREAD_DELTA_GATE_OVERRIDE_ENABLED
                or cfg.POST_SIGNAL_SPREAD_GATE_OVERRIDE_ENABLED
            )
        ):
            return skip("Spread widened since signal", setup_direction=direction,
                        quality_score_value=q_score, compression_ok=compression_ok,
                        sweep_level=sweep_level)

        reasons = [
            f"{('Sweep' if setup_type == 'SWEEP' else 'Bias consolidation' if setup_type == 'BIAS_CONSOLIDATION' else 'Range edge')} {direction} @ {sweep_level:.2f}",
            f"M15 {m15_trade_dir} ({m15_filter.get('reason', 'M15 first')})",
            f"M5 body {ltf_body_ratio:.0%}",
            f"M5 close {ltf_close_position}",
            f"EMA15 {ema15_slope:+.3f}",
            f"EMA20 {ema20_slope:+.3f}",
            f"Vel {tick_snap.get('velocity', 0):.1f}",
            f"Press {pressure_score:+.2f}",
            f"Vol x{volume_ratio:.2f}",
            f"ATR {atr_val:.3f}",
            f"SL:{sl_dist:.2f} TP:{tp_dist:.2f}",
            f"[Q:{q_score:.0%}]",
        ] + q_reasons
        if consolidation:
            reasons.append(
                f"{consolidation.get('label', 'M15 box')} {consolidation['range_low']:.2f}-{consolidation['range_high']:.2f}"
            )
            reasons.append(
                f"TP target ${consolidation['target_profit']:.2f}"
            )
        elif range_edge:
            reasons.append(
                f"{range_edge.get('label', 'Wide range')} {range_edge['range_low']:.2f}-{range_edge['range_high']:.2f}"
            )
            reasons.append(
                f"TP target ${range_edge['target_profit']:.2f}"
            )
        elif sideways_range and sideways_range.get("allowed"):
            reasons.append(
                f"{sideways_range['label']} {sideways_range['range_low']:.2f}-{sideways_range['range_high']:.2f}"
            )

        signal_result = {
            "signal": signal,
            "entry": price,
            "sl": sl,
            "tp": tp,
            "sl_distance": sl_dist,
            "confidence": q_score,
            "reason": " | ".join(reasons),
            "reasons": reasons,
            "rr": round(rr, 2),
            "sweep_level": sweep_level,
            "indicators": {
                "ema15": round(ema15_val, 2),
                "ema15_slope": round(ema15_slope, 4),
                "ema20": round(ema20_val, 2),
                "ema20_slope": round(ema20_slope, 4),
                "m15_trade_direction": m15_trade_dir,
                "m5_body_ratio": round(ltf_body_ratio, 4),
                "m5_close_position": ltf_close_position,
                "atr14": round(atr_val, 4),
                "spread": spread,
                "spread_mean": spread_mean,
                "tick_velocity": tick_snap.get("velocity", 0),
                "volume_ratio": volume_ratio,
                "tick_pressure_score": pressure_score,
                "tick_pressure_bias": pressure_bias,
            },
            "_scalp": True,
            "_signal_family": "SWEEP",
            "_sweep_confirmed": True,
            "_candle_confirmation": True,
            "_scalper_setup_type": setup_type,
            "_exit_profile": "scalp",
            "_be_trigger": _BE_TRIGGER,
            "_be_trigger_r": round(_BE_TRIGGER / sl_dist, 4) if sl_dist > 0 else 0.0,
            "_breakeven_min_hold_seconds": 0 if strict_spec_mode else cfg.EXIT_PROFILE_SCALP_BREAKEVEN_MIN_HOLD_SECONDS,
            "_breakeven_volume_hold_ratio": 0.0 if strict_spec_mode else cfg.EXIT_PROFILE_SCALP_BREAKEVEN_VOLUME_HOLD_RATIO,
            "_timeout": int(
                getattr(cfg, "SCALPER_RANGE_TIMEOUT_SECONDS", cfg.EXIT_PROFILE_SCALP_TIMEOUT_SECONDS)
                if setup_type in {"BIAS_CONSOLIDATION", "RANGE_EDGE"}
                else cfg.EXIT_PROFILE_SCALP_TIMEOUT_SECONDS
            ),
            "_timeout_min_progress_r": 0.0 if setup_type in {"BIAS_CONSOLIDATION", "RANGE_EDGE"} else cfg.EXIT_PROFILE_SCALP_TIMEOUT_MIN_PROGRESS_R,
            "_min_hold_seconds": cfg.EXIT_PROFILE_SCALP_MIN_HOLD_SECONDS,
            "_early_fail": 0.0 if setup_type in {"BIAS_CONSOLIDATION", "RANGE_EDGE"} else _EARLY_FAIL_PTS,
            "_tier1_min_ticks": cfg.EXIT_PROFILE_SCALP_EARLY_FAIL_MIN_TICKS,
            "_tier1_max_ticks": cfg.EXIT_PROFILE_SCALP_EARLY_FAIL_MAX_TICKS,
            "_reversal_arm_r": cfg.EXIT_PROFILE_SCALP_REVERSAL_ARM_R,
            "_reversal_drawdown_pct": cfg.EXIT_PROFILE_SCALP_REVERSAL_DRAWDOWN_PCT,
            "_reversal_floor_r": cfg.EXIT_PROFILE_SCALP_REVERSAL_FLOOR_R,
            "_trail_activate_r": cfg.EXIT_PROFILE_SCALP_TRAIL_ACTIVATE_R,
            "_trail_lock_r": cfg.EXIT_PROFILE_SCALP_TRAIL_LOCK_R,
            "_velocity_drop_enabled": False if setup_type in {"BIAS_CONSOLIDATION", "RANGE_EDGE"} else cfg.EXIT_PROFILE_SCALP_VELOCITY_DROP_ENABLED,
            "_setup_direction": direction,
            "_quality_score": q_score,
            "_threshold": cfg.SCALPER_QUALITY_THRESHOLD,
            "_entry_tick_velocity": tick_snap.get("velocity", 0),
            "_entry_tick_pressure_score": pressure_score,
            "_entry_tick_pressure_bias": pressure_bias,
            "_entry_volume_ratio": volume_ratio,
            "_entry_volume_strong": volume_ratio >= float(getattr(cfg, "SCALPER_MIN_ENTRY_VOLUME_RATIO", 1.0) or 1.0),
            "_entry_spread": spread,
        }
        if consolidation:
            signal_result["_bias_consolidation"] = {
                "label": consolidation.get("label", "M15 box"),
                "range_low": consolidation["range_low"],
                "range_high": consolidation["range_high"],
                "target_profit": consolidation["target_profit"],
                "atr15": consolidation["atr15"],
            }
        if range_edge:
            signal_result["_range_edge"] = {
                "label": range_edge.get("label", "Wide range"),
                "range_low": range_edge["range_low"],
                "range_high": range_edge["range_high"],
                "target_profit": range_edge["target_profit"],
                "atr_ref": range_edge["atr_ref"],
            }
        if sideways_range and sideways_range.get("allowed"):
            signal_result["_sideways_range"] = {
                "label": sideways_range["label"],
                "range_low": sideways_range["range_low"],
                "range_high": sideways_range["range_high"],
                "entry_edge_buffer": sideways_range["entry_edge_buffer"],
            }

        log_scalper_decision(
            setup_direction=direction,
            quality_score=q_score,
            spread_mean=spread_mean,
            spread_std=spread_std,
            compression_ok=compression_ok,
            decision="TRADE_TAKEN",
            reason=" | ".join(reasons),
            price=price,
            sweep_level=sweep_level,
            signal_data=signal_result,
        )
        return signal_result

    def confirm_trade_executed(self, sweep_level: float, candle_ts: float = 0.0):
        """Call this only after MT5 confirms the order. Locks the level and increments session count."""
        level_key = round(sweep_level, 1)
        self._traded_levels[level_key] = time.time()
        self._session_trades += 1
        if candle_ts > 0:
            self._last_candle_ts = candle_ts

    def _resolve_m15_trade_direction(self, m15_df: pd.DataFrame, bias: Dict, ms: Dict = None) -> Dict:
        if m15_df is None or len(m15_df) < 25:
            return {"direction": "NEUTRAL", "reason": "M15 unavailable"}

        bias = bias or {}
        m15_struct = bias.get("m15_structure") or {}
        pullback   = bias.get("m15_pullback") or {}

        pullback_dir = str(pullback.get("direction") or "").upper()
        if bool(pullback.get("active")) and pullback_dir in {"LONG", "SHORT"}:
            return {"direction": pullback_dir, "reason": "M15 pullback"}

        struct_dir = str(m15_struct.get("bos_direction") or "").upper()
        if struct_dir in {"LONG", "SHORT"}:
            return {"direction": struct_dir, "reason": f"M15 BOS {struct_dir}"}

        pattern = str(m15_struct.get("pattern") or "").upper()
        if pattern == "BULLISH": return {"direction": "LONG",  "reason": "M15 bullish structure"}
        if pattern == "BEARISH": return {"direction": "SHORT", "reason": "M15 bearish structure"}

        ema_snap  = ((ms or {}).get("ema") or {}).get("M15_20") or {}
        ema20_val = float(ema_snap.get("value") or 0.0)
        slope     = float(ema_snap.get("slope") or 0.0)
        if ema20_val <= 0:
            closes    = m15_df["close"].values.astype(float)
            ema20_arr = ema(closes, 20)
            if len(ema20_arr) < 3:
                return {"direction": "NEUTRAL", "reason": "M15 EMA unavailable"}
            slope     = float(ema20_arr[-1] - ema20_arr[-3])
            ema20_val = float(ema20_arr[-1])

        price     = float(m15_df["close"].iloc[-1])
        slope_min = float(getattr(cfg, "SCALPER_M15_SLOPE_MIN", 0.15) or 0.15)
        if slope >= slope_min  and price >= ema20_val:
            return {"direction": "LONG",  "reason": f"M15 EMA slope +{slope:.2f}"}
        if slope <= -slope_min and price <= ema20_val:
            return {"direction": "SHORT", "reason": f"M15 EMA slope {slope:.2f}"}
        return {"direction": "NEUTRAL", "reason": f"M15 mixed (slope {slope:+.2f})"}

    def _ltf_candle_stats(self, df: pd.DataFrame) -> Dict:
        if df is None or len(df) < 1:
            return {"body_ratio": 0.0, "range": 0.0, "close_position": "MID"}
        candle = df.iloc[-1]
        high = float(candle["high"])
        low = float(candle["low"])
        open_price = float(candle["open"])
        close = float(candle["close"])
        candle_range = max(0.0, high - low)
        body = abs(close - open_price)
        body_ratio = (body / candle_range) if candle_range > 0 else 0.0
        close_pct = ((close - low) / candle_range) if candle_range > 0 else 0.5
        if close_pct >= 0.70:
            close_position = "HIGH"
        elif close_pct <= 0.30:
            close_position = "LOW"
        else:
            close_position = "MID"
        return {
            "body_ratio": round(body_ratio, 4),
            "range": round(candle_range, 4),
            "close_position": close_position,
        }

    def _setup_candle_confirmation_ok(
        self,
        *,
        setup_type: str,
        direction: str,
        close_position: str,
        last_open: float,
        last_close: float,
    ) -> Tuple[bool, str]:
        direction = str(direction or "").upper()
        setup_type = str(setup_type or "").upper()
        close_position = str(close_position or "MID").upper()
        bullish_close = last_close > last_open
        bearish_close = last_close < last_open

        if setup_type == "SWEEP":
            if direction == "LONG" and close_position != "HIGH":
                return False, "BUY needs M5 close near candle high"
            if direction == "SHORT" and close_position != "LOW":
                return False, "SELL needs M5 close near candle low"
            return True, ""

        if direction == "LONG":
            if not bullish_close:
                return False, "Range LONG needs bullish M5 close"
            if close_position == "LOW":
                return False, "Range LONG cannot close at candle low"
            return True, ""

        if not bearish_close:
            return False, "Range SHORT needs bearish M5 close"
        if close_position == "HIGH":
            return False, "Range SHORT cannot close at candle high"
        return True, ""

    def _compute_micro_trade_plan(
        self,
        *,
        price: float,
        direction: str,
        body_ratio: float,
        volume_ratio: float,
        candle_range: float,
        spread: float,
    ) -> tuple:
        sl_min = float(getattr(cfg, "SCALPER_MICRO_SL_MIN", 0.50) or 0.50)
        sl_max = float(getattr(cfg, "SCALPER_MICRO_SL_MAX", 0.70) or 0.70)
        tp_min = float(getattr(cfg, "SCALPER_MICRO_TP_MIN", 0.50) or 0.50)
        tp_max = float(getattr(cfg, "SCALPER_MICRO_TP_MAX", 1.00) or 1.00)

        sl_dist = max(sl_min, spread * 3.0, candle_range * 0.55)
        sl_dist = round(min(sl_max, sl_dist), 2)

        tp_dist = tp_min
        if body_ratio >= 0.75 and volume_ratio >= 1.5:
            tp_dist = 1.00
        elif body_ratio >= 0.68 and volume_ratio >= 1.2:
            tp_dist = 0.80
        tp_dist = round(min(tp_max, max(tp_min, tp_dist)), 2)

        if direction == "LONG":
            sl = round(price - sl_dist, 2)
            tp = round(price + tp_dist, 2)
        else:
            sl = round(price + sl_dist, 2)
            tp = round(price - tp_dist, 2)
        return sl, tp, sl_dist

    def _quality_score(
        self,
        direction: str,
        body_ratio: float,
        dir_pct: float,
        m1_aligned: bool,
        m5_aligned: bool,
        compression_ok: bool,
        tick_snap: Dict,
    ) -> Tuple[float, list]:
        relaxed_params = anti_starvation.get_relaxed_params()
        tick_threshold = cfg.SCALPER_TICK_DIR_THRESHOLD
        if relaxed_params["active"] and relaxed_params["type"] == "tick_ratio":
            tick_threshold = relaxed_params["tick_ratio_threshold"]
        score, reasons = quality_score(
            direction=direction,
            sweep_present=True,
            body_ratio=body_ratio,
            tick_ratio=dir_pct,
            tick_velocity_increasing=tick_snap.get("vel_increasing", False),
            m1_aligned=m1_aligned,
            m5_aligned=m5_aligned,
            compression_ok=compression_ok,
            tick_ratio_threshold=tick_threshold,
        )
        if relaxed_params["active"] and relaxed_params["type"] == "tick_ratio":
            reasons.append(f"Anti-starvation tick ratio {tick_threshold:.0%}")
        return score, reasons

    def _check_timeframe_alignment(self, df: pd.DataFrame, direction: str) -> bool:
        if df is None or len(df) < 25:
            return False

        closes = df["close"].values.astype(float)
        ema20 = ema(closes, 20)
        price = closes[-1]
        ema_val = ema20[-1]
        slope = ema20[-1] - ema20[-3] if len(ema20) >= 3 else 0
        if direction == "LONG":
            return price >= ema_val and slope > 0.1
        return price <= ema_val and slope < -0.1

    def _check_spread_quality(self, spread: float, tick_snap: Dict) -> Tuple[bool, str]:
        if cfg.ALL_GATES_OVERRIDE_ENABLED or cfg.SPREAD_GATE_OVERRIDE_ENABLED:
            return True, "Spread gate overridden"

        relaxed_params = anti_starvation.get_relaxed_params()
        spread_limit = cfg.SCALPER_SPREAD_MEAN_MAX
        if relaxed_params["active"] and relaxed_params["type"] == "spread_tolerance":
            spread_limit = min(spread_limit + relaxed_params["spread_tolerance_bonus"], cfg.SCALPER_SPREAD_MEAN_MAX)

        if not tick_snap.get("ready"):
            if spread >= spread_limit and not cfg.SPREAD_MEAN_GATE_OVERRIDE_ENABLED:
                return False, f"Spread {spread:.3f} >= {spread_limit:.3f}"
            return True, "OK"

        spread_mean = tick_snap.get("spread_mean", spread)
        spread_std = tick_snap.get("spread_std", 0)
        spread_pctl = tick_snap.get("spread_pctl", 0.5)
        if spread_mean >= spread_limit and not cfg.SPREAD_MEAN_GATE_OVERRIDE_ENABLED:
            return False, f"Mean spread {spread_mean:.3f} >= {spread_limit:.3f}"
        if spread_std > cfg.SCALPER_SPREAD_STD_MAX and not cfg.SPREAD_VOLATILITY_GATE_OVERRIDE_ENABLED:
            return False, f"Spread volatility {spread_std:.3f} > {cfg.SCALPER_SPREAD_STD_MAX:.3f}"
        if spread_pctl > cfg.SCALPER_SPREAD_PERCENTILE_MAX and not cfg.SPREAD_PERCENTILE_GATE_OVERRIDE_ENABLED:
            return False, f"Spread percentile {spread_pctl:.0%} > {cfg.SCALPER_SPREAD_PERCENTILE_MAX:.0%}"
        if spread > spread_mean + cfg.SCALPER_CURRENT_SPREAD_DELTA_MAX and not cfg.SPREAD_DELTA_GATE_OVERRIDE_ENABLED:
            return False, f"Current spread {spread:.3f} > baseline+{cfg.SCALPER_CURRENT_SPREAD_DELTA_MAX:.3f}"
        return True, "Spread OK"

    def _check_compression_gate(self, m1: pd.DataFrame, atr_val: float) -> Tuple[bool, str]:
        if cfg.ALL_GATES_OVERRIDE_ENABLED or cfg.COMPRESSION_GATE_OVERRIDE_ENABLED:
            return True, "Compression gate overridden"

        lookback = max(2, int(cfg.COMPRESSION_RANGE_LOOKBACK))
        if m1 is None or len(m1) < lookback + 7:
            return False, "Insufficient M1 data for compression check"

        recent = m1.iloc[-lookback:]
        range_n = float(recent["high"].max() - recent["low"].min())
        atr_threshold = cfg.COMPRESSION_ATR_MULTIPLIER * atr_val
        if range_n >= atr_threshold and not cfg.COMPRESSION_RANGE_GATE_OVERRIDE_ENABLED:
            return False, f"Range{lookback} {range_n:.1f} >= {atr_threshold:.1f} (too wide)"

        highs = m1["high"].values.astype(float)
        lows = m1["low"].values.astype(float)
        closes = m1["close"].values.astype(float)
        atr_series = atr(highs, lows, closes, 14)
        atr_prev = atr_series[-4] if len(atr_series) >= 4 else atr_series[-1]
        if atr_series[-1] <= atr_prev and not cfg.ATR_RISING_GATE_OVERRIDE_ENABLED:
            return False, f"ATR not rising ({atr_series[-1]:.3f} <= {atr_prev:.3f})"
        return True, f"Compression OK (R{lookback}:{range_n:.1f} < {atr_threshold:.1f}, ATR+)"

    def _compute_cost_aware_sl_tp(
        self,
        price: float,
        signal: str,
        atr_val: float,
        spread: float,
        sweep_level: float,
    ) -> tuple:
        # SL: place below/above the sweep level with a buffer, minimum 1.0 * ATR from entry
        atr_sl_min = max(0.80, atr_val * 0.8)
        if signal == "BUY":
            # SL must be below sweep level (the liquidity that was taken)
            sl_from_sweep = round(sweep_level - 0.20, 2)
            sl_from_atr   = round(price - atr_sl_min, 2)
            sl = min(sl_from_sweep, sl_from_atr)  # further from price = safer
            sl_dist = round(price - sl, 2)
            # TP: minimum RR 1.5, target 2.0
            tp_dist = max(sl_dist * 1.5, atr_val * 1.0, spread * 3.0)
            tp = round(price + tp_dist, 2)
        else:
            sl_from_sweep = round(sweep_level + 0.20, 2)
            sl_from_atr   = round(price + atr_sl_min, 2)
            sl = max(sl_from_sweep, sl_from_atr)  # further from price = safer
            sl_dist = round(sl - price, 2)
            tp_dist = max(sl_dist * 1.5, atr_val * 1.0, spread * 3.0)
            tp = round(price - tp_dist, 2)

        sl_dist = round(abs(sl_dist), 2)
        if sl_dist < 0.5:
            return None, None, 0
        return sl, tp, sl_dist

    def _build_consolidation_box(
        self,
        df: pd.DataFrame,
        *,
        lookback: int,
    ) -> Optional[Dict]:
        if df is None or len(df) < max(20, lookback + 16):
            return None

        last = df.tail(lookback)
        highs = df["high"].values.astype(float)
        lows = df["low"].values.astype(float)
        closes = df["close"].values.astype(float)
        candle_ranges = (last["high"].astype(float) - last["low"].astype(float)).values
        close_std = float(last["close"].astype(float).std() or 0.0)
        range_high = float(last["high"].max())
        range_low = float(last["low"].min())
        box_range = range_high - range_low

        atr_val = float(atr(highs, lows, closes, 14)[-1])
        if atr_val <= 0:
            return None

        if box_range > atr_val * float(getattr(cfg, "SCALPER_BIAS_CONSOLIDATION_RANGE_ATR_MAX", 1.25) or 1.25):
            return None
        if max(candle_ranges) > atr_val * float(getattr(cfg, "SCALPER_BIAS_CONSOLIDATION_CANDLE_RANGE_ATR_MAX", 0.55) or 0.55):
            return None
        if close_std > atr_val * float(getattr(cfg, "SCALPER_BIAS_CONSOLIDATION_CLOSE_STD_ATR_MAX", 0.35) or 0.35):
            return None

        ema20 = ema(closes, 20)
        ema20_val = float(ema20[-1])
        ema20_slope = float(ema20[-1] - ema20[-3]) if len(ema20) >= 3 else 0.0
        return {
            "range_low": round(range_low, 2),
            "range_high": round(range_high, 2),
            "box_range": round(box_range, 4),
            "atr": round(atr_val, 4),
            "ema20": round(ema20_val, 4),
            "ema20_slope": round(ema20_slope, 4),
        }

    def _merge_range_boxes(
        self,
        m15_box: Optional[Dict],
        m30_box: Optional[Dict],
    ) -> Optional[Dict]:
        if not m15_box:
            return None
        if not m30_box:
            return {
                "label": "M15 box",
                "range_low": float(m15_box["range_low"]),
                "range_high": float(m15_box["range_high"]),
                "atr_ref": float(m15_box["atr"]),
            }

        overlap_low = max(float(m15_box["range_low"]), float(m30_box["range_low"]))
        overlap_high = min(float(m15_box["range_high"]), float(m30_box["range_high"]))
        if overlap_high <= overlap_low:
            return None

        overlap_range = overlap_high - overlap_low
        smaller_box = min(float(m15_box["box_range"]), float(m30_box["box_range"]))
        overlap_min = float(getattr(cfg, "SCALPER_SIDEWAYS_RANGE_OVERLAP_MIN", 0.35) or 0.35)
        if smaller_box <= 0 or (overlap_range / smaller_box) < overlap_min:
            return None

        return {
            "label": "M15/M30 box",
            "range_low": round(overlap_low, 2),
            "range_high": round(overlap_high, 2),
            "atr_ref": round(min(float(m15_box["atr"]), float(m30_box["atr"])), 4),
        }

    def _validate_sideways_range_entry(
        self,
        *,
        direction: str,
        price: float,
        spread: float,
        regime: Dict,
        m15_df: pd.DataFrame,
        m30_df: pd.DataFrame,
    ) -> Optional[Dict]:
        if not bool(getattr(cfg, "SCALPER_SIDEWAYS_RANGE_CONFIRM_ENABLED", True)):
            return None
        if str((regime or {}).get("state") or "").upper() != "RANGING":
            return None

        lookback = max(4, int(getattr(cfg, "SCALPER_BIAS_CONSOLIDATION_LOOKBACK_CANDLES", 4) or 4))
        m15_box = self._build_consolidation_box(m15_df, lookback=lookback)
        if not m15_box:
            return {"allowed": False, "reason": "RANGING without steady M15 box"}

        m30_box = self._build_consolidation_box(m30_df, lookback=lookback)
        if bool(getattr(cfg, "SCALPER_SIDEWAYS_RANGE_M30_REQUIRED", True)) and not m30_box:
            return {"allowed": False, "reason": "RANGING requires steady M30 confirmation"}

        merged_box = self._merge_range_boxes(m15_box, m30_box)
        if not merged_box:
            return {"allowed": False, "reason": "M15/M30 range boxes do not overlap cleanly"}

        edge_buffer = max(
            spread * 2.0,
            float(merged_box["atr_ref"]) * float(getattr(cfg, "SCALPER_BIAS_CONSOLIDATION_ENTRY_EDGE_ATR", 0.18) or 0.18),
        )

        direction = str(direction or "").upper()
        if direction == "LONG":
            if price > (float(merged_box["range_low"]) + edge_buffer):
                return {
                    "allowed": False,
                    "reason": (
                        f"RANGING LONG must enter near range low "
                        f"({merged_box['label']} {merged_box['range_low']:.2f}-{merged_box['range_high']:.2f}, "
                        f"price {price:.2f})"
                    ),
                }
        elif direction == "SHORT":
            if price < (float(merged_box["range_high"]) - edge_buffer):
                return {
                    "allowed": False,
                    "reason": (
                        f"RANGING SHORT must enter near range high "
                        f"({merged_box['label']} {merged_box['range_low']:.2f}-{merged_box['range_high']:.2f}, "
                        f"price {price:.2f})"
                    ),
                }

        return {
            "allowed": True,
            "label": str(merged_box["label"]),
            "range_low": float(merged_box["range_low"]),
            "range_high": float(merged_box["range_high"]),
            "entry_edge_buffer": round(edge_buffer, 4),
        }

    def _compute_bias_consolidation_sl_tp(
        self,
        *,
        price: float,
        signal: str,
        anchor_level: float,
        range_low: float,
        range_high: float,
        spread: float,
        atr_val: float,
    ) -> tuple:
        tp_target = float(getattr(cfg, "SCALPER_BIAS_CONSOLIDATION_TP_MIN", 1.0))
        tp_target_max = float(getattr(cfg, "SCALPER_BIAS_CONSOLIDATION_TP_MAX", 2.0))
        if atr_val >= 1.8 or abs(range_high - range_low) >= 1.0:
            tp_target = tp_target_max

        sl_buffer = float(getattr(cfg, "SCALPER_BIAS_CONSOLIDATION_SL_BUFFER", 0.25))
        floor_risk = max(0.60, abs(range_high - range_low) * 0.35, spread * 3.0)

        if signal == "BUY":
            sl = min(anchor_level - sl_buffer, price - floor_risk)
            tp = round(price + tp_target, 2)
            sl = round(sl, 2)
            sl_dist = round(price - sl, 2)
        else:
            sl = max(anchor_level + sl_buffer, price + floor_risk)
            tp = round(price - tp_target, 2)
            sl = round(sl, 2)
            sl_dist = round(sl - price, 2)

        if sl_dist < 0.5:
            return None, None, 0
        return sl, tp, sl_dist

    def _compute_range_edge_sl_tp(
        self,
        *,
        price: float,
        signal: str,
        anchor_level: float,
        range_low: float,
        range_high: float,
        spread: float,
        atr_val: float,
    ) -> tuple:
        tp_target = float(getattr(cfg, "SCALPER_RANGE_EDGE_TP_MIN", 1.0))
        tp_target_max = float(getattr(cfg, "SCALPER_RANGE_EDGE_TP_MAX", 1.8))
        if atr_val >= 2.2 or abs(range_high - range_low) >= 2.0:
            tp_target = tp_target_max

        sl_buffer = max(float(getattr(cfg, "SCALPER_BIAS_CONSOLIDATION_SL_BUFFER", 0.25)), spread * 2.0)
        floor_risk = max(0.65, abs(range_high - range_low) * 0.28, spread * 3.5)

        if signal == "BUY":
            sl = min(anchor_level - sl_buffer, price - floor_risk)
            tp = round(price + tp_target, 2)
            sl = round(sl, 2)
            sl_dist = round(price - sl, 2)
        else:
            sl = max(anchor_level + sl_buffer, price + floor_risk)
            tp = round(price - tp_target, 2)
            sl = round(sl, 2)
            sl_dist = round(sl - price, 2)

        if sl_dist < 0.5:
            return None, None, 0
        return sl, tp, sl_dist

    def _detect_bias_consolidation_setup(
        self,
        *,
        m15_df: pd.DataFrame,
        m30_df: pd.DataFrame,
        bias: Dict,
        regime: Dict,
        price: float,
        spread: float,
    ) -> Optional[Dict]:
        if not bool(getattr(cfg, "SCALPER_BIAS_CONSOLIDATION_ENABLED", True)):
            return None
        if m15_df is None:
            return None

        lookback = max(4, int(getattr(cfg, "SCALPER_BIAS_CONSOLIDATION_LOOKBACK_CANDLES", 4) or 4))
        if len(m15_df) < max(20, lookback + 16):
            return None

        bias_dir = str((bias or {}).get("direction") or "NEUTRAL").upper()
        if bias_dir not in {"LONG", "SHORT"}:
            return None

        closes = m15_df["close"].values.astype(float)
        m15_box = self._build_consolidation_box(m15_df, lookback=lookback)
        if not m15_box:
            return None

        m30_box = None
        if str((regime or {}).get("state") or "").upper() == "RANGING":
            m30_box = self._build_consolidation_box(m30_df, lookback=lookback)
            if bool(getattr(cfg, "SCALPER_SIDEWAYS_RANGE_M30_REQUIRED", True)) and not m30_box:
                return None
        merged_box = self._merge_range_boxes(m15_box, m30_box)
        if not merged_box:
            return None

        ema20_val = float(m15_box["ema20"])
        ema20_slope = float(m15_box["ema20_slope"])
        edge_buffer = max(
            spread * 2.0,
            float(merged_box["atr_ref"]) * float(getattr(cfg, "SCALPER_BIAS_CONSOLIDATION_ENTRY_EDGE_ATR", 0.18) or 0.18),
        )
        range_low = float(merged_box["range_low"])
        range_high = float(merged_box["range_high"])
        box_range = range_high - range_low
        atr15 = float(m15_box["atr"])
        label = str(merged_box["label"])

        if bias_dir == "LONG":
            if closes[-1] < ema20_val or ema20_slope <= 0:
                return None
            if m30_box and float(m30_box["ema20_slope"]) < -float(getattr(cfg, "SCALPER_EMA20_SLOPE_MIN", 0.05) or 0.05):
                return None
            if price > (range_low + edge_buffer):
                return None
            target_profit = float(getattr(cfg, "SCALPER_BIAS_CONSOLIDATION_TP_MIN", 1.0))
            if atr15 >= 1.8 or box_range >= 1.0:
                target_profit = float(getattr(cfg, "SCALPER_BIAS_CONSOLIDATION_TP_MAX", 2.0))
            return {
                "direction": "LONG",
                "anchor_level": round(range_low, 2),
                "range_low": round(range_low, 2),
                "range_high": round(range_high, 2),
                "atr15": round(atr15, 4),
                "target_profit": round(target_profit, 2),
                "label": label,
            }

        if closes[-1] > ema20_val or ema20_slope >= 0:
            return None
        if m30_box and float(m30_box["ema20_slope"]) > float(getattr(cfg, "SCALPER_EMA20_SLOPE_MIN", 0.05) or 0.05):
            return None
        if price < (range_high - edge_buffer):
            return None
        target_profit = float(getattr(cfg, "SCALPER_BIAS_CONSOLIDATION_TP_MIN", 1.0))
        if atr15 >= 1.8 or box_range >= 1.0:
            target_profit = float(getattr(cfg, "SCALPER_BIAS_CONSOLIDATION_TP_MAX", 2.0))
        return {
            "direction": "SHORT",
            "anchor_level": round(range_high, 2),
            "range_low": round(range_low, 2),
            "range_high": round(range_high, 2),
            "atr15": round(atr15, 4),
            "target_profit": round(target_profit, 2),
            "label": label,
        }

    def _detect_range_edge_scalp_setup(
        self,
        *,
        m15_df: pd.DataFrame,
        m30_df: pd.DataFrame,
        bias: Dict,
        regime: Dict,
        price: float,
        spread: float,
    ) -> Optional[Dict]:
        if not bool(getattr(cfg, "SCALPER_RANGE_EDGE_ENABLED", True)):
            return None
        if str((regime or {}).get("state") or "").upper() != "RANGING":
            return None
        if m15_df is None or m30_df is None:
            return None

        lookback = max(4, int(getattr(cfg, "SCALPER_RANGE_EDGE_LOOKBACK_CANDLES", 6) or 6))
        if len(m15_df) < lookback or len(m30_df) < lookback:
            return None

        bias_dir = str((bias or {}).get("direction") or "NEUTRAL").upper()
        if bias_dir not in {"LONG", "SHORT"}:
            return None

        m15_recent = m15_df.tail(lookback)
        m30_recent = m30_df.tail(lookback)
        range_low = max(float(m15_recent["low"].min()), float(m30_recent["low"].min()))
        range_high = min(float(m15_recent["high"].max()), float(m30_recent["high"].max()))
        if range_high <= range_low:
            return None

        m15_closes = m15_df["close"].values.astype(float)
        m15_highs = m15_df["high"].values.astype(float)
        m15_lows = m15_df["low"].values.astype(float)
        atr15 = float(atr(m15_highs, m15_lows, m15_closes, 14)[-1])
        if atr15 <= 0:
            return None

        box_range = range_high - range_low
        min_mult = float(getattr(cfg, "SCALPER_RANGE_EDGE_BOX_ATR_MIN", 1.15) or 1.15)
        max_mult = float(getattr(cfg, "SCALPER_RANGE_EDGE_BOX_ATR_MAX", 3.20) or 3.20)
        if box_range < atr15 * min_mult or box_range > atr15 * max_mult:
            return None

        edge_buffer = max(
            spread * 2.0,
            atr15 * float(getattr(cfg, "SCALPER_RANGE_EDGE_ENTRY_EDGE_ATR", 0.22) or 0.22),
        )

        if bias_dir == "LONG" and price <= (range_low + edge_buffer):
            target_profit = float(getattr(cfg, "SCALPER_RANGE_EDGE_TP_MIN", 1.0))
            if atr15 >= 2.2 or box_range >= 2.0:
                target_profit = float(getattr(cfg, "SCALPER_RANGE_EDGE_TP_MAX", 1.8))
            return {
                "direction": "LONG",
                "anchor_level": round(range_low, 2),
                "range_low": round(range_low, 2),
                "range_high": round(range_high, 2),
                "atr_ref": round(atr15, 4),
                "target_profit": round(target_profit, 2),
                "label": "M15/M30 wide range",
            }

        if bias_dir == "SHORT" and price >= (range_high - edge_buffer):
            target_profit = float(getattr(cfg, "SCALPER_RANGE_EDGE_TP_MIN", 1.0))
            if atr15 >= 2.2 or box_range >= 2.0:
                target_profit = float(getattr(cfg, "SCALPER_RANGE_EDGE_TP_MAX", 1.8))
            return {
                "direction": "SHORT",
                "anchor_level": round(range_high, 2),
                "range_low": round(range_low, 2),
                "range_high": round(range_high, 2),
                "atr_ref": round(atr15, 4),
                "target_profit": round(target_profit, 2),
                "label": "M15/M30 wide range",
            }
        return None

    def _detect_sweep(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        opens: np.ndarray,
    ) -> Optional[Dict]:
        n = len(highs)
        sweep_lookback = max(3, int(cfg.SCALPER_SWEEP_LOOKBACK))
        if n < sweep_lookback + 2:
            return None

        start = n - sweep_lookback - 1
        end = n - 2
        curr = n - 1

        for i in range(start, end - 2):
            for j in range(i + 2, end):
                if abs(lows[i] - lows[j]) < cfg.SCALPER_SWEEP_TOLERANCE:
                    eq_low = min(lows[i], lows[j])
                    for k in range(j + 1, curr + 1):
                        if lows[k] < eq_low - cfg.SCALPER_SWEEP_TOLERANCE and closes[curr] > eq_low and closes[curr] > opens[curr]:
                            return {
                                "direction": "LONG",
                                "level": round(float(lows[k]), 2),
                                "eq_level": round(float(eq_low), 2),
                            }

        for i in range(start, end - 2):
            for j in range(i + 2, end):
                if abs(highs[i] - highs[j]) < cfg.SCALPER_SWEEP_TOLERANCE:
                    eq_high = max(highs[i], highs[j])
                    for k in range(j + 1, curr + 1):
                        if highs[k] > eq_high + cfg.SCALPER_SWEEP_TOLERANCE and closes[curr] < eq_high and closes[curr] < opens[curr]:
                            return {
                                "direction": "SHORT",
                                "level": round(float(highs[k]), 2),
                                "eq_level": round(float(eq_high), 2),
                            }
        return None

    def _cleanup_levels(self):
        now = time.time()
        cooldown = float(getattr(cfg, "SCALPER_LEVEL_COOLDOWN", 1200))
        expired = [level for level, ts in self._traded_levels.items() if now - ts > cooldown]
        for level in expired:
            del self._traded_levels[level]


def _no(reason: str) -> Dict:
    return {"signal": "NO_TRADE", "reason": reason}
