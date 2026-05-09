"""
Trend Channel Strategy — supertrend + trendline channel entries on M15/H1.

Entry logic:
  - Detect valid channel (support + resistance trendlines) on the configured TF
  - Supertrend must agree with direction (uptrend = buy at support, downtrend = sell at resistance)
  - H1 bias must agree (optional, configurable)
  - Enter at channel boundary touch with SL just beyond the supertrend line

Exit logic (swing_trend profile):
  - No timeout — trade held until supertrend flips or channel breaks
  - SL trails the supertrend value each candle via _manage_trend() in trade_manager
  - TP = opposite channel wall
  - Reversal exit if price gives back 50% of peak from 3R+
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from ..strategies.base_strategy import BaseStrategy
import config as cfg


def _no(reason: str, **kw) -> Dict:
    return {"signal": "NO_TRADE", "reason": reason, **kw}


class TrendChannelStrategy(BaseStrategy):
    name = "TREND_CHANNEL"

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #

    def generate_signal(self, data: Dict) -> Dict:
        tf = getattr(cfg, "TREND_CHANNEL_TIMEFRAME", "M15")
        df = data.get(f"{tf.lower()}_df")
        if df is None or (hasattr(df, 'empty') and df.empty):
            df = data.get("m15_df")
        h1   = data.get("h1_df")
        tick = data.get("tick")

        if df is None or len(df) < 50 or tick is None:
            return _no("Insufficient data")

        spread = tick.get("spread", 0)
        max_spread = getattr(cfg, "TREND_CHANNEL_MAX_SPREAD", 0.60)
        if spread > max_spread:
            return _no(f"Spread {spread:.2f} > {max_spread}")

        price = tick["bid"]

        ms = data.get("market_state") or {}
        st = (ms.get("supertrend") or {}).get("M15") or {}
        st_direction = int(st.get("direction", 0))
        st_value     = float(st.get("value", 0.0))
        st_stable    = bool(st.get("stable", False))

        if not st_stable or st_direction == 0:
            return _no("Supertrend not yet stable (< 3 bars)")

        tl = (ms.get("trendlines") or {}).get("M15") or {}
        if not tl:
            return _no("Trendlines not available")

        c = df["close"].values.astype(float)
        channel = self._find_channel(tl, c, price, st_direction)
        if channel is None:
            trendlines = tl.get("trendlines", [])
            sup_proj = next((round(tl["slope"]*(len(c)-1)+tl["intercept"],2) for tl in trendlines if tl["type"]=="support"), None)
            res_proj = next((round(tl["slope"]*(len(c)-1)+tl["intercept"],2) for tl in trendlines if tl["type"]=="resistance"), None)
            return _no(f"No valid channel detected (price={price:.2f} sup_proj={sup_proj} res_proj={res_proj} tls={len(trendlines)})")

        support_price, resistance_price = channel

        # ── H1 bias alignment (optional) ────────────────────────────────
        if getattr(cfg, "TREND_CHANNEL_REQUIRE_H1_ALIGN", True):
            h1_ema = (ms.get("ema") or {}).get("H1_20") or {}
            h1_ema_val = float(h1_ema.get("value", 0.0))
            h1_slope   = float(h1_ema.get("slope", 0.0))
            if h1 is not None and len(h1) >= 50 and h1_ema_val > 0:
                price_h1 = float(h1["close"].iloc[-1])
                if st_direction == 1 and not (price_h1 > h1_ema_val and h1_slope > 0):
                    return _no(f"H1 not aligned with supertrend direction {st_direction:+d}")
                if st_direction == -1 and not (price_h1 < h1_ema_val and h1_slope < 0):
                    return _no(f"H1 not aligned with supertrend direction {st_direction:+d}")

        atr_val = float((ms.get("atr") or {}).get("M15") or 0.0)
        if atr_val <= 0:
            return _no("M15 ATR unavailable")
        touch_tol = atr_val * float(getattr(cfg, "TREND_CHANNEL_TOUCH_ATR_MULT", 0.5))

        direction, entry_zone = None, None
        if st_direction == 1 and abs(price - support_price) <= touch_tol:
            direction, entry_zone = "LONG", support_price
        elif st_direction == -1 and abs(price - resistance_price) <= touch_tol:
            direction, entry_zone = "SHORT", resistance_price

        if direction is None:
            return _no(
                f"Price {price:.2f} not at channel boundary "
                f"(sup={support_price:.2f} res={resistance_price:.2f} tol={touch_tol:.2f})"
            )

        # ── SL / TP ──────────────────────────────────────────────────────
        sl_buffer = atr_val * getattr(cfg, "TREND_CHANNEL_ATR_MULT_SL", 0.5)
        if direction == "LONG":
            sl = round(min(st_value, support_price) - sl_buffer, 2)
            tp = round(resistance_price - sl_buffer, 2)
            signal = "BUY"
        else:
            sl = round(max(st_value, resistance_price) + sl_buffer, 2)
            tp = round(support_price + sl_buffer, 2)
            signal = "SELL"

        sl_dist = round(abs(price - sl), 2)
        tp_dist = round(abs(tp - price), 2)
        if sl_dist < 0.5:
            return _no("SL distance too small")

        rr = round(tp_dist / sl_dist, 2)
        min_rr = getattr(cfg, "TREND_CHANNEL_MIN_RR", 1.5)
        if rr < min_rr:
            return _no(f"RR {rr:.2f} < {min_rr}")

        reason = (
            f"Channel {direction} | ST={st_value:.2f}({'UP' if st_direction==1 else 'DN'}) "
            f"| Sup={support_price:.2f} Res={resistance_price:.2f} "
            f"| RR={rr:.2f} | ATR={atr_val:.3f}"
        )

        return {
            "signal":        signal,
            "entry":         price,
            "sl":            sl,
            "tp":            tp,
            "sl_distance":   sl_dist,
            "confidence":    0.75,
            "reason":        reason,
            "rr":            rr,
            # Exit profile — no timeout, supertrend trailing SL
            "_exit_profile":                  "swing_trend",
            "_signal_family":                 "TREND",
            "_sweep_confirmed":               True,
            "_candle_confirmation":           True,
            "_setup_direction":               direction,
            "_supertrend_value":              st_value,
            "_supertrend_direction":          st_direction,
            "_channel_support":               support_price,
            "_channel_resistance":            resistance_price,
            "_be_trigger_r":                  cfg.EXIT_PROFILE_TREND_BE_TRIGGER_R,
            "_breakeven_min_hold_seconds":    cfg.EXIT_PROFILE_TREND_BREAKEVEN_MIN_HOLD_SECONDS,
            "_breakeven_volume_hold_ratio":   cfg.EXIT_PROFILE_TREND_BREAKEVEN_VOLUME_HOLD_RATIO,
            "_min_hold_seconds":              cfg.EXIT_PROFILE_TREND_MIN_HOLD_SECONDS,
            "_early_fail":                    0.0,
            "_tier1_min_ticks":               0,
            "_tier1_max_ticks":               0,
            "_timeout":                       0,
            "_timeout_min_progress_r":        0.0,
            "_reversal_arm_r":                cfg.EXIT_PROFILE_TREND_REVERSAL_ARM_R,
            "_reversal_drawdown_pct":         cfg.EXIT_PROFILE_TREND_REVERSAL_DRAWDOWN_PCT,
            "_reversal_floor_r":              cfg.EXIT_PROFILE_TREND_REVERSAL_FLOOR_R,
            "_trail_activate_r":              cfg.EXIT_PROFILE_TREND_TRAIL_ACTIVATE_R,
            "_trail_lock_r":                  cfg.EXIT_PROFILE_TREND_TRAIL_LOCK_R,
            "_velocity_drop_enabled":         False,
            "_entry_spread":                  spread,
        }

    # ------------------------------------------------------------------ #
    # Channel detection
    # ------------------------------------------------------------------ #

    def _find_channel(
        self,
        tl_result: Dict,
        closes: np.ndarray,
        price: float,
        st_direction: int,
    ) -> Optional[Tuple[float, float]]:
        """
        Find a valid support + resistance pair that forms a channel.
        Returns (support_price, resistance_price) or None.
        """
        trendlines = tl_result.get("trendlines", [])
        channels   = tl_result.get("channels", [])
        n = len(closes)

        # Prefer detected channels first
        for ch in channels:
            upper_idx = ch.get("upper_line_index")
            lower_idx = ch.get("lower_line_index")
            if upper_idx is None or lower_idx is None:
                continue
            if upper_idx >= len(trendlines) or lower_idx >= len(trendlines):
                continue
            upper_tl = trendlines[upper_idx]
            lower_tl = trendlines[lower_idx]
            res_price = self._project_line(upper_tl, n - 1)
            sup_price = self._project_line(lower_tl, n - 1)
            if res_price > price > sup_price:
                return round(sup_price, 2), round(res_price, 2)

        # Fallback: find nearest support below and resistance above
        # Prefer valid lines but fall back to any line if none are valid
        supports    = [tl for tl in trendlines if tl["type"] == "support"]
        resistances = [tl for tl in trendlines if tl["type"] == "resistance"]
        valid_supports    = [tl for tl in supports    if tl["valid"]]
        valid_resistances = [tl for tl in resistances if tl["valid"]]
        supports    = valid_supports    or supports
        resistances = valid_resistances or resistances
        if not supports or not resistances:
            return None

        sup_prices = [(self._project_line(tl, n - 1), tl) for tl in supports]
        res_prices = [(self._project_line(tl, n - 1), tl) for tl in resistances]

        below = [(p, tl) for p, tl in sup_prices if p < price]
        above = [(p, tl) for p, tl in res_prices if p > price]

        # If no valid resistance trendline above price, use recent swing high as resistance
        if not above:
            swing_highs = tl_result.get("swing_highs", [])
            recent_highs_above = [s["price"] for s in swing_highs[-20:] if s["price"] > price]
            if recent_highs_above:
                nearest_res = min(recent_highs_above)
            else:
                return None
        else:
            nearest_res = min(above, key=lambda x: x[0])[0]

        if not below:
            return None
        nearest_sup = max(below, key=lambda x: x[0])[0]

        min_bars = getattr(cfg, "TREND_CHANNEL_MIN_CHANNEL_BARS", 20)
        channel_width = nearest_res - nearest_sup
        if channel_width < 0.5 or (nearest_res - nearest_sup) < min_bars * 0.01:
            return None

        return round(nearest_sup, 2), round(nearest_res, 2)

    @staticmethod
    def _project_line(tl: Dict, bar_index: int) -> float:
        return tl["slope"] * bar_index + tl["intercept"]

    # ------------------------------------------------------------------ #
    # H1 alignment check — kept for backward compat but no longer called
    # ------------------------------------------------------------------ #

    def _h1_aligned(self, h1: pd.DataFrame, st_direction: int) -> bool:
        from ..indicators import ema as _ema
        c = h1["close"].values.astype(float)
        if len(c) < 20:
            return True
        ema20 = _ema(c, 20)
        slope = ema20[-1] - ema20[-5] if len(ema20) >= 5 else 0
        price = c[-1]
        if st_direction == 1:
            return price > ema20[-1] and slope > 0
        else:
            return price < ema20[-1] and slope < 0
