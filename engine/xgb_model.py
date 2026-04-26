"""
XGBoost Learning Pipeline — learns from trade history, predicts win probability.

Features extracted from each trade:
  - session (LONDON/NY encoded)
  - direction (BUY/SELL encoded)
  - strategy (SMC/SCALPER encoded)
  - atr, atr_ratio, rsi, ema_slope, ema_gap, body_ratio, spread
  - regime (TRENDING/RANGING/EXPANSION encoded)
  - bias_confidence
  - hour of day

Target: won (1/0)

Retrains every N new trades. Predicts on new signals to boost/penalize confidence.
"""
import os
import json
import numpy as np
from typing import Dict, List, Optional
import xgboost as xgb

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODEL_PATH = os.path.join(_BASE_DIR, "xgb_model.json")
_MIN_TRADES = 15       # min trades before first training
_RETRAIN_EVERY = 10    # retrain after N new trades

# Feature columns
_FEATURES = [
    "session", "direction", "hour",
    "atr", "atr_ratio", "rsi", "ema_slope", "body_ratio", "spread",
    "regime", "bias_conf", "sl_distance", "volume",
]


def _encode_session(s: str) -> int:
    return {"ASIAN": 0, "LONDON": 1, "NEW_YORK": 2}.get(s, 0)

def _encode_direction(d: str) -> int:
    return 1 if d in ("BUY", "LONG") else 0

def _encode_regime(r: str) -> int:
    return {"RANGING": 0, "TRENDING": 1, "EXPANSION": 2, "NEWS_VOLATILITY": 3}.get(r, 0)


def _extract_features(trade: Dict) -> Optional[List[float]]:
    """Extract feature vector from a closed trade dict."""
    try:
        # Read from stored features snapshot (set at trade open time)
        feat = trade.get("features", {})
        hour = 12
        # trades store close time as 'time'; open time may be 'open_time'
        ct = trade.get("open_time") or trade.get("time", "")
        if ct:
            try:
                parts = str(ct).replace("T", " ").split(" ")
                if len(parts) >= 2:
                    hour = int(parts[1].split(":")[0])
            except Exception:
                pass

        return [
            _encode_session(feat.get("session", trade.get("session", ""))),
            _encode_direction(trade.get("direction", "")),
            hour,
            feat.get("atr", trade.get("atr", 1.5)),
            feat.get("atr_ratio", trade.get("atr_ratio", 1.0)),
            feat.get("rsi", trade.get("rsi", 50)),
            feat.get("ema_slope", trade.get("ema_slope", 0)),
            feat.get("body_ratio", trade.get("body_ratio", 0.5)),
            feat.get("spread", trade.get("spread", 0.3)),
            _encode_regime(feat.get("regime", trade.get("regime", ""))),
            feat.get("bias_conf", trade.get("bias_conf", 0.5)),
            trade.get("sl_distance", 0.5),
            trade.get("volume", 0.01),
        ]
    except Exception:
        return None


def _extract_signal_features(signal: Dict, indicators: Dict, regime: Dict,
                              bias: Dict, tick: Dict) -> Optional[List[float]]:
    """Extract features from a live signal for prediction."""
    try:
        from datetime import datetime, timezone
        hour = datetime.now(timezone.utc).hour
        from engine.session_filter import get_session
        session = get_session()

        return [
            _encode_session(session),
            _encode_direction(signal.get("signal", "")),
            hour,
            indicators.get("atr", 1.5),
            indicators.get("atr_ratio", 1.0),
            indicators.get("rsi", 50),
            indicators.get("ema9_slope", 0),
            indicators.get("body_ratio", 0.5),
            tick.get("spread", 0.3),
            _encode_regime(regime.get("state", "")),
            bias.get("confidence", 0.5),
            signal.get("sl_distance", 0.5),
            signal.get("volume", 0.01),
        ]
    except Exception:
        return None


class XGBModel:
    def __init__(self):
        self.model: Optional[xgb.XGBClassifier] = None
        self.is_trained = False
        self._trades_at_last_train = 0
        self._load()

    def should_retrain(self, total_trades: int) -> bool:
        if total_trades < _MIN_TRADES:
            return False
        if not self.is_trained:
            return True
        return total_trades - self._trades_at_last_train >= _RETRAIN_EVERY

    def train(self, closed_trades: List[Dict]):
        """Train on closed trade history."""
        X, y = [], []
        seen_keys: set = set()
        for t in closed_trades:
            # Skip trades with zero PnL (SL at entry — no information)
            if t.get("pnl", 0) == 0.0:
                continue
            feat = _extract_features(t)
            if feat is None:
                continue
            # Deduplicate on market-condition features only (indices 0-10).
            # Exclude sl_distance[11] and volume[12] — they vary per trade
            # but don't represent different market conditions.
            dedup_key = (tuple(round(v, 4) if isinstance(v, float) else v for v in feat[:11]),
                         t.get("direction", ""))
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)
            X.append(feat)
            # Use pnl > 0 as ground truth — won field can be corrupted by SL-positive closes
            y.append(1 if t.get("pnl", 0) > 0 else 0)

        if len(X) < _MIN_TRADES:
            return

        X = np.array(X, dtype=float)
        y = np.array(y, dtype=int)

        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

        self.model = xgb.XGBClassifier(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            verbosity=0,
        )
        self.model.fit(X, y)
        self.is_trained = True
        self._trades_at_last_train = len(closed_trades)
        self._save()

        win_rate = sum(y) / len(y)
        print(f"[XGB] Trained on {len(X)} trades (WR: {win_rate:.0%}, scale_pos_weight: {scale_pos_weight:.2f})")

    def predict_win_prob(self, signal: Dict, indicators: Dict,
                         regime: Dict, bias: Dict, tick: Dict) -> float:
        """Predict win probability for a signal. Returns 0.5 if not trained."""
        if not self.is_trained or self.model is None:
            return 0.5
        feat = _extract_signal_features(signal, indicators, regime, bias, tick)
        if feat is None:
            return 0.5
        try:
            X = np.array([feat], dtype=float)
            prob = self.model.predict_proba(X)[0][1]  # probability of class 1 (win)
            return round(float(prob), 3)
        except Exception:
            return 0.5

    def get_feature_importance(self) -> Dict:
        if not self.is_trained or self.model is None:
            return {"trained": False}
        try:
            imp = self.model.feature_importances_
            pairs = sorted(zip(_FEATURES, imp), key=lambda x: x[1], reverse=True)
            return {
                "trained": True,
                "trades": self._trades_at_last_train,
                "features": [{"name": n, "importance": round(float(v), 4)} for n, v in pairs],
            }
        except Exception:
            return {"trained": True, "error": "Cannot read importance"}

    def _save(self):
        if self.model is None:
            return
        try:
            self.model.save_model(_MODEL_PATH)
            meta = {"trades": self._trades_at_last_train}
            with open(_MODEL_PATH + ".meta", "w") as f:
                json.dump(meta, f)
        except Exception:
            pass

    def _load(self):
        if not os.path.isfile(_MODEL_PATH):
            return
        try:
            self.model = xgb.XGBClassifier()
            self.model.load_model(_MODEL_PATH)
            self.is_trained = True
            meta_path = _MODEL_PATH + ".meta"
            if os.path.isfile(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                self._trades_at_last_train = meta.get("trades", 0)
            print(f"[XGB] Loaded model ({self._trades_at_last_train} trades)")
        except Exception:
            self.model = None
            self.is_trained = False


def apply_xgb_filter(signal: Dict, indicators: Dict, regime: Dict, bias: Dict, tick: Dict) -> Dict:
    """Apply XGBoost as a filter only. Returns modified signal or blocks it."""
    if signal.get("signal") not in ("BUY", "SELL"):
        return signal
    
    win_prob = xgb_model.predict_win_prob(signal, indicators, regime, bias, tick)
    
    # Use XGBoost only as a filter with threshold
    threshold = 0.55  # Only trade if win probability > 55%
    
    if win_prob < threshold:
        return {
            "signal": "NO_TRADE",
            "reason": f"XGBoost filter: win probability {win_prob:.0%} < {threshold:.0%}",
            "xgb_prob": win_prob,
            "xgb_threshold": threshold
        }
    
    # Add XGB info to signal but don't modify confidence
    signal["_xgb_prob"] = win_prob
    signal["_xgb_threshold"] = threshold
    return signal

# Singleton
xgb_model = XGBModel()