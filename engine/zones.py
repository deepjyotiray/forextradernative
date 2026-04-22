"""
Zone Detector — DBSCAN-based support/resistance zone clustering.
Ported from forextrader repo.
"""
import numpy as np
import pandas as pd
from typing import List, Dict
from sklearn.cluster import DBSCAN


class ZoneDetector:
    def __init__(self, eps: float = 3.0, min_samples: int = 2, max_width: float = 15.0):
        self.eps = eps
        self.min_samples = min_samples
        self.max_width = max_width

    def detect(self, df: pd.DataFrame) -> Dict[str, List[Dict]]:
        if len(df) < 20:
            return {"support": [], "resistance": []}
        price = float(df["close"].iloc[-1])
        levels = self._collect_levels(df)
        if not levels:
            return {"support": [], "resistance": []}
        zones = self._cluster(levels)
        zones = self._score(zones, df, price)

        support = sorted([z for z in zones if z["zone_high"] < price],
                         key=lambda x: x["strength"], reverse=True)[:8]
        resistance = sorted([z for z in zones if z["zone_low"] > price],
                            key=lambda x: x["strength"], reverse=True)[:8]
        support.sort(key=lambda x: x["zone_high"], reverse=True)
        resistance.sort(key=lambda x: x["zone_low"])
        return {"support": support, "resistance": resistance}

    def _collect_levels(self, df: pd.DataFrame) -> List[Dict]:
        levels = []
        h, l, o, c = df["high"].values, df["low"].values, df["open"].values, df["close"].values
        tr = h - l
        # Swing highs/lows
        for i in range(2, len(df) - 2):
            if h[i] > h[i-1] and h[i] > h[i-2] and h[i] > h[i+1] and h[i] > h[i+2]:
                levels.append({"price": float(h[i]), "idx": i})
            if l[i] < l[i-1] and l[i] < l[i-2] and l[i] < l[i+1] and l[i] < l[i+2]:
                levels.append({"price": float(l[i]), "idx": i})
        # Wick rejections
        for i in range(len(df)):
            if tr[i] == 0:
                continue
            uw = (h[i] - max(o[i], c[i])) / tr[i]
            lw = (min(o[i], c[i]) - l[i]) / tr[i]
            if uw > 0.55:
                levels.append({"price": float(h[i]), "idx": i})
            if lw > 0.55:
                levels.append({"price": float(l[i]), "idx": i})
        # Psychological levels
        cur = float(c[-1])
        for step in [10, 25, 50, 100]:
            base = round(cur / step) * step
            for off in range(-3, 4):
                p = base + off * step
                if abs(p - cur) <= 60:
                    levels.append({"price": float(p), "idx": -1})
        return levels

    def _cluster(self, levels: List[Dict]) -> List[Dict]:
        prices = np.array([l["price"] for l in levels]).reshape(-1, 1)
        db = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(prices)
        zones = []
        for label in set(db.labels_):
            if label == -1:
                continue
            mask = db.labels_ == label
            cp = [levels[i]["price"] for i in range(len(levels)) if mask[i]]
            zl, zh = min(cp), max(cp)
            if zh - zl > self.max_width:
                continue
            zones.append({
                "zone_low": round(zl, 2), "zone_high": round(zh, 2),
                "zone_mid": round(np.mean(cp), 2), "touches": int(mask.sum()),
                "strength": 0.0,
            })
        return zones

    def _score(self, zones: List[Dict], df: pd.DataFrame, price: float) -> List[Dict]:
        n = len(df)
        h, l = df["high"].values, df["low"].values
        recent_start = int(n * 0.8)
        recency = np.ones(n)
        recency[recent_start:] = 3.0
        max_raw = 1.0
        for z in zones:
            margin = max((z["zone_high"] - z["zone_low"]) * 0.3, 1.0)
            touch_mask = (l <= z["zone_high"] + margin) & (h >= z["zone_low"] - margin)
            w = recency[touch_mask].sum()
            prox = max(0, 20 - abs(price - z["zone_mid"]) * 0.5)
            raw = max(0, w * 2 + prox)
            z["_raw"] = raw
            if raw > max_raw:
                max_raw = raw
        for z in zones:
            z["strength"] = round(z.pop("_raw", 0) / max_raw, 2) if max_raw > 0 else 0
        return zones
