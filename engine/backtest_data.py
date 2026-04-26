"""
Historical MT5 data provider for backtests.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
import json
import os
import uuid

import pandas as pd

import config as cfg

try:
    import MetaTrader5 as mt5
except Exception:  # pragma: no cover
    mt5 = None


_TF_MAP = {
    "M1": getattr(mt5, "TIMEFRAME_M1", None),
    "M5": getattr(mt5, "TIMEFRAME_M5", None),
    "M15": getattr(mt5, "TIMEFRAME_M15", None),
    "H1": getattr(mt5, "TIMEFRAME_H1", None),
    "H4": getattr(mt5, "TIMEFRAME_H4", None),
}


@dataclass
class BacktestDataset:
    symbol: str
    requested_start_utc: datetime
    requested_end_utc: datetime
    effective_start_utc: datetime
    effective_end_utc: datetime
    ticks: pd.DataFrame
    candles: Dict[str, pd.DataFrame]
    coverage_warnings: List[str]
    data_source: str = "unknown"


class BacktestDataProvider:
    def __init__(self):
        self._connected = False
        self._cache: Dict[str, BacktestDataset] = {}
        self._cache_root = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "backtests",
            "data_cache",
        )
        self._cache_index_path = os.path.join(self._cache_root, "index.json")
        os.makedirs(self._cache_root, exist_ok=True)

    def connect(self) -> bool:
        if mt5 is None:
            return False
        kwargs = {}
        if cfg.MT5_PATH:
            kwargs["path"] = cfg.MT5_PATH
        if cfg.MT5_LOGIN:
            kwargs["login"] = cfg.MT5_LOGIN
            kwargs["password"] = cfg.MT5_PASSWORD or ""
            kwargs["server"] = cfg.MT5_SERVER or ""
        if not mt5.initialize(**kwargs):
            return False
        self._connected = True
        return True

    def disconnect(self):
        if self._connected and mt5 is not None:
            mt5.shutdown()
        self._connected = False

    def load_dataset(
        self,
        symbol: str,
        start_utc: datetime,
        end_utc: datetime,
        warmup_hours: int = 24 * 3,
        cache_only: bool = False,
    ) -> BacktestDataset:
        if start_utc.tzinfo is None:
            start_utc = start_utc.replace(tzinfo=timezone.utc)
        else:
            start_utc = start_utc.astimezone(timezone.utc)
        if end_utc.tzinfo is None:
            end_utc = end_utc.replace(tzinfo=timezone.utc)
        else:
            end_utc = end_utc.astimezone(timezone.utc)
        if end_utc <= start_utc:
            raise ValueError("end_utc must be after start_utc")
        warmup_start = start_utc - timedelta(hours=warmup_hours)

        key = f"{symbol}|{start_utc.isoformat()}|{end_utc.isoformat()}|{warmup_hours}"
        if key in self._cache:
            return self._cache[key]

        cached = self._load_from_disk_cache(symbol, start_utc, end_utc, warmup_start)
        if cached is not None:
            self._cache[key] = cached
            return cached

        if cache_only:
            raise RuntimeError(
                "Cache-only mode: no cached dataset fully covers requested range "
                f"for {symbol} ({start_utc.isoformat()} -> {end_utc.isoformat()})"
            )

        if not self._connected and not self.connect():
            raise RuntimeError("Failed to initialize MT5")
        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"Failed to select symbol: {symbol}")

        candles = {
            tf: self._fetch_candles(symbol, tf, warmup_start, end_utc)
            for tf in ("M1", "M5", "M15", "H1", "H4")
        }
        ticks = self._fetch_ticks(symbol, start_utc, end_utc)
        if ticks.empty:
            raise RuntimeError("No tick history returned from MT5 for selected range")

        first_tick, last_tick = ticks.iloc[0]["datetime"], ticks.iloc[-1]["datetime"]
        effective_start = max(start_utc, first_tick.to_pydatetime().astimezone(timezone.utc))
        effective_end = min(end_utc, last_tick.to_pydatetime().astimezone(timezone.utc))
        if effective_end <= effective_start:
            raise RuntimeError(
                "Tick history does not overlap requested range. "
                f"First={first_tick.isoformat()} Last={last_tick.isoformat()}"
            )
        coverage_warnings: List[str] = []
        if first_tick > start_utc:
            coverage_warnings.append(
                "Requested start is earlier than first available tick. "
                f"Requested={start_utc.isoformat()} FirstAvailable={first_tick.isoformat()}"
            )
        if last_tick < end_utc:
            coverage_warnings.append(
                "Requested end is later than last available tick. "
                f"Requested={end_utc.isoformat()} LastAvailable={last_tick.isoformat()}"
            )

        dataset = BacktestDataset(
            symbol=symbol,
            requested_start_utc=start_utc,
            requested_end_utc=end_utc,
            effective_start_utc=effective_start,
            effective_end_utc=effective_end,
            ticks=ticks,
            candles=candles,
            coverage_warnings=coverage_warnings,
            data_source="mt5",
        )
        self._cache[key] = dataset
        self._store_to_disk_cache(dataset, warmup_start)
        return dataset

    def _load_index(self) -> List[Dict]:
        if not os.path.exists(self._cache_index_path):
            return []
        try:
            with open(self._cache_index_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            entries = payload.get("entries", [])
            return entries if isinstance(entries, list) else []
        except Exception:
            return []

    def get_cache_availability(self, symbol: Optional[str] = None) -> Dict:
        entries = self._load_index()
        windows: List[Dict] = []
        min_start: Optional[datetime] = None
        max_end: Optional[datetime] = None

        for entry in entries:
            entry_symbol = str(entry.get("symbol", "")).upper().strip()
            if not entry_symbol:
                continue
            if symbol and entry_symbol != symbol.upper().strip():
                continue
            ticks_path = str(entry.get("ticks_path", ""))
            if not ticks_path or not os.path.exists(ticks_path):
                continue
            try:
                start_utc = datetime.fromisoformat(str(entry.get("start_utc"))).astimezone(timezone.utc)
                end_utc = datetime.fromisoformat(str(entry.get("end_utc"))).astimezone(timezone.utc)
            except Exception:
                continue
            if end_utc <= start_utc:
                continue

            tick_count = int(entry.get("tick_count", 0) or 0)
            window = {
                "id": str(entry.get("id", "")),
                "symbol": entry_symbol,
                "start_utc": start_utc.isoformat(),
                "end_utc": end_utc.isoformat(),
                "tick_count": tick_count,
                "duration_hours": round((end_utc - start_utc).total_seconds() / 3600.0, 2),
                "created_at": str(entry.get("created_at", "")),
            }
            windows.append(window)

            if min_start is None or start_utc < min_start:
                min_start = start_utc
            if max_end is None or end_utc > max_end:
                max_end = end_utc

        windows.sort(key=lambda x: x["start_utc"], reverse=True)
        symbols = sorted({w["symbol"] for w in windows})
        return {
            "symbol": symbol.upper().strip() if symbol else None,
            "symbols": symbols,
            "windows": windows,
            "window_count": len(windows),
            "min_start_utc": min_start.isoformat() if min_start else None,
            "max_end_utc": max_end.isoformat() if max_end else None,
        }

    def describe_cache_coverage(
        self,
        symbol: str,
        start_utc: datetime,
        end_utc: datetime,
        warmup_hours: int = 24 * 3,
    ) -> Dict[str, object]:
        symbol = str(symbol).upper().strip()
        if start_utc.tzinfo is None:
            start_utc = start_utc.replace(tzinfo=timezone.utc)
        else:
            start_utc = start_utc.astimezone(timezone.utc)
        if end_utc.tzinfo is None:
            end_utc = end_utc.replace(tzinfo=timezone.utc)
        else:
            end_utc = end_utc.astimezone(timezone.utc)
        warmup_start = start_utc - timedelta(hours=warmup_hours)

        entry = self._find_covering_cache_entry(
            symbol=symbol,
            start_utc=start_utc,
            end_utc=end_utc,
            warmup_start=warmup_start,
        )
        if entry is None:
            return {
                "symbol": symbol,
                "requested_start_utc": start_utc.isoformat(),
                "requested_end_utc": end_utc.isoformat(),
                "warmup_start_utc": warmup_start.isoformat(),
                "cache_entry_found": False,
                "has_tick_overlap": False,
                "effective_start_utc": None,
                "effective_end_utc": None,
                "entry_id": None,
            }

        try:
            effective_start = datetime.fromisoformat(str(entry.get("effective_start_utc"))).astimezone(timezone.utc)
            effective_end = datetime.fromisoformat(str(entry.get("effective_end_utc"))).astimezone(timezone.utc)
        except Exception:
            effective_start = start_utc
            effective_end = end_utc

        has_tick_overlap = max(start_utc, effective_start) < min(end_utc, effective_end)
        return {
            "symbol": symbol,
            "requested_start_utc": start_utc.isoformat(),
            "requested_end_utc": end_utc.isoformat(),
            "warmup_start_utc": warmup_start.isoformat(),
            "cache_entry_found": True,
            "has_tick_overlap": bool(has_tick_overlap),
            "effective_start_utc": effective_start.isoformat(),
            "effective_end_utc": effective_end.isoformat(),
            "entry_id": str(entry.get("id", "")),
        }

    def load_tick_density_profile(
        self,
        symbol: str,
        start_utc: datetime,
        end_utc: datetime,
    ) -> pd.DataFrame:
        """Return minute-level tick density for a cached range."""
        symbol = str(symbol).upper().strip()
        start_utc = start_utc.astimezone(timezone.utc) if start_utc.tzinfo else start_utc.replace(tzinfo=timezone.utc)
        end_utc = end_utc.astimezone(timezone.utc) if end_utc.tzinfo else end_utc.replace(tzinfo=timezone.utc)

        entry = self._find_covering_cache_entry(symbol, start_utc, end_utc)
        if entry is not None:
            profile = self._load_density_from_m1_cache(entry, start_utc, end_utc)
            if not profile.empty:
                profile.attrs["density_source"] = "cache_m1_volume"
                return profile

        try:
            dataset = self.load_dataset(symbol=symbol, start_utc=start_utc, end_utc=end_utc, cache_only=True)
            if not dataset.ticks.empty:
                grouped = (
                    dataset.ticks.assign(bucket_start=dataset.ticks["datetime"].dt.floor("min"))
                    .groupby("bucket_start", as_index=False)
                    .size()
                    .rename(columns={"size": "estimated_ticks"})
                )
                profile = self._finalize_density_profile(grouped, start_utc, end_utc)
                profile.attrs["density_source"] = "cache_ticks"
                return profile
        except Exception:
            pass

        empty = pd.DataFrame(columns=["bucket_start", "estimated_ticks"])
        empty.attrs["density_source"] = "unavailable"
        return empty

    def _save_index(self, entries: List[Dict]) -> None:
        payload = {"entries": entries}
        with open(self._cache_index_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _load_from_disk_cache(
        self,
        symbol: str,
        start_utc: datetime,
        end_utc: datetime,
        warmup_start: datetime,
    ) -> Optional[BacktestDataset]:
        entries = self._load_index()
        for entry in reversed(entries):
            if entry.get("symbol") != symbol:
                continue
            try:
                entry_start = datetime.fromisoformat(str(entry.get("start_utc"))).astimezone(timezone.utc)
                entry_end = datetime.fromisoformat(str(entry.get("end_utc"))).astimezone(timezone.utc)
                entry_warmup = datetime.fromisoformat(str(entry.get("warmup_start_utc"))).astimezone(timezone.utc)
            except Exception:
                continue
            if not (entry_start <= start_utc and entry_end >= end_utc and entry_warmup <= warmup_start):
                continue

            try:
                ticks_path = str(entry.get("ticks_path", ""))
                if not ticks_path or not os.path.exists(ticks_path):
                    continue
                ticks = pd.read_pickle(ticks_path)
                ticks = ticks.loc[(ticks["datetime"] >= start_utc) & (ticks["datetime"] <= end_utc)].reset_index(drop=True)
                if ticks.empty:
                    continue

                candle_paths = entry.get("candles", {}) or {}
                candles: Dict[str, pd.DataFrame] = {}
                for tf in ("M1", "M5", "M15", "H1", "H4"):
                    path = str(candle_paths.get(tf, ""))
                    if not path or not os.path.exists(path):
                        candles[tf] = pd.DataFrame()
                        continue
                    try:
                        cdf = pd.read_pickle(path)
                        if cdf.empty or "datetime" not in cdf.columns:
                            candles[tf] = pd.DataFrame()
                            continue
                        cdf = cdf.loc[
                            (cdf["datetime"] >= warmup_start) & (cdf["datetime"] <= end_utc)
                        ].reset_index(drop=True)
                        candles[tf] = cdf
                    except Exception:
                        # Tolerate malformed candle cache files and keep the usable tick cache.
                        candles[tf] = pd.DataFrame()

                first_tick = ticks.iloc[0]["datetime"]
                last_tick = ticks.iloc[-1]["datetime"]
                effective_start = max(start_utc, first_tick.to_pydatetime().astimezone(timezone.utc))
                effective_end = min(end_utc, last_tick.to_pydatetime().astimezone(timezone.utc))
                if effective_end <= effective_start:
                    continue

                coverage_warnings: List[str] = []
                if first_tick > start_utc:
                    coverage_warnings.append(
                        "Requested start is earlier than first available tick. "
                        f"Requested={start_utc.isoformat()} FirstAvailable={first_tick.isoformat()}"
                    )
                if last_tick < end_utc:
                    coverage_warnings.append(
                        "Requested end is later than last available tick. "
                        f"Requested={end_utc.isoformat()} LastAvailable={last_tick.isoformat()}"
                    )

                return BacktestDataset(
                    symbol=symbol,
                    requested_start_utc=start_utc,
                    requested_end_utc=end_utc,
                    effective_start_utc=effective_start,
                    effective_end_utc=effective_end,
                    ticks=ticks,
                    candles=candles,
                    coverage_warnings=coverage_warnings,
                    data_source="cache",
                )
            except Exception:
                continue
        return None

    def _find_covering_cache_entry(
        self,
        symbol: str,
        start_utc: datetime,
        end_utc: datetime,
        warmup_start: Optional[datetime] = None,
    ) -> Optional[Dict]:
        entries = self._load_index()
        symbol = str(symbol).upper().strip()
        best_entry: Optional[Dict] = None
        best_span: Optional[float] = None
        for entry in reversed(entries):
            if str(entry.get("symbol", "")).upper().strip() != symbol:
                continue
            try:
                entry_start = datetime.fromisoformat(str(entry.get("start_utc"))).astimezone(timezone.utc)
                entry_end = datetime.fromisoformat(str(entry.get("end_utc"))).astimezone(timezone.utc)
            except Exception:
                continue
            if not (entry_start <= start_utc and entry_end >= end_utc):
                continue
            if warmup_start is not None:
                try:
                    entry_warmup = datetime.fromisoformat(str(entry.get("warmup_start_utc"))).astimezone(timezone.utc)
                except Exception:
                    continue
                if entry_warmup > warmup_start:
                    continue
            span = (entry_end - entry_start).total_seconds()
            if best_entry is None or best_span is None or span < best_span:
                best_entry = entry
                best_span = span
        return best_entry

    def _load_density_from_m1_cache(
        self,
        entry: Dict,
        start_utc: datetime,
        end_utc: datetime,
    ) -> pd.DataFrame:
        candle_paths = entry.get("candles", {}) or {}
        path = str(candle_paths.get("M1", ""))
        if not path or not os.path.exists(path):
            return pd.DataFrame(columns=["bucket_start", "estimated_ticks"])
        try:
            m1 = pd.read_pickle(path)
        except Exception:
            return pd.DataFrame(columns=["bucket_start", "estimated_ticks"])
        if m1.empty or "datetime" not in m1.columns:
            return pd.DataFrame(columns=["bucket_start", "estimated_ticks"])

        df = m1.loc[(m1["datetime"] >= start_utc) & (m1["datetime"] < end_utc)].copy()
        if df.empty:
            return pd.DataFrame(columns=["bucket_start", "estimated_ticks"])
        volume = pd.to_numeric(df.get("volume", 0), errors="coerce").fillna(0.0)
        grouped = pd.DataFrame(
            {
                "bucket_start": pd.to_datetime(df["datetime"], utc=True).dt.floor("min"),
                "estimated_ticks": volume.clip(lower=0).round().astype(int),
            }
        )
        return self._finalize_density_profile(grouped, start_utc, end_utc)

    def _finalize_density_profile(
        self,
        grouped: pd.DataFrame,
        start_utc: datetime,
        end_utc: datetime,
    ) -> pd.DataFrame:
        if grouped.empty:
            return pd.DataFrame(columns=["bucket_start", "estimated_ticks"])
        grouped = (
            grouped.groupby("bucket_start", as_index=False)["estimated_ticks"]
            .sum()
            .sort_values("bucket_start")
            .reset_index(drop=True)
        )
        minute_index = pd.date_range(
            start=start_utc.replace(second=0, microsecond=0),
            end=(end_utc - timedelta(minutes=1)).replace(second=0, microsecond=0),
            freq="min",
            tz="UTC",
        )
        if len(minute_index) == 0:
            minute_index = pd.DatetimeIndex([start_utc.replace(second=0, microsecond=0)])
        base = pd.DataFrame({"bucket_start": minute_index})
        merged = base.merge(grouped, on="bucket_start", how="left")
        merged["estimated_ticks"] = merged["estimated_ticks"].fillna(0).astype(int)
        return merged

    def _store_to_disk_cache(self, dataset: BacktestDataset, warmup_start: datetime) -> None:
        cache_id = uuid.uuid4().hex[:12]
        cache_dir = os.path.join(self._cache_root, dataset.symbol, cache_id)
        os.makedirs(cache_dir, exist_ok=True)

        ticks_path = os.path.join(cache_dir, "ticks.pkl")
        dataset.ticks.to_pickle(ticks_path)

        candle_paths: Dict[str, str] = {}
        for tf, cdf in dataset.candles.items():
            path = os.path.join(cache_dir, f"{tf}.pkl")
            cdf.to_pickle(path)
            candle_paths[tf] = path

        entry = {
            "id": cache_id,
            "symbol": dataset.symbol,
            "warmup_start_utc": warmup_start.isoformat(),
            "start_utc": dataset.requested_start_utc.isoformat(),
            "end_utc": dataset.requested_end_utc.isoformat(),
            "effective_start_utc": dataset.effective_start_utc.isoformat(),
            "effective_end_utc": dataset.effective_end_utc.isoformat(),
            "ticks_path": ticks_path,
            "candles": candle_paths,
            "tick_count": int(len(dataset.ticks)),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        entries = self._load_index()
        entries.append(entry)
        self._save_index(entries)

    def _fetch_candles(
        self,
        symbol: str,
        timeframe: str,
        start_utc: datetime,
        end_utc: datetime,
    ) -> pd.DataFrame:
        tf = _TF_MAP.get(timeframe)
        if tf is None:
            return pd.DataFrame()
        rates = mt5.copy_rates_range(symbol, tf, start_utc, end_utc)
        if rates is None or len(rates) == 0:
            return pd.DataFrame()
        df = pd.DataFrame(rates)
        df["datetime"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.rename(columns={"tick_volume": "volume"})
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = df[col].astype(float)
        return df[["datetime", "open", "high", "low", "close", "volume"]].reset_index(drop=True)

    def _fetch_ticks(self, symbol: str, start_utc: datetime, end_utc: datetime) -> pd.DataFrame:
        flags = getattr(mt5, "COPY_TICKS_ALL", 0)
        ticks = mt5.copy_ticks_range(symbol, start_utc, end_utc, flags)
        if ticks is None or len(ticks) == 0:
            return pd.DataFrame()
        df = pd.DataFrame(ticks)
        df["datetime"] = pd.to_datetime(df["time"], unit="s", utc=True)
        if "time_msc" in df.columns:
            df["datetime"] = pd.to_datetime(df["time_msc"], unit="ms", utc=True)
        df["bid"] = df["bid"].astype(float)
        df["ask"] = df["ask"].astype(float)
        df["spread"] = (df["ask"] - df["bid"]).astype(float)
        keep_cols = ["datetime", "bid", "ask", "spread"]
        if "volume" in df.columns:
            keep_cols.append("volume")
        else:
            df["volume"] = 0.0
            keep_cols.append("volume")
        df = df[keep_cols].sort_values("datetime").reset_index(drop=True)
        # Hard clip to requested range to avoid off-by-one ticks.
        mask = (df["datetime"] >= start_utc) & (df["datetime"] <= end_utc)
        return df.loc[mask].reset_index(drop=True)


def candle_tail(df: pd.DataFrame, dt: datetime) -> pd.DataFrame:
    if df.empty:
        return df
    return df.loc[df["datetime"] <= dt].reset_index(drop=True)


def next_candle_timeframes() -> List[str]:
    return ["M1", "M5", "M15", "H1", "H4"]
