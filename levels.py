"""Utilities for calculating key price levels from OHLCV data."""
from __future__ import annotations

from typing import Dict, List
import numpy as np
import pandas as pd

_TIMEFRAME_MAP = {
    "2h": "2h",
    "4h": "4h",
    "8h": "8h",
    "1d": "1D",
    "1w": "1W",
}

class LevelCalculator:
    """Calculate pivot, volume profile, equilibrium and open levels."""

    def __init__(self, df: pd.DataFrame, timeframe: str, n_bins: int = 30) -> None:
        if "open" not in df.columns:
            raise ValueError("DataFrame must contain OHLCV data with 'open' column")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex")
        self.df = df.copy()
        self.tf = _TIMEFRAME_MAP.get(timeframe.lower())
        if not self.tf:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        self.n_bins = n_bins

    def calculate(self) -> List[Dict[str, object]]:
        groups = self.df.groupby(pd.Grouper(freq=self.tf, label="left", closed="left"))
        levels: List[Dict[str, object]] = []
        prev_info = None
        for ts, g in groups:
            if g.empty:
                continue
            info = self._session_info(g)
            if prev_info is not None:
                pivot = self._calc_pivot(prev_info, info)
                if pivot is not None:
                    levels.append(self._fmt(ts, "pivot", pivot))
            vp = self._volume_profile(g)
            levels.append(self._fmt(ts, "poc", vp["poc"]))
            levels.append(self._fmt(ts, "vah", vp["vah"]))
            levels.append(self._fmt(ts, "val", vp["val"]))
            eq = (info["high"] + info["low"]) / 2
            levels.append(self._fmt(ts, "equilibrium", eq))
            levels.append(self._fmt(ts, "open", info["open"]))
            prev_info = info
        return levels

    @staticmethod
    def _session_info(g: pd.DataFrame) -> Dict[str, float]:
        return {
            "open": g["open"].iloc[0],
            "close": g["close"].iloc[-1],
            "high": g["high"].max(),
            "low": g["low"].min(),
        }

    def _calc_pivot(self, prev: Dict[str, float], cur: Dict[str, float]) -> float | None:
        prev_open, prev_close = prev["open"], prev["close"]
        cur_open, cur_close = cur["open"], cur["close"]
        if prev_close == cur_open:
            if (prev_open < prev_close and cur_open > cur_close) or (
                prev_open > prev_close and cur_open < cur_close
            ):
                return cur_open
            return None
        if prev_open < prev_close and cur_open > cur_close:
            return max(prev_close, cur_open)
        if prev_open > prev_close and cur_open < cur_close:
            return min(prev_close, cur_open)
        return None

    def _volume_profile(self, g: pd.DataFrame) -> Dict[str, float]:
        high = g["high"].max()
        low = g["low"].min()
        if high == low:
            return {"poc": high, "vah": high, "val": low}
        bins = np.linspace(low, high, self.n_bins + 1)
        prices = g["close"].to_numpy()
        vols = g["volume"].to_numpy()
        idx = np.searchsorted(bins, prices, side="right") - 1
        idx = np.clip(idx, 0, self.n_bins - 1)
        vol_bins = np.bincount(idx, weights=vols, minlength=self.n_bins)
        total = vol_bins.sum()
        csum = np.cumsum(vol_bins)
        poc_idx = vol_bins.argmax()
        vah_idx = np.searchsorted(csum, total * 0.70)
        val_idx = np.searchsorted(csum, total * 0.30)
        poc = (bins[poc_idx] + bins[poc_idx + 1]) / 2
        vah = bins[min(vah_idx, self.n_bins - 1)]
        val = bins[min(val_idx, self.n_bins - 1)]
        return {"poc": float(poc), "vah": float(vah), "val": float(val)}

    def _fmt(self, ts: pd.Timestamp, level_type: str, price: float) -> Dict[str, object]:
        return {
            "level_type": level_type,
            "timeframe": self.tf.lower(),
            "price": float(price),
            "timestamp": ts,
        }


def get_all_levels(ohlcv: pd.DataFrame, timeframes: List[str]) -> List[Dict[str, object]]:
    """Return a flat list of level dictionaries for the given timeframes."""
    all_levels: List[Dict[str, object]] = []
    for tf in timeframes:
        calc = LevelCalculator(ohlcv, tf)
        all_levels.extend(calc.calculate())
    return all_levels

__all__ = ["get_all_levels", "LevelCalculator"]
