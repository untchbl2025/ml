from __future__ import annotations

import contextlib
import os
import argparse
import random
from typing import Callable, Iterable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from lightgbm import LGBMClassifier
from scipy.stats import zscore
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import (
    ParameterGrid,
    TimeSeriesSplit,
    cross_val_score,
)
from sklearn.utils import resample
from tabulate import tabulate
from xgboost import XGBClassifier
import joblib

try:
    from alive_progress import alive_it as _alive_it, alive_bar as _alive_bar
except Exception:  # pragma: no cover - fallback when alive_progress is missing
    _alive_it = None
    _alive_bar = None


def alive_it(iterable, *args, **kwargs):
    """Fallback-compatible ``alive_it`` implementation."""
    if _alive_it is not None:
        return _alive_it(iterable, *args, **kwargs)

    if isinstance(iterable, int):
        if _alive_bar is not None:
            return _alive_bar(iterable, *args, **kwargs)

        @contextlib.contextmanager
        def _dummy_bar():
            def _noop(_=1):
                pass

            yield _noop

        return _dummy_bar()

    return iterable


alive_bar = _alive_bar

# === Adjustable Parameters ===
SYMBOL = "SPXUSDT"
LIVEDATA_LEN = 1000
TRAIN_N = 500
INVALID_SHARE = 0.10
N_SHARE = 0.05
PUFFER = 0.02

MODEL_PATH = os.environ.get("MODEL_PATH", "elliott_model.joblib")
DATASET_PATH = os.environ.get("DATASET_PATH", "elliott_dataset.joblib")
CONFIDENCE_THRESHOLD = 0.3

# === Pattern Registry ===


class PatternRegistry:
    """Registry for synthetic pattern generators."""

    def __init__(self) -> None:
        self._patterns: Dict[str, Callable[..., object]] = {}
        self._next_wave: Dict[str, List[str]] = {}

    def register(
        self,
        name: str,
        generator: Callable[..., object],
        next_wave: Optional[Iterable[str]] = None,
    ) -> Callable[..., object]:
        """Register a pattern generator with optional follow-up waves."""
        self._patterns[name] = generator
        if next_wave is not None:
            if isinstance(next_wave, str):
                self._next_wave[name] = [next_wave]
            else:
                self._next_wave[name] = list(next_wave)
        return generator

    def has_pattern(self, name: str) -> bool:
        """Return ``True`` if a pattern ``name`` is registered."""
        return name in self._patterns

    def generators(self) -> List[Tuple[Callable[..., object], str]]:
        """Return list of registered generators as ``(func, name)`` tuples."""
        return [(func, name) for name, func in self._patterns.items()]

    def get_next_wave(self, name: str) -> List[str]:
        """Return configured follow-up waves for ``name``."""
        return self._next_wave.get(name, [])


pattern_registry = PatternRegistry()


def register_pattern(name: str, next_wave: Optional[Iterable[str]] = None):
    """Decorator to register a pattern generator."""

    def decorator(func: Callable[..., object]):
        pattern_registry.register(name, func, next_wave)
        return func

    return decorator


def _apply_ohlc_noise(df: pd.DataFrame, noise_level: float) -> pd.DataFrame:
    """Apply gaussian noise to OHLC columns of ``df``."""
    if noise_level and noise_level > 0:
        cols = ["open", "high", "low", "close"]
        noise = np.random.normal(0, noise_level, size=(len(df), len(cols)))
        df[cols] = df[cols].to_numpy() + noise
        df["high"] = df[["open", "close", "high"]].max(axis=1)
        df["low"] = df[["open", "close", "low"]].min(axis=1)
    return df


# === Fibonacci Levels ===


def _current_swing(df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return start and end timestamps for the most recent swing."""

    high_idx = df["high"].idxmax()
    low_idx = df["low"].idxmin()
    if high_idx > low_idx:
        return low_idx, high_idx
    return high_idx, low_idx


def get_fib_levels(
    df: pd.DataFrame, timeframe: str
) -> List[Dict[str, object]]:
    """Return fibonacci levels for the current swing of ``df``."""

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex")

    start_ts, end_ts = _current_swing(df)
    start_price = (
        df.loc[start_ts, "low"]
        if start_ts < end_ts
        else df.loc[start_ts, "high"]
    )
    end_price = (
        df.loc[end_ts, "high"] if end_ts > start_ts else df.loc[end_ts, "low"]
    )

    if end_price >= start_price:
        diff = end_price - start_price
        levels = {
            "fib_0.0": end_price,
            "fib_0.236": end_price - diff * 0.236,
            "fib_0.382": end_price - diff * 0.382,
            "fib_0.5": end_price - diff * 0.5,
            "fib_0.618": end_price - diff * 0.618,
            "fib_0.786": end_price - diff * 0.786,
            "fib_1.0": start_price,
            "fib_1.618": start_price - diff * 0.618,
            "fib_2.618": start_price - diff * 1.618,
        }
    else:
        diff = start_price - end_price
        levels = {
            "fib_0.0": end_price,
            "fib_0.236": end_price + diff * 0.236,
            "fib_0.382": end_price + diff * 0.382,
            "fib_0.5": end_price + diff * 0.5,
            "fib_0.618": end_price + diff * 0.618,
            "fib_0.786": end_price + diff * 0.786,
            "fib_1.0": start_price,
            "fib_1.618": start_price + diff * 0.618,
            "fib_2.618": start_price + diff * 1.618,
        }

    ts = df.index[-1]
    return [
        {
            "level_type": name,
            "timeframe": timeframe.lower(),
            "price": float(val),
            "timestamp": ts,
        }
        for name, val in levels.items()
    ]


# === Level Calculation ===


_TIMEFRAME_MAP = {
    "2h": "2h",
    "4h": "4h",
    "8h": "8h",
    "1d": "1D",
    "1w": "1W",
}


class LevelCalculator:
    """Calculate pivot, volume profile, equilibrium and open levels."""

    def __init__(
        self, df: pd.DataFrame, timeframe: str, n_bins: int = 30
    ) -> None:
        if "open" not in df.columns:
            raise ValueError(
                "DataFrame must contain OHLCV data with 'open' column"
            )
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex")
        self.df = df.copy()
        self.tf = _TIMEFRAME_MAP.get(timeframe.lower())
        if not self.tf:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        self.n_bins = n_bins

    def calculate(self, *, log: bool = False) -> List[Dict[str, object]]:
        grouped = list(
            self.df.groupby(
                pd.Grouper(freq=self.tf, label="left", closed="left")
            )
        )
        levels: List[Dict[str, object]] = []
        prev_info = None
        for ts, g in alive_it(grouped, disable=not log):
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

    def _calc_pivot(
        self, prev: Dict[str, float], cur: Dict[str, float]
    ) -> float | None:
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

    def _fmt(
        self, ts: pd.Timestamp, level_type: str, price: float
    ) -> Dict[str, object]:
        return {
            "level_type": level_type,
            "timeframe": self.tf.lower(),
            "price": float(price),
            "timestamp": ts,
        }


def get_all_levels(
    ohlcv: pd.DataFrame, timeframes: List[str], *, log: bool = False
) -> List[Dict[str, object]]:
    """Return a flat list of level dictionaries for the given timeframes."""
    all_levels: List[Dict[str, object]] = []
    for tf in alive_it(timeframes, disable=not log):
        calc = LevelCalculator(ohlcv, tf)
        all_levels.extend(calc.calculate(log=log))
    return all_levels


# === Parameter ===
FEATURES_BASE = [
    "returns",
    "range",
    "body",
    "ma_diff",
    "vol_ratio",
    "fibo_level",
    "wave_len_ratio",
    "rsi_z",
    "macd",
    "macd_signal",
    "stoch_k",
    "stoch_d",
    "obv",
    "atr",
    "kvo",
    "kvo_signal",
    "cmf",
    "high_z",
    "low_z",
    "vol_z",
    "ema_ratio",
    "bb_width",
    "roc_10",
    "corr_close_vol_10",
    "slope_5",
    "trend_len",
    "vol_atr_ratio",
    "rsi_4h",
    "close_4h",
    "vol_4h",
    "pattern_confidence",
    "wave_structure_score",
    "pos_in_pattern",
    "prev_wave_code",
    "next_wave_code",
    "level_dist",
    "fib_dist_1d",
    "fib_near_1d",
    "fib_dist_1w",
    "fib_near_1w",
]


def save_model(model, path):
    joblib.dump(model, path)


def load_model(path):
    return joblib.load(path)


def save_dataset(df, path):
    joblib.dump(df, path)


def load_dataset(path):
    return joblib.load(path)


def log_feature_stats(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Return mean, std, min and max for each column of ``df``.

    Parameters
    ----------
    df : pandas.DataFrame
        Feature matrix.

    Returns
    -------
    dict
        Mapping of feature name to statistics.
    """

    stats_df = df.agg(["mean", "std", "min", "max"]).T
    print(bold("\nFeature statistics:"))
    print(
        tabulate(
            stats_df.reset_index().values,
            headers=["Feature", "mean", "std", "min", "max"],
        )
    )
    return stats_df.to_dict(orient="index")


PATTERN_PROJ_FACTORS = {
    "TRIANGLE": 1.0,
    "ZIGZAG": 0.9,
    "DOUBLE_ZIGZAG": 1.1,
    "FLAT": 0.8,
    "RUNNING_FLAT": 0.7,
    "EXPANDED_FLAT": 1.2,
    "TREND_REVERSAL": 1.0,
    "FALSE_BREAKOUT": 0.8,
    "GAP_EXTENSION": 1.2,
    "WXY": 1.0,
    "WXYXZ": 1.0,
    "WXYXZY": 1.0,
}
# Mapping for default Elliott wave sequence. Pattern specific follow-ups are
# provided via the pattern registry.
DEFAULT_NEXT_WAVE = {
    "1": "2",
    "2": "3",
    "3": "4",
    "4": "5",
    "5": "A",
    "A": "B",
    "B": "C",
    "C": "1",
}
LABEL_MAP = {
    "1": "Impulswelle 1",
    "2": "Korrekturwelle 2",
    "3": "Impulswelle 3",
    "4": "Korrekturwelle 4",
    "5": "Impulswelle 5",
    "A": "Korrekturwelle A",
    "B": "Korrekturwelle B",
    "C": "Korrekturwelle C",
    "TRIANGLE": "Triangle (Seitwärts)",
    "ZIGZAG": "ZigZag (Korrektur)",
    "DOUBLE_ZIGZAG": "Double ZigZag",
    "FLAT": "Flat (Seitwärts)",
    "RUNNING_FLAT": "Running Flat",
    "EXPANDED_FLAT": "Expanded Flat",
    "TREND_REVERSAL": "Trend Reversal",
    "FALSE_BREAKOUT": "False Breakout",
    "GAP_EXTENSION": "Gap Extension",
    "LEADING_DIAGONAL": "Leading Diagonal",
    "ENDING_DIAGONAL": "Ending Diagonal",
    "RUNNING_TRIANGLE": "Running Triangle",
    "CONTRACTING_TRIANGLE": "Contracting Triangle",
    "FLAT_ZIGZAG": "Flat + ZigZag",
    "TRUNCATED_5": "Truncated Fifth",
    "EXTENDED_5": "Extended Fifth",
    "W": "W",
    "X": "Zwischenwelle",
    "Y": "Y",
    "Z": "Z",
    "WXY": "Double Three",
    "WXYXZ": "Triple Three",
    "WXYXZY": "Complex Triple Three",
    "N": "Kein Muster",
    "INVALID_WAVE": "Ungültig",
}

LABELS = [k for k in LABEL_MAP if k not in ("N", "INVALID_WAVE")]


def bold(x):
    """Return text without ANSI color codes."""
    return str(x)


def blue(x):
    return str(x)


def red(x):
    return str(x)


def green(x):
    return str(x)


def yellow(x):
    return str(x)


# === Indikator-Berechnungen ===
def calc_rsi(series, period=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta > 0, 0, -delta)
    gain_series = pd.Series(gain, index=series.index)
    loss_series = pd.Series(loss, index=series.index)
    avg_gain = gain_series.rolling(window=period, min_periods=period).mean()
    avg_loss = loss_series.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    return 100 - (100 / (1 + rs))


def calc_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def calc_stoch_kd(df, k=14, d=3):
    low_min = df["low"].rolling(window=k).min()
    high_max = df["high"].rolling(window=k).max()
    stoch_k = 100 * (df["close"] - low_min) / (high_max - low_min + 1e-6)
    stoch_d = stoch_k.rolling(window=d).mean()
    return stoch_k, stoch_d


def calc_obv(df):
    obv = [0]
    for i in range(1, len(df)):
        change = np.sign(df["close"].iloc[i] - df["close"].iloc[i - 1])
        obv.append(obv[-1] + df["volume"].iloc[i] * change)
    return pd.Series(obv, index=df.index)


def calc_atr(df, period=14):
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(period).mean()
    return atr


def calc_klinger(df, fast=34, slow=55, signal=13):
    dm = df["high"] - df["low"]
    cm = df["close"] - df["open"]
    vf = np.abs((dm + cm) * df["volume"])
    ema_fast = vf.ewm(span=fast, adjust=False).mean()
    ema_slow = vf.ewm(span=slow, adjust=False).mean()
    kvo = ema_fast - ema_slow
    kvo_signal = kvo.ewm(span=signal, adjust=False).mean()
    return kvo, kvo_signal


def calc_cmf(df, period=20):
    mfv = ((df["close"] - df["low"]) - (df["high"] - df["close"])) * df[
        "volume"
    ]
    mfv_sum = mfv.rolling(window=period).sum()
    vol_sum = df["volume"].rolling(window=period).sum()
    return mfv_sum / (vol_sum + 1e-8)


def calc_slope(series, window=5):
    """Rolling slope of the given series using linear regression."""
    idx = np.arange(window)
    slopes = []
    for i in range(len(series)):
        if i < window - 1:
            slopes.append(np.nan)
            continue
        y = series.iloc[i - window + 1: i + 1]
        m, _ = np.polyfit(idx, y, 1)
        slopes.append(m)
    return pd.Series(slopes, index=series.index)


def calc_trend_length(series):
    """Length of consecutive positive/negative returns."""
    direction = np.sign(series)
    groups = (direction != direction.shift()).cumsum()
    return direction.groupby(groups).cumcount() + 1


def calc_pattern_confidence(series, window=10):
    """Estimate cleanliness of a price series (0-1)."""
    smoothed = series.rolling(window).mean()
    deviation = (series - smoothed).abs().rolling(window).mean()
    amplitude = series.rolling(window).max() - series.rolling(window).min()
    score = 1 - deviation / (amplitude + 1e-8)
    return score.clip(lower=0, upper=1).bfill()


def calc_wave_structure_score(df, label_col="wave"):
    if label_col not in df.columns:
        return pd.Series(1.0, index=df.index)

    def wave_len(w):
        seg = df[df[label_col] == w]
        return (
            abs(seg["close"].iloc[-1] - seg["close"].iloc[0])
            if len(seg) > 1
            else 0
        )

    w1 = df[df[label_col] == "1"]["close"]
    w2 = df[df[label_col] == "2"]["close"]
    w3 = df[df[label_col] == "3"]["close"]
    w4 = df[df[label_col] == "4"]["close"]
    w5 = df[df[label_col] == "5"]["close"]

    checks = 0
    score = 0.0

    if not w1.empty and not w2.empty:
        w1_len = wave_len("1")
        w1_start = w1.iloc[0]
        w2_min = w2.min()
        checks += 1
        if w2_min >= w1_start + 0.01 * w1_len:
            score += 1

    if not w1.empty and not w3.empty and not w5.empty:
        w1_len = wave_len("1")
        w3_len = wave_len("3")
        w5_len = wave_len("5")
        checks += 1
        if w3_len > w1_len and w3_len > w5_len:
            score += 1

    if not w1.empty and not w4.empty:
        w1_max = w1.max()
        w4_min = w4.min()
        checks += 1
        if w4_min >= w1_max:
            score += 1

    final = score / checks if checks else 1.0
    return pd.Series(final, index=df.index)


# === Synthetische Muster & Generatoren ===
def validate_impulse_elliott(df):
    def wave_len(w):
        d = df[df["wave"] == w]
        return (
            abs(d["close"].iloc[-1] - d["close"].iloc[0]) if len(d) > 1 else 0
        )

    w1_len = wave_len("1")
    w3_len = wave_len("3")
    w5_len = wave_len("5")
    if not (w3_len > w1_len * 1.05 and w3_len > w5_len * 1.05):
        df["wave"] = "INVALID_WAVE"
        return df
    w1_max = df[df["wave"] == "1"]["close"].max()
    w4_min = df[df["wave"] == "4"]["close"].min()
    if w4_min < w1_max:
        df["wave"] = "INVALID_WAVE"
        return df
    w1_start = df[df["wave"] == "1"]["close"].iloc[0]
    w2_min = df[df["wave"] == "2"]["close"].min()
    w1_end = df[df["wave"] == "1"]["close"].iloc[-1]
    w1_len = abs(w1_end - w1_start)
    if w2_min < w1_start + 0.01 * w1_len:
        df["wave"] = "INVALID_WAVE"
        return df
    w3_max = df[df["wave"] == "3"]["close"].max()
    w5_max = df[df["wave"] == "5"]["close"].max()
    if w5_max < w3_max:
        df["wave"] = "TRUNCATED_5"
    elif (
        w5_max - df[df["wave"] == "5"]["close"].min()
    ) > 1.618 * max(w1_len, w3_len):
        df["wave"] = "EXTENDED_5"
    return df


def subwaves(length, amp, noise, pattern):
    if pattern == "impulse":
        segs = [length // 5] * 5
        vals = []
        price = amp
        for i in range(5):
            if i % 2 == 0:
                segment = price + np.cumsum(
                    np.abs(np.random.normal(amp / segs[0], noise, segs[i]))
                )
            else:
                segment = price - np.cumsum(
                    np.abs(np.random.normal(amp / segs[0], noise, segs[i]))
                )
            vals.extend(segment)
            price = segment[-1]
        return np.array(vals)
    else:
        return amp + np.cumsum(np.random.normal(amp / length, noise, length))


def synthetic_elliott_wave_rulebased(
    lengths, amp, noise, puffer=PUFFER, *, noise_level: float = 0.0
):
    pattern = ["1", "2", "3", "4", "5", "A", "B", "C"]
    prices, labels = [], []
    price = amp
    wave1_high = None
    for i, (l, w) in enumerate(zip(lengths, pattern)):
        if w in ["1", "3", "5"] and l >= 15:
            seg = subwaves(l, price, noise, "impulse")
            segment = seg
            price = seg[-1]
        elif w == "2":
            tentative = price - np.cumsum(
                np.abs(np.random.normal(amp / l, noise, l))
            )
            max_level = wave1_high * (1 + puffer) if wave1_high else price
            segment = np.minimum(tentative, max_level)
            price = segment[-1]
        elif w == "4":
            segment = price - np.cumsum(
                np.abs(np.random.normal(amp / l, noise, l))
            )
            price = segment[-1]
        else:
            segment = price - np.cumsum(
                np.abs(np.random.normal(amp / l, noise, l))
            )
            price = segment[-1]
        prices.extend(segment)
        labels.extend([w] * l)
        if w == "1":
            wave1_high = segment[-1]
    n = min(len(prices), len(labels))
    df = pd.DataFrame({"close": prices[:n], "wave": labels[:n]})
    df["open"] = df["close"].shift(1).fillna(df["close"][0])
    df["high"] = np.maximum(df["open"], df["close"]) + np.random.uniform(
        0, 1, len(df)
    )
    df["low"] = np.minimum(df["open"], df["close"]) - np.random.uniform(
        0, 1, len(df)
    )
    df["volume"] = np.random.uniform(100, 1000, len(df))
    df = validate_impulse_elliott(df)
    df = _apply_ohlc_noise(df.reset_index(drop=True), noise_level)
    return df


@register_pattern("TRIANGLE", next_wave=["5", "C"])
def synthetic_triangle_pattern(length=40, amp=100, noise=3, *, noise_level: float = 0.0):
    base = amp
    step = amp * 0.04
    a = base + np.cumsum(np.random.normal(step, noise, length // 5))
    b = a[-1] - np.cumsum(
        np.abs(np.random.normal(step * 0.8, noise, length // 5))
    )
    c = b[-1] + np.cumsum(np.random.normal(step * 0.7, noise, length // 5))
    d = c[-1] - np.cumsum(
        np.abs(np.random.normal(step * 0.5, noise, length // 5))
    )
    e = d[-1] + np.cumsum(
        np.random.normal(step * 0.4, noise, length - length // 5 * 4)
    )
    prices = np.concatenate([a, b, c, d, e])
    labels = ["TRIANGLE"] * len(prices)
    n = min(len(prices), len(labels))
    df = pd.DataFrame({"close": prices[:n], "wave": labels[:n]})
    df["open"] = df["close"].shift(1).fillna(df["close"][0])
    df["high"] = np.maximum(df["open"], df["close"]) + np.random.uniform(
        0, 1, len(df)
    )
    df["low"] = np.minimum(df["open"], df["close"]) - np.random.uniform(
        0, 1, len(df)
    )
    df["volume"] = np.random.uniform(100, 1000, len(df))
    df = _apply_ohlc_noise(df, noise_level)
    return df


@register_pattern("ZIGZAG", next_wave=["B", "3"])
def synthetic_zigzag_pattern(length=30, amp=100, noise=3, *, noise_level: float = 0.0):
    len1 = length // 2
    len2 = length // 4
    len3 = length - len1 - len2
    a = amp + np.cumsum(np.random.normal(amp / len1, noise, len1))
    b = a[-1] - np.cumsum(np.abs(np.random.normal(amp / len2, noise, len2)))
    c = b[-1] + np.cumsum(np.random.normal(amp / len3, noise, len3))
    prices = np.concatenate([a, b, c])
    labels = ["ZIGZAG"] * len(prices)
    n = min(len(prices), len(labels))
    df = pd.DataFrame({"close": prices[:n], "wave": labels[:n]})
    df["open"] = df["close"].shift(1).fillna(df["close"][0])
    df["high"] = np.maximum(df["open"], df["close"]) + np.random.uniform(
        0, 1, len(df)
    )
    df["low"] = np.minimum(df["open"], df["close"]) - np.random.uniform(
        0, 1, len(df)
    )
    df["volume"] = np.random.uniform(100, 1000, len(df))
    df = _apply_ohlc_noise(df, noise_level)
    return df


@register_pattern("FLAT", next_wave=["3", "5", "C"])
def synthetic_flat_pattern(length=30, amp=100, noise=3, *, noise_level: float = 0.0):
    len1 = length // 3
    len2 = length // 3
    len3 = length - len1 - len2
    a = amp + np.cumsum(np.random.normal(amp / len1, noise, len1))
    b = a[-1] + np.cumsum(np.random.normal(amp / len2, noise, len2))
    c = b[-1] - np.cumsum(np.abs(np.random.normal(amp / len3, noise, len3)))
    prices = np.concatenate([a, b, c])
    labels = ["FLAT"] * len(prices)
    n = min(len(prices), len(labels))
    df = pd.DataFrame({"close": prices[:n], "wave": labels[:n]})
    df["open"] = df["close"].shift(1).fillna(df["close"][0])
    df["high"] = np.maximum(df["open"], df["close"]) + np.random.uniform(
        0, 1, len(df)
    )
    df["low"] = np.minimum(df["open"], df["close"]) - np.random.uniform(
        0, 1, len(df)
    )
    df["volume"] = np.random.uniform(100, 1000, len(df))
    df = _apply_ohlc_noise(df, noise_level)
    return df


@register_pattern("DOUBLE_ZIGZAG", next_wave=["C", "5"])
def synthetic_double_zigzag_pattern(length=50, amp=100, noise=3, *, noise_level: float = 0.0):
    zz1 = synthetic_zigzag_pattern(length // 2, amp, noise)["close"]
    zz2 = synthetic_zigzag_pattern(length // 2, amp * 0.9, noise)["close"]
    prices = np.concatenate([zz1, zz2])
    labels = ["DOUBLE_ZIGZAG"] * len(prices)
    df = pd.DataFrame({"close": prices, "wave": labels})
    df["open"] = df["close"].shift(1).fillna(df["close"][0])
    df["high"] = np.maximum(df["open"], df["close"]) + np.random.uniform(
        0, 1, len(df)
    )
    df["low"] = np.minimum(df["open"], df["close"]) - np.random.uniform(
        0, 1, len(df)
    )
    df["volume"] = np.random.uniform(100, 1000, len(df))
    df = _apply_ohlc_noise(df, noise_level)
    return df


@register_pattern("RUNNING_FLAT")
def synthetic_running_flat_pattern(length=30, amp=100, noise=3, *, noise_level: float = 0.0):
    a = amp + np.cumsum(np.random.normal(amp / length, noise, length // 3))
    b = a[-1] + np.cumsum(np.random.normal(amp / length, noise, length // 3))
    c = b[-1] + np.cumsum(
        np.random.normal(amp / length, noise, length - length // 3 * 2)
    )
    prices = np.concatenate([a, b, c])
    labels = ["RUNNING_FLAT"] * len(prices)
    df = pd.DataFrame({"close": prices, "wave": labels})
    df["open"] = df["close"].shift(1).fillna(df["close"][0])
    df["high"] = np.maximum(df["open"], df["close"]) + np.random.uniform(
        0, 1, len(df)
    )
    df["low"] = np.minimum(df["open"], df["close"]) - np.random.uniform(
        0, 1, len(df)
    )
    df["volume"] = np.random.uniform(100, 1000, len(df))
    df = _apply_ohlc_noise(df, noise_level)
    return df


@register_pattern("EXPANDED_FLAT", next_wave=["3", "5", "C"])
def synthetic_expanded_flat_pattern(length=30, amp=100, noise=3, *, noise_level: float = 0.0):
    a = amp + np.cumsum(np.random.normal(amp / length, noise, length // 3))
    b = a[-1] + np.cumsum(
        np.random.normal(amp / length * 2, noise, length // 3)
    )
    c = b[-1] - np.cumsum(
        np.abs(np.random.normal(amp / length, noise, length - length // 3 * 2))
    )
    prices = np.concatenate([a, b, c])
    labels = ["EXPANDED_FLAT"] * len(prices)
    df = pd.DataFrame({"close": prices, "wave": labels})
    df["open"] = df["close"].shift(1).fillna(df["close"][0])
    df["high"] = np.maximum(df["open"], df["close"]) + np.random.uniform(
        0, 1, len(df)
    )
    df["low"] = np.minimum(df["open"], df["close"]) - np.random.uniform(
        0, 1, len(df)
    )
    df["volume"] = np.random.uniform(100, 1000, len(df))
    df = _apply_ohlc_noise(df, noise_level)
    return df


@register_pattern("TREND_REVERSAL")
def synthetic_trend_reversal_pattern(
    length=40, amp=100, noise=3, gap_chance=0.1, *, noise_level: float = 0.0
):
    up_len = length // 2
    down_len = length - up_len
    up = amp + np.cumsum(np.abs(np.random.normal(amp / up_len, noise, up_len)))
    gap = (
        np.random.uniform(amp * 0.05, amp * 0.15)
        if np.random.rand() < gap_chance
        else 0
    )
    start_down = up[-1] + gap
    down = synthetic_zigzag_pattern(down_len, start_down, noise)["close"]
    prices = np.concatenate([up, down])
    labels = ["TREND_REVERSAL"] * len(prices)
    df = pd.DataFrame({"close": prices, "wave": labels})
    df["open"] = df["close"].shift(1).fillna(df["close"][0])
    df["high"] = np.maximum(df["open"], df["close"]) + np.random.uniform(
        0, 1, len(df)
    )
    df["low"] = np.minimum(df["open"], df["close"]) - np.random.uniform(
        0, 1, len(df)
    )
    df["volume"] = np.random.uniform(100, 1000, len(df))
    df = _apply_ohlc_noise(df, noise_level)
    return df


@register_pattern("FALSE_BREAKOUT")
def synthetic_false_breakout_pattern(
    length=40, amp=100, noise=3, gap_chance=0.1, *, noise_level: float = 0.0
):
    base_len = length // 3
    breakout_len = length // 4
    end_len = length - base_len - breakout_len
    base = amp + np.cumsum(np.random.normal(0, noise, base_len))
    direction = 1 if np.random.rand() < 0.5 else -1
    breakout = base[-1] + direction * np.cumsum(
        np.abs(np.random.normal(amp / breakout_len, noise, breakout_len))
    )
    if np.random.rand() < gap_chance:
        breakout[0] = base[-1] + direction * np.random.uniform(
            amp * 0.05, amp * 0.1
        )
    ret = breakout[-1] - direction * np.cumsum(
        np.abs(np.random.normal(amp / end_len, noise, end_len))
    )
    prices = np.concatenate([base, breakout, ret])
    labels = ["FALSE_BREAKOUT"] * len(prices)
    df = pd.DataFrame({"close": prices, "wave": labels})
    df["open"] = df["close"].shift(1).fillna(df["close"][0])
    df["high"] = np.maximum(df["open"], df["close"]) + np.random.uniform(
        0, 1, len(df)
    )
    df["low"] = np.minimum(df["open"], df["close"]) - np.random.uniform(
        0, 1, len(df)
    )
    df["volume"] = np.random.uniform(100, 1000, len(df))
    df = _apply_ohlc_noise(df, noise_level)
    return df


@register_pattern("GAP_EXTENSION")
def synthetic_gap_extension_pattern(length=40, amp=100, noise=3, *, noise_level: float = 0.0):
    l1 = length // 3
    l2 = length // 3
    l3 = length - l1 - l2
    first = amp + np.cumsum(np.random.normal(amp / l1, noise, l1))
    gap = np.random.uniform(-amp * 0.15, amp * 0.15)
    mid_start = first[-1] + gap
    mid_df = synthetic_triangle_pattern(l2, amp, noise)
    mid = mid_df["close"].to_numpy()
    mid = mid - mid[0] + mid_start
    ext = mid[-1] + np.cumsum(np.random.normal(amp / l3, noise, l3))
    prices = np.concatenate([first, mid, ext])
    labels = ["GAP_EXTENSION"] * len(prices)
    df = pd.DataFrame({"close": prices, "wave": labels})
    df["open"] = df["close"].shift(1).fillna(df["close"][0])
    df["high"] = np.maximum(df["open"], df["close"]) + np.random.uniform(
        0, 1, len(df)
    )
    df["low"] = np.minimum(df["open"], df["close"]) - np.random.uniform(
        0, 1, len(df)
    )
    df["volume"] = np.random.uniform(100, 1000, len(df))
    df = _apply_ohlc_noise(df, noise_level)
    return df


@register_pattern("WXY", next_wave=["Z", "Abschluss"])
def synthetic_wxy_pattern(length=60, amp=100, noise=3, *, noise_level: float = 0.0):
    lw = length // 3
    lx = length // 6
    ly = length - lw - lx
    w = amp + np.cumsum(np.random.normal(amp / lw, noise, lw))
    x = w[-1] - np.cumsum(np.abs(np.random.normal(amp / lx, noise, lx)))
    y = x[-1] + np.cumsum(np.random.normal(amp / ly, noise, ly))
    prices = np.concatenate([w, x, y])
    labels = ["W"] * lw + ["X"] * lx + ["Y"] * ly
    df = pd.DataFrame({"close": prices, "wave": labels})
    df["open"] = df["close"].shift(1).fillna(df["close"][0])
    df["high"] = np.maximum(df["open"], df["close"]) + np.random.uniform(
        0, 1, len(df)
    )
    df["low"] = np.minimum(df["open"], df["close"]) - np.random.uniform(
        0, 1, len(df)
    )
    df["volume"] = np.random.uniform(100, 1000, len(df))
    df = _apply_ohlc_noise(df, noise_level)
    return df


@register_pattern("WXYXZ", next_wave=["Abschluss"])
def synthetic_wxyxz_pattern(length=80, amp=100, noise=3, *, noise_level: float = 0.0):
    seg = length // 5
    w = amp + np.cumsum(np.random.normal(amp / seg * 2, noise, seg * 2))
    x1 = w[-1] - np.cumsum(np.abs(np.random.normal(amp / seg, noise, seg)))
    y = x1[-1] + np.cumsum(np.random.normal(amp / seg, noise, seg))
    x2 = y[-1] - np.cumsum(
        np.abs(np.random.normal(amp / (length - seg * 4), noise, seg // 2))
    )
    z = x2[-1] + np.cumsum(
        np.random.normal(
            amp / (length - seg * 4), noise, length - seg * 4 - seg // 2
        )
    )
    prices = np.concatenate([w, x1, y, x2, z])
    labels = (
        ["W"] * (seg * 2)
        + ["X"] * seg
        + ["Y"] * seg
        + ["X"] * (seg // 2)
        + ["Z"] * (length - seg * 4 - seg // 2)
    )
    df = pd.DataFrame({"close": prices, "wave": labels})
    df["open"] = df["close"].shift(1).fillna(df["close"][0])
    df["high"] = np.maximum(df["open"], df["close"]) + np.random.uniform(
        0, 1, len(df)
    )
    df["low"] = np.minimum(df["open"], df["close"]) - np.random.uniform(
        0, 1, len(df)
    )
    df["volume"] = np.random.uniform(100, 1000, len(df))
    df = _apply_ohlc_noise(df, noise_level)
    return df


@register_pattern("WXYXZY")
def synthetic_wxyxzy_pattern(length=100, amp=100, noise=3, *, noise_level: float = 0.0):
    seg = length // 6
    w = amp + np.cumsum(np.random.normal(amp / seg, noise, seg))
    x1 = w[-1] - np.cumsum(np.abs(np.random.normal(amp / seg, noise, seg)))
    y1 = x1[-1] + np.cumsum(np.random.normal(amp / seg, noise, seg))
    x2 = y1[-1] - np.cumsum(np.abs(np.random.normal(amp / seg, noise, seg)))
    z = x2[-1] + np.cumsum(np.random.normal(amp / seg, noise, seg))
    y2 = z[-1] - np.cumsum(
        np.abs(np.random.normal(amp / seg, noise, length - seg * 5))
    )
    prices = np.concatenate([w, x1, y1, x2, z, y2])
    labels = (
        ["W"] * seg
        + ["X"] * seg
        + ["Y"] * seg
        + ["X"] * seg
        + ["Z"] * seg
        + ["Y"] * (length - seg * 5)
    )
    df = pd.DataFrame({"close": prices, "wave": labels})
    df["open"] = df["close"].shift(1).fillna(df["close"][0])
    df["high"] = np.maximum(df["open"], df["close"]) + np.random.uniform(
        0, 1, len(df)
    )
    df["low"] = np.minimum(df["open"], df["close"]) - np.random.uniform(
        0, 1, len(df)
    )
    df["volume"] = np.random.uniform(100, 1000, len(df))
    df = _apply_ohlc_noise(df, noise_level)
    return df


@register_pattern("LEADING_DIAGONAL", next_wave=["3"])
def synthetic_leading_diagonal_pattern(length=40, amp=100, noise=3, *, noise_level: float = 0.0):
    seg = length // 5
    step = amp * 0.04
    w1 = amp + np.cumsum(
        np.random.normal(step * 1.0, noise, seg)
    )
    w2 = w1[-1] - np.cumsum(
        np.abs(np.random.normal(step * 0.8, noise, seg))
    )
    w3 = w2[-1] + np.cumsum(
        np.random.normal(step * 1.2, noise, seg)
    )
    w4 = w3[-1] - np.cumsum(
        np.abs(np.random.normal(step * 0.7, noise, seg))
    )
    w5 = w4[-1] + np.cumsum(
        np.random.normal(step * 1.1, noise, length - seg * 4)
    )
    prices = np.concatenate([w1, w2, w3, w4, w5])
    labels = ["LEADING_DIAGONAL"] * len(prices)
    df = pd.DataFrame({"close": prices, "wave": labels})
    df["open"] = df["close"].shift(1).fillna(df["close"][0])
    df["high"] = (
        np.maximum(df["open"], df["close"]) + np.random.uniform(0, 1, len(df))
    )
    df["low"] = (
        np.minimum(df["open"], df["close"]) - np.random.uniform(0, 1, len(df))
    )
    df["volume"] = np.random.uniform(100, 1000, len(df))
    df = _apply_ohlc_noise(df, noise_level)
    return df


@register_pattern("ENDING_DIAGONAL", next_wave=["A"])
def synthetic_ending_diagonal_pattern(length=40, amp=100, noise=3, *, noise_level: float = 0.0):
    seg = length // 5
    step = amp * 0.05
    w1 = amp + np.cumsum(np.random.normal(step * 1.3, noise, seg))
    w2 = w1[-1] - np.cumsum(np.abs(np.random.normal(step * 1.1, noise, seg)))
    w3 = w2[-1] + np.cumsum(np.random.normal(step * 1.2, noise, seg))
    w4 = w3[-1] - np.cumsum(np.abs(np.random.normal(step, noise, seg)))
    w5 = w4[-1] + np.cumsum(
        np.random.normal(step * 1.4, noise, length - seg * 4)
    )
    prices = np.concatenate([w1, w2, w3, w4, w5])
    labels = ["ENDING_DIAGONAL"] * len(prices)
    df = pd.DataFrame({"close": prices, "wave": labels})
    df["open"] = df["close"].shift(1).fillna(df["close"][0])
    df["high"] = (
        np.maximum(df["open"], df["close"]) + np.random.uniform(0, 1, len(df))
    )
    df["low"] = (
        np.minimum(df["open"], df["close"]) - np.random.uniform(0, 1, len(df))
    )
    df["volume"] = np.random.uniform(100, 1000, len(df))
    df = _apply_ohlc_noise(df, noise_level)
    return df


@register_pattern("RUNNING_TRIANGLE", next_wave=["5", "C"])
def synthetic_running_triangle_pattern(length=40, amp=100, noise=3, *, noise_level: float = 0.0):
    seg = length // 5
    step = amp * 0.04
    a = amp + np.cumsum(np.random.normal(step, noise, seg))
    b = a[-1] - np.cumsum(np.abs(np.random.normal(step * 0.8, noise, seg)))
    c = b[-1] + np.cumsum(np.random.normal(step * 0.6, noise, seg))
    d = c[-1] - np.cumsum(np.abs(np.random.normal(step * 0.5, noise, seg)))
    e = d[-1] + np.cumsum(
        np.random.normal(step * 0.7, noise, length - seg * 4)
    )
    prices = np.concatenate([a, b, c, d, e])
    labels = ["RUNNING_TRIANGLE"] * len(prices)
    df = pd.DataFrame({"close": prices, "wave": labels})
    df["open"] = df["close"].shift(1).fillna(df["close"][0])
    df["high"] = (
        np.maximum(df["open"], df["close"]) + np.random.uniform(0, 1, len(df))
    )
    df["low"] = (
        np.minimum(df["open"], df["close"]) - np.random.uniform(0, 1, len(df))
    )
    df["volume"] = np.random.uniform(100, 1000, len(df))
    df = _apply_ohlc_noise(df, noise_level)
    return df


@register_pattern("CONTRACTING_TRIANGLE", next_wave=["5", "C"])
def synthetic_contracting_triangle_pattern(length=40, amp=100, noise=3, *, noise_level: float = 0.0):
    seg = length // 5
    step = amp * 0.05
    a = amp + np.cumsum(np.random.normal(step, noise, seg))
    b = a[-1] - np.cumsum(np.abs(np.random.normal(step * 0.9, noise, seg)))
    c = b[-1] + np.cumsum(np.random.normal(step * 0.8, noise, seg))
    d = c[-1] - np.cumsum(np.abs(np.random.normal(step * 0.7, noise, seg)))
    e = d[-1] + np.cumsum(
        np.random.normal(step * 0.6, noise, length - seg * 4)
    )
    prices = np.concatenate([a, b, c, d, e])
    labels = ["CONTRACTING_TRIANGLE"] * len(prices)
    df = pd.DataFrame({"close": prices, "wave": labels})
    df["open"] = df["close"].shift(1).fillna(df["close"][0])
    df["high"] = (
        np.maximum(df["open"], df["close"]) + np.random.uniform(0, 1, len(df))
    )
    df["low"] = (
        np.minimum(df["open"], df["close"]) - np.random.uniform(0, 1, len(df))
    )
    df["volume"] = np.random.uniform(100, 1000, len(df))
    df = _apply_ohlc_noise(df, noise_level)
    return df


@register_pattern("FLAT_ZIGZAG", next_wave=["C", "5"])
def synthetic_flat_zigzag_series_pattern(length=60, amp=100, noise=3, *, noise_level: float = 0.0):
    half = length // 2
    flat = synthetic_flat_pattern(half, amp, noise)["close"]
    zz = synthetic_zigzag_pattern(length - half, flat.iloc[-1], noise)["close"]
    prices = np.concatenate([flat, zz])
    labels = ["FLAT_ZIGZAG"] * len(prices)
    df = pd.DataFrame({"close": prices, "wave": labels})
    df["open"] = df["close"].shift(1).fillna(df["close"][0])
    df["high"] = (
        np.maximum(df["open"], df["close"]) + np.random.uniform(0, 1, len(df))
    )
    df["low"] = (
        np.minimum(df["open"], df["close"]) - np.random.uniform(0, 1, len(df))
    )
    df["volume"] = np.random.uniform(100, 1000, len(df))
    df = _apply_ohlc_noise(df, noise_level)
    return df


def generate_negative_samples(
    length=100,
    amp=100,
    noise=20,
    outlier_chance=0.1,
    gap_chance=0.1,
    *,
    noise_level: float = 0.0,
):
    prices = amp + np.cumsum(np.random.normal(0, noise, length))
    for i in range(1, length):
        if np.random.rand() < gap_chance:
            gap = np.random.normal(0, noise * 5)
            prices[i:] += gap
    mask = np.random.rand(length) < outlier_chance
    prices[mask] += np.random.normal(0, noise * 10, mask.sum())
    labels = ["N"] * len(prices)
    n = min(len(prices), len(labels))
    df = pd.DataFrame({"close": prices[:n], "wave": labels[:n]})
    df["open"] = df["close"].shift(1).fillna(df["close"][0])
    df["high"] = np.maximum(df["open"], df["close"]) + np.random.uniform(
        0, 1, len(df)
    )
    df["low"] = np.minimum(df["open"], df["close"]) - np.random.uniform(
        0, 1, len(df)
    )
    df["volume"] = np.random.uniform(100, 1000, len(df))
    df = _apply_ohlc_noise(df, noise_level)
    return df


def _simple_wave_segment(label, start_price, length=8, noise=2, *, noise_level: float = 0.0):
    """Create a short price segment for a single wave label."""
    direction = 1 if np.random.rand() < 0.5 else -1
    step = max(start_price * 0.02, 1)
    prices = start_price + direction * np.cumsum(
        np.abs(np.random.normal(step, noise, length))
    )
    df = pd.DataFrame({"close": prices, "wave": [label] * length})
    df["open"] = df["close"].shift(1).fillna(df["close"][0])
    df["high"] = np.maximum(df["open"], df["close"]) + np.random.uniform(
        0, 1, len(df)
    )
    df["low"] = np.minimum(df["open"], df["close"]) - np.random.uniform(
        0, 1, len(df)
    )
    df["volume"] = np.random.uniform(100, 1000, len(df))
    df = _apply_ohlc_noise(df, noise_level)
    return df


def generate_balanced_elliott_dataset(
    n_total: int = TRAIN_N,
    invalid_share: float = INVALID_SHARE,
    n_share: float = N_SHARE,
    pattern_registry: PatternRegistry = pattern_registry,
    log: bool = True,
    test_mode: bool = False,
    test_label_limit: int = 100,
    *,
    noise_level: float = 0.0,
    balance: bool = True,
) -> pd.DataFrame:
    """Generate balanced dataset across all LABELS.

    Optionally include invalid or "N" data.

    Parameters
    ----------
    n_total : int
        Total sample count when ``test_mode`` is False.
    invalid_share : float
        Fraction of invalid samples per label.
    n_share : float
        Fraction of ``N`` samples when ``test_mode`` is False.
    test_mode : bool, optional
        If ``True``, create exactly ``test_label_limit`` samples per label and
        for the ``N`` class, ignoring ``n_total``.
    test_label_limit : int, optional
        Number of samples per label when ``test_mode`` is enabled.
    balance : bool, optional
        When ``True``, resample rows per label so each class has the same
        number of rows after generation.

    Returns
    -------
    pd.DataFrame
        Generated dataset.
    """

    if test_mode:
        n_per_label = test_label_limit
        n_invalid_per_label = int(test_label_limit * invalid_share)
        n_n = test_label_limit if "N" in LABEL_MAP else 0
    else:
        n_n = int(n_total * n_share)
        n_per_label = int(
            (n_total * (1 - n_share)) // (len(LABELS) * (1 + invalid_share))
        )
        n_invalid_per_label = int(n_per_label * invalid_share)

    dfs: List[pd.DataFrame] = []
    total_iterations = len(LABELS) * (n_per_label + n_invalid_per_label) + n_n
    with alive_it(total_iterations, disable=not log) as bar:
        for label in LABELS:
            for _ in range(n_per_label):
                if pattern_registry.has_pattern(label):
                    gen = pattern_registry._patterns[label]
                    length = np.random.randint(30, 60)
                    amp = np.random.uniform(80, 150)
                    noise = np.random.uniform(1, 3.5)
                    df = gen(length=length, amp=amp, noise=noise, noise_level=noise_level)
                    df["wave"] = label
                else:
                    df = synthetic_elliott_wave_rulebased(
                        lengths=np.random.randint(15, 40, size=8),
                        amp=np.random.uniform(80, 150),
                        noise=np.random.uniform(1, 3.0),
                        noise_level=noise_level,
                    )
                    df["wave"] = label
                dfs.append(df)
                bar()
            for _ in range(n_invalid_per_label):
                if pattern_registry.has_pattern(label):
                    gen = pattern_registry._patterns[label]
                    length = np.random.randint(30, 60)
                    amp = np.random.uniform(80, 150)
                    noise = np.random.uniform(4, 8)
                    df = gen(length=length, amp=amp, noise=noise, noise_level=noise_level)
                    df = df.sample(frac=1).reset_index(drop=True)
                    df["wave"] = "INVALID_WAVE"
                else:
                    df = synthetic_elliott_wave_rulebased(
                        lengths=np.random.randint(15, 40, size=8),
                        amp=np.random.uniform(80, 150),
                        noise=np.random.uniform(5, 10),
                        noise_level=noise_level,
                    )
                    df = df.sample(frac=1).reset_index(drop=True)
                    df["wave"] = "INVALID_WAVE"
                dfs.append(df)
                bar()

        for _ in range(n_n):
            length = np.random.randint(40, 100)
            amp = np.random.uniform(60, 140)
            noise = np.random.uniform(15, 40)
            df = generate_negative_samples(
                length=length,
                amp=amp,
                noise=noise,
                outlier_chance=0.2,
                gap_chance=0.25,
                noise_level=noise_level,
            )
            df["wave"] = "N"
            dfs.append(df)
            bar()

    all_data = pd.concat(dfs, ignore_index=True)
    if balance:
        counts = all_data["wave"].value_counts()
        target_count = counts.max()
        if log:
            print("Label-Verteilung vor Balancing:", counts.to_dict())
            print(f"Balancing auf {target_count} Zeilen pro Label")

        resampled: List[pd.DataFrame] = []
        for label, count in counts.items():
            subset = all_data[all_data["wave"] == label]
            replace = count < target_count
            subset_bal = resample(
                subset,
                replace=replace,
                n_samples=target_count,
                random_state=0,
            )
            resampled.append(subset_bal)
        all_data = pd.concat(resampled, ignore_index=True)

    if log:
        print(f"Fertiges Balancing: Gesamtanzahl: {len(all_data)}")
        # print("Label-Verteilung:\n", all_data["wave"].value_counts())
    return all_data


def generate_rulebased_synthetic_with_patterns(
    n: int = 1000,
    negative_ratio: float = 0.15,
    pattern_ratio: float = 0.35,
    log: bool = True,
) -> pd.DataFrame:
    """Generate synthetic dataset consisting of Elliott waves, patterns and
    noise samples.

    When ``log`` is True, progress information is printed.
    """

    num_pattern = int(n * pattern_ratio)
    num_neg = int(n * negative_ratio)
    num_pos = n - num_pattern - num_neg

    total_steps = num_pos + num_pattern + num_neg

    dfs = []
    with alive_it(total_steps, disable=not log) as bar:
        for i in range(num_pos):
            if log:
                bar.title = "Positives"
            lengths = np.random.randint(12, 50, size=8)
            amp = np.random.uniform(60, 150)
            noise = np.random.uniform(1, 4)
            df = synthetic_elliott_wave_rulebased(lengths, amp, noise)
            dfs.append(df)
            bar()

        pattern_funcs = pattern_registry.generators()
        for i in range(num_pattern):
            if log:
                bar.title = "Patterns"
            segs = []
            for _ in range(np.random.randint(1, 3)):
                f, pname = random.choice(pattern_funcs)
                length = np.random.randint(32, 70)
                amp = np.random.uniform(60, 140)
                noise = np.random.uniform(1, 3.5)
                d = f(length=length, amp=amp, noise=noise)
                if np.random.rand() < 0.3:
                    cut = np.random.randint(len(d) // 2, len(d))
                    d = d.iloc[:cut]
                segs.append(d)
                nxt = pattern_registry.get_next_wave(pname)
                if nxt:
                    nxt = nxt if isinstance(nxt, list) else [nxt]
                    start = d["close"].iloc[-1]
                    for wave in nxt:
                        if wave == "Abschluss":
                            continue
                        follow_len = np.random.randint(5, 12)
                        follow = _simple_wave_segment(
                            wave, start, length=follow_len
                        )
                        segs.append(follow)
                        start = follow["close"].iloc[-1]
            df = pd.concat(segs, ignore_index=True)
            dfs.append(df)
            bar()

        for i in range(num_neg):
            if log:
                bar.title = "Noise"
            length = np.random.randint(80, 250)
            amp = np.random.uniform(50, 120)
            noise = np.random.uniform(12, 35)
            df = generate_negative_samples(
                length=length,
                amp=amp,
                noise=noise,
                outlier_chance=0.2,
                gap_chance=0.2,
            )
            dfs.append(df)
            bar()

    if log:
        print()

    combined = pd.concat(dfs, ignore_index=True)
    if log:
        print(
            f"[DataGen] Fertig – Gesamtanzahl Datenpunkte: {len(combined)}"
        )
    return combined


def synthetic_subwaves(df, minlen=4, maxlen=9, *, log: bool = False):
    df = df.copy()
    subwave_id = np.zeros(len(df), dtype=int)
    i = 0
    with alive_it(len(df), disable=not log) as bar:
        while i < len(df):
            sublen = np.random.randint(minlen, maxlen)
            if i + sublen > len(df):
                sublen = len(df) - i
            subwave_id[i: i + sublen] = np.arange(1, sublen + 1)
            i += sublen
            bar(sublen)
    df["subwave"] = subwave_id[: len(df)]
    return df


def compute_wave_fibs(
    df,
    label_col: str = "wave_pred",
    buffer: float = PUFFER,
    *,
    log: bool = False,
):
    """Add fibonacci levels per wave segment to ``df``.

    Parameters
    ----------
    df : DataFrame
        OHLCV data with wave labels/predictions.
    label_col : str, optional
        Column containing wave identifiers.
    buffer : float, optional
        Tolerance for the ``wave_fib_near`` flag.
    """
    if label_col not in df.columns:
        return df

    df = df.copy()
    df["wave_fib_dist"] = np.nan
    df["wave_fib_near"] = np.nan

    start = 0
    cur = df[label_col].iloc[0]
    with alive_it(len(df), disable=not log) as bar:
        for i in range(1, len(df) + 1):
            if i == len(df) or df[label_col].iloc[i] != cur:
                end = i - 1
                start_price = df["close"].iloc[start]
                end_price = df["close"].iloc[end]
                if end_price >= start_price:
                    diff = end_price - start_price
                    fibs = {
                        0.0: end_price,
                        0.236: end_price - diff * 0.236,
                        0.382: end_price - diff * 0.382,
                        0.5: end_price - diff * 0.5,
                        0.618: end_price - diff * 0.618,
                        0.786: end_price - diff * 0.786,
                        1.0: start_price,
                        1.618: start_price - diff * 0.618,
                        2.618: start_price - diff * 1.618,
                    }
                else:
                    diff = start_price - end_price
                    fibs = {
                        0.0: end_price,
                        0.236: end_price + diff * 0.236,
                        0.382: end_price + diff * 0.382,
                        0.5: end_price + diff * 0.5,
                        0.618: end_price + diff * 0.618,
                        0.786: end_price + diff * 0.786,
                        1.0: start_price,
                        1.618: start_price + diff * 0.618,
                        2.618: start_price + diff * 1.618,
                    }

                idx_slice = df.index[start: end + 1]
                closes = df["close"].loc[idx_slice]
                dists = pd.DataFrame(
                    {k: (closes - v).abs() for k, v in fibs.items()}
                )
                min_dist = dists.min(axis=1)
                df.loc[idx_slice, "wave_fib_dist"] = min_dist / closes
                df.loc[idx_slice, "wave_fib_near"] = (
                    min_dist / closes <= buffer
                ).astype(int)

                bar(end - start + 1)
                start = i
                if i < len(df):
                    cur = df[label_col].iloc[i]

    return df


# === Feature Engineering mit 4H-Integration ===
def make_features(
    df,
    df_4h=None,
    levels=None,
    fib_levels=None,
    *,
    log: bool = False,
):
    df = df.copy()
    df["returns"] = df["close"].pct_change().fillna(0)
    df["range"] = (df["high"] - df["low"]) / df["close"]
    df["body"] = (df["close"] - df["open"]).abs() / df["close"]
    df["ma_fast"] = df["close"].rolling(5).mean().bfill()
    df["ma_slow"] = df["close"].rolling(34).mean().bfill()
    df["ma_diff"] = df["ma_fast"] - df["ma_slow"]
    df["vol_ratio"] = df["volume"] / (
        df["volume"].rolling(5).mean().bfill() + 1e-6
    )
    df["fibo_level"] = (df["close"] - df["close"].rolling(21).min()) / (
        df["close"].rolling(21).max() - df["close"].rolling(21).min() + 1e-6
    )
    window = 20
    counts = df.groupby(df["wave"]).cumcount() + 1 if "wave" in df else 1
    df["wave_len_ratio"] = counts / (window)
    df["rsi"] = calc_rsi(df["close"], period=14).bfill()
    df["rsi_z"] = zscore(df["rsi"].fillna(50))
    df["macd"], df["macd_signal"] = calc_macd(df["close"])
    df["stoch_k"], df["stoch_d"] = calc_stoch_kd(df)
    df["obv"] = calc_obv(df)
    df["atr"] = calc_atr(df)
    df["vol_atr_ratio"] = df["volume"] / (df["atr"] + 1e-8)
    df["kvo"], df["kvo_signal"] = calc_klinger(df)
    df["cmf"] = calc_cmf(df)
    df["high_z"] = zscore(df["high"])
    df["low_z"] = zscore(df["low"])
    df["vol_z"] = zscore(df["volume"])
    df["ema_fast"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=26, adjust=False).mean()
    df["ema_ratio"] = df["ema_fast"] / (df["ema_slow"] + 1e-8)
    bb_mid = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    df["bb_width"] = (bb_std * 4) / (bb_mid + 1e-8)
    df["roc_10"] = df["close"].pct_change(10).fillna(0)
    df["corr_close_vol_10"] = (
        df["close"].rolling(10).corr(df["volume"]).fillna(0)
    )
    df["slope_5"] = calc_slope(df["close"], window=5).bfill()
    df["trend_len"] = calc_trend_length(df["returns"]).bfill()
    df["pattern_confidence"] = calc_pattern_confidence(df["close"])
    if "wave" in df.columns:
        df["wave_structure_score"] = calc_wave_structure_score(df, "wave")
    elif "wave_pred" in df.columns:
        df["wave_structure_score"] = calc_wave_structure_score(df, "wave_pred")
    else:
        df["wave_structure_score"] = 1.0
    if "subwave" not in df.columns:
        df = synthetic_subwaves(df, log=log)
    if "wave" in df.columns:
        idx = df.groupby("wave").cumcount()
        total = df.groupby("wave")["wave"].transform("count").replace(0, 1)
        df["pos_in_pattern"] = idx / total
        df["prev_wave"] = df["wave"].shift(1).fillna("START")
        df["next_wave"] = df["wave"].shift(-1).fillna("END")
        df["prev_wave_code"] = pd.Categorical(df["prev_wave"]).codes
        df["next_wave_code"] = pd.Categorical(df["next_wave"]).codes
    else:
        df["pos_in_pattern"] = 0.0
        df["prev_wave_code"] = 0
        df["next_wave_code"] = 0
    if levels is not None:
        level_prices = np.array([lvl["price"] for lvl in levels])
        if len(level_prices):
            df["level_dist"] = df["close"].apply(
                lambda x: np.min(np.abs(level_prices - x))
            )
        else:
            df["level_dist"] = 0.0
    else:
        df["level_dist"] = 0.0

    if fib_levels is not None:
        for tf, fibs in fib_levels.items():
            prices = np.array([f["price"] for f in fibs])
            if len(prices):
                dist = df["close"].apply(
                    lambda x: np.min(np.abs(prices - x)) / x
                )
                df[f"fib_dist_{tf.lower()}"] = dist
                df[f"fib_near_{tf.lower()}"] = (dist <= 0.003).astype(int)
            else:
                df[f"fib_dist_{tf.lower()}"] = 1.0
                df[f"fib_near_{tf.lower()}"] = 0
    else:
        for tf in ["1d", "1w"]:
            df[f"fib_dist_{tf}"] = 1.0
            df[f"fib_near_{tf}"] = 0
    if df_4h is not None:
        df_4h["rsi_4h"] = calc_rsi(df_4h["close"], period=14).bfill()
        df["rsi_4h"] = np.interp(
            df.index, np.linspace(0, len(df) - 1, len(df_4h)), df_4h["rsi_4h"]
        )
        df["close_4h"] = np.interp(
            df.index, np.linspace(0, len(df) - 1, len(df_4h)), df_4h["close"]
        )
        df["vol_4h"] = np.interp(
            df.index, np.linspace(0, len(df) - 1, len(df_4h)), df_4h["volume"]
        )

    # Local fib levels for actual or predicted waves
    if "wave" in df.columns:
        df = compute_wave_fibs(df, "wave", buffer=PUFFER, log=log)
    elif "wave_pred" in df.columns:
        df = compute_wave_fibs(df, "wave_pred", buffer=PUFFER, log=log)

    df = df.dropna().reset_index(drop=True)
    return df


# === OHLCV-Import Bitget API ===
def fetch_bitget_ohlcv_auto(
    symbol, interval="1H", target_len=1000, page_limit=1000, log=False
):
    if isinstance(interval, (list, tuple, set)):
        result = {}
        for iv in interval:
            result[iv] = fetch_bitget_ohlcv_auto(
                symbol,
                iv,
                target_len=target_len,
                page_limit=page_limit,
                log=log,
            )
        return result

    all_data = []
    end_time = None
    total = 0
    with alive_it(target_len, disable=not log) as bar:
        while total < target_len:
            url = (
                "https://api.bitget.com/api/v2/mix/market/candles?"
                f"symbol={symbol}"
                f"&granularity={interval}"
                f"&productType=USDT-FUTURES&limit={page_limit}"
            )
            if end_time:
                url += f"&endTime={end_time}"
            r = requests.get(url)
            if r.status_code != 200:
                raise Exception("Bitget API-Fehler: " + r.text)
            data = r.json()["data"]
            if not data or len(data) < 2:
                break
            df = pd.DataFrame(
                data,
                columns=[
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "baseVolume",
                    "quoteVolume",
                ],
            )
            all_data.append(df)
            min_timestamp = df["timestamp"].astype(int).min()
            end_time = min_timestamp - 1
            total = sum(len(x) for x in all_data)
            bar(len(df))
        if not all_data:
            raise Exception("Keine Candles empfangen!")
    combined = pd.concat(all_data, ignore_index=True)
    combined[["open", "high", "low", "close", "baseVolume", "quoteVolume"]] = (
        combined[
            ["open", "high", "low", "close", "baseVolume", "quoteVolume"]
        ].astype(float)
    )
    combined["timestamp"] = pd.to_datetime(
        combined["timestamp"].astype(int), unit="ms"
    )
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    combined["volume"] = combined["baseVolume"]
    if len(combined) > target_len:
        combined = combined.iloc[-target_len:].reset_index(drop=True)
    if log:
        print(green(f"Loaded {len(combined)} bars for {symbol} {interval}"))
    return combined[["timestamp", "open", "high", "low", "close", "volume"]]


# === ML Training ===
def train_ml(
    skip_grid_search=False,
    max_samples=None,
    model_type="rf",
    feature_selection=False,
    test_mode: bool = False,
    test_label_limit: int = 100,
    *,
    noise_level: float = 0.0,
    log: bool = True,
):
    """Train machine learning model on generated Elliott wave data.

    Parameters
    ----------
    skip_grid_search : bool, optional
        If ``True`` skip hyper parameter search.
    max_samples : int, optional
        Limit of training samples after feature creation.
    model_type : str, optional
        One of ``rf``, ``xgb``, ``lgbm`` or ``voting``.
    feature_selection : bool, optional
        Enable RFECV feature selection.
    test_mode : bool, optional
        Generate a much smaller dataset using ``test_label_limit`` samples per
        label. Useful for quick tests.
    test_label_limit : int, optional
        Number of samples per label when ``test_mode`` is active.
    """

    if os.path.exists(DATASET_PATH):
        print(yellow("Lade vorhandenes Dataset..."))
        df = load_dataset(DATASET_PATH)
    else:
        print(
            bold(
                "Erzeuge balanciertes Datenset mit Muster- und "
                "Wellenbeispielen..."
            )
        )
        df = generate_balanced_elliott_dataset(
            n_total=TRAIN_N,
            invalid_share=INVALID_SHARE,
            n_share=N_SHARE,
            test_mode=test_mode,
            test_label_limit=test_label_limit,
            noise_level=noise_level,
        )
        save_dataset(df, DATASET_PATH)

    print(f"{blue('Gesamtanzahl Datenpunkte:')} {len(df)}")

    counts = df['wave'].value_counts().reset_index()
    counts.columns = ['wave', 'count']
    print(bold("Wave/Pattern Verteilung:"))
    print(tabulate(counts.values, headers=['Wave', 'Count']))
    # Use an early start date to avoid pandas datetime overflow for large
    # datasets. The absolute values do not matter, only the ordering is
    # relevant for feature generation.
    df.index = pd.date_range("1900-01-01", periods=len(df), freq="1h")
    levels = get_all_levels(df, ["2H", "4H", "1D", "1W"], log=log)
    df = make_features(df, levels=levels, log=log)
    df_valid = df[~df["wave"].isin(["X", "INVALID_WAVE"])].reset_index(
        drop=True
    )

    if max_samples is not None and len(df_valid) > max_samples:
        groups = df_valid.groupby("wave")
        per_class = max_samples // len(groups)
        sampled = [
            g.sample(min(len(g), per_class), random_state=42)
            for _, g in groups
        ]
        df_valid = (
            pd.concat(sampled)
            .sample(frac=1, random_state=42)
            .reset_index(drop=True)
        )

    print(f"{blue('Nach Filterung gültige Datenpunkte:')} {len(df_valid)}")
    features = [f for f in FEATURES_BASE if f in df_valid.columns]
    X = df_valid[features]
    y = df_valid["wave"].astype(str)

    tscv = TimeSeriesSplit(n_splits=5)

    def create_model_params(mtype):
        if mtype == "rf":
            base = RandomForestClassifier(random_state=42)
            grid = {
                "n_estimators": [100, 200],
                "max_depth": [None, 10, 20],
                "class_weight": [None, "balanced"],
            }
            defaults = {
                "n_estimators": 200,
                "max_depth": None,
                "class_weight": None,
            }
        elif mtype == "xgb":
            base = XGBClassifier(
                random_state=42, verbosity=0, eval_metric="logloss"
            )
            grid = {
                "n_estimators": [100, 200],
                "max_depth": [3, 6],
                "learning_rate": [0.05, 0.1],
            }
            defaults = {
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.1,
            }
        elif mtype == "lgbm":
            base = LGBMClassifier(random_state=42)
            grid = {
                "n_estimators": [100, 200],
                "max_depth": [-1, 10],
                "learning_rate": [0.05, 0.1],
            }
            defaults = {
                "n_estimators": 200,
                "max_depth": -1,
                "learning_rate": 0.1,
            }
        elif mtype == "voting":
            rf = RandomForestClassifier(n_estimators=200, random_state=42)
            xgb = XGBClassifier(
                n_estimators=200,
                random_state=42,
                verbosity=0,
                eval_metric="logloss",
            )
            lgb = LGBMClassifier(n_estimators=200, random_state=42)
            base = VotingClassifier(
                estimators=[("rf", rf), ("xgb", xgb), ("lgbm", lgb)],
                voting="soft",
            )
            grid = None
            defaults = None
        else:
            raise ValueError(f"Unbekannter Modeltyp: {mtype}")
        return base, grid, defaults

    base_model, param_grid, defaults = create_model_params(model_type)

    if not skip_grid_search and param_grid:
        print(yellow("Starte GridSearch zur Hyperparameteroptimierung..."))
        best_score = -np.inf
        best_params = defaults or {}
        for params in alive_it(
            list(ParameterGrid(param_grid)), disable=not log
        ):
            model_tmp = clone(base_model)
            model_tmp.set_params(**params)
            scores = cross_val_score(model_tmp, X, y, cv=tscv, n_jobs=-1)
            score = scores.mean()
            if score > best_score:
                best_score = score
                best_params = params
        print(
            green(
                f"Beste CV-Genauigkeit: {best_score:.3f} "
                f"| Beste Parameter: {best_params}"
            )
        )
    else:
        best_params = defaults or {}

    def instantiate(mtype, params):
        if mtype == "rf":
            return RandomForestClassifier(**params, random_state=42)
        elif mtype == "xgb":
            return XGBClassifier(
                **params, random_state=42, verbosity=0, eval_metric="logloss"
            )
        elif mtype == "lgbm":
            return LGBMClassifier(**params, random_state=42)
        elif mtype == "voting":
            rf = RandomForestClassifier(
                n_estimators=params.get("n_estimators", 200), random_state=42
            )
            xgb = XGBClassifier(
                n_estimators=params.get("n_estimators", 200),
                random_state=42,
                verbosity=0,
                eval_metric="logloss",
            )
            lgb = LGBMClassifier(
                n_estimators=params.get("n_estimators", 200), random_state=42
            )
            return VotingClassifier(
                estimators=[("rf", rf), ("xgb", xgb), ("lgbm", lgb)],
                voting="soft",
            )

    model = instantiate(model_type, best_params)

    if feature_selection:
        selector = RFECV(model, step=1, cv=tscv, n_jobs=-1)
        print(yellow("Führe Feature Selection mittels RFECV durch..."))
        selector.fit(X, y)
        model = selector.estimator_
        features = [f for f, s in zip(features, selector.support_) if s]
        X = df_valid[features]

    # Compute and log feature statistics of the training data
    feature_stats = log_feature_stats(X)

    print(yellow("Trainiere finales Modell..."))
    with alive_it(1, disable=not log, title="Model Fit") as bar:
        model.fit(X, y)
        bar()

    cv_scores = cross_val_score(model, X, y, cv=tscv, n_jobs=-1)
    print(
        green(
            "Durchschnittliche CV-Genauigkeit: "
            f"{cv_scores.mean():.3f} ± {cv_scores.std():.3f}"
        )
    )

    if hasattr(model, "feature_importances_"):
        importance = pd.Series(
            model.feature_importances_, index=features
        ).sort_values(ascending=False)
    else:
        importance = pd.Series(np.zeros(len(features)), index=features)

    save_model(
        {"model": model, "features": features, "feature_stats": feature_stats},
        MODEL_PATH,
    )
    return model, features, importance, feature_stats


# === Fibo-Bereiche für Entry/TP/SL (automatisch, pro Welle) ===
def get_fibo_zones(df, wave, side="LONG"):
    idx = df[df["wave_pred"] == wave].index
    if len(idx) < 2:
        return None, None
    start, end = idx[0], idx[-1]
    price_start = df["close"].iloc[start]
    price_end = df["close"].iloc[end]
    diff = price_end - price_start

    # Determine actual direction by diff sign
    direction = "LONG" if diff >= 0 else "SHORT"
    diff = abs(diff)

    if direction == "LONG":
        e1 = price_start + diff * 0.236
        e2 = price_start + diff * 0.382
        t1 = price_start + diff * 0.618
        t2 = price_start + diff * 1.0
    else:
        e1 = price_start - diff * 0.236
        e2 = price_start - diff * 0.382
        t1 = price_start - diff * 0.618
        t2 = price_start - diff * 1.0

    # Expand zones using PUFFER
    entry_zone = [min(e1, e2) * (1 - PUFFER), max(e1, e2) * (1 + PUFFER)]
    tp_zone = [min(t1, t2) * (1 - PUFFER), max(t1, t2) * (1 + PUFFER)]
    return entry_zone, tp_zone


# === Zielprojektionen / Pattern-Targets ===
def pattern_target(df_features, current_pattern, last_complete_close):
    idx_pattern = df_features[
        df_features["wave_pred"] == current_pattern
    ].index
    if not len(idx_pattern):
        return last_complete_close * 1.01
    close_last = df_features["close"].iloc[idx_pattern[-1]]
    if current_pattern in ["WXY", "WXYXZ", "WXYXZY"]:
        w_idx = _latest_segment_indices(
            df_features, "W", min_len=5, end_buffer=5
        )
        if len(w_idx) > 1:
            w_len = (
                df_features["close"].iloc[w_idx[-1]]
                - df_features["close"].iloc[w_idx[0]]
            )
            return close_last + 0.8 * w_len
    factor = PATTERN_PROJ_FACTORS.get(current_pattern, 1.0)
    high = df_features["high"].iloc[idx_pattern].max()
    low = df_features["low"].iloc[idx_pattern].min()
    breakout_size = high - low
    return close_last + factor * breakout_size


def _latest_segment_indices(
    df: pd.DataFrame,
    wave_label: str,
    *,
    min_len: int = 1,
    end_buffer: int = 0,
):
    """Return indices of the most recent contiguous segment for ``wave_label``.

    The segment is considered valid only when its length is at least
    ``min_len`` and the last index lies within ``end_buffer`` bars from the
    end of ``df``.  Otherwise an empty array is returned.
    """

    idxs = df[df["wave_pred"] == wave_label].index.to_numpy()
    if len(idxs) == 0:
        return np.array([], dtype=int)
    splits = np.split(idxs, np.where(np.diff(idxs) != 1)[0] + 1)
    seg = splits[-1]
    if len(seg) >= min_len and (df.index[-1] - seg[-1]) <= end_buffer:
        return seg
    return np.array([], dtype=int)


def get_next_wave(current_wave):
    """Return a list of expected follow-up waves for ``current_wave``."""
    wave = str(current_wave)
    nxt = pattern_registry.get_next_wave(wave)
    if nxt:
        if isinstance(nxt, list):
            return [n for n in nxt if n and n != "Abschluss"]
        return [nxt] if nxt and nxt != "Abschluss" else []
    if wave in DEFAULT_NEXT_WAVE:
        return [DEFAULT_NEXT_WAVE[wave]]
    return []


def elliott_target(
    df_features,
    current_wave,
    last_complete_close,
    levels=None,
    level_tolerance=0.001,
):
    """Return projected target price for an Elliott wave segment.

    Parameters
    ----------
    df_features : DataFrame
        Feature dataframe containing OHLC data.
    current_wave : str
        Wave label to project the target for.
    last_complete_close : float
        Last closing price of the previously completed candle.
    levels : list[dict], optional
        Pre-computed price levels used to snap the target.
    level_tolerance : float, optional
        Tolerance as a fraction of ``wave_start_price`` used when snapping
        to nearby levels. If the nearest level differs from
        ``wave_start_price`` by less than this tolerance, the original
        target is kept.

    Notes
    -----
    If no valid segment matching ``current_wave`` is found, a fallback target
    of ``last_complete_close * 1.01`` is returned.
    """
    def idx(wave):
        return _latest_segment_indices(
            df_features, wave, min_len=5, end_buffer=5
        )
    start_idx = idx(current_wave)
    if len(start_idx) == 0:
        fallback = last_complete_close * 1.01
        return (fallback, fallback), last_complete_close, last_complete_close

    wave_start_price = df_features["close"].iloc[start_idx[0]]
    last_wave_close = df_features["close"].iloc[start_idx[-1]]

    target_low = target_high = None
    if str(current_wave) in PATTERN_PROJ_FACTORS:
        t = pattern_target(df_features, str(current_wave), last_complete_close)
        target_low = target_high = t
    elif str(current_wave) == "1":
        target_low = target_high = wave_start_price * 1.02
    elif str(current_wave) == "2":
        idx1 = idx("1")
        if len(idx1) > 0:
            high1 = df_features["close"].iloc[idx1].max()
            low2 = wave_start_price
            retracement = 0.5
            t = high1 - retracement * (high1 - low2)
            if t < low2:
                target_low = target_high = t
            else:
                target_low = target_high = None
        else:
            target_low = target_high = last_complete_close * 0.98
    elif str(current_wave) == "3":
        idx1 = idx("1")
        if len(idx1) > 1 and len(start_idx) > 0:
            w1len = (
                df_features["close"].iloc[idx1[-1]]
                - df_features["close"].iloc[idx1[0]]
            )
            target_low = wave_start_price + 1.618 * w1len
            target_high = wave_start_price + 2.618 * w1len
        else:
            target_low = target_high = last_complete_close * 1.05
    elif str(current_wave) == "4":
        idx3 = idx("3")
        if len(idx3) > 1 and len(start_idx) > 0:
            w3len = (
                df_features["close"].iloc[idx3[-1]]
                - df_features["close"].iloc[idx3[0]]
            )
            t = wave_start_price - 0.382 * abs(w3len)
            if t < wave_start_price:
                target_low = wave_start_price - 0.382 * abs(w3len)
                target_high = wave_start_price - 0.236 * abs(w3len)
            else:
                target_low = target_high = None
        else:
            target_low = target_high = last_complete_close * 0.98
    elif str(current_wave) == "5":
        idx1 = idx("1")
        idx3 = idx("3")
        if len(idx1) > 1 and len(start_idx) > 0:
            w1len = (
                df_features["close"].iloc[idx1[-1]]
                - df_features["close"].iloc[idx1[0]]
            )
            if len(idx3) > 1:
                w3len = (
                    df_features["close"].iloc[idx3[-1]]
                    - df_features["close"].iloc[idx3[0]]
                )
                proj_len = max(w1len, w3len) * 0.618
            else:
                proj_len = w1len
            target_low = wave_start_price + proj_len
            target_high = wave_start_price + max(w1len, w3len)
        else:
            target_low = target_high = last_complete_close * 1.02
    elif str(current_wave) == "A":
        idx5 = idx("5")
        if len(idx5) > 1 and len(start_idx) > 0:
            w5len = (
                df_features["close"].iloc[idx5[-1]]
                - df_features["close"].iloc[idx5[0]]
            )
            t = wave_start_price - w5len
            if t < wave_start_price:
                target_low = target_high = t
            else:
                target_low = target_high = None
        else:
            target_low = target_high = last_complete_close * 0.98
    elif str(current_wave) == "B":
        idxa = idx("A")
        if len(idxa) > 1 and len(start_idx) > 0:
            wa = (
                df_features["close"].iloc[idxa[-1]]
                - df_features["close"].iloc[idxa[0]]
            )
            target_low = wave_start_price + 0.618 * abs(wa)
            target_high = wave_start_price + 1.0 * abs(wa)
        else:
            target_low = target_high = last_complete_close * 1.01
    elif str(current_wave) == "C":
        idxa = idx("A")
        if len(idxa) > 1 and len(start_idx) > 0:
            wa = (
                df_features["close"].iloc[idxa[-1]]
                - df_features["close"].iloc[idxa[0]]
            )
            t = wave_start_price - 1.0 * abs(wa)
            target_low = wave_start_price - 1.618 * abs(wa)
            if t < wave_start_price:
                target_high = t
            else:
                target_low = target_high = None
        else:
            target_low = target_high = last_complete_close * 0.98
    elif str(current_wave) in ["W", "X", "Y", "Z"]:
        idxw = idx("W")
        if len(idxw) > 1 and len(start_idx) > 0:
            wlen = (
                df_features["close"].iloc[idxw[-1]]
                - df_features["close"].iloc[idxw[0]]
            )
            if str(current_wave) in ["Y", "Z"]:
                target_low = wave_start_price + 0.618 * wlen
                target_high = wave_start_price + 1.0 * wlen
            else:
                target_low = wave_start_price - 0.382 * abs(wlen)
                target_high = wave_start_price - 0.236 * abs(wlen)
        else:
            target_low = target_high = last_complete_close
    else:
        target_low = target_high = last_complete_close * 1.01

    if target_low is not None and levels:
        prices = [lvl["price"] for lvl in levels]
        if prices:
            nearest_low = min(prices, key=lambda p: abs(p - target_low))
            nearest_high = min(prices, key=lambda p: abs(p - target_high))
            # Only snap to the nearest level when it differs from the
            # wave's start by more than the tolerance. This avoids
            # returning a target equal to the wave's start unless the
            # level is meaningfully different.
            tolerance = abs(wave_start_price) * level_tolerance
            if abs(nearest_low - wave_start_price) >= tolerance:
                target_low = nearest_low
            if abs(nearest_high - wave_start_price) >= tolerance:
                target_high = nearest_high

    return (target_low, target_high), wave_start_price, last_wave_close


def elliott_target_market_relative(
    df_features,
    current_wave,
    last_close,
    wave1_len=None,
    wave3_len=None,
    *,
    fallback_mult=0.015
):
    """Market-price relative Elliott target projection.

    Parameters
    ----------
    df_features : DataFrame
        Feature dataframe containing OHLC data.
    current_wave : str
        Wave label to project the target for.
    last_close : float
        Current real market price used as reference point.
    wave1_len : float, optional
        Explicit length of wave 1 if already known.
    wave3_len : float, optional
        Explicit length of wave 3 if already known.
    fallback_mult : float, optional
        Fallback percentage for targets when no wave lengths are found.
    """

    if wave1_len is None or wave3_len is None:
        w1_idx = df_features[df_features["wave_pred"] == "1"].index
        w3_idx = df_features[df_features["wave_pred"] == "3"].index
        if len(w1_idx) > 1:
            wave1_len = abs(
                df_features["close"].iloc[w1_idx[-1]]
                - df_features["close"].iloc[w1_idx[0]]
            )
        else:
            wave1_len = last_close * 0.01
        if len(w3_idx) > 1:
            wave3_len = abs(
                df_features["close"].iloc[w3_idx[-1]]
                - df_features["close"].iloc[w3_idx[0]]
            )
        else:
            wave3_len = wave1_len

    if current_wave == "1":
        t_low = t_high = last_close * (1 + 1.0 * fallback_mult)
    elif current_wave == "2":
        t_low = last_close - wave1_len * 0.5
        t_high = last_close - wave1_len * 0.382
    elif current_wave == "3":
        t_low = last_close + 1.618 * wave1_len
        t_high = last_close + 2.618 * wave1_len
    elif current_wave == "4":
        t_low = last_close - 0.382 * wave3_len
        t_high = last_close - 0.236 * wave3_len
    elif current_wave == "5":
        base = max(wave1_len, wave3_len)
        t_low = last_close + 0.618 * base
        t_high = last_close + 1.0 * base
    elif current_wave in ["A", "B", "C"]:
        t_low = last_close * (1 - 1.0 * fallback_mult)
        t_high = last_close * (1 + 1.0 * fallback_mult)
    else:
        t_low = last_close * (1 - 1.5 * fallback_mult)
        t_high = last_close * (1 + 1.5 * fallback_mult)

    min_price, max_price = last_close * 0.5, last_close * 1.5
    t_low = float(np.clip(t_low, min_price, max_price))
    t_high = float(np.clip(t_high, min_price, max_price))
    return (t_low, t_high), last_close, last_close


def suggest_trade(
    df,
    current_wave,
    target_range,
    last_close,
    entry_zone=None,
    tp_zone=None,
    risk=0.01,
    sl_puffer=0.005,
    levels=None,
    probability=None,
):
    """Suggest trade parameters for the given wave.

    When ``probability`` is provided, the risk (and thus resulting SL/TP and
    position size) is scaled by this value to reflect the confidence of the
    prediction.
    """

    entry = last_close
    if (
        target_range is None
        or len(target_range) != 2
        or any(t is None for t in target_range)
    ):
        print(
            red(
                "Kein gültiges Kursziel berechnet "
                "(target_range ist None oder enthält None)."
            )
        )
        return None, None, None, None

    adj_risk = risk * probability if probability is not None else risk

    target_mid = sum(target_range) / 2

    if target_mid > entry:
        direction = "LONG"
        tp = target_range[1]
        sl = entry * (1 - adj_risk - sl_puffer)
    else:
        direction = "SHORT"
        tp = target_range[0]
        sl = entry * (1 + adj_risk + sl_puffer)

    local_cols = [
        c for c in df.columns if c.startswith(f"fib_{current_wave}_")
    ]
    local_prices = []
    if local_cols:
        local_prices = df.iloc[-1][local_cols].dropna().tolist()

    if levels or local_prices:
        prices = []
        if levels:
            prices.extend([lvl["price"] for lvl in levels])
        prices.extend(local_prices)
        if prices:
            tp = min(prices, key=lambda p: abs(p - tp))
            entry = min(prices, key=lambda p: abs(p - entry))

    size = 1000 * adj_risk / abs(entry - sl) if abs(entry - sl) > 0 else 0
    print(
        bold(
            f"\n[TRADE-SETUP] {direction} | Entry: {entry:.4f} | SL: {sl:.4f} "
            f"| TP: {target_range[0]:.4f}-{target_range[1]:.4f} "
            f"| PosSize: {size:.1f}x"
        )
    )
    # Entry- und TP-Zonen werden nicht ausgegeben
    return direction, sl, tp, entry


# === Auswertung der vorhergesagten Wellenstruktur ===
def evaluate_wave_structure(df, label_col="wave_pred"):
    """Prüfe Reihenfolge und Start-/Endpunkte der vorhergesagten Wellen."""
    if label_col not in df.columns:
        print(red("Keine Vorhersagen zur Auswertung gefunden."))
        return False

    segments = []
    current = df[label_col].iloc[0]
    start = 0
    for i in range(1, len(df)):
        if df[label_col].iloc[i] != current:
            segments.append((current, start, i - 1))
            current = df[label_col].iloc[i]
            start = i
    segments.append((current, start, len(df) - 1))

    table = [(lbl, s, e) for lbl, s, e in segments]
    print(bold("\nWave Segments (Prediction):"))
    print(tabulate(table, headers=["Wave", "Start", "End"]))

    order = ["1", "2", "3", "4", "5", "A", "B", "C"]
    last_idx = -1
    for lbl, s, e in segments:
        if lbl in order:
            idx = order.index(lbl)
            if idx < last_idx:
                print(
                    red(
                        f"Ungültige Reihenfolge bei Welle {lbl} (Start {s})"
                    )
                )
                return False
            last_idx = idx
    print(green("Wellenreihenfolge scheint konsistent."))
    return True


def run_pattern_analysis(
    df,
    model,
    features,
    levels=None,
    *,
    log: bool = False,
) -> None:
    df_feat = make_features(df, levels=levels, log=log)
    preds = model.predict(df_feat[features])
    proba = model.predict_proba(df_feat[features])
    classes = [str(c) for c in model.classes_]
    results = []
    df_feat["wave_pred"] = preds
    df_feat = compute_wave_fibs(df_feat, "wave_pred", buffer=PUFFER, log=log)
    for i, row in df_feat.iterrows():
        wave = str(row["wave_pred"])
        prob = proba[i, classes.index(wave)] if wave in classes else 0.0
        target_range, start_price, _ = elliott_target_market_relative(
            df_feat.iloc[: i + 1], wave, row["close"]
        )
        validity = "valid" if target_range[0] is not None else "invalid"
        trade = None
        if validity == "valid":
            direction, sl, tp, entry = suggest_trade(
                df_feat.iloc[: i + 1],
                wave,
                target_range,
                row["close"],
                levels=levels,
                probability=prob,
            )
            trade = {"entry": entry, "sl": sl, "tp": tp, "probability": prob}
        results.append(
            {
                "pattern_type": wave,
                "wave_id": wave,
                "target_projection": {
                    "level": None,
                    "price_range": target_range,
                    "pattern": wave,
                    "probability": prob,
                },
                "tradesetup": trade,
                "validity": validity,
            }
        )
    return results


# === Prediction Smoothing ===
def smooth_predictions(pred, smooth_window=5):
    """Return smoothed labels using majority voting."""
    half = smooth_window // 2
    smoothed = []
    for i in range(len(pred)):
        start = max(0, i - half)
        end = min(len(pred), i + half + 1)
        window_vals = pred[start:end]
        mode = pd.Series(window_vals).mode()
        smoothed.append(mode.iloc[0] if len(mode) else pred[i])
    return np.array(smoothed)


# === Hauptfunktion für Analyse & Grafik ===
def run_ml_on_bitget(
    model,
    features,
    importance,
    feature_stats=None,
    symbol=SYMBOL,
    interval="1H",
    livedata_len=LIVEDATA_LEN,
    extra_intervals=None,
    smooth_window=5,
    *,
    log: bool = True,
):
    df_1h = fetch_bitget_ohlcv_auto(
        symbol, interval, target_len=livedata_len, page_limit=1000, log=True
    )
    df_4h = fetch_bitget_ohlcv_auto(
        symbol, "4H", target_len=800, page_limit=1000, log=True
    )
    df_2h = df_1d = df_1w = None
    if extra_intervals:
        if "2H" in extra_intervals:
            df_2h = fetch_bitget_ohlcv_auto(
                symbol, "2H", target_len=800, page_limit=1000, log=True
            )
        if "1D" in extra_intervals or "D" in extra_intervals:
            df_1d = fetch_bitget_ohlcv_auto(
                symbol, "1D", target_len=400, page_limit=1000, log=True
            )
        if "1W" in extra_intervals or "W" in extra_intervals:
            df_1w = fetch_bitget_ohlcv_auto(
                symbol, "1W", target_len=200, page_limit=1000, log=True
            )
    print(bold("\n==== BITGET DATA ===="))
    parts = [f"{len(df_1h)} (1H)"]
    if df_2h is not None:
        parts.append(f"{len(df_2h)} (2H)")
    parts.append(f"{len(df_4h)} (4H)")
    if df_1d is not None:
        parts.append(f"{len(df_1d)} (1D)")
    if df_1w is not None:
        parts.append(f"{len(df_1w)} (1W)")
    print(
        f"Symbol: {symbol} | Intervall: {interval} | Bars: "
        + " / ".join(parts)
    )
    print(f"Letzter Timestamp: {df_1h['timestamp'].iloc[-1]}")
    last_complete_close = df_1h["close"].iloc[-2]

    levels_base = df_1h.copy()
    levels_base["timestamp"] = pd.to_datetime(levels_base["timestamp"])
    levels_base = levels_base.set_index("timestamp")
    levels = get_all_levels(levels_base, ["2H", "4H", "1D", "1W"], log=log)

    fib_levels = {}
    if df_1d is not None:
        tmp = df_1d.copy()
        tmp["timestamp"] = pd.to_datetime(tmp["timestamp"])
        tmp = tmp.set_index("timestamp")
        fib_levels["1D"] = get_fib_levels(tmp, "1D")
    if df_1w is not None:
        tmp = df_1w.copy()
        tmp["timestamp"] = pd.to_datetime(tmp["timestamp"])
        tmp = tmp.set_index("timestamp")
        fib_levels["1W"] = get_fib_levels(tmp, "1W")
    for fl in fib_levels.values():
        levels.extend(fl)

    df_features = make_features(
        df_1h, df_4h, levels=levels, fib_levels=fib_levels, log=log
    )

    # Sicherstellen, dass alle vom Modell erwarteten Features vorhanden sind
    missing = [f for f in features if f not in df_features.columns]
    if missing:
        msg = "Fehlende Features für Vorhersage: " + ", ".join(missing)
        print(red(msg))
        raise ValueError(msg)

    live_stats = log_feature_stats(df_features[features])
    if feature_stats is not None:
        table = []
        alerts = []
        threshold = 3
        for feat in features:
            train = feature_stats.get(feat, {})
            live = live_stats.get(feat, {})
            table.append([
                feat,
                f"{train.get('mean', float('nan')):.4f}",
                f"{live.get('mean', float('nan')):.4f}",
                f"{train.get('std', float('nan')):.4f}",
                f"{live.get('std', float('nan')):.4f}",
            ])
            if (
                train.get('std', 0) > 0
                and live
                and abs(live.get('mean', 0) - train.get('mean', 0))
                > threshold * train['std']
            ):
                alerts.append(feat)
        print(bold("\nFeature Stats Comparison:"))
        print(
            tabulate(
                table,
                headers=["Feature", "TrainMean", "LiveMean", "TrainStd", "LiveStd"],
            )
        )
        if alerts:
            print(red("Warnung: Starke Abweichungen bei " + ", ".join(alerts)))

    pred_raw = model.predict(df_features[features])
    pred = smooth_predictions(pred_raw, smooth_window=smooth_window)
    pred_proba = model.predict_proba(df_features[features])
    classes = [str(c) for c in model.classes_]
    df_features["wave_pred_raw"] = pred_raw
    df_features["wave_pred"] = pred
    df_features = compute_wave_fibs(
        df_features, "wave_pred", buffer=PUFFER, log=log
    )
    proba_row = pred_proba[-1]
    current_wave = df_features["wave_pred"].iloc[-1]
    main_wave = str(pred[-1])
    main_wave_idx = classes.index(main_wave)
    main_wave_prob = proba_row[main_wave_idx]
    alt_wave, alt_prob = None, None
    if current_wave in ["N", "INVALID_WAVE", "X"]:
        valid_waves = [
            c for c in classes if c not in ["N", "X", "INVALID_WAVE"]
        ]
        valid_indices = [i for i, c in enumerate(classes) if c in valid_waves]
        valid_probs = [proba_row[i] for i in valid_indices]
        if valid_probs:
            alt_idx = valid_indices[np.argmax(valid_probs)]
            alt_wave = classes[alt_idx]
            alt_prob = proba_row[alt_idx]
            print(yellow("\nAktuelle erkannte Welle: Noise/Invalid/X"))
            print(
                yellow(
                    f"Alternativer Vorschlag (höchste ML-Prob.): {alt_wave} "
                    f"({alt_prob*100:.1f}%) → "
                    f"{LABEL_MAP.get(alt_wave, alt_wave)}"
                )
            )
            current_wave = alt_wave
            print(
                green(
                    f"Aktuelle erkannte Welle (endgültig): {current_wave} "
                    f"({alt_prob*100:.1f}%) - "
                    f"{LABEL_MAP.get(current_wave, current_wave)}"
                )
            )
        else:
            print(
                red(
                    "Keine valide Welle mit hinreichender"
                    " Wahrscheinlichkeit erkannt!"
                )
            )
    else:
        print(
            green(
                f"Aktuelle erkannte Welle (endgültig): {current_wave} "
                f"({main_wave_prob*100:.1f}%) - "
                f"{LABEL_MAP.get(current_wave, current_wave)}"
            )
        )

    prob_sorted_idx = np.argsort(proba_row)[::-1]
    print(bold("\nTop-3 ML-Wahrscheinlichkeiten:"))
    for i in range(3):
        idx = prob_sorted_idx[i]
        label = classes[idx]
        print(f"  {LABEL_MAP.get(label,label)}: {proba_row[idx]*100:.1f}%")

    pattern_conf = df_features["pattern_confidence"].iloc[-1]
    if pattern_conf < CONFIDENCE_THRESHOLD:
        print(
            red(
                f"Muster-Konfidenz zu niedrig ({pattern_conf:.2f} < "
                f"{CONFIDENCE_THRESHOLD})"
                " - Trade-Setup übersprungen."
            )
        )
        return

    # ==== Fibo-Zonen berechnen (nicht ausgeben) ====
    entry_zone, tp_zone = get_fibo_zones(
        df_features,
        current_wave,
        side=(
            "LONG"
            if current_wave in [
                "1",
                "3",
                "5",
                "B",
                "ZIGZAG",
                "TRIANGLE",
            ]
            else "SHORT"
        ),
    )

    # ==== Zielprojektionen ====
    (
        target_range,
        wave_start_price,
        last_wave_close,
    ) = elliott_target_market_relative(
        df_features,
        current_wave,
        last_complete_close,
    )
    # Bestmögliche Folgewelle anhand der ML-Wahrscheinlichkeit wählen
    next_wave_candidates = get_next_wave(current_wave)
    next_wave = None
    next_wave_start = None
    next_target_range = None
    next_wave_prob = -1
    for cand in next_wave_candidates:
        cand_prob = (
            proba_row[classes.index(cand)] if cand in classes else 0.0
        )
        if cand_prob > next_wave_prob:
            next_wave = cand
            next_wave_prob = cand_prob
    if next_wave:
        (
            next_target_range,
            next_wave_start,
            _,
        ) = elliott_target_market_relative(
            df_features,
            next_wave,
            target_range[1]
            if target_range[1] is not None
            else last_wave_close,
        )

    # Gleiches Vorgehen für die übernächste Welle
    next_next_wave_candidates = (
        get_next_wave(next_wave) if next_wave else []
    )
    next_next_wave = None
    next_next_wave_start = None
    next_next_target_range = None
    next_next_prob = -1
    for cand in next_next_wave_candidates:
        cand_prob = (
            proba_row[classes.index(cand)] if cand in classes else 0.0
        )
        if cand_prob > next_next_prob:
            next_next_wave = cand
            next_next_prob = cand_prob
    if next_next_wave:
        (
            next_next_target_range,
            next_next_wave_start,
            _,
        ) = elliott_target_market_relative(
            df_features,
            next_next_wave,
            (
                next_target_range[1]
                if next_target_range[1] is not None
                else (
                    target_range[1]
                    if target_range[1] is not None
                    else last_wave_close
                )
            ),
        )

    print(bold("\n==== ZIELPROJEKTIONEN ===="))
    print(
        f"Aktuelle Welle: {current_wave} "
        f"({LABEL_MAP.get(current_wave, current_wave)}) "
        f"| Start: {wave_start_price:.4f} | "
        f"Ziel: {target_range[0]:.4f}-{target_range[1]:.4f}"
    )
    if (
        next_wave
        and next_target_range
        and next_wave_start is not None
        and next_target_range[0] is not None
        and next_target_range[1] is not None
    ):
        print(
            f"Nächste erwartete Welle: {next_wave} "
            f"({LABEL_MAP.get(next_wave, next_wave)}) "
            f"| Start: {next_wave_start:.4f} | "
            f"Ziel: {next_target_range[0]:.4f}-"
            f"{next_target_range[1]:.4f}"
        )
    elif next_wave:
        print(
            "Nächste Welle noch nicht erkannt oder keine "
            "Zielprojektion möglich."
        )
    if (
        next_next_wave
        and next_next_target_range
        and next_next_wave_start is not None
        and next_next_target_range[0] is not None
        and next_next_target_range[1] is not None
    ):
        print(
            f"Darauffolgende erwartete Welle: {next_next_wave} "
            f"({LABEL_MAP.get(next_next_wave, next_next_wave)}) "
            f"| Start: {next_next_wave_start:.4f} | "
            f"Ziel: {next_next_target_range[0]:.4f}-"
            f"{next_next_target_range[1]:.4f}"
        )
    elif next_next_wave:
        print(
            "Übernächste Welle noch nicht erkannt oder keine "
            "Zielprojektion möglich."
        )

    # === Breakout Zone (letzte Patternrange) ===
    idx_pattern = df_features[df_features["wave_pred"] == current_wave].index
    breakout_zone = None
    if len(idx_pattern) > 1:
        high = df_features["high"].iloc[idx_pattern].max()
        low = df_features["low"].iloc[idx_pattern].min()
        breakout_zone = (low, high)

    # === Trade-Setup Output ===
    trade_wave = str(next_wave) if next_target_range else str(current_wave)
    trade_target_range = (
        next_target_range if next_target_range else target_range
    )
    if trade_wave in classes:
        trade_wave_idx = classes.index(trade_wave)
        trade_prob = proba_row[trade_wave_idx]
    else:
        print(
            yellow(
                f"Unbekannte Trade-Welle {trade_wave} –"
                " Wahrscheinlichkeit auf 0 gesetzt."
            )
        )
        trade_prob = 0.0

    fib_near = max(
        df_features.get("fib_near_1d", pd.Series([0])).iloc[-1],
        df_features.get("fib_near_1w", pd.Series([0])).iloc[-1],
    )
    naked_near = int(
        df_features["level_dist"].iloc[-1] / df_features["close"].iloc[-1]
        <= 0.003
    )
    prob_weight = (1 + 0.5 * fib_near) * (1 + 0.5 * naked_near)
    trade_prob *= prob_weight
    entry_exit_score = pattern_conf * prob_weight
    print(bold(f"Entry/Exit-Score: {entry_exit_score:.2f}"))
    direction, sl, tp, entry = suggest_trade(
        df_features,
        trade_wave,
        trade_target_range,
        last_complete_close,
        entry_zone,
        tp_zone,
        levels=levels,
        probability=trade_prob,
    )
    if direction is None:
        print(red("Trade-Setup konnte nicht erstellt werden."))

    evaluate_wave_structure(df_features)

    # Statistik- und Feature-Logs entfernt

    # === PRO-Level Grafik ===
    plt.figure(figsize=(17, 8))
    plt.plot(df_features["close"].values, label="Kurs", linewidth=1.3)
    colors = pd.Categorical(df_features["wave_pred"]).codes
    plt.scatter(
        df_features.index,
        df_features["close"].values,
        c=colors,
        cmap="rainbow",
        s=18,
        label="WaveClass",
        alpha=0.65,
    )
    if target_range[0] is not None:
        plt.axhline(
            sum(target_range) / 2,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label=(
                f"Zielprojektion {current_wave} "
                f"{target_range[0]:.4f}-{target_range[1]:.4f}"
            ),
        )
    if (
        next_wave
        and next_target_range
        and next_target_range[0] is not None
        and next_target_range[1] is not None
    ):
        plt.axhline(
            sum(next_target_range) / 2,
            color="grey",
            linestyle=":",
            linewidth=1.3,
            label=(
                f"Nächste Welle {next_wave} Ziel "
                f"{next_target_range[0]:.4f}-{next_target_range[1]:.4f}"
            ),
        )
    # Breakout-Zone Highlight
    if breakout_zone:
        plt.axhspan(
            breakout_zone[0],
            breakout_zone[1],
            color="orange",
            alpha=0.19,
            label="Breakout-Zone",
        )
    # Fibo Entry/TP-Zonen
    if entry_zone:
        plt.axhspan(
            entry_zone[0],
            entry_zone[1],
            color="green",
            alpha=0.15,
            label="Entry-Zone",
        )
    if tp_zone:
        plt.axhspan(
            tp_zone[0], tp_zone[1], color="blue", alpha=0.14, label="TP-Zone"
        )
    # Wellen/Pattern Text-Annotationen
    for wave in set(df_features["wave_pred"]):
        idxs = df_features[df_features["wave_pred"] == wave].index
        if len(idxs):
            mid = idxs[0] + (idxs[-1] - idxs[0]) // 2
            price = df_features["close"].iloc[mid]
            plt.text(
                mid,
                price,
                f"{LABEL_MAP.get(wave, wave)}",
                fontsize=8,
                ha="center",
                va="bottom",
                color="black",
                alpha=0.8,
            )
    plt.title(
        f"{symbol} 1H Chart – {LABEL_MAP.get(current_wave, current_wave)}"
    )
    plt.legend()
    plt.tight_layout()
    plt.show()


# === Main ===
def main():
    global MODEL_PATH, DATASET_PATH
    parser = argparse.ArgumentParser(description="Elliott Wave ML")
    parser.add_argument(
        "--model-path", default=MODEL_PATH, help="Pfad zum Modell"
    )
    parser.add_argument(
        "--dataset-path", default=DATASET_PATH, help="Pfad zum Dataset"
    )
    parser.add_argument(
        "--skip-grid-search",
        action="store_true",
        help="GridSearch überspringen",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximale Anzahl Trainingssamples",
    )
    parser.add_argument(
        "--model",
        choices=["rf", "xgb", "lgbm", "voting"],
        default="rf",
        help="Modelltyp",
    )
    parser.add_argument(
        "--feature-selection",
        action="store_true",
        help="RFECV Feature Auswahl nutzen",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Kleineren Datensatz zum Testen verwenden",
    )
    parser.add_argument(
        "--test-label-limit",
        type=int,
        default=100,
        help="Anzahl Samples je Label im Test-Modus",
    )
    parser.add_argument(
        "--noise-level",
        type=float,
        default=0.0,
        help="Standardabweichung des zusätzlichen OHLC Rauschens",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="Fenstergröße für das Glätten der Vorhersagen",
    )
    args = parser.parse_args()

    MODEL_PATH = args.model_path
    DATASET_PATH = args.dataset_path

    if os.path.exists(MODEL_PATH):
        print(yellow("Lade gespeichertes Modell..."))
        obj = load_model(MODEL_PATH)
        if isinstance(obj, dict) and "model" in obj:
            model = obj["model"]
            features = obj.get("features", FEATURES_BASE)
            feature_stats = obj.get("feature_stats")
        else:
            model = obj
            if os.path.exists(DATASET_PATH):
                df_tmp = load_dataset(DATASET_PATH)
                df_tmp = make_features(df_tmp, log=True)
                df_tmp = df_tmp[
                    ~df_tmp["wave"].isin(["X", "INVALID_WAVE"])
                ].reset_index(drop=True)
                features = [f for f in FEATURES_BASE if f in df_tmp.columns]
            else:
                features = FEATURES_BASE
            feature_stats = None
        if hasattr(model, "feature_importances_"):
            importance = pd.Series(
                model.feature_importances_, index=features
            ).sort_values(ascending=False)
        else:
            importance = pd.Series(index=features, dtype=float)
    else:
        model, features, importance, feature_stats = train_ml(
            skip_grid_search=args.skip_grid_search,
            max_samples=args.max_samples,
            model_type=args.model,
            feature_selection=args.feature_selection,
            test_mode=args.test_mode,
            test_label_limit=args.test_label_limit,
            noise_level=args.noise_level,
            log=True,
        )
    try:
        run_ml_on_bitget(
            model,
            features,
            importance,
            feature_stats,
            symbol=SYMBOL,
            interval="1H",
            livedata_len=LIVEDATA_LEN,
            extra_intervals=["2H", "4H", "1D", "1W"],
            smooth_window=args.smooth_window,
            log=True,
        )
    except ValueError as e:
        print(red(str(e)))


if __name__ == "__main__":
    main()
