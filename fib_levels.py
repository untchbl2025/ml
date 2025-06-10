"""Compute Fibonacci retracement and extension levels for an OHLCV DataFrame."""
from __future__ import annotations

from typing import Dict, List

import pandas as pd


def _current_swing(df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return start and end timestamps for the most recent swing.

    The swing is determined by taking the global high and low of the
    provided data and ordering them chronologically. The earliest of the
    two becomes the swing start and the later becomes the swing end.
    """
    high_idx = df['high'].idxmax()
    low_idx = df['low'].idxmin()
    if high_idx > low_idx:
        return low_idx, high_idx
    return high_idx, low_idx


def get_fib_levels(df: pd.DataFrame, timeframe: str) -> List[Dict[str, object]]:
    """Return fibonacci levels for the current swing of ``df``.

    Includes common retracement levels between ``fib_0.0`` and ``fib_1.0``
    as well as the extension levels ``fib_1.618`` and ``fib_2.618``.

    Parameters
    ----------
    df : DataFrame
        OHLCV data indexed by timestamp.
    timeframe : str
        Identifier for the timeframe (e.g. "1D", "1W").
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex")

    start_ts, end_ts = _current_swing(df)
    start_price = df.loc[start_ts, 'low'] if start_ts < end_ts else df.loc[start_ts, 'high']
    end_price = df.loc[end_ts, 'high'] if end_ts > start_ts else df.loc[end_ts, 'low']

    if end_price >= start_price:
        diff = end_price - start_price
        levels = {
            'fib_0.0': end_price,
            'fib_0.236': end_price - diff * 0.236,
            'fib_0.382': end_price - diff * 0.382,
            'fib_0.5': end_price - diff * 0.5,
            'fib_0.618': end_price - diff * 0.618,
            'fib_0.786': end_price - diff * 0.786,
            'fib_1.0': start_price,
            'fib_1.618': start_price - diff * 0.618,
            'fib_2.618': start_price - diff * 1.618,
        }
    else:
        diff = start_price - end_price
        levels = {
            'fib_0.0': end_price,
            'fib_0.236': end_price + diff * 0.236,
            'fib_0.382': end_price + diff * 0.382,
            'fib_0.5': end_price + diff * 0.5,
            'fib_0.618': end_price + diff * 0.618,
            'fib_0.786': end_price + diff * 0.786,
            'fib_1.0': start_price,
            'fib_1.618': start_price + diff * 0.618,
            'fib_2.618': start_price + diff * 1.618,
        }

    ts = df.index[-1]
    return [
        {
            'level_type': name,
            'timeframe': timeframe.lower(),
            'price': float(val),
            'timestamp': ts,
        }
        for name, val in levels.items()
    ]


__all__ = ['get_fib_levels']
