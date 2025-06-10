import sys
import types

import numpy as np
import pandas as pd
import pytest

from fib_levels import get_fib_levels
from levels import LevelCalculator

# Stub heavy dependencies before importing ml for make_features
sys.modules.setdefault('lightgbm', types.SimpleNamespace(LGBMClassifier=object))
sys.modules.setdefault('xgboost', types.SimpleNamespace(XGBClassifier=object))
import ml


def test_get_fib_levels_simple_df():
    idx = pd.date_range('2020-01-01', periods=4, freq='D')
    df = pd.DataFrame({
        'open': [1, 2, 3, 4],
        'high': [1.1, 2.1, 3.1, 4.1],
        'low': [0.9, 1.9, 2.9, 3.9],
        'close': [1.0, 2.0, 3.0, 4.0],
        'volume': [10, 10, 10, 10],
    }, index=idx)

    levels = get_fib_levels(df, '1D')
    assert len(levels) == 9
    for lvl in levels:
        assert set(lvl) == {'level_type', 'timeframe', 'price', 'timestamp'}
        assert lvl['timeframe'] == '1d'


def test_level_calculator_calculates_levels():
    idx = pd.date_range('2020-01-01', periods=10, freq='H')
    df = pd.DataFrame({
        'open': np.linspace(1, 10, 10),
        'high': np.linspace(1.1, 10.1, 10),
        'low': np.linspace(0.9, 9.9, 10),
        'close': np.linspace(1.05, 10.05, 10),
        'volume': np.linspace(1, 10, 10),
    }, index=idx)

    calc = LevelCalculator(df, '2h', n_bins=5)
    levels = calc.calculate()
    assert levels, 'No levels returned'
    for lvl in levels:
        assert {'level_type', 'timeframe', 'price', 'timestamp'} <= lvl.keys()


def test_make_features_returns_expected_columns():
    idx = pd.date_range('2020-01-01', periods=60, freq='H')
    df = pd.DataFrame({
        'open': np.random.rand(60) + 1,
        'high': np.random.rand(60) + 1,
        'low': np.random.rand(60),
        'close': np.random.rand(60) + 1,
        'volume': np.random.rand(60) * 10,
    }, index=idx)

    feat = ml.make_features(df)
    expected = {'returns', 'rsi', 'macd', 'atr', 'fib_dist_1d', 'fib_near_1d'}
    assert expected <= set(feat.columns)
