import sys
import types
import numpy as np
import pandas as pd
import pytest

# Provide a dummy lightgbm module if not available
if 'lightgbm' not in sys.modules:
    sys.modules['lightgbm'] = types.ModuleType('lightgbm')
    sys.modules['lightgbm'].LGBMClassifier = object

from ml import _latest_segment_indices, elliott_target, elliott_target_market_relative


def test_latest_segment_indices_fresh():
    df = pd.DataFrame({"wave_pred": list("AABBBCCDD")})
    seg = _latest_segment_indices(df, "B", min_len=2, end_buffer=5)
    assert np.array_equal(seg, np.array([2, 3, 4]))


def test_latest_segment_indices_not_fresh():
    df = pd.DataFrame({"wave_pred": list("AABBBCCDD")})
    seg = _latest_segment_indices(df, "B", min_len=4, end_buffer=5)
    assert seg.size == 0


def test_elliott_target_fallback():
    df = pd.DataFrame({
        "close": np.arange(10.0),
        "high": np.arange(10.0),
        "low": np.arange(10.0),
        "wave_pred": ["1"] * 3 + ["0"] * 7,
    })
    target, start_price, last_close = elliott_target(
        df, "1", last_complete_close=8.0
    )
    assert target[0] == pytest.approx(8.0 * 1.01)
    assert target[1] == pytest.approx(8.0 * 1.01)


def test_elliott_target_market_relative_basic():
    df = pd.DataFrame({
        "close": [100, 110, 120, 130, 140],
        "wave_pred": ["1", "1", "1", "3", "3"],
    })
    target, start_price, last_close = elliott_target_market_relative(
        df, "3", last_close=140
    )
    assert target[0] == pytest.approx(172.36, rel=1e-2)
    assert target[1] == pytest.approx(192.36, rel=1e-2)
