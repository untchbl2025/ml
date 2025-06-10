import sys
import types

import pandas as pd
import numpy as np
import pytest

# Stub optional heavy dependencies before importing ml
sys.modules.setdefault("lightgbm", types.SimpleNamespace(LGBMClassifier=object))
sys.modules.setdefault("xgboost", types.SimpleNamespace(XGBClassifier=object))

import ml


class DummyModel:
    def __init__(self):
        self.classes_ = np.array(["A"])

    def predict(self, X):
        return np.array(["A"] * len(X))

    def predict_proba(self, X):
        return np.array([[1.0]] * len(X))


def test_run_ml_on_bitget_missing_features(monkeypatch):
    df_stub = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=3, freq="h"),
            "open": [1.0, 1.0, 1.0],
            "high": [1.0, 1.0, 1.0],
            "low": [1.0, 1.0, 1.0],
            "close": [1.0, 1.0, 1.0],
            "volume": [1.0, 1.0, 1.0],
        }
    )

    def fake_fetch(*args, **kwargs):
        return df_stub

    def fake_make_features(*args, **kwargs):
        return pd.DataFrame({"foo": [1.0]})

    monkeypatch.setattr(ml, "fetch_bitget_ohlcv_auto", fake_fetch)
    monkeypatch.setattr(ml, "make_features", fake_make_features)
    monkeypatch.setattr(ml, "get_all_levels", lambda *a, **k: [])
    monkeypatch.setattr(ml, "get_fib_levels", lambda *a, **k: [])

    model = DummyModel()
    with pytest.raises(ValueError):
        ml.run_ml_on_bitget(model, ["foo", "bar"], None)
