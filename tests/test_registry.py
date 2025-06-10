import sys
import types

# Stub heavy dependencies before importing ml
sys.modules.setdefault('lightgbm', types.SimpleNamespace(LGBMClassifier=object))
sys.modules.setdefault('xgboost', types.SimpleNamespace(XGBClassifier=object))

import ml
from pattern_registry import pattern_registry

EXPECTED_NEXT_WAVE = {
    "TRIANGLE": ["5", "C"],
    "ZIGZAG": ["B", "3"],
    "FLAT": ["3", "5", "C"],
    "DOUBLE_ZIGZAG": ["C", "5"],
    "EXPANDED_FLAT": ["3", "5", "C"],
    "WXY": ["Z", "Abschluss"],
    "WXYXZ": ["Abschluss"],
}


def test_registered_patterns_present():
    for name in ml.PATTERN_PROJ_FACTORS:
        assert name in pattern_registry._patterns


def test_next_wave_lists():
    for name, expected in EXPECTED_NEXT_WAVE.items():
        assert pattern_registry.get_next_wave(name) == expected
