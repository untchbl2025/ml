import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import ml


def test_no_negative_prices():
    df = ml.generate_rulebased_synthetic_with_patterns(n=50, log=False)
    assert (df[["open", "high", "low", "close"]] >= 0).all().all()


def test_no_zero_prices():
    df = ml.generate_rulebased_synthetic_with_patterns(n=50, log=False)
    assert (df[["open", "high", "low", "close"]] > 0).all().all()


def test_label_distribution():
    df = ml.generate_rulebased_synthetic_with_patterns(n=50, log=False)
    counts = df['wave'].value_counts(normalize=True)
    expected = ["1", "2", "3", "4", "5", "A", "B", "C"]
    counts = counts[counts.index.isin(expected)]
    assert not (counts < 0.01).any()
