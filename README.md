# Machine Learning Wave Analysis

This repository contains the `ml` package along with unit tests.  The
package exposes utilities for data generation, feature engineering, model
training and prediction.

## Installation

Install the dependencies using `pip`:

```bash
pip install -r requirements.txt
```

## Running Tests

After ensuring the repository root is on the Python path (the tests handle this
by inserting the path at runtime), run the test suite with either:

```bash
python -m pytest -q
```

or simply:

```bash
pytest -q
```

## Project Overview

- **ml** implements utilities to generate synthetic price patterns, compute
  indicators such as Fibonacci levels and pivots, and train machine-learning
  models for market analysis. It exposes helper functions like
  `_latest_segment_indices`, `elliott_target`, and
  `elliott_target_market_relative`.
- **tests/test_segments.py** contains unit tests for these helper functions and
  includes a small path fix to import the project modules directly from the
  repository root.

## Command Line Usage

The `ml.core` module can be executed as a script. Use `--smooth-window` to
control how many predictions are considered when smoothing:

```bash
python -m ml.core --smooth-window 7
```

