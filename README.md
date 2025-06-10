# ML Project

This project contains utilities for calculating Fibonacci levels, price levels and machine learning features.

## Running tests

1. Install the required packages (preferably in a virtual environment):

   ```bash
   pip install -r requirements.txt
   ```

2. Execute the test suite using `pytest`:

   ```bash
   pytest
   ```

## Registered Patterns

The project includes several synthetic price patterns used for dataset
generation. Each pattern can optionally declare a list of follow up waves via
the `next_wave` argument of the `@register_pattern` decorator. New patterns must
be registered using this decorator so they become available to the generator.

| Pattern name    | `next_wave` entries |
|-----------------|--------------------|
| TRIANGLE        | `['5', 'C']`       |
| ZIGZAG          | `['B', '3']`       |
| FLAT            | `['3', '5', 'C']` |
| DOUBLE_ZIGZAG   | `['C', '5']`       |
| RUNNING_FLAT    | –                  |
| EXPANDED_FLAT   | `['3', '5', 'C']` |
| TREND_REVERSAL  | –                  |
| FALSE_BREAKOUT  | –                  |
| GAP_EXTENSION   | –                  |
| WXY             | `['Z', 'Abschluss']` |
| WXYXZ           | `['Abschluss']`    |
| WXYXZY          | –                  |
