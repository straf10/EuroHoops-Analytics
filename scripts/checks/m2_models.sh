#!/usr/bin/env bash
# Checklist item 23 (F3/F4): spline, isotonic and LightGBM unit tests, the 5-trial Optuna
# reproducibility test, and the recorded runtimes within budget (backtest < 40 min with 5 seeds,
# Optuna study < 2 h), read from the RUNTIME lines of the progress file.
cd "$ROOT" || exit 1
uv run pytest -q -p no:cacheprovider tests/test_xpts.py tests/test_m2_backtest.py 2>&1 | tail -1
[ "${PIPESTATUS[0]}" = 0 ] || exit 1
uv run python "$ROOT/scripts/checks/m2_runtimes.py"
