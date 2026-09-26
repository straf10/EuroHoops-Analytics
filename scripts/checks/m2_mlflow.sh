#!/usr/bin/env bash
# Checklist item 28 (F9): the last M2 backtest in MLflow (parent with params, data hash and
# commit; one child per variant; the Optuna study child with every trial), and the leakage tests.
cd "$ROOT" || exit 1
uv run python "$ROOT/scripts/checks/m2_mlflow_runs.py" || exit 1
uv run pytest -q -p no:cacheprovider tests/test_m2_leakage.py 2>&1 | tail -1
exit "${PIPESTATUS[0]}"
