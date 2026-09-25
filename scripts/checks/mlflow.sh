#!/usr/bin/env bash
# Checklist item 16: the last M1 backtest of each competition in the local MLflow store (parent +
# one child per variant), and a no-dev environment (as in the daily workflow) that skips
# tracking with a warning but writes the same report.
cd "$ROOT" || exit 1
git check-ignore -q mlruns/ && echo "mlruns/ is gitignored" || exit 1
uv run python "$ROOT/scripts/checks/mlflow_runs.py" || exit 1
export UV_PROJECT_ENVIRONMENT="$SCRATCH/venv_nodev"
uv sync --frozen --no-dev -q || exit 1
uv run --no-dev python -c "import importlib.util as u; assert u.find_spec('mlflow') is None; print('no-dev env: mlflow not installed')" || exit 1
cp reports/backtest_m1_gbl.json "$SCRATCH/gbl_before.json"
flag=$(python -c "import json;print('--score-test' if json.load(open('reports/backtest_m1_gbl.json')).get('test_scored') else '')")
uv run --no-dev eurohoops backtest --model m1 --competition gbl $flag 2>&1 | grep -a "not installed" || exit 1
cmp -s reports/backtest_m1_gbl.json "$SCRATCH/gbl_before.json" \
  && echo "no-dev backtest: same report, tracking skipped with a warning"
