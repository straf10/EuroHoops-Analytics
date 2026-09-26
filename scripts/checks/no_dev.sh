#!/usr/bin/env bash
# Checklist item 30: without the dev group (as in the daily workflow), `eurohoops build` and
# `eurohoops predict` still work and M2's dev dependencies are absent. predict runs with a zero
# window, so no row is appended to the live logs (checked).
cd "$ROOT" || exit 1
export UV_PROJECT_ENVIRONMENT="$SCRATCH/venv_nodev30"
uv sync --frozen --no-dev -q || exit 1
uv run --no-dev python -c "
import importlib.util as u
absent = [m for m in ('lightgbm', 'optuna', 'matplotlib', 'mlflow') if u.find_spec(m) is None]
print('no-dev env without', absent)
assert len(absent) == 4" || exit 1
uv run --no-dev eurohoops build >/dev/null || exit 1
uv run --no-dev eurohoops predict --window-hours 0 || exit 1
uv run --no-dev eurohoops predict --competition gbl --window-hours 0 || exit 1
git diff --quiet -- predictions/ && echo "build + predict ran without the dev group; predictions/ untouched"
