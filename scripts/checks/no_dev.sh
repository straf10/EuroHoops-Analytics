#!/usr/bin/env bash
# Checklist item 30: without the dev group (as in the daily workflow), `eurohoops build` and
# `eurohoops predict` still work and M2's dev dependencies are absent. predict runs with a zero
# window, so it appends no row; a log file it creates only because it did not exist yet (a
# header, no rows) is removed again, and predictions/ must end exactly as it started.
cd "$ROOT" || exit 1
before=$(git status --porcelain -- predictions/)
existing=$(ls predictions/)
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
for f in predictions/*; do
  name=$(basename "$f")
  if ! grep -qx "$name" <<<"$existing"; then
    rows=$(($(wc -l <"$f") - 1))
    [ "$rows" = 0 ] || { echo "$f was created with $rows rows"; exit 1; }
    rm -- "$f" && echo "removed header-only $f (created by this check)"
  fi
done
after=$(git status --porcelain -- predictions/)
[ "$before" = "$after" ] && git diff --quiet -- predictions/ \
  && echo "build + predict ran without the dev group; predictions/ as before"
