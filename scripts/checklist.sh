#!/usr/bin/env bash
# The phase checklist, top to bottom (first used as the weeks 5-7 §6 run). Prints PASS/FAIL per
# item; the exit code is the number of FAILs. Run from anywhere: `bash scripts/checklist.sh`.
#
#   BASE      git ref the predictions and the stint sample are compared with (default origin/main)
#   SCRATCH   working directory for copies and temporary files (default: a fresh mktemp dir)
#   SKIP      space-separated item numbers to skip, e.g. SKIP="16 20" (reported as SKIPPED)
#
# Items 7 and 20 build web/ in a copy, so running astro dev/preview servers (which lock
# web/node_modules) do not break `npm ci`. Item 20 needs Microsoft Edge (Playwright channel).
cd "$(dirname "$0")/.." || exit 99
ROOT=$(pwd)
BASE=${BASE:-origin/main}
SCRATCH=${SCRATCH:-$(mktemp -d)}
CHECKS="$ROOT/scripts/checks"
export ROOT SCRATCH
fails=0
item() { echo; echo "===== $1 ====="; }
res() { if [ "$1" = 0 ]; then echo "PASS"; else echo "FAIL"; fails=$((fails+1)); fi; }
skip() { [[ " $SKIP " == *" $1 "* ]] && { item "$1 $2"; echo "SKIPPED"; }; }
echo "HEAD $(git rev-parse --short HEAD), base $BASE, scratch $SCRATCH, start $(date -u +%H:%MZ)"

skip 1 "uv sync --frozen" || { item "1 uv sync --frozen"; uv sync --frozen 2>&1 | tail -1; res "${PIPESTATUS[0]}"; }
skip 2 "ruff check" || { item "2 ruff check"; uv run ruff check .; res $?; }
skip 3 "ruff format --check" || { item "3 ruff format --check"; uv run ruff format --check .; res $?; }
skip 4 "mypy src" || { item "4 mypy src"; uv run mypy src; res $?; }
skip 5 "pytest + coverage" || {
  item "5 pytest + coverage"
  uv run pytest -q --cov=eurohoops --cov-fail-under=85 -p no:cacheprovider 2>&1 | tail -3
  res "${PIPESTATUS[0]}"
}
skip 6 "vulture" || { item "6 vulture"; uv run vulture src vulture_whitelist.py --min-confidence 60; res $?; }
skip 7 "web build from the fixture" || { item "7 web build from the fixture (copy of web/)"; bash "$CHECKS/web_build.sh"; res $?; }
skip 8 "Elo backtests reproduce the committed reports" || {
  item "8 Elo backtests reproduce the committed reports"
  (uv run eurohoops backtest >/dev/null && uv run eurohoops backtest --competition gbl >/dev/null \
    && uv run pytest -q tests/test_reports.py -p no:cacheprovider 2>&1 | tail -1 \
    && git diff --exit-code --stat -- reports/backtest_elo.json reports/backtest_elo_gbl.json \
      reports/backtest_elo_history.json)
  res $?
}
skip 9 "predictions append-only" || {
  item "9 predictions append-only vs $BASE, Elo model_version unchanged"
  (
    bad=$(git diff "$BASE" -- predictions/ | grep -cE '^-[^-]')
    echo "removed/changed lines vs $BASE: $bad"
    for f in reports/backtest_elo.json reports/backtest_elo_gbl.json; do
      a=$(git show "$BASE:$f" | uv run python -c "import json,sys;print(json.load(sys.stdin)['model_version'])")
      b=$(uv run python -c "import json;print(json.load(open('$f'))['model_version'])")
      echo "$f base=$a now=$b"; [ "$a" = "$b" ] || exit 1
    done
    [ "$bad" = 0 ]
  )
  res $?
}
skip 10 "stint sample" || {
  item "10 stint sample reproduces $BASE"
  (uv run eurohoops stints && git diff --exit-code "$BASE" -- reports/stint_validation.json && echo identical)
  res $?
}
skip 11 "team_games" || { item "11 team_games coverage, points, possessions"; (uv run eurohoops possessions && uv run python "$CHECKS/possessions.py"); res $?; }
skip 12 "stints mart" || { item "12 stints mart thresholds, two builds identical"; bash "$CHECKS/stints_mart.sh"; res $?; }
skip 13 "M1 unit tests + runtime" || { item "13 M1 unit tests, EuroLeague backtest runtime"; bash "$CHECKS/m1_runtime.sh"; res $?; }
skip 14 "leakage tests" || {
  item "14 leakage tests"
  uv run pytest -q -p no:cacheprovider -p no:warnings tests/test_leakage.py 2>&1 | tail -1
  res "${PIPESTATUS[0]}"
}
skip 15 "M1 reports" || { item "15 M1 reports complete and reproducible"; bash "$CHECKS/m1_reports.sh"; res $?; }
skip 16 "MLflow" || { item "16 MLflow parent + children; no-dev run skips tracking"; bash "$CHECKS/mlflow.sh"; res $?; }
skip 17 "live M1 dry run" || { item "17 live M1 dry run"; uv run python "$CHECKS/live_m1_dry_run.py"; res $?; }
skip 18 "build + score + publish twice" || {
  item "18 build + score + publish twice leaves the tree unchanged"
  (
    run() {
      uv run eurohoops build >/dev/null && uv run eurohoops score >/dev/null \
        && uv run eurohoops score --competition gbl >/dev/null && uv run eurohoops publish >/dev/null
    }
    run && s1=$(git status --porcelain; git diff | sha256sum) \
      && run && s2=$(git status --porcelain; git diff | sha256sum) \
      && [ "$s1" = "$s2" ] && echo "unchanged on the second run"
  )
  res $?
}
skip 19 "actionlint" || {
  item "19 actionlint"
  uvx --from actionlint-py actionlint .github/workflows/ci.yml .github/workflows/daily.yml && echo clean
  res $?
}
skip 20 "screenshots" || {
  item "20 screenshots 1440/390, light/dark"
  (bash "$CHECKS/web_build.sh" >/dev/null 2>&1 \
    && uv run --with playwright python "$CHECKS/screenshots.py" "$SCRATCH/web/site" reports/screenshots)
  res $?
}
echo; echo "FAILS: $fails (end $(date -u +%H:%MZ))"; exit "$fails"
