#!/usr/bin/env bash
# Checklist item 24 (F5): reports/backtest_m2.json complete with its gate block, declaration <
# verdict < test in git history, and two backtest runs byte-identical and equal to the committed
# report. Each run is timed against the 40-minute budget.
cd "$ROOT" || exit 1
f=reports/backtest_m2.json
flag=$(uv run python -c "import json;print('--score-test' if json.load(open('$f')).get('test_scored') else '')")
for run in 1 2; do
  start=$(date +%s)
  uv run eurohoops backtest --model m2 $flag >/dev/null 2>&1 || exit 1
  secs=$(( $(date +%s) - start ))
  echo "run $run ($flag): ${secs} s (limit 2400 s)"
  [ "$secs" -lt 2400 ] || exit 1
  cp "$f" "$SCRATCH/m2_run$run.json"
done
cmp -s "$SCRATCH/m2_run1.json" "$SCRATCH/m2_run2.json" && echo "two runs byte-identical" || { echo "runs differ"; exit 1; }
git diff --quiet -- "$f" && echo "equals the committed report" || { echo "$f differs from the committed report"; exit 1; }
uv run python "$ROOT/scripts/checks/m2_order.py" || exit 1
uv run python "$ROOT/scripts/checks/m2_report_fields.py"
