#!/usr/bin/env bash
# Checklist item 24 (F5): reports/backtest_m2.json complete with its gate block, declaration <
# verdict < test in git history, and two backtest runs byte-identical and equal to the committed
# report. Each run is timed against the budget: 60 minutes since the user's decision of
# 2026-09-26 (weeks 7-10b decision 3; it was 40). Both runs always happen (weeks 7-10b G4), so
# reproducibility is checked even when a run is over budget; the item fails if either run is
# over budget, the runs differ, or the report differs from the committed one, and all three
# results are printed.
cd "$ROOT" || exit 1
f=reports/backtest_m2.json
limit=3600
flag=$(uv run python -c "import json;print('--score-test' if json.load(open('$f')).get('test_scored') else '')")
budget_ok=1
for run in 1 2; do
  start=$(date +%s)
  uv run eurohoops backtest --model m2 $flag >/dev/null 2>&1 || { echo "run $run failed"; exit 1; }
  secs=$(( $(date +%s) - start ))
  echo "run $run ($flag): ${secs} s (limit ${limit} s)"
  [ "$secs" -lt "$limit" ] || budget_ok=0
  cp "$f" "$SCRATCH/m2_run$run.json"
done
same=0; committed=0
cmp -s "$SCRATCH/m2_run1.json" "$SCRATCH/m2_run2.json" && same=1
git diff --quiet -- "$f" && committed=1
echo "runtime within budget: $([ $budget_ok = 1 ] && echo yes || echo NO)"
echo "two runs byte-identical: $([ $same = 1 ] && echo yes || echo NO)"
echo "equals the committed report: $([ $committed = 1 ] && echo yes || echo NO)"
uv run python "$ROOT/scripts/checks/m2_order.py" || exit 1
uv run python "$ROOT/scripts/checks/m2_report_fields.py" || exit 1
[ $budget_ok = 1 ] && [ $same = 1 ] && [ $committed = 1 ]
