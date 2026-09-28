#!/usr/bin/env bash
# Checklist item 36: `backtest --model m3` (tuning grid + validation + every variant, plus test
# once it is scored) under 1,800 s (H-j), and the RUNTIME line recorded in the progress file
# under 1,800 s. Reuses item 35's first timed run when it ran in the same checklist.
cd "$ROOT" || exit 1
if [ -s "$SCRATCH/m3_runtime_s" ]; then
  secs=$(cat "$SCRATCH/m3_runtime_s")
  echo "timed by item 35 in this run"
else
  flag=$(uv run python -c "import json;print('--score-test' if json.load(open('reports/backtest_m3.json')).get('test_scored') else '')")
  start=$(date +%s)
  uv run eurohoops backtest --model m3 $flag --tracking-uri "sqlite:///$SCRATCH/mlflow_m3.db" >/dev/null 2>&1 || exit 1
  secs=$(( $(date +%s) - start ))
  git checkout -q -- reports/backtest_m3.json
fi
echo "RUNTIME $secs s (limit 1800 s)"
recorded=$(grep -oE '^RUNTIME [0-9]+' reports/week9-12_progress.md | tail -1 | cut -d' ' -f2)
echo "recorded RUNTIME ${recorded:-none} s"
[ "$secs" -lt 1800 ] && [ -n "$recorded" ] && [ "$recorded" -lt 1800 ]
