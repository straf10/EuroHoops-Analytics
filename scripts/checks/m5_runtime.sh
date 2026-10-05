#!/usr/bin/env bash
# Checklist item 46: `backtest --model m5` under 1,800 s for the EuroLeague and 600 s for the
# GBL (J-k), and the last RUNTIME lines recorded in the progress file within the same limits.
# Reuses item 45's timed runs when it ran in the same checklist.
cd "$ROOT" || exit 1
export MLFLOW_DISABLE_AGENT_HINT=1
flag=$(uv run python -c "import json;print('--score-test' if json.load(open('reports/backtest_m5.json')).get('test_scored') else '')")
timed() {  # $1 = competition, $2 = cache file
  if [ -s "$SCRATCH/$2" ]; then cat "$SCRATCH/$2"; return; fi
  start=$(date +%s)
  uv run eurohoops backtest --model m5 --competition "$1" $flag \
    --tracking-uri "sqlite:///$SCRATCH/mlflow_m5.db" >/dev/null 2>&1 || { echo 99999; return; }
  echo $(( $(date +%s) - start ))
}
el=$(timed euroleague m5_runtime_s)
gbl=$(timed gbl m5_gbl_runtime_s)
git checkout -q -- reports/backtest_m5.json reports/backtest_m5_games.csv \
  reports/backtest_m5_gbl.json reports/backtest_m5_gbl_games.csv 2>/dev/null
echo "RUNTIME euroleague $el s (limit 1800 s), gbl $gbl s (limit 600 s)"
rec_el=$(grep -oE '^RUNTIME euroleague [0-9]+' reports/week14-16_progress.md | tail -1 | awk '{print $3}')
rec_gbl=$(grep -oE '^RUNTIME gbl [0-9]+' reports/week14-16_progress.md | tail -1 | awk '{print $3}')
echo "recorded RUNTIME euroleague ${rec_el:-none} s, gbl ${rec_gbl:-none} s"
[ "$el" -lt 1800 ] && [ "$gbl" -lt 600 ] && [ -n "$rec_el" ] && [ "$rec_el" -lt 1800 ] \
  && [ -n "$rec_gbl" ] && [ "$rec_gbl" -lt 600 ]
