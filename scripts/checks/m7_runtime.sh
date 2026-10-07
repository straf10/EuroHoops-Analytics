#!/usr/bin/env bash
# Checklist item 51: `backtest --model m7` under 1,800 s for the EuroLeague and 600 s for the
# GBL, and `simulate` (10,000 simulations, one competition) under 120 s (K-k); the last
# RUNTIME lines recorded in the progress file within the same limits. Reuses item 50's timed
# runs when it ran in the same checklist.
cd "$ROOT" || exit 1
export MLFLOW_DISABLE_AGENT_HINT=1
p=reports/m7_progress.md
flag=$(uv run python -c "import json;print('--score-test' if json.load(open('reports/backtest_m7.json')).get('test_scored') else '')")
timed() {  # $1 = competition, $2 = cache file
  if [ -s "$SCRATCH/$2" ]; then cat "$SCRATCH/$2"; return; fi
  start=$(date +%s)
  uv run eurohoops backtest --model m7 --competition "$1" $flag \
    --tracking-uri "sqlite:///$(cygpath -m "$SCRATCH")/mlflow_m7.db" >/dev/null 2>&1 || { echo 99999; return; }
  echo $(( $(date +%s) - start ))
}
el=$(timed euroleague m7_runtime_s)
gbl=$(timed gbl m7_gbl_runtime_s)
git checkout -q -- reports/backtest_m7.json reports/backtest_m7_teams.csv \
  reports/backtest_m7_gbl.json reports/backtest_m7_gbl_teams.csv 2>/dev/null
echo "RUNTIME euroleague $el s (limit 1800 s), gbl $gbl s (limit 600 s)"
rec() { grep -oE "^RUNTIME $1 [0-9]+" "$p" | tail -1 | awk '{print $NF}'; }
rec_el=$(rec "m7 euroleague"); rec_gbl=$(rec "m7 gbl"); rec_sim=$(rec "simulate")
echo "recorded RUNTIME m7 euroleague ${rec_el:-none} s, m7 gbl ${rec_gbl:-none} s, simulate ${rec_sim:-none} s"
[ "$el" -lt 1800 ] && [ "$gbl" -lt 600 ] && [ -n "$rec_el" ] && [ "$rec_el" -lt 1800 ] \
  && [ -n "$rec_gbl" ] && [ "$rec_gbl" -lt 600 ] && [ -n "$rec_sim" ] && [ "$rec_sim" -lt 120 ]
