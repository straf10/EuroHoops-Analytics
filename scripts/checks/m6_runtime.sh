#!/usr/bin/env bash
# Checklist item 58: the L-m budgets. Timed now: `backtest --model m6` (EuroLeague < 1,200 s,
# GBL < 300 s; item 56's runs when it ran in the same checklist) and `project --dry-run` (< 90 s).
# Recorded in the progress file: the same three, the export through the API (< 180 s) and the
# Astro build (< 240 s).
cd "$ROOT" || exit 1
export MLFLOW_DISABLE_AGENT_HINT=1
p=reports/week16-18_progress.md
flag=$(uv run python -c "import json;print('--score-test' if json.load(open('reports/backtest_m6.json')).get('test_scored') else '')")
timed() {  # $1 = competition, $2 = cache file
  if [ -s "$SCRATCH/$2" ]; then cat "$SCRATCH/$2"; return; fi
  start=$(date +%s)
  uv run eurohoops backtest --model m6 --competition "$1" $flag \
    --tracking-uri "sqlite:///$(cygpath -m "$SCRATCH")/mlflow_m6.db" >/dev/null 2>&1 || { echo 99999; return; }
  echo $(( $(date +%s) - start ))
}
el=$(timed euroleague m6_runtime_s)
gbl=$(timed gbl m6_gbl_runtime_s)
git checkout -q -- reports/backtest_m6.json reports/backtest_m6_players.csv \
  reports/backtest_m6_gbl.json reports/backtest_m6_gbl_players.csv 2>/dev/null
start=$(date +%s)
uv run eurohoops project --dry-run >/dev/null 2>&1 || exit 1
proj=$(( $(date +%s) - start ))
echo "RUNTIME m6 euroleague $el s (limit 1200 s), m6 gbl $gbl s (limit 300 s), project $proj s (limit 90 s)"
rec() { grep -oE "^RUNTIME $1 [0-9]+" "$p" | tail -1 | awk '{print $NF}'; }
r_el=$(rec "m6 euroleague"); r_gbl=$(rec "m6 gbl"); r_proj=$(rec "project")
r_exp=$(rec "export"); r_astro=$(rec "astro")
echo "recorded m6 euroleague ${r_el:-none}, m6 gbl ${r_gbl:-none}, project ${r_proj:-none}, export ${r_exp:-none}, astro ${r_astro:-none}"
under() { [ -n "$1" ] && [ "$1" -lt "$2" ]; }
under "$el" 1200 && under "$gbl" 300 && under "$proj" 90 && under "$r_el" 1200 && under "$r_gbl" 300 \
  && under "$r_proj" 90 && under "$r_exp" 180 && under "$r_astro" 240
