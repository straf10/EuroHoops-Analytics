#!/usr/bin/env bash
# Checklist item 13: M1 unit tests, and the full EuroLeague M1 backtest under 10 minutes.
cd "$ROOT" || exit 1
uv run pytest -q -p no:cacheprovider tests/test_team_eff.py 2>&1 | tail -1
[ "${PIPESTATUS[0]}" = 0 ] || exit 1
flag=$(uv run python -c "import json;print('--score-test' if json.load(open('reports/backtest_m1.json')).get('test_scored') else '')")
start=$(date +%s)
uv run eurohoops backtest --model m1 $flag >/dev/null 2>&1 || exit 1
secs=$(( $(date +%s) - start ))
echo "full EuroLeague M1 backtest ($flag): ${secs} s (limit 600 s)"
[ "$secs" -lt 600 ]
