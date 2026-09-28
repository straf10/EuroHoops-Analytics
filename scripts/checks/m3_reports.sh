#!/usr/bin/env bash
# Checklist item 35: backtest_m3.json complete with its gate block; two backtest runs give
# byte-identical JSON equal to the committed report; verdict < validation < test in git.
cd "$ROOT" || exit 1
f=reports/backtest_m3.json
flag=$(uv run python -c "import json;print('--score-test' if json.load(open('$f')).get('test_scored') else '')")
uri="sqlite:///$SCRATCH/mlflow_m3.db"
start=$(date +%s)
uv run eurohoops backtest --model m3 $flag --tracking-uri "$uri" >/dev/null 2>&1 || exit 1
echo $(( $(date +%s) - start )) > "$SCRATCH/m3_runtime_s"
cp "$f" "$SCRATCH/m3_run1.json"
uv run eurohoops backtest --model m3 $flag --tracking-uri "$uri" >/dev/null 2>&1 || exit 1
cmp -s "$f" "$SCRATCH/m3_run1.json" && echo "$f: two runs byte-identical" || { echo "$f: runs differ"; exit 1; }
git diff --quiet -- "$f" && echo "$f: equals the committed report" || { echo "$f differs from the committed report"; exit 1; }
uv run python -c "
import json, sys
r = json.load(open('$f'))
assert r['validation_scored'], 'validation not scored'
for split, models in r['metrics'].items():
    for model in ('rapm', 'box_only', 'pir', 'm1', 'b0'):
        m = models[model]
        assert all(m[k] is not None for k in ('n', 'rmse', 'mae', 'log_loss', 'sigma')), (split, model)
g = r['gate']
assert all(g[k] is not None for k in ('rule', 'metric', 'variant', 'reference', 'rmse_diff', 'passed')), g
assert r['oracle_minutes']['label'] == 'oracle, not a forecast'
print('  gate', g['variant'], 'PASS' if g['passed'] else 'FAIL', g['rmse_diff']['mean'], g['rmse_diff']['ci95'])
" || exit 1
uv run python "$ROOT/scripts/checks/m3_order.py"
