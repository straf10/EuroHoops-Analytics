#!/usr/bin/env bash
# Checklist item 15: both M1 reports hold every metric for every split, two backtest runs give
# byte-identical JSON, and that JSON equals the committed report.
cd "$ROOT" || exit 1
for comp in euroleague gbl; do
  f=reports/backtest_m1.json; [ "$comp" = gbl ] && f=reports/backtest_m1_gbl.json
  flag=$(uv run python -c "import json;print('--score-test' if json.load(open('$f')).get('test_scored') else '')")
  uv run eurohoops backtest --model m1 --competition "$comp" $flag >/dev/null 2>&1 || exit 1
  cp "$f" "$SCRATCH/m1_run1_$comp.json"
  uv run eurohoops backtest --model m1 --competition "$comp" $flag >/dev/null 2>&1 || exit 1
  cmp -s "$f" "$SCRATCH/m1_run1_$comp.json" && echo "$f: two runs byte-identical" || { echo "$f differs"; exit 1; }
  git diff --quiet -- "$f" && echo "$f: equals the committed report" || { echo "$f differs from the committed report"; exit 1; }
  uv run python -c "
import json
r = json.load(open('$f'))
keys = ('log_loss', 'brier', 'accuracy', 'ece', 'reliability', 'margin_mae', 'margin_crps', 'totals_mae', 'totals_crps')
for split, models in r['metrics'].items():
    for model in ('m1', 'elo', 'b0'):
        assert all(models[model][k] is not None for k in keys), (split, model)
g = r['gate']
print('  gate', g['variant'], 'passed' if g['passed'] else 'FAILED', g['log_loss_m1_minus_elo']['mean'], g['log_loss_m1_minus_elo']['ci95'])
" || exit 1
done
