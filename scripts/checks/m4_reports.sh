#!/usr/bin/env bash
# Checklist item 41: backtest_m4.json complete with its gate block; two `backtest --model m4`
# runs give byte-identical reports (both files) equal to the committed ones; verdict <
# validation < test in git; RUNTIME of the run and the one recorded in the progress file < 600 s.
cd "$ROOT" || exit 1
f=reports/backtest_m4.json
t=reports/m4_translation.json
flag=$(uv run python -c "import json;print('--score-test' if json.load(open('$f')).get('test_scored') else '')")
PYTHONIOENCODING=utf-8 uv run eurohoops backtest --model m4 $flag > "$SCRATCH/m4_run1.txt" 2>&1 || { tail -5 "$SCRATCH/m4_run1.txt"; exit 1; }
cp "$f" "$SCRATCH/m4_run1.json"; cp "$t" "$SCRATCH/m4_translation_run1.json"
PYTHONIOENCODING=utf-8 uv run eurohoops backtest --model m4 $flag > "$SCRATCH/m4_run2.txt" 2>&1 || exit 1
for g in "$f" "$t"; do
  [ "$g" = "$f" ] && prev="$SCRATCH/m4_run1.json" || prev="$SCRATCH/m4_translation_run1.json"
  cmp -s "$g" "$prev" && echo "$g: two runs byte-identical" || { echo "$g: runs differ"; exit 1; }
  git diff --quiet -- "$g" && echo "$g: equals the committed report" || { echo "$g differs from the committed report"; exit 1; }
done
uv run python -c "
import json
r = json.load(open('$f'))
assert r['validation_scored'], 'validation not scored'
g = r['gate']
assert all(g[k] is not None for k in ('rule', 'metric', 'variant', 'reference', 'n_movers', 'passed')), g
assert g['loss_diff']['ci95'][1] is not None, g
assert r['el_to_gbl']['label'] == 'reported, not gated'
print('  gate', g['variant'], 'PASS' if g['passed'] else 'FAIL', g['loss_diff']['mean'], g['loss_diff']['ci95'], 'pooled' if g['pooled'] else '', g['n_movers'], 'movers')
" || exit 1
uv run python "$ROOT/scripts/checks/m4_order.py" || exit 1
secs=$(grep -oE '^RUNTIME [0-9]+' "$SCRATCH/m4_run1.txt" | cut -d' ' -f2)
recorded=$(grep -oE '^RUNTIME m4 [0-9]+' reports/week12-14_progress.md | tail -1 | cut -d' ' -f3)
echo "RUNTIME ${secs:-none} s (limit 600 s); recorded RUNTIME m4 ${recorded:-none} s"
[ -n "$secs" ] && [ "$secs" -lt 600 ] && [ -n "$recorded" ] && [ "$recorded" -lt 600 ]
