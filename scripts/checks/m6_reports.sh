#!/usr/bin/env bash
# Checklist item 56: backtest_m6.json complete with its gate block (loss differences with CIs,
# coverage per stat), the movers slice and the rest-of-season blocks; two EuroLeague runs give
# byte-identical JSON and per-player CSV, equal to the committed ones; the GBL report (the
# EuroLeague verdict as a fixed choice) reproduces too; verdict < validation < test in git. Times
# the first EuroLeague and the GBL run for item 58.
cd "$ROOT" || exit 1
f=reports/backtest_m6.json
g=reports/backtest_m6_players.csv
flag=$(uv run python -c "import json;print('--score-test' if json.load(open('$f')).get('test_scored') else '')")
uri="sqlite:///$(cygpath -m "$SCRATCH")/mlflow_m6.db"  # a drive path: MLflow needs an absolute Windows path
export MLFLOW_DISABLE_AGENT_HINT=1
start=$(date +%s)
uv run eurohoops backtest --model m6 $flag --tracking-uri "$uri" >"$SCRATCH/m6_out.txt" 2>&1 || { tail -5 "$SCRATCH/m6_out.txt"; exit 1; }
echo $(( $(date +%s) - start )) > "$SCRATCH/m6_runtime_s"
cp "$f" "$SCRATCH/m6_run1.json"; cp "$g" "$SCRATCH/m6_run1.csv"
uv run eurohoops backtest --model m6 $flag --tracking-uri "$uri" >"$SCRATCH/m6_out.txt" 2>&1 || { tail -5 "$SCRATCH/m6_out.txt"; exit 1; }
cmp -s "$f" "$SCRATCH/m6_run1.json" && cmp -s "$g" "$SCRATCH/m6_run1.csv" \
  && echo "$f, $g: two runs byte-identical" || { echo "M6 runs differ"; exit 1; }
git diff --quiet -- "$f" "$g" && echo "$f, $g: equal the committed files" \
  || { echo "M6 report differs from the committed one"; exit 1; }
start=$(date +%s)
uv run eurohoops backtest --model m6 --competition gbl $flag --tracking-uri "$uri" >"$SCRATCH/m6_out.txt" 2>&1 || { tail -5 "$SCRATCH/m6_out.txt"; exit 1; }
echo $(( $(date +%s) - start )) > "$SCRATCH/m6_gbl_runtime_s"
git diff --quiet -- reports/backtest_m6_gbl.json reports/backtest_m6_gbl_players.csv \
  && echo "GBL M6 report equals the committed one" || { echo "GBL M6 report differs"; exit 1; }
uv run python -c "
import json
r = json.load(open('$f'))
assert r['validation_scored'], 'validation not scored'
chosen = '{}@{:g}'.format(r['chosen']['variant'], r['chosen']['half_life'])
for split in ('tuning', 'validation', 'test'):
    block = r[split]
    for model in (chosen, 'marcel', 'same_as_last', 'league_mean'):
        assert block['next_season'][model]['loss'] is not None, (split, model)
        for f in r['checkpoints']:
            assert block['rest_of_season']['{:g}'.format(f)][model]['loss'] is not None, (split, f, model)
    assert r['movers'][split][chosen]['n'] >= 0, split
    assert all(d['ci95'] is not None for d in r['differences'][split].values()), split
gate = r['gate']
for k in ('rule', 'variant', 'loss_chosen', 'loss_marcel', 'loss_same_as_last', 'coverage_ok', 'passed'):
    assert gate.get(k) is not None, k
assert gate['gated'] is True and set(gate['coverage_pooled']) == set(r['stats']), 'coverage per stat'
for k in ('loss_vs_marcel', 'loss_vs_same_as_last'):
    assert gate[k]['ci95'] is not None, k
gbl = json.load(open('reports/backtest_m6_gbl.json'))
assert (gbl['chosen']['variant'], gbl['chosen']['half_life']) == (r['chosen']['variant'], r['chosen']['half_life']), 'GBL not the EL verdict'
assert gbl['gate']['gated'] is False, 'the GBL comparison must be labelled as not gated'
print('  gate', chosen, 'PASS' if gate['passed'] else 'FAIL', gate['loss_chosen'], gate['loss_marcel'],
      gate['loss_same_as_last'], 'coverage ok', gate['coverage_ok'])
" || exit 1
uv run python "$ROOT/scripts/checks/m6_order.py"
