#!/usr/bin/env bash
# Checklist item 45: backtest_m5.json complete with its gate, oracle and gap blocks; two
# EuroLeague runs give byte-identical JSON and per-game CSV, equal to the committed ones; the
# GBL report (the EuroLeague verdict as a fixed choice) reproduces too; verdict < validation <
# test in git. Times the first EuroLeague and the GBL run for item 46.
cd "$ROOT" || exit 1
f=reports/backtest_m5.json
g=reports/backtest_m5_games.csv
flag=$(uv run python -c "import json;print('--score-test' if json.load(open('$f')).get('test_scored') else '')")
uri="sqlite:///$(cygpath -m "$SCRATCH")/mlflow_m5.db"  # a drive path: MLflow needs an absolute Windows path
export MLFLOW_DISABLE_AGENT_HINT=1
start=$(date +%s)
uv run eurohoops backtest --model m5 $flag --tracking-uri "$uri" >"$SCRATCH/m5_out.txt" 2>&1 || { tail -5 "$SCRATCH/m5_out.txt"; exit 1; }
echo $(( $(date +%s) - start )) > "$SCRATCH/m5_runtime_s"
cp "$f" "$SCRATCH/m5_run1.json"; cp "$g" "$SCRATCH/m5_run1.csv"
uv run eurohoops backtest --model m5 $flag --tracking-uri "$uri" >"$SCRATCH/m5_out.txt" 2>&1 || { tail -5 "$SCRATCH/m5_out.txt"; exit 1; }
cmp -s "$f" "$SCRATCH/m5_run1.json" && cmp -s "$g" "$SCRATCH/m5_run1.csv" \
  && echo "$f, $g: two runs byte-identical" || { echo "M5 runs differ"; exit 1; }
git diff --quiet -- "$f" "$g" && echo "$f, $g: equal the committed files" \
  || { echo "M5 report differs from the committed one"; exit 1; }
start=$(date +%s)
uv run eurohoops backtest --model m5 --competition gbl $flag --tracking-uri "$uri" >"$SCRATCH/m5_out.txt" 2>&1 || { tail -5 "$SCRATCH/m5_out.txt"; exit 1; }
echo $(( $(date +%s) - start )) > "$SCRATCH/m5_gbl_runtime_s"
git diff --quiet -- reports/backtest_m5_gbl.json reports/backtest_m5_gbl_games.csv \
  && echo "GBL M5 report equals the committed one" || { echo "GBL M5 report differs"; exit 1; }
uv run python -c "
import json
r = json.load(open('$f'))
assert r['validation_scored'], 'validation not scored'
for split, models in r['metrics'].items():
    for model in ('m5', 'm1', 'elo', 'b0'):
        assert models[model]['log_loss'] is not None, (split, model)
gate = r['gate']
assert all(gate[k] is not None for k in ('rule', 'variant', 'm5_log_loss', 'm1_log_loss', 'passed')), gate
assert r['oracle']['label'] == 'oracle, not a forecast'
assert set(r['gap']) >= {'validation'}
gbl = json.load(open('reports/backtest_m5_gbl.json'))
assert gbl['chosen']['choice'] == r['chosen']['choice'] and 'fixed' in gbl['chosen'], 'GBL not the EL verdict'
assert gbl['gate']['gated'] is False, 'the GBL comparison must be labelled as not gated (J-g)'
ci = gate['log_loss_m5_minus_m1']
print('  gate', gate['variant'], 'PASS' if gate['passed'] else 'FAIL', ci['mean'], ci['ci95'])
" || exit 1
uv run python "$ROOT/scripts/checks/m5_order.py"
