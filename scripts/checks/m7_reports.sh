#!/usr/bin/env bash
# Checklist item 50: backtest_m7.json complete with its gate block (Brier differences with CIs,
# Spiegelhalter z) and reliability; two EuroLeague runs give byte-identical JSON and
# per-team-checkpoint CSV, equal to the committed ones; the GBL report (the EuroLeague verdict as
# a fixed choice) reproduces too; verdict < validation < test in git. Times the first EuroLeague
# and the GBL run for item 51.
cd "$ROOT" || exit 1
f=reports/backtest_m7.json
g=reports/backtest_m7_teams.csv
flag=$(uv run python -c "import json;print('--score-test' if json.load(open('$f')).get('test_scored') else '')")
uri="sqlite:///$(cygpath -m "$SCRATCH")/mlflow_m7.db"  # a drive path: MLflow needs an absolute Windows path
export MLFLOW_DISABLE_AGENT_HINT=1
start=$(date +%s)
uv run eurohoops backtest --model m7 $flag --tracking-uri "$uri" >"$SCRATCH/m7_out.txt" 2>&1 || { tail -5 "$SCRATCH/m7_out.txt"; exit 1; }
echo $(( $(date +%s) - start )) > "$SCRATCH/m7_runtime_s"
cp "$f" "$SCRATCH/m7_run1.json"; cp "$g" "$SCRATCH/m7_run1.csv"
uv run eurohoops backtest --model m7 $flag --tracking-uri "$uri" >"$SCRATCH/m7_out.txt" 2>&1 || { tail -5 "$SCRATCH/m7_out.txt"; exit 1; }
cmp -s "$f" "$SCRATCH/m7_run1.json" && cmp -s "$g" "$SCRATCH/m7_run1.csv" \
  && echo "$f, $g: two runs byte-identical" || { echo "M7 runs differ"; exit 1; }
git diff --quiet -- "$f" "$g" && echo "$f, $g: equal the committed files" \
  || { echo "M7 report differs from the committed one"; exit 1; }
start=$(date +%s)
uv run eurohoops backtest --model m7 --competition gbl $flag --tracking-uri "$uri" >"$SCRATCH/m7_out.txt" 2>&1 || { tail -5 "$SCRATCH/m7_out.txt"; exit 1; }
echo $(( $(date +%s) - start )) > "$SCRATCH/m7_gbl_runtime_s"
git diff --quiet -- reports/backtest_m7_gbl.json reports/backtest_m7_gbl_teams.csv \
  && echo "GBL M7 report equals the committed one" || { echo "GBL M7 report differs"; exit 1; }
uv run python -c "
import json
r = json.load(open('$f'))
assert r['validation_scored'], 'validation not scored'
for split, models in r['metrics'].items():
    for model in ('point_sim', 'elo_sim', 'standings_now', r['chosen']['key']):
        assert models[model]['brier'] is not None, (split, model)
        assert r['reliability'][split][model], (split, model)
gate = r['gate']
for k in ('rule', 'variant', 'brier_chosen', 'brier_point_sim', 'brier_standings_now', 'spiegelhalter_z_pooled', 'passed'):
    assert gate.get(k) is not None, k
assert gate['gated'] is True
diffs = r['differences']['validation']
assert all(diffs[b]['ci95'] is not None for b in diffs), diffs
gbl = json.load(open('reports/backtest_m7_gbl.json'))
assert gbl['chosen']['key'] == r['chosen']['key'] and 'fixed' in gbl['chosen'], 'GBL not the EL verdict'
assert gbl['gate']['gated'] is False, 'the GBL comparison must be labelled as not gated'
print('  gate', gate['variant'], 'PASS' if gate['passed'] else 'FAIL', gate['brier_chosen'],
      gate['brier_point_sim'], gate['brier_standings_now'], gate['spiegelhalter_z_pooled'])
" || exit 1
uv run python "$ROOT/scripts/checks/m7_order.py"
