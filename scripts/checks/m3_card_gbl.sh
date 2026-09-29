#!/usr/bin/env bash
# Checklist item 37: docs/models/m3.md numbers match the reports; gbl_stints.json has pass rates
# per season and check; two GBL stint builds give an identical report equal to the committed one.
cd "$ROOT" || exit 1
uv run pytest -q -p no:cacheprovider tests/test_model_card_m3.py 2>&1 | tail -1
[ "${PIPESTATUS[0]}" = 0 ] || exit 1
f=reports/gbl_stints.json
uv run eurohoops gbl-stints >/dev/null 2>&1 || exit 1
cp "$f" "$SCRATCH/gbl_stints_run1.json"
uv run eurohoops gbl-stints >/dev/null 2>&1 || exit 1
cmp -s "$f" "$SCRATCH/gbl_stints_run1.json" && echo "$f: two builds byte-identical" || { echo "$f: builds differ"; exit 1; }
git diff --quiet -- "$f" && echo "$f: equals the committed report" || { echo "$f differs from the committed report"; exit 1; }
uv run python -c "
import json
r = json.load(open('$f'))
assert r['seasons'], 'no seasons'
for season, s in r['seasons'].items():
    assert s['pass_rate'] is not None and set(s['pass_rate_by_check']) == {'five_on_court', 'points', 'possessions'}, season
    print(' ', season, s['pass_rate'], s['pass_rate_by_check'])
"
f=reports/backtest_m3_gbl.json
flag=$(uv run python -c "import json;print('--score-test' if json.load(open('$f')).get('test_scored') else '')")
uv run eurohoops backtest --model m3 --competition gbl $flag >/dev/null 2>&1 || exit 1
cp "$f" "$SCRATCH/m3_gbl_run1.json"
uv run eurohoops backtest --model m3 --competition gbl $flag >/dev/null 2>&1 || exit 1
cmp -s "$f" "$SCRATCH/m3_gbl_run1.json" && echo "$f: two runs byte-identical" || { echo "$f: runs differ"; exit 1; }
git diff --quiet -- "$f" && echo "$f: equals the committed report" || { echo "$f differs from the committed report"; exit 1; }
uv run python -c "
import json, sys
r = json.load(open('$f'))
assert r['oracle_minutes']['label'] == 'oracle, not a forecast'
assert r['comparisons']['label'].startswith('GBL evidence')
models = ('spm_transfer', 'spm_transfer_scaled', 'box_only', 'pir', 'm1', 'b0')
for split, block in r['metrics'].items():
    for model in models:
        m = block[model]
        assert all(m[k] is not None for k in ('n', 'rmse', 'mae', 'log_loss', 'sigma')), (split, model)
print('  m3_gbl metrics ok for', list(r['metrics']))
" || exit 1
