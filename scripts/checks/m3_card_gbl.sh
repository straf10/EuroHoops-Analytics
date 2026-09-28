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
