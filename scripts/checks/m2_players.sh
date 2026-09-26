#!/usr/bin/env bash
# Checklist item 26 (F7): reports/m2_players.json present with CIs, and its stability verdict is
# what the F-k rule gives on its own numbers (a test recomputes it).
cd "$ROOT" || exit 1
uv run pytest -q -p no:cacheprovider tests/test_m2_reports.py -k players 2>&1 | tail -1
[ "${PIPESTATUS[0]}" = 0 ] || exit 1
uv run python -c "
import json
s = json.load(open('reports/m2_players.json'))['stability']
print('year-to-year', s['year_to_year']['shrunk_shot_making'], 'split-half', s['split_half'], '->', s['verdict'])"
