#!/usr/bin/env bash
# Checklist item 21 (F1): the shot table reconciles with the box scores, exclusions stay <= 1% of
# FGA in every validated season, and two builds give identical tables and an identical report.
cd "$ROOT" || exit 1
tabhash() {
  uv run python -c "
import duckdb
from eurohoops.parse.stints_mart import digest
c = duckdb.connect('data/marts/eurohoops.duckdb', read_only=True)
for t in ('shots', 'shots_excluded'):
    print(t, digest(c.execute(f'SELECT * FROM {t}').df()))
"
}
uv run eurohoops shots || exit 1
h1=$(tabhash); cp reports/shots.json "$SCRATCH/shots_run1.json"
uv run eurohoops shots >/dev/null || exit 1
h2=$(tabhash)
echo "$h1"
[ "$h1" = "$h2" ] && echo "two builds: stored tables identical" || { echo "tables differ"; exit 1; }
cmp -s "$SCRATCH/shots_run1.json" reports/shots.json \
  && echo "two builds: report byte-identical" || { echo "report differs"; exit 1; }
uv run python -c "
import json, sys
r = json.load(open('reports/shots.json'))
rate = r['reconciliation_match_rate_validated']
print('feed = box for', rate, 'of 2011+ team-games (>= 0.99)')
worst = max((b['excluded_share'], s) for s, b in r['seasons'].items() if b['validated'])
print('largest validated-season exclusion share', worst[0], 'in', worst[1], '(<= 0.01)')
for s, b in r['seasons'].items():
    if b['validated']:
        print(' ', s, b['excluded_share'], {k: v['share'] for k, v in b['excluded_by_reason'].items()})
sys.exit(0 if rate >= 0.99 and worst[0] <= 0.01 else 1)"
