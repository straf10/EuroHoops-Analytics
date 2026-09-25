#!/usr/bin/env bash
# Checklist item 12: stints mart pass rates at or above the thresholds, stint points = final in
# every passing game, and two builds give identical tables and an identical report.
cd "$ROOT" || exit 1
tabhash() {
  uv run python -c "
import duckdb
from eurohoops.parse.stints_mart import digest
c = duckdb.connect('data/marts/eurohoops.duckdb', read_only=True)
for t in ('stints', 'stint_game_checks'):
    print(t, digest(c.execute(f'SELECT * FROM {t}').df()))
"
}
uv run eurohoops stints --mart || exit 1
h1=$(tabhash); cp reports/stints_mart.json "$SCRATCH/stints_mart_run1.json"
uv run eurohoops stints --mart >/dev/null || exit 1
h2=$(tabhash)
echo "$h1"
[ "$h1" = "$h2" ] && echo "two builds: stored tables identical" || { echo "tables differ"; exit 1; }
cmp -s "$SCRATCH/stints_mart_run1.json" reports/stints_mart.json \
  && echo "two builds: report byte-identical" || { echo "report differs"; exit 1; }
uv run python -c "
import json, sys
r = json.load(open('reports/stints_mart.json'))
print('2011-14', r['pass_rate_2011_2014'], '(>= 0.95); 2015+', r['pass_rate_2015_on'], '(>= 0.985)')
print('passing games where stint points miss the final:', r['passing_games_where_stint_points_miss_the_final'])
ok = r['pass_rate_2011_2014'] >= 0.95 and r['pass_rate_2015_on'] >= 0.985
sys.exit(0 if ok and not r['passing_games_where_stint_points_miss_the_final'] else 1)"
