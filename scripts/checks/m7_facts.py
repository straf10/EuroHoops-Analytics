"""Week 14-16 M7 §3 facts, re-checked on the local data (part of checklist item 48).

1. Season shapes: every scored season's regular season in the games mart (teams, rounds, games
   per team, played, forfeits) equals ``tests/fixtures/m7_seasons.json``, the summary K2's
   format test reads; 2020-21 has every game played and no forfeit.
2. Official tables: the cached EuroLeague API final tables
   (``data/raw/euroleague/standings/E{season}_r{round}.xml``) equal
   ``tests/fixtures/m7_official_tables.json``, and ``standings.rank`` on the mart's results
   (regulation scores for overtime games, the recorded deductions) reproduces each one.
3. Cut lines: the EuroLeague playoff field is the table's top 8 (2016-2022); from 2023 the play-in
   field is places 7-10 and places 1-6 reach the playoffs. GBL: the quarterfinal field is the
   table's top 8 in every scored season, and it is not in 2023-24 and 2024-25 (why they are
   excluded).
4. Playoff home order: in every scored EuroLeague season, games 1 and 2 of each playoff series
   are at the better-placed team.
5. 2026-27 schedules: EuroLeague 380 games, 38 per team, every ordered pair once; GBL 183 rows,
   one duplicated ordered pair (Olympiacos-AEK), 182 after dropping it.
"""

import json
import re
import sys
from pathlib import Path

from eurohoops.config import EUROLEAGUE, GBL, MART_PATH
from eurohoops.marts import read_games
from eurohoops.sim.played import regulation_scores, season_results
from eurohoops.standings import rank

FIXTURES = Path("tests/fixtures")
SEASONS = json.loads((FIXTURES / "m7_seasons.json").read_text(encoding="utf-8"))
OFFICIAL = json.loads((FIXTURES / "m7_official_tables.json").read_text(encoding="utf-8"))
GBL_EXCLUDED = (2023, 2024)  # quarterfinal field != table top 8 (fact 3)

ok = True
games = {c.name: read_games(MART_PATH, c.name) for c in (EUROLEAGUE, GBL)}


def shape(competition: str, season: int) -> dict[str, object]:
    g = games[competition]
    rs = g[(g["season"] == season) & (g["phase"] == "RS")]
    teams = sorted({*rs["home"], *rs["away"]})
    per_team = sorted({int(((rs["home"] == t) | (rs["away"] == t)).sum()) for t in teams})
    return {
        "teams": len(teams),
        "rounds": int(rs["round"].nunique()),
        "games": len(rs),
        "games_per_team": per_team,
        "played": int(rs["played"].sum()),
        "forfeits": int(rs["forfeit"].sum()),
    }


for key, recorded in SEASONS.items():
    competition, season = key.split("_")
    got = shape(competition, int(season))
    print(f"1 {key}: {got}")
    ok &= got == recorded
ok &= SEASONS["euroleague_2020"]["played"] == SEASONS["euroleague_2020"]["games"]
ok &= SEASONS["euroleague_2020"]["forfeits"] == 0

el = games[EUROLEAGUE.name]
tables: dict[int, list[str]] = {}
for key, entry in OFFICIAL.items():
    season = int(key.split("_")[1])
    path = Path(entry["cache"])
    xml = path.read_text(encoding="utf-8")
    rows = re.findall(r"<code>(\w+)</code><ranking>(\d+)</ranking>", xml)
    cached = [code for code, _ in sorted(rows, key=lambda r: int(r[1]))]
    rs = el[(el["season"] == season) & (el["phase"] == "RS")]
    results = season_results(rs, regulation_scores(EUROLEAGUE.raw_dir, rs))
    ours = rank(results, entry["deducted_wins"])
    print(f"2 {key}: cache = fixture {cached == entry['table']}, rank = official {ours == cached}")
    ok &= cached == entry["table"] and ours == cached
    tables[season] = cached

for season, table in tables.items():
    sg = el[el["season"] == season]
    po = {*sg[sg["phase"] == "PO"]["home"], *sg[sg["phase"] == "PO"]["away"]}
    pi = {*sg[sg["phase"] == "PI"]["home"], *sg[sg["phase"] == "PI"]["away"]}
    if season < 2023:
        good = po == set(table[:8]) and not pi
    else:
        good = pi == set(table[6:10]) and set(table[:6]) <= po and len(po) == 8
    print(f"3 euroleague {season}: playoff/play-in field matches the cut lines {good}")
    ok &= good

gbl = games[GBL.name]
for key in SEASONS:
    competition, season_s = key.split("_")
    if competition != "gbl":
        continue
    season = int(season_s)
    sg = gbl[gbl["season"] == season]
    table = rank(season_results(sg[sg["phase"] == "RS"]))
    po = sg[sg["phase"] == "PO"]
    first = po[po["round"] == po["round"].min()]
    field = {*first["home"], *first["away"]}
    good = field == set(table[:8])
    expected = season not in GBL_EXCLUDED
    print(f"3 gbl {season}: quarterfinal field = table top 8 {good} (expected {expected})")
    ok &= good == expected

for season, table in tables.items():
    po = el[(el["season"] == season) & (el["phase"] == "PO")].sort_values("tipoff_utc")
    bad = 0
    for _, series in po.groupby(po.apply(lambda g: frozenset((g["home"], g["away"])), axis=1)):
        first_two = series.head(2)
        better = min(series["home"].iloc[0], series["away"].iloc[0], key=table.index)
        bad += int((first_two["home"] != better).sum())
    print(f"4 euroleague {season}: playoff games 1-2 away from the better-placed team: {bad}")
    ok &= bad == 0

LIVE = (("euroleague", 380, [38], 380), ("gbl", 183, [26, 27], 182))
for competition, rows, per_team, pairs in LIVE:
    g = games[competition]
    live = g[(g["season"] == 2026) & (g["phase"] == "RS")]
    teams = sorted({*live["home"], *live["away"]})
    counts = sorted({int(((live["home"] == t) | (live["away"] == t)).sum()) for t in teams})
    unique = len(live.drop_duplicates(["home", "away"]))
    print(f"5 {competition} 2026-27: {len(live)} rows, games per team {counts}, pairs {unique}")
    ok &= len(live) == rows and counts == per_team and unique == pairs

print("m7 facts:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
