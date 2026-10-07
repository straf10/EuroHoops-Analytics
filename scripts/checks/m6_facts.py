"""Weeks 16-18 M6 §3 facts, re-checked on the local data (part of checklist item 54).

1. Possessions: every player game with seconds has a possession count, in both leagues and every
   season (EuroLeague 2007+, GBL box seasons 2018+).
2. Scored-set sizes per competition and season (``models.player_seasons``: >= 500 possessions in
   the season and a senior season before it in either league) and the >= 500 counts equal
   ``tests/fixtures/m6_facts.json``.
3. Birth-date coverage among the scored player-seasons equals the fixture (all 1.0 on 2026-10-07).
   Only the share is printed: no date or age is.
4. M4 walk-forward: ``fits_by_target_season`` covers 2019-2025; the targets without a fit
   (2016-2018) cannot have a GBL->EL mover, because GBL box scores start in 2018-19.
5. M3 walk-forward: ``m3_players.json`` holds a season-end BRAPM snapshot for 2011-2025, each
   cut at its own season's last round, so a target's projection reads only earlier snapshots.
6. Four factors: the EuroLeague and GBL player box carries every input (FGM/FGA by type, FTA,
   OREB, DREB, TOV, possessions), so team four factors are derivable from the marts' sources.
"""

import json
import sys
from pathlib import Path

import pandas as pd

from eurohoops.config import EUROLEAGUE, GBL, M4, MART_PATH, PLAYER_BIOS
from eurohoops.marts import read_games, read_table
from eurohoops.models.player_seasons import build_player_seasons
from eurohoops.parse.gbl_box_lines import build_gbl_player_games
from eurohoops.stats.box import build_box_games

FACTS = json.loads(Path("tests/fixtures/m6_facts.json").read_text(encoding="utf-8"))
FOUR_FACTOR_INPUTS = ("fg2m", "fg2a", "fg3m", "fg3a", "fta", "oreb", "dreb", "tov", "poss")

results: list[bool] = []


def check(label: str, passed: bool) -> None:
    results.append(passed)
    print(f"{'PASS' if passed else 'FAIL'}  {label}")


el_games = read_games(MART_PATH, EUROLEAGUE.name)
gbl_games = read_games(MART_PATH, GBL.name)
gbl_team_games = read_table(MART_PATH, "team_games", GBL.name)
xwalk = read_table(MART_PATH, "player_xwalk")
if gbl_team_games is None or xwalk is None:
    sys.exit("marts missing team_games or player_xwalk; run: eurohoops build")
games = {
    "euroleague": build_box_games(EUROLEAGUE.raw_dir, el_games).players,
    "gbl": build_gbl_player_games(GBL.raw_dir, gbl_games, gbl_team_games).table,
}

# 1. possessions
for comp, lines in games.items():
    played = lines[lines["sec"] > 0]
    bad = int((played["poss"].isna() | (played["poss"] <= 0)).sum())
    check(f"1 {comp}: {len(played)} player games with seconds, {bad} without possessions", bad == 0)

# 2-3. scored sets and birth-date coverage
seasons = build_player_seasons(games, xwalk)
seasons = seasons[seasons["season"] < FACTS["live_season"]]
bios = pd.read_parquet(PLAYER_BIOS).dropna(subset=["birth_date"])
dated = {(c, str(s)) for c, s in zip(bios["competition"], bios["source_id"], strict=True)}
keys = zip(xwalk["competition"], xwalk["source_id"].astype(str), strict=True)
dated_persons = set(xwalk.loc[[key in dated for key in keys], "person_id"])
found: dict[str, dict[str, dict[str, float]]] = {}
for (comp, season), rows in seasons.groupby(["competition", "season"]):
    scored = rows[(rows["poss"] >= FACTS["min_poss"]) & (rows["season"] > rows["debut_season"])]
    birth = float(scored["person_id"].isin(dated_persons).mean()) if len(scored) else None
    found.setdefault(str(comp), {})[str(season)] = {
        "ge_min_poss": int((rows["poss"] >= FACTS["min_poss"]).sum()),
        "scored": len(scored),
        "birth_share": None if birth is None else round(birth, 3),
    }
for comp, expected in FACTS["scored"].items():
    for season, want in expected.items():
        got = found.get(comp, {}).get(season)
        check(f"2-3 {comp} {season}: {got} (fixture {want})", got == want)

# 4. M4 fits
m4 = json.loads(M4.translation_report.read_text(encoding="utf-8"))
fits = sorted(int(s) for s in m4["fits_by_target_season"])
check(f"4 M4 fits for target seasons {fits}", fits == FACTS["m4_fit_targets"])
first_gbl = int(seasons.loc[seasons["competition"] == "gbl", "season"].min())
check(f"4 first GBL box season {first_gbl}", first_gbl == FACTS["first_gbl_box_season"])

# 5. M3 snapshots
m3 = json.loads(Path("reports/m3_players.json").read_text(encoding="utf-8"))
snaps = sorted(int(s) for s in m3["seasons"])
check(f"5 M3 BRAPM snapshots {snaps[0]}-{snaps[-1]}", snaps == FACTS["m3_snapshot_seasons"])
late = [
    s
    for s in snaps
    if pd.Timestamp(m3["seasons"][str(s)]["cutoff_utc"])
    > pd.Timestamp(el_games.loc[el_games["season"] == s, "tipoff_utc"].max())
]
check(f"5 snapshots cut after their own season's last game (must be none): {late}", not late)

# 6. four-factor inputs
for comp, lines in games.items():
    missing = [c for c in FOUR_FACTOR_INPUTS if c not in lines.columns]
    check(f"6 {comp}: four-factor inputs missing {missing}", not missing)

ok = all(results)
print("m6 facts:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
