"""Week 14-16 §3 facts, re-checked on the local data (part of checklist item 43).

1. Every tip-off in both competitions is a real time: none sits at 00:00 UTC.
2. No game (2015-2025, either competition) comes <= 1.5 days after the team's previous game in
   either competition (why J-e's ``b2b`` became ``short_rest``, decision D1).
3. PAN/OLY games <= 3 days after a game in the other competition, per season and competition,
   equal the counts recorded in ``reports/week14-16_progress.md``.
4. M1's committed replay inputs are the ones M5 reads (``tuned`` and ``comparison_elo``).
"""

import json
import sys

import pandas as pd

from eurohoops.config import EUROLEAGUE, GBL, GREEK_EL_CLUBS, MART_PATH
from eurohoops.marts import read_games

TWO_COMPETITION_WEEKS = {
    2018: (3, 38),
    2019: (1, 17),
    2020: (5, 16),
    2021: (8, 29),
    2022: (5, 37),
    2023: (7, 41),
    2024: (9, 45),
    2025: (13, 44),
}

ok = True
el = read_games(MART_PATH, EUROLEAGUE.name)
gbl = read_games(MART_PATH, GBL.name)

for name, games in (("EL", el), ("GBL", gbl)):
    t = games["tipoff_utc"].dt.tz_convert("UTC")
    midnight = int(((t.dt.hour == 0) & (t.dt.minute == 0)).sum())
    print(f"1 {name}: {midnight} tip-offs at 00:00 UTC of {len(games)}")
    ok &= midnight == 0

gbl_mapped = gbl.assign(
    home=gbl["home"].map(lambda c: GREEK_EL_CLUBS.get(c, c)),
    away=gbl["away"].map(lambda c: GREEK_EL_CLUBS.get(c, c)),
)
both = pd.concat([el.assign(comp="EL"), gbl_mapped.assign(comp="GBL")])
both = both[both["played"]]
rows = pd.concat(
    [
        both[["game_id", "season", "tipoff_utc", "comp", side]].rename(columns={side: "team"})
        for side in ("home", "away")
    ]
).sort_values(["team", "tipoff_utc", "game_id"])
rows["prev_comp"] = rows.groupby("team")["comp"].shift()
rows["days"] = rows.groupby("team")["tipoff_utc"].diff().dt.total_seconds() / 86400

recent = rows[rows["season"].between(2015, 2025)]
short = int((recent["days"] <= 1.5).sum())
print(f"2 games <= 1.5 days after the previous one (2015-2025): {short}")
ok &= short == 0

greek = rows[rows["team"].isin(["PAN", "OLY"]) & (rows["prev_comp"] != rows["comp"])]
greek = greek[greek["days"] <= 3]
counts = greek.groupby(["season", "comp"]).size().to_dict()
for season, (n_el, n_gbl) in TWO_COMPETITION_WEEKS.items():
    got = (counts.get((season, "EL"), 0), counts.get((season, "GBL"), 0))
    print(f"3 {season}: EL/GBL {got} (recorded {(n_el, n_gbl)})")
    ok &= got == (n_el, n_gbl)

assert EUROLEAGUE.m1 is not None
m1 = json.loads(EUROLEAGUE.m1.report.read_text(encoding="utf-8"))
margin = m1["tuned"]["margin"]
elo = m1["comparison_elo"]
inputs_ok = (
    margin["variant"] == "student_t_const"
    and margin["df"] == 7.0
    and (elo["k"], elo["hca"], elo["reversion"]) == (20.0, 90.0, 0.25)
    and m1["metrics"]["validation"]["m1"]["log_loss"] == 0.587714
)
print(f"4 M1 replay inputs as recorded: {inputs_ok}")
ok &= inputs_ok

print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
