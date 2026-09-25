"""Checklist item 11: team_games coverage, points and the EuroLeague possession sample."""

import json
import sys
from pathlib import Path

AGREEMENT_TARGET = 0.9

report = json.loads(Path("reports/possessions.json").read_text(encoding="utf-8"))
ok = True
for competition, seasons in report["coverage"].items():
    for season, c in seasons.items():
        if c["with_rows"] + c["missing"] != c["rated_games"]:
            print("coverage hole", competition, season, c)
            ok = False
counts = [c for seasons in report["coverage"].values() for c in seasons.values()]
print(
    "rated games:",
    sum(c["rated_games"] for c in counts),
    "with rows:",
    sum(c["with_rows"] for c in counts),
    "missing (listed with reasons):",
    len(report["missing_games"]),
)
if report["points_mismatches"]:
    print("points mismatches", report["points_mismatches"])
    ok = False
else:
    print("points = games scores for every row")
s = report["euroleague_pbp_sample"]
print(
    f"EL sample: {s['games']} games, within ±2: {s['within_tolerance_share_of_teams']:.1%} of "
    f"team-games, {s['within_tolerance_share_of_games']:.1%} of games; mean gap "
    f"{s['mean_gap_pbp_minus_box']}; FT weight matching PBP {s['ft_weight_matching_pbp']}"
)
if s["within_tolerance_share_of_teams"] < AGREEMENT_TARGET:
    doc = Path("docs/data/possessions.md").read_text(encoding="utf-8")
    explained = "Why the target is missed" in doc
    print("below 90%: explanation in docs/data/possessions.md:", explained)
    ok = ok and explained
sys.exit(0 if ok else 1)
