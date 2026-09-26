"""F2: free-throw trips from play-by-play, and-ones tied to shots, out-of-fold expected points."""

import json
from pathlib import Path

import pandas as pd

from eurohoops.ingest.cache import write_atomic
from eurohoops.parse.free_throws import (
    COUNT_COLUMNS,
    build_ft_team_games,
    expected_ft_points,
    fit_rates,
    ft_report,
    out_of_fold_expected,
    pbp_rows,
    trips,
)
from eurohoops.parse.shot_table import Game, add_geometry, game_shots
from tests.conftest import FIXTURES

HALF = json.loads((FIXTURES / "pbp_E2024_5_first_half.json").read_text(encoding="utf-8"))
MAD_MUN = Game("E2024_5", 2024, 5, "MAD", "MUN", neutral=False)


def test_and_ones_are_tied_to_their_made_shot_in_a_real_game() -> None:
    """E2024_5 first half: two and-ones (MUN 07:22, MAD 07:11 in Q2), eleven other trips."""
    found = trips(pbp_rows(HALF))
    and_ones = [t for t in found if t.kind == "and_one"]
    assert [(t.team, t.period, t.clock, t.shot_event) for t in and_ones] == [
        ("MUN", 2, "07:22", 189),
        ("MAD", 2, "07:11", 193),
    ]
    assert len(found) == 13
    shots, _ = game_shots(HALF["points_rows"], MAD_MUN)
    shots = add_geometry(pd.DataFrame(shots)).to_dict("records")
    by_event = {s["event"]: s for s in shots}
    for trip in and_ones:  # NUMBEROFPLAY is the feed's NUM_ANOT: the tied shot is a made 2
        shot = by_event[trip.shot_event or -1]
        assert shot["team"] == trip.team and shot["made"] and shot["value"] == 2
        assert shot["band"] == "rim"  # 1.35 m and 1.40 m from the basket
    # a trip whose free-throw clock is a second after the foul's is still one trip
    assert (found[0].team, found[0].clock, found[0].fta, found[0].ftm) == ("MAD", "08:05", 2, 2)
    pbp_ftm = sum(t.ftm for t in found)
    feed_ftm = sum(r["ID_ACTION"].strip() == "FTM" for r in HALF["points_rows"])
    assert pbp_ftm == feed_ftm


def _pbp(rows: list[tuple[int, str, str, int, str]]) -> dict[str, list[dict[str, object]]]:
    return {
        "FirstQuarter": [
            {
                "NUMBEROFPLAY": n,
                "CODETEAM": team,
                "PLAYTYPE": kind,
                "MINUTE": minute,
                "MARKERTIME": clock,
            }
            for n, team, kind, minute, clock in rows
        ]
    }


def test_trip_rules() -> None:
    rows = [
        (1, "AAA", "2FGM", 1, "09:00"),
        (2, "BBB", "CM", 1, "09:00"),
        (3, "AAA", "FTA", 1, "09:00"),  # and-one (missed)
        (4, "AAA", "2FGM", 1, "08:00"),
        (5, "AAA", "FTM", 1, "08:00"),
        (6, "AAA", "IN", 1, "08:00"),
        (7, "AAA", "FTM", 1, "08:00"),  # two FTs after a make: not an and-one
        (8, "BBB", "2FGM", 1, "07:00"),
        (9, "AAA", "FTM", 1, "07:00"),  # other team's make: not tied
    ]
    found = trips(pbp_rows(_pbp(rows)))
    assert [(t.kind, t.fta, t.ftm, t.shot_event) for t in found] == [
        ("and_one", 1, 0, 1),
        ("other", 2, 2, None),
        ("other", 1, 1, None),
    ]


def _team_games() -> pd.DataFrame:
    rows = []
    for season in (2011, 2012, 2013, 2023):
        for game in range(4):
            row = {c: 0 for c in COUNT_COLUMNS}
            row |= {"season": season, "game_id": f"E{season}_{game}", "team": "AAA"}
            row |= {"fga": 60, "fga_rim": 20, "fga_three": 40, "and_one_trips_rim": 2}
            row |= {"and_one_trips": 2, "and_one_ftm": 1, "other_trips": 8 + season % 10}
            row["other_ftm"] = 12 + season % 10
            row["trips"] = row["and_one_trips"] + row["other_trips"]
            row["ftm"] = row["and_one_ftm"] + row["other_ftm"]
            rows.append(row)
    return pd.DataFrame(rows)


def test_rates_are_ratios_of_totals_and_reproduce_in_sample() -> None:
    tg = _team_games()
    rates = fit_rates(tg)
    assert rates["and_one_trips_per_fga"]["rim"] == 0.1
    assert rates["and_one_trips_per_fga"]["three"] == 0.0
    assert abs(expected_ft_points(tg, rates).sum() - tg["ftm"].sum()) < 1e-9


def test_out_of_fold_expected_never_uses_the_scored_season() -> None:
    tg = _team_games()
    base = out_of_fold_expected(tg, (2011, 2012, 2013), (2023,))
    edited = tg.copy()
    in_2012 = edited["season"] == 2012
    edited.loc[in_2012, ["other_trips", "other_ftm", "trips", "ftm"]] *= 3
    moved = out_of_fold_expected(edited, (2011, 2012, 2013), (2023,))
    assert (moved[in_2012] == base[in_2012]).all()  # 2012's expectation ignores 2012
    assert (moved[~in_2012] != base[~in_2012]).all()  # guard: the edit moves the others
    report = ft_report(tg, (2011, 2012, 2013), (2023,))
    assert report["seasons"]["2023"]["split"] == "validation"
    assert report["seasons"]["2012"]["tolerance"] == 0.1


def test_build_counts_trips_per_team_game(tmp_path: Path) -> None:
    """The real first half as a game: and-ones land in the tied shot's band."""
    shots, _ = game_shots(HALF["points_rows"], MAD_MUN)
    frame = add_geometry(pd.DataFrame(shots)).assign(
        season=2024, game_id="E2024_5", validated_season=True, competition="euroleague"
    )
    excluded = pd.DataFrame(columns=["season", "game_id", "team"])
    write_atomic(
        tmp_path / "playbyplay" / "E2024" / "5.json.gz",
        json.dumps({k: HALF[k] for k in ("FirstQuarter", "SecondQuarter")}).encode(),
    )
    table = build_ft_team_games(tmp_path, frame, excluded)
    mun = table[table["team"] == "MUN"].iloc[0]
    assert (mun["and_one_trips"], mun["and_one_trips_rim"]) == (1, 1)
    assert mun["trips"] == mun["and_one_trips"] + mun["other_trips"]
    assert table["fga"].sum() == len(frame)
