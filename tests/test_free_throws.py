"""F2: free-throw trips from play-by-play, and-ones tied to shots, out-of-fold expected points."""

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from eurohoops.ingest.cache import write_atomic
from eurohoops.parse.free_throws import (
    BAND_FLAG_RATE,
    COUNT_COLUMNS,
    allowed_flags,
    build_ft_team_games,
    expected_and_one_by_band,
    expected_ft_points,
    fit_rates,
    ft_report,
    out_of_fold_expected,
    pbp_rows,
    share_checks,
    share_gate,
    trips,
)
from eurohoops.parse.shot_table import BANDS, Game, add_geometry, game_shots
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


def test_the_report_gates_on_shares_and_keeps_the_level_check_ungated() -> None:
    seasons = [
        synthetic_season(seed, games=250).assign(
            season=season, game_id=lambda f, s=season: f["game_id"].str.replace("2015", str(s))
        )
        for seed, season in enumerate((2011, 2012, 2013, 2023))
    ]
    report = ft_report(pd.concat(seasons, ignore_index=True), (2011, 2012, 2013), (2023,))
    assert report["seasons"]["2023"]["split"] == "validation"
    assert report["seasons"]["2012"]["tolerance"] == 0.1
    assert report["level_check"] == report["seasons"]["2012"]["level_check"]
    assert report["level_check"] == "season level, not gated"
    assert report["share_check"]["resamples"] == 1000 and report["share_check"]["seed"] == 20261001
    gate = report["share_check"]
    assert gate["band_checks"] == 4 * len(BANDS)
    assert gate["band_flags"] == sum(s["shares"]["band_flags"] for s in report["seasons"].values())
    assert gate["band_flags_allowed"] == allowed_flags(4 * len(BANDS))
    assert report["share_checks_pass"] == gate["passed"]
    assert report["share_checks_pass"]  # rates drawn from one model: shares reconcile


def test_the_count_rule_is_the_binomial_95th_percentile() -> None:
    assert pytest.approx(0.0455, abs=1e-4) == BAND_FLAG_RATE
    assert allowed_flags(78) == 7  # 13 held-out seasons x 6 bands
    assert allowed_flags(0) == 0
    rate = BAND_FLAG_RATE
    cdf = [
        sum(math.comb(78, j) * rate**j * (1 - rate) ** (78 - j) for j in range(k + 1))
        for k in (6, 7)
    ]
    assert cdf[0] < 0.95 <= cdf[1]
    one = {"shares": {"bands": dict.fromkeys(BANDS), "band_flags": 0, "teams_pass": True}}
    seasons = {str(s): one for s in range(13)}
    assert share_gate(seasons)["passed"]
    flagged = {**seasons, "0": {"shares": {**one["shares"], "band_flags": 8}}}
    assert not share_gate(flagged)["passed"]  # 8 flags > 7 allowed
    team_miss = {**seasons, "0": {"shares": {**one["shares"], "teams_pass": False}}}
    assert not share_gate(team_miss)["passed"]


RATES = {
    "and_one_trips_per_fga": dict(zip(BANDS, (0.06, 0.04, 0.02, 0.01, 0.003, 0.001), strict=True)),
    "points_per_and_one_trip": 0.75,
    "other_trips_per_fga": 0.13,
    "points_per_other_trip": 1.5,
}


def synthetic_season(seed: int = 7, teams: int = 18, games: int = 400) -> pd.DataFrame:
    """One season of team-games drawn from ``RATES`` (the model is right); teams differ in shot
    volume (+-10%) and band mix, so their expected shares differ."""
    rng = np.random.default_rng(seed)
    volume = np.linspace(0.9, 1.1, teams)
    mix = rng.dirichlet(np.full(len(BANDS), 8.0), teams) * 0.5 + 0.5 / len(BANDS)
    rows = []
    for g in range(games):
        for t in rng.choice(teams, 2, replace=False):
            fga = rng.poisson(60 * volume[t] * mix[t])
            trips = rng.poisson(fga * np.array(list(RATES["and_one_trips_per_fga"].values())))
            ftm_band = rng.binomial(trips, RATES["points_per_and_one_trip"])
            other = rng.poisson(fga.sum() * RATES["other_trips_per_fga"])
            other_ftm = rng.binomial(2 * other, RATES["points_per_other_trip"] / 2)
            row = {c: 0 for c in COUNT_COLUMNS} | {
                "season": 2015,
                "game_id": f"E2015_{g}",
                "team": f"T{t:02d}",
                "fga": int(fga.sum()),
                "other_trips": int(other),
                "other_ftm": int(other_ftm),
                "and_one_trips": int(trips.sum()),
                "and_one_ftm": int(ftm_band.sum()),
                "ftm": int(ftm_band.sum() + other_ftm),
                "trips": int(trips.sum() + other),
            }
            for i, b in enumerate(BANDS):
                row |= {f"fga_{b}": fga[i], f"and_one_trips_{b}": trips[i]}
                row |= {f"and_one_ftm_{b}": ftm_band[i]}
            rows.append(row)
    return pd.DataFrame(rows).astype({c: "float64" for c in COUNT_COLUMNS})


def _checks(frame: pd.DataFrame) -> dict[str, Any]:
    return share_checks(
        frame, expected_and_one_by_band(frame, RATES), expected_ft_points(frame, RATES)
    )


def _team_rates(frame: pd.DataFrame, column: pd.Series) -> pd.Series:
    by_team = column.groupby(frame["team"]).sum() / frame.groupby("team").size()
    return by_team / by_team.sum()


def _plant_team_shares(frame: pd.DataFrame, shift: pd.Series) -> pd.DataFrame:
    """Scale each team's FT points so its actual share = expected share + shift."""
    actual = _team_rates(frame, frame["ftm"])
    expected = _team_rates(frame, expected_ft_points(frame, RATES))
    target = (expected + shift) / (expected + shift).sum()
    factor = frame["team"].map(target / actual)
    return frame.assign(ftm=frame["ftm"] * factor)


def test_share_checks_pass_when_the_model_is_right() -> None:
    result = _checks(synthetic_season())
    assert result["band_flags"] == 0 and result["teams_pass"], result
    assert set(result["bands"]) == set(BANDS)
    for band in result["bands"].values():
        assert band["tolerance"] == pytest.approx(2 * band["bootstrap_se"], abs=2e-6)


def test_a_band_share_error_of_three_se_fails_the_band_check() -> None:
    frame = synthetic_season()
    clean = _checks(frame)["bands"]["rim"]
    columns = [f"and_one_ftm_{b}" for b in BANDS]
    rim = frame["and_one_ftm_rim"].sum()
    rest = frame[columns].sum().sum() - rim
    target = clean["expected_share"] + 3 * clean["bootstrap_se"]
    planted = frame.assign(
        and_one_ftm_rim=frame["and_one_ftm_rim"] * target * rest / (1 - target) / rim
    )
    result = _checks(planted)
    assert result["bands"]["rim"]["gap"] == pytest.approx(3 * clean["bootstrap_se"], abs=1e-5)
    assert not result["bands"]["rim"]["within_tolerance"] and result["band_flags"] >= 1


def test_team_share_errors_of_three_se_fail_the_mean_gap_check() -> None:
    frame = synthetic_season()
    clean = _checks(frame)["teams"]
    names = sorted(set(frame["team"]))
    sign = pd.Series([1.0 if i % 2 else -1.0 for i in range(len(names))], index=names)
    planted = _plant_team_shares(frame, sign * 3 * clean["gap_se_rms"])
    result = _checks(planted)["teams"]
    assert result["mean_abs_share_gap"] > 2.5 * clean["gap_se_rms"]
    assert not result["mean_abs_within_tolerance"]


def test_team_share_errors_of_three_se_against_the_signal_fail_the_r_check() -> None:
    frame = synthetic_season()
    clean = _checks(frame)["teams"]
    assert clean["r_within_tolerance"]
    expected = _team_rates(frame, expected_ft_points(frame, RATES))
    against = -np.sign(expected - expected.mean()) * 3 * clean["gap_se_rms"]
    result = _checks(_plant_team_shares(frame, against))["teams"]
    assert not result["r_within_tolerance"]


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
