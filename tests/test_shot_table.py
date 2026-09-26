"""F1: the shot table, its exclusions and the feed facts of docs/data/shots.md on real games."""

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from eurohoops.ingest.cache import write_atomic
from eurohoops.parse.games import conform
from eurohoops.parse.shot_table import (
    REASONS,
    SHOT_VALUE,
    Game,
    box_shooting,
    build_shot_table,
    game_shots,
    label_contradicts_geometry,
    period_of,
    reconcile,
    seconds_left,
    shot_band,
    shot_report,
)
from eurohoops.parse.stints_mart import digest
from tests.conftest import FIXTURES, REPO

REAL = json.loads((FIXTURES / "shots_real_games.json").read_text(encoding="utf-8"))
E2024_1 = json.loads((FIXTURES / "points_E2024_1.json").read_text(encoding="utf-8"))["Rows"]
BER_PAN = Game("E2024_1", 2024, 1, "BER", "PAN", neutral=False)


def _game(block: dict[str, Any]) -> Game:
    g = block["game"]
    return Game(g["game_id"], g["season"], g["game_code"], g["home"], g["away"], g["neutral"])


def test_points_a_b_are_the_home_and_away_score_after_the_row() -> None:
    """§3: POINTS_A is the schedule home team's score (BER hosted E2024_1), after the action."""
    run = {"BER": 0, "PAN": 0}
    for row in E2024_1:
        run[row["TEAM"].strip()] += row["POINTS"]
        assert (row["POINTS_A"], row["POINTS_B"]) == (run["BER"], run["PAN"])
    assert run == {"BER": 77, "PAN": 87}  # the final score


def test_margin_before_is_the_score_before_the_shot_from_the_shooters_side() -> None:
    shots, excluded = game_shots(E2024_1, BER_PAN)
    by_event = {s["event"]: s for s in shots}
    assert by_event[55]["margin_before"] == 0  # PAN's first basket, 0-0 before it
    assert by_event[62]["margin_before"] == 2  # PAN 2-0 up before Nunn's basket
    assert by_event[63]["margin_before"] == -4  # BER 0-4 down before Olinde's basket
    assert by_event[63]["home"] and not by_event[62]["home"]
    assert not excluded
    assert len(shots) == sum(r["ID_ACTION"].strip() != "FTM" for r in E2024_1)


def test_overtime_minutes_and_console_give_period_and_seconds_left() -> None:
    """§3: MINUTE counts on through overtime (41-45, 46-50); CONSOLE is the time left."""
    rows = REAL["overtime"]["rows"]
    assert max(r["MINUTE"] for r in rows) > 45  # a double-overtime game
    for row in rows:
        period = period_of(row["MINUTE"])
        left = seconds_left(row["CONSOLE"], period)
        length = 600 if period <= 4 else 300
        start = (period - 1) * 600 if period <= 4 else 2400 + (period - 5) * 300
        elapsed_minute = (start + length - left) / 60
        assert row["MINUTE"] - 1 <= elapsed_minute <= row["MINUTE"]
    assert [period_of(m) for m in (1, 10, 11, 40, 41, 45, 46, 50, 51)] == [
        1,
        1,
        2,
        4,
        5,
        5,
        6,
        6,
        7,
    ]


def test_buzzer_console_minus_one_second_reads_as_zero() -> None:
    assert seconds_left("00:-1", 4) == 0
    assert seconds_left("05:00", 5) == 300
    assert seconds_left("10:00", 1) == 600


def test_2011_codes_layups_dunks_and_blocked_attempts() -> None:
    """§3: 2011-12 logs layups/dunks with their own codes and blocked attempts as xFGAB."""
    block = REAL["codes_2011"]
    shots, excluded = game_shots(block["rows"], _game(block))
    assert not excluded
    raw = {r["NUM_ANOT"]: r["ID_ACTION"].strip() for r in block["rows"]}
    seen = {raw[s["event"]]: (s["value"], s["made"]) for s in shots}
    assert seen["LAYUPMD"] == (2, True)
    assert seen["DUNK"] == (2, True)
    assert seen["LAYUPATT"] == (2, False)
    assert seen["2FGAB"] == (2, False)
    assert seen["3FGAB"] == (3, False)


def test_box_gap_e2017_14_is_one_player_missing_from_the_box_score() -> None:
    """The one 2011+ reconciliation mismatch: the box score lacks a KHI player the feed has."""
    block = REAL["box_gap"]
    shots, _ = game_shots(block["rows"], _game(block))
    box = box_shooting(block["box"])
    assert not isinstance(box, str)
    in_box = {p["Player_ID"].strip() for s in block["box"]["Stats"] for p in s["PlayersStats"]}
    for team in ("KHI", "ZAL"):
        own = [s for s in shots if s["team"] == team]
        fga, points = len(own), sum(s["value"] * s["made"] for s in own)
        missing = [s for s in own if s["shooter"] not in in_box]
        gap = (len(missing), sum(s["value"] * s["made"] for s in missing))
        assert (fga - box[team][0], points - box[team][1]) == gap
    assert box["KHI"] != (67, 72) and gap == (0, 0)  # ZAL (last) matches; KHI does not
    feed_final = {r["TEAM"].strip(): 0 for r in block["rows"]}
    for r in block["rows"]:
        feed_final[r["TEAM"].strip()] += r["POINTS"]
    assert feed_final == {"KHI": block["game"]["home_score"], "ZAL": block["game"]["away_score"]}


def _row(action: str, points: int, x: int, y: int, team: str = "BER") -> dict[str, Any]:
    return {
        "NUM_ANOT": 1,
        "TEAM": team.ljust(10),
        "ID_PLAYER": "P1",
        "ID_ACTION": action,
        "POINTS": points,
        "COORD_X": x,
        "COORD_Y": y,
        "ZONE": "C",
        "FASTBREAK": "0",
        "SECOND_CHANCE": "1",
        "POINTS_OFF_TURNOVER": "0",
        "MINUTE": 3,
        "CONSOLE": "07:30",
        "POINTS_A": points,
        "POINTS_B": 0,
    }


@pytest.mark.parametrize(
    ("row", "reason"),
    [
        (_row("2FGA", 0, 0, 0), "zero_coordinates"),
        (_row("2FGA", 0, -1, -1), "unparseable"),
        (_row("2FGA", 0, 100, 100, team="XXX"), "unparseable"),
        (_row("2FGM", 0, 100, 100), "unparseable"),
        (_row("3FGA", 3, 700, 100), "unparseable"),
        (_row("ZZZ", 0, 100, 100), "unparseable"),
    ],
)
def test_unusable_rows_are_excluded_with_a_reason(row: dict[str, Any], reason: str) -> None:
    shots, excluded = game_shots([row], BER_PAN)
    assert not shots
    assert [e["reason"] for e in excluded] == [reason]
    assert reason in REASONS


def test_free_throws_are_not_shots_and_flags_parse() -> None:
    shots, excluded = game_shots([_row("FTM", 1, -1, -1), _row("2FGM", 2, 50, 50)], BER_PAN)
    assert not excluded
    assert len(shots) == 1
    assert shots[0]["second_chance"] and not shots[0]["fastbreak"]


def test_label_geometry_contradictions() -> None:
    x = pd.Series([0.0, 0.0, 0.0, 0.0])
    y = pd.Series([6.50, 6.95, 6.65, 7.00])
    value = pd.Series([3, 2, 3, 3])
    # a 3 at 6.50 m (0.25 inside) and a 2 at 6.95 m (0.20 outside) contradict; 6.65 m is within
    assert label_contradicts_geometry(x, y, value).tolist() == [True, True, False, False]


def test_bands() -> None:
    assert [shot_band(d, 2) for d in (0.5, 2.0, 4.0, 6.0)] == ["rim", "short", "mid", "long2"]
    assert [shot_band(d, 3) for d in (6.8, 8.5)] == ["three", "deep3"]


def _write_raw(root: Path) -> pd.DataFrame:
    games = []
    for key in ("box_gap", "overtime", "codes_2011"):
        g = REAL[key]["game"]
        path = root / "points" / f"E{g['season']}" / f"{g['game_code']}.json.gz"
        write_atomic(path, json.dumps({"Rows": REAL[key]["rows"]}).encode())
        games.append(g)
    box = root / "boxscore" / "E2017" / "14.json.gz"
    write_atomic(box, json.dumps(REAL["box_gap"]["box"]).encode())
    frame = pd.DataFrame(
        [
            {
                "game_id": g["game_id"],
                "season": g["season"],
                "game_code": g["game_code"],
                "phase": "RS",
                "round": 1,
                "round_label": "Round 1",
                "tipoff_utc": pd.Timestamp(f"{g['season']}-10-10T18:00:00Z"),
                "home": g["home"],
                "away": g["away"],
                "home_score": g["home_score"],
                "away_score": g["away_score"],
                "played": True,
                "forfeit": False,
                "neutral": g["neutral"],
                "confirmed_date": True,
            }
            for g in games
        ]
        + [
            {
                "game_id": "E2016_9",  # played, no cached feed
                "season": 2016,
                "game_code": 9,
                "phase": "RS",
                "round": 1,
                "round_label": "Round 1",
                "tipoff_utc": pd.Timestamp("2016-10-10T18:00:00Z"),
                "home": "AAA",
                "away": "BBB",
                "home_score": 80,
                "away_score": 70,
                "played": True,
                "forfeit": False,
                "neutral": False,
                "confirmed_date": True,
            }
        ]
    )
    return conform(frame)


def test_build_is_deterministic_and_reconciles_against_the_box(tmp_path: Path) -> None:
    games = _write_raw(tmp_path)
    first = build_shot_table(tmp_path, games)
    second = build_shot_table(tmp_path, games)
    assert digest(first.shots) == digest(second.shots)
    assert digest(first.excluded) == digest(second.excluded)
    assert first.not_cached == ["E2016_9"]
    assert set(first.shots["season"]) == {2011, 2017, 2023}
    assert first.shots["validated_season"].all()
    checks = reconcile(first, tmp_path, games)
    khi = checks[(checks["game_id"] == "E2017_14") & (checks["team"] == "KHI")].iloc[0]
    assert khi["status"] == "mismatch"
    assert (khi["feed_fga"] - khi["box_fga"], khi["feed_fg_points"] - khi["box_fg_points"]) == (
        15,
        18,
    )
    assert set(checks.loc[checks["game_id"] != "E2017_14", "status"]) == {"no_box"}
    report = shot_report(first, checks)
    season = report["seasons"]["2017"]
    assert season["reconciliation"]["mismatches"] == [
        {
            "game_id": "E2017_14",
            "team": "KHI",
            "fga_feed_minus_box": 15,
            "fg_points_feed_minus_box": 18,
        }
    ]
    assert season["actions"]["FTM"] == 39
    assert report["table_sha256"]["shots"] == digest(first.shots)


def test_committed_report_meets_the_f1_thresholds() -> None:
    """F1 Done-when on the full cache: feed = box for >= 99% of 2011+ team-games, exclusions
    <= 1% of FGA in every validated season, and no field-goal code outside the known set."""
    report = json.loads((REPO / "reports/shots.json").read_text(encoding="utf-8"))
    assert report["reconciliation_match_rate_validated"] >= 0.99
    for season, block in report["seasons"].items():
        assert set(block["actions"]) <= {*SHOT_VALUE, "FTM"}, season
        if block["validated"]:
            assert block["excluded_share"] <= 0.01, season
            mismatches = block["reconciliation"]["mismatches"]
            assert all(m["game_id"] == "E2017_14" for m in mismatches), season
