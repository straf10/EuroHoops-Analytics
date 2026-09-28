import gzip
import json
from pathlib import Path

import pandas as pd
import pytest
import typer

from eurohoops import research
from eurohoops.marts import read_table, write_tables
from eurohoops.parse.gbl_pbp import parse_export
from eurohoops.parse.gbl_stints import (
    CHECKS,
    H_I_THRESHOLD,
    POSSESSION_TOLERANCE,
    build_gbl_stints_mart,
    check_game,
    game_stints,
    mart_report,
)
from eurohoops.parse.stints_mart import digest
from tests.conftest import FIXTURES
from tests.test_gbl_pbp import Row, export, starters

# --- The real 2018-19 fixture (M3 H0): Kolossos 68-83 Promitheas, cached export ------------------

FIXTURE_META = json.loads((FIXTURES / "gbl_stints" / "game_17144DAC.json").read_text())


def _real_events() -> pd.DataFrame:
    xlsx = gzip.decompress((FIXTURES / "gbl_stints" / "17144DAC.xlsx.gz").read_bytes())
    g = FIXTURE_META["game"]
    return parse_export(xlsx, g["game_id"], g["home_score"], g["away_score"])


def _real_team_poss() -> dict[str, float]:
    return {row["team"]: row["poss_raw"] for row in FIXTURE_META["team_games"]}


def test_real_fixture_all_three_checks_pass() -> None:
    """68-83, poss_raw 69.76/71.14 (reports/week9-12_progress.md, D4's fixture game)."""
    g = FIXTURE_META["game"]
    result = game_stints(
        g["game_id"],
        g["season"],
        _real_events(),
        g["home"],
        g["away"],
        home_score=g["home_score"],
        away_score=g["away_score"],
        team_poss=_real_team_poss(),
    )
    assert result.reasons == {c: [] for c in CHECKS}
    home_points = sum(s.points[0] for s in result.stints)
    away_points = sum(s.points[1] for s in result.stints)
    assert (home_points, away_points) == (68, 83)
    home_poss = sum(s.possessions[0] for s in result.stints)
    away_poss = sum(s.possessions[1] for s in result.stints)
    assert abs(home_poss - 69.76) <= POSSESSION_TOLERANCE
    assert abs(away_poss - 71.14) <= POSSESSION_TOLERANCE
    # Every positive-length stint has exactly 5 players per side.
    for stint in result.stints:
        if stint.end > stint.start:
            assert len(stint.players[0]) == 5
            assert len(stint.players[1]) == 5


def _real_games_and_team_games() -> tuple[pd.DataFrame, pd.DataFrame]:
    g = FIXTURE_META["game"]
    games = pd.DataFrame(
        [
            {
                "game_id": g["game_id"],
                "season": g["season"],
                "home": g["home"],
                "away": g["away"],
                "home_score": g["home_score"],
                "away_score": g["away_score"],
                "played": g["played"],
                "forfeit": g["forfeit"],
            }
        ]
    )
    team_games = pd.DataFrame(FIXTURE_META["team_games"])
    return games, team_games


def test_two_builds_of_the_real_fixture_are_byte_identical() -> None:
    games, team_games = _real_games_and_team_games()
    pbp = _real_events()
    first = build_gbl_stints_mart(pbp, games, team_games)
    second = build_gbl_stints_mart(pbp, games, team_games)
    assert digest(first.stints) == digest(second.stints)
    assert digest(first.checks) == digest(second.checks)
    report_1 = mart_report(first)
    report_2 = mart_report(second)
    assert report_1 == report_2
    assert report_1["games"] == 1
    assert report_1["overall_pass_rate"] == 1.0
    assert report_1["h_i_rule_holds"] is True
    assert report_1["seasons_with_play_by_play"] == [2018]


# --- Synthetic exports for the three planted defects ---------------------------------------
#
# Each game is minimal: 10 starters at 00:00, "Start of game"/"Start of quarter 1", then a
# handful of period-1 rows. No further "Start of quarter" markers are logged, so every row
# stays tagged period 1 (``gbl_pbp.parse_export`` only advances ``period`` on a row reading
# "Start of quarter N"); periods 2-4 close as empty, lineup-carried stints, same as any GBL
# game whose PBP export ends early. All clocks stay inside period 1's 600 s, so this is safe.

HOME, AWAY = "HOM", "AWY"


def _home_five() -> list[Row]:
    return starters(0, range(1, 6))


def _away_five() -> list[Row]:
    return starters(1, range(11, 16))


START: list[Row] = [
    ("00:00", None, None, None, "Start of game"),
    ("00:00", None, None, None, "Start of quarter 1"),
]


def _events(rows: list[Row], home_score: int, away_score: int, game_id: str) -> pd.DataFrame:
    return parse_export(export(rows), game_id, home_score, away_score)


# five_on_court: a substitution pair at 00:20; dropping the "left the court" row leaves #6 on
# court alongside #1, who was never removed -> 6 home players for the rest of the quarter.
FIVE_SUB: list[Row] = [
    ("00:10", "2-0", "(1) Player 1 performed a 2 points lay-up", None, None),
    ("00:20", None, "(1) Player 1 left the court", None, None),
    ("00:20", None, "(6) Player 6 entered the court", None, None),
]


def _five_rows(dropped: bool) -> list[Row]:
    rows = [*_home_five(), *_away_five(), *START, FIVE_SUB[0]]
    rows += [FIVE_SUB[2]] if dropped else [FIVE_SUB[1], FIVE_SUB[2]]
    return rows


def test_dropped_substitution_row_fails_five_on_court_only() -> None:
    ok = game_stints(
        "G_FIVE_OK",
        2018,
        _events(_five_rows(False), 2, 0, "G_FIVE_OK"),
        HOME,
        AWAY,
        home_score=2,
        away_score=0,
        team_poss={HOME: 1.0, AWAY: 0.0},
    )
    assert ok.reasons == {c: [] for c in CHECKS}

    bad = game_stints(
        "G_FIVE_BAD",
        2018,
        _events(_five_rows(True), 2, 0, "G_FIVE_BAD"),
        HOME,
        AWAY,
        home_score=2,
        away_score=0,
        team_poss={HOME: 1.0, AWAY: 0.0},
    )
    assert bad.reasons["five_on_court"] and "6" in bad.reasons["five_on_court"][0]
    assert bad.reasons["points"] == []
    assert bad.reasons["possessions"] == []


# points: a made 3 by away, logged with its correct running score ("2-3", so the export's own
# score-reconciliation check in ``gbl_pbp.parse_export`` still passes -- exactly as a game that
# is missing one play's *attribution* but not its scoreboard update would look). The "dropped"
# variant garbles the play's text so ``classify`` files it as ``other`` instead of ``fg_made``:
# the score cell survives, but nothing credits the make to a stint.
POINTS_HOME_MAKE: Row = ("00:10", "2-0", "(1) Player 1 performed a 2 points lay-up", None, None)
POINTS_AWAY_MAKE: Row = (
    "00:30",
    "2-3",
    None,
    "(11) Player 11 performed a 3 points jump shot",
    None,
)
POINTS_AWAY_GARBLED: Row = ("00:30", "2-3", None, "(11) Player 11 logged with a data gap", None)


def _points_rows(dropped: bool) -> list[Row]:
    away_row = POINTS_AWAY_GARBLED if dropped else POINTS_AWAY_MAKE
    return [*_home_five(), *_away_five(), *START, POINTS_HOME_MAKE, away_row]


def test_dropped_scoring_row_fails_points_only() -> None:
    ok = game_stints(
        "G_PTS_OK",
        2018,
        _events(_points_rows(False), 2, 3, "G_PTS_OK"),
        HOME,
        AWAY,
        home_score=2,
        away_score=3,
        team_poss={HOME: 1.0, AWAY: 1.0},
    )
    assert ok.reasons == {c: [] for c in CHECKS}

    bad = game_stints(
        "G_PTS_BAD",
        2018,
        _events(_points_rows(True), 2, 3, "G_PTS_BAD"),
        HOME,
        AWAY,
        home_score=2,
        away_score=3,
        team_poss={HOME: 1.0, AWAY: 1.0},
    )
    assert bad.reasons["points"] == [f"{AWAY}: 0 from stints, 3 final"]
    assert bad.reasons["five_on_court"] == []
    # The garbled row also stops crediting away's possession end, but by only 1: inside +-2.
    assert bad.reasons["possessions"] == []


# possessions (plural rows, per the task): one made home basket anchors the export's own score
# check, then four away turnovers that score nothing; dropping three of the turnovers shifts
# away's counted possessions by 3, past the +-2 tolerance.
POSS_HOME_MAKE: Row = ("00:05", "2-0", "(1) Player 1 performed a 2 points lay-up", None, None)
POSS_ROWS: list[Row] = [
    ("00:10", None, None, "(11) Player 11 made a bad pass", None),
    ("00:20", None, None, "(12) Player 12 made a bad pass", None),
    ("00:30", None, None, "(13) Player 13 made a bad pass", None),
    ("00:40", None, None, "(14) Player 14 made a bad pass", None),
]


def _poss_rows(n_dropped: int) -> list[Row]:
    keep = len(POSS_ROWS) - n_dropped
    return [*_home_five(), *_away_five(), *START, POSS_HOME_MAKE, *POSS_ROWS[:keep]]


def test_dropped_possession_ending_rows_fail_possessions_only() -> None:
    ok = game_stints(
        "G_POSS_OK",
        2018,
        _events(_poss_rows(0), 2, 0, "G_POSS_OK"),
        HOME,
        AWAY,
        home_score=2,
        away_score=0,
        team_poss={HOME: 1.0, AWAY: 4.0},
    )
    assert ok.reasons == {c: [] for c in CHECKS}

    bad = game_stints(
        "G_POSS_BAD",
        2018,
        _events(_poss_rows(3), 2, 0, "G_POSS_BAD"),
        HOME,
        AWAY,
        home_score=2,
        away_score=0,
        team_poss={HOME: 1.0, AWAY: 4.0},
    )
    assert bad.reasons["possessions"] == [f"{AWAY}: 1 counted, 4.00 poss_raw"]
    assert bad.reasons["five_on_court"] == []
    assert bad.reasons["points"] == []


def test_a_game_without_team_games_rows_fails_possessions_with_a_reason() -> None:
    result = game_stints(
        "G_NO_TG",
        2018,
        _events(_poss_rows(0), 2, 0, "G_NO_TG"),
        HOME,
        AWAY,
        home_score=2,
        away_score=0,
        team_poss=None,
    )
    assert result.reasons["possessions"] == ["no team_games rows for this game"]


def test_a_team_missing_from_team_games_fails_possessions_with_its_own_reason() -> None:
    """Distinct from having no ``team_games`` rows at all: only one side's row is missing."""
    result = game_stints(
        "G_HALF_TG",
        2018,
        _events(_poss_rows(0), 2, 0, "G_HALF_TG"),
        HOME,
        AWAY,
        home_score=2,
        away_score=0,
        team_poss={HOME: 1.0},
    )
    assert result.reasons["possessions"] == [f"{AWAY}: not in team_games"]
    assert result.reasons["five_on_court"] == []
    assert result.reasons["points"] == []


def test_check_game_directly_on_a_planted_defect() -> None:
    """``check_game`` (not just ``game_stints``) sees a five_on_court failure directly."""
    result = game_stints(
        "G_FIVE_BAD2",
        2018,
        _events(_five_rows(True), 2, 0, "G_FIVE_BAD2"),
        HOME,
        AWAY,
        home_score=2,
        away_score=0,
        team_poss={HOME: 1.0, AWAY: 0.0},
    )
    reasons = check_game(
        result.stints, HOME, AWAY, home_score=2, away_score=0, team_poss={HOME: 1.0, AWAY: 0.0}
    )
    assert reasons == result.reasons


# --- The mart over all six synthetic games + the real fixture --------------------------------


def _synthetic_games_table() -> pd.DataFrame:
    rows = [
        ("G_FIVE_OK", 2, 0),
        ("G_FIVE_BAD", 2, 0),
        ("G_PTS_OK", 2, 3),
        ("G_PTS_BAD", 2, 3),
        ("G_POSS_OK", 2, 0),
        ("G_POSS_BAD", 2, 0),
    ]
    return pd.DataFrame(
        [
            {
                "game_id": gid,
                "season": 2018,
                "home": HOME,
                "away": AWAY,
                "home_score": hs,
                "away_score": as_,
                "played": True,
                "forfeit": False,
            }
            for gid, hs, as_ in rows
        ]
    )


def _synthetic_team_games_table() -> pd.DataFrame:
    poss = {
        "G_FIVE_OK": (1.0, 0.0),
        "G_FIVE_BAD": (1.0, 0.0),
        "G_PTS_OK": (1.0, 1.0),
        "G_PTS_BAD": (1.0, 1.0),
        "G_POSS_OK": (1.0, 4.0),
        "G_POSS_BAD": (1.0, 4.0),
    }
    rows = []
    for game_id, (home_poss, away_poss) in poss.items():
        rows.append({"game_id": game_id, "team": HOME, "poss_raw": home_poss})
        rows.append({"game_id": game_id, "team": AWAY, "poss_raw": away_poss})
    return pd.DataFrame(rows)


def _synthetic_pbp_table() -> pd.DataFrame:
    frames = [
        _events(_five_rows(False), 2, 0, "G_FIVE_OK"),
        _events(_five_rows(True), 2, 0, "G_FIVE_BAD"),
        _events(_points_rows(False), 2, 3, "G_PTS_OK"),
        _events(_points_rows(True), 2, 3, "G_PTS_BAD"),
        _events(_poss_rows(0), 2, 0, "G_POSS_OK"),
        _events(_poss_rows(3), 2, 0, "G_POSS_BAD"),
    ]
    return pd.concat(frames, ignore_index=True)


def test_mart_report_over_three_baselines_and_three_defects() -> None:
    mart = build_gbl_stints_mart(
        _synthetic_pbp_table(), _synthetic_games_table(), _synthetic_team_games_table()
    )
    checks = mart.checks.set_index("game_id")
    assert bool(checks.loc["G_FIVE_OK", "passed"])
    assert not bool(checks.loc["G_FIVE_BAD", "passed"])
    assert not bool(checks.loc["G_FIVE_BAD", "five_on_court"])
    assert bool(checks.loc["G_FIVE_BAD", "points"])
    assert bool(checks.loc["G_FIVE_BAD", "possessions"])

    assert bool(checks.loc["G_PTS_OK", "passed"])
    assert not bool(checks.loc["G_PTS_BAD", "passed"])
    assert not bool(checks.loc["G_PTS_BAD", "points"])

    assert bool(checks.loc["G_POSS_OK", "passed"])
    assert not bool(checks.loc["G_POSS_BAD", "passed"])
    assert not bool(checks.loc["G_POSS_BAD", "possessions"])

    report = mart_report(mart)
    assert report["games"] == 6
    assert report["overall_pass_rate"] == 0.5
    assert report["h_i_threshold"] == H_I_THRESHOLD
    assert report["h_i_rule_holds"] is False  # 0.5 < 0.95
    assert report["seasons"]["2018"]["failures_by_check"] == {
        "five_on_court": 1,
        "points": 1,
        "possessions": 1,
    }

    # Two builds of the same inputs are byte-identical (the report has no timestamps).
    again = build_gbl_stints_mart(
        _synthetic_pbp_table(), _synthetic_games_table(), _synthetic_team_games_table()
    )
    assert digest(mart.stints) == digest(again.stints)
    assert digest(mart.checks) == digest(again.checks)
    assert mart_report(again) == report


def test_a_game_without_a_cached_export_is_left_out_of_the_mart() -> None:
    games = pd.DataFrame(
        [
            {
                "game_id": "G_NO_PBP",
                "season": 2018,
                "home": HOME,
                "away": AWAY,
                "home_score": 1,
                "away_score": 0,
                "played": True,
                "forfeit": False,
            }
        ]
    )
    empty_pbp = pd.DataFrame(columns=list(_synthetic_pbp_table().columns))
    mart = build_gbl_stints_mart(empty_pbp, games, pd.DataFrame())
    assert mart.stints.empty
    assert mart.checks.empty
    report = mart_report(mart)
    assert report["games"] == 0
    assert report["overall_pass_rate"] is None
    assert report["h_i_rule_holds"] is False
    assert report["seasons_with_play_by_play"] == []


# --- research.gbl_stints(): the command function, on a temporary DuckDB from fixtures --------


def _games_schema_row() -> pd.DataFrame:
    g = FIXTURE_META["game"]
    return pd.DataFrame(
        [
            {
                "game_id": g["game_id"],
                "season": g["season"],
                "game_code": g["game_code"],
                "phase": g["phase"],
                "round": g["round"],
                "round_label": str(g["round"]),
                "tipoff_utc": pd.Timestamp(g["tipoff_utc"], tz="UTC"),
                "home": g["home"],
                "away": g["away"],
                "home_score": g["home_score"],
                "away_score": g["away_score"],
                "played": g["played"],
                "forfeit": g["forfeit"],
                "neutral": g["neutral"],
                "confirmed_date": True,
                "competition": "gbl",
            }
        ]
    )


def _team_games_schema_rows() -> pd.DataFrame:
    rows = [dict(row, competition="gbl") for row in FIXTURE_META["team_games"]]
    return pd.DataFrame(rows)


def test_gbl_stints_command_builds_the_mart_and_writes_the_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    mart_path = tmp_path / "eurohoops.duckdb"
    pbp_path = tmp_path / "gbl_pbp.parquet"
    report_path = tmp_path / "gbl_stints.json"

    write_tables(mart_path, {"games": _games_schema_row(), "team_games": _team_games_schema_rows()})
    _real_events().to_parquet(pbp_path, engine="pyarrow", index=False)

    monkeypatch.setattr(research, "MART_PATH", mart_path)
    monkeypatch.setattr(research, "GBL_PBP", pbp_path)
    monkeypatch.setattr(research, "GBL_STINTS_REPORT", report_path)

    research.gbl_stints()

    out = capsys.readouterr().out
    assert "2018" in out
    assert str(report_path) in out

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["games"] == 1
    assert report["overall_pass_rate"] == 1.0
    assert report["h_i_rule_holds"] is True

    stints = read_table(mart_path, "gbl_stints")
    checks = read_table(mart_path, "gbl_stint_game_checks")
    assert stints is not None and not stints.empty
    assert checks is not None and checks["passed"].all()

    # A second run overwrites the marts with byte-identical tables and report.
    research.gbl_stints()
    assert json.loads(report_path.read_text(encoding="utf-8")) == report
    stints_2 = read_table(mart_path, "gbl_stints")
    checks_2 = read_table(mart_path, "gbl_stint_game_checks")
    assert digest(stints) == digest(stints_2)
    assert digest(checks) == digest(checks_2)


def test_gbl_stints_command_without_a_staged_pbp_exits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(research, "GBL_PBP", tmp_path / "missing.parquet")
    with pytest.raises(typer.Exit):
        research.gbl_stints()


def test_gbl_stints_command_without_team_games_exits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mart_path = tmp_path / "eurohoops.duckdb"
    pbp_path = tmp_path / "gbl_pbp.parquet"
    write_tables(mart_path, {"games": _games_schema_row()})
    _real_events().to_parquet(pbp_path, engine="pyarrow", index=False)
    monkeypatch.setattr(research, "MART_PATH", mart_path)
    monkeypatch.setattr(research, "GBL_PBP", pbp_path)
    with pytest.raises(typer.Exit):
        research.gbl_stints()
