from pathlib import Path

import pandas as pd
import pytest

from eurohoops.config import GBL, GBL_PLAYER_BOX, GBL_TEAM_BOX, MART_PATH, SQL_DIR
from eurohoops.ingest.gbl import ingest_gbl
from eurohoops.marts import box_invariants, build_marts, read_games, read_teams
from eurohoops.parse.box import build_box_tables
from eurohoops.parse.games import build_gbl_tables, write_table
from tests.conftest import REPO, make_games, teams_table, write_pipeline
from tests.test_gbl_ingest import FakeEsake, fetcher_for


@pytest.fixture
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.mark.usefixtures("workdir")
def test_marts_round_trip_both_competitions() -> None:
    euroleague = make_games({2024: True, 2025: False})
    gbl = make_games({2024: True}, gbl_like=True)
    write_pipeline(euroleague, gbl)
    pd.testing.assert_frame_equal(read_games(MART_PATH, "euroleague"), euroleague)
    pd.testing.assert_frame_equal(read_games(MART_PATH, "gbl"), gbl)
    assert read_teams(MART_PATH, "gbl").equals(teams_table(gbl))


@pytest.mark.usefixtures("workdir", "no_sleep")
def test_failing_box_scores_are_flagged_not_dropped(tmp_path: Path) -> None:
    rounds = ingest_gbl(fetcher_for(FakeEsake()), tmp_path / "raw", [2018], True, 2026)
    games, teams = build_gbl_tables(rounds)
    player_box, team_box = build_box_tables(tmp_path / "raw", games)
    # The fake serves one recorded box score (a PAOK 81-64 game) for every game, so points
    # never reconcile; also doctor one player line to trip the shot and points checks.
    doctored = player_box.index[(player_box["game_id"] == "GBL2018_B90F050D")][0]
    player_box.loc[doctored, "fg2m"] = 99
    write_table(games, GBL.staging_games)
    write_table(teams, GBL.staging_teams)
    write_table(player_box, GBL_PLAYER_BOX)
    write_table(team_box, GBL_TEAM_BOX)
    write_pipeline(make_games({2024: True}), games)
    assert build_marts(MART_PATH, REPO / SQL_DIR)

    season = box_invariants(MART_PATH)["seasons"]["2018"]
    assert season["games"] == 4  # the forfeit is not box-checked; nothing is dropped
    assert season["passed"] == 0
    assert season["failed_games"]["GBL2018_B90F050D"] == [
        "points_mismatch",
        "bad_shot_lines",
        "bad_point_lines",
    ]
    assert "points_mismatch" in season["failed_games"]["GBL2018_E0ABEE8A"]
