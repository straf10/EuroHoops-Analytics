"""E1: the ``team_games`` table (both competitions) and its validation report."""

import gzip
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from eurohoops.parse.gbl_pbp import parse_export
from eurohoops.parse.possession_report import (
    coverage,
    euroleague_sample,
    gbl_pbp_vs_esake,
    points_mismatches,
    possession_report,
)
from eurohoops.parse.team_box import build_team_games
from tests.conftest import FIXTURES, esake_fixture
from tests.test_gbl_pbp import GAME, export

BOX = json.loads((FIXTURES / "box_E2023_1.json").read_text(encoding="utf-8"))  # RED 94-73 ASV


def cache(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(payload, mtime=0))


def frame(
    rows: list[tuple[str, int, int, str, str, int, int]], neutral: bool = False
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [r[0] for r in rows],
            "season": [r[1] for r in rows],
            "game_code": [r[2] for r in rows],
            "home": [r[3] for r in rows],
            "away": [r[4] for r in rows],
            "home_score": [r[5] for r in rows],
            "away_score": [r[6] for r in rows],
            "played": True,
            "forfeit": False,
            "neutral": neutral,
        }
    )


@pytest.fixture
def raw(tmp_path: Path) -> tuple[Path, Path]:
    el, gbl = tmp_path / "euroleague", tmp_path / "gbl"
    cache(el / "boxscore" / "E2023" / "1.json.gz", json.dumps(BOX).encode())
    empty = {"ByQuarter": [], "Stats": [{"PlayersStats": []}, {"PlayersStats": []}]}
    cache(el / "boxscore" / "E2023" / "2.json.gz", json.dumps(empty).encode())
    cache(el / "boxscore" / "E2023" / "3.json.gz", json.dumps(BOX).encode())
    cache(el / "boxscore" / "E2023" / "4.json.gz", json.dumps(BOX).encode())
    cache(
        gbl / "boxscore" / "2024" / "8FC479F6.html.gz", esake_fixture("box_8FC479F6.html").encode()
    )
    cache(
        gbl / "boxscore" / "2018" / "0000000A.html.gz", esake_fixture("box_F26689D1.html").encode()
    )
    return el, gbl


EL_GAMES = [
    ("E2023_1", 2023, 1, "RED", "ASV", 94, 73),
    ("E2023_2", 2023, 2, "RED", "ASV", 80, 70),  # API placeholder box
    ("E2023_3", 2023, 3, "MAD", "ASV", 94, 73),  # box of other teams
    ("E2023_4", 2023, 4, "RED", "ASV", 95, 73),  # box points differ from the result
    ("E2023_5", 2023, 5, "RED", "ASV", 90, 80),  # not cached
]
GBL_GAMES = [
    ("GBL2024_8FC479F6", 2024, 0x8FC479F6, "HOM", "AWY", 93, 101),
    ("GBL2018_0000000A", 2018, 10, "HOM", "AWY", 6, 2),  # ESAKE totals 81-64: PBP instead
    ("GBL2018_0000000B", 2018, 11, "HOM", "AWY", 70, 60),  # nothing cached
]


def build(raw: tuple[Path, Path], pbp: pd.DataFrame | None = None) -> Any:
    if pbp is None:
        pbp = parse_export(export(GAME), "GBL2018_0000000A", 6, 2)
    return build_team_games(frame(EL_GAMES), frame(GBL_GAMES), raw, pbp)


def test_team_games_rows_and_missing_reasons(raw: tuple[Path, Path]) -> None:
    built = build(raw)
    table = built.table.set_index(["game_id", "team"])
    red = table.loc[("E2023_1", "RED")]
    assert (red["opponent"], bool(red["home"]), red["source"]) == ("ASV", True, "euroleague_box")
    assert red["poss_raw"] == pytest.approx(69.46)
    assert red["poss_game"] == pytest.approx((69.46 + 69.82) / 2)
    assert not table.loc[("E2023_1", "ASV"), "home"]
    esake = table.loc[("GBL2024_8FC479F6", "HOM")]
    assert (esake["minutes"], esake["source"], esake["poss_raw"]) == (45.0, "esake_box", 85.34)
    pbp = table.loc[("GBL2018_0000000A", "HOM")]
    assert (pbp["source"], pbp["points"], pbp["fga"], pbp["fta"]) == ("gbl_pbp", 6, 2, 2)
    assert table.loc[("GBL2018_0000000A", "AWY"), "fga"] == 3
    assert pbp["dreb"] == 1  # "(4) Player 4 made a defensive rebound"
    reasons = dict(zip(built.missing["game_id"], built.missing["reason"], strict=True))
    assert reasons == {
        "E2023_2": "box score has no players (API placeholder)",
        "E2023_3": "box score teams ['ASV', 'RED'] are not the game's ['MAD', 'ASV']",
        "E2023_4": "box points 94-73 are not the result",
        "E2023_5": "no cached box score (ingest --details)",
        "GBL2018_0000000B": "no cached ESAKE box score; no GBL play-by-play",
    }
    assert len(built.table) == 2 * 3


def test_gbl_pbp_that_does_not_reproduce_the_result_is_not_used(raw: tuple[Path, Path]) -> None:
    pbp = parse_export(export(GAME), "GBL2018_0000000A", 6, 2).assign(side="home")
    built = build(raw, pbp)
    reasons = dict(zip(built.missing["game_id"], built.missing["reason"], strict=True))
    assert reasons["GBL2018_0000000A"] == (
        "ESAKE totals 81-64 are not the result; play-by-play without both teams"
    )
    wrong = parse_export(export(GAME), "GBL2018_0000000A", 6, 2)
    wrong.loc[wrong["action"] == "ft_made", "value"] = 0
    reasons = dict(
        zip(build(raw, wrong).missing["game_id"], build(raw, wrong).missing["reason"], strict=True)
    )
    assert reasons["GBL2018_0000000A"].endswith("play-by-play points 5-2 are not the result")


def test_a_page_without_stat_tables_is_a_missing_game(tmp_path: Path) -> None:
    gbl = tmp_path / "gbl"
    cache(gbl / "boxscore" / "2018" / "0000000A.html.gz", b"<div>header only</div>")
    built = build_team_games(
        frame([]), frame(GBL_GAMES[1:2]), (tmp_path / "el", gbl), pd.DataFrame()
    )
    assert built.missing["reason"].tolist() == ["ESAKE page has no box score; no GBL play-by-play"]


def test_neutral_games_have_no_home_team(raw: tuple[Path, Path]) -> None:
    built = build_team_games(frame(EL_GAMES[:1], neutral=True), frame([]), raw, None)
    assert not built.table["home"].any()


def test_coverage_and_points_checks(raw: tuple[Path, Path]) -> None:
    built = build(raw)
    games = pd.concat(
        [
            frame(EL_GAMES).assign(competition="euroleague"),
            frame(GBL_GAMES).assign(competition="gbl"),
        ]
    )
    cov = coverage(built.table, built.missing, games)
    assert cov["euroleague"]["2023"] == {"rated_games": 5, "with_rows": 1, "missing": 4}
    assert cov["gbl"]["2018"] == {"rated_games": 2, "with_rows": 1, "missing": 1}
    assert points_mismatches(built.table, games) == []
    moved = built.table.assign(points=built.table["points"] + (built.table["game_id"] == "E2023_1"))
    assert points_mismatches(moved, games) == ["E2023_1"]


def pbp_json(rows: list[tuple[str, str, str]]) -> dict[str, Any]:
    return {
        "FirstQuarter": [
            {"PLAYTYPE": k, "CODETEAM": t, "PLAYER_ID": "", "MARKERTIME": c, "MINUTE": 1}
            for k, t, c in rows
        ]
    }


def test_euroleague_sample_compares_box_and_pbp(tmp_path: Path) -> None:
    raw = tmp_path / "el"
    cache(raw / "boxscore" / "E2023" / "1.json.gz", json.dumps(BOX).encode())
    pbp = pbp_json([("2FGM", "RED", "09:00")] * 70 + [("2FGM", "ASV", "08:00")] * 66)
    cache(raw / "playbyplay" / "E2023" / "1.json.gz", json.dumps(pbp).encode())
    empty = {"ByQuarter": [], "Stats": [{"PlayersStats": []}, {"PlayersStats": []}]}
    cache(raw / "boxscore" / "E2024" / "7.json.gz", json.dumps(empty).encode())
    cache(raw / "playbyplay" / "E2024" / "7.json.gz", json.dumps(pbp_json([])).encode())
    report = euroleague_sample(raw)
    assert report["games"] == 1
    assert report["excluded"] == {"E2024_7": "box score has no players (API placeholder)"}
    assert [(r["team"], r["box"], r["pbp"]) for r in report["per_team"]] == [
        ("RED", 69.46, 70),
        ("ASV", 69.82, 66),
    ]
    assert report["within_tolerance_share_of_teams"] == 0.5
    assert report["within_tolerance_share_of_games"] == 0.0
    assert report["mean_gap_pbp_minus_box"] == pytest.approx((0.54 - 3.82) / 2, abs=1e-4)


def test_gbl_comparison_and_full_report(raw: tuple[Path, Path]) -> None:
    pbp = parse_export(export(GAME), "GBL2018_0000000A", 6, 2)
    gbl = frame(GBL_GAMES)
    compared = gbl_pbp_vs_esake(raw[1], gbl, pbp)
    assert compared["games"] == 1
    assert compared["mean_gap_pbp_minus_esake"]["points"] == pytest.approx((6 - 81 + 2 - 64) / 2)
    built = build(raw)
    games = pd.concat(
        [frame(EL_GAMES).assign(competition="euroleague"), gbl.assign(competition="gbl")]
    )
    report = possession_report(
        built.table,
        missing=built.missing,
        games=games,
        euroleague_raw=raw[0],
        gbl_raw=raw[1],
        gbl_pbp=None,
    )
    assert report["euroleague_pbp_sample"] is None  # no EuroLeague PBP cached
    assert report["gbl_pbp_vs_esake"] is None
    assert len(report["missing_games"]) == 5
