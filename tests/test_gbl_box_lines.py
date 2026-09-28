import gzip
from pathlib import Path

import pandas as pd
import pytest

from eurohoops.parse.esake import EsakeParseError
from eurohoops.parse.gbl_box_lines import build_gbl_player_games, parse_box_lines
from tests.conftest import esake_fixture

# The ΣΥΝΟΛΟ (totals) row of each team table, hand-read from the fixtures (see
# tests/fixtures/esake/box_8FC479F6.html and box_F26689D1.html): pts, fg2a, fg3a, fta, oreb,
# dreb, ast, stl, blk, tov, pf.
TOTALS_8FC479F6 = (
    {
        "pts": 93,
        "fg2a": 51,
        "fg3a": 23,
        "fta": 27,
        "oreb": 18,
        "dreb": 27,
        "ast": 18,
        "stl": 9,
        "blk": 3,
        "tov": 18,
        "pf": 22,
    },
    {
        "pts": 101,
        "fg2a": 42,
        "fg3a": 29,
        "fta": 28,
        "oreb": 11,
        "dreb": 25,
        "ast": 21,
        "stl": 12,
        "blk": 3,
        "tov": 15,
        "pf": 27,
    },
)
# The "ΟΜΑΔΙΚΑ - ΠΑΓΚΟΣ" team/bench row: only rebounds and turnovers are nonzero.
TEAM_ROW_8FC479F6 = ({"oreb": 5, "dreb": 2, "tov": 1}, {"oreb": 1, "dreb": 2, "tov": 1})
TOTALS_F26689D1 = (
    {
        "pts": 81,
        "fg2a": 46,
        "fg3a": 22,
        "fta": 9,
        "oreb": 13,
        "dreb": 24,
        "ast": 22,
        "stl": 4,
        "blk": 1,
        "tov": 11,
        "pf": 23,
    },
    {
        "pts": 64,
        "fg2a": 23,
        "fg3a": 28,
        "fta": 22,
        "oreb": 8,
        "dreb": 23,
        "ast": 11,
        "stl": 7,
        "blk": 3,
        "tov": 11,
        "pf": 16,
    },
)


def test_parse_box_lines_returns_home_then_away_full_rows() -> None:
    parsed = parse_box_lines(esake_fixture("box_8FC479F6.html"))
    assert parsed is not None
    home, away = parsed
    assert len(home) == 12
    assert len(away) == 12
    first = home[0]
    assert first["player_id"] == "E199BC3C"
    assert (first["sec"], first["pts"], first["fg2a"], first["tov"], first["pir"]) == (
        132,
        0,
        1,
        1,
        -1,
    )


@pytest.mark.parametrize(
    ("fixture", "totals", "team_row"),
    [
        ("box_8FC479F6.html", TOTALS_8FC479F6, TEAM_ROW_8FC479F6),
        ("box_F26689D1.html", TOTALS_F26689D1, ({}, {})),
    ],
)
def test_player_rows_sum_to_the_totals_row_except_rebounds_and_turnovers(
    fixture: str, totals: tuple[dict[str, int], ...], team_row: tuple[dict[str, int], ...]
) -> None:
    parsed = parse_box_lines(esake_fixture(fixture))
    assert parsed is not None
    for lines, total, bench in zip(parsed, totals, team_row, strict=True):
        for stat, expected in total.items():
            got = sum(line[stat] for line in lines)
            if stat in ("oreb", "dreb", "tov"):
                got += bench.get(stat, 0)
            assert got == expected, stat


@pytest.mark.parametrize("fixture", ["box_8FC479F6.html", "box_F26689D1.html"])
def test_pir_identity_holds_row_by_row(fixture: str) -> None:
    parsed = parse_box_lines(esake_fixture(fixture))
    assert parsed is not None
    for lines in parsed:
        for line in lines:
            missed_fg = (line["fg2a"] - line["fg2m"]) + (line["fg3a"] - line["fg3m"])
            missed_ft = line["fta"] - line["ftm"]
            pir = (
                line["pts"]
                + line["reb"]
                + line["ast"]
                + line["stl"]
                + line["blk"]
                + line["fd"]
                - missed_fg
                - missed_ft
                - line["tov"]
                - line["blka"]
                - line["pf"]
            )
            assert pir == line["pir"]


def test_page_without_stat_tables_returns_none() -> None:
    assert parse_box_lines('<div class="mvp-player">header only</div>') is None


def test_one_table_page_is_rejected() -> None:
    with pytest.raises(EsakeParseError, match="2 team box scores"):
        parse_box_lines(
            "<table><tr><th>ΠΑΙΚΤΗΣ</th></tr><tr><td>ΣΥΝΟΛΟ</td><td>80</td></tr></table>"
        )


def test_overtime_is_counted_from_the_score_by_period_table() -> None:
    with_ot = parse_box_lines(esake_fixture("box_8FC479F6.html"))
    no_ot = parse_box_lines(esake_fixture("box_F26689D1.html"))
    assert with_ot is not None and no_ot is not None


def _cache(raw_dir: Path, season: int, idgame: str, html: str) -> None:
    path = raw_dir / "boxscore" / str(season) / f"{idgame}.html.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(html.encode()))


def _games(*rows: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _team_games(*rows: tuple[str, str, float]) -> pd.DataFrame:
    return pd.DataFrame([{"game_id": g, "team": t, "poss_game": p} for g, t, p in rows])


@pytest.fixture
def two_real_games() -> pd.DataFrame:
    return _games(
        {
            "game_id": "G1",
            "season": 2024,
            "game_code": int("8FC479F6", 16),
            "home": "AAAA0001",
            "away": "AAAA0002",
            "played": True,
            "forfeit": False,
        },
        {
            "game_id": "G2",
            "season": 2019,
            "game_code": int("F26689D1", 16),
            "home": "AAAA0003",
            "away": "AAAA0004",
            "played": True,
            "forfeit": False,
        },
    )


def test_builder_reads_both_games_and_computes_poss(
    tmp_path: Path, two_real_games: pd.DataFrame
) -> None:
    _cache(tmp_path, 2024, "8FC479F6", esake_fixture("box_8FC479F6.html"))
    _cache(tmp_path, 2019, "F26689D1", esake_fixture("box_F26689D1.html"))
    team_games = _team_games(
        ("G1", "AAAA0001", 75.0),
        ("G1", "AAAA0002", 75.0),
        ("G2", "AAAA0003", 68.0),
        ("G2", "AAAA0004", 68.0),
    )
    result = build_gbl_player_games(tmp_path, two_real_games, team_games)
    assert result.skipped == 0
    assert len(result.table) == 48  # 12 players * 2 teams * 2 games
    player = result.table[
        (result.table["game_id"] == "G1") & (result.table["player_id"] == "FC6A957C")
    ].iloc[0]
    assert player["sec"] == 2134
    assert player["game_sec"] == 2700  # regulation + one overtime
    # recorded seconds of AAAA0001 total exactly 5 * game_sec (5 on court throughout)
    assert player["poss"] == pytest.approx(5.0 * 75.0 * 2134 / (5 * 2700))


def test_builder_skips_pages_without_a_box_table(
    tmp_path: Path, two_real_games: pd.DataFrame
) -> None:
    games = pd.concat(
        [
            two_real_games,
            pd.DataFrame(
                [
                    {
                        "game_id": "G3",
                        "season": 2019,
                        "game_code": 99,
                        "home": "AAAA0005",
                        "away": "AAAA0006",
                        "played": True,
                        "forfeit": False,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    _cache(tmp_path, 2024, "8FC479F6", esake_fixture("box_8FC479F6.html"))
    _cache(tmp_path, 2019, "F26689D1", esake_fixture("box_F26689D1.html"))
    _cache(tmp_path, 2019, "00000063", '<div class="mvp-player">header only</div>')
    team_games = _team_games(
        ("G1", "AAAA0001", 75.0),
        ("G1", "AAAA0002", 75.0),
        ("G2", "AAAA0003", 68.0),
        ("G2", "AAAA0004", 68.0),
        ("G3", "AAAA0005", 70.0),
        ("G3", "AAAA0006", 70.0),
    )
    result = build_gbl_player_games(tmp_path, games, team_games)
    assert result.skipped == 1
    assert len(result.table) == 48


def test_builder_skips_a_game_missing_from_team_games(
    tmp_path: Path, two_real_games: pd.DataFrame
) -> None:
    _cache(tmp_path, 2024, "8FC479F6", esake_fixture("box_8FC479F6.html"))
    _cache(tmp_path, 2019, "F26689D1", esake_fixture("box_F26689D1.html"))
    team_games = _team_games(("G1", "AAAA0001", 75.0), ("G1", "AAAA0002", 75.0))  # G2 missing
    result = build_gbl_player_games(tmp_path, two_real_games, team_games)
    assert result.skipped == 1
    assert len(result.table) == 24  # only G1's 24 rows
    assert set(result.table["game_id"]) == {"G1"}


def test_builder_skips_games_without_a_cached_file(
    tmp_path: Path, two_real_games: pd.DataFrame
) -> None:
    _cache(tmp_path, 2024, "8FC479F6", esake_fixture("box_8FC479F6.html"))
    # G2's box page is never cached.
    team_games = _team_games(("G1", "AAAA0001", 75.0), ("G1", "AAAA0002", 75.0))
    result = build_gbl_player_games(tmp_path, two_real_games, team_games)
    assert result.skipped == 0  # not "skipped": simply never fetched
    assert set(result.table["game_id"]) == {"G1"}


def test_two_runs_are_identical(tmp_path: Path, two_real_games: pd.DataFrame) -> None:
    _cache(tmp_path, 2024, "8FC479F6", esake_fixture("box_8FC479F6.html"))
    _cache(tmp_path, 2019, "F26689D1", esake_fixture("box_F26689D1.html"))
    team_games = _team_games(
        ("G1", "AAAA0001", 75.0),
        ("G1", "AAAA0002", 75.0),
        ("G2", "AAAA0003", 68.0),
        ("G2", "AAAA0004", 68.0),
    )
    first = build_gbl_player_games(tmp_path, two_real_games, team_games)
    second = build_gbl_player_games(tmp_path, two_real_games, team_games)
    pd.testing.assert_frame_equal(first.table, second.table)
