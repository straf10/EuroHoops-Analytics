from datetime import datetime

import pandas as pd
import pytest

from eurohoops.ingest.gbl import RoundPage
from eurohoops.parse.esake import (
    EsakeParseError,
    GblGame,
    ResultsPage,
    month_number,
    parse_box_score,
    parse_results_page,
)
from eurohoops.parse.games import build_gbl_tables, build_gbl_team_seasons
from tests.conftest import esake_fixture


def game_block(idgame: str, when: str, score: str, home: str = "0000000A") -> str:
    """The minimal markup of one ESAKE game block."""
    return (
        '<div class="esake-program-game"><h5>01η Αγωνιστική</h5>'
        f'<div class="esake-program-game-info"><img src="/skn/clock.svg">{when}</div>'
        f'<a href="/el/action/EsakegameView?idgame={idgame}&mode=3">stats</a>'
        '<div class="esake-program-game-final-score">'
        f'<div><span>HOME &amp; CO</span><img src="/dat/esaketeam/{home}/logo.png"></div>'
        f"<div><span>{score}</span></div>"
        '<div><img src="/dat/esaketeam/00000010/logo.png"><span>ΑΕΚ</span></div></div></div>'
    )


def header(month: str, year: int) -> str:
    return f'<div class="esake-program-series-title">{month} <span>{year}</span></div>'


def one_round(season: int, *games: GblGame, code: str = "01") -> RoundPage:
    return RoundPage(season, "00000001", code, ResultsPage(round_codes=(), games=games))


def test_played_round_parses_teams_scores_and_round_list() -> None:
    page = parse_results_page(esake_fixture("results_2025_rs_24.html"))
    assert len(page.games) == 6
    assert page.round_codes == tuple(f"{n:02d}" for n in range(1, 27))
    first = page.games[0]
    assert (first.idgame, first.home, first.away) == ("A005D853", "0000000D", "0000000A")
    assert (first.home_score, first.away_score, first.forfeit) == (87, 83, False)
    assert first.round_label == "24η Αγωνιστική"
    assert first.tipoff_local == datetime(2026, 4, 4, 16, 0)


def test_new_format_playoffs_expose_round_codes() -> None:
    page = parse_results_page(esake_fixture("results_2025_po_01.html"))
    assert page.round_codes[:3] == ("201", "202", "203")
    assert page.games[0].round_label == "QF1"


def test_unplayed_and_unconfirmed_games() -> None:
    unplayed = parse_results_page(esake_fixture("results_2026_rs_01.html")).games
    assert all(g.home_score is None and g.away_score is None for g in unplayed)
    assert all(g.confirmed_date for g in unplayed)
    late_round = parse_results_page(esake_fixture("results_2026_rs_26.html")).games
    unconfirmed = [g for g in late_round if not g.confirmed_date]
    assert unconfirmed
    assert unconfirmed[0].tipoff_local.hour == 0


def test_forfeit_is_played_20_0_and_flagged() -> None:
    games = parse_results_page(esake_fixture("results_2018_po_01.html")).games
    forfeit = next(g for g in games if g.forfeit)
    assert (forfeit.home, forfeit.away) == ("00000001", "00000002")
    assert (forfeit.home_score, forfeit.away_score) == (20, 0)


def test_empty_phase_has_no_games() -> None:
    page = parse_results_page(esake_fixture("results_2019_po_01.html"))
    assert page == ResultsPage(round_codes=(), games=())


def test_year_comes_from_the_month_header_across_new_year() -> None:
    html = (
        header("Δεκέμβριος", 2025)
        + game_block("00000001", "Σαβ 27 Δεκ - 17:00", "80&nbsp;-&nbsp;70")
        + header("Ιανουάριος", 2026)
        + game_block("00000002", "Σαβ 3 Ιαν - 17:00", "&nbsp;-&nbsp;")
    )
    games = parse_results_page(html).games
    assert [g.tipoff_local for g in games] == [datetime(2025, 12, 27, 17), datetime(2026, 1, 3, 17)]
    assert games[0].home_name == "HOME & CO"


@pytest.mark.parametrize(
    ("html", "match"),
    [
        (game_block("00000001", "Σαβ 27 Δεκ - 17:00", "80 - 70"), "before any month header"),
        (header("Ιανουάριος", 2026) + game_block("1", "Σαβ 3 Ιαν", "-"), "incomplete"),
        (header("Ιανουάριος", 2026) + game_block("00000001", "Σαβ 3 Ιαν", "7 - x"), "score"),
        (header("Ιανουάριος", 2026) + game_block("00000001", "Σαβ 3 Δεκ", "-"), "header month"),
    ],
)
def test_unexpected_markup_fails_loudly(html: str, match: str) -> None:
    with pytest.raises(EsakeParseError, match=match):
        parse_results_page(html)


@pytest.mark.parametrize(
    ("name", "month"), [("Οκτ", 10), ("Μάϊος", 5), ("Μαϊ", 5), ("Ιουν", 6), ("Ιουλ", 7)]
)
def test_month_names_and_abbreviations(name: str, month: int) -> None:
    assert month_number(name) == month


def test_unknown_month_is_rejected() -> None:
    with pytest.raises(EsakeParseError):
        month_number("Ιου")  # ambiguous: June or July


@pytest.mark.parametrize(
    ("local", "utc"),
    [
        (datetime(2026, 3, 28, 16, 0), "2026-03-28T14:00:00Z"),  # EET, the day before DST
        (datetime(2026, 4, 4, 16, 0), "2026-04-04T13:00:00Z"),  # EEST
        (datetime(2026, 10, 24, 17, 0), "2026-10-24T14:00:00Z"),  # EEST, the day before DST ends
        (datetime(2026, 10, 31, 17, 0), "2026-10-31T15:00:00Z"),  # EET
    ],
)
def test_athens_tipoffs_convert_to_utc_across_dst(local: datetime, utc: str) -> None:
    game = GblGame("0000ABCD", "01", local, True, "A", "B", "A", "B", None, None, False)
    games, _ = build_gbl_tables([one_round(2026, game)])
    assert games.loc[0, "tipoff_utc"] == pd.Timestamp(utc)


def test_gbl_tables_from_real_pages() -> None:
    rounds = [
        RoundPage(
            2018, "00000002", "01", parse_results_page(esake_fixture("results_2018_po_01.html"))
        ),
        RoundPage(
            2018, "00000002", "12", parse_results_page(esake_fixture("results_2018_po_12.html"))
        ),
    ]
    games, teams = build_gbl_tables(rounds)
    assert len(games) == 5
    assert set(games["phase"]) == {"PO"}
    assert games["forfeit"].sum() == 1
    assert games["played"].all()
    assert games.loc[games["forfeit"], "game_id"].item() == "GBL2018_C9EC888D"
    assert teams.set_index("team").loc["0000000C", "name"] == "ΠΑΟΚ"
    team_seasons = build_gbl_team_seasons(rounds)
    assert set(team_seasons["season"]) == {2018}
    assert sorted(team_seasons["team"]) == sorted(teams["team"])


def test_box_score_reconciles_with_totals() -> None:
    boxes = parse_box_score(esake_fixture("box_F26689D1.html"))
    assert boxes is not None
    home, away = boxes
    assert (home.total_points, away.total_points) == (81, 64)
    assert sum(p.points for p in home.players) == 81
    assert sum(p.seconds for p in home.players) == 200 * 60
    first = home.players[0]
    assert (first.player_id, first.fg2m, first.fg2a, first.seconds) == ("000006FC", 2, 5, 1330)


def test_box_score_needs_two_teams() -> None:
    with pytest.raises(EsakeParseError, match="2 team box scores"):
        parse_box_score(
            "<table><tr><th>ΠΑΙΚΤΗΣ</th></tr><tr><td>ΣΥΝΟΛΟ</td><td>80</td></tr></table>"
        )


def test_game_page_without_stat_tables_has_no_box_score() -> None:
    assert parse_box_score('<div class="mvp-player">header only</div>') is None
