import gzip
import io
import logging
from pathlib import Path

import httpx
import openpyxl
import pandas as pd
import pytest

from eurohoops.ingest.gbl import (
    ExportIdError,
    _Session,
    export_id,
    signed_game_id,
)
from eurohoops.parse.box import FILL_SEASONS, build_box_tables
from eurohoops.parse.esake import BoxLine, parse_box_score
from eurohoops.parse.gbl_pbp import (
    PbpFormatError,
    PbpScoreError,
    build_pbp_table,
    classify,
    parse_export,
    player_lines,
)
from tests.conftest import esake_fixture
from tests.test_gbl_ingest import fetcher_for

Row = tuple[str, str | None, str | None, str | None, str | None]
HEADER = ("Time", "Score", "HOME TEAM", "AWAY TEAM", "Game Actions")


def export(rows: list[Row], header: tuple[str, ...] = HEADER) -> bytes:
    book = openpyxl.Workbook()
    sheet = book.active
    assert sheet is not None
    sheet.append(header)
    for row in rows:
        sheet.append(row)
    buffer = io.BytesIO()
    book.save(buffer)
    return buffer.getvalue()


def starters(column: int, numbers: range) -> list[Row]:
    rows: list[Row] = []
    for n in numbers:
        cells: list[str | None] = [None, None]
        cells[column] = f"({n}) Player {n} entered the court"
        rows.append(("00:00", None, cells[0], cells[1], None))
    return rows


# First column scores 2 + 3 + 1 = 6, second column 2; #1 is subbed for #6 at 05:00; the last
# player event is at 39:30 but the game ends at 40:00.
GAME: list[Row] = [
    *starters(0, range(1, 6)),
    *starters(1, range(11, 16)),
    ("00:00", None, None, None, "Start of game"),
    ("00:00", None, None, None, "Start of quarter 1"),
    ("01:00", "2-0", "(1) Player 1 performed a 2 points lay-up", None, None),
    ("02:00", None, None, "(11) Player 11 missed a 3 points jump shot", None),
    ("03:00", None, None, "(12) Player 12 blocked while attempting a 2 points dunk", None),
    ("03:00", None, "(2)  Player 2 blocked a shot", None, None),
    ("05:00", None, "(1) Player 1 left the court", None, None),
    ("05:00", None, "(6) Player 6 entered the court", None, None),
    ("06:00", "5-0", "(6) Player 6 performed a 3 points jump shot", None, None),
    ("07:00", "5-2", None, "(11) Player 11 performed a 2 points jump shot", None),
    ("08:00", None, "(3) Player 3 missed a free throw (1 of 2)", None, None),
    ("08:00", "6-2", "(3) Player 3 made a free throw (2 of 2)", None, None),
    ("08:30", None, None, "(13) Player 13 perfomed a steal", None),
    ("10:00", None, None, None, "End of quarter 1"),
    ("10:00", None, None, None, "Start of quarter 2"),
    ("39:30", None, "(4) Player 4 made a defensive rebound", None, None),
    ("40:00", None, None, None, "End of quarter 4"),
    ("40:00", None, None, None, "End of game"),
]


@pytest.mark.parametrize(
    ("sentence", "expected"),
    [
        ("(24) Frank BARTLEY made a free throw (2 of 2)", ("24", "Frank BARTLEY", "ft_made", 1)),
        ("(9) A B missed a free throw (1 of 3)", ("9", "A B", "ft_missed", 1)),
        (
            "(1) Chevez GOODWIN performed a 2 points jump shot",
            ("1", "Chevez GOODWIN", "fg_made", 2),
        ),
        ("(5) X Y performed a 3 points jump shot", ("5", "X Y", "fg_made", 3)),
        ("(5) X Y missed a 2 points lay-up", ("5", "X Y", "fg_missed", 2)),
        ("(20) D L blocked while attempting a 3 points jump shot", ("20", "D L", "fg_missed", 3)),
        ("(9) Andreas PETROPOULOS entered the court", ("9", "Andreas PETROPOULOS", "in", 0)),
        ("(9) Andreas PETROPOULOS left the court", ("9", "Andreas PETROPOULOS", "out", 0)),
        ("(30) Chris SILVA perfomed a steal", ("30", "Chris SILVA", "other", 0)),
        ("(1) Danny Agbelese blocked a shot", ("1", "Danny Agbelese", "other", 0)),
        ("(7) E. J. Singler passed the ball out of bounds", ("7", "E. J. Singler", "other", 0)),
        ("Defensive rebound", ("", "", "other", 0)),
    ],
)
def test_classify(sentence: str, expected: tuple[str, str, str, int]) -> None:
    assert classify(sentence) == expected


@pytest.mark.parametrize(("home", "away", "first"), [(6, 2, "home"), (2, 6, "away")])
def test_columns_map_to_home_and_away_by_the_final_score(home: int, away: int, first: str) -> None:
    events = parse_export(export(GAME), "G1", home, away)
    scorer = events[events["text"] == "(6) Player 6 performed a 3 points jump shot"].iloc[0]
    assert scorer["side"] == first
    assert (scorer["home_score"], scorer["away_score"]) == ((5, 0) if first == "home" else (0, 5))
    assert events["period"].tolist()[-1] == 2  # periods count the "Start of ..." rows
    assert (
        events.loc[events["text"].str.contains("Player 2 blocked"), "player"].item() == "Player 2"
    )


def test_a_final_score_that_matches_neither_order_is_refused() -> None:
    with pytest.raises(PbpScoreError, match=r"sheet final \(6, 2\), result 6-3"):
        parse_export(export(GAME), "G1", 6, 3)


def test_an_unexpected_sheet_layout_is_refused() -> None:
    with pytest.raises(PbpFormatError):
        parse_export(export(GAME, ("When", "Score", "A", "B", "Game Actions")), "G1", 6, 2)


def test_player_lines_count_shots_and_minutes_to_the_end_of_game() -> None:
    lines = player_lines(parse_export(export(GAME), "G1", 6, 2)).set_index("number")
    assert lines.loc["1", ["points", "fg2m", "fg2a", "seconds"]].tolist() == [2, 1, 1, 300]
    assert lines.loc["6", ["points", "fg3m", "fg3a", "seconds"]].tolist() == [3, 1, 1, 2100]
    assert lines.loc["3", ["points", "ftm", "fta"]].tolist() == [1, 1, 2]
    assert lines.loc["12", ["fg2m", "fg2a"]].tolist() == [0, 1]
    # #4 plays the whole game: the "End of game" row (40:00), not the last event (39:30).
    assert lines.loc["4", "seconds"] == 2400
    assert lines.groupby("side")["points"].sum().to_dict() == {"away": 2, "home": 6}
    assert lines.groupby("side")["seconds"].sum().to_dict() == {"away": 12000, "home": 12000}


def cache(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(payload, mtime=0))


def games_frame(rows: list[tuple[str, int, int, int, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [r[0] for r in rows],
            "season": [r[1] for r in rows],
            "game_code": [r[2] for r in rows],
            "home": ["HOM"] * len(rows),
            "away": ["AWY"] * len(rows),
            "home_score": [r[3] for r in rows],
            "away_score": [r[4] for r in rows],
            "played": True,
            "forfeit": False,
        }
    )


def test_pbp_table_skips_exports_that_do_not_match(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    cache(tmp_path / "pbp" / "2018" / "0000000A.xlsx.gz", export(GAME))
    cache(tmp_path / "pbp" / "2018" / "0000000B.xlsx.gz", export(GAME))
    games = games_frame([("G_A", 2018, 10, 6, 2), ("G_B", 2018, 11, 9, 9), ("G_C", 2018, 12, 1, 0)])
    with caplog.at_level(logging.WARNING):
        table = build_pbp_table(tmp_path, games)
    assert set(table["game_id"]) == {"G_A"}
    assert "G_B" in caplog.text
    assert build_pbp_table(tmp_path, games.iloc[2:]).empty


# --- The 2018-20 box fill -----------------------------------------------------------------


def events_for(lines: list[BoxLine], column: int, numbers: list[int]) -> list[Row]:
    """Shot rows that reproduce ``lines`` (one jersey number each) in the given team column."""
    rows: list[Row] = []
    for line, number in zip(lines, numbers, strict=True):
        who = f"({number}) Player {number}"
        sentences = (
            [f"{who} performed a 2 points jump shot"] * line.fg2m
            + [f"{who} missed a 2 points jump shot"] * (line.fg2a - line.fg2m)
            + [f"{who} performed a 3 points jump shot"] * line.fg3m
            + [f"{who} missed a 3 points jump shot"] * (line.fg3a - line.fg3m)
            + [f"{who} made a free throw (1 of 1)"] * line.ftm
            + [f"{who} missed a free throw (1 of 1)"] * (line.fta - line.ftm)
        )
        for sentence in sentences:
            cells: list[str | None] = [None, None]
            cells[column] = sentence
            rows.append(("10:00", None, cells[0], cells[1], None))
    return rows


def with_final(rows: list[Row], home: int, away: int) -> list[Row]:
    return [*rows, ("40:00", f"{home}-{away}", None, None, "End of game")]


BOX = parse_box_score(esake_fixture("box_F26689D1.html"))  # PAOK 81-64, 12 players each
EXTRA = BoxLine("X", points=7, fg2m=2, fg2a=3, fg3m=1, fg3a=2, ftm=0, fta=0, seconds=600)


def write_box_game(raw: Path, season: int, code: int, box: bool) -> None:
    if box:
        (raw / "boxscore" / str(season)).mkdir(parents=True, exist_ok=True)
        page = esake_fixture("box_F26689D1.html").encode()
        cache(raw / "boxscore" / str(season) / f"{code:08X}.html.gz", page)


def test_a_game_without_an_esake_box_gets_pbp_team_totals(tmp_path: Path) -> None:
    assert BOX is not None
    home, away = list(BOX[0].players), list(BOX[1].players)
    rows = events_for(home, 0, list(range(len(home)))) + events_for(
        away, 1, list(range(50, 50 + len(away)))
    )
    cache(tmp_path / "pbp" / "2018" / "0000000A.xlsx.gz", export(with_final(rows, 81, 64)))
    tables = build_box_tables(tmp_path, games_frame([("G_A", 2018, 10, 81, 64)]))
    assert tables.team_box.to_dict("records") == [
        {"game_id": "G_A", "team": "HOM", "total_points": 81, "source": "pbp"},
        {"game_id": "G_A", "team": "AWY", "total_points": 64, "source": "pbp"},
    ]
    assert tables.player_box.empty  # team totals only (decision 2026-09-25)
    assert tables.fill["fill"].tolist() == ["team_totals_from_pbp"] * 2


def test_a_short_esake_box_gets_the_missing_players_pbp_line(tmp_path: Path) -> None:
    assert BOX is not None
    home, away = list(BOX[0].players), list(BOX[1].players)
    rows = events_for([*home, EXTRA], 0, list(range(len(home) + 1)))
    rows += events_for(away, 1, list(range(50, 50 + len(away))))
    write_box_game(tmp_path, 2019, 11, box=True)
    cache(tmp_path / "pbp" / "2019" / "0000000B.xlsx.gz", export(with_final(rows, 88, 64)))
    tables = build_box_tables(tmp_path, games_frame([("G_B", 2019, 11, 88, 64)]))
    pbp = tables.player_box[tables.player_box["source"] == "pbp"]
    assert pbp[["team", "player_id", "points", "fg3m"]].to_dict("records") == [
        {"team": "HOM", "player_id": f"pbp:{len(home)}:Player {len(home)}", "points": 7, "fg3m": 1}
    ]
    assert set(tables.team_box["source"]) == {"esake"}  # ESAKE's own totals row stays as it was
    [log] = tables.fill.to_dict("records")
    assert (log["team"], log["fill"]) == ("HOM", "missing_player_from_pbp")
    assert log["detail"] == f"#{len(home)} Player {len(home)}: 7 points"


def test_an_ambiguous_gap_is_not_filled(tmp_path: Path) -> None:
    assert BOX is not None
    home, away = list(BOX[0].players), list(BOX[1].players)
    two = [EXTRA, BoxLine("Y", 3, 0, 0, 1, 1, 0, 0, 100)]
    rows = events_for([*home, *two], 0, list(range(len(home) + 2)))
    rows += events_for(away, 1, list(range(50, 50 + len(away))))
    write_box_game(tmp_path, 2019, 12, box=True)
    cache(tmp_path / "pbp" / "2019" / "0000000C.xlsx.gz", export(with_final(rows, 91, 64)))
    tables = build_box_tables(tmp_path, games_frame([("G_C", 2019, 12, 91, 64)]))
    assert set(tables.player_box["source"]) == {"esake"}
    [log] = tables.fill.to_dict("records")
    assert log["fill"] == "not_filled"
    assert log["detail"].startswith("2 unmatched PBP lines")


def test_fill_reasons_without_pbp_or_outside_the_scope(tmp_path: Path) -> None:
    assert 2022 not in FILL_SEASONS
    write_box_game(tmp_path, 2022, 13, box=True)
    games = games_frame([("G_D", 2018, 14, 70, 60), ("G_E", 2022, 13, 90, 64)])
    fill = build_box_tables(tmp_path, games).fill.set_index(["game_id", "team"])["detail"]
    assert fill[("G_D", "HOM")].startswith("no cached PBP export")
    assert fill[("G_E", "HOM")].startswith("outside the 2018-19/2019-20 fill scope")
    assert ("G_E", "AWY") not in fill.index  # 64 = 64: not short


def test_a_pbp_that_does_not_add_up_is_not_used(tmp_path: Path) -> None:
    rows = with_final([("01:00", "2-0", "(1) P performed a 2 points jump shot", None, None)], 5, 0)
    cache(tmp_path / "pbp" / "2018" / "0000000F.xlsx.gz", export(rows))
    fill = build_box_tables(tmp_path, games_frame([("G_F", 2018, 15, 5, 0)])).fill
    assert fill.loc[fill["team"] == "HOM", "detail"].item() == "PBP events sum to 2, result 5"


# --- Fetching -------------------------------------------------------------------------------


def test_signed_game_id_matches_basket_hotel() -> None:
    assert signed_game_id("D6867DA7") == -695829081  # docs/spikes/gbl-pbp.md
    assert signed_game_id("06C28DF5") == 113413621


def test_export_id_from_the_widget_response() -> None:
    widget = b'url += \\"&game_id=\\" + 4453393;\\r\\n'
    assert export_id(widget) == 4453393
    assert export_id(b'url += "&game_id=" + 6084033;') == 6084033
    with pytest.raises(ExportIdError):
        export_id(b"MBT.API.update('x', '<div>no export</div>')")


class FakeBasketHotel:
    def __init__(self) -> None:
        self.calls: list[httpx.URL] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request.url)
        if request.url.path.endswith("/show"):
            assert request.url.params["request[0][param][game_id]"] == "113413621"
            assert request.url.params["request[0][param][show_export_link]"] == "1"
            return httpx.Response(200, text='url += "&game_id=" + 4453393;')
        assert request.url.params["game_id"] == "4453393"
        return httpx.Response(200, content=export(GAME))


@pytest.mark.usefixtures("no_sleep")
def test_pbp_takes_two_requests_once(tmp_path: Path) -> None:
    api = FakeBasketHotel()
    session = _Session(fetcher_for(api), tmp_path, live_season=2026)  # type: ignore[arg-type]
    session.pbp(2018, "06C28DF5")
    session.pbp(2018, "06C28DF5")
    assert len(api.calls) == 2
    assert (tmp_path / "pbp_widget" / "2018" / "06C28DF5.js.gz").exists()
    xlsx = gzip.decompress((tmp_path / "pbp" / "2018" / "06C28DF5.xlsx.gz").read_bytes())
    assert len(parse_export(xlsx, "G", 6, 2)) == len(GAME)
    # A cached widget response spares its request when only the export is missing.
    (tmp_path / "pbp" / "2018" / "06C28DF5.xlsx.gz").unlink()
    session.pbp(2018, "06C28DF5")
    assert len(api.calls) == 3
    assert api.calls[-1].path.endswith("play_by_play")
