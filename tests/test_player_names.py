import gzip
from pathlib import Path

import pandas as pd
import pytest

from eurohoops.parse.player_names import (
    PLAYER_NAMES_SCHEMA,
    euroleague_names,
    gbl_names,
    parse_name_cells,
    pbp_links,
)
from tests.conftest import esake_fixture

ROW = (
    '<tr><td class="table-left"><a href="/el/action/EsakeplayerView?idplayer={pid}&amp;mode=1">'
    '{head}<div class="table-image-wrapper"><img src="x.png"></div><span>{surname}</span> {first}'
    "</a></td><td>9</td></tr>"
)
MVP = (
    '<div class="mvp-player-name"><a href="/el/action/EsakeplayerView?idplayer={pid}&amp;mode=3">'
    "<span>{surname}</span><br>{first}</a></div>"
)


def page(*rows: str) -> str:
    return "<html><body><table>" + "".join(rows) + "</table></body></html>"


def test_cells_read_jersey_surname_and_first_name() -> None:
    html = page(
        ROW.format(pid="0000ABCD", head="#17", surname="ΜΟΥΡAΤΟΣ", first="ΒAΣΙΛΗΣ"),
        ROW.format(pid="0000ABCE", head="##", surname="SKORDILIS", first="GAIOS"),
        ROW.format(pid="0000ABCF", head="#0", surname="ΓΟΥΙΛΙAΜΣ", first="- ΓΚΟΣ ΝAΙΤΖΕΛ"),
    )
    cells = parse_name_cells(html + MVP.format(pid="0000ABCD", surname="X", first="Y"))
    assert [(c.player_id, c.jersey, c.surname, c.first) for c in cells] == [
        ("0000ABCD", "17", "ΜΟΥΡAΤΟΣ", "ΒAΣΙΛΗΣ"),  # the Latin A is kept as printed
        ("0000ABCE", None, "SKORDILIS", "GAIOS"),
        ("0000ABCF", None, "ΓΟΥΙΛΙAΜΣ", "- ΓΚΟΣ ΝAΙΤΖΕΛ"),  # #0 is ESAKE's placeholder
    ]


def test_real_fixture_page_has_a_cell_per_player() -> None:
    cells = parse_name_cells(esake_fixture("box_8FC479F6.html"))
    assert len(cells) == 24
    assert len({c.player_id for c in cells}) == 24
    assert all(c.surname for c in cells)


def write_page(root: Path, season: int, idgame: str, html: str) -> None:
    path = root / "boxscore" / str(season) / f"{idgame}.html.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(html.encode("utf-8")))


def test_gbl_names_aggregate_per_team_season_and_name_pbp_keys(tmp_path: Path) -> None:
    write_page(
        tmp_path,
        2018,
        "00000A01",
        page(ROW.format(pid="0000ABCD", head="#7", surname="ΜΟΥΡAΤΟΣ", first="ΒAΣΙΛΗΣ")),
    )
    write_page(
        tmp_path,
        2018,
        "00000A02",
        page(ROW.format(pid="0000ABCD", head="##", surname="ΜΟΥΡAΤΟΣ", first="ΒAΣΙΛΗΣ")),
    )
    box = pd.DataFrame(
        {
            "game_id": ["GBL2018_00000A01", "GBL2018_00000A02", "GBL2018_00000A02"],
            "team": ["00000005", "00000005", "00000005"],
            "player_id": ["0000ABCD", "0000ABCD", "pbp:23:Giorgos Aggelou"],
            "seconds": [600, 0, 800],
        }
    )
    names = gbl_names(tmp_path, box)
    PLAYER_NAMES_SCHEMA.validate(names)
    rows = names.astype(object).where(names.notna(), None).to_dict("records")
    assert rows == [
        {
            "competition": "gbl",
            "source_id": "0000ABCD",
            "season": 2018,
            "team": "00000005",
            "jersey": "7",
            "surname_raw": "ΜΟΥΡAΤΟΣ",
            "first_raw": "ΒAΣΙΛΗΣ",
            "games": 1,
            "minutes": 10.0,
        },
        {
            "competition": "gbl",
            "source_id": "pbp:23:Giorgos Aggelou",
            "season": 2018,
            "team": "00000005",
            "jersey": "23",
            "surname_raw": "AGGELOU",
            "first_raw": "GIORGOS",
            "games": 1,
            "minutes": 13.33,
        },
    ]


def test_gbl_names_refuse_an_official_id_without_a_name(tmp_path: Path) -> None:
    box = pd.DataFrame(
        {"game_id": ["GBL2018_00000A01"], "team": ["T"], "player_id": ["0000ABCD"], "seconds": [1]}
    )
    with pytest.raises(ValueError, match="no name"):
        gbl_names(tmp_path, box)


def test_euroleague_names_strip_padding_and_split_the_name() -> None:
    players = pd.DataFrame(
        {
            "player_id": ["P012099", "P012099", "P004193"],
            "player": ["GROSBER, DORIAN", "GROSBER, DORIAN", "JBAM"],
            "dorsal": ["2", "2", ""],
            "team": ["BER", "BER", "CHL"],
            "season": [2024, 2024, 2012],
            "sec": [600, 300, 120],
        }
    )
    names = euroleague_names(players).astype(object)
    names = names.where(names.notna(), None)
    columns = ["source_id", "jersey", "surname_raw", "first_raw", "games", "minutes"]
    assert names[columns].to_numpy().tolist() == [
        ["P004193", None, "JBAM", "", 1, 2.0],
        ["P012099", "2", "GROSBER", "DORIAN", 2, 15.0],
    ]


def test_pbp_keys_link_only_to_one_same_jersey_same_name_id() -> None:
    def row(source_id: str, jersey: str | None, surname: str, first: str) -> dict[str, object]:
        return {
            "competition": "gbl",
            "source_id": source_id,
            "season": 2018,
            "team": "00000005",
            "jersey": jersey,
            "surname_raw": surname,
            "first_raw": first,
            "games": 1,
            "minutes": 1.0,
        }

    names = pd.DataFrame(
        [
            row("pbp:23:Giorgos Aggelou", "23", "AGGELOU", "GIORGOS"),
            row("pbp:9:Nick Other", "9", "OTHER", "NICK"),
            row("pbp:0:Gabe York", None, "YORK", "GABE"),
            row("0000AAAA", "23", "ΑΓΓΕΛΟΥ", "ΓΙΩΡΓΟΣ"),
            row("0000BBBB", "23", "ΑΛΛΟΣ", "ΚΑΠΟΙΟΣ"),
            row("0000CCCC", "9", "ΑΛΛΟΣ", "ΚΑΠΟΙΟΣ"),
        ]
    )

    def same(a: str, b: str) -> bool:
        return (a, b) == ("AGGELOU GIORGOS", "ΑΓΓΕΛΟΥ ΓΙΩΡΓΟΣ")

    links = pbp_links(names, same).set_index("source_id")
    assert links.loc["pbp:23:Giorgos Aggelou", "linked_source_id"] == "0000AAAA"
    assert pd.isna(links.loc["pbp:9:Nick Other", "linked_source_id"])
    assert links.loc["pbp:9:Nick Other", "reason"] == "1 ids with jersey 9, 0 with the name"
    assert links.loc["pbp:0:Gabe York", "reason"] == "no jersey in the key"
