from datetime import date
from pathlib import Path

import httpx
import pytest

from eurohoops.ingest.bios import (
    build_bios,
    el_people_path,
    gbl_bio_path,
    ingest_bios,
    parse_el_people,
    parse_esake_bio,
)
from eurohoops.ingest.http import Fetcher

FIXTURES = Path(__file__).parent / "fixtures" / "bios"
ESAKE_PAGE = (FIXTURES / "esake_player_TEST0001.html").read_bytes()
EL_PEOPLE = (FIXTURES / "el_people_E2024.json").read_bytes()


def test_esake_bio_reads_birth_date_and_country() -> None:
    assert parse_esake_bio(ESAKE_PAGE.decode("utf-8")) == (date(1991, 3, 9), "ΕΛΛΑΔΑ")
    assert parse_esake_bio("<table><tr><td>ΘΕΣΗ</td><td>C</td></tr></table>") == (None, None)


def test_el_people_keeps_players_and_prefixes_the_box_id() -> None:
    assert parse_el_people(EL_PEOPLE) == [
        ("P900002", date(1998, 11, 24), "GRE"),
        ("P900003", None, None),
    ]


@pytest.mark.usefixtures("no_sleep")
def test_ingest_caches_once_and_skips_pbp_keys(tmp_path: Path) -> None:
    calls: list[str] = []

    def api(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        body = EL_PEOPLE if "euroleague" in request.url.host else ESAKE_PAGE
        return httpx.Response(200, content=body)

    fetcher = Fetcher(httpx.Client(transport=httpx.MockTransport(api)), 0.0)
    gbl_raw, el_raw = tmp_path / "gbl", tmp_path / "euroleague"
    kwargs = {
        "gbl_raw": gbl_raw,
        "el_raw": el_raw,
        "gbl_ids": ["0000ABCD", "pbp:7:Some One", "0000ABCD"],
        "el_seasons": [2024],
    }
    assert ingest_bios(fetcher, fetcher, **kwargs) == (1, 1)  # type: ignore[arg-type]
    assert ingest_bios(fetcher, fetcher, **kwargs) == (0, 0)  # type: ignore[arg-type]
    assert len(calls) == 2
    assert "limit=2000" in calls[0]
    assert gbl_bio_path(gbl_raw, "0000ABCD").exists()
    assert el_people_path(el_raw, 2024).exists()

    bios = build_bios(gbl_raw, el_raw).astype(object)
    assert bios.where(bios.notna(), None).to_dict("records") == [
        {
            "competition": "euroleague",
            "source_id": "P900002",
            "birth_date": date(1998, 11, 24),
            "country": "GRE",
        },
        {"competition": "euroleague", "source_id": "P900003", "birth_date": None, "country": None},
        {
            "competition": "gbl",
            "source_id": "0000ABCD",
            "birth_date": date(1991, 3, 9),
            "country": "ΕΛΛΑΔΑ",
        },
    ]
