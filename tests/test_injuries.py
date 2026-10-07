import gzip
import logging
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pandas as pd
import pytest

from eurohoops.injuries import (
    INJURY_COLUMNS,
    InjuryPaths,
    InjuryReportError,
    parse_report,
    record_injuries,
    team_name_map,
)

FIXTURE = Path(__file__).parent / "fixtures" / "basketnews" / "injury_report.html"
HTML = FIXTURE.read_bytes()
NAMES = {"Anadolu Efes Istanbul": "IST", "Fenerbahce Beko Istanbul": "ULK"}
T1 = datetime(2026, 10, 7, 8, 0, tzinfo=UTC)
T2 = datetime(2026, 10, 8, 8, 0, tzinfo=UTC)
UPDATED = "2026-10-07T05:36:50Z"


def paths(tmp_path: Path) -> InjuryPaths:
    return InjuryPaths(tmp_path / "injuries" / "log.csv", tmp_path / "raw")


def client(status: int = 200, body: bytes = HTML) -> httpx.Client:
    calls = httpx.MockTransport(lambda request: httpx.Response(status, content=body))
    return httpx.Client(transport=calls)


def logged(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def test_parse_report_reads_teams_players_and_update_time() -> None:
    updated, rows = parse_report(HTML)
    assert updated == UPDATED  # the date beside the title, not the first date on the page
    assert [(r["team_name_bn"], r["player"]) for r in rows] == [
        ("Anadolu Efes Istanbul", "Isaia Cordinier"),
        ("Anadolu Efes Istanbul", "Burak Can Yildizli"),
        ("Fenerbahce Beko Istanbul", "Nando De Colo"),
        ("Example Unknown Club", "John Doe"),
        ("Example Unknown Club", "Jane Roe"),
    ]
    first, empty_comment, markup = rows[0], rows[1], rows[2]
    assert first == {
        "page_updated_utc": UPDATED,
        "team": "",
        "team_name_bn": "Anadolu Efes Istanbul",
        "player": "Isaia Cordinier",
        "bn_player_id": "23741",
        "position": "PG",
        "status": "Out",
        "return_round": "Indefinitely",
        "comment": "Rehab after surgery",
    }
    assert empty_comment["comment"] == "" and empty_comment["return_round"] == "Out for season"
    assert (markup["status"], markup["comment"]) == ("Game-time", "Knee & ankle, day-to-day")


def test_parse_report_without_update_time_leaves_it_empty() -> None:
    html = HTML.replace(b'data-time="1791351410"', b'data-time="soon"')
    updated, rows = parse_report(html)
    assert updated == ""
    assert len(rows) == 5


def test_parse_report_without_a_table_raises() -> None:
    with pytest.raises(InjuryReportError, match="table not found"):
        parse_report(b"<html><body><p>maintenance</p></body></html>")


def test_team_name_map_uses_season_teams_and_the_sponsor_alias() -> None:
    teams = pd.DataFrame(
        {
            "team": ["IST", "ULK", "OLD", "ZZZ"],
            "name": ["Anadolu Efes Istanbul", "Fenerbahce Tarfin Istanbul", "EWE Baskets", "X"],
        }
    )
    games = pd.DataFrame(
        {
            "season": [2026, 2026, 2025],
            "home": ["IST", "ULK", "OLD"],
            "away": ["ULK", "IST", "ZZZ"],
        }
    )
    assert team_name_map(teams, games, 2026) == {
        "Anadolu Efes Istanbul": "IST",
        "Fenerbahce Tarfin Istanbul": "ULK",
        "Fenerbahce Beko Istanbul": "ULK",
    }


def test_record_appends_a_snapshot_with_mapped_teams_and_warns_on_unmapped(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    p = paths(tmp_path)
    with caplog.at_level(logging.WARNING):
        summary = record_injuries(client(), NAMES, p, lambda: T1)
    assert (summary["status"], summary["rows"], summary["teams"]) == ("appended", 5, 3)
    assert summary["unmapped"] == ["Example Unknown Club"]
    assert "Example Unknown Club" in caplog.text
    table = logged(p.log)
    assert tuple(table.columns) == INJURY_COLUMNS
    assert set(table["snapshot_utc"]) == {"2026-10-07T08:00:00Z"}
    assert table["team"].tolist() == ["IST", "IST", "ULK", "", ""]  # unmapped kept, empty team
    assert table["bn_player_id"].tolist()[:2] == ["23741", "14714"]
    [cached] = (tmp_path / "raw").glob("*.html.gz")
    assert cached.name == "20261007T080000Z.html.gz"
    assert gzip.decompress(cached.read_bytes()) == HTML


def test_unchanged_snapshot_is_not_appended_again_but_a_change_is(tmp_path: Path) -> None:
    p = paths(tmp_path)
    record_injuries(client(), NAMES, p, lambda: T1)
    before = p.log.read_bytes()
    again = record_injuries(client(), NAMES, p, lambda: T2)
    assert again["status"] == "unchanged"
    assert p.log.read_bytes() == before
    changed = HTML.replace(b"Back soon", b"Out two weeks")
    third = record_injuries(client(body=changed), NAMES, p, lambda: T2)
    assert third["status"] == "appended"
    assert p.log.read_bytes().startswith(before)  # existing rows untouched
    table = logged(p.log)
    assert table["snapshot_utc"].value_counts().to_dict() == {
        "2026-10-07T08:00:00Z": 5,
        "2026-10-08T08:00:00Z": 5,
    }


def test_new_page_update_time_with_same_rows_is_a_new_snapshot(tmp_path: Path) -> None:
    p = paths(tmp_path)
    record_injuries(client(), NAMES, p, lambda: T1)
    later = HTML.replace(b"1791351410", b"1791437810")
    result = record_injuries(client(body=later), NAMES, p, lambda: T2)
    assert result["status"] == "appended"
    assert len(logged(p.log)) == 10


@pytest.mark.parametrize(
    ("status", "body"),
    [
        (500, b"error"),
        (404, b"gone"),
        (200, b"<html><body>no table</body></html>"),
        (
            200,
            b'<table class="table_02"><tr class="injury_reports__head"><th>P</th></tr></table>',
        ),
    ],
)
def test_failures_warn_and_write_nothing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, status: int, body: bytes
) -> None:
    p = paths(tmp_path)
    with caplog.at_level(logging.WARNING):
        summary = record_injuries(client(status, body), NAMES, p, lambda: T1)
    assert summary["status"] == "failed"
    assert "nothing recorded" in caplog.text
    assert not p.log.exists()
    assert not (tmp_path / "raw").exists()


def test_transport_error_is_a_failure_too(tmp_path: Path) -> None:
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    p = paths(tmp_path)
    summary = record_injuries(
        httpx.Client(transport=httpx.MockTransport(boom)), NAMES, p, lambda: T1
    )
    assert summary["status"] == "failed"
    assert not p.log.exists()


def test_a_failure_leaves_an_existing_log_untouched(tmp_path: Path) -> None:
    p = paths(tmp_path)
    record_injuries(client(), NAMES, p, lambda: T1)
    before = p.log.read_bytes()
    record_injuries(client(500, b"x"), NAMES, p, lambda: T2)
    assert p.log.read_bytes() == before


def test_a_torn_log_is_refused(tmp_path: Path) -> None:
    p = paths(tmp_path)
    p.log.parent.mkdir()
    p.log.write_bytes(b"snapshot_utc,page_updated_utc")  # no trailing newline
    with pytest.raises(ValueError, match="newline"):
        record_injuries(client(), NAMES, p, lambda: T1)
