import csv
import gzip
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import pytest

from eurohoops.odds import (
    MIN_REMAINING,
    OddsApiError,
    OddsPaths,
    api_key,
    consensus,
    devig_two_way,
    last_remaining,
    record_odds,
)

FAKE_KEY = "0123456789abcdef0123456789abcdef"
NOW = datetime(2026, 10, 1, 8, 0, tzinfo=UTC)
HEADERS = {"x-requests-last": "3", "x-requests-used": "3", "x-requests-remaining": "497"}


def book(key: str, home: float, away: float, spread: float, total: float) -> dict[str, Any]:
    return {
        "key": key,
        "title": key,
        "markets": [
            {
                "key": "h2h",
                "outcomes": [
                    {"name": "Real Madrid", "price": home},
                    {"name": "Olympiacos", "price": away},
                ],
            },
            {
                "key": "spreads",
                "outcomes": [
                    {"name": "Real Madrid", "price": 1.9, "point": spread},
                    {"name": "Olympiacos", "price": 1.9, "point": -spread},
                ],
            },
            {
                "key": "totals",
                "outcomes": [
                    {"name": "Over", "price": 1.9, "point": total},
                    {"name": "Under", "price": 1.9, "point": total},
                ],
            },
        ],
    }


def event(home: str, away: str, commence: str, books: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": f"{home}-{away}",
        "sport_key": "basketball_euroleague",
        "commence_time": commence,
        "home_team": home,
        "away_team": away,
        "bookmakers": books,
    }


THREE_WAY = {
    "key": "threeway",
    "markets": [
        {
            "key": "h2h",
            "outcomes": [
                {"name": "Real Madrid", "price": 1.5},
                {"name": "Olympiacos", "price": 3.0},
                {"name": "Draw", "price": 15.0},
            ],
        }
    ],
}
EVENTS = [
    event(
        "Real Madrid",
        "Olympiacos",
        "2026-10-01T18:45:00Z",
        [
            book("a", 1.80, 2.10, -3.5, 162.5),
            book("b", 1.70, 2.20, -4.5, 160.5),
            book("c", 1.75, 2.05, -2.5, 161.5),
            THREE_WAY,
        ],
    ),
    event("Olympiacos", "Real Madrid", "2026-10-01T07:00:00Z", [book("a", 2, 2, 0, 160)]),
    event("Unknown BC", "Real Madrid", "2026-10-02T18:00:00Z", []),
    event("Real Madrid", "Olympiacos", "2026-12-01T18:00:00Z", []),
]


def games() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["E2026_40", "E2026_41"],
            "season": [2026, 2026],
            "round": [2, 3],
            "tipoff_utc": pd.to_datetime(["2026-10-01T18:45:00Z", "2026-10-08T18:00:00Z"]),
            "home": ["MAD", "OLY"],
            "away": ["OLY", "MAD"],
            "played": [False, False],
        }
    )


class FakeOddsApi:
    def __init__(self, status: int = 200, body: Any = None) -> None:
        self.status = status
        self.body = EVENTS if body is None else body
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(self.status, json=self.body, headers=HEADERS)


@pytest.fixture
def paths(tmp_path: Path) -> OddsPaths:
    teams = tmp_path / "teams.csv"
    teams.write_text("odds_api_name,team,verified\nReal Madrid,MAD,no\nOlympiacos,OLY,no\n")
    return OddsPaths(tmp_path / "odds.csv", tmp_path / "calls.csv", teams, tmp_path / "raw")


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_devig_on_a_two_way_market() -> None:
    # 1/1.80 + 1/2.10 = 1.0317 (3.2% margin); P(home) = 2.10 / (2.10 + 1.80) = 0.538462.
    assert devig_two_way(1.80, 2.10) == pytest.approx(2.10 / 3.90)
    assert devig_two_way(1.80, 2.10) + devig_two_way(2.10, 1.80) == pytest.approx(1.0)


def test_consensus_medians_skip_three_way_moneylines() -> None:
    row = consensus(EVENTS[0])
    assert row["bookmakers"] == 3  # the three-way (draw) book is left out
    median = sorted([2.10 / 3.90, 2.20 / 3.90, 2.05 / 3.80])[1]
    assert row["p_home"] == f"{median:.4f}"
    assert (row["spread_books"], row["spread_home"]) == (3, "-3.5")
    assert (row["total_books"], row["total"]) == (3, "161.5")


def test_consensus_without_markets_leaves_blanks() -> None:
    assert consensus(EVENTS[2]) == {
        "bookmakers": 0,
        "p_home": "",
        "spread_books": 0,
        "spread_home": "",
        "total_books": 0,
        "total": "",
    }


def run(paths: OddsPaths, api: FakeOddsApi) -> dict[str, Any]:
    with httpx.Client(transport=httpx.MockTransport(api)) as client:
        return record_odds(client, FAKE_KEY, games(), paths, lambda: NOW)


def test_one_call_records_consensus_quota_and_raw_cache(
    paths: OddsPaths, caplog: pytest.LogCaptureFixture
) -> None:
    api = FakeOddsApi()
    # httpx logs request URLs (with the key) at INFO: open every logger up to DEBUG.
    with caplog.at_level(logging.DEBUG), caplog.at_level(logging.DEBUG, logger="httpx"):
        httpx_level = logging.getLogger("httpx").level
        summary = run(paths, api)
        assert logging.getLogger("httpx").level == httpx_level  # restored after the call
    assert len(api.requests) == 1
    query = dict(api.requests[0].url.params)
    assert query["apiKey"] == FAKE_KEY
    assert query["markets"] == "h2h,spreads,totals"
    assert (query["regions"], query["oddsFormat"]) == ("eu", "decimal")
    # Event 2 is in play (skipped), event 3 has an unknown team, event 4 has no game nearby.
    assert summary["rows_written"] == 1
    assert len(summary["unmatched"]) == 2
    assert "Unknown BC" in summary["unmatched"][0]
    assert "no scheduled MAD-OLY game" in summary["unmatched"][1]
    [row] = rows(paths.log)
    assert row["game_id"] == "E2026_40"
    assert row["fetched_at_utc"] == "2026-10-01T08:00:00Z"
    assert row["tipoff_utc"] == "2026-10-01T18:45:00Z"
    [call] = rows(paths.calls)
    assert call == {
        "called_at_utc": "2026-10-01T08:00:00Z",
        "status": "200",
        "cost": "3",
        "requests_used": "3",
        "requests_remaining": "497",
        "events": "4",
        "rows_written": "1",
    }
    raw = gzip.decompress((paths.raw_dir / "20261001T080000Z.json.gz").read_bytes())
    assert json.loads(raw) == EVENTS
    assert "remaining 497" in caplog.text
    public = paths.log.read_text() + paths.calls.read_text() + caplog.text
    assert FAKE_KEY not in public


def test_runs_only_append(paths: OddsPaths) -> None:
    run(paths, FakeOddsApi())
    first_log, first_calls = paths.log.read_bytes(), paths.calls.read_bytes()
    run(paths, FakeOddsApi())
    assert paths.log.read_bytes().startswith(first_log)
    assert paths.calls.read_bytes().startswith(first_calls)
    assert len(rows(paths.log)) == 2  # the same game gains a second snapshot
    assert last_remaining(paths.calls) == 497


@pytest.mark.parametrize(("remaining", "calls"), [(MIN_REMAINING - 1, 0), (MIN_REMAINING, 1)])
def test_quota_guard(paths: OddsPaths, remaining: int, calls: int) -> None:
    paths.calls.write_text(
        "called_at_utc,status,cost,requests_used,requests_remaining,events,rows_written\n"
        f"2026-09-30T08:00:00Z,200,3,480,{remaining},4,1\n"
        "2026-09-30T09:00:00Z,500,,,,,0\n"  # a call without headers does not hide the count
    )
    api = FakeOddsApi()
    if calls:
        run(paths, api)
    else:
        with pytest.raises(OddsApiError, match="refusing to call"):
            run(paths, api)
    assert len(api.requests) == calls


def test_rejected_key_is_logged_without_the_key(paths: OddsPaths) -> None:
    api = FakeOddsApi(401, {"message": "API key is not valid", "error_code": "INVALID_KEY"})
    with pytest.raises(OddsApiError) as err:
        run(paths, api)
    assert str(err.value) == "HTTP 401 INVALID_KEY"
    assert rows(paths.calls)[0]["status"] == "401"
    assert not paths.log.exists()


def test_transport_errors_hide_the_url(paths: OddsPaths) -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"cannot reach {request.url}", request=request)

    with (
        httpx.Client(transport=httpx.MockTransport(refuse)) as client,
        pytest.raises(OddsApiError) as err,
    ):
        record_odds(client, FAKE_KEY, games(), paths, lambda: NOW)
    assert FAKE_KEY not in str(err.value)
    assert err.value.__cause__ is None
    assert err.value.__suppress_context__


def test_non_json_error_body(paths: OddsPaths) -> None:
    def broken(request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, text="<html>bad gateway</html>")

    with (
        httpx.Client(transport=httpx.MockTransport(broken)) as client,
        pytest.raises(OddsApiError, match=r"^HTTP 502$"),
    ):
        record_odds(client, FAKE_KEY, games(), paths, lambda: NOW)


def test_api_key_sources(tmp_path: Path) -> None:
    dotenv = tmp_path / ".env"
    assert api_key({}, dotenv) is None
    dotenv.write_text('OTHER=1\nODDS_API_KEY="from-file"\n')
    assert api_key({}, dotenv) == "from-file"
    assert api_key({"ODDS_API_KEY": "from-env"}, dotenv) == "from-env"
    dotenv.write_text("ODDS_API_KEY=\n")
    assert api_key({}, dotenv) is None
