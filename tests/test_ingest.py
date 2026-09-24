import json
from pathlib import Path

import httpx
import pytest

from eurohoops.ingest.euroleague import MIN_INTERVAL_S, ingest_seasons
from eurohoops.ingest.http import MAX_ATTEMPTS, Fetcher
from tests.conftest import FIXTURES

DETAIL_BODY = b'{"Live": false}'
LIVE = 2026


class FakeApi:
    """Serves recorded schedule fixtures and a tiny detail body; counts every request."""

    def __init__(self, empty_details: bool = False) -> None:
        self.calls: list[str] = []
        self.empty_details = empty_details

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(str(request.url))
        if request.url.host == "api-live.euroleague.net":
            season = request.url.path.split("/")[-2].removeprefix("E")
            return httpx.Response(200, content=(FIXTURES / f"schedule_E{season}.json").read_bytes())
        return httpx.Response(200, content=b"" if self.empty_details else DETAIL_BODY)


def fetcher_for(api: FakeApi) -> Fetcher:
    return Fetcher(httpx.Client(transport=httpx.MockTransport(api)), MIN_INTERVAL_S)


@pytest.mark.usefixtures("no_sleep")
def test_second_ingest_of_completed_season_makes_zero_http_calls(tmp_path: Path) -> None:
    first = FakeApi()
    schedules = ingest_seasons(fetcher_for(first), tmp_path, [2024], details=True, live_season=LIVE)
    assert len(schedules[2024]) == 3
    assert len(first.calls) == 1 + 3 * 3  # schedule + 3 endpoints x 3 played games

    second = FakeApi()
    again = ingest_seasons(fetcher_for(second), tmp_path, [2024], details=True, live_season=LIVE)
    assert second.calls == []
    assert again == schedules


@pytest.mark.usefixtures("no_sleep")
def test_live_season_schedule_is_refreshed_and_unplayed_games_get_no_details(
    tmp_path: Path,
) -> None:
    ingest_seasons(fetcher_for(FakeApi()), tmp_path, [2026], details=True, live_season=LIVE)
    api = FakeApi()
    ingest_seasons(fetcher_for(api), tmp_path, [2026], details=True, live_season=LIVE)
    assert len(api.calls) == 1
    assert "seasons/E2026/games" in api.calls[0]


@pytest.mark.usefixtures("no_sleep")
def test_raw_cache_layout_is_gzip_and_file_existence_based(tmp_path: Path) -> None:
    ingest_seasons(fetcher_for(FakeApi()), tmp_path, [2024], details=True, live_season=LIVE)
    assert (tmp_path / "schedule" / "E2024.json.gz").exists()
    detail = tmp_path / "playbyplay" / "E2024" / "327.json.gz"
    assert detail.read_bytes()[:2] == b"\x1f\x8b"
    assert not list(tmp_path.rglob("*.tmp"))


@pytest.mark.usefixtures("no_sleep")
def test_empty_detail_bodies_are_not_cached(tmp_path: Path) -> None:
    ingest_seasons(
        fetcher_for(FakeApi(empty_details=True)), tmp_path, [2024], details=True, live_season=LIVE
    )
    assert not (tmp_path / "boxscore").exists()


@pytest.mark.usefixtures("no_sleep")
def test_schedules_only_without_details_flag(tmp_path: Path) -> None:
    api = FakeApi()
    ingest_seasons(fetcher_for(api), tmp_path, [2024], details=False, live_season=LIVE)
    assert len(api.calls) == 1


def test_fetcher_throttles_between_requests(no_sleep: list[float]) -> None:
    fetcher = fetcher_for(FakeApi())
    fetcher.get("https://live.euroleague.net/api/Boxscore")
    fetcher.get("https://live.euroleague.net/api/Boxscore")
    assert len(no_sleep) == 1
    assert 0 < no_sleep[0] <= MIN_INTERVAL_S


def test_fetcher_retries_server_errors_with_backoff(no_sleep: list[float]) -> None:
    responses = iter([httpx.Response(503), httpx.Response(429), httpx.Response(200, json={})])
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return next(responses)

    fetcher = Fetcher(httpx.Client(transport=httpx.MockTransport(handler)), MIN_INTERVAL_S)
    assert json.loads(fetcher.get("https://x.test/")) == {}
    assert len(calls) == 3
    assert no_sleep  # backoff waited


def test_fetcher_gives_up_after_max_attempts(no_sleep: list[float]) -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        raise httpx.ConnectError("down", request=request)

    with pytest.raises(httpx.ConnectError):
        Fetcher(httpx.Client(transport=httpx.MockTransport(handler)), MIN_INTERVAL_S).get(
            "https://x.test/"
        )
    assert len(calls) == MAX_ATTEMPTS


def test_fetcher_does_not_retry_client_errors(no_sleep: list[float]) -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(404)

    with pytest.raises(httpx.HTTPStatusError):
        Fetcher(httpx.Client(transport=httpx.MockTransport(handler)), MIN_INTERVAL_S).get(
            "https://x.test/"
        )
    assert len(calls) == 1


@pytest.mark.usefixtures("no_sleep")
def test_past_season_is_final_even_with_unplayed_games(tmp_path: Path) -> None:
    """2019-20 (COVID) and 2021-22 (voided games) keep unplayed games forever."""
    ingest_seasons(fetcher_for(FakeApi()), tmp_path, [2026], details=False, live_season=2027)
    api = FakeApi()
    ingest_seasons(fetcher_for(api), tmp_path, [2026], details=False, live_season=2027)
    assert api.calls == []
