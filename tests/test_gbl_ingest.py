from pathlib import Path

import httpx
import pytest

from eurohoops.ingest.gbl import MIN_INTERVAL_S, SEASON_IDS, ingest_gbl
from eurohoops.ingest.http import Fetcher
from eurohoops.parse.box import build_box_tables
from eurohoops.parse.games import build_gbl_tables
from tests.conftest import esake_fixture

EMPTY = "results_2019_po_01.html"
# (season, phase, series) -> fixture; anything else is an empty page.
PAGES = {
    # 2018-19: old format, no round list -> enumerated until an empty page.
    (2018, "00000001", "01"): "results_2018_po_01.html",
    (2018, "00000001", "02"): "results_2018_po_12.html",
    # 2026-27 (live): round list 01..26 on the first page; round 02 is already complete.
    (2026, "00000001", "01"): "results_2026_rs_01.html",
    (2026, "00000001", "02"): "results_2025_rs_24.html",
}
SEASON_BY_ID = {season_id: season for season, season_id in SEASON_IDS.items()}


class FakeEsake:
    def __init__(self) -> None:
        self.calls: list[httpx.URL] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request.url)
        params = request.url.params
        if "idgame" in params:
            return httpx.Response(200, text=esake_fixture("box_F26689D1.html"))
        key = (SEASON_BY_ID[params["idchampionship"]], params["idseason"], params["series"])
        return httpx.Response(200, text=esake_fixture(PAGES.get(key, EMPTY)))


def fetcher_for(api: FakeEsake) -> Fetcher:
    return Fetcher(httpx.Client(transport=httpx.MockTransport(api)), MIN_INTERVAL_S)


@pytest.mark.usefixtures("no_sleep")
def test_past_season_enumerates_old_rounds_and_second_run_makes_zero_calls(
    tmp_path: Path,
) -> None:
    first = FakeEsake()
    rounds = ingest_gbl(fetcher_for(first), tmp_path, [2018], details=True, live_season=2026)
    assert [(r.phase, r.code, len(r.page.games)) for r in rounds] == [
        ("00000001", "01", 4),
        ("00000001", "02", 1),
    ]
    # phase A: 01, 02, 03 (empty); phase B: 01 (empty); box scores of 4 played non-forfeits.
    assert len(first.calls) == 4 + 4
    second = FakeEsake()
    again = ingest_gbl(fetcher_for(second), tmp_path, [2018], details=True, live_season=2026)
    assert second.calls == []
    assert again == rounds


@pytest.mark.usefixtures("no_sleep")
def test_live_season_refetches_only_incomplete_rounds(tmp_path: Path) -> None:
    ingest_gbl(fetcher_for(FakeEsake()), tmp_path, [2026], details=False, live_season=2026)
    api = FakeEsake()
    rounds = ingest_gbl(fetcher_for(api), tmp_path, [2026], details=False, live_season=2026)
    series = [url.params["series"] for url in api.calls]
    assert "02" not in series  # complete round -> cached
    assert series.count("01") == 2  # unplayed discovery page (phase A) + empty phase B
    assert len(series) == 1 + 24 + 1
    assert [r.code for r in rounds] == [f"{n:02d}" for n in range(1, 27)]


def test_esake_throttle_is_at_least_two_seconds(no_sleep: list[float]) -> None:
    assert MIN_INTERVAL_S >= 2.0
    fetcher = fetcher_for(FakeEsake())
    fetcher.get("https://www.esake.gr/el/action/EsakegameView?idgame=00000001&mode=3")
    fetcher.get("https://www.esake.gr/el/action/EsakegameView?idgame=00000002&mode=3")
    assert len(no_sleep) == 1
    assert 1.9 < no_sleep[0] <= MIN_INTERVAL_S


@pytest.mark.usefixtures("no_sleep")
def test_box_tables_cover_played_non_forfeit_games(tmp_path: Path) -> None:
    rounds = ingest_gbl(fetcher_for(FakeEsake()), tmp_path, [2018], details=True, live_season=2026)
    games, _ = build_gbl_tables(rounds)
    player_box, team_box = build_box_tables(tmp_path, games)
    assert len(team_box) == 2 * 4
    assert set(team_box["game_id"]) == set(games.loc[~games["forfeit"], "game_id"])
    assert len(player_box) == 4 * 24
    assert (team_box.groupby("game_id")["total_points"].sum() == 81 + 64).all()
