"""Greek Basket League (ESAKE) ingestion into a gzip raw-HTML cache.

Cache layout (``root`` = data/raw/gbl):
  results/{season}/{phase}/{round}.html.gz   final for past seasons, and for live-season rounds
                                             whose games all have a score; other pages refresh
  boxscore/{season}/{idgame}.html.gz         played games only; written once, never refetched
  pbp_widget/{season}/{idgame}.js.gz         BasketHotel widget response (holds the export id)
  pbp/{season}/{idgame}.xlsx.gz              BasketHotel play-by-play export; written once

Rounds of a phase come from the list embedded in its first page (``series=01``). Old-format
playoff pages (2018-19 to 2022-23) have no list, so their rounds are enumerated until a page
is empty.
"""

import logging
import re
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

from eurohoops.ingest.cache import read_cached, write_atomic
from eurohoops.ingest.http import Fetcher
from eurohoops.parse.esake import ResultsPage, parse_results_page

RESULTS_URL = (
    "https://www.esake.gr/el/action/EsakeResults"
    "?idchampionship={season_id}&idteam=&idseason={phase}&series={code}"
)
BOX_URL = "https://www.esake.gr/el/action/EsakegameView?idgame={idgame}&mode=3"
# BasketHotel serves the ESAKE PBP widget (docs/spikes/gbl-pbp.md). ESAKE's main.js hard-codes
# the api and league ids; its default season id resolves any game via external game ids.
BASKETHOTEL_API = "55b4cf328e78a7a16e07aefd9518ccb2fb1afa29"
PBP_WIDGET_URL = (
    "https://widgets.baskethotel.com/widget-service/show?api=" + BASKETHOTEL_API + "&lang=en"
    "&request[0][container]=games-pbp&request[0][widget]=400"
    "&request[0][param][league_id]=30409&request[0][param][season_id]=122843"
    "&request[0][param][game_id]={game_id}&request[0][param][use_external_game_ids]=1"
    "&request[0][param][show_tabs][0]=play_by_play&request[0][param][show_export_link]=1"
)
PBP_EXPORT_URL = (
    "https://widgets.baskethotel.com/widget-service/export/view/play_by_play"
    "?api=" + BASKETHOTEL_API + "&game_id={export_id}"
)
EXPORT_ID = re.compile(r'&game_id=\\?"\s*\+\s*(\d+)')  # the JS may be escaped inside a string
MIN_INTERVAL_S = 2.0
SEASON_IDS = {
    2018: "A12E05CD",
    2019: "49EEB365",
    2020: "03FFA3AC",
    2021: "8C367D67",
    2022: "DC917125",
    2023: "C1AF5EF5",
    2024: "4820C134",
    2025: "44B80BEB",
    2026: "184645B9",
}
# ESAKE phase A (regular season) and phase B (playoffs, play-outs, classification games).
PHASES = {"00000001": "RS", "00000002": "PO"}
DISCOVERY_CODE = "01"

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RoundPage:
    season: int
    phase: str  # ESAKE phase id, a key of PHASES
    code: str  # ESAKE round ("series") code
    page: ResultsPage


@dataclass
class _Session:
    fetcher: Fetcher
    root: Path
    live_season: int
    hits: int = 0
    fetches: int = 0

    def results(self, season: int, phase: str, code: str) -> ResultsPage:
        path = self.root / "results" / str(season) / phase / f"{code}.html.gz"
        if path.exists():
            self.hits += 1
            return parse_results_page(read_cached(path).decode("utf-8"))
        url = RESULTS_URL.format(season_id=SEASON_IDS[season], phase=phase, code=code)
        payload = self.fetcher.get(url)
        self.fetches += 1
        if season < self.live_season:  # cache before parsing: a parser bug never costs a refetch
            write_atomic(path, payload)
        page = parse_results_page(payload.decode("utf-8"))
        complete = bool(page.games) and all(g.home_score is not None for g in page.games)
        if season >= self.live_season and complete:
            write_atomic(path, payload)
        return page

    def pbp(self, season: int, idgame: str) -> None:
        """Two requests on a cold cache: the widget (for the export id), then the xlsx export."""
        export = self.root / "pbp" / str(season) / f"{idgame}.xlsx.gz"
        if export.exists():
            self.hits += 1
            return
        widget_path = self.root / "pbp_widget" / str(season) / f"{idgame}.js.gz"
        if widget_path.exists():
            self.hits += 1
            widget = read_cached(widget_path)
        else:
            widget = self.fetcher.get(PBP_WIDGET_URL.format(game_id=signed_game_id(idgame)))
            self.fetches += 1
            write_atomic(widget_path, widget)
        write_atomic(export, self.fetcher.get(PBP_EXPORT_URL.format(export_id=export_id(widget))))
        self.fetches += 1

    def box_score(self, season: int, idgame: str) -> None:
        path = self.root / "boxscore" / str(season) / f"{idgame}.html.gz"
        if path.exists():
            self.hits += 1
            return
        write_atomic(path, self.fetcher.get(BOX_URL.format(idgame=idgame)))
        self.fetches += 1


def signed_game_id(idgame: str) -> int:
    """BasketHotel reads ESAKE's hex game id as a signed 32-bit integer."""
    value = int(idgame, 16)
    return value - 2**32 if value >= 2**31 else value


class ExportIdError(ValueError):
    """The widget response carries no play-by-play export id."""


def export_id(widget: bytes) -> int:
    """BasketHotel's internal game id, from the export link in the widget response."""
    match = EXPORT_ID.search(widget.decode("utf-8", errors="replace"))
    if match is None:
        raise ExportIdError("no export game id in the widget response")
    return int(match.group(1))


def _phase_rounds(session: _Session, season: int, phase: str) -> list[RoundPage]:
    discovery = session.results(season, phase, DISCOVERY_CODE)
    if discovery.round_codes:
        return [
            RoundPage(
                season,
                phase,
                code,
                discovery if code == DISCOVERY_CODE else session.results(season, phase, code),
            )
            for code in discovery.round_codes
        ]
    rounds: list[RoundPage] = []
    number, page = 1, discovery
    while page.games:
        rounds.append(RoundPage(season, phase, f"{number:02d}", page))
        number += 1
        page = session.results(season, phase, f"{number:02d}")
    return rounds


def ingest_gbl(
    fetcher: Fetcher,
    root: Path,
    seasons: list[int],
    details: bool,
    live_season: int,
    *,
    pbp_seasons: Collection[int] = (),
) -> list[RoundPage]:
    """Return every round page of ``seasons``.

    ``details`` also caches played games' box scores; played games of ``pbp_seasons`` get
    their play-by-play export cached (a local backfill: never run in CI).
    """
    session = _Session(fetcher, root, live_season)
    rounds = [
        round_page
        for season in seasons
        for phase in PHASES
        for round_page in _phase_rounds(session, season, phase)
    ]
    if details:
        for round_page in rounds:
            for game in round_page.page.games:
                if game.home_score is not None and not game.forfeit:
                    session.box_score(round_page.season, game.idgame)
    for round_page in rounds:
        if round_page.season not in pbp_seasons:
            continue
        for game in round_page.page.games:
            if game.home_score is not None and not game.forfeit:
                session.pbp(round_page.season, game.idgame)
    log.info("gbl pages: %d cached, %d fetched", session.hits, session.fetches)
    return rounds
