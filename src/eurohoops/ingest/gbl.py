"""Greek Basket League (ESAKE) ingestion into a gzip raw-HTML cache.

Cache layout (``root`` = data/raw/gbl):
  results/{season}/{phase}/{round}.html.gz   final for past seasons, and for live-season rounds
                                             whose games all have a score; other pages refresh
  boxscore/{season}/{idgame}.html.gz         played games only; written once, never refetched

Rounds of a phase come from the list embedded in its first page (``series=01``). Old-format
playoff pages (2018-19 to 2022-23) have no list, so their rounds are enumerated until a page
is empty.
"""

import logging
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

    def box_score(self, season: int, idgame: str) -> None:
        path = self.root / "boxscore" / str(season) / f"{idgame}.html.gz"
        if path.exists():
            self.hits += 1
            return
        write_atomic(path, self.fetcher.get(BOX_URL.format(idgame=idgame)))
        self.fetches += 1


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
    fetcher: Fetcher, root: Path, seasons: list[int], details: bool, live_season: int
) -> list[RoundPage]:
    """Return every round page of ``seasons``; ``details`` also caches played games' box scores."""
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
    log.info("gbl pages: %d cached, %d fetched", session.hits, session.fetches)
    return rounds
