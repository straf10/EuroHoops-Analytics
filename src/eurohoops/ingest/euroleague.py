"""EuroLeague ingestion into an immutable, gzip-compressed raw cache.

Cache layout (``root`` = data/raw/euroleague):
  schedule/E{season}.json.gz                  refreshed until every game in it is played
  {endpoint}/E{season}/{game_code}.json.gz    completed games only; written once, never refetched
"Already have it" is a file-existence check; there is no index file.
"""

import gzip
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eurohoops.ingest.http import Fetcher

SCHEDULE_URL = "https://api-live.euroleague.net/v2/competitions/E/seasons/E{season}/games"
DETAIL_URL = "https://live.euroleague.net/api/{endpoint}?gamecode={game_code}&seasoncode=E{season}"
DETAIL_ENDPOINTS = ("Boxscore", "PlaybyPlay", "Points")

log = logging.getLogger(__name__)

RawGame = dict[str, Any]


@dataclass
class CacheStats:
    schedule_hits: int = 0
    schedule_fetches: int = 0
    detail_hits: int = 0
    detail_fetches: int = 0
    detail_empty: int = 0

    def summary(self) -> str:
        return (
            f"schedules: {self.schedule_hits} cached, {self.schedule_fetches} fetched; "
            f"details: {self.detail_hits} cached, {self.detail_fetches} fetched, "
            f"{self.detail_empty} empty (not cached)"
        )


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(gzip.compress(payload, mtime=0))
    tmp.replace(path)


def _schedule_games(payload: bytes) -> list[RawGame]:
    games: list[RawGame] = json.loads(payload)["data"]
    return games


def _load_schedule(fetcher: Fetcher, root: Path, season: int, stats: CacheStats) -> list[RawGame]:
    path = root / "schedule" / f"E{season}.json.gz"
    if path.exists():
        games = _schedule_games(gzip.decompress(path.read_bytes()))
        if all(game["played"] for game in games):
            stats.schedule_hits += 1
            return games
    payload = fetcher.get(SCHEDULE_URL.format(season=season))
    stats.schedule_fetches += 1
    _write_atomic(path, payload)
    return _schedule_games(payload)


def _fetch_details(
    fetcher: Fetcher, root: Path, season: int, games: list[RawGame], stats: CacheStats
) -> None:
    for game in games:
        if not game["played"]:
            continue
        for endpoint in DETAIL_ENDPOINTS:
            path = root / endpoint.lower() / f"E{season}" / f"{game['gameCode']}.json.gz"
            if path.exists():
                stats.detail_hits += 1
                continue
            url = DETAIL_URL.format(endpoint=endpoint, game_code=game["gameCode"], season=season)
            payload = fetcher.get(url)
            stats.detail_fetches += 1
            if payload.strip():
                _write_atomic(path, payload)
            else:
                stats.detail_empty += 1
                log.warning("empty %s body for E%d game %s", endpoint, season, game["gameCode"])


def ingest_seasons(
    fetcher: Fetcher, root: Path, seasons: list[int], details: bool
) -> dict[int, list[RawGame]]:
    """Return raw schedules per season; with ``details``, also cache detail JSON of played games."""
    stats = CacheStats()
    schedules = {season: _load_schedule(fetcher, root, season, stats) for season in seasons}
    if details:
        for season, games in schedules.items():
            _fetch_details(fetcher, root, season, games, stats)
    log.info(stats.summary())
    return schedules
