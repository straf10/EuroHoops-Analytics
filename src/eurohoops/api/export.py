"""The static export as a client of the read-only API (weeks 16-18, L10).

``eurohoops publish`` and ``eurohoops export-stats`` no longer build their JSON themselves: they
GET each route the site reads through FastAPI's ``TestClient`` (in-process, no server, no
network) and write the response bytes unchanged, so every file under ``web/src/data/`` is
exactly the API's answer for its route. ``file_for`` and ``route_for`` are the one mapping
between the two; the contract test walks the files and checks each against its route.

Files, relative to ``web/``:

- ``src/data/site.json`` from ``/site``; ``src/data/stats/<path>`` from ``/stats/<path>``;
- ``src/data/api/...`` from the routes the pages of L11 read (see ``file_for``);
- ``public/api/openapi.json`` from ``/openapi.json``, published with the site.

A person id has a colon (``P:003733``); the file key replaces it with ``_`` (``file_key``).
"""

import json
import re
import shutil
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from fastapi.testclient import TestClient
from httpx import Response

from eurohoops.api.readmodel import Store
from eurohoops.config import (
    EUROLEAGUE,
    LIVE_SEASON,
    M6_PROJECTIONS,
    M6_SIMILARITY,
    SITE_DATA,
)
from eurohoops.publish import DISPLAY_CODES

WEB_DIR = SITE_DATA.parent.parent.parent  # web/
DATA_DIR = PurePosixPath(SITE_DATA.parent.relative_to(WEB_DIR).as_posix())  # src/data
SITE_FILE = DATA_DIR / "site.json"
STATS_PREFIX = DATA_DIR / "stats"
API_PREFIX = DATA_DIR / "api"
OPENAPI_FILE = PurePosixPath("public/api/openapi.json")
OPENAPI_ROUTE = "/openapi.json"

PERSON_ID = re.compile(r"^[A-Z]:[0-9A-Za-z]+$")
SCOUTING = ("board",)
FIXED = {
    "/site": SITE_FILE,
    OPENAPI_ROUTE: OPENAPI_FILE,
    "/metrics/live": API_PREFIX / "metrics" / "live.json",
    **{f"/scouting/{name}": API_PREFIX / "scouting" / f"{name}.json" for name in SCOUTING},
}
FIXED_ROUTES = {file: route for route, file in FIXED.items()}
PLAYER_FILES = {"projection": "projection.json", "similar": "similar.json", None: "index.json"}
SIMULATION = re.compile(r"^/simulations/latest\?competition=(?P<c>\w+)$")
PLAYER = re.compile(r"^/players/(?P<id>[^/]+)(?:/(?P<what>projection|similar))?$")
TEAM = re.compile(
    rf"^/teams/(?P<c>\w+)/(?P<team>[^/]+)/(?:factors\?season={LIVE_SEASON}|(?P<r>ratings))$"
)


class ExportError(RuntimeError):
    """A route answered something other than 200 where the export needs a file."""


@dataclass(frozen=True)
class Route:
    """A route to export; an ``optional`` one that answers 404 writes no file (and is reported)."""

    path: str
    optional: bool = False


@dataclass(frozen=True)
class RawStore(Store):
    """A ``Store`` whose EuroLeague raw cache is ``raw_dir`` (``export-stats --raw-dir``)."""

    raw_dir: Path = EUROLEAGUE.raw_dir

    def path(self, relative: Path) -> Path:
        return self.raw_dir if relative == EUROLEAGUE.raw_dir else super().path(relative)


def file_key(person_id: str) -> str:
    """The Windows-safe file name of a person id: ``P:003733`` -> ``P_003733``."""
    if not PERSON_ID.match(person_id):
        raise ValueError(f"not a person id: {person_id!r}")
    return person_id.replace(":", "_")


def person_of(key: str) -> str:
    """Inverse of ``file_key``: the first ``_`` (an id holds no other) is the colon."""
    return key.replace("_", ":", 1)


def file_for(route: str) -> PurePosixPath:
    """The file (relative to ``web/``) a route is written to."""
    if route in FIXED:
        return FIXED[route]
    if route.startswith("/stats/"):
        return STATS_PREFIX / route.removeprefix("/stats/")
    if match := SIMULATION.match(route):
        return API_PREFIX / "simulations" / f"{match['c']}.json"
    if match := PLAYER.match(route):
        return API_PREFIX / "players" / file_key(match["id"]) / PLAYER_FILES[match["what"]]
    if match := TEAM.match(route):
        name = "ratings.json" if match["r"] else "factors.json"
        return API_PREFIX / "teams" / match["c"] / match["team"] / name
    raise ValueError(f"no file for route {route!r}")


def route_for(file: PurePosixPath) -> str:
    """The route a file (relative to ``web/``) is the response of; inverse of ``file_for``."""
    if file in FIXED_ROUTES:
        return FIXED_ROUTES[file]
    if file.is_relative_to(STATS_PREFIX):
        return "/stats/" + file.relative_to(STATS_PREFIX).as_posix()
    if file.is_relative_to(API_PREFIX):
        match file.relative_to(API_PREFIX).parts:
            case ("simulations", name) if name.endswith(".json"):
                return f"/simulations/latest?competition={name.removesuffix('.json')}"
            case ("players", key, name) if name in PLAYER_FILES.values():
                what = next(k for k, v in PLAYER_FILES.items() if v == name)
                return f"/players/{person_of(key)}" + ("" if what is None else f"/{what}")
            case ("teams", competition, team, "ratings.json"):
                return f"/teams/{competition}/{team}/ratings"
            case ("teams", competition, team, "factors.json"):
                return f"/teams/{competition}/{team}/factors?season={LIVE_SEASON}"
    raise ValueError(f"no route for file {file.as_posix()!r}")


def _body(route: str, response: Response) -> bytes:
    if response.status_code != 200:
        raise ExportError(f"GET {route} answered {response.status_code}: {response.text.strip()}")
    return response.content


def get(client: TestClient, route: str) -> bytes:
    """The body of a 200 response; anything else is an ``ExportError`` naming route and body."""
    return _body(route, client.get(route))


def export_files(
    client: TestClient, routes: Iterable[Route], skipped: Callable[[str], None] = lambda _: None
) -> dict[PurePosixPath, bytes]:
    """Each route's response bytes by file (relative to ``web/``). A 404 on an optional route is
    passed to ``skipped`` with its reason and writes no file; any other failure raises."""
    files: dict[PurePosixPath, bytes] = {}
    for route in routes:
        response = client.get(route.path)
        if response.status_code == 404 and route.optional:
            skipped(f"skipped {route.path}: {response.text.strip()}")
        else:
            files[file_for(route.path)] = _body(route.path, response)
    return files


def _report(store: Store, relative: Path) -> dict[str, Any]:
    path = store.path(relative)
    report: dict[str, Any] = (
        json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"players": []}
    )
    return report


def _source_codes(competition: str) -> dict[str, str]:
    """Display code -> source code (``DISPLAY_CODES`` maps the other way)."""
    return {shown: code for code, shown in DISPLAY_CODES.get(competition, {}).items()}


def api_routes(store: Store, site: Mapping[str, Any]) -> list[Route]:
    """The routes behind ``src/data/api/``: scouting, simulations, metrics, every projected or
    similar person, and every team of the live season (from the ``/site`` ratings)."""
    routes = [Route(f"/scouting/{name}", optional=True) for name in SCOUTING]
    routes.append(Route(f"/simulations/latest?competition={EUROLEAGUE.name}", optional=True))
    routes.append(Route("/metrics/live", optional=True))
    projected = _report(store, M6_PROJECTIONS)["players"]
    similar = _report(store, M6_SIMILARITY)["players"]
    people = [row["person_id"] for row in projected if row["competition"] == EUROLEAGUE.name]
    routes += [Route(f"/players/{person}/projection", optional=True) for person in people]
    routes += [
        Route(f"/players/{person}/similar", optional=True) for person in similar if person in people
    ]
    routes += [Route(f"/players/{person}", optional=True) for person in people]
    for section in site["competitions"]:
        source = _source_codes(section["key"])
        for rating in section["ratings"]:
            shown = rating["code"]
            base = f"/teams/{section['key']}/{source.get(shown, shown)}"
            routes.append(Route(f"{base}/factors?season={LIVE_SEASON}", optional=True))
            routes.append(Route(f"{base}/ratings", optional=True))
    return routes


def publish_files(
    client: TestClient, store: Store, skipped: Callable[[str], None] = lambda _: None
) -> dict[PurePosixPath, bytes]:
    """Everything ``eurohoops publish`` writes: the site, the API files and the OpenAPI document."""
    site = get(client, "/site")
    files = {SITE_FILE: site}
    files.update(export_files(client, api_routes(store, json.loads(site)), skipped))
    files[OPENAPI_FILE] = get(client, OPENAPI_ROUTE)
    return files


def stats_files(client: TestClient) -> dict[str, bytes]:
    """Every stats file by its path below the stats directory (the ``/stats`` index)."""
    index = json.loads(get(client, "/stats"))
    return {path: get(client, f"/stats/{path}") for path in index["files"]}


def write_publish(web: Path, files: Mapping[PurePosixPath, bytes]) -> None:
    """Write ``files`` below ``web``, replacing the ``api`` directory so no stale file stays."""
    if (web / API_PREFIX).exists():
        shutil.rmtree(web / API_PREFIX)
    for relative, body in files.items():
        target = web / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)


def write_stats_files(out_dir: Path, files: Mapping[str, bytes]) -> None:
    """Replace the stats directory's ``seasons`` and write ``files`` (as ``write_stats`` does)."""
    if (out_dir / "seasons").exists():
        shutil.rmtree(out_dir / "seasons")
    for relative, body in files.items():
        target = out_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)
