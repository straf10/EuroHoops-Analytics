"""The read-only API (weeks 16-18, L5): FastAPI over ``api.readmodel``.

Every route is a GET that returns what a read-model function builds from the marts, logs and
committed reports; no route fits or computes a model, and the app never writes a file. A
404 body is ``{"detail": "<reason>"}`` (``/simulations/latest`` adds the M7 gate verdict).

Bytes: a response body is ``json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\\n"``
in UTF-8, exactly what ``stats.export.write_stats`` writes, so the static export can write the
response bytes as they are. ``/site`` is the one exception: it returns the indent-2 form
``logs.write_json`` writes for ``site.json``. ``/openapi.json`` is FastAPI's own and is stable
across app instances.

``create_app(store, now)`` injects the data tree and the clock (``now`` stamps the site and
``meta.json``; tests pass a fixed one). ``local_app`` is the uvicorn factory the Dockerfile and
``docker compose up`` run: ``uvicorn --factory eurohoops.api.app:local_app``.
"""

import json
import os
from collections.abc import Callable
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

from fastapi import FastAPI, Request
from fastapi.responses import Response
from pydantic import BaseModel
from starlette.exceptions import HTTPException

from eurohoops.api import readmodel
from eurohoops.api.readmodel import NotFound, Store

TAGS = [
    {"name": "site", "description": "The front page's data: forecasts, results, scorecard, Elo."},
    {"name": "stats", "description": "The EuroLeague stats-site files (players, teams, shots)."},
    {"name": "games", "description": "Upcoming games and every logged forecast of a game."},
    {"name": "teams", "description": "Elo ratings, four factors and schedule strength."},
    {"name": "players", "description": "Player seasons, impact, M6 projections and similarity."},
    {
        "name": "scouting",
        "description": "The M6 over/under board, undervalued list and M4 factors.",
    },
    {
        "name": "simulations",
        "description": "The ungated M7 season simulation and its gate verdict.",
    },
    {"name": "metrics", "description": "Live scorecards of the pre-registered forecasts."},
]


class CompactJSON(Response):
    """``json.dumps(..., ensure_ascii=False, separators=(",", ":")) + "\\n"``, UTF-8."""

    media_type = "application/json"

    def render(self, content: Any) -> bytes:
        text = json.dumps(content, ensure_ascii=False, separators=(",", ":"))
        return (text + "\n").encode("utf-8")


class IndentedJSON(Response):
    """``logs.write_json``'s form: ``json.dumps(payload, indent=2) + "\\n"``."""

    media_type = "application/json"

    def render(self, content: Any) -> bytes:
        return (json.dumps(content, indent=2) + "\n").encode("utf-8")


class NotFoundBody(BaseModel):
    detail: str


def create_app(store: Store, now: Callable[[], datetime]) -> FastAPI:
    app = FastAPI(
        title="EuroHoops read-only API",
        version=version("eurohoops"),
        description="EuroLeague and Greek Basket League analytics. Read-only: every response is "
        "read from the marts, the logs and the committed reports; nothing is computed or fitted.",
        openapi_tags=TAGS,
    )
    missing: dict[int | str, dict[str, Any]] = {
        404: {"model": NotFoundBody, "description": "Not found; the body says why."}
    }

    def not_found(_: Request, error: Exception) -> Response:
        missing_reason = cast(NotFound, error)  # registered for NotFound only
        extra = readmodel.jsonable(missing_reason.extra)
        return CompactJSON({"detail": missing_reason.reason, **extra}, 404)

    def http_error(_: Request, error: Exception) -> Response:
        failure = cast(HTTPException, error)  # registered for HTTPException only
        return CompactJSON({"detail": failure.detail}, failure.status_code)

    app.add_exception_handler(NotFound, not_found)
    app.add_exception_handler(HTTPException, http_error)

    @app.get("/site", tags=["site"], response_class=IndentedJSON, responses=missing)
    def site() -> Response:
        """``site.json``: forecasts, results, scorecard and Elo of both competitions."""
        return IndentedJSON(readmodel.site(store, now()))

    @app.get("/stats", tags=["stats"], response_class=CompactJSON)
    def stats_index() -> Response:
        """The relative path of every stats file."""
        return CompactJSON(readmodel.stats_index(store, now()))

    @app.get("/stats/{path:path}", tags=["stats"], response_class=CompactJSON, responses=missing)
    def stats_file(path: str) -> Response:
        """One stats file, e.g. ``meta.json`` or ``seasons/2025/players.json``."""
        return CompactJSON(readmodel.stats_file(store, path, now()))

    @app.get("/games/upcoming", tags=["games"], response_class=CompactJSON, responses=missing)
    def upcoming_games(competition: str = "euroleague") -> Response:
        """Unplayed live-season games after now, with their pre-registered forecast."""
        return CompactJSON(readmodel.upcoming_games(store, competition, now()))

    @app.get(
        "/predictions/{game_id}", tags=["games"], response_class=CompactJSON, responses=missing
    )
    def prediction(game_id: str) -> Response:
        """Every logged forecast of a game per model: the pre-registered row, then later rows."""
        return CompactJSON(readmodel.prediction(store, game_id))

    @app.get(
        "/teams/{competition}/{team}/ratings",
        tags=["teams"],
        response_class=CompactJSON,
        responses=missing,
    )
    def team_ratings(competition: str, team: str) -> Response:
        """The team's published Elo rating, change and rank (source team code)."""
        return CompactJSON(readmodel.team_ratings(store, competition, team))

    @app.get(
        "/teams/{competition}/{team}/factors",
        tags=["teams"],
        response_class=CompactJSON,
        responses=missing,
    )
    def team_factors(competition: str, team: str, season: int | None = None) -> Response:
        """Four factors and schedule strength of a season (the latest with box scores)."""
        return CompactJSON(readmodel.team_factors(store, competition, team, season))

    @app.get(
        "/players/{person_id}", tags=["players"], response_class=CompactJSON, responses=missing
    )
    def player(person_id: str) -> Response:
        """A person's seasons, M3 impact snapshots and M2 shot-making."""
        return CompactJSON(readmodel.player(store, person_id))

    @app.get(
        "/players/{person_id}/projection",
        tags=["players"],
        response_class=CompactJSON,
        responses=missing,
    )
    def player_projection(person_id: str) -> Response:
        """The M6 projection of a person (a committed report; nothing is fitted here)."""
        return CompactJSON(readmodel.player_projection(store, person_id))

    @app.get(
        "/players/{person_id}/similar",
        tags=["players"],
        response_class=CompactJSON,
        responses=missing,
    )
    def player_similar(person_id: str) -> Response:
        """The players whose shot and box profile is closest to this person's."""
        return CompactJSON(readmodel.player_similar(store, person_id))

    @app.get("/scouting/board", tags=["scouting"], response_class=CompactJSON, responses=missing)
    def scouting_board() -> Response:
        """The M6 over/under board."""
        return CompactJSON(readmodel.scouting_board(store))

    @app.get(
        "/scouting/undervalued", tags=["scouting"], response_class=CompactJSON, responses=missing
    )
    def scouting_undervalued() -> Response:
        """Young GBL players whose projection translates well to the EuroLeague."""
        return CompactJSON(readmodel.scouting_undervalued(store))

    @app.get(
        "/scouting/translation", tags=["scouting"], response_class=CompactJSON, responses=missing
    )
    def scouting_translation() -> Response:
        """M4's GBL to EuroLeague translation factors and its gate verdict."""
        return CompactJSON(readmodel.scouting_translation(store))

    @app.get(
        "/simulations/latest", tags=["simulations"], response_class=CompactJSON, responses=missing
    )
    def simulation_latest(competition: str = "euroleague") -> Response:
        """The latest season simulation, ``gated: false``, with M7's gate verdict. 404 (with the
        verdict) when no ungated report exists."""
        return CompactJSON(readmodel.simulation_latest(store, competition))

    @app.get("/metrics/live", tags=["metrics"], response_class=CompactJSON, responses=missing)
    def metrics_live() -> Response:
        """The live scorecards of the pre-registered forecasts (Elo, B0, M1 and M5 shadow)."""
        return CompactJSON(readmodel.metrics_live(store))

    return app


def local_app() -> FastAPI:
    """The app on the local tree: ``EUROHOOPS_ROOT`` (default the working directory) and, when
    ``EUROHOOPS_FROM_CACHE`` is set, the stats routes read the cached schedules."""
    root = Path(os.environ.get("EUROHOOPS_ROOT", "."))
    store = Store(root, from_cache=bool(os.environ.get("EUROHOOPS_FROM_CACHE")))
    return create_app(store, lambda: datetime.now(UTC))
