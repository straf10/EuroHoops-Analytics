# The read-only API

EuroHoops serves its data through a small FastAPI app (`src/eurohoops/api/`). It is **read-only**:
every response is read from the DuckDB marts, the prediction logs and the committed reports. Nothing
is computed, fitted or written when a route answers. The API is **not deployed publicly**. You run it
locally with Docker (below), and the static site is built from its answers.

## Routes

The table lists the route, the read-model function behind it (`api/readmodel.py`) and what it reads.
Team codes are the **source** codes (`ULK`, `PAN`, …). Person ids look like `P:003733`.

| Route | Read model | Source |
|---|---|---|
| `GET /site` | `site` | the forecasts, results, scorecards and Elo of both competitions (today's `site.json`) |
| `GET /stats` | `stats_index` | the path of every stats file |
| `GET /stats/{path}` | `stats_file` | one stats file, e.g. `meta.json`, `seasons/2025/players.json` |
| `GET /games/upcoming?competition=` | `upcoming_games` | unplayed live-season games with their pre-registered forecast |
| `GET /predictions/{game_id}` | `prediction` | every logged forecast of a game per model: the pre-registered row, then later rows |
| `GET /teams/{competition}/{team}/ratings` | `team_ratings` | published Elo rating, change and rank |
| `GET /teams/{competition}/{team}/factors?season=` | `team_factors` | four factors and schedule strength. A missing input is `value: null` with a `reason` |
| `GET /players/{person_id}` | `player` | the person's seasons, M3 impact (RAPM/SPM with intervals) and M2 shot-making |
| `GET /players/{person_id}/projection` | `player_projection` | `reports/m6_projections.json` |
| `GET /players/{person_id}/similar` | `player_similar` | `reports/m6_similarity.json` |
| `GET /scouting/board` | `scouting_board` | `reports/m6_board.json` (the over/under board) |
| `GET /scouting/undervalued` | `scouting_undervalued` | the `undervalued` list in `reports/m6_projections.json` |
| `GET /scouting/translation` | `scouting_translation` | M4's factors and its gate verdict (it **failed** its gate) |
| `GET /simulations/latest?competition=` | `simulation_latest` | `reports/sim_ungated_{competition}.json`, always `gated: false` with M7's gate verdict and reason |
| `GET /metrics/live` | `metrics_live` | the live scorecards: Elo, the baseline, M1 and the M5 shadow |
| `GET /openapi.json` | FastAPI | the schema; identical across runs |

A route with nothing to return answers **404** with `{"detail": "<reason>"}`. The simulation route
also adds M7's gate verdict to its 404 body. No birth date or age is in any response.

## Running it locally

```bash
docker compose up            # builds the image, then serves http://localhost:8000
# http://localhost:8000/docs  is the interactive schema
```

`docker-compose.yml` mounts `data/`, `reports/` and `predictions/` **read-only** into a read-only
container, so the API cannot write a file even by mistake. `data/` is not in the repository, so run
the pipeline first (`uv run eurohoops ingest`, then `build`; see the README).

Without Docker:

```bash
uv run uvicorn --factory eurohoops.api.app:local_app --port 8000
```

The first `/players/...` request builds the player-season frame from the marts, and the frame is
then kept for later requests. That first request took about 10 s on the local tree and about a
minute in Docker on a Windows bind mount; later ones answer in well under a second.

`EUROHOOPS_ROOT` sets the tree the API reads (the default is the working directory).
`EUROHOOPS_FROM_CACHE=1` makes the stats routes read the cached schedules.

## The site is built from the API

`eurohoops publish` and `eurohoops export-stats` are thin clients (`api/export.py`). They call each
route the site reads through FastAPI's in-process `TestClient`, with no server and no network. They
write the response bytes unchanged:

| Route | File, relative to `web/` |
|---|---|
| `/site` | `src/data/site.json` |
| `/stats/{path}` | `src/data/stats/{path}` |
| `/players/{id}`, `/projection`, `/similar` | `src/data/api/players/{key}/index.json`, `projection.json`, `similar.json` (`key` = id with `:` → `_`) |
| `/scouting/{board,undervalued,translation}` | `src/data/api/scouting/{name}.json` |
| `/simulations/latest?competition=c` | `src/data/api/simulations/{c}.json` |
| `/teams/{c}/{team}/ratings`, `/factors?season=<live>` | `src/data/api/teams/{c}/{team}/ratings.json`, `factors.json` |
| `/metrics/live` | `src/data/api/metrics/live.json` |
| `/openapi.json` | `public/api/openapi.json` (published with the site) |

`file_for` and `route_for` in `api/export.py` are the only mapping between routes and files. Both
directories are build outputs and are never committed. The site loads every `api/` file as
optional: a route that answered 404 writes no file, and the page then shows "not available".

## Checks

- **Contract test** (`tests/test_api_contract.py`): it walks the exported files, not the route list.
  Every file under `web/src/data/` and `web/public/api/` must equal its route's response, byte for
  byte, so a file that no route produces fails the test.
- **Route and read-model tests**: `tests/test_api_app.py`, `tests/test_api_readmodel.py` and
  `tests/test_api_export.py` run on a fixture tree (`tests/api_tree.py`). They check every route,
  the 404 bodies, read-only access (no file changes during a request), and a stable OpenAPI schema.
- **Checklist item 59** (`scripts/checks/api_export.py`) runs on the real data. It checks that the
  old export and the export through the API are byte-identical, that the contract holds on the real
  output, that the export runtime is recorded, and that `docker compose config` is valid.

```bash
uv run pytest -q tests/test_api_app.py tests/test_api_readmodel.py tests/test_api_export.py tests/test_api_contract.py
```
