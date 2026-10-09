"""The read-only API served over HTTP (FastAPI's TestClient) on the tree of ``tests/api_tree.py``.

Per route: status, shape and a hand-checked value (the numbers themselves are checked against the
raw fixtures in ``test_api_readmodel.py``); the 404 bodies; byte-identity with what the pipeline
writes today (``write_stats`` for ``/stats``, ``logs.write_json`` through ``eurohoops publish``
for ``/site``); a stable OpenAPI document; and that no route writes anything.
"""

import hashlib
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from eurohoops import cli
from eurohoops.api import app as api_app
from eurohoops.api.app import CompactJSON, IndentedJSON, create_app
from eurohoops.api.readmodel import Store
from eurohoops.config import EUROLEAGUE, MART_PATH, SITE_DATA
from eurohoops.logs import write_json
from eurohoops.publish import DISPLAY_CODES
from eurohoops.stats.box import build_box_games
from eurohoops.stats.export import Inputs, build_payloads, load_cached_games, write_stats
from eurohoops.stats.shots import build_shots
from tests.api_tree import EL_PERSON, GBL_ONLY_PERSON, NOW, build_tree
from tests.conftest import FIXTURES, REPO

STATS_NOW = datetime(2026, 9, 26, 8, tzinfo=UTC)  # the clock the committed stats fixture used
runner = CliRunner()


def make_client(root: Path, *, from_cache: bool = False, now: datetime = NOW) -> TestClient:
    return TestClient(create_app(Store(root, from_cache=from_cache), lambda: now))


@pytest.fixture(scope="module")
def root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tree = tmp_path_factory.mktemp("api") / "tree"
    tree.mkdir()
    build_tree(tree)
    return tree


@pytest.fixture(scope="module")
def client(root: Path) -> TestClient:
    return make_client(root)


OK_ROUTES = [
    "/site",
    "/stats",
    "/stats/meta.json",
    "/stats/seasons/2024/players.json",
    "/games/upcoming",
    "/games/upcoming?competition=gbl",
    "/predictions/E2026_2",
    "/teams/euroleague/BAR/ratings",
    "/teams/euroleague/LJU/factors?season=2010",
    "/teams/gbl/AAAA0001/factors",
    f"/players/{EL_PERSON}",
    f"/players/{EL_PERSON}/projection",
    f"/players/{EL_PERSON}/similar",
    "/scouting/board",
    "/simulations/latest",
    "/metrics/live",
    "/openapi.json",
]
NOT_FOUND_ROUTES = [
    ("/stats/seasons/1999/players.json", "no stats file 'seasons/1999/players.json'"),
    ("/games/upcoming?competition=nba", "unknown competition 'nba'"),
    ("/predictions/E2026_5", "no prediction is logged for game 'E2026_5'"),
    ("/teams/euroleague/XXX/ratings", "unknown team 'XXX' in euroleague"),
    ("/teams/euroleague/ASV/ratings", "'ASV' has no game in the euroleague 2026 season"),
    ("/teams/acb/BAR/factors", "unknown competition 'acb'"),
    ("/teams/gbl/XXX/factors", "unknown team 'XXX' in gbl"),
    ("/players/P:NOBODY", "unknown person_id 'P:NOBODY'"),
    ("/players/P:NOBODY/projection", "no M6 projection for person_id 'P:NOBODY'"),
    ("/players/P:NOBODY/similar", "no similarity list for person_id 'P:NOBODY'"),
    ("/simulations/latest?competition=nba", "unknown competition 'nba'"),
    ("/nowhere", "Not Found"),
]


@pytest.mark.parametrize("path", OK_ROUTES)
def test_a_route_answers_in_the_pipelines_bytes(client: TestClient, path: str) -> None:
    response = client.get(path)
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    payload = json.loads(response.content)
    if path == "/site":
        expected = json.dumps(payload, indent=2) + "\n"  # logs.write_json's form
    elif path == "/openapi.json":
        return  # FastAPI's own serializer; stability is tested below
    else:
        expected = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
    assert response.content == expected.encode("utf-8")


@pytest.mark.parametrize(("path", "reason"), NOT_FOUND_ROUTES)
def test_a_404_body_is_a_detail_with_a_reason(client: TestClient, path: str, reason: str) -> None:
    response = client.get(path)
    assert response.status_code == 404
    body = json.loads(response.content)
    assert body == {"detail": body["detail"]} and reason in body["detail"]
    assert response.content == (json.dumps(body, separators=(",", ":")) + "\n").encode("utf-8")


def test_the_compact_and_indented_responses_keep_unicode_and_end_with_a_newline() -> None:
    compact = CompactJSON({"n": "Ολυμπιακός", "xs": [1, 2]}).body
    assert compact == '{"n":"Ολυμπιακός","xs":[1,2]}\n'.encode()
    assert IndentedJSON({"n": "Ολ"}).body == b'{\n  "n": "\\u039f\\u03bb"\n}\n'


def test_the_stats_listing_and_a_bad_query_value(client: TestClient) -> None:
    files = client.get("/stats").json()["files"]
    assert "meta.json" in files and "seasons/2024/shots.json" in files
    assert files == sorted(files)
    response = client.get("/teams/euroleague/BAR/factors?season=last")
    assert response.status_code == 422


def test_route_values(client: TestClient) -> None:
    upcoming = client.get("/games/upcoming").json()
    assert upcoming["games"][0]["game_id"] == "E2026_4"  # the default competition is euroleague
    assert (
        client.get("/games/upcoming?competition=gbl").json()["games"][0]["game_id"] == "GBL2026_4"
    )
    models = client.get("/predictions/E2026_4").json()["models"]
    assert (
        models["elo"]["pre_registered"]["p_home"],
        models["m1"]["pre_registered"]["p_home"],
    ) == (
        0.61,
        0.63,
    )
    ratings = client.get("/teams/euroleague/BAR/ratings").json()
    assert ratings["elo"]["rank"] == 1 and ratings["team"]["display_code"] == "BAR"
    factors = client.get("/teams/euroleague/LJU/factors?season=2010").json()
    assert factors["factors"]["efg_pct"]["value"] == pytest.approx(38 / 78)  # 26 + 1.5 * 8 over 78
    player = client.get(f"/players/{EL_PERSON}").json()
    assert player["name"] == "Levi Randolph" and len(player["seasons"]) == 1
    assert client.get(f"/players/{GBL_ONLY_PERSON}").status_code == 404  # not on the site
    assert client.get(f"/players/{EL_PERSON}/projection").json()["projections"][0]["team"] == "TEL"
    similar = client.get(f"/players/{EL_PERSON}/similar").json()["similar"]
    assert [s["rank"] for s in similar] == [1]  # the GBL row is not published
    assert client.get("/scouting/board").json()["rows"][0]["label"] == "within noise"
    metrics = client.get("/metrics/live").json()["competitions"]
    assert metrics[0]["scorecard"]["elo"]["log_loss"] == 0.61
    meta = client.get("/stats/meta.json").json()
    assert meta["generated_at"] == "2026-10-08T08:00Z" and meta["teams"]["PAO"]


def test_the_simulation_is_ungated_with_the_m7_verdict(client: TestClient) -> None:
    response = client.get("/simulations/latest")
    assert response.status_code == 200
    body = response.json()
    assert body["gated"] is False and body["teams"][0]["team"] == "BAR"
    assert body["gate"]["passed"] is False
    assert body["gate"]["reason"].startswith("M7 failed its validation gate: validation Brier")
    assert client.get("/simulations/latest?competition=gbl").status_code == 404


def test_without_an_ungated_report_the_404_carries_the_verdict(tmp_path: Path) -> None:
    build_tree(tmp_path, ungated=False)
    response = make_client(tmp_path).get("/simulations/latest")
    assert response.status_code == 404
    body = response.json()
    assert body["gated"] is False and body["gate"]["passed"] is False
    assert body["detail"] == (
        "no ungated simulation report for euroleague: M7 failed its validation gate: "
        "validation Brier 0.0920 is not below the point rule's 0.0883"
    )
    assert body["gate"]["reason"] in body["detail"]


def test_a_missing_report_is_a_404_naming_the_file(tmp_path: Path) -> None:
    build_tree(tmp_path)
    (tmp_path / "reports" / "m6_board.json").unlink()
    response = make_client(tmp_path).get("/scouting/board")
    assert response.status_code == 404
    assert "m6_board.json not found: the M6 over/under board" in response.json()["detail"]


def _independent_stats(raw: Path, now: datetime) -> dict[str, dict[str, Any]]:
    """The payloads built the way ``cli export-stats --from-cache`` builds them."""
    games, teams = load_cached_games(raw)
    return build_payloads(
        Inputs(
            games=games,
            names=dict(teams.itertuples(index=False)),
            box=build_box_games(raw, games),
            shots=build_shots(raw, games),
            codes=DISPLAY_CODES["euroleague"],
            live_season=2026,
        ),
        now,
    )


@pytest.mark.parametrize("from_cache", [True, False], ids=["from-cache", "marts"])
def test_stats_responses_are_the_bytes_write_stats_writes(
    root: Path, tmp_path: Path, from_cache: bool
) -> None:
    client = make_client(root, from_cache=from_cache, now=STATS_NOW)
    written = tmp_path / "written"
    write_stats(written, _independent_stats(root / EUROLEAGUE.raw_dir, STATS_NOW))
    paths = sorted(p.relative_to(written).as_posix() for p in written.rglob("*.json"))
    assert client.get("/stats").json()["files"] == paths and len(paths) == 14
    for path in paths:
        response = client.get(f"/stats/{path}")
        assert response.content == (written / path).read_bytes(), path
        # and the file the committed stats fixture holds for the same raw slice
        assert response.content == (FIXTURES / "stats" / path).read_bytes(), path


def test_site_is_the_bytes_eurohoops_publish_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    build_tree(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "utc_now", lambda: NOW)
    assert runner.invoke(cli.app, ["publish"]).exit_code == 0
    written = (tmp_path / SITE_DATA).read_bytes()
    assert make_client(tmp_path).get("/site").content == written
    assert written.startswith(b'{\n  "generated_at_utc": "2026-10-08T08:00:00Z"')
    again = tmp_path / "again.json"
    write_json(again, json.loads(written))
    assert again.read_bytes() == written  # logs.write_json's form, the one /site returns


def _snapshot(root: Path) -> dict[str, tuple[int, int, str]]:
    return {
        p.relative_to(root).as_posix(): (
            p.stat().st_size,
            p.stat().st_mtime_ns,
            hashlib.sha256(p.read_bytes()).hexdigest(),
        )
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def _directories(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_dir())


def test_no_route_writes_and_duckdb_is_only_opened_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    build_tree(tmp_path)
    opened: list[dict[str, Any]] = []
    connect = duckdb.connect

    def spy(database: str, *args: Any, **kwargs: Any) -> Any:
        opened.append({"database": database, "read_only": kwargs.get("read_only")})
        return connect(database, *args, **kwargs)

    monkeypatch.setattr(duckdb, "connect", spy)
    before, directories = _snapshot(tmp_path), _directories(tmp_path)
    mart_before = (tmp_path / MART_PATH).read_bytes()
    client = make_client(tmp_path)
    for path in [*OK_ROUTES, *(p for p, _ in NOT_FOUND_ROUTES)]:
        client.get(path)
    for path in client.get("/stats").json()["files"]:
        assert client.get(f"/stats/{path}").status_code == 200
    assert opened and all(call["read_only"] is True for call in opened)
    assert {call["database"] for call in opened} == {str(tmp_path / MART_PATH)}
    assert _snapshot(tmp_path) == before  # sizes, mtimes and bytes of every file
    assert _directories(tmp_path) == directories
    assert (tmp_path / MART_PATH).read_bytes() == mart_before
    assert not list((tmp_path / MART_PATH).parent.glob("*.wal"))
    with (
        duckdb.connect(str(tmp_path / MART_PATH), read_only=True) as con,
        pytest.raises(duckdb.Error),
    ):
        con.execute("CREATE TABLE intruder AS SELECT 1")  # what read_only=True guarantees


def test_stats_from_the_cache_never_open_duckdb(
    root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shutil.copytree(root / "data" / "raw", tmp_path / "data" / "raw")  # the daily workflow's tree
    opened: list[str] = []
    monkeypatch.setattr(duckdb, "connect", lambda *a, **k: opened.append("x"))
    client = make_client(tmp_path, from_cache=True)
    assert client.get("/stats/meta.json").status_code == 200
    assert client.get("/stats/seasons/2024/teams.json").status_code == 200
    assert opened == []
    upcoming = client.get("/games/upcoming")
    assert upcoming.status_code == 404 and "the marts are not built" in upcoming.json()["detail"]
    assert opened == []  # a missing mart is reported before any connection is tried


def test_the_openapi_document_is_stable_across_instances(root: Path) -> None:
    apps = [create_app(Store(root), lambda: NOW) for _ in range(2)]
    assert json.dumps(apps[0].openapi()) == json.dumps(apps[1].openapi())
    first, second = (TestClient(a).get("/openapi.json").content for a in apps)
    assert first == second


def test_the_openapi_document_lists_every_route_under_its_tag(client: TestClient) -> None:
    document = client.get("/openapi.json").json()
    tags = [t["name"] for t in document["tags"]]
    assert tags == [
        "site",
        "stats",
        "games",
        "teams",
        "players",
        "scouting",
        "simulations",
        "metrics",
    ]
    expected = {
        "/site": "site",
        "/stats": "stats",
        "/stats/{path}": "stats",
        "/games/upcoming": "games",
        "/predictions/{game_id}": "games",
        "/teams/{competition}/{team}/ratings": "teams",
        "/teams/{competition}/{team}/factors": "teams",
        "/players/{person_id}": "players",
        "/players/{person_id}/projection": "players",
        "/players/{person_id}/similar": "players",
        "/scouting/board": "scouting",
        "/simulations/latest": "simulations",
        "/metrics/live": "metrics",
    }
    assert {
        p: next(iter(op.values()))["tags"][0] for p, op in document["paths"].items()
    } == expected
    assert all(set(op) == {"get"} for op in document["paths"].values())  # read-only: GET only
    for path, operations in document["paths"].items():
        if path != "/stats":
            assert "404" in operations["get"]["responses"], path
    ids = [op["get"]["operationId"] for op in document["paths"].values()]
    assert len(set(ids)) == len(ids)


def test_a_write_method_is_not_routed(client: TestClient) -> None:
    for method in ("post", "put", "delete", "patch"):
        assert getattr(client, method)("/site").status_code == 405


def test_the_uvicorn_factory_reads_the_tree_from_the_environment(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EUROHOOPS_ROOT", str(root))
    monkeypatch.setenv("EUROHOOPS_FROM_CACHE", "1")
    app = api_app.local_app()
    assert isinstance(app, FastAPI)
    client = TestClient(app)
    meta = client.get("/stats/meta.json")
    assert meta.status_code == 200 and meta.json()["seasons"][0]["season"] == 2010
    assert client.get("/metrics/live").status_code == 200


def test_the_compose_file_runs_the_api_on_the_local_data_read_only() -> None:
    compose = (REPO / "docker-compose.yml").read_text(encoding="utf-8")
    for needle in (
        "services:",
        "  api:",
        "    build: .",
        '      - "8000:8000"',
        "      - ./data:/srv/eurohoops/data:ro",
        "      - ./reports:/srv/eurohoops/reports:ro",
        "      - ./predictions:/srv/eurohoops/predictions:ro",
        "    read_only: true",
    ):
        assert needle in compose.splitlines(), needle
    dockerfile = (REPO / "Dockerfile").read_text(encoding="utf-8")
    assert "EXPOSE 8000" in dockerfile
    assert '"eurohoops.api.app:local_app"' in dockerfile and '"--factory"' in dockerfile
    assert "uv sync --frozen --no-dev" in dockerfile
    ignored = (REPO / ".dockerignore").read_text(encoding="utf-8").split()
    assert {"data", ".git", ".venv"} <= set(ignored)  # the data is mounted, never baked in


@pytest.mark.skipif(shutil.which("docker") is None, reason="docker is not installed")
def test_docker_compose_config_is_valid() -> None:
    done = subprocess.run(
        ["docker", "compose", "-f", str(REPO / "docker-compose.yml"), "config", "--quiet"],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO,
    )
    assert done.returncode == 0, done.stderr
