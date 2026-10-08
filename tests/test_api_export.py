"""The static export as a client of the API: the file<->route mapping, the errors, and that the
new export writes byte-for-byte the files the old ``publish`` and ``export-stats`` wrote."""

import shutil
from pathlib import Path, PurePosixPath
from typing import Any

import duckdb
import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from eurohoops import cli
from eurohoops.api.app import create_app
from eurohoops.api.export import (
    API_PREFIX,
    OPENAPI_FILE,
    SITE_FILE,
    STATS_PREFIX,
    ExportError,
    RawStore,
    Route,
    export_files,
    file_for,
    file_key,
    get,
    person_of,
    publish_files,
    route_for,
    stats_files,
)
from eurohoops.api.readmodel import Store
from eurohoops.config import EUROLEAGUE, LIVE_SEASON, MART_PATH
from tests.api_check import load_script
from tests.api_tree import EL_PERSON, GBL_ONLY_PERSON, NOW, build_tree

runner = CliRunner()
script = load_script()
ROUTES = [
    "/site",
    "/openapi.json",
    "/stats/meta.json",
    "/stats/seasons/2024/players.json",
    "/scouting/board",
    "/scouting/undervalued",
    "/scouting/translation",
    "/metrics/live",
    "/simulations/latest?competition=euroleague",
    "/simulations/latest?competition=gbl",
    f"/players/{EL_PERSON}",
    f"/players/{EL_PERSON}/projection",
    f"/players/{GBL_ONLY_PERSON}/similar",
    "/teams/euroleague/BAR/ratings",
    f"/teams/gbl/AAAA0001/factors?season={LIVE_SEASON}",
]


def client_for(root: Path, *, from_cache: bool = False) -> TestClient:
    return TestClient(create_app(Store(root, from_cache=from_cache), lambda: NOW))


@pytest.fixture
def tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The fixture tree as the working directory, with the CLI clock fixed."""
    build_tree(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "utc_now", lambda: NOW)
    return tmp_path


@pytest.mark.parametrize("route", ROUTES)
def test_the_file_of_a_route_maps_back_to_the_route(route: str) -> None:
    assert route_for(file_for(route)) == route


def test_the_files_of_the_routes() -> None:
    assert file_for("/site") == SITE_FILE == PurePosixPath("src/data/site.json")
    assert file_for("/openapi.json") == PurePosixPath("public/api/openapi.json")
    assert file_for("/stats/seasons/2024/players.json") == (
        STATS_PREFIX / "seasons" / "2024" / "players.json"
    )
    assert file_for("/scouting/board") == API_PREFIX / "scouting" / "board.json"
    assert file_for("/simulations/latest?competition=gbl") == API_PREFIX / "simulations/gbl.json"
    assert file_for("/players/P:003733/projection") == (
        API_PREFIX / "players" / "P_003733" / "projection.json"
    )
    assert file_for("/players/G:0000ABCD") == API_PREFIX / "players" / "G_0000ABCD" / "index.json"
    assert file_for(f"/teams/euroleague/PAM/factors?season={LIVE_SEASON}") == (
        API_PREFIX / "teams" / "euroleague" / "PAM" / "factors.json"
    )


def test_a_person_key_has_no_colon_and_round_trips() -> None:
    for person in ("P:003733", "G:0000ABCD", EL_PERSON):
        key = file_key(person)
        assert ":" not in key and "_" in key
        assert person_of(key) == person
    for bad in ("003733", "P:00_33", "P:../x", "PP:1"):
        with pytest.raises(ValueError, match="not a person id"):
            file_key(bad)


@pytest.mark.parametrize("route", ["/games/upcoming", "/players/P:1/other", "/teams/a/b/factors"])
def test_a_route_without_a_file_is_an_error(route: str) -> None:
    with pytest.raises(ValueError, match="no file for route"):
        file_for(route)


@pytest.mark.parametrize(
    "file", ["src/data/api/mystery.json", "src/data/other.json", "src/data/api/players/x.json"]
)
def test_a_file_without_a_route_is_an_error(file: str) -> None:
    with pytest.raises(ValueError, match="no route for file"):
        route_for(PurePosixPath(file))


def test_a_non_200_response_is_an_error_naming_the_route_and_the_body(tree: Path) -> None:
    client = client_for(tree)
    with pytest.raises(ExportError) as raised:
        get(client, "/teams/euroleague/NOPE/ratings")
    assert "GET /teams/euroleague/NOPE/ratings answered 404" in str(raised.value)
    assert "unknown team 'NOPE'" in str(raised.value)
    with pytest.raises(ExportError, match="answered 422"):
        get(client, "/teams/euroleague/BAR/factors?season=last")
    with pytest.raises(ExportError, match="answered 404"):
        export_files(client, [Route("/teams/euroleague/NOPE/ratings")])


def test_an_optional_404_writes_no_file_and_is_reported(tree: Path) -> None:
    reasons: list[str] = []
    files = export_files(
        client_for(tree),
        [
            Route("/simulations/latest?competition=gbl", optional=True),
            Route("/simulations/latest?competition=euroleague", optional=True),
        ],
        reasons.append,
    )
    assert list(files) == [API_PREFIX / "simulations" / "euroleague.json"]
    assert len(reasons) == 1 and "skipped /simulations/latest?competition=gbl" in reasons[0]
    assert "no ungated simulation report for gbl" in reasons[0]


def test_an_optional_route_that_fails_otherwise_still_raises(tree: Path) -> None:
    with pytest.raises(ExportError, match="answered 422"):
        export_files(
            client_for(tree), [Route("/teams/euroleague/BAR/factors?season=x", optional=True)]
        )


def _written(web: Path) -> dict[str, bytes]:
    return {p.relative_to(web).as_posix(): p.read_bytes() for p in web.rglob("*") if p.is_file()}


@pytest.mark.parametrize("from_cache", [True, False], ids=["from-cache", "marts"])
def test_the_new_export_is_byte_identical_to_the_old_one(tree: Path, from_cache: bool) -> None:
    old = tree / "old"
    script.legacy_site(old / SITE_FILE, NOW)
    script.legacy_stats(old / STATS_PREFIX, NOW, EUROLEAGUE.raw_dir, from_cache=from_cache)
    assert runner.invoke(cli.app, ["publish"]).exit_code == 0
    args = ["export-stats", "--out", str(tree / "web" / STATS_PREFIX)]
    assert runner.invoke(cli.app, [*args, *(["--from-cache"] if from_cache else [])]).exit_code == 0

    web = tree / "web"
    assert script.tree_differences(old, web) == []
    old_files = _written(old)
    assert len(old_files) == 15  # site.json and the 14 stats files
    new_files = _written(web)
    assert set(old_files) < set(new_files)  # the new export adds the api/ and openapi files
    assert all(new_files[name] == body for name, body in old_files.items())


def test_the_planted_difference_between_old_and_new_is_found(tmp_path: Path) -> None:
    old, new = tmp_path / "old", tmp_path / "new"
    for root in (old, new):
        (root / "a").mkdir(parents=True)
        (root / "a" / "x.json").write_bytes(b"{}\n")
    assert script.tree_differences(old, new) == []
    (new / "a" / "x.json").write_bytes(b'{"x":1}\n')
    assert script.tree_differences(old, new) == ["a/x.json differs between old and new"]
    (new / "a" / "x.json").unlink()
    assert script.tree_differences(old, new) == ["a/x.json is missing from the new export"]


def test_publish_writes_the_new_route_files(tree: Path) -> None:
    result = runner.invoke(cli.app, ["publish"])
    assert result.exit_code == 0, result.output
    assert "skipped /simulations/latest?competition=gbl" in result.output
    written = {p.relative_to(tree / "web").as_posix() for p in (tree / "web").rglob("*.json")}
    key = file_key(EL_PERSON)
    assert {
        "src/data/site.json",
        "public/api/openapi.json",
        "src/data/api/scouting/board.json",
        "src/data/api/scouting/undervalued.json",
        "src/data/api/scouting/translation.json",
        "src/data/api/metrics/live.json",
        "src/data/api/simulations/euroleague.json",
        f"src/data/api/players/{key}/projection.json",
        f"src/data/api/players/{key}/similar.json",
        f"src/data/api/players/{key}/index.json",
        "src/data/api/teams/euroleague/BAR/ratings.json",
        "src/data/api/teams/euroleague/BAR/factors.json",
        "src/data/api/teams/gbl/AAA/factors.json",
    } <= written
    assert "src/data/api/simulations/gbl.json" not in written  # the 404 writes no file
    assert not any(":" in name for name in written)
    gbl_only = file_key("G:ABCD1234")
    assert f"src/data/api/players/{gbl_only}/projection.json" in written
    assert f"src/data/api/players/{gbl_only}/index.json" not in written  # only EuroLeague persons


def test_publish_replaces_the_api_directory_and_the_openapi_file_is_stable(tree: Path) -> None:
    assert runner.invoke(cli.app, ["publish"]).exit_code == 0
    first = _written(tree / "web")
    stale = tree / "web" / API_PREFIX / "players" / "P_STALE" / "index.json"
    stale.parent.mkdir(parents=True)
    stale.write_text("{}")
    assert runner.invoke(cli.app, ["publish"]).exit_code == 0
    assert not stale.exists() and _written(tree / "web") == first  # incl. openapi.json bytes
    assert (tree / "web" / OPENAPI_FILE).read_bytes().endswith(b"}")


def test_export_stats_replaces_stale_season_files_and_honours_raw_dir(tree: Path) -> None:
    shutil.copytree(tree / EUROLEAGUE.raw_dir, tree / "elsewhere")
    shutil.rmtree(tree / "data" / "raw")
    out = tree / "stats_out"
    (out / "seasons" / "1999").mkdir(parents=True)
    (out / "seasons" / "1999" / "players.json").write_text("{}")
    args = ["export-stats", "--from-cache", "--raw-dir", "elsewhere", "--out", str(out)]
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    assert not (out / "seasons" / "1999").exists()
    assert "wrote 14 files" in result.output and "2 seasons" in result.output
    assert (out / "meta.json").read_bytes().endswith(b"}\n")


def test_a_failing_route_fails_the_command(tree: Path) -> None:
    (tree / "reports" / "live_scorecard.json").unlink()
    result = runner.invoke(cli.app, ["publish"])
    assert result.exit_code != 0 and isinstance(result.exception, ExportError)
    assert "GET /site answered 404" in str(result.exception)


def test_the_export_opens_duckdb_only_read_only(
    tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[dict[str, Any]] = []
    connect = duckdb.connect

    def spy(database: str, *args: Any, **kwargs: Any) -> Any:
        opened.append({"database": database, "read_only": kwargs.get("read_only")})
        return connect(database, *args, **kwargs)

    monkeypatch.setattr(duckdb, "connect", spy)
    mart_before = (tree / MART_PATH).read_bytes()
    store = Store(tree)
    publish_files(TestClient(create_app(store, lambda: NOW)), store)
    stats_files(TestClient(create_app(RawStore(tree), lambda: NOW)))
    assert opened and all(call["read_only"] is True for call in opened)
    assert (tree / MART_PATH).read_bytes() == mart_before
    assert not (tree / "web").exists()  # the builders return bytes; only the writers touch disk


def test_the_stats_from_the_cache_never_open_duckdb(
    tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[str] = []
    monkeypatch.setattr(duckdb, "connect", lambda *a, **k: opened.append("x"))
    stats_files(TestClient(create_app(RawStore(tree, from_cache=True), lambda: NOW)))
    assert opened == []


def test_one_clock_stamps_the_whole_run(tree: Path) -> None:
    store = Store(tree)
    files = publish_files(TestClient(create_app(store, lambda: NOW)), store)
    assert files[SITE_FILE].startswith(b'{\n  "generated_at_utc": "2026-10-08T08:00:00Z"')
