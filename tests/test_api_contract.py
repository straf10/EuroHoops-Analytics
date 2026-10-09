"""The contract: every file under ``web/src/data`` (and ``web/public/api``) is the API's response
bytes for the route it maps to. The check walks the output directory, not the route list, so a
file no route produces fails it."""

import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from eurohoops.api.app import create_app
from eurohoops.api.export import (
    API_PREFIX,
    OPENAPI_FILE,
    SITE_FILE,
    STATS_PREFIX,
    publish_files,
    stats_files,
    write_publish,
    write_stats_files,
)
from eurohoops.api.readmodel import Store
from tests.api_check import load_script
from tests.api_tree import NOW, build_tree

script = load_script()


@pytest.fixture(scope="module")
def exported(tmp_path_factory: pytest.TempPathFactory) -> tuple[TestClient, Path]:
    root = tmp_path_factory.mktemp("contract") / "tree"
    root.mkdir()
    build_tree(root)
    store = Store(root)
    client = TestClient(create_app(store, lambda: NOW))
    web = root.parent / "web"
    write_publish(web, publish_files(client, store))
    write_stats_files(web / STATS_PREFIX, stats_files(client))
    return client, web


def test_every_exported_file_is_the_response_of_its_route(
    exported: tuple[TestClient, Path],
) -> None:
    client, web = exported
    assert script.contract_mismatches(client, web) == []
    walked = [p for d in script.WALKED for p in (web / d).rglob("*") if p.is_file()]
    names = {p.relative_to(web).as_posix() for p in walked}
    assert SITE_FILE.as_posix() in names and OPENAPI_FILE.as_posix() in names
    assert any(n.startswith(API_PREFIX.as_posix() + "/players/") for n in names)
    assert any(n.startswith(API_PREFIX.as_posix() + "/teams/") for n in names)
    assert any(n.startswith(STATS_PREFIX.as_posix() + "/seasons/") for n in names)


def test_a_planted_mismatch_is_detected(exported: tuple[TestClient, Path], tmp_path: Path) -> None:
    client, web = exported
    copy = tmp_path / "web"
    shutil.copytree(web, copy)
    board = copy / API_PREFIX / "scouting" / "board.json"
    board.write_bytes(board.read_bytes().replace(b"within noise", b"WITHIN noise", 1))
    site = copy / SITE_FILE
    site.write_bytes(site.read_bytes() + b" ")
    assert script.contract_mismatches(client, copy) == [
        "src/data/api/scouting/board.json differs from GET /scouting/board",
        "src/data/site.json differs from GET /site",
    ]


def test_a_file_that_maps_to_no_route_fails(
    exported: tuple[TestClient, Path], tmp_path: Path
) -> None:
    client, _ = exported
    (tmp_path / API_PREFIX).mkdir(parents=True)
    (tmp_path / API_PREFIX / "mystery.json").write_text("{}")
    assert script.contract_mismatches(client, tmp_path) == [
        "no route for file 'src/data/api/mystery.json'"
    ]
