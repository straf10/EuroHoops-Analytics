"""Checklist item (weeks 16-18 L10): the static export through the API equals the old export.

Run from the repository root on the real data (never writes into ``web/``):

    uv run python scripts/checks/api_export.py [--stats-mode cache|marts|both]

1. builds the OLD outputs (``site.json`` as the pre-L10 ``eurohoops publish`` assembled it,
   the stats files as ``build_payloads`` + ``write_stats`` wrote them) into a temporary directory;
2. builds the NEW outputs (``publish_files``, ``stats_files``: the API's response bytes) into a
   second one, with the same clock;
3. compares every old file with the new file at the same path, byte for byte (the new tree
   also holds ``src/data/api/`` and ``public/api/openapi.json``, which the old export lacked);
4. the contract check: every file under the new tree's ``src/data`` and ``public/api`` equals
   the API's response for the route it maps to;
5. prints ``RUNTIME export N`` (seconds for the new export of everything, incl. writing);
6. ``docker compose config -q``: PASS, FAIL, or BLOCKED when no docker binary is on the PATH.

Exits non-zero on any difference, contract mismatch or invalid compose file.
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd
from fastapi.testclient import TestClient

from eurohoops.api.app import create_app
from eurohoops.api.export import (
    DATA_DIR,
    OPENAPI_FILE,
    SITE_FILE,
    STATS_PREFIX,
    RawStore,
    get,
    publish_files,
    route_for,
    stats_files,
    write_publish,
    write_stats_files,
)
from eurohoops.api.readmodel import FORECAST_MODELS, Store
from eurohoops.cli import _live, utc_now
from eurohoops.config import EUROLEAGUE, GBL, LIVE_SEASON, MART_PATH
from eurohoops.eval.forecasts import build_forecasts
from eurohoops.logs import write_json
from eurohoops.marts import read_games, read_teams
from eurohoops.publish import DISPLAY_CODES, Section, site_data
from eurohoops.stats.box import build_box_games
from eurohoops.stats.export import Inputs, build_payloads, load_cached_games, write_stats
from eurohoops.stats.shots import build_shots

WALKED = (DATA_DIR, OPENAPI_FILE.parent)  # the directories the contract check walks, under web/


def legacy_site(out: Path, now: datetime) -> None:
    """``site.json`` as ``eurohoops publish`` built it before L10 (paths relative to the cwd)."""
    sections = []
    for title, comp in (("EuroLeague", EUROLEAGUE), ("Greek Basket League", GBL)):
        live = _live(comp)
        sections.append(
            Section(
                key=comp.name,
                title=title,
                log=(
                    pd.read_csv(comp.prediction_log, dtype={"game_id": str})
                    if comp.prediction_log.exists()
                    else pd.DataFrame()
                ),
                scorecard=_json(comp.scorecard),
                backtest=_json(comp.live_backtest.report),
                games=live.games,
                names=dict(read_teams(MART_PATH, comp.name).itertuples(index=False)),
                model=live.model,
                season=live.season,
                replay_from=live.replay_from,
                forecasts=build_forecasts(
                    {
                        "elo": comp.prediction_log,
                        "m1": comp.m1_prediction_log,
                        "m5": comp.m5_prediction_log,
                    },
                    FORECAST_MODELS,
                    live.games,
                    live.season,
                    comp.manual_pushes,
                ),
            )
        )
    write_json(out, site_data(sections, now))


def _json(path: Path) -> dict[str, Any]:
    report: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return report


def legacy_stats(out: Path, now: datetime, raw_dir: Path, *, from_cache: bool) -> None:
    """The stats directory as ``eurohoops export-stats`` built it before L10."""
    if from_cache:
        games, teams = load_cached_games(raw_dir)
    else:
        games = read_games(MART_PATH, EUROLEAGUE.name)
        teams = read_teams(MART_PATH, EUROLEAGUE.name)
    inputs = Inputs(
        games=games,
        names=dict(teams.itertuples(index=False)),
        box=build_box_games(raw_dir, games),
        shots=build_shots(raw_dir, games),
        codes=DISPLAY_CODES[EUROLEAGUE.name],
        live_season=LIVE_SEASON,
    )
    write_stats(out, build_payloads(inputs, now))


def contract_mismatches(client: TestClient, web: Path) -> list[str]:
    """Every file under ``web``'s ``src/data`` and ``public/api`` that is not its route's response
    (found by walking the directories, not by iterating the routes); one line per file."""
    problems = []
    for directory in WALKED:
        for path in sorted(p for p in (web / directory).rglob("*") if p.is_file()):
            relative = PurePosixPath(path.relative_to(web).as_posix())
            try:
                route = route_for(relative)
            except ValueError as error:
                problems.append(str(error))
                continue
            if path.read_bytes() != get(client, route):
                problems.append(f"{relative} differs from GET {route}")
    return problems


def tree_differences(old: Path, new: Path) -> list[str]:
    """Every file under ``old`` that is missing from or different in ``new`` (byte for byte)."""
    problems = []
    for path in sorted(p for p in old.rglob("*") if p.is_file()):
        relative = path.relative_to(old)
        other = new / relative
        if not other.exists():
            problems.append(f"{relative.as_posix()} is missing from the new export")
        elif path.read_bytes() != other.read_bytes():
            problems.append(f"{relative.as_posix()} differs between old and new")
    return problems


def docker_compose() -> str:
    """``PASS``, ``FAIL`` or ``BLOCKED`` (no docker binary) for ``docker compose config -q``."""
    if shutil.which("docker") is None:
        return "BLOCKED"
    run = subprocess.run(
        ["docker", "compose", "config", "-q"],
        capture_output=True,
        text=True,
        check=False,
        cwd=Path(__file__).resolve().parents[2],
    )
    if run.returncode:
        print(run.stderr.strip())
    return "FAIL" if run.returncode else "PASS"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else "")
    parser.add_argument("--stats-mode", choices=("cache", "marts", "both"), default="cache")
    modes = {"cache": [True], "marts": [False], "both": [True, False]}
    cache_modes = modes[parser.parse_args().stats_mode]
    now = utc_now()
    failures: list[str] = []
    with tempfile.TemporaryDirectory() as scratch:
        old, new = Path(scratch, "old"), Path(scratch, "new")
        legacy_site(old / SITE_FILE, now)
        for from_cache in cache_modes:
            legacy_stats(old / STATS_PREFIX, now, EUROLEAGUE.raw_dir, from_cache=from_cache)
            started = time.perf_counter()
            site_store = Store()
            skipped: list[str] = []
            files = publish_files(
                TestClient(create_app(site_store, lambda: now)), site_store, skipped.append
            )
            stats_store = RawStore(from_cache=from_cache)
            stats = stats_files(TestClient(create_app(stats_store, lambda: now)))
            write_publish(new, files)
            write_stats_files(new / STATS_PREFIX, stats)
            seconds = time.perf_counter() - started
            mode = "from-cache" if from_cache else "marts"
            differences = tree_differences(old, new)
            print(f"stats {mode}: {len(stats)} stats files, {len(files)} publish files")
            print(f"skipped routes: {len(skipped)}")
            for line in skipped[:10]:
                print(f"  {line}")
            failures += [f"[{mode}] {line}" for line in differences]
            client = TestClient(create_app(stats_store, lambda: now))
            failures += [f"[{mode}] {line}" for line in contract_mismatches(client, new)]
            print(f"RUNTIME export {seconds:.0f}")
            shutil.rmtree(old / STATS_PREFIX)
            shutil.rmtree(new)
    compose = docker_compose()
    print(f"docker compose config -q: {compose}")
    for line in failures:
        print(f"FAIL {line}")
    print("PASS" if not failures and compose != "FAIL" else "FAIL")
    return 1 if failures or compose == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
