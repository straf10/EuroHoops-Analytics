import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml

from tests.conftest import REPO

DAILY = (REPO / ".github/workflows/daily.yml").read_text(encoding="utf-8")
BASH = shutil.which("bash") or "bash"


def test_daily_workflow_runs_every_step_for_both_competitions() -> None:
    for command in ("ingest", "predict", "score"):
        assert re.search(rf"eurohoops {command}\s*$", DAILY, re.MULTILINE), command
        assert f"eurohoops {command} --competition gbl" in DAILY, command


def test_daily_caches_box_scores_before_build() -> None:
    """E7: the team model's box scores are cached (only new games fetched) before the build."""
    build = DAILY.index("eurohoops build")
    el_cache = DAILY.index("path: data/raw/euroleague")
    el_details = DAILY.index("eurohoops ingest --details")
    gbl_details = DAILY.index(
        "eurohoops ingest --competition gbl --details --pbp --seasons 2018 2019"
    )
    assert el_cache < el_details < build
    assert DAILY.index("path: data/raw/gbl") < gbl_details < build
    assert DAILY.index("eurohoops predict") > build


def test_daily_stages_el_history_and_stints_for_m5_softly() -> None:
    """M5 needs 2007-on games before the build and the stints mart after it, before predict; a
    failure of either step is a warning so Elo, M1, odds and injuries still run and commit."""
    stage = DAILY.index("eurohoops ingest --from-season 2007")
    build = DAILY.index("eurohoops build")
    stints = DAILY.index("eurohoops stints --mart")
    assert DAILY.index("eurohoops ingest --details") < stage < build < stints
    assert stints < DAILY.index("- name: Predict upcoming games")
    assert DAILY.count("will be skipped") == 2
    for command in ("ingest --from-season 2007", "stints --mart"):
        start = DAILY.index(command)
        assert '|| echo "::warning::' in DAILY[start : DAILY.index("- name:", start)]


def test_daily_runs_are_closer_together_than_the_prediction_window() -> None:
    """With one run a day and a 36 h window, every game is inside at least one run's window."""
    (cron,) = re.findall(r'cron: "([^"]+)"', DAILY)
    assert cron.split()[2:] == ["*", "*", "*"]
    assert "--window-hours" not in DAILY  # predict keeps its 36 h default


def test_daily_builds_the_site_from_fresh_data() -> None:
    publish = DAILY.index("eurohoops publish")
    build = DAILY.index("npm run build")
    upload = DAILY.index("upload-pages-artifact")
    assert publish < build < upload
    assert "working-directory: web" in DAILY


def odds_step() -> str:
    start = DAILY.index("- name: Record EuroLeague odds")
    return DAILY[start : DAILY.index("- name:", start + 1)]


def test_daily_records_odds_after_build_and_before_scoring() -> None:
    assert DAILY.index("eurohoops build") < DAILY.index("eurohoops odds")
    assert DAILY.index("eurohoops odds") < DAILY.index("eurohoops score")
    assert "git add predictions/ reports/ odds/ injuries/" in DAILY


def test_daily_records_injuries_between_odds_and_prediction() -> None:
    assert DAILY.index("- name: Record EuroLeague odds") < DAILY.index(
        "- name: Record EuroLeague injuries"
    )
    assert DAILY.index("uv run --no-dev eurohoops injuries") < DAILY.index(
        "- name: Predict upcoming games"
    )


def test_odds_step_is_skipped_with_a_warning_without_the_secret() -> None:
    step = odds_step()
    assert "ODDS_API_KEY: ${{ secrets.ODDS_API_KEY }}" in step
    body = step.split("run: |\n", 1)[1]
    script = "\n".join(line.removeprefix("          ") for line in body.splitlines())
    fake_uv = 'uv() { echo "uv $*"; }\n'  # stands in for the real command

    def run_step(secret: str) -> subprocess.CompletedProcess[str]:
        env = {**os.environ, "ODDS_API_KEY": secret}
        return subprocess.run(
            [BASH, "-c", fake_uv + script], env=env, capture_output=True, text=True, check=False
        )

    skipped = run_step("")
    assert skipped.returncode == 0
    assert "::warning::ODDS_API_KEY secret is not set" in skipped.stdout
    assert "uv run" not in skipped.stdout
    ran = run_step("x")
    assert ran.returncode == 0
    assert "uv run --no-dev eurohoops odds" in ran.stdout


CI = (REPO / ".github/workflows/ci.yml").read_text(encoding="utf-8")


def test_ci_builds_the_site_from_the_committed_fixture() -> None:
    web = CI[CI.index("\n  web:") :]
    assert "node-version: 24" in web
    assert "cp tests/fixtures/site.json web/src/data/site.json" in web
    assert web.index("site.json web/src/data") < web.index("npm ci") < web.index("npm run build")
    assert "working-directory: web" in web
    assert (REPO / "tests/fixtures/site.json").exists()


def daily_jobs() -> dict[str, Any]:
    jobs: dict[str, Any] = yaml.safe_load(DAILY)["jobs"]
    return jobs


def test_npm_never_runs_while_the_job_can_write_to_the_repository() -> None:
    """Third-party npm packages run in ``npm ci``: that job gets no write token."""
    top = yaml.safe_load(DAILY).get("permissions", {})
    assert top in ({}, None) or top.get("contents") != "write"
    npm_jobs = [
        name
        for name, job in daily_jobs().items()
        if any("npm ci" in str(step.get("run", "")) for step in job["steps"])
    ]
    assert npm_jobs
    for name in npm_jobs:
        assert daily_jobs()[name].get("permissions", {}).get("contents") != "write", name


def test_the_write_job_keeps_its_token_out_of_the_checkout() -> None:
    for name, job in daily_jobs().items():
        if job.get("permissions", {}).get("contents") != "write":
            continue
        checkout = next(
            s for s in job["steps"] if str(s.get("uses", "")).startswith("actions/checkout")
        )
        assert checkout.get("with", {}).get("persist-credentials") is False, name


def commit_step() -> dict[str, Any]:
    steps = [s for job in daily_jobs().values() for s in job["steps"]]
    return next(s for s in steps if s.get("name") == "Commit predictions and reports if changed")


def git(cwd: Path, *args: str) -> str:
    done = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True)
    return done.stdout


def sandbox(tmp_path: Path) -> tuple[Path, Path, dict[str, str]]:
    """A bare 'GitHub' remote, the runner's clone, and an env that routes the step's
    https://x-access-token URL to the bare remote."""
    bare, runner, other = tmp_path / "remote.git", tmp_path / "runner", tmp_path / "other"
    git(tmp_path, "init", "-q", "--bare", "-b", "main", str(bare))
    git(tmp_path, "clone", "-q", str(bare), str(other))
    for key, value in (("user.name", "t"), ("user.email", "t@t"), ("commit.gpgsign", "false")):
        git(other, "config", key, value)
    (other / "predictions").mkdir()
    (other / "predictions/log.csv").write_text("game_id,p\nG1,0.5\n", newline="\n")
    for folder in ("reports", "odds", "injuries"):  # the step adds all four folders
        (other / folder).mkdir()
        (other / folder / "keep.txt").write_text("x\n", newline="\n")
    git(other, "add", ".")
    git(other, "commit", "-qm", "base")
    git(other, "push", "-q", "origin", "main")
    git(tmp_path, "clone", "-q", str(bare), str(runner))
    url = "https://x-access-token:tok@github.com/o/r.git"
    env = {
        **os.environ,
        "GH_TOKEN": "tok",
        "REPO": "o/r",
        "BRANCH": "main",
        "GIT_CONFIG_COUNT": "2",
        "GIT_CONFIG_KEY_0": f"url.{bare.as_posix()}.insteadOf",
        "GIT_CONFIG_VALUE_0": url,
        "GIT_CONFIG_KEY_1": "commit.gpgsign",
        "GIT_CONFIG_VALUE_1": "false",
    }
    return other, runner, env


def run_commit_step(runner: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    step = commit_step()
    assert set(step["env"]) >= {"GH_TOKEN", "REPO", "BRANCH"}
    return subprocess.run(
        [BASH, "-e", "-c", step["run"]],
        cwd=runner,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_push_replays_the_run_on_a_main_that_moved(tmp_path: Path) -> None:
    other, runner, env = sandbox(tmp_path)
    (other / "README").write_text("moved\n")
    git(other, "add", ".")
    git(other, "commit", "-qm", "owner commit during the run")
    git(other, "push", "-q", "origin", "main")
    with (runner / "predictions/log.csv").open("a", newline="\n") as fh:
        fh.write("G2,0.6\n")
    done = run_commit_step(runner, env)
    assert done.returncode == 0, done.stderr
    git(other, "pull", "-q", "origin", "main")
    assert (other / "predictions/log.csv").read_text() == "game_id,p\nG1,0.5\nG2,0.6\n"
    assert (other / "README").exists()


def test_a_conflict_on_an_append_only_log_fails_without_rewriting_rows(tmp_path: Path) -> None:
    other, runner, env = sandbox(tmp_path)
    with (other / "predictions/log.csv").open("a", newline="\n") as fh:
        fh.write("G9,0.1\n")  # a row pushed while the run was going
    git(other, "commit", "-qam", "hand-logged row")
    git(other, "push", "-q", "origin", "main")
    with (runner / "predictions/log.csv").open("a", newline="\n") as fh:
        fh.write("G2,0.6\n")
    done = run_commit_step(runner, env)
    assert done.returncode != 0
    assert "::error::" in done.stdout
    git(other, "pull", "-q", "origin", "main")
    assert (other / "predictions/log.csv").read_text() == "game_id,p\nG1,0.5\nG9,0.1\n"
    assert not (runner / ".git/rebase-merge").exists()  # the rebase was aborted, not resolved


def test_an_unreachable_remote_fails_with_the_error_annotation(tmp_path: Path) -> None:
    """A pull that fails before any rebase starts (network, auth) still reports ::error::."""
    _, runner, env = sandbox(tmp_path)
    with (runner / "predictions/log.csv").open("a", newline="\n") as fh:
        fh.write("G2,0.6\n")
    env["GIT_CONFIG_KEY_0"] = f"url.{(tmp_path / 'missing.git').as_posix()}.insteadOf"
    done = run_commit_step(runner, env)
    assert done.returncode != 0
    assert "::error::" in done.stdout
