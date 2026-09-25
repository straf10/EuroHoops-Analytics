import os
import re
import shutil
import subprocess

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
    assert "git add predictions/ reports/ odds/" in DAILY


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
