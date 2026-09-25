import re

from tests.conftest import REPO

DAILY = (REPO / ".github/workflows/daily.yml").read_text(encoding="utf-8")


def test_daily_workflow_runs_every_step_for_both_competitions() -> None:
    for command in ("ingest", "predict", "score"):
        assert re.search(rf"eurohoops {command}\s*$", DAILY, re.MULTILINE), command
        assert f"eurohoops {command} --competition gbl" in DAILY, command


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
