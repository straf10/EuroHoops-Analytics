import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from eurohoops import cli
from eurohoops.config import BACKTEST_REPORT, GAMES_PATH, PREDICTION_LOG, SCORECARD_REPORT
from eurohoops.eval.backtest import GRID_HCA, GRID_K, GRID_REVERSION, load_tuned_model
from eurohoops.parse.games import read_games, write_games
from tests.conftest import make_games
from tests.test_ingest import FakeApi

runner = CliRunner()
NOW = datetime(2026, 10, 1, 8, 0, tzinfo=UTC)


@pytest.fixture
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def pipeline(workdir: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Synthetic staging table: three played seasons plus an unplayed live season."""
    write_games(make_games({2023: True, 2024: True, 2025: True, 2026: False}), GAMES_PATH)
    monkeypatch.setattr(cli, "utc_now", lambda: NOW)
    return workdir


@pytest.mark.usefixtures("workdir", "no_sleep")
def test_ingest_accepts_space_separated_seasons(monkeypatch: pytest.MonkeyPatch) -> None:
    api = FakeApi()
    monkeypatch.setattr(
        cli, "make_client", lambda: httpx.Client(transport=httpx.MockTransport(api))
    )
    result = runner.invoke(cli.app, ["ingest", "--seasons", "2024", "2026", "--details"])
    assert result.exit_code == 0, result.output
    assert len(api.calls) == 2 + 3 * 3
    assert list(read_games(GAMES_PATH)["season"].unique()) == [2024, 2026]


@pytest.mark.usefixtures("workdir")
def test_ingest_rejects_junk_arguments() -> None:
    result = runner.invoke(cli.app, ["ingest", "--seasons", "2024", "--detials"])
    assert result.exit_code != 0


@pytest.mark.usefixtures("pipeline")
def test_backtest_predict_score_end_to_end() -> None:
    result = runner.invoke(cli.app, ["backtest"])
    assert result.exit_code == 0, result.output
    assert "tuned: K=" in result.output
    report = json.loads(BACKTEST_REPORT.read_text())
    tuned = report["tuned"]
    assert (tuned["k"], tuned["hca"], tuned["reversion"]) in {
        (k, h, r) for k in GRID_K for h in GRID_HCA for r in GRID_REVERSION
    }
    assert report["grid"]["size"] == 60
    assert report["model_version"] == load_tuned_model(BACKTEST_REPORT).version()
    ci = report["test_log_loss_diff_elo_minus_b0"]
    assert ci["ci95"][0] <= ci["mean"] <= ci["ci95"][1]

    assert runner.invoke(cli.app, ["predict"]).exit_code == 0
    first = PREDICTION_LOG.read_bytes()
    assert runner.invoke(cli.app, ["predict"]).exit_code == 0
    assert PREDICTION_LOG.read_bytes() == first

    result = runner.invoke(cli.app, ["score"])
    assert result.exit_code == 0, result.output
    card = json.loads(SCORECARD_REPORT.read_text())
    assert card["rows_in_log"] == 3
    assert card["elo"]["n"] == 0


@pytest.mark.usefixtures("pipeline")
def test_predict_exits_non_zero_when_stamp_is_not_before_tipoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert runner.invoke(cli.app, ["backtest"]).exit_code == 0
    ticks: Iterator[datetime] = iter([NOW, datetime(2026, 10, 1, 18, 0, tzinfo=UTC)])
    monkeypatch.setattr(cli, "utc_now", lambda: next(ticks))
    result = runner.invoke(cli.app, ["predict"])
    assert result.exit_code == 1
    assert not PREDICTION_LOG.exists()
