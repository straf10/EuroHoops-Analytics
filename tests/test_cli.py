import json
from collections.abc import Iterator
from datetime import UTC, datetime
from itertools import product
from pathlib import Path

import httpx
import pandas as pd
import pytest
from typer.testing import CliRunner

from eurohoops import cli
from eurohoops.config import (
    BOX_INVARIANTS_REPORT,
    EUROLEAGUE,
    GBL,
    MART_PATH,
    SITE_DIR,
    SQL_DIR,
)
from eurohoops.eval.backtest import load_tuned_model
from eurohoops.marts import read_games
from tests.conftest import REPO, make_games, write_pipeline
from tests.test_gbl_ingest import FakeEsake
from tests.test_ingest import FakeApi

runner = CliRunner()
NOW = datetime(2026, 10, 1, 8, 0, tzinfo=UTC)


@pytest.fixture
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    (tmp_path / SQL_DIR).mkdir()
    for sql in (REPO / SQL_DIR).glob("*.sql"):
        (tmp_path / SQL_DIR / sql.name).write_text(sql.read_text(encoding="utf-8"))
    return tmp_path


@pytest.fixture
def pipeline(workdir: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Staged + built synthetic data covering every backtest split of both competitions."""
    write_pipeline(
        make_games({s: True for s in range(2007, 2026)} | {2026: False}),
        make_games({s: True for s in range(2018, 2026)} | {2026: False}, gbl_like=True),
    )
    monkeypatch.setattr(cli, "utc_now", lambda: NOW)
    return workdir


def invoke(*args: str) -> str:
    result = runner.invoke(cli.app, list(args))
    assert result.exit_code == 0, result.output
    return result.output


@pytest.mark.usefixtures("workdir", "no_sleep")
def test_ingest_both_competitions_then_build(monkeypatch: pytest.MonkeyPatch) -> None:
    el_api, esake = FakeApi(), FakeEsake()
    transports = iter([httpx.MockTransport(el_api), httpx.MockTransport(esake)])
    monkeypatch.setattr(cli, "make_client", lambda: httpx.Client(transport=next(transports)))
    invoke("ingest", "--seasons", "2024", "2026", "--details")
    assert len(el_api.calls) == 2 + 3 * 3
    invoke("ingest", "--competition", "gbl", "--seasons", "2018", "--details")
    assert len(esake.calls) == 4 + 4
    output = invoke("build")
    assert "gbl box scores 2018: 0/4 pass" in output  # one recorded box served for every game
    report = json.loads(BOX_INVARIANTS_REPORT.read_text())
    assert report["seasons"]["2018"]["games"] == 4
    assert list(read_games(MART_PATH, "euroleague")["season"].unique()) == [2024, 2026]
    assert read_games(MART_PATH, "gbl")["forfeit"].sum() == 1


@pytest.mark.usefixtures("workdir")
def test_ingest_rejects_junk_arguments() -> None:
    result = runner.invoke(cli.app, ["ingest", "--seasons", "2024", "--detials"])
    assert result.exit_code != 0


@pytest.mark.usefixtures("pipeline")
def test_backtest_predict_score_publish_end_to_end() -> None:
    output = invoke("backtest")
    assert "backtest_elo_history.json" in output
    for spec in (EUROLEAGUE.live_backtest, *EUROLEAGUE.history_backtests):
        grid = set(product(spec.grid.k, spec.grid.hca, spec.grid.reversion))
        report = json.loads(spec.report.read_text())
        tuned = report["tuned"]
        assert (tuned["k"], tuned["hca"], tuned["reversion"]) in grid
        assert not tuned["frozen"]
        assert {k: report["grid"]["best"][k] for k in ("k", "hca", "reversion")} == {
            k: tuned[k] for k in ("k", "hca", "reversion")
        }
        assert report["grid"]["best_minus_tuned_log_loss"]["test"]["mean"] == 0.0
        assert report["seasons"]["test"] == list(spec.test)
        assert report["model_version"] == load_tuned_model(spec.report).version()
    output = invoke("backtest", "--competition", "gbl")
    assert "[frozen]" in output
    gbl_report = json.loads(GBL.live_backtest.report.read_text())
    assert GBL.live_backtest.frozen is not None
    assert load_tuned_model(GBL.live_backtest.report).params == GBL.live_backtest.frozen
    assert gbl_report["tuned"]["frozen"]
    assert gbl_report["grid"]["size"] == 9 * 10 * 5
    assert gbl_report["seasons"]["tuning"] == [2022, 2023]
    ci = gbl_report["test_margin_abs_error_diff_elo_minus_b0"]
    assert ci["ci95"][0] <= ci["mean"] <= ci["ci95"][1]

    invoke("predict")
    el_log = EUROLEAGUE.prediction_log.read_bytes()
    invoke("predict", "--competition", "gbl")
    invoke("predict", "--competition", "gbl")
    assert EUROLEAGUE.prediction_log.read_bytes() == el_log  # a GBL run never touches it
    gbl_log = pd.read_csv(GBL.prediction_log)
    assert len(gbl_log) == 3
    assert gbl_log["game_id"].str.startswith("GBL2026_").all()

    invoke("score")
    invoke("score", "--competition", "gbl")
    assert json.loads(GBL.scorecard.read_text())["rows_in_log"] == 3
    invoke("publish")
    page = (SITE_DIR / "index.html").read_text(encoding="utf-8")
    assert page.count("Team AAA &amp; Co") >= 2  # names escaped, both sections


@pytest.mark.usefixtures("pipeline")
def test_predict_exits_non_zero_when_stamp_is_not_before_tipoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invoke("backtest")
    ticks: Iterator[datetime] = iter([NOW, datetime(2026, 10, 1, 18, 0, tzinfo=UTC)])
    monkeypatch.setattr(cli, "utc_now", lambda: next(ticks))
    result = runner.invoke(cli.app, ["predict"])
    assert result.exit_code == 1
    assert not EUROLEAGUE.prediction_log.exists()
