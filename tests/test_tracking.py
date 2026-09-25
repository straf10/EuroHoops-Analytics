"""E5: MLflow tracking of M1 backtests (temporary stores only; never the repo's mlruns/)."""

import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from eurohoops.eval.m1_backtest import run_m1_backtest
from eurohoops.eval.tracking import (
    data_hash,
    default_tracking_uri,
    flatten_metrics,
    git_state,
    log_backtest,
)
from tests.conftest import REPO, make_games, make_team_games
from tests.test_m1_backtest import SPEC


def test_flatten_metrics_keeps_numeric_leaves_only() -> None:
    value = {"a": {"b": 1, "c": [0.5, {"d": 2}]}, "ok": True, "none": None, "s": "x", "e f(g)": 3}
    assert flatten_metrics(value) == {"a.b": 1.0, "a.c.0": 0.5, "a.c.1.d": 2.0, "e f_g_": 3.0}


def test_data_hash_ignores_row_order_but_not_values() -> None:
    games = make_games({2020: True})
    rows = make_team_games(games)
    first = data_hash(games, rows)
    assert data_hash(games.sample(frac=1.0, random_state=1), rows.iloc[::-1]) == first
    changed = rows.copy()
    changed.loc[0, "fga"] += 1
    assert data_hash(games, changed) != first


def test_git_state_in_and_outside_a_checkout(tmp_path: Path) -> None:
    commit, dirty = git_state(REPO)
    assert len(commit) == 40 and isinstance(dirty, bool)
    assert git_state(tmp_path) == ("unknown", True)


@pytest.fixture(scope="module")
def report() -> dict[str, Any]:
    games = make_games({s: True for s in range(2016, 2024)})
    return run_m1_backtest(games, make_team_games(games), SPEC)


def test_one_parent_and_a_child_per_variant(tmp_path: Path, report: dict[str, Any]) -> None:
    import mlflow  # noqa: PLC0415 - the dev dependency under test

    uri = default_tracking_uri(tmp_path / "mlruns")
    run_id = log_backtest(report, "euroleague", report["data_sha256"], uri)
    assert run_id is not None
    again = log_backtest(report, "euroleague", report["data_sha256"], uri)  # reuses the experiment
    mlflow.set_tracking_uri(uri)
    runs = mlflow.search_runs(experiment_names=["m1-backtest"])
    assert isinstance(runs, pd.DataFrame)
    mine = runs[(runs["run_id"] == run_id) | (runs["tags.mlflow.parentRunId"] == run_id)]
    parent = mine[mine["run_id"] == run_id].iloc[0]
    children = mine[mine["tags.mlflow.parentRunId"] == run_id]
    assert len(children) == len(report["variants"])
    assert set(children["params.variant"]) == set(report["variants"])
    for run in (parent, *[row for _, row in children.iterrows()]):
        assert run["params.data_sha256"] == report["data_sha256"]
        assert len(run["params.git_commit"]) == 40
        assert run["params.git_dirty"] in {"true", "false"}
    assert parent["params.rating.ridge"] == str(report["tuned"]["rating"]["ridge"])
    assert parent["metrics.gate.m1_log_loss"] == report["gate"]["m1_log_loss"]
    assert children["metrics.validation.log_loss"].notna().all()
    artifacts = mlflow.MlflowClient().list_artifacts(run_id)
    assert [a.path for a in artifacts] == ["backtest_m1_euroleague.json"]
    assert again != run_id
    assert not (REPO / "mlruns").exists() or not any((REPO / "mlruns").glob("**/*" + run_id + "*"))


def test_tracking_is_skipped_without_mlflow(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, report: dict[str, Any]
) -> None:
    monkeypatch.setitem(sys.modules, "mlflow", None)
    with caplog.at_level(logging.WARNING):
        assert log_backtest(report, "gbl", "hash", "sqlite:///unused.db") is None
    assert "not installed" in caplog.text
