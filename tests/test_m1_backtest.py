"""E4: the M1 walk-forward backtest, the validation gate and the once-only test split."""

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from eurohoops.config import DecayGrid, Grid, M1Backtest
from eurohoops.eval.m1_backtest import (
    _decay_edges,
    format_m1_table,
    run_m1_backtest,
)
from eurohoops.models.team_eff import VARIANTS, DecayParams
from tests.conftest import make_games, make_team_games

SPEC = M1Backtest(
    Path("unused.json"),
    warmup=(2017, 2018),
    tuning=(2019, 2020),
    validation=(2021,),
    test=(2022, 2023),
    elo_grid=Grid(k=(20.0, 30.0), hca=(50.0, 90.0), reversion=(0.25,)),
    rating_grid=DecayGrid(half_life_days=(60.0, 240.0), carry=(0.5, 1.0), ridge=(250.0, 1000.0)),
    pace_grid=DecayGrid(half_life_days=(60.0,), carry=(1.0,), ridge=(2.0, 5.0)),
)


@pytest.fixture(scope="module", params=[False, True], ids=["euroleague-like", "gbl-like"])
def history(request: pytest.FixtureRequest) -> tuple[pd.DataFrame, pd.DataFrame]:
    games = make_games({s: True for s in range(2016, 2024)}, gbl_like=request.param)
    return games, make_team_games(games)


@pytest.fixture(scope="module")
def report(history: tuple[pd.DataFrame, pd.DataFrame]) -> dict[str, Any]:
    return run_m1_backtest(*history, SPEC)


def test_report_has_every_split_model_and_variant(report: dict[str, Any]) -> None:
    assert list(report["metrics"]) == ["tuning", "validation"]  # no test before the verdict
    assert not report["test_scored"]
    for models in report["metrics"].values():
        assert list(models) == ["m1", "elo", "b0"]
        for m in models.values():
            assert m["n"] > 0
            for key in ("log_loss", "brier", "accuracy", "ece", "margin_mae", "margin_crps"):
                assert m[key] is not None, key
            assert {"totals_mae", "totals_crps", "reliability"} <= set(m)
    assert list(report["variants"]) == list(VARIANTS)
    assert all(v["declared"] and not v["post_hoc"] for v in report["variants"].values())
    losses = report["grid"]["variant_tuning_log_loss"]
    assert report["tuned"]["variant"] == min(losses, key=losses.get)
    assert report["variants"][report["tuned"]["variant"]]["chosen"]
    gate = report["gate"]
    assert gate["passed"] == (gate["m1_log_loss"] < gate["elo_log_loss"])
    assert gate["m1_log_loss"] == report["metrics"]["validation"]["m1"]["log_loss"]
    for key in ("log_loss_m1_minus_elo", "margin_crps_m1_minus_elo", "totals_crps_m1_minus_elo"):
        ci = gate[key]
        assert ci["ci95"][0] <= ci["mean"] <= ci["ci95"][1]
    rating = report["tuned"]["rating"]
    assert rating["half_life_days"] in SPEC.rating_grid.half_life_days
    assert rating["ridge"] in SPEC.rating_grid.ridge
    assert report["tuned"]["pace"]["ridge"] in SPEC.pace_grid.ridge
    assert report["model_version"].split("+")[1].startswith("m1.")
    assert report["heteroscedasticity_tuning"]["n"] == report["metrics"]["tuning"]["m1"]["n"]
    segments = report["validation_segments"]
    assert segments["lopsided"]["n"] + segments["balanced"]["n"] == gate_n(report)


def gate_n(report: dict[str, Any]) -> int:
    n: int = report["metrics"]["validation"]["m1"]["n"]
    return n


def test_the_backtest_is_reproducible(
    history: tuple[pd.DataFrame, pd.DataFrame], report: dict[str, Any]
) -> None:
    assert json.dumps(run_m1_backtest(*history, SPEC)) == json.dumps(report)


def test_scoring_the_test_split_changes_nothing_else(
    history: tuple[pd.DataFrame, pd.DataFrame], report: dict[str, Any]
) -> None:
    with_test = run_m1_backtest(*history, SPEC, score_test=True)
    assert with_test["test_scored"]
    assert with_test["metrics"]["test"]["m1"]["n"] > 0
    assert "test_vs_elo" in with_test and "test_segments" in with_test
    for key in ("tuned", "grid", "gate", "comparison_elo", "b0", "model_version"):
        assert with_test[key] == report[key], key
    for split in ("tuning", "validation"):
        assert with_test["metrics"][split] == report["metrics"][split]


def test_games_without_box_lines_are_forecast_but_do_not_feed_the_fit(
    history: tuple[pd.DataFrame, pd.DataFrame], report: dict[str, Any]
) -> None:
    games, rows = history
    dropped = set(games.loc[games["season"] == 2021, "game_id"].iloc[:4])
    thinner = run_m1_backtest(games, rows[~rows["game_id"].isin(dropped)], SPEC)
    assert thinner["games_without_box_lines"]["validation"] == 4
    assert thinner["metrics"]["validation"]["m1"]["n"] == report["metrics"]["validation"]["m1"]["n"]


def test_format_table_shows_the_gate(report: dict[str, Any]) -> None:
    table = format_m1_table(report)
    assert "validation m1" in table
    assert "gate (" in table and ("PASS" in table or "FAIL" in table)


def test_decay_grid_edges_treat_full_carry_as_a_natural_bound() -> None:
    grid = DecayGrid((60.0, 120.0), (0.5, 1.0), (1.0, 2.0, 4.0))
    assert _decay_edges(grid, DecayParams(60.0, 1.0, 2.0)) == ["half_life_days"]
    assert _decay_edges(grid, DecayParams(120.0, 0.5, 4.0)) == ["half_life_days", "carry", "ridge"]
