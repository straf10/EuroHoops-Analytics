"""The M3 backtest harness (week 9-12 H1): tuning-only mode, the report shape, the gate, and
byte-identical reruns. ``box_only``/``pir`` are test doubles (subagent B owns the real ones)."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd
import pytest

from eurohoops import cli
from eurohoops.config import M3Backtest, M3Grid
from eurohoops.eval.m3_backtest import (
    BaselineResult,
    format_m3_table,
    m1_margins,
    run_m3_backtest,
)
from eurohoops.models.elo import FloatArray
from eurohoops.models.team_eff import DecayParams, prepare_history
from eurohoops.models.team_eff import forecast as team_eff_forecast
from tests.m3_synthetic import SyntheticM3, make_synthetic_m3

SPEC = M3Backtest(
    report=None,  # type: ignore[arg-type]  # unused by run_m3_backtest, which returns the dict
    warmup=(2015, 2016),
    tuning=(2017, 2018),
    validation=(2019,),
    test=(2020, 2021),
    grid=M3Grid(half_life_days=(365.0, 730.0), ridge=(200.0, 800.0)),
)

TUNED_M1: dict[str, Any] = {
    "rating": asdict(DecayParams(half_life_days=365.0, carry=1.0, ridge=500.0)),
    "pace": asdict(DecayParams(half_life_days=365.0, carry=1.0, ridge=2.0)),
    "margin": {"variant": "normal_const", "scale": 10.0, "df": None, "ref_pace": None},
    "totals_sigma": 12.0,
}


def _zero_baseline(
    games: pd.DataFrame,
    player_games: pd.DataFrame,
    shares: pd.DataFrame,
    possessions: pd.Series,
    tuning: FloatArray,
) -> BaselineResult:
    return BaselineResult(margin=np.zeros(len(games)), params={"kind": "zero"})


def _pir_like_baseline(
    games: pd.DataFrame,
    player_games: pd.DataFrame,
    shares: pd.DataFrame,
    possessions: pd.Series,
    tuning: FloatArray,
) -> BaselineResult:
    """A deliberately noisier constant-zero double, distinct from box_only's, so the
    ``comparisons`` block has something other than an identical zero to compare against."""
    rng = np.random.default_rng(0)
    return BaselineResult(margin=rng.normal(0.0, 0.01, len(games)), params={"kind": "pir_like"})


@pytest.fixture(scope="module")
def data() -> SyntheticM3:
    return make_synthetic_m3(
        list(range(2015, 2022)),
        teams=8,
        games_per_season=20,
        stints_per_game=10,
        roster=8,
        seed=5,
    )


def test_tuning_only_report_has_no_validation_or_test_metric(data: SyntheticM3) -> None:
    report = run_m3_backtest(
        data.games,
        data.team_games,
        data.player_games,
        data.stints,
        data.checks,
        spec=SPEC,
        tuned_m1=TUNED_M1,
        box_only_fn=_zero_baseline,
        pir_fn=_pir_like_baseline,
        tuning_only=True,
    )
    assert report["tuning_only"] is True
    assert not report["validation_scored"]
    assert not report["test_scored"]
    assert list(report["metrics"]) == ["tuning"]  # tuning numbers only, per H5
    assert list(report["games_per_split"]) == ["tuning"]
    assert set(report["metrics"]["tuning"]) == {"rapm", "box_only", "pir", "m1", "b0"}
    assert "gate" not in report
    assert "comparisons" not in report
    assert report["chosen"]["variant"] in report["grid"]


@pytest.fixture(scope="module")
def report(data: SyntheticM3) -> dict[str, Any]:
    return run_m3_backtest(
        data.games,
        data.team_games,
        data.player_games,
        data.stints,
        data.checks,
        spec=SPEC,
        tuned_m1=TUNED_M1,
        box_only_fn=_zero_baseline,
        pir_fn=_pir_like_baseline,
    )


def test_full_report_scores_tuning_and_validation_not_test(report: dict[str, Any]) -> None:
    assert list(report["metrics"]) == ["tuning", "validation"]
    assert report["validation_scored"]
    assert not report["test_scored"]
    for split, models in report["metrics"].items():
        assert set(models) == {"rapm", "box_only", "pir", "m1", "b0"}
        for name, m in models.items():
            assert m["n"] >= 0, (split, name)
    gate = report["gate"]
    assert gate["reference"] == "box_only"
    assert gate["variant"] == report["chosen"]["variant"]
    if gate["rmse_diff"] is not None:
        ci = gate["rmse_diff"]
        assert ci["ci95"][0] <= ci["mean"] <= ci["ci95"][1]
        assert gate["passed"] == (ci["ci95"][1] < 0.0)
    assert set(report["comparisons"]) == {"m1", "pir"}
    for _split, counts in report["games_per_split"].items():
        assert counts["scored"] + counts["dropped"] == counts["rated"]
    assert report["oracle_minutes"]["label"] == "oracle, not a forecast"


def test_score_test_adds_the_test_split(data: SyntheticM3) -> None:
    report = run_m3_backtest(
        data.games,
        data.team_games,
        data.player_games,
        data.stints,
        data.checks,
        spec=SPEC,
        tuned_m1=TUNED_M1,
        box_only_fn=_zero_baseline,
        pir_fn=_pir_like_baseline,
        score_test=True,
    )
    assert list(report["metrics"]) == ["tuning", "validation", "test"]
    assert report["test_scored"]


def test_two_runs_are_byte_identical(data: SyntheticM3) -> None:
    args = (data.games, data.team_games, data.player_games, data.stints, data.checks)
    kwargs = {
        "spec": SPEC,
        "tuned_m1": TUNED_M1,
        "box_only_fn": _zero_baseline,
        "pir_fn": _pir_like_baseline,
    }
    first = json.dumps(run_m3_backtest(*args, **kwargs), sort_keys=True)
    second = json.dumps(run_m3_backtest(*args, **kwargs), sort_keys=True)
    assert first == second


def test_m1_margins_equal_team_eff_forecast_with_the_committed_params(data: SyntheticM3) -> None:
    rating = DecayParams(**TUNED_M1["rating"])
    pace = DecayParams(**TUNED_M1["pace"])
    frame = data.games.sort_values("tipoff_utc").reset_index(drop=True)
    history = prepare_history(frame, data.team_games)
    expected = team_eff_forecast(history, rating, pace).margin
    got = m1_margins(frame, data.team_games, TUNED_M1)
    np.testing.assert_array_equal(got, expected)


def test_format_m3_table_runs_for_both_modes(data: SyntheticM3, report: dict[str, Any]) -> None:
    tuning_only = run_m3_backtest(
        data.games,
        data.team_games,
        data.player_games,
        data.stints,
        data.checks,
        spec=SPEC,
        tuned_m1=TUNED_M1,
        box_only_fn=_zero_baseline,
        pir_fn=_pir_like_baseline,
        tuning_only=True,
    )
    assert "chosen" in format_m3_table(tuning_only)
    assert "gate" in format_m3_table(report)


def test_cli_declares_m3_and_soft_imports_box_impact() -> None:
    """``eurohoops backtest --model m3`` exists and its box_impact wiring degrades cleanly
    when subagent B's ``models/box_impact.py`` (written in a parallel branch) is not merged
    into this worktree yet -- exercised directly (not through ``test_cli.py``, which this
    subagent does not own) so the CLI plumbing itself is covered."""
    assert cli.ModelName.m3 == "m3"
    assert cli._box_impact() is None  # box_impact.py does not exist in this worktree
