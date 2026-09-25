"""Predictions before time T must be bit-identical whatever happens to games at or after T."""

from typing import Any

import numpy as np
import pandas as pd
import pytest

from eurohoops.eval.backtest import run_backtest, win_probabilities
from eurohoops.models.elo import EloParams, FloatArray, prepare, replay
from tests.conftest import make_games
from tests.test_backtest import SPEC

PARAMS = EloParams(k=30.0, hca=80.0, reversion=0.5)


@pytest.fixture(params=[False, True], ids=["euroleague-like", "gbl-like"])
def history(request: pytest.FixtureRequest) -> pd.DataFrame:
    return make_games({2023: True, 2024: True}, gbl_like=request.param)


def cutoff(games: pd.DataFrame) -> pd.Timestamp:
    """A time in the middle of the second season."""
    second = games[games["season"] == 2024]
    return second["tipoff_utc"].iloc[len(second) // 2]


def predictions_before(games: pd.DataFrame, t: pd.Timestamp) -> FloatArray:
    ordered = games.sort_values(["tipoff_utc", "game_code"], ignore_index=True)
    probs: FloatArray = win_probabilities(replay(prepare(ordered), PARAMS))
    return probs[(ordered["tipoff_utc"] < t).to_numpy()]


def flip_scores(games: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    flipped = games.copy()
    flipped.loc[mask, ["home_score", "away_score"]] = games.loc[
        mask, ["away_score", "home_score"]
    ].to_numpy()
    return flipped


def test_deleting_future_games_changes_nothing(history: pd.DataFrame) -> None:
    t = cutoff(history)
    baseline = predictions_before(history, t)
    assert len(baseline) > 0
    assert np.array_equal(baseline, predictions_before(history[history["tipoff_utc"] < t], t))


def test_altering_future_results_changes_nothing(history: pd.DataFrame) -> None:
    t = cutoff(history)
    future = history["tipoff_utc"] >= t
    altered = flip_scores(history, future)
    assert not altered.equals(history)
    assert np.array_equal(predictions_before(history, t), predictions_before(altered, t))


def test_duplicating_future_games_changes_nothing(history: pd.DataFrame) -> None:
    t = cutoff(history)
    future = history[history["tipoff_utc"] >= t]
    dupes = future.assign(game_code=future["game_code"] + 10_000)
    duplicated = pd.concat([history, dupes], ignore_index=True)
    assert np.array_equal(predictions_before(history, t), predictions_before(duplicated, t))


def test_a_games_own_result_does_not_change_its_prediction(history: pd.DataFrame) -> None:
    ordered = history.reset_index(drop=True)
    target = len(ordered) // 2
    flipped = flip_scores(ordered, ordered.index == target)
    before = replay(prepare(ordered), PARAMS)[target]
    after = replay(prepare(flipped), PARAMS)[target]
    assert before == after


def test_prepare_rejects_unsorted_games(history: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="sorted"):
        prepare(history.iloc[::-1])


# Everything the backtest fits before the test seasons: sigma, B0, the Elo parameters, the
# tuning split's metrics (incl. ECE and reliability bins) and the totals baselines up to s.
def fitted_before(report: dict[str, Any], season: int) -> dict[str, Any]:
    baseline = report["totals"]["baseline_by_season"]
    return {
        "tuned": report["tuned"],
        "b0": report["b0"],
        "tuning": report["metrics"]["tuning"],
        "totals_tuning": report["totals"]["tuning"],
        "baseline": {s: v for s, v in baseline.items() if int(s) <= season},
    }


def only_baselines(report: dict[str, Any], season: int) -> dict[str, Any]:
    return {s: v for s, v in report["totals"]["baseline_by_season"].items() if int(s) <= season}


def mutations(games: pd.DataFrame, season: int) -> dict[str, pd.DataFrame]:
    """Delete, alter and add games of ``season`` and later (every third one, so none is empty)."""
    later = (games["season"] >= season) & games["played"]
    some = later & (games["game_code"] % 3 == 0)
    extra = games[later].assign(game_code=games.loc[later, "game_code"] + 10_000)
    added = pd.concat([games, extra], ignore_index=True)
    return {
        "deleted": games[~some].reset_index(drop=True),
        "altered": flip_scores(games, some),
        "added": added.sort_values(["tipoff_utc", "game_code"], ignore_index=True),
    }


@pytest.fixture(scope="module")
def backtest_games() -> pd.DataFrame:
    return make_games({s: True for s in range(2017, 2024)} | {2024: False}, gbl_like=True)


@pytest.mark.parametrize("season", [2022, 2023])
def test_nothing_fitted_before_a_test_season_sees_it(
    backtest_games: pd.DataFrame, season: int
) -> None:
    baseline = fitted_before(run_backtest(backtest_games, SPEC), season)
    for name, mutated in mutations(backtest_games, season).items():
        assert fitted_before(run_backtest(mutated, SPEC), season) == baseline, name


@pytest.mark.parametrize("season", [2020, 2021, 2022, 2023, 2024])
def test_the_totals_baseline_for_a_season_ignores_that_season_and_later(
    backtest_games: pd.DataFrame, season: int
) -> None:
    baseline = only_baselines(run_backtest(backtest_games, SPEC), season)
    assert baseline[str(season)] is not None
    for name, mutated in mutations(backtest_games, season).items():
        assert only_baselines(run_backtest(mutated, SPEC), season) == baseline, name


def test_the_mutations_do_change_later_results(backtest_games: pd.DataFrame) -> None:
    """Guard against a vacuous check: the same edits do move the test-season numbers."""
    before = run_backtest(backtest_games, SPEC)["metrics"]["test"]
    for name, mutated in mutations(backtest_games, 2022).items():
        assert run_backtest(mutated, SPEC)["metrics"]["test"] != before, name
