"""Predictions before time T must be bit-identical whatever happens to games at or after T."""

import numpy as np
import pandas as pd
import pytest

from eurohoops.eval.backtest import win_probabilities
from eurohoops.models.elo import EloParams, FloatArray, prepare, replay
from tests.conftest import make_games

PARAMS = EloParams(k=30.0, hca=80.0, reversion=0.5)


@pytest.fixture
def history() -> pd.DataFrame:
    return make_games({2023: True, 2024: True})


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
