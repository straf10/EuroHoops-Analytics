"""Predictions before time T must be bit-identical whatever happens to games at or after T."""

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from eurohoops.eval.backtest import run_backtest, win_probabilities
from eurohoops.eval.m1_backtest import run_m1_backtest
from eurohoops.models.elo import EloParams, FloatArray, prepare, replay
from eurohoops.models.team_eff import DecayParams, forecast, prepare_history
from tests.conftest import make_games, make_team_games
from tests.test_backtest import SPEC
from tests.test_m1_backtest import SPEC as M1_TEST_SPEC

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


# --- M1 (E6) ---------------------------------------------------------------------------------

M1_RATING = DecayParams(half_life_days=120.0, carry=0.5, ridge=500.0)
M1_PACE = DecayParams(half_life_days=120.0, carry=0.5, ridge=5.0)


@pytest.fixture(params=[False, True], ids=["euroleague-like", "gbl-like"])
def m1_history(request: pytest.FixtureRequest) -> tuple[pd.DataFrame, pd.DataFrame]:
    games = make_games({2022: True, 2023: True, 2024: True}, gbl_like=request.param)
    return games, make_team_games(games)


def m1_forecasts(games: pd.DataFrame, rows: pd.DataFrame, t: pd.Timestamp) -> dict[str, Any]:
    """game id -> (home points, away points, pace) forecast, for games tipping off before t."""
    ordered = games.sort_values(["tipoff_utc", "game_code"], ignore_index=True)
    result = forecast(prepare_history(ordered, rows), M1_RATING, M1_PACE)
    before = (ordered["tipoff_utc"] < t).to_numpy()
    return {
        game_id: (h, a, p)
        for game_id, h, a, p in zip(
            ordered.loc[before, "game_id"],
            result.home_points[before],
            result.away_points[before],
            result.pace[before],
            strict=True,
        )
    }


def same(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Bit-identical, NaN (no forecast yet) equal to NaN."""
    return a.keys() == b.keys() and all(
        np.array_equal(np.array(a[k]), np.array(b[k]), equal_nan=True) for k in a
    )


def m1_mutations(
    games: pd.DataFrame, rows: pd.DataFrame, t: pd.Timestamp
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Delete, alter (score, box line, possessions) and add games tipping off at or after t.

    The added games include a team never seen before t.
    """
    later = games["tipoff_utc"] >= t
    later_ids = set(games.loc[later, "game_id"])
    in_later = rows["game_id"].isin(later_ids)
    swapped = rows.copy()
    for game_id in later_ids:
        pair = swapped.index[swapped["game_id"] == game_id]
        swapped.loc[pair, "points"] = swapped.loc[pair[::-1], "points"].to_numpy()
    box = rows.assign(
        fga=rows["fga"] + 7 * in_later,
        poss_raw=rows["poss_raw"] + 7 * in_later,
        poss_game=rows["poss_game"] + 7 * in_later,
        minutes=rows["minutes"] + 5.0 * in_later,
    )
    played = games[later & games["played"]]
    extra = played.assign(game_code=played["game_code"] + 10_000, game_id=played["game_id"] + "x")
    newcomer = extra.assign(
        home="NEW", game_id=extra["game_id"] + "n", game_code=extra["game_code"] + 5_000
    )
    added = pd.concat([games, extra, newcomer], ignore_index=True).sort_values(
        ["tipoff_utc", "game_code"], ignore_index=True
    )
    added_rows = pd.concat([rows, make_team_games(pd.concat([extra, newcomer]))])
    return {
        "deleted": (games[~later], rows[~in_later]),
        "scores_altered": (flip_scores(games, later), swapped),
        "box_lines_altered": (games, box),
        "added_incl_a_new_team": (added, added_rows),
    }


def m1_cutoff(games: pd.DataFrame) -> pd.Timestamp:
    """The first tip-off of a round in the middle of the last season."""
    last = games[games["season"] == games["season"].max()]
    middle = sorted(last["round"].unique())[len(last["round"].unique()) // 2]
    return last.loc[last["round"] == middle, "tipoff_utc"].min()


def test_m1_forecasts_before_t_ignore_every_later_edit(
    m1_history: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    games, rows = m1_history
    t = m1_cutoff(games)
    baseline = m1_forecasts(games, rows, t)
    assert len(baseline) > 60
    for name, (g, r) in m1_mutations(games, rows, t).items():
        assert same(m1_forecasts(g, r, t), baseline), name


def test_the_same_edits_do_change_later_m1_forecasts(
    m1_history: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Guard against a vacuous check: forecasts of the rounds after t do move."""
    games, rows = m1_history
    t = m1_cutoff(games)
    horizon = games["tipoff_utc"].max() + pd.Timedelta(days=1)
    baseline = m1_forecasts(games, rows, horizon)
    for name, (g, r) in m1_mutations(games, rows, t).items():
        if name == "deleted":
            continue  # the later games themselves are gone
        mutated = m1_forecasts(g, r, horizon)
        changed = [k for k in baseline if not same({k: baseline[k]}, {k: mutated[k]})]
        assert changed, name


def test_a_games_own_box_line_never_feeds_its_forecast(
    m1_history: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    games, rows = m1_history
    ordered = games.sort_values(["tipoff_utc", "game_code"], ignore_index=True)
    for target in (len(ordered) // 2, len(ordered) - 1):
        own = rows["game_id"] == ordered.loc[target, "game_id"]
        edited = rows.assign(
            points=rows["points"] + 30 * own,
            poss_game=rows["poss_game"] + 20 * own,
            poss_raw=rows["poss_raw"] + 20 * own,
        )
        before = forecast(prepare_history(ordered, rows), M1_RATING, M1_PACE)
        after = forecast(prepare_history(ordered, edited), M1_RATING, M1_PACE)
        assert before.home_points[target] == after.home_points[target]
        assert before.pace[target] == after.pace[target]


M1_SPEC = replace(
    M1_TEST_SPEC, warmup=(2017, 2018), tuning=(2019, 2020), validation=(2021,), test=(2022, 2023)
)


@pytest.fixture(scope="module", params=[False, True], ids=["euroleague-like", "gbl-like"])
def m1_backtest_history(request: pytest.FixtureRequest) -> tuple[pd.DataFrame, pd.DataFrame]:
    games = make_games({s: True for s in range(2017, 2024)}, gbl_like=request.param)
    return games, make_team_games(games)


def tuning_fitted(report: dict[str, Any]) -> dict[str, Any]:
    """Everything the backtest fits on the tuning seasons (hyperparameters, sigma, df, Elo)."""
    return {
        "tuned": report["tuned"],
        "grid": report["grid"],
        "comparison_elo": report["comparison_elo"],
        "b0": report["b0"],
        "tuning": report["metrics"]["tuning"],
        "heteroscedasticity": report["heteroscedasticity_tuning"],
        "variants": {
            k: (v["df"], v["scale"], v["ref_pace"]) for k, v in report["variants"].items()
        },
    }


def test_nothing_tuned_sees_validation_or_test_games(
    m1_backtest_history: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    games, rows = m1_backtest_history
    baseline = run_m1_backtest(games, rows, M1_SPEC)
    t = games.loc[games["season"] == M1_SPEC.validation[0], "tipoff_utc"].min()
    for name, (g, r) in m1_mutations(games, rows, t).items():
        mutated = run_m1_backtest(g, r, M1_SPEC)
        assert tuning_fitted(mutated) == tuning_fitted(baseline), name
        # guard: the same edit does reach the validation numbers
        assert mutated["metrics"]["validation"] != baseline["metrics"]["validation"], name
