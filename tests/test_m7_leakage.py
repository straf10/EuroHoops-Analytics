"""Leakage tests for the M7 backtest (week 14-16 K5): nothing a checkpoint uses or outputs depends
on a game result at or after its cutoff (reports/m7_progress.md, section 1: "the schedule of the
remaining fixtures is known in advance and may be used; their results may not").

What is guaranteed, at every checkpoint of the checkpoint season (2018 of the synthetic league):

* **Score edits** (every game at or after the cutoff gets other scores: the season's remaining
  fixtures, its playoffs and Final Four, every later season): the inputs (M1's posterior, pace
  point, both noise scales, the Elo ratings, the played results) and every forecast (each model's
  rank distribution, P(direct), P(play-in), P(top 10), P(title), hence the expected rank) are
  bit-identical. The outcome columns of the CSV (final_place, made_direct, made_top10, champion,
  rps) are targets and do move.
* **Deleting or adding games** at or after the cutoff: the posterior, the pace point, the Elo
  ratings and the played results are bit-identical. The remaining *schedule* is an input, so
  deleting or adding a remaining fixture changes the net noise scale and the simulated fixtures
  and is not expected to leave the probabilities alone; the same edit in later seasons or in the
  playoffs leaves every forecast bit-identical as well.
* **Sensitivity**: changing a result before the cutoff (the season, or the previous one) moves the
  posterior mean and a probability, so the comparisons are not vacuous.
* **Tuning-only discipline**: validation and test outcomes never reach a ``tuning_only`` report or
  its CSV.

Each check is a helper that returns the list of what changed (empty = clean). The real tests assert
it is empty; the planted-leak tests monkeypatch a leak into the harness and assert the very same
helper reports a change, which proves the helpers can fail. The same seed is used on both sides of
every comparison (the seed is the run's seed plus the checkpoint's index, and no edit changes the
checkpoint plan).
"""

import re
from collections.abc import Callable, Sequence
from dataclasses import replace
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from eurohoops.eval import m7_backtest
from eurohoops.eval.m7_backtest import (
    CheckpointInputs,
    Forecast,
    M7Inputs,
    Strengths,
    _forecasts,
    checkpoint_inputs,
    run_m7_backtest,
)
from eurohoops.logs import write_json
from eurohoops.models.team_eff import RatingPosterior
from eurohoops.sim.season import StrengthSampler
from tests.m7_synthetic import ROUNDS, build_inputs, small_spec, synthetic_formats

SEASON = 2018  # the checkpoint season: has a points deduction and three seasons of history
FRACTIONS = (0.25, 0.5, 0.75)
# Few simulations: every comparison is bit for bit, so any sample size shows a dependence.
SPEC = replace(small_spec(), n_sims=50, tuning=(SEASON,), validation=(), test=())
FULL_SPEC = replace(small_spec(), n_sims=50)  # tuning 2017-18, validation 2019, test 2020-21
INPUTS = build_inputs()
FIRST_EDITED_ROUND = 18  # deleted rounds follow every checkpoint's round k + 1 (<= 17)

# Forecast fields of ``Forecast`` (the expected rank is a function of ``rank_probs``).
FORECAST_FIELDS = ("rank_probs", "p_direct", "p_play_in", "p_top10", "p_title")
# The CSV columns that are forecasts; the others (final_place, made_direct, made_top10, champion,
# rps) are targets scored against the real final table and may change with the late results.
FORECAST_COLUMNS = ("p_direct", "p_play_in", "p_top10", "p_title", "exp_rank")
TARGET_COLUMNS = ("final_place", "made_direct", "made_top10", "champion", "rps")

Forecaster = Callable[[CheckpointInputs], dict[str, Forecast]]


# --- Edits of the inputs ----------------------------------------------------------------------


def _cutoffs() -> dict[float, pd.Timestamp]:
    return {c.checkpoint.fraction: c.checkpoint.cutoff for c in _states("base")}


def _rescore(inputs: M7Inputs, mask: Any) -> M7Inputs:
    """Other scores for the played games selected by ``mask`` (games mart and team_games points):
    the sides swap and the new home score gains 17 (never a draw)."""
    games = inputs.games.copy()
    sel = mask & games["played"]
    home, away = games["home_score"].copy(), games["away_score"].copy()
    new_home = away + 17 + ((away + 17) == home).astype("Int64")
    games.loc[sel, "home_score"] = new_home[sel]
    games.loc[sel, "away_score"] = home[sel]
    ids = set(games.loc[sel, "game_id"])
    team_games = inputs.team_games.copy()
    rows = team_games["game_id"].isin(ids)
    by_game = games.set_index("game_id")
    keys = team_games.loc[rows, "game_id"]
    home_points = by_game["home_score"].reindex(keys).to_numpy(dtype=np.int64)
    away_points = by_game["away_score"].reindex(keys).to_numpy(dtype=np.int64)
    team_games.loc[rows, "points"] = np.where(
        team_games.loc[rows, "home"], home_points, away_points
    )
    return replace(inputs, games=games, team_games=team_games)


def _remove(inputs: M7Inputs, mask: Any) -> M7Inputs:
    """The games selected by ``mask`` removed from the games mart and team_games."""
    ids = set(inputs.games.loc[mask, "game_id"])
    return replace(
        inputs,
        games=inputs.games[~inputs.games["game_id"].isin(ids)].reset_index(drop=True),
        team_games=inputs.team_games[~inputs.team_games["game_id"].isin(ids)].reset_index(
            drop=True
        ),
    )


def _add(inputs: M7Inputs, template: str, game_id: str, tipoff: pd.Timestamp) -> M7Inputs:
    """A copy of the game ``template`` as a new game ``game_id`` (round 99) at ``tipoff``."""
    row = inputs.games[inputs.games["game_id"] == template].copy()
    row["game_id"], row["round"], row["tipoff_utc"] = game_id, 99, tipoff
    rows = inputs.team_games[inputs.team_games["game_id"] == template].copy()
    rows["game_id"] = game_id
    games = pd.concat([inputs.games, row]).sort_values("tipoff_utc", kind="stable")
    return replace(
        inputs,
        games=games.reset_index(drop=True),
        team_games=pd.concat([inputs.team_games, rows], ignore_index=True),
    )


def _season_mask(inputs: M7Inputs, season: int, phase: str | None = None) -> Any:
    games = inputs.games
    return (games["season"] == season) & (games["phase"] == phase if phase else True)


def _round_mask(inputs: M7Inputs, season: int, rounds: Sequence[int]) -> Any:
    games = inputs.games
    return _season_mask(inputs, season, "RS") & games["round"].isin(rounds)


def _late_scores(inputs: M7Inputs, cutoff: pd.Timestamp) -> M7Inputs:
    """Other scores for every game at or after ``cutoff``: the season's remaining fixtures and
    playoffs, and all later seasons."""
    return _rescore(inputs, inputs.games["tipoff_utc"] >= cutoff)


def _late_games_removed(inputs: M7Inputs) -> M7Inputs:
    """The season's rounds 18-22 and its playoffs and Final Four removed (the rounds that open
    each checkpoint's remaining schedule, k + 1 <= 17, stay)."""
    games = inputs.games
    late = _round_mask(inputs, SEASON, range(FIRST_EDITED_ROUND, ROUNDS + 1))
    return _remove(inputs, late | (_season_mask(inputs, SEASON) & (games["phase"] != "RS")))


def _playoffs_removed(inputs: M7Inputs) -> M7Inputs:
    return _remove(inputs, _season_mask(inputs, SEASON) & (inputs.games["phase"] != "RS"))


def _game_added(inputs: M7Inputs, cutoff: pd.Timestamp) -> M7Inputs:
    """A new fixture of the season one hour after ``cutoff`` (inside round k + 1's week, so the
    cutoff itself stays the first tip-off of round k + 1)."""
    games = inputs.games
    last_round = games[_round_mask(inputs, SEASON, [ROUNDS])].iloc[0]
    return _add(inputs, str(last_round["game_id"]), "E2018_9001", cutoff + pd.Timedelta(hours=1))


def _later_seasons_edited(inputs: M7Inputs) -> M7Inputs:
    """Later seasons: 2019 rescored, half of 2020 removed, a game added to 2021."""
    games = inputs.games
    rescored = _rescore(inputs, _season_mask(inputs, 2019))
    half = _round_mask(rescored, 2020, range(1, 12)) | _season_mask(rescored, 2020, "PO")
    halved = _remove(rescored, half)
    template = games[_round_mask(inputs, 2021, [ROUNDS])].iloc[0]
    end = games.loc[_season_mask(inputs, 2021), "tipoff_utc"].max()
    return _add(halved, str(template["game_id"]), "E2021_9001", end + pd.Timedelta(days=1))


def _earlier_scores(inputs: M7Inputs, season: int, rounds: Sequence[int]) -> M7Inputs:
    return _rescore(inputs, _round_mask(inputs, season, rounds))


EDITS: dict[str, Callable[[M7Inputs], M7Inputs]] = {
    "base": lambda inputs: inputs,
    "deleted": _late_games_removed,
    "playoffs_removed": _playoffs_removed,
    "later_seasons": _later_seasons_edited,
    "earlier_in_season": lambda inputs: _earlier_scores(inputs, SEASON, [1, 2]),
    "previous_season": lambda inputs: _earlier_scores(inputs, SEASON - 1, [19, 20, 21, 22]),
}


# --- The checks -------------------------------------------------------------------------------


@cache
def _states(edit: str, fraction: float | None = None) -> tuple[CheckpointInputs, ...]:
    """The checkpoints of the league under a named edit; the score and addition edits are made
    relative to a checkpoint's cutoff, so they name its ``fraction``."""
    if edit == "scores":
        edited = _late_scores(INPUTS, _cutoffs()[fraction or 0.0])
    elif edit == "added":
        edited = _game_added(INPUTS, _cutoffs()[fraction or 0.0])
    else:
        edited = EDITS[edit](INPUTS)
    return tuple(checkpoint_inputs(edited, SPEC, synthetic_formats))


def _at(states: Sequence[CheckpointInputs], fraction: float) -> CheckpointInputs:
    (found,) = [c for c in states if c.checkpoint.fraction == fraction]
    return found


def _same(a: Any, b: Any) -> bool:
    """Bit-identical: floats compare with NaN equal to NaN, anything else by equality."""
    if isinstance(a, np.ndarray | float):
        return bool(np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True))
    return bool(a == b)


def input_differences(
    before: CheckpointInputs, after: CheckpointInputs, *, schedule_free: bool = True
) -> list[str]:
    """Names of the checkpoint inputs that are not bit-identical. ``schedule_free`` leaves out
    what the remaining schedule legitimately moves: the schedule itself and the noise scales,
    which average over the remaining games."""
    assert before.checkpoint == after.checkpoint, "the checkpoint plan must not change"
    pairs: dict[str, tuple[Any, Any]] = {
        "posterior.labels": (before.posterior.labels, after.posterior.labels),
        "posterior.mean": (before.posterior.mean, after.posterior.mean),
        "posterior.cov": (before.posterior.cov, after.posterior.cov),
        "posterior.sigma2": (before.posterior.sigma2, after.posterior.sigma2),
        "strengths.mean": (before.strengths.mean, after.strengths.mean),
        "strengths.cov": (before.strengths.cov, after.strengths.cov),
        "pace.mu": (before.pace.mu, after.pace.mu),
        "pace.team": (before.pace.team, after.pace.team),
        "elo_ratings": (before.elo_ratings, after.elo_ratings),
        "played": (before.played, after.played),
    }
    changed = [name for name, (a, b) in pairs.items() if not _same(a, b)]
    if not schedule_free:
        changed += [
            name
            for name, same in {
                "noise": before.noise == after.noise,
                "net": before.net == after.net,
                "remaining": before.remaining.equals(after.remaining),
            }.items()
            if not same
        ]
    return changed


def forecast_differences(
    before: CheckpointInputs,
    after: CheckpointInputs,
    forecaster: Forecaster = lambda ci: _forecasts(SPEC, ci),
) -> list[str]:
    """``model.field`` of every forecast (unrounded) that is not bit-identical."""
    old, new = forecaster(before), forecaster(after)
    return [
        f"{model}.{field}"
        for model in old
        for field in FORECAST_FIELDS
        if not _same(getattr(old[model], field), getattr(new[model], field))
    ]


def score_edit_differences(fraction: float) -> tuple[list[str], list[str]]:
    """Input and forecast differences of one checkpoint when every result at or after its cutoff
    is rescored."""
    before, after = _at(_states("base"), fraction), _at(_states("scores", fraction), fraction)
    return input_differences(before, after, schedule_free=False), forecast_differences(
        before, after
    )


# --- Edits at or after the cutoff change nothing ----------------------------------------------


def test_the_edits_are_real_and_keep_the_checkpoint_plan() -> None:
    base = _states("base")
    assert [c.checkpoint.fraction for c in base] == list(FRACTIONS)
    for fraction in FRACTIONS:
        edited = _late_scores(INPUTS, _cutoffs()[fraction])
        cut = INPUTS.games["tipoff_utc"] >= _cutoffs()[fraction]
        assert (edited.games.loc[cut, "home_score"] != INPUTS.games.loc[cut, "home_score"]).any()
        assert (edited.games.loc[~cut, "home_score"] == INPUTS.games.loc[~cut, "home_score"]).all()
        assert [c.checkpoint for c in _states("scores", fraction)] == [c.checkpoint for c in base]
    assert len(_states("deleted")) == len(_states("added", 0.25)) == len(base)
    assert len(_late_games_removed(INPUTS).games) < len(INPUTS.games)
    assert len(_game_added(INPUTS, _cutoffs()[0.25]).games) == len(INPUTS.games) + 1


@pytest.mark.parametrize("fraction", FRACTIONS)
def test_late_scores_leave_the_inputs_and_every_forecast_bit_identical(fraction: float) -> None:
    inputs, forecasts = score_edit_differences(fraction)
    assert inputs == []
    assert forecasts == []


@pytest.mark.parametrize("fraction", FRACTIONS)
def test_late_scores_move_only_the_targets_of_the_csv(fraction: float) -> None:
    edited = _late_scores(INPUTS, _cutoffs()[fraction])
    (_, before), (_, after) = (
        run_m7_backtest(i, spec=SPEC, formats=synthetic_formats) for i in (INPUTS, edited)
    )

    def one(rows: pd.DataFrame) -> pd.DataFrame:
        return rows[rows["checkpoint"] == fraction].reset_index(drop=True)

    old, new = one(before), one(after)
    assert len(old) == len(new) > 0
    for column in FORECAST_COLUMNS:
        assert old[column].equals(new[column]), column
    # the targets are scored against the real final table, which the edit does change
    assert any(not old[c].equals(new[c]) for c in TARGET_COLUMNS)


@pytest.mark.parametrize("edit", ["deleted", "playoffs_removed"])
@pytest.mark.parametrize("fraction", FRACTIONS)
def test_deleted_games_leave_the_posterior_pace_elo_and_played_bit_identical(
    edit: str, fraction: float
) -> None:
    assert input_differences(_at(_states("base"), fraction), _at(_states(edit), fraction)) == []


@pytest.mark.parametrize("fraction", FRACTIONS)
def test_an_added_game_leaves_the_posterior_pace_elo_and_played_bit_identical(
    fraction: float,
) -> None:
    before, after = _at(_states("base"), fraction), _at(_states("added", fraction), fraction)
    assert input_differences(before, after) == []
    assert not before.remaining.equals(after.remaining)  # the new fixture is in the schedule


@pytest.mark.parametrize("fraction", FRACTIONS)
def test_removed_playoffs_leave_every_forecast_bit_identical(fraction: float) -> None:
    before, after = _at(_states("base"), fraction), _at(_states("playoffs_removed"), fraction)
    assert input_differences(before, after, schedule_free=False) == []
    assert forecast_differences(before, after) == []


@pytest.mark.parametrize("fraction", FRACTIONS)
def test_edits_of_later_seasons_leave_inputs_and_every_forecast_bit_identical(
    fraction: float,
) -> None:
    before, after = _at(_states("base"), fraction), _at(_states("later_seasons"), fraction)
    assert input_differences(before, after, schedule_free=False) == []
    assert forecast_differences(before, after) == []


def test_the_remaining_schedule_is_an_input() -> None:
    """The exception to 'nothing after the cutoff': deleting remaining fixtures moves the net noise
    scale (it averages over the remaining games), which is allowed and is why the deletion tests
    leave the noise scales out."""
    before, after = _at(_states("base"), 0.25), _at(_states("deleted"), 0.25)
    assert len(after.remaining) < len(before.remaining)
    assert "remaining" in input_differences(before, after, schedule_free=False)
    assert forecast_differences(before, after)  # fewer fixtures to play, other probabilities


# --- Earlier results do matter ----------------------------------------------------------------


@pytest.mark.parametrize("edit", ["earlier_in_season", "previous_season"])
@pytest.mark.parametrize("fraction", FRACTIONS)
def test_earlier_results_move_the_posterior_and_the_probabilities(
    edit: str, fraction: float
) -> None:
    before, after = _at(_states("base"), fraction), _at(_states(edit), fraction)
    assert "posterior.mean" in input_differences(before, after)
    assert forecast_differences(before, after)
    edited = EDITS[edit](INPUTS)
    early = edited.games["tipoff_utc"] < before.checkpoint.cutoff
    assert (edited.games.loc[~early, "home_score"] == INPUTS.games.loc[~early, "home_score"]).all()


# --- Planted leaks are caught -----------------------------------------------------------------


def _sampler_reading_the_final_table(ci: CheckpointInputs) -> dict[str, Forecast]:
    """The forecasts with a leak: the strength sampler shifts every team's offence by its final
    place of the checkpoint season, taken from all the season's games."""
    teams, place = ci.info.teams, ci.info.place
    shift = 0.5 * np.array([len(teams) / 2.0 - place[team] for team in teams])
    honest = m7_backtest.gaussian_sampler

    def leaky(strengths: Strengths, spread: float) -> StrengthSampler:
        sample = honest(strengths, spread)

        def draw(
            n_sims: int, rng: np.random.Generator
        ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
            off, def_, home = sample(n_sims, rng)
            return off + shift, def_, home

        return draw

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(m7_backtest, "gaussian_sampler", leaky)
        return _forecasts(SPEC, ci)


def test_a_sampler_reading_the_final_table_is_caught() -> None:
    fraction = 0.5
    before, after = _at(_states("base"), fraction), _at(_states("scores", fraction), fraction)
    leaks = forecast_differences(before, after, _sampler_reading_the_final_table)
    assert any(name.startswith("sim_full.") for name in leaks)
    assert "sim_full.p_direct" in leaks
    # while the honest sampler passes the same check on the same pair
    assert forecast_differences(before, after) == []


@pytest.fixture
def fitted_through_the_season_end(monkeypatch: pytest.MonkeyPatch) -> None:
    """The posterior is fitted on every game up to one second after the checkpoint season's last
    tip-off (regular season, playoffs and Final Four) instead of up to the cutoff."""
    last = INPUTS.games.groupby("season")["tipoff_utc"].max() + pd.Timedelta(seconds=1)
    honest = m7_backtest.rating_posteriors

    def leaky(
        history: Any, params: Any, cutoffs: Sequence[tuple[float, int]]
    ) -> list[RatingPosterior]:
        through = [(float(last[season].as_unit("ns").value) / 1e9, season) for _, season in cutoffs]
        return honest(history, params, through)

    monkeypatch.setattr(m7_backtest, "rating_posteriors", leaky)


def test_a_posterior_fitted_through_the_season_end_is_caught(
    fitted_through_the_season_end: None,
) -> None:
    fraction = 0.5
    # check 1: the late-score edit
    reference = _at(checkpoint_inputs(INPUTS, SPEC, synthetic_formats), fraction)
    edited = _late_scores(INPUTS, reference.checkpoint.cutoff)
    scored = _at(checkpoint_inputs(edited, SPEC, synthetic_formats), fraction)
    assert "posterior.mean" in input_differences(reference, scored, schedule_free=False)
    assert forecast_differences(reference, scored)
    # check 2: removing the season's playoffs and Final Four
    removed = _at(checkpoint_inputs(_playoffs_removed(INPUTS), SPEC, synthetic_formats), fraction)
    assert "posterior.mean" in input_differences(reference, removed)
    # the same checks pass on the honest harness (the tests above)


# --- Validation and test outcomes never reach a tuning-only report ------------------------------


def _tokens(value: Any) -> list[Any]:
    """Every dict key and string value of a JSON tree."""
    if isinstance(value, dict):
        return [t for k, v in value.items() for t in [k, *_tokens(v)]]
    if isinstance(value, list):
        return [t for v in value for t in _tokens(v)]
    return [value] if isinstance(value, str) else []


# a season number on its own (not part of a number, a decimal or a calendar date: the cutoffs of
# the 2018 season fall in 2019)
LATER_SEASON = re.compile(r"(?<![\d.-])(2019|2020|2021)(?![\d-])")


def _written(report: dict[str, Any], rows: pd.DataFrame, directory: Path) -> tuple[bytes, bytes]:
    """The JSON and CSV bytes of a run as the CLI writes them; the JSON without ``data_sha256``,
    which hashes every input game by design (so it moves with any edit, validation and test
    games included, and is the one key excluded from the comparison)."""
    write_json(directory / "report.json", {k: v for k, v in report.items() if k != "data_sha256"})
    rows.to_csv(directory / "rows.csv", index=False, lineterminator="\n")
    return (directory / "report.json").read_bytes(), (directory / "rows.csv").read_bytes()


def test_validation_and_test_outcomes_never_reach_a_tuning_only_report(tmp_path: Path) -> None:
    spec = FULL_SPEC
    held_out = INPUTS.games["season"].isin([*spec.validation, *spec.test])
    edited = _rescore(INPUTS, held_out)
    assert (
        edited.games.loc[held_out, "home_score"] != INPUTS.games.loc[held_out, "home_score"]
    ).all()

    def tuning(inputs: M7Inputs) -> tuple[dict[str, Any], pd.DataFrame]:
        return run_m7_backtest(inputs, spec=spec, formats=synthetic_formats, tuning_only=True)

    (report, rows), (other, other_rows) = tuning(INPUTS), tuning(edited)
    # data_sha256 hashes all input games by design, so it is the one thing that moves
    assert report["data_sha256"] != other["data_sha256"]
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    json_a, csv_a = _written(report, rows, tmp_path / "a")
    json_b, csv_b = _written(other, other_rows, tmp_path / "b")
    assert json_a == json_b
    assert csv_a == csv_b
    for text in (json_a.decode(), csv_a.decode()):
        assert not LATER_SEASON.search(text)
    assert "validation" not in _tokens(report) and "test" not in _tokens(report)
    assert set(rows["split"]) == {"tuning"}
    # the edit does reach a run that scores validation, so the identity above is not vacuous
    full = run_m7_backtest(INPUTS, spec=spec, formats=synthetic_formats)
    full_edited = run_m7_backtest(edited, spec=spec, formats=synthetic_formats)
    assert full[0]["metrics"]["validation"] != full_edited[0]["metrics"]["validation"]
    assert full[0]["metrics"]["tuning"] == full_edited[0]["metrics"]["tuning"]
