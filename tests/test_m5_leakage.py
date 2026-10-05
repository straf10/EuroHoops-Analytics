"""Leakage tests for M5 (week 14-16 J5, decision D6): no forecast for a game g uses g itself or
anything at or after g's round cutoff (``minutes.round_cutoffs``: the first tip-off of g's
``(season, phase, round)``).

The guarantee (D6):

* ``exp_margin`` and ``exp_total`` of every game never depend on that game or on any game at or
  after its cutoff (any season, tuning included);
* ``p_home``, ``margin_sigma``, ``margin_df`` and ``total_sigma`` of validation and test games never
  depend on a validation or test outcome at or after the game's cutoff. (The distribution is fit
  in-sample on tuning, so a tuning game's ``p_home`` may depend on later *tuning* outcomes; that is
  the declared exception and is not tested against. What is tested is that no validation or test
  outcome reaches a tuning-fit number.)

Everything runs on the synthetic league of ``tests/m5_synthetic.py`` with a fixed seed. Comparisons
are made on the frame ``predict_games`` returns, which is rounded to 6 decimals, so a dependence
smaller than 5e-7 is invisible; ``np.array_equal`` (NaN equal to NaN) is applied to those rounded
values. Each check is a helper that returns the list of ``column:game_id`` that changed (empty =
clean). The real tests assert it is empty; the planted-leak tests monkeypatch a leak into the model
and assert the very same helper reports a change, which proves the helpers can fail.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from functools import cache
from typing import Any

import numpy as np
import pandas as pd
import pytest

from eurohoops.eval import m5_backtest
from eurohoops.eval.m5_backtest import Choice, M5Inputs, predict_games, run_m5_backtest
from eurohoops.models import minutes
from eurohoops.parse.games import conform
from tests.conftest import make_team_games
from tests.m5_synthetic import build_inputs, fake_player_part, small_spec

SPEC = small_spec()
SEED = 0

# Every form, every shares variant, both totals variants and Platt on/off, on the small grid.
CORE = Choice("proj_hc", None, "core", 180.0, 10.0, None, True, "total_m1")
CORE_REST = Choice("proj_decay", 3.0, "core_rest", 180.0, 40.0, 25.0, True, "total_m1_rest")
BLEND = Choice("proj_avail", 3.0, "blend", 180.0, 10.0, 25.0, False, "total_m1_rest")
BLEND_HC = Choice("proj_hc", None, "blend", 180.0, 40.0, 25.0, False, "total_m1")
CHOICES = (CORE, CORE_REST, BLEND, BLEND_HC)

# Target games (round 8 of 2019, round 6 of 2020, round 9 of 2022; 4 games per round, so each has
# round mates both before and after it in tip-off order). Not the forfeit or the unplayed game.
TUNING_GAME = "E2019_31"
VALIDATION_GAME = "E2020_23"
TEST_GAME = "E2022_35"
TARGETS = (TUNING_GAME, VALIDATION_GAME, TEST_GAME)

EXP_COLUMNS = ("exp_margin", "exp_total")
DIST_COLUMNS = ("p_home", "margin_sigma", "margin_df", "total_sigma")

EpochSeconds = np.ndarray[Any, np.dtype[np.float64]]


@cache
def _base() -> M5Inputs:
    return build_inputs(SEED)


def _tip(games: pd.DataFrame) -> EpochSeconds:
    seconds = (games["tipoff_utc"] - pd.Timestamp(0, tz="UTC")).dt.total_seconds()
    return seconds.to_numpy(dtype=np.float64)


def _predict(inputs: M5Inputs, choice: Choice, *, oracle: bool = False) -> pd.DataFrame:
    return predict_games(
        inputs, spec=SPEC, player_part=fake_player_part, choice=choice, oracle=oracle
    )


@cache
def _reference(choice: Choice, oracle: bool, mode: str) -> pd.DataFrame:
    """The unedited league's frame. ``mode`` names the planted leak active during the call ('' =
    none), so that the cache never mixes a leaky reference with a clean one."""
    return _predict(_base(), choice, oracle=oracle)


def _is_tuning(game_id: str) -> bool:
    games = _base().games
    season = int(games.loc[games["game_id"] == game_id, "season"].iloc[0])
    return season in SPEC.tuning


def _guarded_columns(game_id: str) -> tuple[str, ...]:
    """Columns D6 guarantees for ``game_id``: the distribution columns only outside tuning."""
    return EXP_COLUMNS if _is_tuning(game_id) else (*EXP_COLUMNS, *DIST_COLUMNS)


def _differences(
    before: pd.DataFrame, after: pd.DataFrame, ids: set[str], columns: tuple[str, ...]
) -> list[str]:
    """``column:game_id`` of every guaranteed value that is not bit-identical (rounded frame)."""
    shared = sorted(ids & set(before["game_id"]) & set(after["game_id"]))
    assert shared, "nothing to compare"
    old = before.set_index("game_id").loc[shared]
    new = after.set_index("game_id").loc[shared]
    changed = []
    for column in columns:
        a = old[column].to_numpy(dtype=np.float64)
        b = new[column].to_numpy(dtype=np.float64)
        same = (a == b) | (np.isnan(a) & np.isnan(b))
        changed += [f"{column}:{game_id}" for game_id in np.asarray(shared)[~same]]
    return changed


# --- Edits of the inputs ----------------------------------------------------------------------


def _ids(games: pd.DataFrame, mask: Any) -> set[str]:
    return set(games.loc[mask, "game_id"].astype(str))


def _late_ids(inputs: M5Inputs, game_id: str) -> set[str]:
    """Target-competition games at or after the round cutoff of ``game_id`` (g and its round mates
    included)."""
    games = inputs.games
    cutoff = float(minutes.round_cutoffs(games)[games["game_id"] == game_id].iloc[0])
    return _ids(games, _tip(games) >= cutoff)


def _rescore(inputs: M5Inputs, ids: set[str]) -> M5Inputs:
    """Different scores for the played games in ``ids`` (games mart and team_games points): the
    sides swap and the new home score gains 17 (never a draw)."""
    games = inputs.games.copy()
    sel = games["game_id"].isin(ids) & games["played"]
    home, away = games["home_score"].copy(), games["away_score"].copy()
    new_home = away + 17 + ((away + 17) == home).astype("Int64")
    games.loc[sel, "home_score"] = new_home[sel]
    games.loc[sel, "away_score"] = home[sel]
    team_games = inputs.team_games.copy()
    rows = team_games["game_id"].isin(ids)
    by_game = games.set_index("game_id")
    keys = team_games.loc[rows, "game_id"]
    home_points = by_game["home_score"].reindex(keys).to_numpy(dtype=np.int64)
    away_points = by_game["away_score"].reindex(keys).to_numpy(dtype=np.int64)
    team_games.loc[rows, "points"] = np.where(
        team_games.loc[rows, "home"], home_points, away_points
    )
    return M5Inputs(
        games, team_games, inputs.player_games, inputs.other_games, inputs.club_map,
        inputs.tuned_m1, inputs.elo,
    )  # fmt: skip


def _reseconds(inputs: M5Inputs, ids: set[str]) -> M5Inputs:
    """Different minutes for every player row of the games in ``ids``."""
    player_games = inputs.player_games.copy()
    rows = player_games["game_id"].isin(ids).to_numpy()
    bump = 90 * (np.arange(int(rows.sum())) % 4)
    player_games.loc[rows, "sec"] = player_games.loc[rows, "sec"].to_numpy() * 2 + bump
    return M5Inputs(
        inputs.games, inputs.team_games, player_games, inputs.other_games, inputs.club_map,
        inputs.tuned_m1, inputs.elo,
    )  # fmt: skip


def _remove(inputs: M5Inputs, ids: set[str]) -> M5Inputs:
    """The games in ``ids`` removed from the games mart, team_games and player_games."""
    games = inputs.games[~inputs.games["game_id"].isin(ids)].reset_index(drop=True)
    return M5Inputs(
        games,
        inputs.team_games[~inputs.team_games["game_id"].isin(ids)].reset_index(drop=True),
        inputs.player_games[~inputs.player_games["game_id"].isin(ids)].reset_index(drop=True),
        inputs.other_games, inputs.club_map, inputs.tuned_m1, inputs.elo,
    )  # fmt: skip


def edit_scores(inputs: M5Inputs, game_id: str) -> M5Inputs:
    return _rescore(inputs, _late_ids(inputs, game_id))


def edit_seconds(inputs: M5Inputs, game_id: str) -> M5Inputs:
    return _reseconds(inputs, _late_ids(inputs, game_id))


def edit_delete(inputs: M5Inputs, game_id: str) -> M5Inputs:
    """Delete g's round mates and every later game of g's season (g itself stays)."""
    games = inputs.games
    season = games.loc[games["game_id"] == game_id, "season"].iloc[0]
    doomed = _late_ids(inputs, game_id) & _ids(games, games["season"] == season)
    # The game that tips off first in g's round defines g's cutoff: deleting it would move the
    # cutoff itself (and with it the decay clock), which is not an edit "at or after" the cutoff.
    cutoffs = minutes.round_cutoffs(games).to_numpy(dtype=np.float64)
    defining = _ids(games, _tip(games) == cutoffs[(games["game_id"] == game_id).to_numpy()][0])
    return _remove(inputs, doomed - {game_id} - defining)


def edit_add(inputs: M5Inputs, game_id: str) -> M5Inputs:
    """A new game (round 99, a lopsided score) three days after the last game of g's season."""
    games = inputs.games
    template = games[games["game_id"] == game_id].copy()
    season = template["season"].iloc[0]
    last = games.loc[games["season"] == season, "tipoff_utc"].max()
    new_id = f"{game_id}_new"
    template = template.assign(
        game_id=new_id, game_code=9999, round=99, round_label="Round 99",
        tipoff_utc=last + pd.Timedelta(days=3), home_score=160, away_score=40,
        played=True, forfeit=False,
    )  # fmt: skip
    player_games = inputs.player_games[inputs.player_games["game_id"] == game_id].copy()
    return M5Inputs(
        conform(pd.concat([games, template], ignore_index=True)),
        pd.concat([inputs.team_games, make_team_games(template, seed=5)], ignore_index=True),
        pd.concat([inputs.player_games, player_games.assign(game_id=new_id)], ignore_index=True),
        inputs.other_games, inputs.club_map, inputs.tuned_m1, inputs.elo,
    )  # fmt: skip


def edit_other(inputs: M5Inputs, game_id: str) -> M5Inputs:
    """The other competition after g's tip-off: scores changed, tip-offs moved a day later and
    every third game deleted."""
    assert inputs.other_games is not None
    games = inputs.games
    tip = float(_tip(games)[(games["game_id"] == game_id).to_numpy()][0])
    other = inputs.other_games.copy()
    later = _tip(other) > tip
    other.loc[later, "tipoff_utc"] = other.loc[later, "tipoff_utc"] + pd.Timedelta(days=1)
    swap = other.loc[later, "home_score"].copy()
    other.loc[later, "home_score"] = other.loc[later, "away_score"] + 9
    other.loc[later, "away_score"] = swap
    drop = np.zeros(len(other), dtype=bool)
    drop[np.flatnonzero(later)[::3]] = True
    other = other[~drop].reset_index(drop=True)
    return M5Inputs(
        inputs.games, inputs.team_games, inputs.player_games, other,
        inputs.club_map, inputs.tuned_m1, inputs.elo,
    )  # fmt: skip


def edit_all(inputs: M5Inputs, game_id: str) -> M5Inputs:
    for edit in (edit_scores, edit_seconds, edit_delete, edit_add, edit_other):
        inputs = edit(inputs, game_id)
    return inputs


EDITS: dict[str, Callable[[M5Inputs, str], M5Inputs]] = {
    "scores": edit_scores,
    "seconds": edit_seconds,
    "delete": edit_delete,
    "add": edit_add,
    "other_competition": edit_other,
    "all": edit_all,
}


# --- The checks (each returns the list of changed values; empty means no leak) ----------------


def future_edit_diff(
    choice: Choice, game_id: str, kind: str, *, mode: str = "", oracle: bool = False
) -> list[str]:
    """What changes for the games whose information was not touched when the games at or after
    ``game_id``'s cutoff are edited (``kind``). The checked set is every game whose own cutoff is
    at or before g's (g, its round mates, everything earlier), so one run covers many games; for
    ``other_competition`` it is every game that tips off at or before g (the edit is after g's
    tip-off). The distribution columns are checked only when g is a validation or test game."""
    base = _base()
    games = base.games
    if kind == "other_competition":
        keep = _tip(games) <= float(_tip(games)[(games["game_id"] == game_id).to_numpy()][0])
    else:
        cutoffs = minutes.round_cutoffs(games).to_numpy(dtype=np.float64)
        cutoff = float(cutoffs[(games["game_id"] == game_id).to_numpy()][0])
        keep = cutoffs <= cutoff
    ids = _ids(games, keep)
    after = _predict(EDITS[kind](base, game_id), choice, oracle=oracle)
    return _differences(_reference(choice, oracle, mode), after, ids, _guarded_columns(game_id))


def own_event_diff(
    choice: Choice, game_id: str, *, mode: str = "", oracle: bool = False
) -> list[str]:
    """What changes for g when g's own player_games and team_games rows are deleted and its score
    is set to something else (the games-mart row stays, so g is still forecast)."""
    base = _base()
    games = base.games.copy()
    own = games["game_id"] == game_id
    games.loc[own, ["home_score", "away_score"]] = [150, 31]
    edited = M5Inputs(
        games,
        base.team_games[base.team_games["game_id"] != game_id].reset_index(drop=True),
        base.player_games[base.player_games["game_id"] != game_id].reset_index(drop=True),
        base.other_games, base.club_map, base.tuned_m1, base.elo,
    )  # fmt: skip
    return _differences(
        _reference(choice, oracle, mode),
        _predict(edited, choice, oracle=oracle),
        {game_id},
        _guarded_columns(game_id),
    )


def tuning_fit_diff(choice: Choice, *, mode: str = "") -> list[str]:
    """What changes when every validation and test outcome is rewritten: the numbers fit on tuning
    (``margin_sigma``, ``margin_df``, ``total_sigma`` of every game) and everything of the tuning
    seasons themselves (``p_home``, ``exp_margin``, ``exp_total``)."""
    base = _base()
    games = base.games
    later = _ids(games, ~games["season"].isin(SPEC.tuning) & (games["season"] > SPEC.tuning[-1]))
    before = _reference(choice, False, mode)
    after = _predict(_rescore(base, later), choice)
    assert (before["margin_sigma"] == before["margin_sigma"].iloc[0]).all()  # one for every game
    every = set(before["game_id"])
    tuning = _ids(before, before["season"].isin(SPEC.tuning))
    return _differences(
        before, after, every, ("margin_sigma", "margin_df", "total_sigma")
    ) + _differences(before, after, tuning, ("p_home", *EXP_COLUMNS))


# --- 1. Future edits --------------------------------------------------------------------------

KINDS = ("scores", "seconds", "delete", "add", "other_competition")
# Every (edit, target) pair once, the choice rotating so each form, shares variant and totals
# variant meets each edit kind and each target split without running the full cross product.
MATRIX = [
    pytest.param(kind, target, CHOICES[(i + j) % len(CHOICES)], id=f"{kind}-{target}")
    for i, kind in enumerate(KINDS)
    for j, target in enumerate(TARGETS)
]


@pytest.mark.parametrize(("kind", "target", "choice"), MATRIX)
def test_future_edits_change_nothing_guaranteed(kind: str, target: str, choice: Choice) -> None:
    assert future_edit_diff(choice, target, kind) == []


@pytest.mark.parametrize("choice", CHOICES, ids=lambda c: f"{c.form}-{c.shares}")
def test_every_future_edit_at_once_changes_nothing_guaranteed(choice: Choice) -> None:
    target = TARGETS[CHOICES.index(choice) % len(TARGETS)]
    assert future_edit_diff(choice, target, "all") == []


# --- 2. Own events ----------------------------------------------------------------------------

OWN = [
    pytest.param(choice, target, id=f"{target}-{choice.form}-{choice.shares}")
    for target, choices in (
        (TUNING_GAME, (CORE, BLEND)),
        (VALIDATION_GAME, (CORE_REST, BLEND_HC)),
        (TEST_GAME, (CORE, BLEND)),
    )
    for choice in choices
]


@pytest.mark.parametrize(("choice", "target"), OWN)
def test_own_events_change_nothing_in_projected_mode(choice: Choice, target: str) -> None:
    assert own_event_diff(choice, target) == []


@pytest.mark.parametrize(
    ("choice", "target"), [(CORE, VALIDATION_GAME), (BLEND, TEST_GAME)], ids=["core", "blend"]
)
def test_own_events_change_the_oracle(choice: Choice, target: str) -> None:
    """The oracle may see the game: removing g's minutes changes g's exp_margin (so the check
    above is not blind to the very input an oracle would read)."""
    diff = own_event_diff(choice, target, oracle=True)
    assert f"exp_margin:{target}" in diff


# --- 3. Sensitivity: an edit before g's cutoff must change g ----------------------------------


def _earlier_ids(game_id: str) -> set[str]:
    """Games of g's season in rounds 1-3 (g is in round 6 or later, so before its cutoff)."""
    games = _base().games
    season = games.loc[games["game_id"] == game_id, "season"].iloc[0]
    earlier = (games["season"] == season) & (games["round"] <= 3)
    assert float(_tip(games)[earlier.to_numpy()].max()) < float(
        minutes.round_cutoffs(games)[games["game_id"] == game_id].iloc[0]
    )
    return _ids(games, earlier)


def _changed_value(before: pd.DataFrame, after: pd.DataFrame, game_id: str, column: str) -> bool:
    a = before.loc[before["game_id"] == game_id, column].to_numpy(dtype=np.float64)
    b = after.loc[after["game_id"] == game_id, column].to_numpy(dtype=np.float64)
    return not np.array_equal(a, b, equal_nan=True)


@pytest.mark.parametrize("choice", CHOICES, ids=lambda c: f"{c.form}-{c.shares}")
def test_an_earlier_score_edit_changes_exp_margin(choice: Choice) -> None:
    after = _predict(_rescore(_base(), _earlier_ids(VALIDATION_GAME)), choice)
    assert _changed_value(_reference(choice, False, ""), after, VALIDATION_GAME, "exp_margin")


def test_an_earlier_minutes_edit_changes_exp_margin() -> None:
    after = _predict(_reseconds(_base(), _earlier_ids(VALIDATION_GAME)), CORE)
    assert _changed_value(_reference(CORE, False, ""), after, VALIDATION_GAME, "exp_margin")


def test_an_earlier_possessions_edit_changes_exp_total() -> None:
    base = _base()
    team_games = base.team_games.copy()
    rows = team_games["game_id"].isin(_earlier_ids(VALIDATION_GAME))
    team_games.loc[rows, ["poss_raw", "poss_game"]] += 8.0
    edited = M5Inputs(
        base.games, team_games, base.player_games, base.other_games, base.club_map,
        base.tuned_m1, base.elo,
    )  # fmt: skip
    after = _predict(edited, CORE)
    assert _changed_value(_reference(CORE, False, ""), after, VALIDATION_GAME, "exp_total")


def test_a_rest_relevant_edit_changes_exp_margin() -> None:
    """Moving the other competition's games before g (rest features of g's club) changes the
    rest-aware forms, so the other-competition check above is capable of failing."""
    base = _base()
    games = base.games
    row = games[games["game_id"] == VALIDATION_GAME].iloc[0]
    assert base.other_games is not None
    other = base.other_games.copy()
    near = (other["season"] == row["season"]) & (other["round"] == row["round"] - 1)
    other.loc[near, "tipoff_utc"] = row["tipoff_utc"] - pd.Timedelta(hours=20)
    edited = M5Inputs(
        games, base.team_games, base.player_games, other, base.club_map, base.tuned_m1, base.elo
    )
    after = _predict(edited, CORE_REST)
    reference = _reference(CORE_REST, False, "")
    assert not np.array_equal(
        reference["exp_margin"].to_numpy(), after["exp_margin"].to_numpy(), equal_nan=True
    )


# --- 4. Planted leaks: the helpers can fail ---------------------------------------------------


def _own_minutes_leak(
    variant: str, games: pd.DataFrame, player_games: pd.DataFrame, **_: Any
) -> pd.DataFrame:
    """A leak: each game's own minutes as its 'projected' shares."""
    return minutes.oracle_shares(games, player_games)


def test_planted_leak_is_detected_by_the_own_event_check(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(m5_backtest, "projected_shares_variant", _own_minutes_leak)
    diff = own_event_diff(CORE, VALIDATION_GAME, mode="own_minutes")
    assert f"exp_margin:{VALIDATION_GAME}" in diff


def test_planted_leak_is_detected_by_the_future_edit_check(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(m5_backtest, "projected_shares_variant", _own_minutes_leak)
    diff = future_edit_diff(CORE, VALIDATION_GAME, "seconds", mode="own_minutes")
    assert f"exp_margin:{VALIDATION_GAME}" in diff


def test_the_unpatched_checks_are_clean_on_the_same_targets() -> None:
    """The control of the two tests above: same helpers, same game, no leak."""
    assert own_event_diff(CORE, VALIDATION_GAME) == []
    assert future_edit_diff(CORE, VALIDATION_GAME, "seconds") == []


# --- 5. Validation and test outcomes never reach tuning-fit numbers ---------------------------


@pytest.mark.parametrize("choice", CHOICES, ids=lambda c: f"{c.form}-{c.shares}")
def test_validation_and_test_outcomes_do_not_reach_tuning_fit_numbers(choice: Choice) -> None:
    assert tuning_fit_diff(choice) == []


def test_planted_fit_on_validation_is_detected_by_the_tuning_fit_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A leak: the margin distribution and total sigma are fit on every rated game, not tuning."""

    def fit_everywhere(ctx: Any, margin: Any) -> Any:
        return ctx.rated & ctx.comparison_ok & np.isfinite(margin)

    monkeypatch.setattr(m5_backtest, "_fit_mask", fit_everywhere)
    diff = tuning_fit_diff(CORE, mode="fit_everywhere")
    assert any(item.startswith("margin_sigma:") for item in diff)


def _tuning_only_report(inputs: M5Inputs) -> tuple[str, pd.DataFrame]:
    """The tuning-only report (without ``data_sha256``, a hash of every input table, which of
    course moves when a validation score does) as canonical JSON, and its per-game frame."""
    report, frame = run_m5_backtest(
        inputs, spec=SPEC, player_part=fake_player_part, tuning_only=True
    )
    report = {key: value for key, value in report.items() if key != "data_sha256"}
    return json.dumps(report, sort_keys=True, default=str), frame


@cache
def _base_tuning_only() -> tuple[str, pd.DataFrame]:
    return _tuning_only_report(_base())


@cache
def _tuning_only_after_later_outcomes() -> tuple[str, pd.DataFrame]:
    """The tuning-only report after every validation and test outcome was rewritten."""
    base = _base()
    later = _ids(base.games, base.games["season"] > SPEC.tuning[-1])
    return _tuning_only_report(_rescore(base, later))


def test_tuning_only_verdict_cannot_see_validation_or_test() -> None:
    report_before, frame_before = _base_tuning_only()
    report_after, frame_after = _tuning_only_after_later_outcomes()
    assert report_after == report_before
    pd.testing.assert_frame_equal(frame_after, frame_before)


def test_tuning_only_report_differs_only_in_platt_coefficients_of_later_seasons() -> None:
    """Isolates what the strict test above reports: apart from ``chosen.platt.coefficients`` of
    the seasons after tuning (fitted on the validation and test outcomes of the seasons before
    them), the tuning-only report and its frame are identical. Not a weaker replacement of that
    test: it pins down exactly how far the leak goes (the choice and every tuning number are
    unaffected)."""
    report_before, frame_before = _base_tuning_only()
    report_after, frame_after = _tuning_only_after_later_outcomes()
    before, after = json.loads(report_before), json.loads(report_after)
    for report in (before, after):
        for season in [s for s in report["chosen"]["platt"]["coefficients"] if int(s) > 2019]:
            del report["chosen"]["platt"]["coefficients"][season]
    assert after == before
    pd.testing.assert_frame_equal(frame_after, frame_before)


def test_tuning_only_report_does_see_tuning_outcomes() -> None:
    """Control for the test above: rewriting tuning outcomes does change the report."""
    base = _base()
    games = base.games
    tuning = _ids(games, games["season"].isin(SPEC.tuning))
    report_after, _ = _tuning_only_report(_rescore(base, tuning))
    assert report_after != _base_tuning_only()[0]
