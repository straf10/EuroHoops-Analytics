"""Leakage tests (week 9-12 H1, done-when #3): editing any stint row of game g or later, or any
box row of game g or later, changes no rating, weight or prediction used for game g. A
deliberately planted leak (a copy of ``fit_walk_forward`` with the wrong cutoff boundary,
kept local to this file -- ``rapm.py`` itself is never touched) is shown to make the same check
fail, proving the check has power to catch a real regression.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from eurohoops.models.minutes import expected_possessions, projected_shares
from eurohoops.models.rapm import (
    BASE_OFFSET,
    DecayedRidgeSparse,
    DesignRows,
    ModelColumns,
    RatingLookup,
    WalkForward,
    _round_groups,  # local, deliberately-broken use only (module docstring)
    build_design_rows,
    build_spell_index,
    fit_walk_forward,
    plain_player_columns,
    rating_lookup,
)
from tests.m3_synthetic import SyntheticM3, make_synthetic_m3

FIT_KWARGS = {"half_life_days": 365.0, "ridge_o": 300.0, "ridge_d": 300.0}


def _fit(data: SyntheticM3) -> tuple[WalkForward, int]:
    """``data.games`` is already chronologically sorted (``parse.games.conform``)."""
    spell_index = build_spell_index(data.stints)
    columns = plain_player_columns(spell_index)
    rows = build_design_rows(data.stints, data.checks, data.games, spell_index)
    wf = fit_walk_forward(data.games, rows, columns, **FIT_KWARGS)
    # A game with real prior history: not the first round of the first season.
    g = next(i for i in range(len(data.games)) if wf.lookups[i] is not None and i > 20)
    return wf, g


def _leaky_walk_forward(
    games: pd.DataFrame, rows: DesignRows, columns: ModelColumns, **kwargs: float
) -> WalkForward:
    """A byte-for-byte copy of ``rapm.fit_walk_forward``, except the round-cutoff comparison
    uses ``side="right"``: a stint that tips off in the same instant as a round's cutoff then
    leaks into that round's own fit. Used only to prove the leakage check below has power; the
    real ``fit_walk_forward`` (``side="left"``) is never modified."""
    n_base = rows.n_base if len(rows.cols) else BASE_OFFSET
    model = DecayedRidgeSparse(n_base, kwargs["half_life_days"], columns)
    n_games = len(games)
    home_coef = np.full(n_games, np.nan)
    lookups: list[RatingLookup | None] = [None] * n_games
    theta = np.zeros(columns.n_model)
    added = 0
    for time, _season, game_idx in _round_groups(games):
        model.advance(time)
        stop = int(np.searchsorted(rows.time, time, side="right"))  # the planted leak
        if stop > added:
            span = slice(added, stop)
            model.add(
                rows.cols[span],
                rows.vals[span],
                target=rows.target[span],
                weight=rows.weight[span],
                time=rows.time[span],
            )
            added = stop
        if not added:
            continue
        theta = model.solve(kwargs["ridge_o"], kwargs["ridge_d"], x0=theta)
        coef = 2.0 * theta[1]
        lookup = rating_lookup(theta, columns)
        home_coef[game_idx] = coef
        for g in game_idx:
            lookups[g] = lookup
    return WalkForward(home_coef, lookups, columns, model)


def test_editing_a_later_stint_row_does_not_change_game_g() -> None:
    data = make_synthetic_m3(
        [2020, 2021], teams=8, games_per_season=24, stints_per_game=12, roster=8, seed=9
    )
    wf_before, g = _fit(data)
    game_id = str(data.games.loc[g, "game_id"])

    # Edit a stint row of game g itself, and of the very last game (both "game g or later").
    edited = data.stints.copy()
    own_row = edited.index[edited["game_id"] == game_id][0]
    later_game_id = str(data.games.iloc[-1]["game_id"])
    later_row = edited.index[edited["game_id"] == later_game_id][0]
    edited.loc[own_row, "home_points"] = int(edited.loc[own_row, "home_points"]) + 7
    edited.loc[own_row, "home_poss"] = int(edited.loc[own_row, "home_poss"]) + 2
    edited.loc[later_row, "away_points"] = int(edited.loc[later_row, "away_points"]) + 11

    edited_data = SyntheticM3(
        data.games,
        edited,
        data.checks,
        data.team_games,
        data.player_games,
        data.true_o,
        data.true_d,
    )
    wf_after, g_after = _fit(edited_data)
    assert g_after == g  # same game (the fixture and the edit did not change round structure)

    assert wf_before.home_coef[g] == wf_after.home_coef[g] or (
        np.isnan(wf_before.home_coef[g]) and np.isnan(wf_after.home_coef[g])
    )
    before, after = wf_before.lookups[g], wf_after.lookups[g]
    assert (before is None) == (after is None)
    if before is not None:
        assert before.o == after.o
        assert before.d == after.d


def test_planted_cutoff_leak_is_caught() -> None:
    """The same before/after-edit comparison, run through the deliberately broken
    ``_leaky_walk_forward`` (``side="right"``): editing game g's own stint row now DOES change
    game g's own rating, because g's own same-instant stints leak into g's own fit. This shows
    the check above has the power to catch a real off-by-one in the cutoff boundary."""
    data = make_synthetic_m3(
        [2020, 2021], teams=8, games_per_season=24, stints_per_game=12, roster=8, seed=9
    )
    spell_index = build_spell_index(data.stints)
    columns = plain_player_columns(spell_index)
    rows = build_design_rows(data.stints, data.checks, data.games, spell_index)
    wf_before = _leaky_walk_forward(data.games, rows, columns, **FIT_KWARGS)
    g = next(i for i in range(len(data.games)) if wf_before.lookups[i] is not None and i > 20)
    game_id = str(data.games.loc[g, "game_id"])

    edited = data.stints.copy()
    own_row = edited.index[edited["game_id"] == game_id][0]
    edited.loc[own_row, "home_points"] = int(edited.loc[own_row, "home_points"]) + 25
    edited.loc[own_row, "home_poss"] = int(edited.loc[own_row, "home_poss"]) + 3

    edited_index = build_spell_index(edited)
    edited_columns = plain_player_columns(edited_index)
    edited_rows = build_design_rows(edited, data.checks, data.games, edited_index)
    wf_after = _leaky_walk_forward(data.games, edited_rows, edited_columns, **FIT_KWARGS)

    before, after = wf_before.lookups[g], wf_after.lookups[g]
    assert before is not None and after is not None
    assert before.o != after.o or before.d != after.d  # the planted leak changed g's own rating


def test_editing_a_later_box_row_does_not_change_shares_or_possessions_for_g() -> None:
    """Editing a ``player_games``/``team_games`` row of game g's own game or a later game must
    not change ``projected_shares``/``expected_possessions`` computed for game g (H-c: only
    games strictly before g's round cutoff feed them; verified end to end here, on top of
    ``minutes.py``'s own tests, since H1 consumes both functions through this pipeline)."""
    data = make_synthetic_m3(
        [2020, 2021], teams=8, games_per_season=24, stints_per_game=6, roster=8, seed=13
    )
    games = data.games.sort_values("tipoff_utc").reset_index(drop=True)
    g = 30  # well past the first round, so g has real projection history
    game_id = str(games.loc[g, "game_id"])

    shares_before = projected_shares(games, data.player_games)
    poss_before = expected_possessions(games, data.team_games)
    before_rows = shares_before[shares_before["game_id"] == game_id].sort_values(
        ["side", "player_id"]
    )
    before_poss = float(poss_before.loc[game_id])

    edited_players = data.player_games.copy()
    own = edited_players.index[edited_players["game_id"] == game_id][0]
    edited_players.loc[own, "sec"] = int(edited_players.loc[own, "sec"]) + 37
    later_game_id = str(games.iloc[-1]["game_id"])
    later = edited_players.index[edited_players["game_id"] == later_game_id][0]
    edited_players.loc[later, "sec"] = int(edited_players.loc[later, "sec"]) + 41

    edited_teams = data.team_games.copy()
    own_team_rows = edited_teams.index[edited_teams["game_id"] == game_id]
    edited_teams.loc[own_team_rows, "poss_raw"] = edited_teams.loc[own_team_rows, "poss_raw"] + 5.0

    shares_after = projected_shares(games, edited_players)
    poss_after = expected_possessions(games, edited_teams)
    after_rows = shares_after[shares_after["game_id"] == game_id].sort_values(["side", "player_id"])
    after_poss = float(poss_after.loc[game_id])

    pd.testing.assert_frame_equal(
        before_rows.reset_index(drop=True), after_rows.reset_index(drop=True)
    )
    assert before_poss == after_poss
