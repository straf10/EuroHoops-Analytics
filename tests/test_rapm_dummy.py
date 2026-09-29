"""``rapm_dummy`` unit tests (week 9-12 H6, subagent E): a hand-computed pooled example, the
threshold-0-reproduces-plain-rapm property, rating-lookup semantics, leakage, the minutes-rows
sort fix, and the tuner's threshold selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from eurohoops.config import M3Backtest, M3Grid
from eurohoops.eval.m3_backtest import (
    BaselineResult,
    Data,
    TunedRapm,
    build_rapm_inputs,
    prepare_data,
    rapm_margins,
    run_m3_backtest,
    tune_rapm,
)
from eurohoops.models.elo import FloatArray
from eurohoops.models.rapm import (
    BASE_OFFSET,
    HOME_COL,
    INTERCEPT_COL,
    DecayedMinutes,
    DesignRows,
    MinutesRows,
    ModelColumns,
    Spell,
    SpellIndex,
    WalkForward,
    _round_groups,  # local, deliberately-broken use only (mirrors test_m3_leakage.py)
    build_design_rows,
    build_minutes_rows,
    build_spell_index,
    fit_decayed_minutes,
    fit_walk_forward,
    plain_player_columns,
)
from eurohoops.models.rapm_dummy import (
    BaseAccumulator,
    DummyColumns,
    DummyCutoffSolve,
    DummyRatingLookup,
    build_dummy_columns,
    dummy_rating_lookup,
    fit_rapm_dummy,
    fit_walk_forward_dummy,
    solve_dummy_cutoff,
    threshold_map,
    tune_rapm_dummy,
)
from tests.m3_synthetic import SyntheticM3, make_synthetic_m3


def _manual_spell_index(spells: list[Spell]) -> SpellIndex:
    positions = {s: i for i, s in enumerate(spells)}
    lookup = pd.DataFrame(
        {
            "player_id": [s.player_id for s in spells],
            "team": [s.team for s in spells],
            "season": [s.season for s in spells],
            "pos": np.arange(len(spells)),
        }
    )
    return SpellIndex(tuple(spells), positions, lookup)


# --- Hand-computed pooling example -----------------------------------------------------------


def _build_pooling_example_rows(
    spell_index: SpellIndex,
    season: int,
    lineups: list[tuple[list[str], list[str]]],
    stint_scores: list[tuple[int, int, int, int]],
    duration: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Design rows and minutes rows (module-level helper, kept out of the hand test itself so
    that test stays under the statement-count lint limit) for a home team "AAA" hosting an away
    team "BBB", one 100 x home_poss/home_pts/away_poss/away_pts row per stint."""
    rows_cols: list[np.ndarray] = []
    rows_vals: list[np.ndarray] = []
    target: list[float] = []
    weight: list[float] = []
    minutes_position: list[int] = []
    minutes_seconds: list[float] = []
    for (home_five, away_five), (home_poss, home_pts, away_poss, away_pts) in zip(
        lineups, stint_scores, strict=True
    ):
        home_pos = np.array([spell_index.position(p, "AAA", season) for p in home_five])
        away_pos = np.array([spell_index.position(p, "BBB", season) for p in away_five])

        cols = np.concatenate(
            [[INTERCEPT_COL, HOME_COL], spell_index.o_col(home_pos), spell_index.d_col(away_pos)]
        )
        vals = np.concatenate([[1.0, 1.0], np.ones(5), -np.ones(5)])
        rows_cols.append(cols)
        rows_vals.append(vals)
        target.append(100.0 * home_pts / home_poss)
        weight.append(float(home_poss))

        cols = np.concatenate(
            [[INTERCEPT_COL, HOME_COL], spell_index.o_col(away_pos), spell_index.d_col(home_pos)]
        )
        vals = np.concatenate([[1.0, -1.0], np.ones(5), -np.ones(5)])
        rows_cols.append(cols)
        rows_vals.append(vals)
        target.append(100.0 * away_pts / away_poss)
        weight.append(float(away_poss))

        minutes_position.extend([*home_pos.tolist(), *away_pos.tolist()])
        minutes_seconds.extend([duration] * 10)

    return (
        np.stack(rows_cols),
        np.stack(rows_vals),
        np.array(target),
        np.array(weight),
        np.array(minutes_position, dtype=np.int64),
        np.array(minutes_seconds),
    )


def test_hand_computed_pooled_replacement_matches_a_manual_dense_ridge_solve() -> None:
    """Six AAA players over 3 stints: A1-A4 and all of B1-B5 are on court every stint, A5 plays
    only the first two and A6 only the third (A6 replaces A5 in that lineup). A6's total
    decayed minutes (one stint) sit below a threshold every other player clears (two or three
    stints), so his own O/D design contributions must be pooled into (AAA, 2020)'s replacement
    columns while everyone else keeps his own column -- checked against an independent, plain
    dense weighted-ridge solve of that exact pooled design (H6's hand example)."""
    season = 2020
    a_players = [f"A{i}" for i in range(1, 7)]  # A1..A6
    b_players = [f"B{i}" for i in range(1, 6)]
    spells = [Spell(p, "AAA", season) for p in a_players] + [
        Spell(p, "BBB", season) for p in b_players
    ]
    spell_index = _manual_spell_index(spells)
    dummy = build_dummy_columns(spell_index)
    assert dummy.n_players == 11
    assert dummy.n_team_seasons == 2  # (AAA, 2020) and (BBB, 2020)
    assert dummy.columns.n_model == BASE_OFFSET + 2 * (11 + 2)

    lineups = [
        (["A1", "A2", "A3", "A4", "A5"], b_players),
        (["A1", "A2", "A3", "A4", "A5"], b_players),
        (["A1", "A2", "A3", "A4", "A6"], b_players),  # A6 replaces A5
    ]
    # (home_poss, home_pts, away_poss, away_pts) per stint
    stint_scores = [(4, 8, 3, 3), (5, 5, 4, 8), (2, 0, 2, 2)]
    duration = 100.0  # seconds per stint, same instant for every row (see below): no decay
    cols_arr, vals_arr, target_arr, weight_arr, minutes_pos_arr, minutes_sec_arr = (
        _build_pooling_example_rows(spell_index, season, lineups, stint_scores, duration)
    )

    ridge_o, ridge_d = 137.0, 251.0
    threshold_minutes = 2.5  # A6: 100s = 1.667 min (below); everyone else: >= 200s = 3.33 min

    # A6's spell decayed minutes must indeed fall on the pooled side of the threshold, and
    # everyone else's on the individual side -- otherwise this example tests nothing.
    decayed_seconds = np.zeros(spell_index.n_spells)
    np.add.at(decayed_seconds, minutes_pos_arr, minutes_sec_arr)
    decayed_minutes = decayed_seconds / 60.0
    a6_pos = spell_index.position("A6", "AAA", season)
    a5_pos = spell_index.position("A5", "AAA", season)
    assert decayed_minutes[a6_pos] < threshold_minutes <= decayed_minutes[a5_pos]

    col_map = threshold_map(dummy, decayed_minutes, threshold_minutes)
    pooled_columns = ModelColumns(
        dummy.columns.name,
        dummy.columns.n_model,
        col_map,
        dummy.columns.o_labels,
        dummy.columns.d_labels,
        dummy.columns.o_start,
        dummy.columns.d_start,
    )
    # A6's own O/D columns must never appear in the pooled mapping; his contributions moved to
    # the (AAA, 2020) replacement columns instead.
    a6_o_col = BASE_OFFSET + a6_pos
    a6_d_col = BASE_OFFSET + spell_index.n_spells + a6_pos
    a6_own_o_model = dummy.columns.o_start + dummy.player_pos["A6"]
    a6_own_d_model = dummy.columns.d_start + dummy.player_pos["A6"]
    assert col_map[a6_o_col] != a6_own_o_model
    assert col_map[a6_d_col] != a6_own_d_model
    repl_idx = dummy.team_season_pos[("AAA", season)]
    assert col_map[a6_o_col] == dummy.columns.o_start + dummy.n_players + repl_idx
    assert col_map[a6_d_col] == dummy.columns.d_start + dummy.n_players + repl_idx

    # -- Independent, plain dense weighted ridge solve of the pooled design. --
    n_base = spell_index.n_base
    x_base = np.zeros((6, n_base))
    for r in range(6):
        x_base[r, cols_arr[r]] = vals_arr[r]
    x_model = x_base @ pooled_columns.M.toarray()
    w = np.diag(weight_arr)
    gram = x_model.T @ w @ x_model
    rhs = x_model.T @ w @ target_arr
    penalty = dummy.columns.penalty(ridge_o, ridge_d)
    manual_theta = np.linalg.solve(gram + np.diag(penalty), rhs)

    base = BaseAccumulator(n_base, half_life_days=365.0)
    base.advance(1000.0)
    base.add(cols_arr, vals_arr, target=target_arr, weight=weight_arr, time=np.full(6, 1000.0))
    solve = solve_dummy_cutoff(
        base.gram,
        base.rhs,
        dummy,
        decayed_minutes,
        threshold_minutes,
        ridge_o=ridge_o,
        ridge_d=ridge_d,
    )

    np.testing.assert_allclose(solve.theta, manual_theta, atol=1e-9, rtol=0.0)


# --- Threshold 0 reproduces plain rapm ---------------------------------------------------------


def test_threshold_zero_reproduces_plain_rapm() -> None:
    data = make_synthetic_m3(
        [2020, 2021], teams=6, games_per_season=16, stints_per_game=8, roster=8, seed=41
    )
    spell_index = build_spell_index(data.stints)
    rows = build_design_rows(data.stints, data.checks, data.games, spell_index)
    minutes_rows = build_minutes_rows(data.stints, data.checks, data.games, spell_index)
    dummy = build_dummy_columns(spell_index)
    plain_columns = plain_player_columns(spell_index)
    half_life_days, ridge_o, ridge_d = 365.0, 400.0, 400.0

    plain = fit_walk_forward(
        data.games,
        rows,
        plain_columns,
        half_life_days=half_life_days,
        ridge_o=ridge_o,
        ridge_d=ridge_d,
    )
    dummy_wf = fit_walk_forward_dummy(
        data.games,
        rows,
        minutes_rows,
        spell_index,
        dummy,
        half_life_days=half_life_days,
        ridge_o=ridge_o,
        ridge_d=ridge_d,
        dummy_minutes=0.0,
    )

    for i, (plain_lookup, dummy_lookup) in enumerate(
        zip(plain.lookups, dummy_wf.lookups, strict=True)
    ):
        if plain_lookup is None:
            assert dummy_lookup is None
            continue
        assert isinstance(dummy_lookup, DummyRatingLookup)
        assert dummy_lookup.repl_o == {}  # threshold 0: nothing is ever pooled
        assert dummy_lookup.repl_d == {}
        players = set(plain_lookup.o) | set(dummy_lookup.o)
        for p in players:
            assert plain_lookup.o.get(p, 0.0) == pytest.approx(
                dummy_lookup.o.get(p, 0.0), abs=1e-6
            ), (i, p)
            assert plain_lookup.d.get(p, 0.0) == pytest.approx(
                dummy_lookup.d.get(p, 0.0), abs=1e-6
            ), (i, p)


# --- Rating lookup semantics: above / below / unseen / replacement not yet fitted --------------


def test_lookup_at_or_above_threshold_reads_the_players_own_column() -> None:
    dummy_lookup = DummyRatingLookup(
        o={"P1": 5.0}, d={"P1": -2.0}, repl_o={("AAA", 2020): 9.0}, repl_d={("AAA", 2020): -9.0}
    )
    assert dummy_lookup.rating("P1", "AAA", 2020) == (5.0, -2.0)


def test_lookup_below_threshold_reads_the_team_season_replacement() -> None:
    dummy_lookup = DummyRatingLookup(
        o={}, d={}, repl_o={("AAA", 2020): 3.0}, repl_d={("AAA", 2020): -1.5}
    )
    assert dummy_lookup.rating("P2", "AAA", 2020) == (3.0, -1.5)


def test_lookup_unseen_player_reads_the_replacement_like_a_below_threshold_player() -> None:
    dummy_lookup = DummyRatingLookup(
        o={"P1": 5.0}, d={"P1": -2.0}, repl_o={("BBB", 2021): 1.0}, repl_d={("BBB", 2021): -1.0}
    )
    assert dummy_lookup.rating("nobody", "BBB", 2021) == (1.0, -1.0)


def test_lookup_replacement_not_yet_fitted_returns_zero() -> None:
    dummy_lookup = DummyRatingLookup(o={}, d={}, repl_o={}, repl_d={})
    assert dummy_lookup.rating("nobody", "CCC", 2019) == (0.0, 0.0)
    assert dummy_lookup.rating("below_threshold_player", "CCC", 2019) == (0.0, 0.0)


def test_dummy_rating_lookup_builder_matches_the_semantics_end_to_end() -> None:
    """``dummy_rating_lookup`` itself (not a hand-built ``DummyRatingLookup``): a player above
    threshold reads his own column, a player below reads his (team, season) replacement only if
    it was touched this cutoff, and an untouched replacement is simply absent (0 via ``.get``)."""
    season = 2022
    spells = [Spell(p, "AAA", season) for p in ("A1", "A2")] + [Spell("B1", "BBB", season)]
    spell_index = _manual_spell_index(spells)
    dummy = build_dummy_columns(spell_index)
    theta = np.zeros(dummy.columns.n_model)
    theta[dummy.columns.o_start + dummy.player_pos["A1"]] = 7.0  # A1: above threshold, own O
    repl_idx = dummy.team_season_pos[("AAA", season)]
    a1_o_col = dummy.columns.o_start + dummy.player_pos["A1"]
    repl_o_col = dummy.columns.o_start + dummy.n_players + repl_idx
    theta[repl_o_col] = -4.0  # (AAA, 2022) O replacement
    used = np.array([a1_o_col, repl_o_col], dtype=np.int64)
    solve = DummyCutoffSolve(theta, used)
    decayed_minutes = np.array([10.0, 0.0, 5.0])  # A1 above, A2 (pos 1) below, B1 above
    threshold_minutes = 5.0
    lookup = dummy_rating_lookup(solve, dummy, decayed_minutes, threshold_minutes)

    assert lookup.rating("A1", "AAA", season) == (7.0, 0.0)
    assert lookup.rating("A2", "AAA", season) == (-4.0, 0.0)  # below threshold: the replacement
    # (BBB, 2022) replacement was never touched (not in `used`): 0, not the AAA one.
    assert lookup.rating("B1", "BBB", season) == (0.0, 0.0)


# --- The minutes-rows sort fix (D6/H6 known issue) --------------------------------------------


def test_build_minutes_rows_is_sorted_by_time_regardless_of_stint_order() -> None:
    """``fit_decayed_minutes`` (and ``fit_walk_forward_dummy``) use ``searchsorted`` on
    ``rows.time``, which needs ascending order; ``build_minutes_rows`` used to return rows in
    the raw ``stints`` table's own order, not tip-off order -- this is the exact bug D6/H6
    flags. Shuffling the input stints must not change the decayed-minutes result."""
    data = make_synthetic_m3(
        [2019, 2020, 2021], teams=6, games_per_season=15, stints_per_game=8, roster=8, seed=31
    )
    spell_index = build_spell_index(data.stints)
    ordered_rows = build_minutes_rows(data.stints, data.checks, data.games, spell_index)
    assert np.all(np.diff(ordered_rows.time) >= 0.0)

    shuffled_stints = data.stints.sample(frac=1.0, random_state=7).reset_index(drop=True)
    shuffled_rows = build_minutes_rows(shuffled_stints, data.checks, data.games, spell_index)
    assert np.all(np.diff(shuffled_rows.time) >= 0.0)

    ordered_final = fit_decayed_minutes(data.games, ordered_rows, spell_index, half_life_days=365.0)
    shuffled_final = fit_decayed_minutes(
        data.games, shuffled_rows, spell_index, half_life_days=365.0
    )
    np.testing.assert_allclose(ordered_final, shuffled_final, atol=1e-9)


# --- Leakage: editing game g or later must not change game g's mapping, rating or prediction ---


FIT_KWARGS = {"half_life_days": 365.0, "ridge_o": 300.0, "ridge_d": 300.0, "dummy_minutes": 50.0}


def _fit_dummy(data: SyntheticM3) -> tuple[list[DummyRatingLookup | None], int]:
    spell_index = build_spell_index(data.stints)
    dummy = build_dummy_columns(spell_index)
    rows = build_design_rows(data.stints, data.checks, data.games, spell_index)
    minutes_rows = build_minutes_rows(data.stints, data.checks, data.games, spell_index)
    wf = fit_walk_forward_dummy(
        data.games,
        rows,
        minutes_rows,
        spell_index,
        dummy,
        half_life_days=FIT_KWARGS["half_life_days"],
        ridge_o=FIT_KWARGS["ridge_o"],
        ridge_d=FIT_KWARGS["ridge_d"],
        dummy_minutes=FIT_KWARGS["dummy_minutes"],
    )
    lookups = wf.lookups
    g = next(i for i in range(len(data.games)) if lookups[i] is not None and i > 20)
    return lookups, g  # type: ignore[return-value]


def test_editing_a_later_stint_row_does_not_change_game_g() -> None:
    data = make_synthetic_m3(
        [2020, 2021], teams=8, games_per_season=24, stints_per_game=12, roster=8, seed=9
    )
    lookups_before, g = _fit_dummy(data)
    game_id = str(data.games.loc[g, "game_id"])

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
    lookups_after, g_after = _fit_dummy(edited_data)
    assert g_after == g

    before, after = lookups_before[g], lookups_after[g]
    assert (before is None) == (after is None)
    if before is not None:
        assert isinstance(after, DummyRatingLookup)
        assert before.o == after.o
        assert before.d == after.d
        assert before.repl_o == after.repl_o
        assert before.repl_d == after.repl_d


def test_editing_a_later_minutes_row_does_not_change_game_g() -> None:
    """Same check, but editing a game's *box on-court time* (``player_games.sec``, which feeds
    ``inputs.minutes`` through ``build_minutes_rows`` via the stints table's own start/end, so
    here we edit the stint window itself, which is what actually drives the threshold) of game
    g's own game or a later game leaves g's mapping/rating untouched -- the minutes side of the
    same leakage guarantee the stint-row test above checks for the regression targets."""
    data = make_synthetic_m3(
        [2020, 2021], teams=8, games_per_season=24, stints_per_game=12, roster=8, seed=17
    )
    lookups_before, g = _fit_dummy(data)
    game_id = str(data.games.loc[g, "game_id"])

    edited = data.stints.copy()
    own_rows = edited.index[edited["game_id"] == game_id]
    later_game_id = str(data.games.iloc[-1]["game_id"])
    later_rows = edited.index[edited["game_id"] == later_game_id]
    # Widen the stint windows (more on-court seconds credited) without touching points/possessions
    # (keeps the design rows -- and so the leakage test above's scope -- untouched; only minutes
    # change).
    edited.loc[own_rows, "end_s"] = edited.loc[own_rows, "end_s"] + 5
    edited.loc[later_rows, "end_s"] = edited.loc[later_rows, "end_s"] + 5

    edited_data = SyntheticM3(
        data.games,
        edited,
        data.checks,
        data.team_games,
        data.player_games,
        data.true_o,
        data.true_d,
    )
    lookups_after, g_after = _fit_dummy(edited_data)
    assert g_after == g

    before, after = lookups_before[g], lookups_after[g]
    assert (before is None) == (after is None)
    if before is not None:
        assert isinstance(after, DummyRatingLookup)
        assert before.o == after.o
        assert before.d == after.d
        assert before.repl_o == after.repl_o
        assert before.repl_d == after.repl_d


def _leaky_fit_walk_forward_dummy(
    games: pd.DataFrame,
    rows: DesignRows,
    minutes_rows: MinutesRows,
    spell_index: SpellIndex,
    dummy: DummyColumns,
    **kwargs: float,
) -> list[DummyRatingLookup | None]:
    """A byte-for-byte copy of ``rapm_dummy.fit_walk_forward_dummy``, except the *minutes*
    cutoff comparison uses ``side="right"``: a minutes row that ticks in the same instant as a
    round's cutoff then leaks into that round's own threshold decision. Used only to prove the
    leakage check above has power; the real ``fit_walk_forward_dummy`` (``side="left"``) is
    never modified."""
    n_base = spell_index.n_base
    base = BaseAccumulator(n_base, kwargs["half_life_days"])
    minutes = DecayedMinutes(spell_index.n_spells, kwargs["half_life_days"])
    n_games = len(games)
    lookups: list[DummyRatingLookup | None] = [None] * n_games
    theta = np.zeros(dummy.columns.n_model)
    added_rows = 0
    added_minutes = 0
    for time, _season, game_idx in _round_groups(games):
        base.advance(time)
        minutes.advance(time)
        stop = int(np.searchsorted(rows.time, time, side="left"))
        if stop > added_rows:
            span = slice(added_rows, stop)
            base.add(
                rows.cols[span],
                rows.vals[span],
                target=rows.target[span],
                weight=rows.weight[span],
                time=rows.time[span],
            )
            added_rows = stop
        # The planted leak: side="right" lets same-instant minutes rows into THIS round's own
        # threshold decision.
        m_stop = int(np.searchsorted(minutes_rows.time, time, side="right"))
        if m_stop > added_minutes:
            m_span = slice(added_minutes, m_stop)
            minutes.add(
                minutes_rows.position[m_span],
                minutes_rows.seconds[m_span],
                minutes_rows.time[m_span],
            )
            added_minutes = m_stop
        if not added_rows:
            continue
        decayed_minutes = minutes.seconds / 60.0
        solve = solve_dummy_cutoff(
            base.gram,
            base.rhs,
            dummy,
            decayed_minutes,
            kwargs["dummy_minutes"],
            ridge_o=kwargs["ridge_o"],
            ridge_d=kwargs["ridge_d"],
            x0=theta,
        )
        theta = solve.theta
        _ = 2.0 * theta[HOME_COL]
        lookup = dummy_rating_lookup(solve, dummy, decayed_minutes, kwargs["dummy_minutes"])
        for g in game_idx:
            lookups[g] = lookup
    return lookups


def test_planted_minutes_side_right_leak_is_caught() -> None:
    """The same before/after-edit comparison, run through the deliberately broken minutes
    cutoff (``side="right"``): editing game g's own stint window now DOES change game g's own
    threshold mapping and rating, because g's own same-instant minutes leak into g's own
    threshold decision. Shows the leakage check has the power to catch a real off-by-one."""
    data = make_synthetic_m3(
        [2020, 2021], teams=8, games_per_season=24, stints_per_game=12, roster=8, seed=9
    )
    spell_index = build_spell_index(data.stints)
    dummy = build_dummy_columns(spell_index)
    rows = build_design_rows(data.stints, data.checks, data.games, spell_index)
    minutes_rows = build_minutes_rows(data.stints, data.checks, data.games, spell_index)
    lookups_before = _leaky_fit_walk_forward_dummy(
        data.games, rows, minutes_rows, spell_index, dummy, **FIT_KWARGS
    )
    g = next(i for i in range(len(data.games)) if lookups_before[i] is not None and i > 20)
    game_id = str(data.games.loc[g, "game_id"])

    edited = data.stints.copy()
    own_rows = edited.index[edited["game_id"] == game_id]
    edited.loc[own_rows, "end_s"] = edited.loc[own_rows, "end_s"] + 45

    edited_spell_index = build_spell_index(edited)
    edited_dummy = build_dummy_columns(edited_spell_index)
    edited_rows = build_design_rows(edited, data.checks, data.games, edited_spell_index)
    edited_minutes_rows = build_minutes_rows(edited, data.checks, data.games, edited_spell_index)
    lookups_after = _leaky_fit_walk_forward_dummy(
        data.games, edited_rows, edited_minutes_rows, edited_spell_index, edited_dummy, **FIT_KWARGS
    )

    before, after = lookups_before[g], lookups_after[g]
    assert before is not None and after is not None
    changed = before.o != after.o or before.d != after.d
    changed = changed or before.repl_o != after.repl_o or before.repl_d != after.repl_d
    assert changed  # the planted leak changed g's own mapping/rating


# --- Two runs identical -------------------------------------------------------------------


def test_two_runs_are_identical() -> None:
    data = make_synthetic_m3(
        [2019, 2020, 2021], teams=6, games_per_season=14, stints_per_game=8, roster=8, seed=23
    )
    spell_index = build_spell_index(data.stints)
    dummy = build_dummy_columns(spell_index)
    rows = build_design_rows(data.stints, data.checks, data.games, spell_index)
    minutes_rows = build_minutes_rows(data.stints, data.checks, data.games, spell_index)

    def _run() -> WalkForward:
        return fit_walk_forward_dummy(
            data.games,
            rows,
            minutes_rows,
            spell_index,
            dummy,
            half_life_days=365.0,
            ridge_o=400.0,
            ridge_d=400.0,
            dummy_minutes=60.0,
        )

    first = _run()
    second = _run()

    np.testing.assert_array_equal(first.home_coef, second.home_coef)
    for a, b in zip(first.lookups, second.lookups, strict=True):
        if a is None:
            assert b is None
            continue
        assert isinstance(a, DummyRatingLookup)
        assert isinstance(b, DummyRatingLookup)
        assert a.o == b.o
        assert a.d == b.d
        assert a.repl_o == b.repl_o
        assert a.repl_d == b.repl_d


# --- The tuner: threshold selection -------------------------------------------------------


def test_tune_rapm_dummy_selects_the_lowest_rmse_threshold() -> None:
    data = make_synthetic_m3(
        list(range(2015, 2021)),
        teams=8,
        games_per_season=20,
        stints_per_game=10,
        roster=8,
        seed=55,
    )
    spec = M3Backtest(
        report=None,  # type: ignore[arg-type]
        warmup=(2015, 2016),
        tuning=(2017, 2018),
        validation=(2019,),
        test=(2020,),
        grid=M3Grid(
            half_life_days=(365.0,), ridge=(400.0,), dummy_minutes=(0.0, 50.0, 100.0, 200.0)
        ),
    )
    fake_games = data.games
    fake_team_games = data.team_games
    fake_player_games = data.player_games
    fake_data: Data = prepare_data(
        fake_games, fake_team_games, fake_player_games, data.stints, data.checks, spec=spec
    )
    inputs = build_rapm_inputs(data.stints, data.checks, fake_data.games)

    tuned_rapm = tune_rapm(fake_data, spec, inputs, {})
    tuned_dummy = tune_rapm_dummy(fake_data, spec, inputs, {"rapm": tuned_rapm})

    assert tuned_dummy.half_life_days == tuned_rapm.half_life_days
    assert tuned_dummy.ridge_o == tuned_rapm.ridge_o
    assert tuned_dummy.ridge_d == tuned_rapm.ridge_d
    threshold = tuned_dummy.extra["dummy_minutes"]
    assert threshold in (50.0, 100.0, 200.0)  # 0 is never retried (already `rapm`'s own point)

    # Recompute every nonzero threshold's tuning RMSE independently and check the tuner picked
    # the minimum, exactly the way `tune_rapm` computes its own tuning RMSE.
    dummy_columns = build_dummy_columns(inputs.spell_index)
    last_tuning_season = max(spec.tuning)
    sub_mask = fake_data.games["season"].to_numpy() <= last_tuning_season
    games_tuning = fake_data.games[sub_mask].reset_index(drop=True)
    tuning_mask = fake_data.split["tuning"][sub_mask]
    actual = fake_data.margin[sub_mask]
    tuning_game_ids = set(games_tuning["game_id"].astype(str))
    shares_tuning = fake_data.shares[fake_data.shares["game_id"].astype(str).isin(tuning_game_ids)]

    def rmse_for(threshold_minutes: float) -> float:
        wf = fit_walk_forward_dummy(
            games_tuning,
            inputs.rows,
            inputs.minutes,
            inputs.spell_index,
            dummy_columns,
            half_life_days=tuned_rapm.half_life_days,
            ridge_o=tuned_rapm.ridge_o,
            ridge_d=tuned_rapm.ridge_d,
            dummy_minutes=threshold_minutes,
        )
        pred = rapm_margins(games_tuning, wf, shares_tuning, fake_data.possessions)
        sel = tuning_mask & np.isfinite(pred)
        return float(np.sqrt(np.mean((pred[sel] - actual[sel]) ** 2)))

    rmses = {t: rmse_for(t) for t in (50.0, 100.0, 200.0)}
    best_expected = min(rmses, key=lambda t: rmses[t])
    assert threshold == best_expected
    assert tuned_dummy.tuning_rmse == pytest.approx(rmses[best_expected])


def test_tune_rapm_dummy_falls_back_to_rapm_when_the_grid_has_no_nonzero_threshold() -> None:
    tuned_rapm = TunedRapm(365.0, 500.0, 500.0, 12.34, {"info": "rapm's own grid report"})
    spec = M3Backtest(
        report=None,  # type: ignore[arg-type]
        warmup=(2015,),
        tuning=(2016,),
        validation=(2017,),
        test=(2018,),
        grid=M3Grid(dummy_minutes=(0.0,)),
    )
    data = make_synthetic_m3(
        [2015, 2016, 2017, 2018], teams=6, games_per_season=10, stints_per_game=6, roster=6
    )
    prepared = prepare_data(
        data.games, data.team_games, data.player_games, data.stints, data.checks, spec=spec
    )
    inputs = build_rapm_inputs(data.stints, data.checks, prepared.games)
    tuned_dummy = tune_rapm_dummy(prepared, spec, inputs, {"rapm": tuned_rapm})
    assert tuned_dummy.extra == {"dummy_minutes": 0.0}
    assert tuned_dummy.tuning_rmse == tuned_rapm.tuning_rmse
    assert tuned_dummy.half_life_days == tuned_rapm.half_life_days


# --- fit_rapm_dummy / end-to-end registration ---------------------------------------------


def test_fit_rapm_dummy_uses_the_tuned_threshold() -> None:
    data = make_synthetic_m3(
        [2017, 2018], teams=6, games_per_season=12, stints_per_game=8, roster=8, seed=61
    )
    inputs = build_rapm_inputs(data.stints, data.checks, data.games)
    tuned = TunedRapm(365.0, 300.0, 300.0, 0.0, {}, extra={"dummy_minutes": 75.0})
    wf = fit_rapm_dummy(data.games, inputs, tuned)
    dummy = build_dummy_columns(inputs.spell_index)
    reference = fit_walk_forward_dummy(
        data.games,
        inputs.rows,
        inputs.minutes,
        inputs.spell_index,
        dummy,
        half_life_days=365.0,
        ridge_o=300.0,
        ridge_d=300.0,
        dummy_minutes=75.0,
    )
    np.testing.assert_array_equal(wf.home_coef, reference.home_coef)


def test_variant_is_registered_and_chosen_only_when_it_wins_on_tuning() -> None:
    """End-to-end through ``run_m3_backtest``: ``rapm_dummy`` is a declared variant (appears in
    the tuning grid report) and the report's chosen variant is whichever has the lower tuning
    RMSE -- exactly the acceptance rule H6 states."""
    data = make_synthetic_m3(
        list(range(2015, 2021)),
        teams=8,
        games_per_season=18,
        stints_per_game=10,
        roster=8,
        seed=71,
    )
    spec = M3Backtest(
        report=None,  # type: ignore[arg-type]
        warmup=(2015, 2016),
        tuning=(2017, 2018),
        validation=(2019,),
        test=(2020,),
        grid=M3Grid(half_life_days=(365.0,), ridge=(400.0,), dummy_minutes=(0.0, 50.0, 100.0)),
    )

    def _zero_baseline(
        games: pd.DataFrame,
        player_games: pd.DataFrame,
        shares: pd.DataFrame,
        possessions: pd.Series,
        tuning: FloatArray,
    ) -> BaselineResult:
        return BaselineResult(margin=np.zeros(len(games)), params={"kind": "zero"})

    tuned_m1 = {
        "rating": {"half_life_days": 365.0, "carry": 1.0, "ridge": 500.0},
        "pace": {"half_life_days": 365.0, "carry": 1.0, "ridge": 2.0},
        "margin": {"variant": "normal_const", "scale": 10.0, "df": None, "ref_pace": None},
        "totals_sigma": 12.0,
    }
    report = run_m3_backtest(
        data.games,
        data.team_games,
        data.player_games,
        data.stints,
        data.checks,
        spec=spec,
        tuned_m1=tuned_m1,
        box_only_fn=_zero_baseline,
        pir_fn=_zero_baseline,
        tuning_only=True,
    )
    # rapm_dummy is a declared, tuned variant: it appears in the grid report with the expected
    # (nonzero-only) threshold candidates, and `run_m3_backtest`'s own `min(tuned, key=...)`
    # chooses it -- as opposed to plain `rapm` -- only when it reports the lower tuning RMSE
    # (the module's own `min` logic, exercised generically by test_m3_backtest.py; this test
    # only checks that `rapm_dummy` is correctly wired into that same machinery).
    assert "rapm_dummy" in report["grid"]
    assert report["grid"]["rapm_dummy"]["dummy_minutes"] == [50.0, 100.0]  # 0 never retried
    chosen_name = report["chosen"]["variant"]
    assert chosen_name in ("rapm", "rapm_dummy")
    if chosen_name == "rapm_dummy":
        assert "dummy_minutes" in report["chosen"]


# --- Edge-case branches -------------------------------------------------------------------


def test_base_accumulator_add_with_no_rows_is_a_no_op() -> None:
    base = BaseAccumulator(n_base=5, half_life_days=365.0)
    base.advance(1000.0)
    base.add(
        np.empty((0, 3), dtype=np.int64),
        np.empty((0, 3)),
        target=np.empty(0),
        weight=np.empty(0),
        time=np.empty(0),
    )
    assert base.gram.nnz == 0
    np.testing.assert_array_equal(base.rhs, np.zeros(5))


def test_solve_dummy_cutoff_with_nothing_touched_returns_all_zero() -> None:
    season = 2020
    spells = [Spell("A1", "AAA", season), Spell("B1", "BBB", season)]
    spell_index = _manual_spell_index(spells)
    dummy = build_dummy_columns(spell_index)
    empty_gram = sparse.csr_matrix((spell_index.n_base, spell_index.n_base))
    empty_rhs = np.zeros(spell_index.n_base)
    decayed_minutes = np.zeros(spell_index.n_spells)
    solve = solve_dummy_cutoff(
        empty_gram, empty_rhs, dummy, decayed_minutes, 50.0, ridge_o=100.0, ridge_d=100.0
    )
    assert len(solve.used) == 0
    np.testing.assert_array_equal(solve.theta, np.zeros(dummy.columns.n_model))
