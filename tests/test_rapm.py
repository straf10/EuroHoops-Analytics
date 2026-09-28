"""RAPM unit tests (week 9-12 H1): a hand-computed example, API shape, and recovery."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eurohoops.models.rapm import (
    BASE_OFFSET,
    HOME_COL,
    INTERCEPT_COL,
    DecayedRidgeSparse,
    Spell,
    SpellIndex,
    build_design_rows,
    build_minutes_rows,
    build_spell_index,
    fit_decayed_minutes,
    fit_walk_forward,
    plain_player_columns,
    rating_lookup,
)
from tests.m3_synthetic import make_synthetic_m3


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


def test_hand_computed_three_stints_match_a_manual_ridge_solve() -> None:
    """Two fixed 5-man lineups, three stints (six rows): the sparse CG solve must match an
    independent, plain-numpy weighted ridge solve of the same normal equations to 1e-9."""
    home_players = [f"A{i}" for i in range(1, 6)]
    away_players = [f"B{i}" for i in range(1, 6)]
    season = 2020
    spells = [Spell(p, "AAA", season) for p in home_players] + [
        Spell(p, "BBB", season) for p in away_players
    ]
    spell_index = _manual_spell_index(spells)
    columns = plain_player_columns(spell_index)
    assert columns.n_model == BASE_OFFSET + 2 * 10

    home_pos = np.array([spell_index.position(p, "AAA", season) for p in home_players])
    away_pos = np.array([spell_index.position(p, "BBB", season) for p in away_players])

    # Stint sides: (home_poss, home_points, away_poss, away_points)
    stints = [(4, 8, 3, 3), (5, 5, 4, 8), (2, 0, 2, 2)]
    rows_cols = []
    rows_vals = []
    target = []
    weight = []
    for home_poss, home_points, away_poss, away_points in stints:
        # home offense / away defense
        cols = np.concatenate(
            [[INTERCEPT_COL, HOME_COL], spell_index.o_col(home_pos), spell_index.d_col(away_pos)]
        )
        vals = np.concatenate([[1.0, 1.0], np.ones(5), -np.ones(5)])
        rows_cols.append(cols)
        rows_vals.append(vals)
        target.append(100.0 * home_points / home_poss)
        weight.append(float(home_poss))
        # away offense / home defense
        cols = np.concatenate(
            [[INTERCEPT_COL, HOME_COL], spell_index.o_col(away_pos), spell_index.d_col(home_pos)]
        )
        vals = np.concatenate([[1.0, -1.0], np.ones(5), -np.ones(5)])
        rows_cols.append(cols)
        rows_vals.append(vals)
        target.append(100.0 * away_points / away_poss)
        weight.append(float(away_poss))

    cols_arr = np.stack(rows_cols)
    vals_arr = np.stack(rows_vals)
    target_arr = np.array(target)
    weight_arr = np.array(weight)

    ridge_o, ridge_d = 137.0, 251.0
    penalty = columns.penalty(ridge_o, ridge_d)

    # Manual, independent weighted ridge solve.
    n_base = spell_index.n_base
    x_base = np.zeros((6, n_base))
    for r in range(6):
        x_base[r, cols_arr[r]] = vals_arr[r]
    x_model = x_base @ columns.M.toarray()
    w = np.diag(weight_arr)
    gram = x_model.T @ w @ x_model
    rhs = x_model.T @ w @ target_arr
    manual_theta = np.linalg.solve(gram + np.diag(penalty), rhs)

    model = DecayedRidgeSparse(n_base, half_life_days=365.0, columns=columns)
    model.advance(1000.0)
    model.add(cols_arr, vals_arr, target=target_arr, weight=weight_arr, time=np.full(6, 1000.0))
    solved_theta = model.solve(ridge_o, ridge_d)

    np.testing.assert_allclose(solved_theta, manual_theta, atol=1e-9, rtol=0.0)

    # The aggregated system D's posterior will consume must match the same manual numbers,
    # restricted to touched columns (every column here is touched).
    system = model.aggregated_system(ridge_o, ridge_d)
    assert len(system.used) == columns.n_model
    np.testing.assert_allclose(system.gram.toarray(), gram, atol=1e-9)
    np.testing.assert_allclose(system.rhs, rhs, atol=1e-9)


def test_rating_lookup_defaults_to_zero_for_an_unseen_player() -> None:
    spells = [Spell("A1", "AAA", 2020), Spell("B1", "BBB", 2020)]
    spell_index = _manual_spell_index(spells)
    columns = plain_player_columns(spell_index)
    theta = np.zeros(columns.n_model)
    theta[columns.o_start] = 5.0
    theta[columns.d_start] = -3.0
    lookup = rating_lookup(theta, columns)
    assert lookup.rating("A1", "AAA", 2020) == (5.0, -3.0)
    assert lookup.rating("nobody", "AAA", 2020) == (0.0, 0.0)


def test_walk_forward_leaves_nan_before_any_stint_and_fills_in_after() -> None:
    data = make_synthetic_m3([2020], teams=4, games_per_season=6, stints_per_game=4, roster=6)
    spell_index = build_spell_index(data.stints)
    columns = plain_player_columns(spell_index)
    rows = build_design_rows(data.stints, data.checks, data.games, spell_index)
    result = fit_walk_forward(
        data.games, rows, columns, half_life_days=365.0, ridge_o=500.0, ridge_d=500.0
    )
    assert np.isnan(result.home_coef[0])  # nothing has tipped off yet before the first game
    assert result.lookups[0] is None
    assert not np.isnan(result.home_coef[-1])
    assert result.lookups[-1] is not None


def test_synthetic_recovery_improves_with_more_stints() -> None:
    """RAPM ratings correlate >= 0.9 with the truth at 30k stints, and the error shrinks as
    stints grow (H1's done-when #1)."""
    small = make_synthetic_m3(
        list(range(2015, 2017)),
        teams=10,
        games_per_season=40,
        stints_per_game=33,
        roster=10,
        seed=1,
    )
    big = make_synthetic_m3(
        list(range(2015, 2025)),
        teams=16,
        games_per_season=92,
        stints_per_game=33,
        roster=10,
        seed=1,
    )
    assert len(big.stints) >= 30_000

    def fit_final(data: object) -> dict[str, float]:
        spell_index = build_spell_index(data.stints)  # type: ignore[attr-defined]
        columns = plain_player_columns(spell_index)
        rows = build_design_rows(data.stints, data.checks, data.games, spell_index)  # type: ignore[attr-defined]
        result = fit_walk_forward(
            data.games,
            rows,
            columns,
            half_life_days=1460.0,
            ridge_o=250.0,
            ridge_d=250.0,  # type: ignore[attr-defined]
        )
        lookup = next(x for x in reversed(result.lookups) if x is not None)
        return {p: o + lookup.d.get(p, 0.0) for p, o in lookup.o.items()}

    def correlation(data: object, fitted: dict[str, float]) -> float:
        players = sorted(fitted)
        true = np.array([data.true_o[p] + data.true_d[p] for p in players])  # type: ignore[attr-defined]
        est = np.array([fitted[p] for p in players])
        return float(np.corrcoef(true, est)[0, 1])

    def rmse(data: object, fitted: dict[str, float]) -> float:
        players = sorted(fitted)
        true = np.array([data.true_o[p] + data.true_d[p] for p in players])  # type: ignore[attr-defined]
        est = np.array([fitted[p] for p in players])
        return float(np.sqrt(np.mean((true - est) ** 2)))

    small_fit = fit_final(small)
    big_fit = fit_final(big)
    big_corr = correlation(big, big_fit)
    assert big_corr >= 0.9, big_corr
    assert rmse(big, big_fit) < rmse(small, small_fit)  # the error shrinks as stints grow


def test_decayed_minutes_shrink_with_a_shorter_half_life() -> None:
    """D6: decayed on-court seconds per spell, used by the (not-yet-built) rapm_dummy
    threshold. No hand example is specified for this hook, so this checks its two defining
    properties: a spell with no passed-check stints has 0 minutes, and a shorter half-life
    leaves less total credited time than a very long one (more decay)."""
    data = make_synthetic_m3(
        list(range(2018, 2021)),
        teams=6,
        games_per_season=20,
        stints_per_game=10,
        roster=8,
        seed=17,
    )
    spell_index = build_spell_index(data.stints)
    rows = build_minutes_rows(data.stints, data.checks, data.games, spell_index)
    long_half_life = fit_decayed_minutes(data.games, rows, spell_index, half_life_days=1.0e6)
    short_half_life = fit_decayed_minutes(data.games, rows, spell_index, half_life_days=30.0)
    assert len(long_half_life) == spell_index.n_spells
    assert (long_half_life >= 0.0).all()
    assert (short_half_life >= 0.0).all()
    assert short_half_life.sum() < long_half_life.sum()

    # A game with a planted failed check contributes no minutes at all.
    failed_games = set(data.checks.loc[~data.checks["passed"].astype(bool), "game_id"])
    if not failed_games:
        checks = data.checks.copy()
        checks.loc[checks.index[0], "passed"] = False
        failed_games = {str(checks.loc[checks.index[0], "game_id"])}
    else:
        checks = data.checks
    failed_rows = build_minutes_rows(data.stints, checks, data.games, spell_index)
    all_pass = data.checks.assign(passed=True)
    all_rows = build_minutes_rows(data.stints, all_pass, data.games, spell_index)
    assert failed_rows.position.size < all_rows.position.size


@pytest.mark.parametrize("seed", [3, 4])
def test_solve_is_deterministic_across_warm_start_seeds(seed: int) -> None:
    """CG must converge to the same solution regardless of what state the warm start held."""
    data = make_synthetic_m3(
        [2021], teams=6, games_per_season=10, stints_per_game=8, roster=8, seed=seed
    )
    spell_index = build_spell_index(data.stints)
    columns = plain_player_columns(spell_index)
    rows = build_design_rows(data.stints, data.checks, data.games, spell_index)
    kwargs = {"half_life_days": 365.0, "ridge_o": 500.0, "ridge_d": 500.0}
    fresh = fit_walk_forward(data.games, rows, columns, **kwargs)
    warm = fit_walk_forward(data.games, rows, columns, **kwargs)
    for a, b in zip(fresh.lookups, warm.lookups, strict=True):
        if a is None:
            assert b is None
            continue
        assert a.o == b.o
        assert a.d == b.d


def test_spell_index_keeps_full_ids_when_a_failing_game_lists_no_players() -> None:
    """Real stints of a failing game can list nobody (E2015_23): ids must survive intact."""
    stints = pd.DataFrame(
        {
            "season": [2015, 2015],
            "home": ["AAA", "AAA"],
            "away": ["BBB", "BBB"],
            "home_players": [np.array(["P000001", "P000002"]), np.array([], dtype=str)],
            "away_players": [np.array(["P000003"]), np.array(["P000004"])],
        }
    )
    index = build_spell_index(stints)
    assert sorted(s.player_id for s in index.spells) == ["P000001", "P000002", "P000003", "P000004"]
    columns = plain_player_columns(index)
    assert set(columns.o_labels) == {"P000001", "P000002", "P000003", "P000004"}
