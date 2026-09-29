"""SPM unit tests (week 9-12 H6, subagent F): coefficient recovery, the ridge-toward-prior
mechanism (``rapm.fit_walk_forward``'s ``prior_fn`` hook), lookup semantics, D2's walk-forward
window (a planted edit and a planted leak), and determinism."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eurohoops.eval.m3_backtest import build_rapm_inputs
from eurohoops.models import box_impact as bi
from eurohoops.models.box_impact import STAT_COLUMNS
from eurohoops.models.rapm import (
    BASE_OFFSET,
    HOME_COL,
    INTERCEPT_COL,
    DecayedRidgeSparse,
    DesignRows,
    Spell,
    SpellIndex,
    WalkForward,
    _round_groups,  # local, deliberately-broken use only (test_m3_leakage.py's own convention)
    build_design_rows,
    build_spell_index,
    fit_walk_forward,
    plain_player_columns,
    rating_lookup,
)
from eurohoops.models.spm import (
    SeasonSpm,
    SpmModel,
    _weighted_mean_std,
    build_prior_fn,
    build_season_spm,
    fit_season_spm,
)
from tests.m3_synthetic import SyntheticM3, make_synthetic_m3

# ---------------------------------------------------------------------------------------------
# fit_season_spm: coefficient recovery on synthetic data (H6 done-when: "coefficients recovered")
# ---------------------------------------------------------------------------------------------


def test_fit_season_spm_recovers_known_coefficients() -> None:
    """Targets generated from a known linear model of the exact features/standardisation
    ``fit_season_spm`` uses internally must be recovered (a near-0 alpha isolates recovery from
    ridge shrinkage bias)."""
    rng = np.random.default_rng(0)
    n_players, n_stats = 500, len(STAT_COLUMNS)
    features = rng.normal(0.0, 4.0, size=(n_players, n_stats))
    weight = rng.uniform(100.0, 3000.0, size=n_players)
    mean, std = _weighted_mean_std(features, weight)
    standardized = (features - mean) / std

    o_coef_true = np.linspace(0.3, 1.8, n_stats)
    d_coef_true = np.linspace(-1.4, 0.6, n_stats)
    o_intercept_true, d_intercept_true = 2.1, -1.3
    noise = rng.normal(0.0, 1e-6, size=n_players)
    target_o = o_intercept_true + standardized @ o_coef_true + noise
    target_d = d_intercept_true + standardized @ d_coef_true + noise

    model = fit_season_spm(2020, features, weight, target_o, target_d, alpha=1e-6)

    assert model.o_intercept == pytest.approx(o_intercept_true, abs=1e-3)
    assert model.d_intercept == pytest.approx(d_intercept_true, abs=1e-3)
    np.testing.assert_allclose(model.o_coef, o_coef_true, atol=1e-3)
    np.testing.assert_allclose(model.d_coef, d_coef_true, atol=1e-3)

    o_pred, d_pred = model.predict(features)
    np.testing.assert_allclose(o_pred, target_o, atol=1e-2)
    np.testing.assert_allclose(d_pred, target_d, atol=1e-2)


def test_fit_season_spm_shrinks_toward_0_as_alpha_grows() -> None:
    """A sanity check that ``alpha`` really is a ridge penalty: a huge alpha must shrink the
    standardised-feature coefficients toward 0, not change them arbitrarily."""
    rng = np.random.default_rng(1)
    n_players, n_stats = 200, len(STAT_COLUMNS)
    features = rng.normal(0.0, 2.0, size=(n_players, n_stats))
    weight = rng.uniform(50.0, 500.0, size=n_players)
    target_o = rng.normal(5.0, 2.0, size=n_players)
    target_d = rng.normal(-3.0, 2.0, size=n_players)

    light = fit_season_spm(2020, features, weight, target_o, target_d, alpha=1e-6)
    heavy = fit_season_spm(2020, features, weight, target_o, target_d, alpha=1.0e9)

    assert np.linalg.norm(heavy.o_coef) < np.linalg.norm(light.o_coef)
    assert np.linalg.norm(heavy.d_coef) < np.linalg.norm(light.d_coef)


# ---------------------------------------------------------------------------------------------
# The ridge-toward-prior mechanism: a hand example against a dense solve, and prior 0 == plain.
# ---------------------------------------------------------------------------------------------


def _hand_example_system() -> tuple[SpellIndex, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The same 3-stint, 2-lineup hand example ``tests/test_rapm.py`` uses (two 5-man lineups,
    three stint sides each), returned as ``(spell_index, cols, vals, target, weight)``."""
    home_players = [f"A{i}" for i in range(1, 6)]
    away_players = [f"B{i}" for i in range(1, 6)]
    season = 2020
    spells = [Spell(p, "AAA", season) for p in home_players] + [
        Spell(p, "BBB", season) for p in away_players
    ]
    positions = {s: i for i, s in enumerate(spells)}
    lookup = pd.DataFrame(
        {
            "player_id": [s.player_id for s in spells],
            "team": [s.team for s in spells],
            "season": [s.season for s in spells],
            "pos": np.arange(len(spells)),
        }
    )
    spell_index = SpellIndex(tuple(spells), positions, lookup)
    home_pos = np.array([spell_index.position(p, "AAA", season) for p in home_players])
    away_pos = np.array([spell_index.position(p, "BBB", season) for p in away_players])

    stints = [(4, 8, 3, 3), (5, 5, 4, 8), (2, 0, 2, 2)]
    rows_cols, rows_vals, target, weight = [], [], [], []
    for home_poss, home_points, away_poss, away_points in stints:
        cols = np.concatenate(
            [[INTERCEPT_COL, HOME_COL], spell_index.o_col(home_pos), spell_index.d_col(away_pos)]
        )
        vals = np.concatenate([[1.0, 1.0], np.ones(5), -np.ones(5)])
        rows_cols.append(cols)
        rows_vals.append(vals)
        target.append(100.0 * home_points / home_poss)
        weight.append(float(home_poss))
        cols = np.concatenate(
            [[INTERCEPT_COL, HOME_COL], spell_index.o_col(away_pos), spell_index.d_col(home_pos)]
        )
        vals = np.concatenate([[1.0, -1.0], np.ones(5), -np.ones(5)])
        rows_cols.append(cols)
        rows_vals.append(vals)
        target.append(100.0 * away_points / away_poss)
        weight.append(float(away_poss))
    return spell_index, np.stack(rows_cols), np.stack(rows_vals), np.array(target), np.array(weight)


def test_ridge_toward_prior_hand_example_matches_a_dense_solve() -> None:
    """``DecayedRidgeSparse.solve``'s ridge-toward-a-nonzero-prior result must match an
    independent dense solve of ``(gram + diag(penalty)) theta = rhs + penalty * prior_mean`` to
    1e-9 (H6 done-when)."""
    spell_index, cols_arr, vals_arr, target_arr, weight_arr = _hand_example_system()
    columns = plain_player_columns(spell_index)
    ridge_o, ridge_d = 300.0, 700.0
    penalty = columns.penalty(ridge_o, ridge_d)
    prior_mean = np.random.default_rng(11).normal(0.0, 5.0, size=columns.n_model)

    n_base = spell_index.n_base
    x_base = np.zeros((6, n_base))
    for r in range(6):
        x_base[r, cols_arr[r]] = vals_arr[r]
    x_model = x_base @ columns.M.toarray()
    w = np.diag(weight_arr)
    gram = x_model.T @ w @ x_model
    rhs = x_model.T @ w @ target_arr
    manual_theta = np.linalg.solve(gram + np.diag(penalty), rhs + penalty * prior_mean)

    model = DecayedRidgeSparse(n_base, half_life_days=365.0, columns=columns)
    model.advance(1000.0)
    model.add(cols_arr, vals_arr, target=target_arr, weight=weight_arr, time=np.full(6, 1000.0))
    solved_theta = model.solve(ridge_o, ridge_d, prior_mean=prior_mean)

    np.testing.assert_allclose(solved_theta, manual_theta, atol=1e-9, rtol=0.0)


def test_prior_fn_returning_zero_reproduces_plain_rapm_exactly() -> None:
    """A ``prior_fn`` that always returns an all-0 array must give byte-identical lookups and
    home coefficients to not passing a ``prior_fn`` at all (H6 done-when)."""
    data = make_synthetic_m3(
        [2020, 2021], teams=8, games_per_season=20, stints_per_game=10, roster=8, seed=21
    )
    spell_index = build_spell_index(data.stints)
    columns = plain_player_columns(spell_index)
    rows = build_design_rows(data.stints, data.checks, data.games, spell_index)
    kwargs = {"half_life_days": 365.0, "ridge_o": 500.0, "ridge_d": 500.0}

    plain = fit_walk_forward(data.games, rows, columns, **kwargs)
    zero_prior = fit_walk_forward(
        data.games, rows, columns, prior_fn=lambda _t, _s, c: np.zeros(c.n_model), **kwargs
    )

    both_nan = np.isnan(plain.home_coef) & np.isnan(zero_prior.home_coef)
    assert (both_nan | (plain.home_coef == zero_prior.home_coef)).all()
    for a, b in zip(plain.lookups, zero_prior.lookups, strict=True):
        assert (a is None) == (b is None)
        if a is not None:
            assert a.o == b.o
            assert a.d == b.d


# ---------------------------------------------------------------------------------------------
# Lookup semantics (H6 done-when): fitted -> theta; unseen column -> the prior; no box history
# either -> the SPM of league-average rates.
# ---------------------------------------------------------------------------------------------


def test_lookup_fills_an_unseen_column_with_the_prior_not_zero() -> None:
    """A spell no design row ever touches ("GHOST") must be reported at the prior ``prior_fn``
    gives it, not at the solver's default 0 for an untouched column."""
    home_players = [f"A{i}" for i in range(1, 6)]
    away_players = [f"B{i}" for i in range(1, 6)]
    season = 2020
    spells = (
        [Spell(p, "AAA", season) for p in home_players]
        + [Spell(p, "BBB", season) for p in away_players]
        + [Spell("GHOST", "AAA", season)]
    )
    positions = {s: i for i, s in enumerate(spells)}
    lookup = pd.DataFrame(
        {
            "player_id": [s.player_id for s in spells],
            "team": [s.team for s in spells],
            "season": [s.season for s in spells],
            "pos": np.arange(len(spells)),
        }
    )
    spell_index = SpellIndex(tuple(spells), positions, lookup)
    columns = plain_player_columns(spell_index)
    home_pos = np.array([spell_index.position(p, "AAA", season) for p in home_players])
    away_pos = np.array([spell_index.position(p, "BBB", season) for p in away_players])

    base_cols = np.concatenate(
        [[INTERCEPT_COL, HOME_COL], spell_index.o_col(home_pos), spell_index.d_col(away_pos)]
    ).reshape(1, -1)
    vals = np.concatenate([[1.0, 1.0], np.ones(5), -np.ones(5)]).reshape(1, -1)
    games = pd.DataFrame(
        {
            "game_id": ["G1"],
            "season": [season],
            "phase": ["RS"],
            "round": [1],
            "tipoff_utc": [pd.Timestamp("2020-10-01T18:00:00Z")],
        }
    )
    ghost_o, ghost_d = 4.5, -2.5

    def prior_fn(_time: float, _season: int, cols_: object) -> np.ndarray:
        pm = np.zeros(cols_.n_model)  # type: ignore[attr-defined]
        ghost_pos = columns.o_labels.index("GHOST")
        pm[columns.o_start + ghost_pos] = ghost_o
        pm[columns.d_start + ghost_pos] = ghost_d
        return pm

    def rows_with_weight(weight: float) -> DesignRows:
        return DesignRows(
            cols=base_cols,
            vals=vals,
            target=np.array([200.0]),
            weight=np.array([weight]),
            time=np.array([0.0]),
            season=np.array([season], dtype=np.int64),
            game_id=np.array(["G1"], dtype=str),
        )

    fit_kwargs = {"half_life_days": 365.0, "ridge_o": 300.0, "ridge_d": 300.0, "prior_fn": prior_fn}
    light = fit_walk_forward(games, rows_with_weight(4.0), columns, **fit_kwargs)
    heavy = fit_walk_forward(games, rows_with_weight(4.0e6), columns, **fit_kwargs)
    light_lookup, heavy_lookup = light.lookups[0], heavy.lookups[0]
    assert light_lookup is not None and heavy_lookup is not None

    # GHOST's column is never touched by a row either way: always exactly the prior, whatever
    # the *other* (seen) columns' evidence looks like -- the posterior equals the prior with no
    # data, regardless of ridge strength.
    assert light_lookup.o["GHOST"] == pytest.approx(ghost_o)
    assert heavy_lookup.o["GHOST"] == pytest.approx(ghost_o)
    assert light_lookup.d["GHOST"] == pytest.approx(ghost_d)
    assert heavy_lookup.d["GHOST"] == pytest.approx(ghost_d)
    # A1 was actually touched by the one stint (prior 0 for it): a much heavier weight on that
    # same stint must move its solved rating, proving it reflects the fit, not a fixed prior.
    assert light_lookup.o["A1"] != pytest.approx(heavy_lookup.o["A1"])


def test_prior_fn_falls_back_to_league_average_for_a_player_missing_from_box_data() -> None:
    """A rapm player entirely missing from the box's ``player_index`` (no box history at all)
    must predict exactly the SPM at 0 features (league-average rates), while a player who *does*
    have box evidence gets a different, feature-driven prediction."""
    spells = [Spell("A1", "AAA", 2020), Spell("A2", "AAA", 2020)]
    positions = {s: i for i, s in enumerate(spells)}
    lookup = pd.DataFrame(
        {
            "player_id": [s.player_id for s in spells],
            "team": [s.team for s in spells],
            "season": [s.season for s in spells],
            "pos": [0, 1],
        }
    )
    spell_index = SpellIndex(tuple(spells), positions, lookup)
    columns = plain_player_columns(spell_index)

    n_stats = len(STAT_COLUMNS)
    o_coef = np.zeros(n_stats)
    o_coef[0] = 2.0
    model = SpmModel(
        season=2020,
        stats=STAT_COLUMNS,
        mean=np.zeros(n_stats),
        std=np.ones(n_stats),
        o_intercept=3.0,
        o_coef=o_coef,
        d_intercept=-2.0,
        d_coef=np.zeros(n_stats),
        alpha=1.0,
        n_players=1,
    )
    player_rate = np.zeros((1, n_stats))
    player_rate[0, 0] = 50.0  # A1: real evidence on stat 0
    snapshot = bi.RoundSnapshot(
        games_idx=np.array([], dtype=np.int64),
        player_rate=player_rate,
        league_rate=np.zeros(n_stats),
        weight=np.array([1.0e9]),  # weight >> k: the shrinkage factor is ~1
    )
    season_spm = SeasonSpm(
        models={2020: model}, snapshot_by_time={0.0: snapshot}, player_index={"A1": 0}
    )  # A2 is absent from player_index entirely: no box history

    prior_fn = build_prior_fn(season_spm, columns)
    prior = prior_fn(0.0, 2020, columns)
    assert prior is not None
    a1_o = prior[columns.o_start + columns.o_labels.index("A1")]
    a2_o = prior[columns.o_start + columns.o_labels.index("A2")]
    assert a1_o == pytest.approx(103.0, abs=1e-3)  # 3.0 + 50.0 * 2.0
    assert a2_o == pytest.approx(3.0, abs=1e-9)  # the intercept alone: league-average rates


def test_prior_fn_returns_none_for_a_season_with_no_spm_model_yet() -> None:
    season_spm = SeasonSpm(models={}, snapshot_by_time={}, player_index={})
    spells = [Spell("A1", "AAA", 2011)]
    lookup = pd.DataFrame({"player_id": ["A1"], "team": ["AAA"], "season": [2011], "pos": [0]})
    columns = plain_player_columns(SpellIndex(tuple(spells), {spells[0]: 0}, lookup))
    prior_fn = build_prior_fn(season_spm, columns)
    assert prior_fn(0.0, 2011, columns) is None


# ---------------------------------------------------------------------------------------------
# D2's walk-forward window: a planted edit changes no SPM coefficient used in season s, but does
# change the one used in season s + 1.
# ---------------------------------------------------------------------------------------------


def _season_spm_for(data: SyntheticM3, **kwargs: float) -> SeasonSpm:
    inputs = build_rapm_inputs(data.stints, data.checks, data.games, data.player_games)
    defaults: dict[str, float] = {
        "rapm_half_life_days": 365.0,
        "rapm_ridge_o": 300.0,
        "rapm_ridge_d": 300.0,
    }
    defaults.update(kwargs)
    return build_season_spm(data.games, data.player_games, inputs, **defaults)  # type: ignore[arg-type]


def test_editing_a_seasons_stints_changes_only_its_own_and_later_seasons_spm() -> None:
    data = make_synthetic_m3(
        [2020, 2021, 2022], teams=8, games_per_season=24, stints_per_game=10, roster=8, seed=31
    )
    before = _season_spm_for(data)
    assert {2021, 2022} <= set(before.models)  # 2020 (the first season) has no earlier rating

    edited_stints = data.stints.copy()
    season_2021_games = set(data.games.loc[data.games["season"] == 2021, "game_id"].astype(str))
    row = edited_stints.index[edited_stints["game_id"].isin(season_2021_games)][0]
    edited_stints.loc[row, "home_points"] = int(edited_stints.loc[row, "home_points"]) + 9
    edited_stints.loc[row, "home_poss"] = int(edited_stints.loc[row, "home_poss"]) + 2

    edited_data = SyntheticM3(
        data.games,
        edited_stints,
        data.checks,
        data.team_games,
        data.player_games,
        data.true_o,
        data.true_d,
    )
    after = _season_spm_for(edited_data)

    assert before.models[2021].to_json() == after.models[2021].to_json()
    assert before.models[2022].to_json() != after.models[2022].to_json()


def test_editing_a_seasons_box_rows_changes_only_its_own_and_later_seasons_spm() -> None:
    data = make_synthetic_m3(
        [2020, 2021, 2022], teams=8, games_per_season=24, stints_per_game=10, roster=8, seed=32
    )
    before = _season_spm_for(data)
    assert {2021, 2022} <= set(before.models)

    edited_players = data.player_games.copy()
    season_2021_games = set(data.games.loc[data.games["season"] == 2021, "game_id"].astype(str))
    rows_2021 = edited_players.index[edited_players["game_id"].isin(season_2021_games)]
    row = rows_2021[0]
    edited_players.loc[row, "pts"] = int(edited_players.loc[row, "pts"]) + 50
    edited_players.loc[row, "poss"] = float(edited_players.loc[row, "poss"]) + 20.0

    edited_data = SyntheticM3(
        data.games,
        data.stints,
        data.checks,
        data.team_games,
        edited_players,
        data.true_o,
        data.true_d,
    )
    after = _season_spm_for(edited_data)

    assert before.models[2021].to_json() == after.models[2021].to_json()
    assert before.models[2022].to_json() != after.models[2022].to_json()


# ---------------------------------------------------------------------------------------------
# Leakage (H6 done-when + item 34): editing any stint or box row of game g or later changes no
# prior, rating or prediction used for game g; a planted leak is shown to fail, then reverted.
# ---------------------------------------------------------------------------------------------

FIT_KWARGS = {"half_life_days": 365.0, "ridge_o": 300.0, "ridge_d": 300.0}


def _fit_rapm_spm(data: SyntheticM3) -> tuple[WalkForward, int]:
    spell_index = build_spell_index(data.stints)
    columns = plain_player_columns(spell_index)
    rows = build_design_rows(data.stints, data.checks, data.games, spell_index)
    inputs = build_rapm_inputs(data.stints, data.checks, data.games, data.player_games)
    season_spm = build_season_spm(
        data.games,
        data.player_games,
        inputs,
        rapm_half_life_days=FIT_KWARGS["half_life_days"],
        rapm_ridge_o=FIT_KWARGS["ridge_o"],
        rapm_ridge_d=FIT_KWARGS["ridge_d"],
    )
    prior_fn = build_prior_fn(season_spm, columns)
    wf = fit_walk_forward(data.games, rows, columns, prior_fn=prior_fn, **FIT_KWARGS)
    g = next(i for i in range(len(data.games)) if wf.lookups[i] is not None and i > 20)
    return wf, g


def test_editing_a_later_stint_row_does_not_change_game_g() -> None:
    data = make_synthetic_m3(
        [2020, 2021], teams=8, games_per_season=24, stints_per_game=12, roster=8, seed=41
    )
    wf_before, g = _fit_rapm_spm(data)
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
    wf_after, g_after = _fit_rapm_spm(edited_data)
    assert g_after == g

    before, after = wf_before.lookups[g], wf_after.lookups[g]
    assert (before is None) == (after is None)
    if before is not None:
        assert before.o == after.o
        assert before.d == after.d


def test_editing_a_later_box_row_does_not_change_game_g() -> None:
    data = make_synthetic_m3(
        [2020, 2021], teams=8, games_per_season=24, stints_per_game=12, roster=8, seed=42
    )
    wf_before, g = _fit_rapm_spm(data)
    game_id = str(data.games.loc[g, "game_id"])

    edited_players = data.player_games.copy()
    own = edited_players.index[edited_players["game_id"] == game_id][0]
    later_game_id = str(data.games.iloc[-1]["game_id"])
    later = edited_players.index[edited_players["game_id"] == later_game_id][0]
    edited_players.loc[own, "pts"] = int(edited_players.loc[own, "pts"]) + 50
    edited_players.loc[own, "poss"] = float(edited_players.loc[own, "poss"]) + 20.0
    edited_players.loc[later, "pts"] = int(edited_players.loc[later, "pts"]) + 50

    edited_data = SyntheticM3(
        data.games,
        data.stints,
        data.checks,
        data.team_games,
        edited_players,
        data.true_o,
        data.true_d,
    )
    wf_after, g_after = _fit_rapm_spm(edited_data)
    assert g_after == g

    before, after = wf_before.lookups[g], wf_after.lookups[g]
    assert (before is None) == (after is None)
    if before is not None:
        assert before.o == after.o
        assert before.d == after.d


def _leaky_walk_forward(
    games: pd.DataFrame,
    rows: DesignRows,
    columns: object,
    *,
    half_life_days: float,
    ridge_o: float,
    ridge_d: float,
    prior_fn: object = None,
) -> WalkForward:
    """A byte-for-byte copy of ``rapm.fit_walk_forward`` (with the ``prior_fn`` branch it added),
    except the round-cutoff comparison uses ``side="right"`` -- the same planted leak
    ``tests/test_m3_leakage.py`` uses for the plain variant, reproduced here to prove the check
    above has power with a nonzero per-cutoff SPM prior wired in too. The real
    ``fit_walk_forward`` (``side="left"``) is never modified."""
    n_base = rows.n_base if len(rows.cols) else BASE_OFFSET
    model = DecayedRidgeSparse(n_base, half_life_days, columns)  # type: ignore[arg-type]
    n_games = len(games)
    home_coef = np.full(n_games, np.nan)
    lookups: list = [None] * n_games
    theta = np.zeros(columns.n_model)  # type: ignore[attr-defined]
    added = 0
    for time, season, game_idx in _round_groups(games):
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
        if prior_fn is not None:
            pm = prior_fn(time, season, columns)  # type: ignore[operator]
            theta = model.solve(ridge_o, ridge_d, pm, x0=theta)
            filled = theta if pm is None else np.where(model.model_seen, theta, pm)
        else:
            theta = model.solve(ridge_o, ridge_d, x0=theta)
            filled = theta
        coef = 2.0 * theta[HOME_COL]
        lookup = rating_lookup(filled, columns)  # type: ignore[arg-type]
        home_coef[game_idx] = coef
        for g in game_idx:
            lookups[g] = lookup
    return WalkForward(home_coef, lookups, columns, model)  # type: ignore[arg-type]


def test_planted_cutoff_leak_is_caught_for_rapm_spm() -> None:
    data = make_synthetic_m3(
        [2020, 2021], teams=8, games_per_season=24, stints_per_game=12, roster=8, seed=43
    )
    spell_index = build_spell_index(data.stints)
    columns = plain_player_columns(spell_index)
    rows = build_design_rows(data.stints, data.checks, data.games, spell_index)
    inputs = build_rapm_inputs(data.stints, data.checks, data.games, data.player_games)
    season_spm = build_season_spm(
        data.games,
        data.player_games,
        inputs,
        rapm_half_life_days=FIT_KWARGS["half_life_days"],
        rapm_ridge_o=FIT_KWARGS["ridge_o"],
        rapm_ridge_d=FIT_KWARGS["ridge_d"],
    )
    prior_fn = build_prior_fn(season_spm, columns)
    wf_before = _leaky_walk_forward(data.games, rows, columns, prior_fn=prior_fn, **FIT_KWARGS)
    g = next(i for i in range(len(data.games)) if wf_before.lookups[i] is not None and i > 20)
    game_id = str(data.games.loc[g, "game_id"])

    edited = data.stints.copy()
    own_row = edited.index[edited["game_id"] == game_id][0]
    edited.loc[own_row, "home_points"] = int(edited.loc[own_row, "home_points"]) + 25
    edited.loc[own_row, "home_poss"] = int(edited.loc[own_row, "home_poss"]) + 3

    edited_spell_index = build_spell_index(edited)
    edited_columns = plain_player_columns(edited_spell_index)
    edited_rows = build_design_rows(edited, data.checks, data.games, edited_spell_index)
    edited_inputs = build_rapm_inputs(edited, data.checks, data.games, data.player_games)
    edited_season_spm = build_season_spm(
        data.games,
        data.player_games,
        edited_inputs,
        rapm_half_life_days=FIT_KWARGS["half_life_days"],
        rapm_ridge_o=FIT_KWARGS["ridge_o"],
        rapm_ridge_d=FIT_KWARGS["ridge_d"],
    )
    edited_prior_fn = build_prior_fn(edited_season_spm, edited_columns)
    wf_after = _leaky_walk_forward(
        data.games, edited_rows, edited_columns, prior_fn=edited_prior_fn, **FIT_KWARGS
    )

    before, after = wf_before.lookups[g], wf_after.lookups[g]
    assert before is not None and after is not None
    assert before.o != after.o or before.d != after.d  # the planted leak changed g's own rating


# ---------------------------------------------------------------------------------------------
# Determinism: two runs on the same inputs give identical SPM models and walk-forward results.
# ---------------------------------------------------------------------------------------------


def test_two_runs_are_identical() -> None:
    data = make_synthetic_m3(
        [2020, 2021], teams=8, games_per_season=20, stints_per_game=10, roster=8, seed=51
    )
    first = _season_spm_for(data)
    second = _season_spm_for(data)
    assert {s: m.to_json() for s, m in first.models.items()} == {
        s: m.to_json() for s, m in second.models.items()
    }

    wf_first, _ = _fit_rapm_spm(data)
    wf_second, _ = _fit_rapm_spm(data)
    both_nan = np.isnan(wf_first.home_coef) & np.isnan(wf_second.home_coef)
    assert (both_nan | (wf_first.home_coef == wf_second.home_coef)).all()
    for a, b in zip(wf_first.lookups, wf_second.lookups, strict=True):
        assert (a is None) == (b is None)
        if a is not None:
            assert a.o == b.o
            assert a.d == b.d
