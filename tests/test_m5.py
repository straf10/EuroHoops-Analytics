"""J3: M5 model core: walk-forward residual and rest fit, probit blend, Platt, over/under."""

import math

import numpy as np
import pandas as pd
import pytest
from scipy import special

from eurohoops.models.m5 import (
    apply_platt,
    blend_by_season,
    fit_blend,
    fit_platt,
    fit_residual,
    p_cover,
    p_over,
)
from eurohoops.models.team_eff import MarginModel

KW = {"half_life_days": 200.0, "residual_ridge": 2.0, "rest_ridge": 1.0}


def schedule(n_teams: int, n_rounds: int, rng: np.random.Generator) -> pd.DataFrame:
    """Every team plays once per round (random pairing, random home side); one season."""
    rows = []
    start = pd.Timestamp("2025-10-01", tz="UTC")
    for rnd in range(n_rounds):
        order = rng.permutation(n_teams)
        for i in range(n_teams // 2):
            rows.append(
                {
                    "game_id": f"g{rnd}-{i}",
                    "season": 2025,
                    "phase": "RS",
                    "round": rnd + 1,
                    "tipoff_utc": start + pd.Timedelta(days=7 * rnd, hours=i),
                    "home": f"T{order[2 * i]}",
                    "away": f"T{order[2 * i + 1]}",
                    "played": True,
                    "forfeit": False,
                    "neutral": False,
                }
            )
    return pd.DataFrame(rows)


def design(games: pd.DataFrame, rest: np.ndarray, teams: list[str]) -> np.ndarray:
    """Dense rows [h, r_team…, β…] of every game (an independent construction of the design)."""
    x = np.zeros((len(games), 1 + len(teams) + rest.shape[1]))
    for g, (home, away, neutral) in enumerate(
        zip(games["home"], games["away"], games["neutral"], strict=True)
    ):
        x[g, 0] = 0.0 if neutral else 1.0
        x[g, 1 + teams.index(home)] = 1.0
        x[g, 1 + teams.index(away)] = -1.0
    x[:, 1 + len(teams) :] = rest
    return x


def test_synthetic_recovery_200_datasets() -> None:
    """Late-season adjustment, h and β agree with the planted values.

    Per quantity c·θ the tolerance is |expected ridge bias| + 5 sd. With A = XᵀX + P (X = the
    known design of the rows before the last round, P the penalty) and unit weights (infinite
    half-life), θ̂ = A⁻¹Xᵀy has mean A⁻¹XᵀXθ and covariance σ² A⁻¹XᵀX A⁻¹, so the bias is
    -A⁻¹Pθ and sd = σ sqrt(c A⁻¹XᵀX A⁻¹ c). Both come from the planted truth and the design,
    never from the fit. 5 sd keeps the chance of any miss over ~6000 checks below 1e-2.
    """
    n_teams, n_rounds, sigma, ridge, rest_ridge = 8, 30, 8.0, 0.5, 0.5
    h_true, beta_true = 3.0, 0.7
    teams = [f"T{i}" for i in range(n_teams)]
    penalty = np.diag([1e-6, *[ridge] * n_teams, rest_ridge])
    for seed in range(200):
        rng = np.random.default_rng(seed)
        games = schedule(n_teams, n_rounds, rng)
        rest = rng.integers(-3, 4, size=(len(games), 1)).astype(np.float64)
        theta = np.concatenate([[h_true], rng.normal(0.0, 2.0, n_teams), [beta_true]])
        x = design(games, rest, teams)
        margin = x @ theta + rng.normal(0.0, sigma, len(games))
        fit = fit_residual(
            games,
            margin,
            rest,
            half_life_days=math.inf,
            residual_ridge=ridge,
            rest_ridge=rest_ridge,
        )
        train = (games["round"] < n_rounds).to_numpy()
        gram = x[train].T @ x[train]
        inv = np.linalg.inv(gram + penalty)
        cov = sigma**2 * inv @ gram @ inv
        bias = -inv @ penalty @ theta
        last = np.flatnonzero(~train)
        unit_h = np.eye(len(theta))[0]
        unit_b = np.eye(len(theta))[-1]
        for g in last:
            for functional, estimate in (
                (x[g], fit.adjustment[g]),
                (unit_h, fit.home[g]),
                (unit_b, fit.rest_coef[g, 0]),
            ):
                tolerance = abs(functional @ bias) + 5.0 * math.sqrt(functional @ cov @ functional)
                assert abs(estimate - functional @ theta) < tolerance, (seed, g)


def test_matches_direct_normal_equations() -> None:
    rng = np.random.default_rng(7)
    teams = [f"T{i}" for i in range(6)]
    games = schedule(6, 3, rng)
    target = rng.normal(0.0, 5.0, len(games))
    ridge = 1.7
    fit = fit_residual(
        games, target, None, half_life_days=math.inf, residual_ridge=ridge, rest_ridge=1.0
    )
    x = design(games, np.zeros((len(games), 0)), teams)
    earlier = (games["round"] < 3).to_numpy()  # rounds 1-2 feed the round-3 cutoff
    penalty = np.diag([1e-6, *[ridge] * len(teams)])
    theta = np.linalg.solve(x[earlier].T @ x[earlier] + penalty, x[earlier].T @ target[earlier])
    third = np.flatnonzero(~earlier)
    np.testing.assert_allclose(fit.adjustment[third], x[third] @ theta, atol=1e-9, rtol=0)
    np.testing.assert_allclose(fit.home[third], theta[0], atol=1e-9, rtol=0)
    assert fit.rest_coef.shape == (len(games), 0)
    assert np.isnan(fit.adjustment[games["round"] == 1]).all()  # nothing earlier


def test_no_leakage_and_earlier_rows_matter() -> None:
    rng = np.random.default_rng(11)
    games = schedule(6, 8, rng)
    target = rng.normal(0.0, 5.0, len(games))
    rest = rng.normal(0.0, 1.0, (len(games), 2))
    base = fit_residual(games, target, rest, **KW)
    rnd = 5
    in_round = (games["round"] == rnd).to_numpy()
    cutoff = games.loc[in_round, "tipoff_utc"].min()
    later = (games["tipoff_utc"] >= cutoff).to_numpy()
    # at or after the cutoff: every target, and the rest features of the other games (a game's own
    # rest features are a legitimate predictor of its own adjustment, not a fit input)
    target2, rest2 = target.copy(), rest.copy()
    target2[later] += rng.normal(0.0, 50.0, later.sum())
    rest2[later & ~in_round] += 9.0
    moved = fit_residual(games, target2, rest2, **KW)
    for name in ("adjustment", "home", "rest_coef"):
        np.testing.assert_array_equal(
            getattr(moved, name)[in_round], getattr(base, name)[in_round], err_msg=name
        )
        np.testing.assert_array_equal(
            getattr(moved, name)[~later], getattr(base, name)[~later], err_msg=name
        )
    # the check can fail: an earlier row moves the round's fit
    early = np.flatnonzero((games["round"] == 2).to_numpy())[0]
    target3 = target.copy()
    target3[early] += 10.0
    assert not np.array_equal(
        fit_residual(games, target3, rest, **KW).home[in_round], base.home[in_round]
    )
    rest3 = rest.copy()
    rest3[early] += 5.0
    assert not np.array_equal(
        fit_residual(games, target, rest3, **KW).rest_coef[in_round], base.rest_coef[in_round]
    )


def test_nan_target_and_forfeit_rows_are_never_fitted() -> None:
    rng = np.random.default_rng(3)
    games = schedule(6, 6, rng)
    target = rng.normal(0.0, 5.0, len(games))
    nan_target = target.copy()
    nan_target[[4, 9]] = np.nan
    forfeit = games.copy()
    forfeit.loc[[4, 9], "forfeit"] = True
    a = fit_residual(games, nan_target, None, **KW)
    b = fit_residual(forfeit, target, None, **KW)
    np.testing.assert_array_equal(a.adjustment, b.adjustment)


def test_zero_residual_gives_zero_adjustment_so_core_margin_is_rapm() -> None:
    """target ≡ 0 (actual margin = player part) => every coefficient is 0 => adjustment 0, so
    the core margin ``player_part + adjustment`` equals ``rapm_margins`` exactly."""
    rng = np.random.default_rng(5)
    games = schedule(6, 6, rng)
    rest = rng.normal(0.0, 1.0, (len(games), 1))
    player_part = rng.normal(0.0, 6.0, len(games))
    fit = fit_residual(games, np.zeros(len(games)), rest, **KW)
    fitted = np.isfinite(fit.adjustment)
    assert fitted.any()
    np.testing.assert_allclose(fit.adjustment[fitted], 0.0, atol=1e-9)
    np.testing.assert_allclose(
        (player_part + fit.adjustment)[fitted], player_part[fitted], atol=1e-9
    )


def test_blend_puts_weight_on_the_generating_component() -> None:
    """n = 20000: each wrong component's probit coefficient has sd about
    1 / (sqrt(n) · sd(m) · 0.7) < 0.003 against a true 1/8 = 0.125, so its weight is far
    below the 0.1 that would break the > 0.9 bound."""
    rng = np.random.default_rng(1)
    n = 20_000
    components = rng.normal(0.0, 8.0, (n, 3))
    home_won = (components[:, 1] / 8.0 + rng.normal(size=n) > 0).astype(np.float64)
    weights = fit_blend(components, home_won)
    assert weights[1] > 0.9
    assert (weights >= 0).all()
    assert weights.sum() == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_array_equal(weights, fit_blend(components, home_won))
    one_hot = np.array([0.0, 1.0, 0.0])
    np.testing.assert_array_equal(components @ one_hot, components[:, 1])


def test_blend_equal_weights_when_all_coefficients_are_zero() -> None:
    m = np.array([[1.0, 2.0], [1.0, 2.0], [-1.0, -2.0], [-1.0, -2.0]])
    # outcomes opposite to every margin: the likelihood is maximised at b = 0
    np.testing.assert_array_equal(fit_blend(m, np.array([0.0, 0.0, 1.0, 1.0])), [0.5, 0.5])


def test_blend_by_season_uses_only_earlier_seasons() -> None:
    rng = np.random.default_rng(2)
    n = 600
    season = np.repeat([1, 2, 3], n // 3).astype(np.int64)
    components = rng.normal(0.0, 8.0, (n, 2))
    home_won = (components[:, 0] / 8.0 + rng.normal(size=n) > 0).astype(np.float64)
    rated = np.ones(n, dtype=bool)
    base = blend_by_season(components, home_won, season, rated)
    assert np.isnan(base[season == 1]).all()
    np.testing.assert_array_equal(
        base[season == 3][0], fit_blend(components[season < 3], home_won[season < 3])
    )
    for s in (2, 3):
        flipped = home_won.copy()
        flipped[season >= s] = 1.0 - flipped[season >= s]
        moved = blend_by_season(components, flipped, season, rated)
        np.testing.assert_array_equal(moved[season <= s], base[season <= s])
    flipped = home_won.copy()
    flipped[season == 2] = 1.0 - flipped[season == 2]
    assert not np.array_equal(
        blend_by_season(components, flipped, season, rated)[season == 3], base[season == 3]
    )
    unrated = rated & (season != 1)
    assert np.isnan(blend_by_season(components, home_won, season, unrated)[season == 2]).all()


def test_over_and_cover_are_monotone_and_half_at_the_centre() -> None:
    total = np.array([150.0, 160.0, 171.5])
    lines = np.linspace(120.0, 200.0, 41)
    over = np.stack([p_over(total, 12.0, line) for line in lines])
    assert (np.diff(over, axis=0) < 0).all()
    np.testing.assert_allclose(p_over(total, 12.0, total), 0.5, atol=1e-15)
    model = MarginModel("normal_const", 11.0)
    margin = np.array([-8.0, 0.5, 6.0])
    cover = np.stack([p_cover(margin, model, line) for line in np.linspace(-20, 20, 41)])
    assert (np.diff(cover, axis=0) < 0).all()
    np.testing.assert_allclose(p_cover(margin, model, margin), 0.5, atol=1e-15)
    student = MarginModel("student_t_const", 11.0, df=5.0)
    np.testing.assert_allclose(p_cover(margin, student, margin), 0.5, atol=1e-15)
    with pytest.raises(ValueError):
        p_cover(margin, MarginModel("normal_pace", 11.0, None, 70.0), 0.0)


def test_platt_recovers_identity_on_calibrated_probabilities() -> None:
    """y ~ Bernoulli(p) so the truth is (a, b) = (0, 1). The tolerance is 5 sd from the Fisher
    information Σ p(1-p)[[1, x], [x, x²]] (x = logit p) of the known probabilities."""
    rng = np.random.default_rng(4)
    p = rng.uniform(0.05, 0.95, 40_000)
    y = (rng.uniform(size=p.size) < p).astype(np.float64)
    a, b = fit_platt(p, y)
    x = special.logit(p)
    info = (p * (1 - p) * np.stack([np.ones_like(x), x, x, x**2])).reshape(4, -1).sum(axis=1)
    sd = np.sqrt(np.diag(np.linalg.inv(info.reshape(2, 2))))
    assert abs(a) < 5.0 * sd[0]
    assert abs(b - 1.0) < 5.0 * sd[1]
    assert fit_platt(p, y) == (a, b)


def test_apply_platt_identity_and_clipping() -> None:
    p = np.random.default_rng(6).uniform(0.01, 0.99, 500)
    np.testing.assert_allclose(apply_platt(p, 0.0, 1.0), p, atol=1e-12, rtol=0)
    out = apply_platt(np.array([0.0, 1.0]), 0.0, 1.0)
    assert np.isfinite(out).all()
    assert out[0] == pytest.approx(1e-9, rel=1e-6)


def test_all_deterministic() -> None:
    rng = np.random.default_rng(9)
    games = schedule(6, 6, rng)
    target = rng.normal(0.0, 5.0, len(games))
    rest = rng.normal(0.0, 1.0, (len(games), 1))
    one, two = fit_residual(games, target, rest, **KW), fit_residual(games, target, rest, **KW)
    for name in ("adjustment", "home", "rest_coef"):
        np.testing.assert_array_equal(getattr(one, name), getattr(two, name))
    m = rng.normal(0.0, 8.0, (300, 2))
    y = (m[:, 0] + rng.normal(0.0, 8.0, 300) > 0).astype(np.float64)
    np.testing.assert_array_equal(fit_blend(m, y), fit_blend(m, y))
    p = special.expit(m[:, 0] / 8.0)
    assert fit_platt(p, y) == fit_platt(p, y)
    np.testing.assert_array_equal(apply_platt(p, 0.1, 0.9), apply_platt(p, 0.1, 0.9))
