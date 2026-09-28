"""M3 H4 tests: the closed-form Bayesian RAPM posterior (``models/rapm_posterior.py``).

The coverage test (H4's Done-when) simulates 200 RAPM-like synthetic seasons: random 5-v-5
lineups from a player pool, offense +1 / defense -1 blocks, possession weights, a true rating
per player drawn from the declared prior, noise at a known sigma, sigma re-estimated from the
fit's own residuals via ``noise_variance`` (never the true value), then checks the share of
(player, season) pairs whose 90% interval covers the truth lands in [0.87, 0.93].

A first, smaller design (more overlap between lineups, fewer stints per season) gave coverage
that swung well outside that band from one seed to the next even though the *mean* z-score was
correctly centred: every rating column in a season shares one estimated sigma^2 and a highly
overlapping design, so a season's ~120 (player, column) checks are far from 120 independent
draws — one unlucky sigma^2 estimate shifts all of them together. The parameters below (a
60-player pool, so any two players share only ~1/6 of stints, and enough stints that sigma^2's
own sampling noise is small) were chosen empirically to make the 200-season estimate land
inside [0.87, 0.93] for every seed tried (1, 2, 3, 7, 42, 999, 12345, 20261001), not just the
one fixed below; see ``reports/week9-12_progress.md`` for the exploration.
"""

import numpy as np
import pytest
from scipy import sparse

from eurohoops.models.elo import FloatArray
from eurohoops.models.rapm_posterior import (
    MAX_DENSE_PARAMS,
    Posterior,
    noise_variance,
    posterior,
    posterior_from_normal_equations,
    ridge_solution,
)

# --- H4 Done-when: posterior mean equals an independent ridge solution to 1e-9 ----------------


def test_posterior_mean_equals_an_independently_solved_ridge_solution() -> None:
    """A mix of penalised and ~unpenalised columns, a nonzero prior mean, moderate n."""
    rng = np.random.default_rng(1)
    n, p = 200, 6
    x_dense = rng.normal(size=(n, p))
    x = sparse.csr_matrix(x_dense)
    weight = rng.uniform(1.0, 5.0, n)
    theta = rng.normal(size=p)
    target = x @ theta + rng.normal(scale=0.5, size=n)
    # columns 0-1 ~unpenalised (like an intercept/home column), 2-3 lightly, 4-5 heavily.
    penalty = np.array([1e-8, 1e-8, 3.0, 3.0, 25.0, 25.0])
    prior_mean = np.array([1.0, -2.0, 0.5, 0.0, -0.75, 2.0])

    fitted = posterior(x, weight, target, penalty=penalty, prior_mean=prior_mean, noise_var=4.0)
    reference = ridge_solution(x, weight, target, penalty, prior_mean)

    np.testing.assert_allclose(fitted.mean, reference, atol=1e-9)


def test_posterior_mean_matches_reference_from_normal_equations_directly() -> None:
    """Same check via ``posterior_from_normal_equations``, gram/rhs built by hand."""
    rng = np.random.default_rng(2)
    n, p = 150, 5
    x_dense = rng.normal(size=(n, p))
    weight = rng.uniform(0.5, 4.0, n)
    theta = rng.normal(size=p)
    target = x_dense @ theta + rng.normal(scale=0.3, size=n)
    penalty = np.array([1e-8, 2.0, 2.0, 40.0, 40.0])
    prior_mean = np.array([0.0, 1.5, -1.5, 3.0, -3.0])

    gram = x_dense.T @ (weight[:, None] * x_dense)
    rhs = x_dense.T @ (weight * target)
    fitted = posterior_from_normal_equations(gram, rhs, penalty, prior_mean, noise_var=1.0)
    reference = ridge_solution(x_dense, weight, target, penalty, prior_mean)

    np.testing.assert_allclose(fitted.mean, reference, atol=1e-9)


# --- H4 Done-when: 90% interval coverage over 200 synthetic seasons ---------------------------

SEED = 20261001  # the project's declared bootstrap seed (H-d); reused here for one fixed choice
N_PLAYERS = 60  # each stint uses 10 of 60: ~1/6 overlap between any two players' stints
TAU = 6.0  # prior sd of a true rating
SIGMA_TRUE = 9.0  # true observation noise sd (points per 100, before the 1/sqrt(weight) scale)
PRIOR_MEAN = 0.3  # a single nonzero prior mean shared by every rating column
N_SEASONS = 200
STINTS_PER_SEASON = 20_000  # keeps sigma^2's own season-to-season sampling noise small
N_RATING_COLUMNS = 2 * N_PLAYERS  # O and D block per player
PENALTY = np.full(N_RATING_COLUMNS, SIGMA_TRUE**2 / TAU**2)  # lambda_j = sigma^2 / tau^2
PRIOR_MEAN_VECTOR = np.full(N_RATING_COLUMNS, PRIOR_MEAN)


def _synthetic_season_design(
    rng: np.random.Generator, n_stints: int
) -> tuple[sparse.csr_matrix, FloatArray]:
    """A stint-side design: for each stint, a random offense five (+1) and defense five (-1)
    drawn without overlap from the player pool, plus a possession-like row weight."""
    keys = rng.random((n_stints, N_PLAYERS))
    lineup = np.argsort(keys, axis=1)[:, :10]  # 10 distinct players per stint
    offense, defense = lineup[:, :5], lineup[:, 5:]
    stint = np.repeat(np.arange(n_stints), 5)
    rows = np.concatenate([stint, stint])
    cols = np.concatenate([offense.ravel(), N_PLAYERS + defense.ravel()])
    data = np.concatenate([np.ones(n_stints * 5), -np.ones(n_stints * 5)])
    x = sparse.csr_matrix((data, (rows, cols)), shape=(n_stints, N_RATING_COLUMNS))
    weight = rng.uniform(30.0, 150.0, n_stints)
    return x, weight


def _fit_one_season(rng: np.random.Generator) -> tuple[FloatArray, Posterior]:
    """One synthetic season: true ratings, a fit, sigma^2 estimated from its own residuals."""
    theta_true = rng.normal(PRIOR_MEAN, TAU, size=N_RATING_COLUMNS)
    x, weight = _synthetic_season_design(rng, STINTS_PER_SEASON)
    noise = rng.normal(0.0, SIGMA_TRUE / np.sqrt(weight))
    target = x @ theta_true + noise

    # A first fit (an arbitrary placeholder noise_var: it does not affect the mean) to get
    # residuals, exactly as the orchestrator would on a tuning fit.
    preliminary = posterior(
        x, weight, target, penalty=PENALTY, prior_mean=PRIOR_MEAN_VECTOR, noise_var=1.0
    )
    residual = target - x @ preliminary.mean
    sigma2_hat = noise_variance(residual, weight, dof=N_RATING_COLUMNS)
    fitted = posterior(
        x, weight, target, penalty=PENALTY, prior_mean=PRIOR_MEAN_VECTOR, noise_var=sigma2_hat
    )
    return theta_true, fitted


def test_90pct_intervals_cover_the_truth_87_to_93_pct_of_the_time_over_200_seasons() -> None:
    rng = np.random.default_rng(SEED)
    covered = 0
    total = 0
    for _season in range(N_SEASONS):
        theta_true, fitted = _fit_one_season(rng)
        lo, hi = fitted.interval(0.9)
        covered += int(np.sum((theta_true >= lo) & (theta_true <= hi)))
        total += N_RATING_COLUMNS
    coverage = covered / total
    assert 0.87 <= coverage <= 0.93, f"coverage {coverage:.4f} outside [0.87, 0.93]"


# --- Other Done-when checks --------------------------------------------------------------------


def test_sd_shrinks_as_the_number_of_observations_grows() -> None:
    rng = np.random.default_rng(3)
    n, p = 60, 4
    x_dense = rng.normal(size=(n, p))
    x = sparse.csr_matrix(x_dense)
    weight = rng.uniform(1.0, 3.0, n)
    target = rng.normal(size=n)
    penalty = np.full(p, 2.0)
    prior_mean = np.zeros(p)

    small = posterior(x, weight, target, penalty=penalty, prior_mean=prior_mean, noise_var=1.0)
    # Doubling the rows (same distribution of evidence, twice as much of it) can only add
    # information: the posterior sd of every column must shrink.
    x_doubled = sparse.vstack([x, x]).tocsr()
    weight_doubled = np.concatenate([weight, weight])
    target_doubled = np.concatenate([target, target])
    grown = posterior(
        x_doubled,
        weight_doubled,
        target_doubled,
        penalty=penalty,
        prior_mean=prior_mean,
        noise_var=1.0,
    )

    assert np.all(grown.sd < small.sd)


def test_a_flat_columns_sd_matches_the_ols_standard_error_on_a_hand_example() -> None:
    """One column, weighted regression through the origin, penalty ~ 0 (flat prior).

    Weighted OLS: Var(beta_hat) = sigma^2 / sum(w_i x_i^2). x = [1, 2, 3, 4], w = [2, 1, 3, 1]:
    sum(w x^2) = 2*1 + 1*4 + 3*9 + 1*16 = 49. sigma^2 = 5 -> SE = sqrt(5 / 49).
    """
    x = np.array([[1.0], [2.0], [3.0], [4.0]])
    weight = np.array([2.0, 1.0, 3.0, 1.0])
    target = np.zeros(4)  # sd does not depend on the target, only on x, weight, penalty
    penalty = np.array([1e-8])
    prior_mean = np.array([0.0])
    noise_var = 5.0

    fitted = posterior(
        sparse.csr_matrix(x),
        weight,
        target,
        penalty=penalty,
        prior_mean=prior_mean,
        noise_var=noise_var,
    )

    expected_se = np.sqrt(noise_var / 49.0)
    assert fitted.sd[0] == pytest.approx(expected_se, rel=1e-6)


def test_dense_and_sparse_gram_inputs_give_the_same_result() -> None:
    rng = np.random.default_rng(4)
    p = 8
    a = rng.normal(size=(p, p))
    gram_dense = a.T @ a + np.eye(p)  # symmetric positive definite
    rhs = rng.normal(size=p)
    penalty = np.full(p, 1.5)
    prior_mean = rng.normal(size=p)

    from_dense = posterior_from_normal_equations(gram_dense, rhs, penalty, prior_mean, 2.0)
    from_sparse = posterior_from_normal_equations(
        sparse.csr_matrix(gram_dense), rhs, penalty, prior_mean, 2.0
    )

    np.testing.assert_allclose(from_dense.mean, from_sparse.mean, atol=1e-10)
    np.testing.assert_allclose(from_dense.sd, from_sparse.sd, atol=1e-10)


def test_more_than_6000_params_raises() -> None:
    p = MAX_DENSE_PARAMS + 1
    gram = sparse.eye(p, format="csr")
    rhs = np.zeros(p)
    penalty = np.ones(p)
    prior_mean = np.zeros(p)

    with pytest.raises(ValueError, match="exceeds the dense limit"):
        posterior_from_normal_equations(gram, rhs, penalty, prior_mean, 1.0)


# --- interval() and noise_variance() ------------------------------------------------------------


def test_interval_is_symmetric_around_the_mean_and_widens_with_level() -> None:
    fitted = Posterior(mean=np.array([1.0, -2.0]), sd=np.array([0.5, 2.0]))
    lo90, hi90 = fitted.interval(0.9)
    lo50, hi50 = fitted.interval(0.5)

    np.testing.assert_allclose(fitted.mean - lo90, hi90 - fitted.mean)
    assert np.all(hi90 - lo90 > hi50 - lo50)


def test_noise_variance_matches_the_textbook_formula() -> None:
    residual = np.array([1.0, -2.0, 3.0, -1.0])
    weight = np.array([1.0, 2.0, 1.0, 4.0])
    # sum(w r^2) = 1*1 + 2*4 + 1*9 + 4*1 = 22; dof = 1 -> n - dof = 3
    assert noise_variance(residual, weight, dof=1.0) == pytest.approx(22.0 / 3.0)


def test_noise_variance_raises_when_dof_leaves_no_degrees_of_freedom() -> None:
    residual = np.array([1.0, -1.0])
    weight = np.array([1.0, 1.0])
    with pytest.raises(ValueError, match="degrees of freedom"):
        noise_variance(residual, weight, dof=2.0)
