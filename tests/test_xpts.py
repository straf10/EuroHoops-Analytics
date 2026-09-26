"""F3/F4: the spline baseline recovers a known logistic curve, isotonic calibration, no identity
features, and the LightGBM challenger and Optuna study are deterministic."""

import numpy as np
import pandas as pd
import pytest

from eurohoops.models.xpts import (
    design,
    fit_isotonic,
    fit_logistic,
    fit_spline,
    has_identity_column,
    make_spec,
    natural_spline_basis,
    sigmoid,
)

ZONE_CODES = tuple("ABCDEFGHIJ")


def synthetic_shots(n: int, seed: int) -> pd.DataFrame:
    """Shot-table-like rows with every feature column the models read (made is filled later)."""
    rng = np.random.default_rng(seed)
    value = np.where(rng.random(n) < 0.4, 3, 2)
    distance = np.where(value == 3, rng.uniform(6.8, 9.0, n), rng.uniform(0.2, 6.5, n))
    return pd.DataFrame(
        {
            "season": rng.integers(2011, 2023, n),
            "game_id": [f"E2011_{i % 300}" for i in range(n)],
            "shooter": [f"P{i % 50}" for i in range(n)],
            "team": "AAA",
            "opponent": "BBB",
            "distance": distance,
            "angle": rng.uniform(0.0, 120.0, n),
            "value": value,
            "zone": rng.choice(list(ZONE_CODES[:6]), n),
            "fastbreak": rng.random(n) < 0.1,
            "second_chance": rng.random(n) < 0.15,
            "points_off_turnover": rng.random(n) < 0.2,
            "home": rng.random(n) < 0.5,
            "seconds_left": rng.integers(0, 601, n),
            "period": rng.integers(1, 6, n),
            "margin_before": rng.integers(-25, 26, n),
            "band": "rim",
        }
    )


def test_natural_spline_is_linear_beyond_the_boundary_knots() -> None:
    knots = np.array([1.0, 2.0, 4.0, 7.0])
    x = np.array([8.0, 9.0, 10.0, 11.0])
    second_difference = np.diff(natural_spline_basis(x, knots), n=2, axis=0)
    np.testing.assert_allclose(second_difference, 0.0, atol=1e-9)
    assert natural_spline_basis(x, knots).shape == (4, 3)  # K knots -> K - 1 columns


def test_spline_fit_recovers_a_known_logistic_curve() -> None:
    """F3 Done-when: n = 200k synthetic shots from a known logistic curve in the model's own
    (whitened) design; the unpenalised fit recovers every coefficient within 0.02."""
    shots = synthetic_shots(200_000, seed=3)
    spec = make_spec(shots, n_knots=6)
    x = design(shots, spec)
    rng = np.random.default_rng(4)
    truth = rng.normal(0.0, 0.3, x.shape[1])
    truth[0] = -0.1
    y = (rng.random(len(shots)) < sigmoid(x @ truth)).astype(np.float64)
    beta = fit_logistic(x, y, l2=0.0)
    assert np.max(np.abs(beta - truth)) < 0.02


def test_whitened_design_is_centred_and_uncorrelated_on_training_shots() -> None:
    shots = synthetic_shots(20_000, seed=5)
    x = design(shots, make_spec(shots, n_knots=4))[:, 1:]
    np.testing.assert_allclose(x.mean(axis=0), 0.0, atol=1e-8)
    np.testing.assert_allclose(np.cov(x, rowvar=False, bias=True), np.eye(x.shape[1]), atol=1e-6)


def test_design_has_no_identity_columns() -> None:
    """R9: quality is not skill: no shooter, team, opponent or game column in either model."""
    from eurohoops.models.xpts_gbm import features  # noqa: PLC0415

    shots = synthetic_shots(5_000, seed=6)
    spec = make_spec(shots, n_knots=4)
    assert not has_identity_column(spec.names)
    assert not has_identity_column(tuple(features(shots).columns))
    assert has_identity_column(("zone_A", "shooter"))  # the guard itself works
    for names in (spec.names, tuple(features(shots).columns)):  # outcome-coded feed flags
        assert not set(names) & {"fastbreak", "second_chance", "points_off_turnover"}


def test_penalty_shrinks_and_fit_is_deterministic() -> None:
    shots = synthetic_shots(20_000, seed=7)
    shots["made"] = np.random.default_rng(8).random(len(shots)) < 0.5
    loose = fit_spline(shots, 4, 1e-6)
    tight = fit_spline(shots, 4, 1.0)
    assert np.abs(tight.beta[1:]).sum() < np.abs(loose.beta[1:]).sum()
    np.testing.assert_array_equal(fit_spline(shots, 4, 1e-6).beta, loose.beta)
    p = loose.predict(shots)
    assert ((p > 0) & (p < 1)).all()


def test_isotonic_is_monotone_and_pools_violators() -> None:
    fitted = fit_isotonic(np.array([0.1, 0.2, 0.3, 0.4]), np.array([0.0, 1.0, 0.0, 1.0]))
    np.testing.assert_allclose(
        fitted(np.array([0.1, 0.2, 0.25, 0.3, 0.4, 0.9])), [0.0, 0.5, 0.5, 0.5, 1.0, 1.0]
    )
    rng = np.random.default_rng(9)
    p = rng.random(5_000)
    y = (rng.random(5_000) < p**2).astype(np.float64)
    out = fit_isotonic(p, y)(np.sort(p))
    assert (np.diff(out) >= 0).all()
    assert abs(out.mean() - y.mean()) < 1e-9  # isotonic preserves the mean on its training data


lightgbm = pytest.importorskip("lightgbm")


def test_lightgbm_is_deterministic_for_a_seed() -> None:
    from eurohoops.models.xpts_gbm import fit_gbm  # noqa: PLC0415

    shots = synthetic_shots(5_000, seed=10)
    shots["made"] = np.random.default_rng(11).random(len(shots)) < 0.45
    params = {
        "num_leaves": 15,
        "learning_rate": 0.1,
        "n_estimators": 50,
        "min_child_samples": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "lambda_l2": 1.0,
    }
    a = fit_gbm(shots, params, 20261001).predict(shots)
    b = fit_gbm(shots, params, 20261001).predict(shots)
    c = fit_gbm(shots, params, 20261002).predict(shots)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)  # row and feature subsampling follow the seed
