"""Week 14-16 J7: the M5 rest report (OLS with HC0 errors; the report on the synthetic league)."""

import numpy as np

from eurohoops.eval.m5_backtest import run_m5_backtest
from eurohoops.eval.m5_rest import FEATURES, Z90, ols_hc0, rest_report
from tests.m5_synthetic import CLUB_MAP, build_inputs, fake_player_part, small_spec


def test_ols_hc0_recovers_a_planted_effect_and_matches_a_direct_computation() -> None:
    rng = np.random.default_rng(20261020)
    n = 4000
    x = np.column_stack(
        [
            rng.normal(0, 2, n),
            rng.integers(-1, 2, n).astype(float),
            rng.integers(-2, 3, n).astype(float),
            rng.integers(-1, 2, n).astype(float),
        ]
    )
    beta = np.array([0.5, -1.0, 0.0, 2.0])
    y = 1.5 + x @ beta + rng.normal(0, 10, n)
    out = ols_hc0(x, y)
    # direct computation: the same estimator written out independently
    design = np.column_stack([np.ones(n), x])
    coef = np.linalg.lstsq(design, y, rcond=None)[0]
    resid = y - design @ coef
    bread = np.linalg.inv(design.T @ design)
    se = np.sqrt(np.diag(bread @ (design.T * resid**2) @ design @ bread))
    for j, name in enumerate(FEATURES):
        assert abs(out[name]["coef"] - coef[j + 1]) < 1e-6
        assert abs(out[name]["se"] - se[j + 1]) < 1e-6
        low, high = out[name]["ci90"]
        assert abs(high - low - 2 * Z90 * se[j + 1]) < 1e-5
        # the planted value lies within 4 standard errors (noise sd 10, n 4000)
        assert abs(coef[j + 1] - beta[j]) < 4 * se[j + 1]


def test_ols_hc0_reports_a_constant_feature_as_unidentified() -> None:
    rng = np.random.default_rng(1)
    x = np.column_stack([rng.normal(size=50), np.zeros(50), rng.normal(size=50), np.ones(50)])
    out = ols_hc0(x, rng.normal(size=50))
    assert out["short_rest"] == {"coef": None, "se": None, "ci90": None}
    assert out["other_comp_prev"]["coef"] is None
    assert out["days_rest"]["coef"] is not None


def test_rest_report_on_the_synthetic_league() -> None:
    inputs = build_inputs()
    spec = small_spec()
    _, predictions = run_m5_backtest(inputs, spec=spec, player_part=fake_player_part)
    report = rest_report(
        inputs.games,
        inputs.other_games,
        inputs.club_map,
        predictions,
        tuning=spec.tuning,
        greek=tuple(CLUB_MAP.values()),
    )
    by_split = report["residual_on_rest_diff"]["by_split"]
    assert set(by_split) == {"tuning", "validation"}
    m5 = predictions[predictions["model"] == "m5"]
    assert by_split["tuning"]["n"] == int((m5["split"] == "tuning").sum())
    assert set(report["residual_on_rest_diff"]["by_tuning_season"]) <= {str(s) for s in spec.tuning}
    dist = report["distribution"]["tuning"]
    assert 0.0 <= dist["short_rest_share"] <= 1.0
    assert sum(report["greek_within_3_days_of_other_competition"].values()) > 0
