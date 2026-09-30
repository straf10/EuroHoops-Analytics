"""M4 I7 tests: per-stat GBL↔EL translation (``models/translation.py``)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eurohoops.models.box_impact import STAT_COLUMNS
from eurohoops.models.translation import (
    StatFit,
    fit_translation,
    fits_report,
    to_el,
    to_gbl,
)

STAT = "pts"
N_SIM = 200
SIM_BASE_SEED = 20261015
N_PERSONS = 150
DELTA_TRUE = -0.12
BETA_TRUE = 0.08
TAU_TRUE = 0.10
SIGMA_TRUE = 0.25
C_TRUE = 8.0


def _pair_row(
    *,
    person_id: str,
    pair_type: str,
    later_season: int,
    minutes_gbl: float,
    minutes_el: float,
    rate_gbl: float,
    rate_el: float,
    team_gap: float = 0.0,
    season_gbl: int | None = None,
    season_el: int | None = None,
) -> dict[str, object]:
    if season_gbl is None:
        season_gbl = later_season if pair_type != "el_to_gbl" else later_season - 1
    if season_el is None:
        season_el = later_season if pair_type != "gbl_to_el" else later_season - 1
    row: dict[str, object] = {
        "person_id": person_id,
        "pair_type": pair_type,
        "season_gbl": season_gbl,
        "season_el": season_el,
        "later_season": later_season,
        "minutes_gbl": minutes_gbl,
        "minutes_el": minutes_el,
        "team_gbl": "T1",
        "team_el": "T2",
        "team_gap": team_gap,
    }
    for k in STAT_COLUMNS:
        row[f"rate_gbl_{k}"] = rate_gbl if k == STAT else 1.0
        row[f"rate_el_{k}"] = rate_el if k == STAT else 1.0
    return row


def _simulate_dataset(rng: np.random.Generator) -> pd.DataFrame:
    """Draw pairs from the I-j generative model with known δ, β, τ, σ (normalised weights)."""
    meta: list[tuple[str, str, float, float, float, float, int]] = []
    harms: list[float] = []
    for p in range(N_PERSONS):
        n_pairs = int(rng.integers(1, 4))
        for _ in range(n_pairs):
            is_dual = bool(rng.random() < 0.35)
            pair_type = "dual" if is_dual else ("gbl_to_el" if rng.random() < 0.5 else "el_to_gbl")
            minutes_gbl = float(rng.uniform(200.0, 800.0))
            minutes_el = float(rng.uniform(200.0, 800.0))
            harm = 2.0 * minutes_gbl * minutes_el / (minutes_gbl + minutes_el)
            harms.append(harm)
            meta.append(
                (
                    f"P{p:04d}",
                    pair_type,
                    minutes_gbl,
                    minutes_el,
                    float(rng.normal(0.0, 3.0)),
                    harm,
                    2018 + int(rng.integers(0, 4)),
                )
            )
    mean_harm = float(np.mean(harms))
    u_by_person = {f"P{p:04d}": float(rng.normal(0.0, TAU_TRUE)) for p in range(N_PERSONS)}
    rows: list[dict[str, object]] = []
    for person_id, pair_type, minutes_gbl, minutes_el, team_gap, harm, later in meta:
        dual = 1.0 if pair_type == "dual" else 0.0
        w = harm / mean_harm
        e = float(rng.normal(0.0, SIGMA_TRUE / np.sqrt(w)))
        y = DELTA_TRUE + BETA_TRUE * dual + u_by_person[person_id] + e
        rate_gbl = float(rng.gamma(shape=2.0, scale=10.0))
        rate_el = max((rate_gbl + C_TRUE) * np.exp(y) - C_TRUE, 0.0)
        rows.append(
            _pair_row(
                person_id=person_id,
                pair_type=pair_type,
                later_season=later,
                minutes_gbl=minutes_gbl,
                minutes_el=minutes_el,
                rate_gbl=rate_gbl,
                rate_el=rate_el,
                team_gap=team_gap,
            )
        )
    return pd.DataFrame(rows)


def test_delta_interval_coverage_on_synthetic_datasets() -> None:
    """200 seeded datasets: 90% intervals for δ cover the truth in [0.87, 0.93] (one stat)."""
    covers = 0
    for i in range(N_SIM):
        rng = np.random.default_rng(SIM_BASE_SEED + i)
        pairs = _simulate_dataset(rng)
        # Re-scale generative noise: the fit uses w normalised to mean 1, so regenerate y
        # through the same rates with a noise scale that matches after normalisation would
        # be exact; here we fit as-is and check empirical coverage of the declared truth.
        fit = fit_translation(pairs, before=2030, team=False, stats=(STAT,))[STAT]
        if fit.delta_lo90 <= DELTA_TRUE <= fit.delta_hi90:
            covers += 1
    fraction = covers / N_SIM
    assert 0.87 <= fraction <= 0.93, f"coverage {fraction:.4f} outside [0.87, 0.93]"


def test_force_tau0_matches_weighted_least_squares() -> None:
    rng = np.random.default_rng(42)
    pairs = _simulate_dataset(rng)
    fit = fit_translation(pairs, before=2030, team=False, stats=(STAT,), force_tau0=True)[STAT]

    used = pairs
    rate_gbl = used[f"rate_gbl_{STAT}"].to_numpy(dtype=np.float64)
    rate_el = used[f"rate_el_{STAT}"].to_numpy(dtype=np.float64)
    c = fit.c
    y = np.log((rate_el + c) / (rate_gbl + c))
    minutes_gbl = used["minutes_gbl"].to_numpy(dtype=np.float64)
    minutes_el = used["minutes_el"].to_numpy(dtype=np.float64)
    harm = 2.0 * minutes_gbl * minutes_el / (minutes_gbl + minutes_el)
    w = harm / float(np.mean(harm))
    dual = (used["pair_type"].to_numpy() == "dual").astype(np.float64)
    x = np.column_stack([np.ones(len(used)), dual])
    sqrt_w = np.sqrt(w)
    beta_wls, _, _, _ = np.linalg.lstsq(x * sqrt_w[:, None], y * sqrt_w, rcond=None)

    assert fit.tau2 == 0.0
    assert fit.delta == pytest.approx(float(beta_wls[0]), abs=1e-9)
    assert fit.beta_dual == pytest.approx(float(beta_wls[1]), abs=1e-9)


def test_zero_rates_finite() -> None:
    rows = [
        _pair_row(
            person_id="A",
            pair_type="dual",
            later_season=2020,
            minutes_gbl=400.0,
            minutes_el=400.0,
            rate_gbl=0.0,
            rate_el=0.0,
        ),
        _pair_row(
            person_id="B",
            pair_type="gbl_to_el",
            later_season=2020,
            minutes_gbl=300.0,
            minutes_el=350.0,
            rate_gbl=0.0,
            rate_el=5.0,
        ),
        _pair_row(
            person_id="C",
            pair_type="el_to_gbl",
            later_season=2021,
            minutes_gbl=250.0,
            minutes_el=280.0,
            rate_gbl=4.0,
            rate_el=0.0,
        ),
    ]
    fits = fit_translation(pd.DataFrame(rows), before=2030, stats=(STAT,))
    fit = fits[STAT]
    assert np.isfinite(fit.delta)
    assert np.isfinite(fit.delta_lo90)
    assert np.isfinite(fit.delta_hi90)
    assert fit.c > 0.0


def test_single_person_forces_tau0() -> None:
    rows = [
        _pair_row(
            person_id="ONLY",
            pair_type="dual",
            later_season=2019,
            minutes_gbl=500.0,
            minutes_el=480.0,
            rate_gbl=20.0,
            rate_el=18.0,
        ),
        _pair_row(
            person_id="ONLY",
            pair_type="gbl_to_el",
            later_season=2020,
            minutes_gbl=400.0,
            minutes_el=420.0,
            rate_gbl=22.0,
            rate_el=19.0,
        ),
    ]
    fit = fit_translation(pd.DataFrame(rows), before=2030, stats=(STAT,))[STAT]
    assert fit.n_persons == 1
    assert fit.tau2 == 0.0


def test_no_dual_drops_beta() -> None:
    rows = [
        _pair_row(
            person_id="A",
            pair_type="gbl_to_el",
            later_season=2019,
            minutes_gbl=500.0,
            minutes_el=480.0,
            rate_gbl=20.0,
            rate_el=16.0,
        ),
        _pair_row(
            person_id="B",
            pair_type="el_to_gbl",
            later_season=2020,
            minutes_gbl=400.0,
            minutes_el=420.0,
            rate_gbl=18.0,
            rate_el=15.0,
        ),
        _pair_row(
            person_id="C",
            pair_type="gbl_to_el",
            later_season=2021,
            minutes_gbl=450.0,
            minutes_el=460.0,
            rate_gbl=25.0,
            rate_el=20.0,
        ),
    ]
    fit = fit_translation(pd.DataFrame(rows), before=2030, stats=(STAT,))[STAT]
    assert fit.beta_dual == 0.0


def test_constant_team_gap_drops_gamma() -> None:
    rows = [
        _pair_row(
            person_id="A",
            pair_type="dual",
            later_season=2019,
            minutes_gbl=500.0,
            minutes_el=480.0,
            rate_gbl=20.0,
            rate_el=16.0,
            team_gap=2.5,
        ),
        _pair_row(
            person_id="B",
            pair_type="gbl_to_el",
            later_season=2020,
            minutes_gbl=400.0,
            minutes_el=420.0,
            rate_gbl=18.0,
            rate_el=15.0,
            team_gap=2.5,
        ),
        _pair_row(
            person_id="C",
            pair_type="el_to_gbl",
            later_season=2021,
            minutes_gbl=450.0,
            minutes_el=460.0,
            rate_gbl=25.0,
            rate_el=20.0,
            team_gap=2.5,
        ),
    ]
    fit = fit_translation(pd.DataFrame(rows), before=2030, team=True, stats=(STAT,))[STAT]
    assert fit.gamma_team == 0.0


def test_walk_forward_ignores_later_pairs() -> None:
    base = [
        _pair_row(
            person_id="A",
            pair_type="dual",
            later_season=2019,
            minutes_gbl=500.0,
            minutes_el=480.0,
            rate_gbl=20.0,
            rate_el=16.0,
        ),
        _pair_row(
            person_id="B",
            pair_type="gbl_to_el",
            later_season=2020,
            minutes_gbl=400.0,
            minutes_el=420.0,
            rate_gbl=18.0,
            rate_el=15.0,
        ),
        _pair_row(
            person_id="C",
            pair_type="el_to_gbl",
            later_season=2021,
            minutes_gbl=450.0,
            minutes_el=460.0,
            rate_gbl=25.0,
            rate_el=20.0,
        ),
        _pair_row(
            person_id="D",
            pair_type="dual",
            later_season=2023,
            minutes_gbl=600.0,
            minutes_el=610.0,
            rate_gbl=30.0,
            rate_el=10.0,
        ),
    ]
    pairs = pd.DataFrame(base)
    fits_full = fit_translation(pairs, before=2022, stats=(STAT,))

    edited = pairs.copy()
    mask = edited["later_season"] >= 2022
    edited.loc[mask, f"rate_el_{STAT}"] = 999.0
    edited.loc[mask, f"rate_gbl_{STAT}"] = 0.01
    fits_edited = fit_translation(edited, before=2022, stats=(STAT,))

    deleted = pairs.loc[pairs["later_season"] < 2022].copy()
    fits_deleted = fit_translation(deleted, before=2022, stats=(STAT,))

    assert fits_full[STAT] == fits_edited[STAT] == fits_deleted[STAT]


def test_empty_before_raises() -> None:
    rows = [
        _pair_row(
            person_id="A",
            pair_type="dual",
            later_season=2024,
            minutes_gbl=500.0,
            minutes_el=480.0,
            rate_gbl=20.0,
            rate_el=16.0,
        )
    ]
    with pytest.raises(ValueError, match="later_season"):
        fit_translation(pd.DataFrame(rows), before=2020, stats=(STAT,))


def test_to_el_to_gbl_round_trip() -> None:
    rows = [
        _pair_row(
            person_id="A",
            pair_type="dual",
            later_season=2019,
            minutes_gbl=500.0,
            minutes_el=480.0,
            rate_gbl=20.0,
            rate_el=18.0,
            team_gap=1.0,
        ),
        _pair_row(
            person_id="B",
            pair_type="gbl_to_el",
            later_season=2020,
            minutes_gbl=400.0,
            minutes_el=420.0,
            rate_gbl=22.0,
            rate_el=19.0,
            team_gap=-2.0,
        ),
        _pair_row(
            person_id="C",
            pair_type="el_to_gbl",
            later_season=2021,
            minutes_gbl=450.0,
            minutes_el=460.0,
            rate_gbl=25.0,
            rate_el=21.0,
            team_gap=0.5,
        ),
    ]
    fits = fit_translation(pd.DataFrame(rows), before=2030, team=True)
    gbl = pd.DataFrame(
        {f"rate_gbl_{k}": [15.0, 20.0, 8.0] for k in STAT_COLUMNS},
        index=["p1", "p2", "p3"],
    )
    gap = pd.Series([1.0, -1.5, 0.0], index=gbl.index)
    el = to_el(gbl, fits, gap)
    back = to_gbl(el, fits, gap)
    for k in STAT_COLUMNS:
        # No flooring when rates are comfortably above 0 and the factor is moderate.
        np.testing.assert_allclose(
            back[f"rate_gbl_{k}"].to_numpy(),
            gbl[f"rate_gbl_{k}"].to_numpy(),
            atol=1e-9,
        )


def test_prediction_floors_at_zero() -> None:
    fit = StatFit(
        stat=STAT,
        c=5.0,
        delta=-3.0,
        delta_lo90=-3.5,
        delta_hi90=-2.5,
        beta_dual=0.0,
        gamma_team=0.0,
        tau2=0.0,
        sigma2=1.0,
        n_pairs=3,
        n_persons=2,
    )
    gbl = pd.DataFrame({f"rate_gbl_{STAT}": [0.1]}, index=["p"])
    el = to_el(gbl, {STAT: fit})
    assert float(el[f"rate_el_{STAT}"].iloc[0]) == 0.0


def test_fits_report_rounded_and_deterministic() -> None:
    rows = [
        _pair_row(
            person_id="A",
            pair_type="dual",
            later_season=2019,
            minutes_gbl=500.0,
            minutes_el=480.0,
            rate_gbl=20.0,
            rate_el=16.0,
        ),
        _pair_row(
            person_id="B",
            pair_type="gbl_to_el",
            later_season=2020,
            minutes_gbl=400.0,
            minutes_el=420.0,
            rate_gbl=18.0,
            rate_el=15.0,
        ),
        _pair_row(
            person_id="C",
            pair_type="el_to_gbl",
            later_season=2021,
            minutes_gbl=450.0,
            minutes_el=460.0,
            rate_gbl=25.0,
            rate_el=20.0,
        ),
    ]
    pairs = pd.DataFrame(rows)
    a = fits_report(fit_translation(pairs, before=2030, stats=(STAT,)))
    b = fits_report(fit_translation(pairs, before=2030, stats=(STAT,)))
    assert a == b
    assert isinstance(a[STAT]["delta"], float)
    assert a[STAT]["n_pairs"] == 3


def test_mover_prediction_does_not_apply_beta() -> None:
    """A dual-only beta shift must not enter to_el (movers are not duals)."""
    # Build a hand StatFit with known delta=0, beta=1 so applying beta would change the prediction.
    fit = StatFit(
        stat=STAT,
        c=1.0,
        delta=0.0,
        delta_lo90=-0.1,
        delta_hi90=0.1,
        beta_dual=1.0,
        gamma_team=0.0,
        tau2=0.0,
        sigma2=1.0,
        n_pairs=4,
        n_persons=4,
    )
    gbl = pd.DataFrame({f"rate_gbl_{STAT}": [10.0]})
    el = to_el(gbl, {STAT: fit})
    assert float(el[f"rate_el_{STAT}"].iloc[0]) == pytest.approx(10.0, abs=1e-12)
