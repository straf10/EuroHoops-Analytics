"""M4 I7: per-stat GBL <-> EuroLeague translation (PLAN section 5.5, I-j).

For each box rate k (``STAT_COLUMNS``), pairs of the same person across leagues give a
log-ratio of per-100 rates. Duals (same season, both leagues) and movers in both directions
enter the same regression; for an EL target season ``t`` only pairs whose later season is
strictly before ``t`` are used (walk-forward).

Model, one stat k, fitted independently:

    y_i = log((r_EL,i + c_k) / (r_GBL,i + c_k))
    y_i = delta_k + beta_k * dual_i + gamma_k * team_gap_i + u_{person(i)} + e_i

    u_p ~ N(0, tau_k^2),   e_i ~ N(0, sigma_k^2 / w_i)

``c_k`` is half the median of all rate values of k on both sides among the fit's pairs
(half the mean if that median is 0; 0.5 if the mean is also 0). ``w_i`` is the harmonic mean
of the pair's two minutes, renormalised to mean 1 over the fit. ``gamma_k`` is present only in
the ``team=True`` variant (``translate_team``); movers are not duals, so prediction applies
delta and gamma but never beta.

``tau_k^2`` and ``sigma_k^2`` maximise the REML (restricted) marginal likelihood over
log-variances (``scipy.optimize``); fixed effects are GLS given those variances; delta's 90%
interval is the normal quantile 1.64485 times the GLS standard error. Degenerate designs force
``tau^2 = 0`` (fewer than two persons, or fewer pairs than fixed effects + 1), drop beta when
no dual exists, and drop gamma when ``team_gap`` has no variance -- reported as 0.0 rather than
a singular fit.

``to_el`` / ``to_gbl`` invert the mean structure for a mover (no dual term), flooring at 0.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import optimize, special

from eurohoops.models.box_impact import STAT_COLUMNS
from eurohoops.models.elo import FloatArray

IntArray = npt.NDArray[np.int64]
Z90 = float(special.ndtri(0.5 + 0.9 / 2.0))  # ~1.64485
_LOG_VAR_FLOOR = -20.0  # tau2, sigma2 >= exp(floor) ~ 2e-9 during REML search
_START_LOG_TAU2 = -2.0
_START_LOG_SIGMA2 = -1.0


@dataclass(frozen=True)
class StatFit:
    """Closed-form GLS + REML fit for one box rate's GBL<->EL log-ratio."""

    stat: str
    c: float
    delta: float
    delta_lo90: float
    delta_hi90: float
    beta_dual: float
    gamma_team: float
    tau2: float
    sigma2: float
    n_pairs: int
    n_persons: int


def fit_translation(
    pairs: pd.DataFrame,
    *,
    before: int,
    team: bool = False,
    stats: tuple[str, ...] = STAT_COLUMNS,
    force_tau0: bool = False,
) -> dict[str, StatFit]:
    """Fit per-stat translation factors on pairs with ``later_season < before``.

    ``force_tau0`` is for tests only: skip the REML search and set ``tau2 = 0`` so the fixed
    effects match weighted least squares exactly.
    """
    used = pairs.loc[pairs["later_season"].astype(int) < int(before)].copy()
    if used.empty:
        raise ValueError(f"no translation pairs with later_season < {before}")
    return {stat: _fit_stat(used, stat, team=team, force_tau0=force_tau0) for stat in stats}


def to_el(
    rates_gbl: pd.DataFrame,
    fits: dict[str, StatFit],
    team_gap: pd.Series | None = None,
) -> pd.DataFrame:
    """Translate GBL per-100 rates to EL: ``(r + c) * exp(delta + gamma*gap) - c``, floored at 0."""
    return _translate(rates_gbl, fits, team_gap, direction="to_el")


def to_gbl(
    rates_el: pd.DataFrame,
    fits: dict[str, StatFit],
    team_gap: pd.Series | None = None,
) -> pd.DataFrame:
    """Translate EL rates to GBL: ``(r + c) * exp(-(delta + gamma*gap)) - c``, floored at 0."""
    return _translate(rates_el, fits, team_gap, direction="to_gbl")


def fits_report(fits: dict[str, StatFit]) -> dict[str, dict[str, float | int]]:
    """JSON-ready per-stat fit summary; floats rounded to 6 decimals."""
    out: dict[str, dict[str, float | int]] = {}
    for stat, fit in fits.items():
        row: dict[str, float | int] = {}
        for key, value in asdict(fit).items():
            if key == "stat":
                continue
            if isinstance(value, float):
                row[key] = round(value, 6)
            else:
                row[key] = int(value)
        out[stat] = row
    return out


def _translate(
    rates: pd.DataFrame,
    fits: dict[str, StatFit],
    team_gap: pd.Series | None,
    *,
    direction: str,
) -> pd.DataFrame:
    if team_gap is None:
        gap = np.zeros(len(rates), dtype=np.float64)
    else:
        gap = team_gap.reindex(rates.index).to_numpy(dtype=np.float64)
        if np.any(np.isnan(gap)):
            raise ValueError("team_gap has missing values for some rate rows")
    src_prefix = "rate_gbl_" if direction == "to_el" else "rate_el_"
    dst_prefix = "rate_el_" if direction == "to_el" else "rate_gbl_"
    out = pd.DataFrame(index=rates.index)
    for stat, fit in fits.items():
        src = rates[f"{src_prefix}{stat}"].to_numpy(dtype=np.float64)
        linear = fit.delta + fit.gamma_team * gap
        if direction == "to_el":
            pred = (src + fit.c) * np.exp(linear) - fit.c
        else:
            pred = (src + fit.c) * np.exp(-linear) - fit.c
        out[f"{dst_prefix}{stat}"] = np.maximum(pred, 0.0)
    return out


def _fit_stat(
    pairs: pd.DataFrame,
    stat: str,
    *,
    team: bool,
    force_tau0: bool,
) -> StatFit:
    rate_gbl = pairs[f"rate_gbl_{stat}"].to_numpy(dtype=np.float64)
    rate_el = pairs[f"rate_el_{stat}"].to_numpy(dtype=np.float64)
    c = _pseudocount(np.concatenate([rate_gbl, rate_el]))
    y = np.log((rate_el + c) / (rate_gbl + c))

    minutes_gbl = pairs["minutes_gbl"].to_numpy(dtype=np.float64)
    minutes_el = pairs["minutes_el"].to_numpy(dtype=np.float64)
    harm = _harmonic_mean(minutes_gbl, minutes_el)
    w = harm / float(np.mean(harm))

    dual = (pairs["pair_type"].to_numpy() == "dual").astype(np.float64)
    team_gap = pairs["team_gap"].to_numpy(dtype=np.float64)
    person_codes, uniques = pd.factorize(pairs["person_id"].astype(str), sort=True)
    person = np.asarray(person_codes, dtype=np.int64)
    n_persons = len(uniques)

    include_beta = bool(np.any(dual > 0.0))
    include_gamma = bool(team and float(np.std(team_gap)) > 0.0)
    x = _design(dual, team_gap, include_beta=include_beta, include_gamma=include_gamma)
    n_pairs, n_fe = x.shape

    use_tau0 = force_tau0 or n_persons < 2 or n_pairs < n_fe + 1
    if use_tau0:
        tau2, sigma2 = 0.0, _sigma2_wls(x, y, w, n_fe)
        beta_hat, cov = _gls(x, y, w, person, n_persons, tau2=0.0, sigma2=sigma2)
    else:
        tau2, sigma2 = _reml_variances(x, y, w, person, n_persons)
        beta_hat, cov = _gls(x, y, w, person, n_persons, tau2=tau2, sigma2=sigma2)

    delta = float(beta_hat[0])
    se_delta = float(np.sqrt(max(cov[0, 0], 0.0)))
    beta_dual = float(beta_hat[1]) if include_beta else 0.0
    if include_gamma:
        gamma_idx = 1 + int(include_beta)
        gamma_team = float(beta_hat[gamma_idx])
    else:
        gamma_team = 0.0

    return StatFit(
        stat=stat,
        c=float(c),
        delta=delta,
        delta_lo90=delta - Z90 * se_delta,
        delta_hi90=delta + Z90 * se_delta,
        beta_dual=beta_dual,
        gamma_team=gamma_team,
        tau2=float(tau2),
        sigma2=float(sigma2),
        n_pairs=int(n_pairs),
        n_persons=int(n_persons),
    )


def _pseudocount(rates: FloatArray) -> float:
    med = float(np.median(rates))
    if med > 0.0:
        return 0.5 * med
    mean = float(np.mean(rates))
    if mean > 0.0:
        return 0.5 * mean
    return 0.5


def _harmonic_mean(a: FloatArray, b: FloatArray) -> FloatArray:
    return (2.0 * a * b) / (a + b)


def _design(
    dual: FloatArray,
    team_gap: FloatArray,
    *,
    include_beta: bool,
    include_gamma: bool,
) -> FloatArray:
    cols: list[FloatArray] = [np.ones(dual.shape[0], dtype=np.float64)]
    if include_beta:
        cols.append(dual)
    if include_gamma:
        cols.append(team_gap)
    return np.column_stack(cols)


def _person_weight_sums(w: FloatArray, person: IntArray, n_persons: int) -> FloatArray:
    sums = np.zeros(n_persons, dtype=np.float64)
    np.add.at(sums, person, w)
    return sums


def _apply_g_inv(
    v: FloatArray,
    w: FloatArray,
    person: IntArray,
    person_w: FloatArray,
    lam: float,
) -> FloatArray:
    """Apply ``G^{-1}`` with ``G = W^{-1} + lam ZZ'`` to a vector (or each column of a matrix)."""
    if v.ndim == 1:
        if lam <= 0.0:
            return w * v
        weighted = w * v
        person_sum = np.zeros(person_w.shape[0], dtype=np.float64)
        np.add.at(person_sum, person, weighted)
        factor = lam / (1.0 + lam * person_w)
        return weighted - w * (factor[person] * person_sum[person])
    out = np.empty_like(v, dtype=np.float64)
    for j in range(v.shape[1]):
        out[:, j] = _apply_g_inv(v[:, j], w, person, person_w, lam)
    return out


def _log_det_g(w: FloatArray, person_w: FloatArray, lam: float) -> float:
    base = float(-np.sum(np.log(w)))
    if lam <= 0.0:
        return base
    return base + float(np.sum(np.log1p(lam * person_w)))


def _gls(
    x: FloatArray,
    y: FloatArray,
    w: FloatArray,
    person: IntArray,
    n_persons: int,
    *,
    tau2: float,
    sigma2: float,
) -> tuple[FloatArray, FloatArray]:
    """GLS beta and Cov = ``(X' V^{-1} X)^{-1}`` for ``V = tau2 ZZ' + sigma2 W^{-1}``."""
    person_w = _person_weight_sums(w, person, n_persons)
    lam = 0.0 if sigma2 <= 0.0 else tau2 / sigma2
    g_inv_x = _apply_g_inv(x, w, person, person_w, lam)
    g_inv_y = _apply_g_inv(y, w, person, person_w, lam)
    xtgx = x.T @ g_inv_x
    xtgy = x.T @ g_inv_y
    xtgx = 0.5 * (xtgx + xtgx.T)
    beta = np.linalg.solve(xtgx, xtgy)
    # Cov(beta) = (X'V^{-1}X)^{-1} = sigma2 (X'G^{-1}X)^{-1}; sigma2 cancels in beta itself.
    cov = float(max(sigma2, 0.0)) * np.linalg.inv(xtgx)
    return beta, cov


def _sigma2_wls(x: FloatArray, y: FloatArray, w: FloatArray, n_fe: int) -> float:
    sqrt_w = np.sqrt(w)
    beta, _, _, _ = np.linalg.lstsq(x * sqrt_w[:, None], y * sqrt_w, rcond=None)
    resid = y - x @ beta
    denom = max(len(y) - n_fe, 1)
    return float(max(np.sum(w * resid**2) / denom, np.exp(_LOG_VAR_FLOOR)))


def _reml_objective(  # noqa: PLR0917 -- scipy.optimize.minimize passes args positionally
    log_vars: FloatArray,
    x: FloatArray,
    y: FloatArray,
    w: FloatArray,
    person: IntArray,
    person_w: FloatArray,
) -> float:
    """Negative REML log-likelihood (up to an additive constant)."""
    log_tau2 = float(np.clip(log_vars[0], _LOG_VAR_FLOOR, 20.0))
    log_sigma2 = float(np.clip(log_vars[1], _LOG_VAR_FLOOR, 20.0))
    tau2 = float(np.exp(log_tau2))
    sigma2 = float(np.exp(log_sigma2))
    n, p = x.shape
    lam = tau2 / sigma2
    g_inv_x = _apply_g_inv(x, w, person, person_w, lam)
    g_inv_y = _apply_g_inv(y, w, person, person_w, lam)
    xtgx = x.T @ g_inv_x
    xtgy = x.T @ g_inv_y
    xtgx = 0.5 * (xtgx + xtgx.T)
    try:
        beta = np.linalg.solve(xtgx, xtgy)
    except np.linalg.LinAlgError:
        return 1e300
    resid = y - x @ beta
    g_inv_r = _apply_g_inv(resid, w, person, person_w, lam)
    quad = float(resid @ g_inv_r) / sigma2
    # |V| = sigma2^n |G|; |X'V^{-1}X| = sigma2^{-p} |X'G^{-1}X|
    sign_x, logdet_x = np.linalg.slogdet(xtgx)
    if sign_x <= 0.0:
        return 1e300
    log_det_v = n * log_sigma2 + _log_det_g(w, person_w, lam)
    log_det_xtvx = float(logdet_x) - p * log_sigma2
    return float(0.5 * (log_det_v + log_det_xtvx + quad))


def _reml_variances(
    x: FloatArray,
    y: FloatArray,
    w: FloatArray,
    person: IntArray,
    n_persons: int,
) -> tuple[float, float]:
    person_w = _person_weight_sums(w, person, n_persons)
    start = np.array([_START_LOG_TAU2, _START_LOG_SIGMA2], dtype=np.float64)
    result = optimize.minimize(
        _reml_objective,
        start,
        args=(x, y, w, person, person_w),
        method="Nelder-Mead",
        options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 2_000},
    )
    log_tau2 = float(np.clip(result.x[0], _LOG_VAR_FLOOR, 20.0))
    log_sigma2 = float(np.clip(result.x[1], _LOG_VAR_FLOOR, 20.0))
    return float(np.exp(log_tau2)), float(np.exp(log_sigma2))
