"""M5 model core (weeks 14-16): a player-based margin plus a team residual and a rest effect.

M5's margin for game g is

    margin[g] = player_part[g] + adjustment[g]
    adjustment[g] = h · home_flag[g] + r[home] - r[away] + β · rest_diff[g]

``player_part`` is H-c's ``eval.m3_backtest.rapm_margins`` (it already contains its own home
term). ``h``, the team residuals ``r`` and the rest coefficients ``β`` are fitted walk-forward on
the residual ``actual_margin - player_part`` by a time-decayed ridge (``fit_residual``). With a
residual that is identically 0 every coefficient is 0, so the adjustment is exactly 0 and the core
margin equals ``rapm_margins``: the adjustment can only move the player part where the data ask.

Walk-forward as in M1: a game is predicted from rated games that tipped off strictly before the
first tip-off of its round (season, phase, round). ``rest_diff`` (home minus away rest features) is
known from the schedule, so it is a predictor for any game, but only rows of earlier games enter
the fit.

The rest of the module is the evaluation plumbing: a probit blend of component margins fitted by
season on earlier seasons (``blend_by_season``), Platt recalibration of win probabilities and
over/under and spread-cover probabilities.
"""

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import optimize, special

from eurohoops.models.elo import FloatArray
from eurohoops.models.minutes import _epoch, round_cutoffs
from eurohoops.models.team_eff import DecayedRidge, DecayParams, IntArray, MarginModel

BoolArray = npt.NDArray[np.bool_]
P_CLIP = 1e-9  # keeps logit finite at p = 0 or 1
HOME_PENALTY = 1e-6  # h is almost free (see UNPENALISED in team_eff)


@dataclass(frozen=True)
class ResidualFit:
    """Per game, the coefficients used to forecast it (NaN before any game has tipped off)."""

    adjustment: FloatArray  # h·home_flag + r_home - r_away + β·rest_diff
    rest_coef: FloatArray  # (n_games, k) β; k = 0 columns when there is no rest feature
    home: FloatArray  # h


def fit_residual(
    games: pd.DataFrame,
    target: FloatArray,
    rest_diff: FloatArray | None,
    *,
    half_life_days: float,
    residual_ridge: float,
    rest_ridge: float,
) -> ResidualFit:
    """Walk-forward fit of [h, r_team…, β…] on ``target`` (= actual margin - player part).

    ``games`` is in tip-off order. A row is fitted only if it is played, not a forfeit, and its
    target and rest features are finite (NaN target: never used). Row weight is
    ``0.5 ** (age_days / half_life_days)`` with no season carry, so ``DecayedRidge`` is reused
    with ``carry=1``. Penalties: h ``HOME_PENALTY``, r ``residual_ridge``, β ``rest_ridge``.
    """
    tipoff = _epoch(games["tipoff_utc"])
    if np.any(np.diff(tipoff) < 0):
        raise ValueError("games must be sorted by tipoff_utc")
    n = len(games)
    teams = sorted({*games["home"], *games["away"]})
    index = {team: i for i, team in enumerate(teams)}
    n_teams = len(teams)
    home = games["home"].map(index).to_numpy(dtype=np.int64)
    away = games["away"].map(index).to_numpy(dtype=np.int64)
    home_flag = np.where(games["neutral"].to_numpy(dtype=bool), 0.0, 1.0)
    rest = np.zeros((n, 0)) if rest_diff is None else np.asarray(rest_diff, dtype=np.float64)
    k = rest.shape[1]
    usable = (
        games["played"].to_numpy(dtype=bool)
        & ~games["forfeit"].to_numpy(dtype=bool)
        & np.isfinite(target)
        & np.isfinite(rest).all(axis=1)
    )
    columns: IntArray = np.concatenate(
        [
            np.zeros((n, 1), dtype=np.int64),
            (1 + home)[:, None],
            (1 + away)[:, None],
            np.broadcast_to(1 + n_teams + np.arange(k, dtype=np.int64), (n, k)),
        ],
        axis=1,
    )
    values = np.concatenate([home_flag[:, None], np.ones((n, 1)), -np.ones((n, 1)), rest], axis=1)
    penalty = np.concatenate(
        [[HOME_PENALTY], np.full(n_teams, residual_ridge), np.full(k, rest_ridge)]
    )
    model = DecayedRidge(penalty, DecayParams(half_life_days, 1.0, 0.0))
    seasons = games["season"].to_numpy(dtype=np.int64)
    cutoff = round_cutoffs(games).to_numpy(dtype=np.float64)

    adjustment = np.full(n, np.nan)
    rest_coef = np.full((n, k), np.nan)
    home_coef = np.full(n, np.nan)
    added = 0
    fitted_any = False
    for time in np.unique(cutoff):
        due = np.flatnonzero(cutoff == time)
        model.advance(float(time), int(seasons[due[0]]))
        stop = int(np.searchsorted(tipoff, time, side="left"))
        rows = np.flatnonzero(usable[added:stop]) + added
        added = stop
        if len(rows):
            model.add(
                columns[rows],
                values[rows],
                target=target[rows],
                weight=np.ones(len(rows)),
                time=tipoff[rows],
                season=seasons[rows],
            )
            fitted_any = True
        if not fitted_any:
            continue
        theta = model.solve()
        h, r, beta = theta[0], theta[1 : 1 + n_teams], theta[1 + n_teams :]
        home_coef[due] = h
        rest_coef[due] = beta
        adjustment[due] = h * home_flag[due] + r[home[due]] - r[away[due]] + rest[due] @ beta
    return ResidualFit(adjustment, rest_coef, home_coef)


def _probit_loglik(
    b: FloatArray, components: FloatArray, home_won: FloatArray
) -> tuple[float, FloatArray]:
    """Negative log-likelihood of p = Φ(components · b) and its gradient."""
    z = components @ b
    log_p, log_q = special.log_ndtr(z), special.log_ndtr(-z)
    log_pdf = -0.5 * z**2 - 0.5 * math.log(2.0 * math.pi)
    score = home_won * np.exp(log_pdf - log_p) - (1.0 - home_won) * np.exp(log_pdf - log_q)
    loss = -float(np.sum(home_won * log_p + (1.0 - home_won) * log_q))
    gradient: FloatArray = -(components.T @ score)
    return loss, gradient


def fit_blend(components: FloatArray, home_won: FloatArray) -> FloatArray:
    """Weights of a probit blend p = Φ(Σ_k b_k m_k) with b ≥ 0 and no intercept, b / Σb.

    Maximum likelihood on rows where every component is finite; equal weights if Σb = 0.
    """
    finite = np.isfinite(components).all(axis=1) & np.isfinite(home_won)
    if not finite.any():
        raise ValueError("no rows with finite components to fit a blend on")
    k = components.shape[1]
    found = optimize.minimize(
        _probit_loglik,
        np.full(k, 1.0 / (10.0 * k)),
        args=(components[finite], home_won[finite]),
        jac=True,
        method="L-BFGS-B",
        bounds=[(0.0, None)] * k,
    )
    total = float(np.sum(found.x))
    if total == 0.0:
        return np.full(k, 1.0 / k)
    weights: FloatArray = found.x / total
    return weights


def blend_by_season(
    components: FloatArray, home_won: FloatArray, season: IntArray, rated: BoolArray
) -> FloatArray:
    """(n_games, k) blend weights: each season's games use ``fit_blend`` on the rated rows of
    earlier seasons only (NaN weights when no earlier rated row has finite components).
    Multiply by ``components`` and sum over k to get the blended margin."""
    weights = np.full(components.shape, np.nan)
    usable = rated & np.isfinite(components).all(axis=1) & np.isfinite(home_won)
    for s in np.unique(season):
        earlier = usable & (season < s)
        if earlier.any():
            weights[season == s] = fit_blend(components[earlier], home_won[earlier])
    return weights


def _logit(p: FloatArray) -> FloatArray:
    clipped = np.clip(p, P_CLIP, 1.0 - P_CLIP)
    out: FloatArray = np.log(clipped) - np.log1p(-clipped)
    return out


def _platt_loss(params: FloatArray, x: FloatArray, y: FloatArray) -> tuple[float, FloatArray]:
    t = params[0] + params[1] * x
    loss = float(np.sum(np.logaddexp(0.0, t) - y * t))
    residual = special.expit(t) - y
    return loss, np.array([np.sum(residual), np.sum(residual * x)])


def fit_platt(p: FloatArray, home_won: FloatArray) -> tuple[float, float]:
    """(a, b) of p' = sigmoid(a + b·logit(p)) by maximum likelihood (convex, starts at identity)."""
    found = optimize.minimize(
        _platt_loss,
        np.array([0.0, 1.0]),
        args=(_logit(p), home_won),
        jac=True,
        method="L-BFGS-B",
        options={"ftol": 1e-15, "gtol": 1e-10},
    )
    return float(found.x[0]), float(found.x[1])


def apply_platt(p: FloatArray, a: float, b: float) -> FloatArray:
    out: FloatArray = special.expit(a + b * _logit(p))
    return out


def p_over(total: FloatArray, sigma: float, line: float | FloatArray) -> FloatArray:
    """P(total > line) when the game total is Normal(``total``, ``sigma``)."""
    out: FloatArray = special.ndtr((total - line) / sigma)
    return out


def p_cover(margin: FloatArray, model: MarginModel, line: float | FloatArray) -> FloatArray:
    """P(home margin > line) under ``model``'s (constant-scale) distribution."""
    if model.ref_pace is not None:
        raise ValueError("p_cover needs a constant-scale model, not a pace-scaled one")
    return model.cdf((margin - line) / model.scale)
