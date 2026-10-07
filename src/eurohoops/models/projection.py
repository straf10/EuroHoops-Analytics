"""M6 player projections (weeks 16-18 L1): per-100 rates, shooting percentages and impact ratings
for a target season (the next season, or the rest of a season at a checkpoint), with an 80%
interval, from the player's earlier seasons.

Per stat, a player's usable seasons (every row with ``season < t`` and, for a checkpoint target,
the partial season-``t`` row of the target's competition: rows at or after the cutoff change
nothing) are blended with weights ``decay * exposure`` (``decay = 0.5 ** (k / half_life)``, k the
seasons before the target, 0 for the partial current season; exposure is possessions for counts,
attempts for percentages) and shrunk to a position-free league prior::

    m    = sum(w_i r_i) / sum(w_i)                        # the player's decayed rate
    S    = u(mu) * sum(w_i^2 f_i^2 / n_i) / sum(w_i)^2    # its sampling variance
    w    = tau2 / (tau2 + S)                              # empirical-Bayes weight on his own data
    mean = w * m + (1 - w) * mu0

``u(mu)`` is the variance of one unit of exposure, Poisson-style for counts (``100 * mu``) and
binomial-style for percentages (``mu * (1 - mu)``), taken at the shrunk rate (one fixed-point
step from the league mean); ``f_i`` is the translation scale of a translated row (1 otherwise).
``mu0`` is the exposure-weighted mean of the target competition's rows in seasons ``t-3 .. t-1``
and ``tau2`` the between-player variance of the same rows by method of moments (observed
variance minus mean sampling variance, exact for unequal weights). No grid: the prior strength
is whatever the data say.

The predictive variance is ``w^2 (S + D * staleness) + (1 - w)^2 tau2 + T``: the staleness is the
squared cumulative weight of the rows older than each season up to the target, summed over the
seasons (the player's true rate drifts by ``D`` per season, so a stale blend is a worse
estimate: with one row ``g`` seasons old it is ``g``), and ``T`` the sampling variance of the
target-season rate at the reference ``exposure``. A player with no usable row gets the league
mean and ``tau2 + T``. The interval is the Normal quantile, clipped at 0 below.

Hooks, never fitted here: ``aging`` (``AgeAdjust``: change in a rate between two ages, from the
aging curve) moves every row to the target age; ``translation`` (M4's delta and pseudocount)
moves other-league rows to the target league as ``(r + c) * exp(+-delta) - c``, floored at 0.
Percentages are not translated. Ages are inputs only: they are held in memory, never stored.

Impact stats (``spm``, ``brapm``) are measurements with a known ``sd``: rows blend with weights
``decay / sd^2``, the prior is the possession-weighted mean of the competition's impact rows in
``t-3 .. t-1`` (or, for ``brapm`` with ``impact_prior="spm"``, the target's own projected SPM,
its spread being the BRAPM-minus-SPM second moment plus the SPM projection variance), and there
is no sampling term. A stat whose league prior cannot be formed (fewer than two impact rows with
possessions in the window, e.g. BRAPM for GBL) is left out for those targets.

Drift (``fit_drift``) is estimated by the caller's choice of ``before_season`` (walk-forward on
tuning targets, frozen for validation and test, D6); it uses complete seasons strictly before it.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import pandera.pandas as pa
from scipy import special

from eurohoops.models.elo import FloatArray
from eurohoops.models.player_seasons import (
    COMPETITIONS,
    COUNT_STATS,
    IMPACT_STATS,
    PCT_STATS,
    PROJECTED_STATS,
    rate_table,
)
from eurohoops.parse.schemas import schema_dtypes, validated

AgeAdjust = Callable[[str, FloatArray, FloatArray], FloatArray]
FLAGS = ("no_history", "no_age", "translated", "partial_season", "no_impact_input")
VARIANTS = ("proj_shrunk", "proj_age", "proj_full", "proj_full_spm")
PRIOR_SEASONS = 3  # the league prior reads seasons t-3 .. t-1
_OUTPUT_COLUMNS = (
    "tid",
    "order",
    "mean",
    "sd",
    "lo80",
    "hi80",
    "prior_mean",
    "weight",
    "exposure",
    "n_seasons",
    "flags",
)
_RATE_COLUMNS = (
    "poss",
    *(c for stat in COUNT_STATS + PCT_STATS for c in (stat, f"{stat}_n")),
)
_UNIT_FLOOR = 1e-9  # a stat at 0 (or 1) still has sampling noise
_SD_FLOOR = 1e-3  # an impact row with sd 0 is a very precise measurement, not an infinite one

TARGET_SCHEMA = pa.DataFrameSchema(
    {
        "person_id": pa.Column(str),
        "competition": pa.Column(str, pa.Check.isin(COMPETITIONS)),
        "season": pa.Column("int64"),
        "checkpoint": pa.Column("float64", pa.Check.in_range(0.0, 1.0, include_max=False)),
        "exposure": pa.Column("float64", pa.Check.gt(0.0)),
    },
    unique=["person_id", "competition", "season", "checkpoint"],
    strict=True,
)

IMPACT_SCHEMA = pa.DataFrameSchema(
    {
        "person_id": pa.Column(str),
        "competition": pa.Column(str, pa.Check.isin(COMPETITIONS)),
        "season": pa.Column("int64"),
        "stat": pa.Column(str, pa.Check.isin(IMPACT_STATS)),
        "value": pa.Column("float64"),
        "sd": pa.Column("float64", pa.Check.ge(0.0)),
    },
    unique=["person_id", "competition", "season", "stat"],
    strict=True,
)

PROJECTIONS_SCHEMA = pa.DataFrameSchema(
    {
        "person_id": pa.Column(str),
        "competition": pa.Column(str, pa.Check.isin(COMPETITIONS)),
        "season": pa.Column("int64"),
        "checkpoint": pa.Column("float64", pa.Check.in_range(0.0, 1.0, include_max=False)),
        "stat": pa.Column(str, pa.Check.isin(PROJECTED_STATS)),
        "mean": pa.Column("float64"),
        "sd": pa.Column("float64", pa.Check.ge(0.0)),
        "lo80": pa.Column("float64"),
        "hi80": pa.Column("float64"),
        "prior_mean": pa.Column("float64"),
        "weight": pa.Column("float64", pa.Check.in_range(0.0, 1.0)),
        "exposure": pa.Column("float64", pa.Check.ge(0.0)),
        "n_seasons": pa.Column("int64", pa.Check.ge(0)),
        "flags": pa.Column(str),
    },
    unique=["person_id", "competition", "season", "checkpoint", "stat"],
    strict=True,
)


@dataclass(frozen=True)
class ProjectionParams:
    """One projection variant at one half-life (L-f)."""

    half_life: float
    aging: bool
    translation: bool
    impact_prior: Literal["league", "spm"]
    interval: float = 0.8


@dataclass(frozen=True)
class Translation:
    """M4's translation factors for one target season (passed in, never fitted here)."""

    delta: Mapping[str, float]
    c: Mapping[str, float]
    target_season: int


def variant_params(variant: str, half_life: float) -> ProjectionParams:
    """The parameters of ``proj_shrunk`` / ``proj_age`` / ``proj_full`` / ``proj_full_spm``."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown projection variant {variant!r}; expected one of {VARIANTS}")
    return ProjectionParams(
        half_life=half_life,
        aging=variant != "proj_shrunk",
        translation=variant in ("proj_full", "proj_full_spm"),
        impact_prior="spm" if variant == "proj_full_spm" else "league",
    )


def _unit(stat: str, rate: FloatArray) -> FloatArray:
    """Variance of one unit of exposure at ``rate``: Poisson per 100 possessions for counts,
    binomial per attempt for percentages."""
    unit = 100.0 * rate if stat in COUNT_STATS else rate * (1.0 - rate)
    return np.maximum(unit, _UNIT_FLOOR)


def _moment_variance(
    x: FloatArray, v: FloatArray, w: FloatArray, *, centered: bool = True
) -> tuple[float, float]:
    """Weighted mean of ``x`` and the variance of its true values by method of moments.

    ``x_i`` has true value variance ``tau2`` plus sampling variance ``v_i``; with reliability
    weights ``w_i`` and ``W = sum w``, ``E sum w_i (x_i - mean_w)^2 = sum w_i (1 - w_i / W)
    (tau2 + v_i)``, which is solved for ``tau2`` (floored at 0). ``centered=False`` measures the
    second moment about 0 instead of the weighted mean (a prior mean fixed from outside).
    """
    if len(x) < 2:
        raise ValueError("a variance by moments needs at least two rows")
    total = w.sum()
    mean = float((w * x).sum() / total)
    centre = mean if centered else 0.0
    share = w * (1.0 - w / total) if centered else w
    tau2 = ((w * (x - centre) ** 2).sum() - (share * v).sum()) / share.sum()
    return mean, max(float(tau2), 0.0)


def fit_drift(
    history: pd.DataFrame, before_season: int, impact: pd.DataFrame | None = None
) -> dict[str, float]:
    """Season-to-season variance of a player's true rate, per stat, from the same person's
    consecutive complete seasons strictly before ``before_season`` (D6).

    The change between two seasons has variance ``D`` (the drift) plus the two sampling
    variances; ``D`` is the second moment of the changes about their weighted mean (the mean is
    the aging effect, a separate hook) minus the sampling variances, weights the precision of the
    change. Impact stats are included only when ``impact`` has at least two such pairs for them.
    """
    complete = history[~history["partial"] & (history["season"] < before_season)]
    rates = rate_table(complete)
    later = rates.assign(season=rates["season"] - 1)
    pairs = rates.merge(later, on=["person_id", "competition", "season"], suffixes=("", "_b"))
    out = {}
    for stat in COUNT_STATS + PCT_STATS:
        n_a, n_b = pairs[f"{stat}_n"].to_numpy(), pairs[f"{stat}_n_b"].to_numpy()
        r_a, r_b = pairs[stat].to_numpy(), pairs[f"{stat}_b"].to_numpy()
        ok = np.isfinite(r_a) & np.isfinite(r_b) & (n_a > 0) & (n_b > 0)
        n_a, n_b, r_a, r_b = n_a[ok], n_b[ok], r_a[ok], r_b[ok]
        precision = 1.0 / (1.0 / n_a + 1.0 / n_b)
        sampling = _unit(stat, (r_a + r_b) / 2.0) / precision
        out[stat] = _drift_of(stat, r_b - r_a, sampling, precision, before_season)
    if impact is not None:
        for stat in IMPACT_STATS:
            rows = impact[(impact["stat"] == stat) & (impact["season"] < before_season)]
            first = rows[["person_id", "competition", "season", "value", "sd"]]
            later = first.assign(season=first["season"] - 1)
            pairs = first.merge(
                later, on=["person_id", "competition", "season"], suffixes=("", "_b")
            )
            sampling = (
                np.maximum(pairs["sd"], _SD_FLOOR) ** 2 + np.maximum(pairs["sd_b"], _SD_FLOOR) ** 2
            )
            if len(pairs) < 2:
                continue  # no such ratings (e.g. BRAPM for GBL): not an error, no drift
            out[stat] = _drift_of(
                stat,
                (pairs["value_b"] - pairs["value"]).to_numpy(),
                sampling.to_numpy(),
                1.0 / sampling.to_numpy(),
                before_season,
            )
    return out


def _drift_of(
    stat: str, change: FloatArray, sampling: FloatArray, weight: FloatArray, before_season: int
) -> float:
    if len(change) < 2:
        raise ValueError(f"no consecutive complete seasons of {stat} before {before_season}")
    return _moment_variance(change, sampling, weight)[1]


def _staleness(
    tid: np.ndarray, season: np.ndarray, share: FloatArray, t: np.ndarray, n_targets: int
) -> FloatArray:
    """Per target: the sum over seasons ``j`` from the oldest row's to ``t`` of the squared
    weight share of the rows older than ``j``. The true rate takes one drift step per season, so
    the error of the blend about the target season's rate has variance ``D`` times this.
    Rows are sorted by (target, season): row i holds from its season to the next row's (or to
    ``t``) the cumulative share of the rows up to itself."""
    totals = np.bincount(tid, weights=share, minlength=n_targets)
    cum = np.cumsum(share) - (np.cumsum(totals) - totals)[tid]
    last = np.append(tid[1:] != tid[:-1], True) if len(tid) else np.zeros(0, dtype=bool)
    upcoming = np.where(last, t[tid], np.append(season[1:], 0)) if len(tid) else season
    return np.bincount(tid, weights=cum**2 * (upcoming - season), minlength=n_targets)


@dataclass
class _Stat:
    """One stat's projection for every target (arrays of length n_targets)."""

    mean: FloatArray
    var: FloatArray
    prior_mean: FloatArray
    weight: FloatArray
    exposure: FloatArray
    n_seasons: np.ndarray
    translated: np.ndarray
    partial: np.ndarray
    no_input: np.ndarray
    present: np.ndarray


def _empty(n_targets: int) -> _Stat:
    """A stat that is projected for no target."""
    zeros = np.zeros(n_targets)
    flags = np.zeros(n_targets, dtype=bool)
    return _Stat(
        zeros, zeros, zeros, zeros, zeros, flags.astype("int64"), flags, flags, flags, flags
    )


def _shrink(
    m: FloatArray,
    has: np.ndarray,
    sampling: FloatArray,
    staleness: FloatArray,
    *,
    drift: float,
    prior_mean: FloatArray,
    prior_var: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Posterior mean, weight on the player's data and the variance of the mean about the true
    rate of the target season (before the target-season sampling noise)."""
    weight = np.where(has, prior_var / np.maximum(prior_var + sampling, 1e-300), 0.0)
    mean = np.where(has, weight * np.where(has, m, 0.0) + (1.0 - weight) * prior_mean, prior_mean)
    var = weight**2 * (sampling + drift * staleness) + (1.0 - weight) ** 2 * prior_var
    return mean, weight, var


def _prior_window(rates: pd.DataFrame, competition: str, season: int) -> dict[str, FloatArray]:
    """The competition's rate rows in seasons ``season-3 .. season-1``, as arrays."""
    window = rates[
        (rates["competition"] == competition)
        & (rates["season"] >= season - PRIOR_SEASONS)
        & (rates["season"] < season)
    ]
    return {c: window[c].to_numpy(dtype="float64") for c in window.columns if c in _RATE_COLUMNS}


def _league_prior(stat: str, window: Mapping[str, FloatArray]) -> tuple[float, float, float]:
    """Mean, between-player variance and attempts per possession of a prior window."""
    x, n = window[stat], window[f"{stat}_n"]
    seen = np.isfinite(x) & (n > 0)
    if seen.sum() < 2:
        raise ValueError(f"no league prior for {stat}: need two rows in the prior seasons")
    x, n = x[seen], n[seen]
    mean = float((n * x).sum() / n.sum())
    _, tau2 = _moment_variance(x, _unit(stat, np.array([mean])) / n, n)
    return mean, tau2, float(window[f"{stat}_n"].sum() / window["poss"].sum())


@dataclass(frozen=True)
class _Targets:
    """The targets as arrays, with each one's league-prior window."""

    t: np.ndarray
    exposure: FloatArray
    keys: list[tuple[str, int]]
    windows: Mapping[tuple[str, int], Mapping[str, FloatArray]]

    @property
    def n(self) -> int:
        return len(self.t)


def _box_stat(
    stat: str,
    rows: Mapping[str, np.ndarray],
    targets: _Targets,
    params: ProjectionParams,
    *,
    drift: float,
    aging: AgeAdjust | None,
    translation: Translation | None,
) -> _Stat:
    n_targets, t = targets.n, targets.t
    rate = rows[stat].copy()
    n = rows[f"{stat}_n"]
    factor = np.ones(len(rate))
    moved = np.zeros(len(rate), dtype=bool)
    if params.translation and translation is not None and stat in COUNT_STATS:
        moved = rows["other"]
        sign = np.where(rows["t_comp"] == "euroleague", 1.0, -1.0)
        scale = np.exp(sign * translation.delta[stat])
        c = translation.c[stat]
        rate = np.where(moved, np.maximum((rate + c) * scale - c, 0.0), rate)
        factor = np.where(moved, scale, 1.0)
    if params.aging and aging is not None:
        aged = np.isfinite(rows["age_from"]) & np.isfinite(rows["age_to"])
        if aged.any():
            rate[aged] += aging(stat, rows["age_from"][aged], rows["age_to"][aged])
        rate = np.clip(rate, 0.0, 1.0 if stat in PCT_STATS else None)
    ok = np.isfinite(rate) & (n > 0)
    tid = rows["tid"][ok]
    rate, n, factor, moved = rate[ok], n[ok], factor[ok], moved[ok]
    decay = rows["decay"][ok]
    w = decay * n
    total = np.bincount(tid, weights=w, minlength=n_targets)
    has = total > 0
    safe = np.where(has, total, 1.0)
    m = np.bincount(tid, weights=w * rate, minlength=n_targets) / safe
    spread = np.bincount(tid, weights=w**2 * factor**2 / n, minlength=n_targets) / safe**2
    stale = _staleness(tid, rows["season"][ok], w / safe[tid], t, n_targets)
    keys = targets.keys
    priors = {key: _league_prior(stat, targets.windows[key]) for key in set(keys)}
    mu0 = np.array([priors[k][0] for k in keys])
    tau2 = np.array([priors[k][1] for k in keys])
    league_ratio = np.array([priors[k][2] for k in keys])
    first = tau2 / np.maximum(tau2 + _unit(stat, mu0) * spread, 1e-300)
    mu1 = np.where(has, first * m + (1.0 - first) * mu0, mu0)
    mean, weight, var = _shrink(
        m, has, _unit(stat, mu1) * spread, stale, drift=drift, prior_mean=mu0, prior_var=tau2
    )
    mean = np.clip(mean, 0.0, 1.0 if stat in PCT_STATS else None)
    exposure = targets.exposure
    if stat in PCT_STATS:
        # the reference exposure is in possessions: attempts follow his own attempts per possession
        poss = np.bincount(rows["tid"], weights=rows["decay"] * rows["poss"], minlength=n_targets)
        made = np.bincount(
            rows["tid"], weights=rows["decay"] * rows[f"{stat}_n"], minlength=n_targets
        )
        own = np.divide(made, poss, out=league_ratio.copy(), where=poss > 0)
        exposure = np.maximum(exposure * own, 1.0)
    return _Stat(
        mean=mean,
        var=var + _unit(stat, mean) / exposure,
        prior_mean=mu0,
        weight=weight,
        exposure=np.bincount(tid, weights=decay * n, minlength=n_targets),
        n_seasons=np.bincount(tid, minlength=n_targets),
        translated=np.bincount(tid, weights=moved, minlength=n_targets) > 0,
        partial=np.bincount(tid, weights=rows["current"][ok], minlength=n_targets) > 0,
        no_input=np.zeros(n_targets, dtype=bool),
        present=np.ones(n_targets, dtype=bool),
    )


def _impact_window(
    impact: pd.DataFrame, poss: pd.DataFrame, competition: str, season: int
) -> pd.DataFrame:
    """The competition's impact rows in seasons ``season-3 .. season-1`` that have possessions."""
    window = impact[
        (impact["competition"] == competition)
        & (impact["season"] >= season - PRIOR_SEASONS)
        & (impact["season"] < season)
    ]
    window = window.merge(poss, on=["person_id", "competition", "season"])
    return window[window["poss"] > 0]


def _impact_prior(
    stat: str,
    impact: pd.DataFrame,
    poss: pd.DataFrame,
    params: ProjectionParams,
    *,
    competition: str,
    season: int,
) -> tuple[float, float] | None:
    """The league mean and spread of ``stat``; with ``impact_prior="spm"`` for BRAPM, the
    second moment of BRAPM minus SPM about 0 instead (its mean is the target's own SPM). None
    when fewer than two rows can form it."""
    window = _impact_window(impact[impact["stat"] == stat], poss, competition, season)
    if params.impact_prior == "spm" and stat == "brapm":
        spm = _impact_window(impact[impact["stat"] == "spm"], poss, competition, season)
        both = window.merge(spm, on=["person_id", "competition", "season"], suffixes=("", "_s"))
        if len(both) < 2:
            return None
        v = np.maximum(both["sd"], _SD_FLOOR) ** 2 + np.maximum(both["sd_s"], _SD_FLOOR) ** 2
        return _moment_variance(
            (both["value"] - both["value_s"]).to_numpy(),
            v.to_numpy(),
            both["poss"].to_numpy(),
            centered=False,
        )
    if len(window) < 2:
        return None
    return _moment_variance(
        window["value"].to_numpy(),
        np.maximum(window["sd"].to_numpy(), _SD_FLOOR) ** 2,
        window["poss"].to_numpy(),
    )


def _impact_stat(
    stat: str,
    keys: pd.DataFrame,
    impact: pd.DataFrame | None,
    poss: pd.DataFrame,
    params: ProjectionParams,
    *,
    drift: float,
    aging: AgeAdjust | None,
    offsets: pd.Series,
    spm: _Stat | None,
) -> _Stat:
    n_targets = len(keys)
    if impact is None:
        return _empty(n_targets)
    own = impact[impact["stat"] == stat]
    rows = keys.merge(own, left_on=["person_id", "t_comp"], right_on=["person_id", "competition"])
    current = (rows["season"] == rows["t"]) & (rows["checkpoint"] > 0)
    rows = rows[(rows["season"] < rows["t"]) | current].assign(current=current)
    rows = rows.merge(poss, on=["person_id", "competition", "season"], how="left")
    rows = rows.sort_values(["tid", "season"], kind="stable").reset_index(drop=True)
    rows["poss"] = rows["poss"].fillna(0.0)
    value = rows["value"].to_numpy(dtype="float64").copy()
    if params.aging and aging is not None:
        offset = rows["person_id"].map(offsets)
        aged = offset.notna().to_numpy()
        if aged.any():
            with contextlib.suppress(KeyError):  # a curve without this stat leaves it as is
                value[aged] += aging(
                    stat,
                    (offset[aged] + rows.loc[aged, "season"]).to_numpy(dtype="float64"),
                    (offset[aged] + rows.loc[aged, "t"]).to_numpy(dtype="float64"),
                )
    tid = rows["tid"].to_numpy()
    sd2 = np.maximum(rows["sd"].to_numpy(), _SD_FLOOR) ** 2
    decay = 0.5 ** (((rows["t"] - rows["season"]) / params.half_life).to_numpy())
    g = decay / sd2
    total = np.bincount(tid, weights=g, minlength=n_targets)
    has = total > 0
    safe = np.where(has, total, 1.0)
    m = np.bincount(tid, weights=g * value, minlength=n_targets) / safe
    sampling = np.bincount(tid, weights=decay * g, minlength=n_targets) / safe**2
    t = keys["t"].to_numpy()
    stale = _staleness(tid, rows["season"].to_numpy(), g / safe[tid], t, n_targets)
    mu0, tau2, present = np.zeros(n_targets), np.zeros(n_targets), np.zeros(n_targets, dtype=bool)
    cache = {}
    for i, key in enumerate(zip(keys["t_comp"], t, strict=True)):
        if key not in cache:
            cache[key] = _impact_prior(
                stat, impact, poss, params, competition=key[0], season=key[1]
            )
        found = cache[key]
        if found is not None:
            mu0[i], tau2[i], present[i] = found[0], found[1], True
    if params.impact_prior == "spm" and stat == "brapm" and spm is not None:
        present &= spm.present
        mu0, tau2 = spm.mean, tau2 + spm.var
    mean, weight, var = _shrink(
        m, has, sampling, stale, drift=drift, prior_mean=mu0, prior_var=tau2
    )
    return _Stat(
        mean=mean,
        var=var,
        prior_mean=mu0,
        weight=weight,
        exposure=np.bincount(tid, weights=decay * rows["poss"].to_numpy(), minlength=n_targets),
        n_seasons=np.bincount(tid, minlength=n_targets),
        translated=np.zeros(n_targets, dtype=bool),
        partial=np.bincount(tid, weights=rows["current"].to_numpy(), minlength=n_targets) > 0,
        no_input=~has,
        present=present,
    )


def _usable_rows(
    targets: pd.DataFrame, rates: pd.DataFrame, params: ProjectionParams, offsets: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Each target's usable history rows (module docstring) with their decay and ages."""
    keys = targets[["person_id", "competition", "season", "checkpoint"]].rename(
        columns={"competition": "t_comp", "season": "t"}
    )
    keys["tid"] = np.arange(len(keys))
    rows = keys.merge(rates, on="person_id")
    current = (
        (rows["season"] == rows["t"])
        & rows["partial"]
        & (rows["checkpoint"] > 0)
        & (rows["competition"] == rows["t_comp"])
    )
    rows = rows[(rows["season"] < rows["t"]) | current].assign(current=current)
    rows = rows.sort_values(["tid", "season"], kind="stable").reset_index(drop=True)
    rows["decay"] = 0.5 ** ((rows["t"] - rows["season"]) / params.half_life)
    rows["other"] = rows["competition"] != rows["t_comp"]
    offset = rows["person_id"].map(offsets)
    rows["age_from"] = offset + rows["season"]
    rows["age_to"] = offset + rows["t"]
    return keys, rows


def project(
    history: pd.DataFrame,
    target: pd.DataFrame,
    params: ProjectionParams,
    *,
    drift: Mapping[str, float],
    impact: pd.DataFrame | None = None,
    ages: pd.DataFrame | None = None,
    aging: AgeAdjust | None = None,
    translation: Translation | None = None,
) -> pd.DataFrame:
    """Projections (``PROJECTIONS_SCHEMA``) for every target, from the rows it may use.

    ``drift`` is ``fit_drift`` (box stats required; an impact stat missing from it has drift 0).
    ``aging`` is required when ``params.aging``; ``translation`` when ``params.translation`` and a
    target has a usable row of the other competition, and its ``target_season`` may not be after
    any such target's season (walk-forward). A target with no age is not aged (flag ``no_age``).
    """
    # cast only: a duplicate, a checkpoint outside [0, 1) or a non-positive exposure fails the
    # PROJECTIONS_SCHEMA check of the output
    targets = target[list(TARGET_SCHEMA.columns)].astype(schema_dtypes(TARGET_SCHEMA))
    targets = targets.reset_index(drop=True)
    if params.aging and aging is None:
        raise ValueError("params.aging needs an aging callable")
    if impact is not None:
        impact = validated(impact, IMPACT_SCHEMA)
    rates = rate_table(history)
    rates["poss"] = rates["pts_n"]
    # age at season s = offset + s: any known age of the person fixes it (in memory only)
    offsets = (
        (ages["age"] - ages["season"]).groupby(ages["person_id"]).mean()
        if ages is not None
        else pd.Series(dtype="float64")
    )
    keys, rows = _usable_rows(targets, rates, params, offsets)
    if params.translation and rows["other"].any():
        if translation is None:
            raise ValueError("params.translation needs M4's factors for other-league rows")
        if (rows.loc[rows["other"], "t"] < translation.target_season).any():
            raise ValueError("translation fitted for a season after the target (walk-forward)")
    n_targets = len(targets)
    z = float(special.ndtri(0.5 + params.interval / 2.0))
    pairs = list(zip(targets["competition"], targets["season"].astype(int), strict=True))
    arrays = _Targets(
        t=targets["season"].to_numpy(),
        exposure=targets["exposure"].to_numpy(),
        keys=pairs,
        windows={key: _prior_window(rates, *key) for key in set(pairs)},
    )
    columns = {c: rows[c].to_numpy() for c in rows.columns}
    results: dict[str, _Stat] = {}
    for stat in COUNT_STATS + PCT_STATS:
        results[stat] = _box_stat(
            stat, columns, arrays, params,
            drift=drift[stat], aging=aging, translation=translation,
        )  # fmt: skip
    poss = rates[["person_id", "competition", "season", "poss"]]
    for stat in IMPACT_STATS:
        results[stat] = _impact_stat(
            stat, keys, impact, poss, params,
            drift=drift.get(stat, 0.0), aging=aging, offsets=offsets, spm=results.get("spm"),
        )  # fmt: skip
    has_rows = np.isin(np.arange(n_targets), columns["tid"])
    known_age = targets["person_id"].isin(offsets.index).to_numpy()
    names = sorted(FLAGS)
    spelled = np.array(
        ["|".join(n for j, n in enumerate(names) if m >> j & 1) for m in range(2 ** len(names))],
        dtype=object,
    )
    parts: dict[str, list[np.ndarray]] = {c: [] for c in _OUTPUT_COLUMNS}
    for order, stat in enumerate(PROJECTED_STATS):
        res = results[stat]
        picked = {
            "no_history": ~has_rows,
            "no_age": has_rows & ~known_age & params.aging & (res.n_seasons > 0),
            "translated": res.translated,
            "partial_season": res.partial,
            "no_impact_input": res.no_input,
        }
        mask = sum(picked[name].astype("int64") << j for j, name in enumerate(names))
        sd = np.sqrt(res.var)
        low = res.mean - z * sd
        keep = res.present
        columns_out = {
            "tid": np.arange(n_targets),
            "order": np.full(n_targets, order),
            "mean": res.mean,
            "sd": sd,
            "lo80": low if stat in IMPACT_STATS else np.maximum(low, 0.0),
            "hi80": res.mean + z * sd,
            "prior_mean": res.prior_mean,
            "weight": np.clip(res.weight, 0.0, 1.0),
            "exposure": res.exposure,
            "n_seasons": res.n_seasons,
            "flags": spelled[mask],
        }
        for name, values in columns_out.items():
            parts[name].append(values[keep])
    joined = {name: np.concatenate(values) for name, values in parts.items()}
    rank = np.lexsort((joined["order"], joined["tid"]))
    joined = {name: values[rank] for name, values in joined.items()}
    out = targets[["person_id", "competition", "season", "checkpoint"]].iloc[joined["tid"]]
    out = out.reset_index(drop=True)
    out["stat"] = np.array(PROJECTED_STATS, dtype=object)[joined["order"]]
    for name in (
        "mean",
        "sd",
        "lo80",
        "hi80",
        "prior_mean",
        "weight",
        "exposure",
        "n_seasons",
        "flags",
    ):
        out[name] = joined[name]
    return validated(out[list(PROJECTIONS_SCHEMA.columns)], PROJECTIONS_SCHEMA)
