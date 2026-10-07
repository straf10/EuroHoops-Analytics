"""Aging curve (weeks 16-18 L2): the expected year-over-year change of each projected stat by age,
and ``apply`` to move a rate from one age to another (the projection, L1, calls it).

Method (L-d): the delta method on pairs of consecutive complete seasons of the same person and
competition. A pair (s, s+1) has age = the person's age in season s (rounded to an integer; a
model input only, never stored per person) and delta = rate(s+1) - rate(s). Only pairs with
``s + 1 < cutoff_season`` are used, complete seasons only (``partial`` rows and every row with
season >= the cutoff never enter any fit, prior or mean: walk-forward).

Survivor correction. A player with fewer than ``min_poss`` possessions in s+1 was benched or left
because he got worse; keeping only players who stay on court biases every delta upward. A pair
whose season s has at least ``min_poss`` possessions but whose season s+1 has 0 < poss < min_poss
is kept: its s+1 rate is shrunk (empirical Bayes) toward the exposure-weighted mean of the
short s+1 rates of the same competition and age (the players who stayed on court are a different
population, and so is every other age; a global mean would erase part of the decline). The
true-talent variance is the spread of those short rates about their group means minus their
sampling variance; a group with fewer than ``MIN_SHORT_ROWS`` pairs has no prior and is left out.
A player with no s+1 row at all is not observable and cannot be paired: the correction recovers
the short seasons, not the departures. ``survivor_correction=False`` drops the short-season pairs
(the planted bias in the tests).

Sampling variance of a rate: Poisson, 100 * mean / possessions, for per-100 counts; p(1-p) /
attempts for percentages; the supplied sd squared for impact ratings. A pair's variance is the
sum of its two seasons' (for a shrunk season, its posterior variance B * var, B being the
shrinkage factor). Weights (D10): without the correction a pair is weighted by 1 / (variance +
tau_d^2), the inverse sum of the two exposures (the harmonic-mean-of-exposure weight) plus a
method-of-moments random effect for the spread of true changes between players. With the
correction every pair is weighted by its season-s exposure only, 1 / (var_s + tau_d^2): season s
is the qualified season, fixed before the outcome is known, whereas a weight that grows with the
s+1 exposure would under-represent exactly the players who then lost minutes (the decliners).
The standard error of a weighted age mean always uses the pairs' own variances (the posterior
variance for a shrunk rate). Per age the weighted mean is partially pooled toward a weighted
quadratic through all ages (random effect across ages, DerSimonian-Laird), then smoothed with a
least-squares quadratic spline (C1, three interior knots at most). The standard error is the
spline's, propagated from the per-age variances (the contraction from pooling is not credited:
conservative).

Ages come from ``player_ages`` (``models/player_seasons.py``, built from ``ingest/bios.py`` in
memory); birth dates and exact ages never reach a committed file, a report or the site (L-e).
``AgingCurve`` holds integer age grid points and deltas only, nothing per person.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from eurohoops.models.player_seasons import (
    COUNT_STATS,
    IMPACT_STATS,
    PCT_STATS,
    PROJECTED_STATS,
    rate_table,
)

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

MIN_AGE_PAIRS = 30  # an age is on the grid ends only with this many pairs
MAX_INTERIOR_KNOTS = 3
MIN_SHORT_ROWS = 5  # short-season rows a competition-season needs to have a shrinkage prior
_KEYS = ["person_id", "competition", "season"]
_TINY = 1e-12


@dataclass(frozen=True)
class AgingCurve:
    cutoff_season: int  # fitted on pairs (s, s+1) with s + 1 < cutoff_season, complete seasons only
    ages: FloatArray  # integer ages ascending; ends set by data with >= MIN_AGE_PAIRS pairs
    delta: Mapping[str, FloatArray]  # stat -> expected change from age a to a + 1 at each ages[i]
    se: Mapping[str, FloatArray]  # stat -> its standard error
    n_pairs: Mapping[str, IntArray]  # stat -> pairs behind each age


def _stat_rows(
    history: pd.DataFrame, cutoff: int, impact: pd.DataFrame | None
) -> dict[str, pd.DataFrame]:
    """Per stat, the usable complete pre-cutoff rows: value, exposure ``n`` and the season's
    possessions (impact rows also carry their sampling variance ``var``; the others get theirs in
    ``_with_noise``)."""
    complete = history[~history["partial"] & (history["season"] < cutoff)].reset_index(drop=True)
    rates = rate_table(complete)
    out: dict[str, pd.DataFrame] = {}
    for stat in COUNT_STATS + PCT_STATS:
        frame = rates[_KEYS].copy()
        frame["value"] = rates[stat].to_numpy()
        frame["n"] = rates[f"{stat}_n"].to_numpy()
        frame["poss"] = complete["poss"].to_numpy()
        out[stat] = frame[np.isfinite(frame["value"]) & (frame["n"] > 0)].reset_index(drop=True)
    if impact is not None:
        base = complete[[*_KEYS, "poss"]]
        for stat in IMPACT_STATS:
            part = impact[(impact["stat"] == stat) & (impact["season"] < cutoff)]
            joined = part.merge(base, on=_KEYS, how="inner")
            frame = joined[_KEYS].copy()
            frame["value"] = joined["value"].to_numpy()
            frame["n"] = joined["poss"].to_numpy()
            frame["poss"] = joined["poss"].to_numpy()
            frame["var"] = np.maximum(joined["sd"].to_numpy(dtype="float64") ** 2, _TINY)
            out[stat] = frame[np.isfinite(frame["value"]) & (frame["n"] > 0)].reset_index(drop=True)
    return out


def _with_noise(stat: str, rows: pd.DataFrame, min_poss: float) -> pd.DataFrame:
    """Add the sampling variance of each rate (pooled mean from the qualified rows)."""
    rows = rows.copy()
    if stat in IMPACT_STATS:
        return rows
    qualified = rows[rows["poss"] >= min_poss]
    pool = qualified if len(qualified) else rows
    mean = float(np.average(pool["value"], weights=pool["n"]))
    n = rows["n"].to_numpy(dtype="float64")
    if stat in COUNT_STATS:
        rows["var"] = np.maximum(100.0 * mean / n, _TINY)
    else:
        rows["var"] = np.maximum(mean * (1.0 - mean) / n, _TINY)
    return rows


def _shrink_short(short: pd.DataFrame) -> pd.DataFrame:
    """Shrink the s+1 rate of the pairs with a short next season toward the exposure-weighted
    mean of those pairs' s+1 rates at the same competition and age (the prior of players with this
    little exposure at this age: the players who stayed on court are a different population, and
    so is every other age). The shrinkage factor ``B = tau^2 / (tau^2 + var)`` takes the
    true-talent variance ``tau^2`` from the spread of the short rates about their group means
    minus their sampling variance. A group with fewer than ``MIN_SHORT_ROWS`` pairs has no prior
    and its pairs are left out. Returns the pairs with ``value_next`` and ``var_next`` shrunk."""
    keys = ["competition", "age_int"]
    short = short.assign(wv=short["value_next"] * short["n_next"])
    grouped = short.groupby(keys)
    prior = (grouped["wv"].sum() / grouped["n_next"].sum()).rename("prior")
    size = grouped.size().rename("size")
    short = short.merge(pd.concat([prior, size], axis=1).reset_index(), on=keys)
    short = short[short["size"] >= MIN_SHORT_ROWS]
    if short.empty:
        return short
    spread = float(
        np.mean((short["value_next"] - short["prior"]) ** 2) - np.mean(short["var_next"])
    )
    tau2 = max(spread, _TINY)
    factor = tau2 / (tau2 + short["var_next"])
    short["value_next"] = short["prior"] + factor * (short["value_next"] - short["prior"])
    short["var_next"] = factor * short["var_next"]
    return short


def _pairs(
    rows: pd.DataFrame, ages: pd.DataFrame, min_poss: float, survivors: bool
) -> pd.DataFrame:
    """Pairs (s, s+1) of qualified rows, plus, with ``survivors``, those whose s+1 row is a
    short season (shrunk): integer age, delta and sampling variance."""
    now = rows[rows["poss"] >= min_poss]
    nxt = rows.assign(season=rows["season"] - 1)
    nxt = nxt.rename(columns={"value": "value_next", "var": "var_next", "n": "n_next"})
    pairs = now.merge(
        nxt[[*_KEYS, "value_next", "var_next", "n_next", "poss"]], on=_KEYS, suffixes=("", "_next")
    )
    pairs = pairs.merge(ages[["person_id", "season", "age"]], on=["person_id", "season"])
    pairs["age_int"] = np.floor(pairs["age"] + 0.5).astype("int64")
    regular = pairs["poss_next"] >= min_poss
    kept = [pairs[regular]]
    if survivors:
        kept.append(_shrink_short(pairs[~regular]))
    pairs = pd.concat(kept, ignore_index=True)
    return pd.DataFrame(
        {
            "age_int": pairs["age_int"],
            "delta": pairs["value_next"] - pairs["value"],
            "var": pairs["var"] + pairs["var_next"],
            "wvar": pairs["var"] if survivors else pairs["var"] + pairs["var_next"],
        }
    )


def _basis(x: FloatArray, lo: float, hi: float, degree: int, knots: FloatArray) -> FloatArray:
    """Truncated-power basis of the quadratic spline (C1 at each knot), x scaled to [-1, 1]."""
    span = max(hi - lo, 1.0)
    z = (x - (lo + hi) / 2.0) / (span / 2.0)
    zk = (knots - (lo + hi) / 2.0) / (span / 2.0)
    cols = [z**p for p in range(degree + 1)]
    cols += [np.maximum(z - k, 0.0) ** degree for k in zk]
    return np.column_stack(cols)


def _fit_stat(grid: FloatArray, pairs: pd.DataFrame) -> tuple[FloatArray, FloatArray, IntArray]:
    """Delta, standard error and pair count at each grid age from one stat's pairs."""
    lo, hi = float(grid[0]), float(grid[-1])
    pairs = pairs[pairs["age_int"].isin(grid.astype("int64"))]
    idx = (pairs["age_int"].to_numpy() - int(lo)).astype("int64")
    count = np.bincount(idx, minlength=len(grid)).astype("int64")
    if len(pairs) == 0:
        raise ValueError("no pairs on the age grid")
    d = pairs["delta"].to_numpy(dtype="float64")
    v = pairs["var"].to_numpy(dtype="float64")
    # Random effect for the spread of true changes between players (DerSimonian-Laird).
    w = 1.0 / v
    mean_w = np.bincount(idx, weights=w * d, minlength=len(grid)) / np.maximum(
        np.bincount(idx, weights=w, minlength=len(grid)), _TINY
    )
    q = float(np.sum(w * (d - mean_w[idx]) ** 2))
    dof = len(d) - int(np.count_nonzero(count))
    c = float(
        w.sum()
        - np.sum(
            np.bincount(idx, weights=w**2, minlength=len(grid))
            / np.maximum(np.bincount(idx, weights=w, minlength=len(grid)), _TINY)
        )
    )
    tau2 = max((q - dof) / c, 0.0) if dof > 0 and c > 0 else 0.0
    w = 1.0 / (pairs["wvar"].to_numpy(dtype="float64") + tau2)
    sw = np.bincount(idx, weights=w, minlength=len(grid))
    seen = count > 0
    m = np.bincount(idx, weights=w * d, minlength=len(grid))[seen] / sw[seen]
    var_m = np.bincount(idx, weights=w**2 * (v + tau2), minlength=len(grid))[seen] / sw[seen] ** 2
    x = grid[seen]
    # Partial pooling across ages toward a weighted quadratic through the age means.
    degree = min(2, len(x) - 1)
    design = _basis(x, lo, hi, degree, np.empty(0))
    coef = np.linalg.lstsq(design / np.sqrt(var_m)[:, None], m / np.sqrt(var_m), rcond=None)[0]
    trend = design @ coef
    dof_a = len(x) - (degree + 1)
    wa = 1.0 / var_m
    ca = float(wa.sum() - np.sum(wa**2) / wa.sum())
    qa = float(np.sum(wa * (m - trend) ** 2))
    tau_a = max((qa - dof_a) / ca, 0.0) if dof_a > 0 and ca > 0 else 0.0
    b = tau_a / (tau_a + var_m)
    theta = trend + b * (m - trend)
    # Quadratic spline through the pooled per-age estimates; SE from their sampling variances.
    n_int = max(0, min(MAX_INTERIOR_KNOTS, len(x) - (degree + 1))) if degree == 2 else 0
    knots = np.linspace(lo, hi, n_int + 2)[1:-1]
    fit = _basis(x, lo, hi, degree, knots)
    weight = 1.0 / var_m
    info = np.linalg.pinv(fit.T @ (fit * weight[:, None]))
    beta = info @ (fit.T @ (weight * theta))
    full = _basis(grid, lo, hi, degree, knots)
    delta = full @ beta
    se = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", full, info, full), 0.0))
    return delta, se, count


def aging_curve(
    history: pd.DataFrame,
    ages: pd.DataFrame,
    cutoff_season: int,
    *,
    impact: pd.DataFrame | None = None,
    min_poss: float = 500.0,
    survivor_correction: bool = True,
) -> AgingCurve:
    """The aging curve fitted on pairs (s, s+1) with ``s + 1 < cutoff_season``.

    ``history``: player-season frame (I1; partial rows and rows from the cutoff on are ignored);
    ``ages``: ``player_ages`` output (in memory); ``impact``: impact frame (IMPACT_SCHEMA) to add
    ``spm``/``brapm``. The grid runs over the integer ages from the lowest to the highest age
    with at least ``MIN_AGE_PAIRS`` pairs of ``pts``; a stat with no pair on the grid is left out.
    """
    rows = _stat_rows(history, cutoff_season, impact)
    pairs: dict[str, pd.DataFrame] = {}
    for stat, frame in rows.items():
        noisy = _with_noise(stat, frame, min_poss)
        pairs[stat] = _pairs(noisy, ages, min_poss, survivor_correction)
    base = pairs["pts"]["age_int"].value_counts() if len(pairs["pts"]) else pd.Series(dtype="int64")
    enough = base[base >= MIN_AGE_PAIRS]
    if enough.empty:
        raise ValueError(f"fewer than {MIN_AGE_PAIRS} pairs at every age before {cutoff_season}")
    grid = np.arange(int(enough.index.min()), int(enough.index.max()) + 1, dtype="float64")
    delta: dict[str, FloatArray] = {}
    se: dict[str, FloatArray] = {}
    n_pairs: dict[str, IntArray] = {}
    for stat in PROJECTED_STATS:
        found = pairs.get(stat)
        if found is None or not found["age_int"].isin(grid.astype("int64")).any():
            continue
        delta[stat], se[stat], n_pairs[stat] = _fit_stat(grid, found)
    return AgingCurve(cutoff_season, grid, delta, se, n_pairs)


def _cumulative(curve: AgingCurve, stat: str, age: FloatArray) -> FloatArray:
    """Summed change from the lowest grid age to ``age``: piecewise linear between grid ages, the
    end delta's slope beyond them."""
    delta = curve.delta[stat]
    grid = curve.ages
    at_grid = np.concatenate([[0.0], np.cumsum(delta[:-1])])
    inside = np.interp(age, grid, at_grid)
    below = delta[0] * (age - grid[0])
    above = at_grid[-1] + delta[-1] * (age - grid[-1])
    return np.asarray(np.where(age < grid[0], below, np.where(age > grid[-1], above, inside)))


def apply(curve: AgingCurve, stat: str, age_from: FloatArray, age_to: FloatArray) -> FloatArray:
    """Summed expected change in ``stat`` from ``age_from`` to ``age_to`` (elementwise; fractional
    ages interpolate linearly; beyond the grid ends the end delta continues). Zero when the ages
    are equal; going back in age returns the negative of the forward sum."""
    return _cumulative(curve, stat, np.asarray(age_to, dtype="float64")) - _cumulative(
        curve, stat, np.asarray(age_from, dtype="float64")
    )
