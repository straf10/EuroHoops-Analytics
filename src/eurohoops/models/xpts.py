"""Shot-quality model M2 (xPTS): the spline baseline (F3), fitted with numpy only.

P(make) of a field-goal attempt from the shot context alone (F-b; no shooter, team or opponent
identity: quality is not skill, PLAN R9). xPTS = P(make) * shot value.

Baseline design (F-c): a natural cubic spline in distance for each shot type (2s and 3s, knots
at development quantiles of that type's distances), a 3-point indicator, and linear terms for
|angle|, ``ZONE`` (levels with at least ``MIN_ZONE_SHOTS`` training shots), the three context
flags, seconds left in the period, period (1-4, overtime as one level), the pre-shot margin
(clipped at ±``MARGIN_CLIP``), home and season as a numeric trend.

Every non-intercept column is whitened on the training shots (centred, then multiplied by the
inverse of the Cholesky factor of their covariance; columns constant on the training shots are
dropped), a pure reparametrisation that makes the
Newton steps well conditioned; the L2 penalty ``l2`` applies to the whitened coefficients, on
the mean log-loss scale. The fit is penalised Newton-Raphson (IRLS), deterministic.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from eurohoops.models.elo import FloatArray

MIN_ZONE_SHOTS = 1000
MARGIN_CLIP = 40.0
SEASON_CENTRE = 2016
KNOT_RANGE = (0.025, 0.975)  # outer knots at these quantiles, inner knots evenly between
IDENTITY_COLUMNS = ("shooter", "team", "opponent", "game_id", "player")
MAX_ITERATIONS = 100
CONSTANT_SD = 1e-9
TOLERANCE = 1e-10


def natural_spline_basis(x: FloatArray, knots: FloatArray) -> FloatArray:
    """Natural cubic spline basis without the constant (ESL eq. 5.4-5.5): x, then K-2 columns
    d_k - d_{K-1}, linear beyond the boundary knots."""
    k = len(knots)
    last = knots[-1]

    def d(j: int) -> FloatArray:
        out: FloatArray = (np.maximum(x - knots[j], 0.0) ** 3 - np.maximum(x - last, 0.0) ** 3) / (
            last - knots[j]
        )
        return out

    columns = [x] + [d(j) - d(k - 2) for j in range(k - 2)]
    return np.column_stack(columns)


def spline_knots(distance: FloatArray, n_knots: int) -> FloatArray:
    probabilities = np.linspace(KNOT_RANGE[0], KNOT_RANGE[1], n_knots)
    knots: FloatArray = np.quantile(distance, probabilities)
    return knots


@dataclass(frozen=True)
class SplineSpec:
    """Everything the design matrix needs, fitted on the training shots only."""

    knots_two: FloatArray
    knots_three: FloatArray
    zones: tuple[str, ...]
    centre: FloatArray
    whiten: FloatArray  # upper-triangular inverse Cholesky factor
    names: tuple[str, ...]
    kept: npt.NDArray[np.bool_]  # raw columns that vary on the training shots


def _raw_design(
    shots: pd.DataFrame, knots_two: FloatArray, knots_three: FloatArray, zones: tuple[str, ...]
) -> tuple[FloatArray, tuple[str, ...]]:
    distance = shots["distance"].to_numpy(dtype=np.float64)
    three = (shots["value"].to_numpy() == 3).astype(np.float64)
    two_basis = natural_spline_basis(distance, knots_two) * (1.0 - three)[:, None]
    three_basis = natural_spline_basis(distance, knots_three) * three[:, None]
    period = shots["period"].to_numpy()
    zone = shots["zone"].astype(str).to_numpy()
    columns = [
        two_basis,
        three_basis,
        three[:, None],
        (shots["angle"].to_numpy(dtype=np.float64) / 90.0)[:, None],
        np.column_stack([(zone == z).astype(np.float64) for z in zones])
        if zones
        else np.empty((len(shots), 0)),
        shots[["fastbreak", "second_chance", "points_off_turnover", "home"]].to_numpy(
            dtype=np.float64
        ),
        (shots["seconds_left"].to_numpy(dtype=np.float64) / 600.0)[:, None],
        np.column_stack([(period == p).astype(np.float64) for p in (2, 3, 4)]),
        (period >= 5).astype(np.float64)[:, None],
        (
            np.clip(shots["margin_before"].to_numpy(dtype=np.float64), -MARGIN_CLIP, MARGIN_CLIP)
            / 10.0
        )[:, None],
        ((shots["season"].to_numpy(dtype=np.float64) - SEASON_CENTRE) / 10.0)[:, None],
    ]
    names = (
        *(f"distance_2pt_{i}" for i in range(two_basis.shape[1])),
        *(f"distance_3pt_{i}" for i in range(three_basis.shape[1])),
        "three",
        "angle",
        *(f"zone_{z}" for z in zones),
        "fastbreak",
        "second_chance",
        "points_off_turnover",
        "home",
        "seconds_left",
        "period_2",
        "period_3",
        "period_4",
        "period_ot",
        "margin_before",
        "season",
    )
    return np.column_stack(columns), names


def make_spec(shots: pd.DataFrame, n_knots: int) -> SplineSpec:
    """Knots, zone levels and the whitening transform from the training shots."""
    distance = shots["distance"].to_numpy(dtype=np.float64)
    three = shots["value"].to_numpy() == 3
    counts = shots["zone"].astype(str).value_counts()
    zones = tuple(sorted(str(z) for z, n in counts.items() if n >= MIN_ZONE_SHOTS))[1:]
    knots_two = spline_knots(distance[~three], n_knots)
    knots_three = spline_knots(distance[three], n_knots)
    raw, names = _raw_design(shots, knots_two, knots_three, zones)
    kept = raw.std(axis=0) > CONSTANT_SD  # e.g. season when training on a single season
    raw = raw[:, kept]
    centre = raw.mean(axis=0)
    covariance = np.cov(raw - centre, rowvar=False, bias=True)
    whiten = np.linalg.inv(np.linalg.cholesky(covariance).T)
    names = tuple(n for n, k in zip(names, kept, strict=True) if k)
    if has_identity_column(names):
        raise ValueError(f"identity column in the M2 design: {names}")
    return SplineSpec(knots_two, knots_three, zones, centre, whiten, names, kept)


def design(shots: pd.DataFrame, spec: SplineSpec) -> FloatArray:
    """Whitened design matrix with a leading intercept column."""
    raw, _ = _raw_design(shots, spec.knots_two, spec.knots_three, spec.zones)
    white = (raw[:, spec.kept] - spec.centre) @ spec.whiten
    return np.column_stack([np.ones(len(shots)), white])


def sigmoid(z: FloatArray) -> FloatArray:
    out: FloatArray = 0.5 * (1.0 + np.tanh(0.5 * z))
    return out


def fit_logistic(x: FloatArray, y: FloatArray, l2: float) -> FloatArray:
    """Penalised Newton-Raphson on mean log loss + l2/2 * |beta[1:]|^2 (intercept free)."""
    n, p = x.shape
    beta = np.zeros(p)
    penalty = np.full(p, l2)
    penalty[0] = 0.0
    for _ in range(MAX_ITERATIONS):
        prob = sigmoid(x @ beta)
        gradient = x.T @ (prob - y) / n + penalty * beta
        weights = prob * (1.0 - prob)
        hessian = (x * weights[:, None]).T @ x / n + np.diag(penalty)
        step = np.linalg.solve(hessian, gradient)
        beta = beta - step
        if np.max(np.abs(step)) < TOLERANCE:
            break
    return beta


@dataclass(frozen=True)
class SplineModel:
    spec: SplineSpec
    beta: FloatArray
    n_knots: int
    l2: float

    def predict(self, shots: pd.DataFrame) -> FloatArray:
        return sigmoid(design(shots, self.spec) @ self.beta)


def fit_spline(shots: pd.DataFrame, n_knots: int, l2: float) -> SplineModel:
    spec = make_spec(shots, n_knots)
    y = shots["made"].to_numpy(dtype=np.float64)
    return SplineModel(spec, fit_logistic(design(shots, spec), y, l2), n_knots, l2)


def has_identity_column(names: tuple[str, ...]) -> bool:
    return any(part in name for name in names for part in IDENTITY_COLUMNS)


def fit_isotonic(p: FloatArray, y: FloatArray) -> Callable[[FloatArray], FloatArray]:
    """Isotonic (non-decreasing) regression of y on p by pool-adjacent-violators; the returned
    map interpolates linearly between the fitted points and is flat beyond them."""
    xs, first = np.unique(p, return_inverse=True)
    sums = np.bincount(first, weights=y).astype(np.float64)
    counts = np.bincount(first).astype(np.float64)
    block_sum: list[float] = []
    block_n: list[float] = []
    block_len: list[int] = []
    for s, n in zip(sums, counts, strict=True):
        block_sum.append(float(s))
        block_n.append(float(n))
        block_len.append(1)
        while len(block_sum) > 1 and (block_sum[-2] / block_n[-2] >= block_sum[-1] / block_n[-1]):
            last_sum, last_n, last_len = block_sum.pop(), block_n.pop(), block_len.pop()
            block_sum[-1] += last_sum
            block_n[-1] += last_n
            block_len[-1] += last_len
    fitted = np.repeat(np.array(block_sum) / np.array(block_n), np.array(block_len, dtype=np.int64))

    def apply(q: FloatArray) -> FloatArray:
        out: FloatArray = np.interp(q, xs, fitted)
        return out

    return apply
