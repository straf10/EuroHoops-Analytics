"""Shot-level scoring for M2: log loss, Brier, equal-count ECE and reliability (F-f), and the
game-level (cluster) bootstrap every M2 interval uses (F-g)."""

from typing import Any

import numpy as np
import numpy.typing as npt

from eurohoops.models.elo import FloatArray

ECE_BINS = 20
RELIABILITY_MIN_SHOTS = 500
RELIABILITY_TOLERANCE = 0.02
ECE_TARGET = 0.010
PROB_FLOOR = 1e-6


def shot_log_loss(p: FloatArray, y: FloatArray) -> FloatArray:
    """Per-shot log loss; p is clipped to [1e-6, 1 - 1e-6] (an isotonic step can reach 0 or 1)."""
    q = np.clip(p, PROB_FLOOR, 1.0 - PROB_FLOOR)
    loss: FloatArray = -(y * np.log(q) + (1.0 - y) * np.log(1.0 - q))
    return loss


def equal_count_bins(p: FloatArray, bins: int = ECE_BINS) -> npt.NDArray[np.int64]:
    """Bin index of each shot: ``bins`` groups of (nearly) equal size by sorted P (stable ties)."""
    order = np.argsort(p, kind="stable")
    index = np.empty(len(p), dtype=np.int64)
    index[order] = np.arange(len(p)) * bins // max(len(p), 1)
    return index


def reliability(p: FloatArray, y: FloatArray, bins: int = ECE_BINS) -> list[dict[str, Any]]:
    idx = equal_count_bins(p, bins)
    rows = []
    for b in range(bins):
        mask = idx == b
        n = int(mask.sum())
        if not n:
            continue
        mean_p, observed = float(p[mask].mean()), float(y[mask].mean())
        rows.append(
            {
                "bin": b,
                "n": n,
                "p_low": round(float(p[mask].min()), 6),
                "p_high": round(float(p[mask].max()), 6),
                "mean_p": round(mean_p, 6),
                "observed": round(observed, 6),
                "gap": round(observed - mean_p, 6),
            }
        )
    return rows


def ece(p: FloatArray, y: FloatArray, bins: int = ECE_BINS) -> float:
    idx = equal_count_bins(p, bins)
    total = 0.0
    for b in range(bins):
        mask = idx == b
        if mask.any():
            total += mask.sum() * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return total / len(p) if len(p) else float("nan")


def scores(p: FloatArray, y: FloatArray) -> dict[str, Any]:
    """Log loss, Brier, ECE and the reliability bins; rounded to 6 decimals."""
    if not len(p):
        return {"n": 0, "log_loss": None, "brier": None, "ece": None, "reliability": []}
    return {
        "n": len(p),
        "log_loss": round(float(shot_log_loss(p, y).mean()), 6),
        "brier": round(float(np.mean((p - y) ** 2)), 6),
        "mean_p": round(float(p.mean()), 6),
        "observed": round(float(y.mean()), 6),
        "ece": round(ece(p, y), 6),
        "reliability": reliability(p, y),
    }


def calibrated(block: dict[str, Any]) -> bool:
    """F-f: ECE <= 0.010 and every bin with >= 500 shots within +-0.02."""
    bins_ok = all(
        abs(r["gap"]) <= RELIABILITY_TOLERANCE
        for r in block["reliability"]
        if r["n"] >= RELIABILITY_MIN_SHOTS
    )
    return bool(block["ece"] is not None and block["ece"] <= ECE_TARGET and bins_ok)


def cluster_bootstrap(
    values: FloatArray, clusters: npt.NDArray[Any], resamples: int, seed: int, level: float = 0.95
) -> tuple[float, float, float]:
    """Shot-weighted mean of ``values`` and its percentile CI, resampling whole clusters (games).

    Each resample draws as many games as there are, with replacement; its statistic is the
    sum of the drawn games' values over the number of their shots.
    """
    codes, inverse = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inverse, weights=values, minlength=len(codes))
    counts = np.bincount(inverse, minlength=len(codes)).astype(np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(codes), size=(resamples, len(codes)))
    stats = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    tail = (1.0 - level) / 2.0 * 100.0
    low, high = np.percentile(stats, [tail, 100.0 - tail])
    return float(values.mean()), float(low), float(high)
