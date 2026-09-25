"""Proper scoring rules, calibration and a paired bootstrap for per-game loss differences."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from eurohoops.models.elo import FloatArray


@dataclass(frozen=True)
class Metrics:
    n: int
    log_loss: float | None
    brier: float | None
    accuracy: float | None
    margin_mae: float | None
    ece: float | None
    reliability: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready, rounded to 6 decimals so reports diff cleanly."""
        return {
            "n": self.n,
            "log_loss": _round(self.log_loss),
            "brier": _round(self.brier),
            "accuracy": _round(self.accuracy),
            "margin_mae": _round(self.margin_mae),
            "ece": _round(self.ece),
            "reliability": self.reliability,
        }


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 6)


def per_game_log_loss(p_home: FloatArray, home_won: FloatArray) -> FloatArray:
    loss: FloatArray = -(home_won * np.log(p_home) + (1.0 - home_won) * np.log(1.0 - p_home))
    return loss


def _bin_index(p: FloatArray, bins: int) -> npt.NDArray[np.int64]:
    """Equal-width bins [i/bins, (i+1)/bins); p = 1 falls in the last bin."""
    return np.minimum(np.floor(p * bins), bins - 1).astype(np.int64)


def reliability(p: FloatArray, y: FloatArray, bins: int = 10) -> list[dict[str, Any]]:
    """Per bin: edges, number of games, mean predicted P and observed rate (null when empty)."""
    idx = _bin_index(p, bins)
    rows = []
    for b in range(bins):
        in_bin = idx == b
        n = int(in_bin.sum())
        rows.append(
            {
                "low": round(b / bins, 6),
                "high": round((b + 1) / bins, 6),
                "n": n,
                "mean_p": _round(float(p[in_bin].mean())) if n else None,
                "observed": _round(float(y[in_bin].mean())) if n else None,
            }
        )
    return rows


def ece(p: FloatArray, y: FloatArray, bins: int = 10) -> float:
    """Expected calibration error: |mean P - observed rate| per bin, weighted by bin count."""
    idx = _bin_index(p, bins)
    total = 0.0
    for b in np.unique(idx):
        in_bin = idx == b
        total += in_bin.sum() * abs(float(p[in_bin].mean()) - float(y[in_bin].mean()))
    return total / len(p)


def score(p_home: FloatArray, exp_margin: FloatArray, margin: FloatArray) -> Metrics:
    """Accuracy counts p_home >= 0.5 as a home pick. Empty input gives n=0 and null metrics."""
    n = len(p_home)
    if n == 0:
        return Metrics(
            n=0,
            log_loss=None,
            brier=None,
            accuracy=None,
            margin_mae=None,
            ece=None,
            reliability=reliability(p_home, p_home),
        )
    home_won = (margin > 0).astype(np.float64)
    return Metrics(
        n=n,
        log_loss=float(per_game_log_loss(p_home, home_won).mean()),
        brier=float(np.mean((p_home - home_won) ** 2)),
        accuracy=float(np.mean((p_home >= 0.5) == (home_won == 1.0))),
        margin_mae=float(np.mean(np.abs(exp_margin - margin))),
        ece=ece(p_home, home_won),
        reliability=reliability(p_home, home_won),
    )


def paired_bootstrap_ci(diff: FloatArray, resamples: int, seed: int) -> tuple[float, float, float]:
    """Mean of ``diff`` and the 95% percentile CI of its mean over paired resamples."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(resamples, len(diff)))
    means = diff[idx].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(diff.mean()), float(low), float(high)
