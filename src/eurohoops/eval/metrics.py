"""Proper scoring rules and a paired bootstrap for per-game loss differences."""

from dataclasses import asdict, dataclass

import numpy as np

from eurohoops.models.elo import FloatArray


@dataclass(frozen=True)
class Metrics:
    n: int
    log_loss: float | None
    brier: float | None
    accuracy: float | None
    margin_mae: float | None

    def as_dict(self) -> dict[str, float | int | None]:
        """JSON-ready, rounded to 6 decimals so reports diff cleanly."""
        return {k: v if v is None or k == "n" else round(v, 6) for k, v in asdict(self).items()}


def per_game_log_loss(p_home: FloatArray, home_won: FloatArray) -> FloatArray:
    loss: FloatArray = -(home_won * np.log(p_home) + (1.0 - home_won) * np.log(1.0 - p_home))
    return loss


def score(p_home: FloatArray, exp_margin: FloatArray, margin: FloatArray) -> Metrics:
    """Accuracy counts p_home >= 0.5 as a home pick. Empty input gives n=0 and null metrics."""
    n = len(p_home)
    if n == 0:
        return Metrics(n=0, log_loss=None, brier=None, accuracy=None, margin_mae=None)
    home_won = (margin > 0).astype(np.float64)
    return Metrics(
        n=n,
        log_loss=float(per_game_log_loss(p_home, home_won).mean()),
        brier=float(np.mean((p_home - home_won) ** 2)),
        accuracy=float(np.mean((p_home >= 0.5) == (home_won == 1.0))),
        margin_mae=float(np.mean(np.abs(exp_margin - margin))),
    )


def paired_bootstrap_ci(diff: FloatArray, resamples: int, seed: int) -> tuple[float, float, float]:
    """Mean of ``diff`` and the 95% percentile CI of its mean over paired resamples."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(resamples, len(diff)))
    means = diff[idx].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(diff.mean()), float(low), float(high)
