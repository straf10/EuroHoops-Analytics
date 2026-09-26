"""Season-level variants of M2 (weeks 7-10b G5): ``lgbm_level`` and ``spline_level``.

Declared before any run in reports/week7-10b_progress.md (LEVEL_DECLARATION). A base model
fitted on other seasons cannot know a new season's shooting level, so its logit gets a season
offset estimated only from earlier games of that season, shrunk toward the previous season's
level:

    p_level = sigmoid(logit(p_base) + o),   o = w * d + (1 - w) * prior,   w = n / (n + k)

For a shot in game g, d is the maximum-likelihood intercept shift over the season's shots in
games that tipped off strictly before g (n of them; none -> w = 0), and prior is the previous
season's full-season shift, with base predictions from a model trained on neither season.
k = 1 / (v_bar * tau^2) by empirical Bayes: tau^2 is the season-to-season variance of the
full-season shift (net of its sampling variance 1/I, I = sum p(1 - p)) and v_bar the mean
p(1 - p) (so w is the posterior weight of n shots' evidence against a prior of variance tau^2).
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from eurohoops.models.elo import FloatArray
from eurohoops.models.xpts import sigmoid

P_CLIP = 1e-6
NEWTON_ITERATIONS = 100
NEWTON_TOLERANCE = 1e-12
MAX_STEP = 1.0  # a Newton step is capped (a to-date sample of all makes has no finite MLE)
MAX_SHIFT = 10.0


def logit(p: FloatArray) -> FloatArray:
    q = np.clip(p, P_CLIP, 1.0 - P_CLIP)
    out: FloatArray = np.log(q / (1.0 - q))
    return out


def mle_shift(logits: FloatArray, y: FloatArray, start: float = 0.0) -> float:
    """The intercept shift d solving sum(y - sigmoid(logits + d)) = 0 (Newton); 0 for no shots."""
    if not len(logits):
        return 0.0
    d = start
    for _ in range(NEWTON_ITERATIONS):
        p = sigmoid(logits + d)
        step = float((y - p).sum() / max(float((p * (1.0 - p)).sum()), 1e-12))
        step = min(max(step, -MAX_STEP), MAX_STEP)
        d = min(max(d + step, -MAX_SHIFT), MAX_SHIFT)
        if abs(step) < NEWTON_TOLERANCE:
            break
    return d


@dataclass(frozen=True)
class Shrinkage:
    """The empirical-Bayes weight constant k and what it came from."""

    k: float  # inf when tau^2 <= 0: the prior alone
    tau2: float
    v_bar: float
    pairs: int

    def weight(self, n: int) -> float:
        return 0.0 if not n or math.isinf(self.k) else n / (n + self.k)

    def to_json(self) -> dict[str, Any]:
        return {
            "k": None if math.isinf(self.k) else round(self.k, 3),
            "tau2": round(self.tau2, 8),
            "v_bar": round(self.v_bar, 6),
            "consecutive_pairs": self.pairs,
        }


def fit_shrinkage(season_preds: dict[int, tuple[FloatArray, FloatArray]]) -> Shrinkage:
    """k from full-season shifts of the given seasons ({season: (p_base, y)}): tau^2 over
    consecutive seasons both present, net of the shifts' sampling variance."""
    shift = {t: mle_shift(logit(p), y) for t, (p, y) in season_preds.items()}
    info = {t: float((p * (1.0 - p)).sum()) for t, (p, _) in season_preds.items()}
    pairs = [(t - 1, t) for t in sorted(shift) if t - 1 in shift]
    all_p = np.concatenate([p for p, _ in season_preds.values()])
    v_bar = float((all_p * (1.0 - all_p)).mean())
    if not pairs:
        return Shrinkage(math.inf, 0.0, v_bar, 0)
    jumps = float(np.mean([(shift[b] - shift[a]) ** 2 for a, b in pairs]))
    noise = float(np.mean([1.0 / info[a] + 1.0 / info[b] for a, b in pairs]))
    tau2 = jumps - noise
    k = 1.0 / (v_bar * tau2) if tau2 > 0 else math.inf
    return Shrinkage(k, tau2, v_bar, len(pairs))


def season_offsets(
    p_base: FloatArray,
    y: FloatArray,
    tipoff: npt.NDArray[np.int64],
    prior: float,
    shrink: Shrinkage,
) -> FloatArray:
    """The offset of every shot of one season: games at one tip-off time share it, and it uses
    only shots of games that tipped off strictly earlier."""
    order = np.argsort(tipoff, kind="stable")
    times = tipoff[order]
    starts = np.flatnonzero(np.r_[True, times[1:] != times[:-1]])
    ends = np.r_[starts[1:], len(times)]
    logits, labels = logit(p_base)[order], y[order]
    out = np.empty(len(p_base))
    d = 0.0
    for a, b in zip(starts, ends, strict=True):
        n = int(a)
        if n:
            d = mle_shift(logits[:n], labels[:n], d)
        w = shrink.weight(n)
        out[order[a:b]] = w * d + (1.0 - w) * prior
    return out


def shifted(p_base: FloatArray, offsets: FloatArray) -> FloatArray:
    return sigmoid(logit(p_base) + offsets)


class FoldPredictions(Protocol):
    """What ``m2_backtest.Folds`` holds: LOSO out-of-fold, leave-two-out pairs, later."""

    index: dict[int, npt.NDArray[np.intp]]
    oof: FloatArray
    pair: dict[tuple[int, int], FloatArray]
    later: FloatArray


@dataclass(frozen=True)
class LevelPredictions:
    oof: FloatArray
    later: FloatArray
    offsets_oof: FloatArray
    offsets_later: FloatArray
    shrinkage: dict[str, Any]
    priors: dict[str, float]


def level_predictions(
    folds: FoldPredictions,
    y_dev: FloatArray,
    tip_dev: npt.NDArray[np.int64],
    *,
    later_season: npt.NDArray[np.int64],
    y_later: FloatArray,
    tip_later: npt.NDArray[np.int64],
    development: Sequence[int],
) -> LevelPredictions:
    """The level variant of one base (see the module doc and the declaration).

    Development season s: base = its LOSO prediction; prior and k from the leave-two-out models
    (s, t), trained on neither s nor t. Later seasons: base = the development-fitted prediction;
    k from the development LOSO predictions; prior from the previous season's LOSO prediction
    (a development season) or development-fitted prediction (a later season)."""
    oof, off_oof = np.full(len(y_dev), np.nan), np.full(len(y_dev), np.nan)
    shrinkage: dict[str, Any] = {}
    priors: dict[str, float] = {}
    for s in development:
        others = {t: (folds.pair[(s, t)], y_dev[folds.index[t]]) for t in development if t != s}
        shrink = fit_shrinkage(others)
        prior = mle_shift(logit(others[s - 1][0]), others[s - 1][1]) if s - 1 in others else 0.0
        rows = folds.index[s]
        off_oof[rows] = season_offsets(folds.oof[rows], y_dev[rows], tip_dev[rows], prior, shrink)
        oof[rows] = shifted(folds.oof[rows], off_oof[rows])
        shrinkage[str(s)], priors[str(s)] = shrink.to_json(), round(prior, 6)
    loso = {t: (folds.oof[folds.index[t]], y_dev[folds.index[t]]) for t in development}
    shrink = fit_shrinkage(loso)
    shrinkage["later"] = shrink.to_json()
    later, off_later = np.full(len(y_later), np.nan), np.full(len(y_later), np.nan)
    for u in sorted(set(later_season.tolist())):
        before = later_season == u - 1
        if u - 1 in loso:
            prior = mle_shift(logit(loso[u - 1][0]), loso[u - 1][1])
        elif before.any():
            prior = mle_shift(logit(folds.later[before]), y_later[before])
        else:
            prior = 0.0
        rows = np.flatnonzero(later_season == u)
        off_later[rows] = season_offsets(
            folds.later[rows], y_later[rows], tip_later[rows], prior, shrink
        )
        later[rows] = shifted(folds.later[rows], off_later[rows])
        priors[str(u)] = round(prior, 6)
    return LevelPredictions(oof, later, off_oof, off_later, shrinkage, priors)
