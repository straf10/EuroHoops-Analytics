"""Weeks 7-10b G5: the season-level offset. The MLE shift, the empirical-Bayes weight, the
causal offset (no shot of game g or later reaches game g's offset), other seasons reaching a
season's offsets only through the declared prior and k, and the weighted ECE bootstrap."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from eurohoops.eval.shot_metrics import ece, ece_diff_bootstrap, weighted_ece
from eurohoops.models import season_level
from eurohoops.models.season_level import (
    Shrinkage,
    fit_shrinkage,
    level_predictions,
    logit,
    mle_shift,
    season_offsets,
)
from eurohoops.models.xpts import sigmoid


def _season(n: int, shift: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Base P(make), outcomes drawn with the given logit shift, and tip-offs (30 games)."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.2, 0.8, n)
    y = (rng.random(n) < sigmoid(logit(p) + shift)).astype(np.float64)
    tipoff = np.sort(rng.integers(0, 30, n)).astype(np.int64)
    return p, y, tipoff


def test_the_mle_shift_recovers_a_known_level() -> None:
    p, y, _ = _season(200_000, 0.3, 1)
    assert abs(mle_shift(logit(p), y) - 0.3) < 0.02
    assert mle_shift(logit(p[:0]), y[:0]) == 0.0
    d = mle_shift(logit(p), y)
    assert abs(float((y - sigmoid(logit(p) + d)).sum())) < 1e-6  # the score equation holds


def test_the_weight_is_n_over_n_plus_k() -> None:
    shrink = Shrinkage(k=1000.0, tau2=0.01, v_bar=0.1, pairs=3)
    assert shrink.weight(0) == 0.0 and shrink.weight(1000) == 0.5
    assert Shrinkage(math.inf, -0.1, 0.2, 3).weight(10_000) == 0.0
    assert Shrinkage(math.inf, -0.1, 0.2, 3).to_json()["k"] is None


def test_k_follows_the_season_to_season_variance() -> None:
    rng = np.random.default_rng(3)
    levels = np.cumsum(rng.normal(0.0, 0.1, 30))  # a random walk: tau = 0.1
    seasons = {2000 + t: _season(20_000, levels[t], 10 + t)[:2] for t in range(30)}
    shrink = fit_shrinkage(seasons)
    assert shrink.pairs == 29
    assert 0.005 < shrink.tau2 < 0.02
    assert shrink.k == pytest.approx(1.0 / (shrink.v_bar * shrink.tau2))
    flat = {2000 + t: _season(20_000, 0.0, 50 + t)[:2] for t in range(6)}
    assert math.isinf(fit_shrinkage(flat).k)  # no level change beyond noise: the prior alone


def test_an_offset_uses_only_games_that_tipped_off_earlier() -> None:
    p, y, tipoff = _season(6_000, 0.4, 5)
    shrink = Shrinkage(k=500.0, tau2=0.01, v_bar=0.2, pairs=5)
    base = season_offsets(p, y, tipoff, 0.1, shrink)
    assert np.all(base[tipoff == 0] == 0.1)  # the first tip-off: the prior alone
    for g in (1, 12, 29):
        edited = np.where(tipoff >= g, 1.0 - y, y)  # every shot of game g and later flipped
        moved = season_offsets(p, edited, tipoff, 0.1, shrink)
        np.testing.assert_array_equal(moved[tipoff <= g], base[tipoff <= g])
        if g < 29:  # guard: the edit reaches the later games' offsets
            assert not np.array_equal(moved[tipoff > g], base[tipoff > g])
    # one offset per game, and shots of a later tip-off see more of the season
    assert all(len(set(base[tipoff == t])) == 1 for t in range(30))


def _folds(n: int = 3_000) -> tuple[SimpleNamespace, np.ndarray, np.ndarray, list[int]]:
    """Fixed base predictions for four development seasons (a Folds stand-in)."""
    development = [2011, 2012, 2013, 2014]
    rng = np.random.default_rng(9)
    season = np.repeat(development, n)
    p = rng.uniform(0.3, 0.7, len(season))
    shift = dict(zip(development, (0.2, -0.1, 0.15, -0.2), strict=True))
    y = (rng.random(len(season)) < sigmoid(logit(p) + np.vectorize(shift.get)(season))).astype(
        np.float64
    )
    index = {s: np.flatnonzero(season == s) for s in development}
    pair = {
        (s, t): np.clip(p[index[t]] + 0.01 * (s - 2011), 0.01, 0.99)
        for s in development
        for t in development
        if s != t
    }
    folds = SimpleNamespace(index=index, oof=p, pair=pair, later=np.empty(0))
    tip = (np.arange(len(season)) % 25).astype(np.int64)
    return folds, y, tip, development


def _offsets(folds: SimpleNamespace, y: np.ndarray, tip: np.ndarray, dev: list[int]) -> np.ndarray:
    empty = np.empty(0, dtype=np.int64)
    return level_predictions(
        folds, y, tip, later_season=empty, y_later=np.empty(0), tip_later=empty, development=dev
    ).offsets_oof


def test_other_seasons_reach_an_offset_only_through_the_prior_and_k(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    folds, y, tip, dev = _folds()
    s, rows = 2013, folds.index[2013]
    flip = lambda t: np.where(np.isin(np.arange(len(y)), folds.index[t]), 1.0 - y, y)  # noqa: E731
    base = _offsets(folds, y, tip, dev)
    # guard: without holding k, a far season (2011) reaches season 2013 through k
    assert not np.array_equal(_offsets(folds, flip(2011), tip, dev)[rows], base[rows])
    fixed = Shrinkage(k=800.0, tau2=0.01, v_bar=0.2, pairs=2)
    monkeypatch.setattr(season_level, "fit_shrinkage", lambda _: fixed)
    held = _offsets(folds, y, tip, dev)
    far = _offsets(folds, flip(2011), tip, dev)
    np.testing.assert_array_equal(far[rows], held[rows])  # k held: 2011 does not reach 2013
    previous = _offsets(folds, flip(s - 1), tip, dev)
    assert not np.array_equal(previous[rows], held[rows])  # 2012 reaches it: the prior


def test_the_weighted_ece_is_the_ece_with_unit_weights() -> None:
    p, y, _ = _season(5_000, 0.0, 7)
    order = np.argsort(p, kind="stable")
    assert weighted_ece(p, y, np.ones(len(p)), order) == pytest.approx(ece(p, y), abs=1e-12)
    games = np.arange(len(p)) // 20
    diff, low, high = ece_diff_bootstrap(
        p, np.clip(p + 0.05, 0, 1), y, games, resamples=200, seed=1
    )
    assert low <= diff <= high
    again = ece_diff_bootstrap(p, np.clip(p + 0.05, 0, 1), y, games, resamples=200, seed=1)
    assert again == (diff, low, high)
