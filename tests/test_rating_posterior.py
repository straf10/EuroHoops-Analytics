"""K1: M1's rating posterior (Gaussian over [μ, h, off, def]) and pace point, per cutoff."""

import math
from dataclasses import replace

import numpy as np
import pytest
from scipy import stats

from eurohoops.models.elo import FloatArray
from eurohoops.models.team_eff import (
    SECONDS_PER_DAY,
    DecayedRidge,
    DecayParams,
    History,
    RatingPosterior,
    _cutoffs,
    forecast,
    pace_fits,
    pace_penalty,
    pace_points,
    prepare_history,
    rating_design,
    rating_fits,
    rating_penalty,
    rating_posteriors,
)
from tests.test_team_eff import league

RATING = DecayParams(half_life_days=90.0, carry=0.6, ridge=300.0)
PACE = DecayParams(half_life_days=60.0, carry=1.0, ridge=2.0)


def reference_history() -> History:
    """Four teams over 2020-2021 with noisy ratings; DDD only appears in 2021, one 2021 game is
    at a neutral venue and the last round of 2021 is unplayed (forecast only)."""
    ratings = {
        "AAA": (4.0, 1.0, 1.5),
        "BBB": (0.0, -2.0, -1.0),
        "CCC": (-3.0, 1.0, 0.5),
        "DDD": (1.0, 2.0, -0.5),
    }
    games, rows = league(ratings, [2020, 2021], noise=7.0, seed=11, unplayed_last_round=True)
    ddd = (games["home"] == "DDD") | (games["away"] == "DDD")
    early = games["season"] == 2020
    games = games[~(ddd & early)].reset_index(drop=True)
    rows = rows[rows["game_id"].isin(games["game_id"])]
    neutral = games.index[games["season"] == 2021][3]
    games.loc[neutral, "neutral"] = True
    return prepare_history(games, rows)


def _old_rating_fits(history: History, params: DecayParams) -> tuple[FloatArray, FloatArray]:
    """``rating_fits`` as it was before the K1 refactor (291b312), verbatim."""
    n = len(history.teams)
    model = DecayedRidge(rating_penalty(n, params.ridge), params)
    columns, values = rating_design(history)
    home_ortg = np.full(len(history.home), np.nan)
    away_ortg = np.full(len(history.home), np.nan)
    added = 0
    for time, season, games in _cutoffs(history):
        model.advance(time, season)
        stop = int(np.searchsorted(history.row_time, time, side="left"))
        if stop > added:
            span = slice(added, stop)
            model.add(
                columns[span],
                values[span],
                target=history.row_ortg[span],
                weight=history.row_poss[span],
                time=history.row_time[span],
                season=history.row_season[span],
            )
            added = stop
        if not added:
            continue
        theta = model.solve()
        mu, h = theta[0], theta[1]
        off, dfn = theta[2 : 2 + n], theta[2 + n :]
        hf = history.home_flag[games]
        home, away = history.home[games], history.away[games]
        home_ortg[games] = mu + h * hf + off[home] - dfn[away]
        away_ortg[games] = mu - h * hf + off[away] - dfn[home]
    return home_ortg, away_ortg


def _old_pace_fits(history: History, params: DecayParams) -> FloatArray:
    """``pace_fits`` as it was before the K1 refactor (291b312), verbatim."""
    n = len(history.teams)
    model = DecayedRidge(pace_penalty(n, params.ridge), params)
    columns = np.stack(
        [np.zeros_like(history.pace_home), 1 + history.pace_home, 1 + history.pace_away], axis=1
    )
    values = np.ones(columns.shape, dtype=np.float64)
    weight = np.ones(len(history.pace_home))
    pace = np.full(len(history.home), np.nan)
    added = 0
    for time, season, games in _cutoffs(history):
        model.advance(time, season)
        stop = int(np.searchsorted(history.pace_time, time, side="left"))
        if stop > added:
            span = slice(added, stop)
            model.add(
                columns[span],
                values[span],
                target=history.pace_poss40[span],
                weight=weight[span],
                time=history.pace_time[span],
                season=history.pace_season[span],
            )
            added = stop
        if not added:
            continue
        theta = model.solve()
        pace[games] = theta[0] + theta[1 + history.home[games]] + theta[1 + history.away[games]]
    return pace


def test_m1_fits_are_byte_identical_to_the_pre_refactor_code() -> None:
    """The K1 refactor (a shared walk-forward generator) changes no M1 output, bit for bit.

    The reference is the pre-refactor code run on the same machine: a stored fixture would only
    hold one platform's floating-point rounding (Windows and Linux BLAS/libm differ in the last
    bits), so it cannot be compared exactly elsewhere.
    """
    history = reference_history()
    old_home, old_away = _old_rating_fits(history, RATING)
    old_pace = _old_pace_fits(history, PACE)
    home, away = rating_fits(history, RATING)
    got = forecast(history, RATING, PACE)
    pairs = {
        "home_ortg": (home, old_home),
        "away_ortg": (away, old_away),
        "pace_fits": (pace_fits(history, PACE), old_pace),
        "home_points": (got.home_points, old_pace * old_home / 100.0),
        "away_points": (got.away_points, old_pace * old_away / 100.0),
    }
    for name, (new, old) in pairs.items():
        assert np.array_equal(new, old, equal_nan=True), name
    assert np.isfinite(home).sum() > 10  # the comparison covers real forecasts, not only NaN


def round_cutoffs(history: History) -> list[tuple[float, int]]:
    return sorted({(float(t), int(s)) for t, s in zip(history.cutoff, history.season, strict=True)})


def implied_ortg(post: RatingPosterior, home: str, away: str, hf: float) -> tuple[float, float]:
    """μ + h·hf + off[H] - def[A] (and the away side) from the posterior mean."""
    theta = dict(zip(post.labels, post.mean, strict=True))
    base = theta["mu"]
    off_home, off_away = theta.get(f"off:{home}", 0.0), theta.get(f"off:{away}", 0.0)
    def_home, def_away = theta.get(f"def:{home}", 0.0), theta.get(f"def:{away}", 0.0)
    return (
        base + theta["h"] * hf + off_home - def_away,
        base - theta["h"] * hf + off_away - def_home,
    )


def test_mean_equals_the_ratings_rating_fits_uses_at_every_round_cutoff() -> None:
    history = reference_history()
    cutoffs = round_cutoffs(history)
    assert {s for _, s in cutoffs} == {2020, 2021}
    posts = rating_posteriors(history, RATING, cutoffs)
    home_ortg, away_ortg = rating_fits(history, RATING)
    checked = 0
    for (time, season), post in zip(cutoffs, posts, strict=True):
        for g in np.flatnonzero((history.cutoff == time) & (history.season == season)):
            if np.isnan(home_ortg[g]):
                assert len(post.columns) == 0
                continue
            home, away = history.teams[history.home[g]], history.teams[history.away[g]]
            got = implied_ortg(post, home, away, float(history.home_flag[g]))
            assert abs(got[0] - home_ortg[g]) < 1e-9
            assert abs(got[1] - away_ortg[g]) < 1e-9
            checked += 1
    assert checked == np.count_nonzero(~np.isnan(home_ortg))


def test_a_cutoff_between_rounds_matches_a_hand_built_decayed_ridge() -> None:
    history = reference_history()
    times = np.unique(history.row_time[history.row_season == 2021])
    cutoff, season = float((times[-3] + times[-2]) / 2.0), 2021
    n = len(history.teams)
    use = np.flatnonzero(history.row_time < cutoff)
    assert 0 < len(use) < len(history.row_time)
    columns, values = rating_design(history)
    model = DecayedRidge(rating_penalty(n, RATING.ridge), RATING)
    model.advance(cutoff, season)
    model.add(
        columns[use],
        values[use],
        target=history.row_ortg[use],
        weight=history.row_poss[use],
        time=history.row_time[use],
        season=history.row_season[use],
    )
    theta = model.solve()
    seen = np.flatnonzero(model.seen)
    (post,) = rating_posteriors(history, RATING, [(cutoff, season)])
    assert np.array_equal(post.columns, seen)
    assert np.max(np.abs(post.mean - theta[seen])) < 1e-9
    # σ² and the precision straight from the definitions
    decay = np.array(
        [
            0.5 ** ((cutoff - history.row_time[i]) / SECONDS_PER_DAY / RATING.half_life_days)
            * RATING.carry ** (season - history.row_season[i])
            for i in use
        ]
    )
    x = np.zeros((len(use), 2 + 2 * n))
    for r, i in enumerate(use):
        x[r, columns[i]] = values[i]
    resid = history.row_ortg[use] - x @ theta
    weighted = x.T @ ((decay * history.row_poss[use])[:, None] * x)
    precision = weighted + np.diag(rating_penalty(n, RATING.ridge))
    p_eff = np.trace(np.linalg.solve(precision[np.ix_(seen, seen)], weighted[np.ix_(seen, seen)]))
    sigma2 = float(np.sum(decay * history.row_poss[use] * resid**2) / (np.sum(decay) - p_eff))
    assert post.sigma2 == pytest.approx(sigma2, rel=1e-9)
    expected = sigma2 * np.linalg.inv(precision[np.ix_(seen, seen)])
    assert np.max(np.abs(post.cov - expected)) < 1e-9 * np.max(np.abs(expected))


def test_labels_columns_and_covariance_shape() -> None:
    history = reference_history()
    n = len(history.teams)
    early = float(history.row_time[6])  # 2020: DDD has not played yet
    posts = rating_posteriors(history, RATING, [(early, 2020), (float(history.cutoff[-1]), 2021)])
    for post, ddd_seen in zip(posts, (False, True), strict=True):
        k = len(post.columns)
        assert post.teams == history.teams
        assert np.all(np.diff(post.columns) > 0)
        assert len(post.labels) == k == len(post.mean) == len(post.cov)
        for column, label in zip(post.columns, post.labels, strict=True):
            if column < 2:
                assert label == ("mu", "h")[column]
            elif column < 2 + n:
                assert label == f"off:{history.teams[column - 2]}"
            else:
                assert label == f"def:{history.teams[column - 2 - n]}"
        assert ("off:DDD" in post.labels) is ddd_seen
        assert ("def:DDD" in post.labels) is ddd_seen
        assert np.array_equal(post.cov, post.cov.T)
        np.linalg.cholesky(post.cov)  # raises unless positive definite
        assert post.sigma2 > 0


def test_a_cutoff_before_any_row_is_an_empty_posterior() -> None:
    history = reference_history()
    (post,) = rating_posteriors(history, RATING, [(float(history.row_time[0]), 2020)])
    assert post.columns.size == 0 and post.labels == () and post.mean.size == 0
    assert post.cov.shape == (0, 0) and math.isnan(post.sigma2)
    (point,) = pace_points(history, PACE, [(float(history.pace_time[0]), 2020)])
    assert np.array_equal(point, np.zeros(1 + len(history.teams)))


def test_pace_points_give_the_pace_that_pace_fits_uses() -> None:
    history = reference_history()
    cutoffs = round_cutoffs(history)
    points = pace_points(history, PACE, cutoffs)
    fits = pace_fits(history, PACE)
    checked = 0
    for (time, season), point in zip(cutoffs, points, strict=True):
        assert point.shape == (1 + len(history.teams),)
        for g in np.flatnonzero((history.cutoff == time) & (history.season == season)):
            if not np.isnan(fits[g]):
                got = point[0] + point[1 + history.home[g]] + point[1 + history.away[g]]
                assert abs(got - fits[g]) < 1e-9
                checked += 1
    assert checked == np.count_nonzero(~np.isnan(fits))


def test_the_order_of_the_cutoffs_does_not_matter() -> None:
    history = reference_history()
    cutoffs = round_cutoffs(history)
    between = (cutoffs[2][0] + cutoffs[3][0]) / 2.0
    cutoffs = [*cutoffs, (between, cutoffs[2][1]), cutoffs[1]]  # a repeated cutoff too
    shuffled = [cutoffs[i] for i in np.random.default_rng(5).permutation(len(cutoffs))]
    forward = rating_posteriors(history, RATING, cutoffs)
    other = rating_posteriors(history, RATING, shuffled)
    pace_forward = pace_points(history, PACE, cutoffs)
    pace_other = pace_points(history, PACE, shuffled)
    for i, cutoff in enumerate(shuffled):
        j = cutoffs.index(cutoff)
        assert np.array_equal(other[i].mean, forward[j].mean)
        assert np.array_equal(other[i].cov, forward[j].cov, equal_nan=True)
        assert np.array_equal([other[i].sigma2], [forward[j].sigma2], equal_nan=True)
        assert np.array_equal(pace_other[i], pace_forward[j])


def truncated(history: History, cutoff: float) -> History:
    """The history with every rating and pace row at or after ``cutoff`` deleted."""
    keep = history.row_time < cutoff
    pace = history.pace_time < cutoff
    return replace(
        history,
        row_time=history.row_time[keep],
        row_season=history.row_season[keep],
        row_team=history.row_team[keep],
        row_opp=history.row_opp[keep],
        row_home=history.row_home[keep],
        row_ortg=history.row_ortg[keep],
        row_poss=history.row_poss[keep],
        pace_time=history.pace_time[pace],
        pace_season=history.pace_season[pace],
        pace_home=history.pace_home[pace],
        pace_away=history.pace_away[pace],
        pace_poss40=history.pace_poss40[pace],
    )


def snapshot(history: History, cutoff: tuple[float, int]) -> tuple[RatingPosterior, FloatArray]:
    (post,) = rating_posteriors(history, RATING, [cutoff])
    (point,) = pace_points(history, PACE, [cutoff])
    return post, point


def same(a: tuple[RatingPosterior, FloatArray], b: tuple[RatingPosterior, FloatArray]) -> bool:
    (p, x), (q, y) = a, b
    return (
        np.array_equal(p.columns, q.columns)
        and p.labels == q.labels
        and np.array_equal(p.mean, q.mean)
        and np.array_equal(p.cov, q.cov)
        and p.sigma2 == q.sigma2
        and np.array_equal(x, y)
    )


def test_rows_at_or_after_the_cutoff_change_nothing() -> None:
    history = reference_history()
    first = int(np.searchsorted(history.row_time, history.row_time[6], side="left"))
    cutoff = (float(history.row_time[first]), 2020)  # a row tips off exactly at the cutoff
    assert 0 < first < len(history.row_time) - 4
    base = snapshot(history, cutoff)
    # perturbed values
    ortg = np.where(history.row_time >= cutoff[0], history.row_ortg + 37.0, history.row_ortg)
    poss40 = np.where(
        history.pace_time >= cutoff[0], history.pace_poss40 * 1.3, history.pace_poss40
    )
    assert same(snapshot(replace(history, row_ortg=ortg, pace_poss40=poss40), cutoff), base)
    # deleted
    assert same(snapshot(truncated(history, cutoff[0]), cutoff), base)
    # added: the last rows again, a day after the last tip-off, with the sides swapped
    last = slice(-4, None)
    day = SECONDS_PER_DAY
    added = replace(
        history,
        row_time=np.append(history.row_time, history.row_time[last] + day),
        row_season=np.append(history.row_season, history.row_season[last]),
        row_team=np.append(history.row_team, history.row_opp[last]),
        row_opp=np.append(history.row_opp, history.row_team[last]),
        row_home=np.append(history.row_home, history.row_home[last]),
        row_ortg=np.append(history.row_ortg, history.row_ortg[last] - 20.0),
        row_poss=np.append(history.row_poss, history.row_poss[last]),
        pace_time=np.append(history.pace_time, history.pace_time[last] + day),
        pace_season=np.append(history.pace_season, history.pace_season[last]),
        pace_home=np.append(history.pace_home, history.pace_away[last]),
        pace_away=np.append(history.pace_away, history.pace_home[last]),
        pace_poss40=np.append(history.pace_poss40, history.pace_poss40[last] + 9.0),
    )
    assert same(snapshot(added, cutoff), base)


def test_a_row_before_the_cutoff_does_change_the_posterior() -> None:
    history = reference_history()
    cutoff = (float(history.row_time[6]), 2020)
    base_post, base_point = snapshot(history, cutoff)
    ortg = history.row_ortg.copy()
    ortg[0] += 5.0
    moved, _ = snapshot(replace(history, row_ortg=ortg), cutoff)
    assert not np.array_equal(moved.mean, base_post.mean)
    assert moved.sigma2 != base_post.sigma2
    assert not np.array_equal(moved.cov, base_post.cov)
    poss = history.row_poss.copy()
    poss[0] += 5.0
    heavier, _ = snapshot(replace(history, row_poss=poss), cutoff)
    assert not np.array_equal(heavier.cov, base_post.cov)
    pace = history.pace_poss40.copy()
    pace[0] += 5.0
    assert not np.array_equal(snapshot(replace(history, pace_poss40=pace), cutoff)[1], base_point)


def league_history(
    rng: np.random.Generator, teams: int, sigma2: float, ridge: float
) -> tuple[History, FloatArray, FloatArray, float]:
    """A balanced double round robin whose ORtg rows follow the model exactly: the true off/def
    come from the ridge prior N(0, σ²/ridge) and each row has noise N(0, σ²/poss).

    Returns the history, the true off and def, and a cutoff after every row."""
    off = rng.normal(0.0, math.sqrt(sigma2 / ridge), teams)
    dfn = rng.normal(0.0, math.sqrt(sigma2 / ridge), teams)
    mu, edge = 108.0, 2.0
    pairs = [(h, a) for h in range(teams) for a in range(teams) if h != a]
    poss = rng.uniform(70.0, 75.0, len(pairs))
    team = np.array([t for h, a in pairs for t in (h, a)], dtype=np.int64)
    opp = np.array([t for h, a in pairs for t in (a, h)], dtype=np.int64)
    sign = np.tile([1.0, -1.0], len(pairs))
    row_poss = np.repeat(poss, 2)
    mean = mu + edge * sign + off[team] - dfn[opp]
    ortg = mean + rng.normal(0.0, 1.0, len(mean)) * np.sqrt(sigma2 / row_poss)
    time = np.repeat(np.arange(len(pairs), dtype=np.float64) * 3600.0, 2)
    zero_f, zero_i = np.zeros(0), np.zeros(0, dtype=np.int64)
    history = History(
        teams=tuple(f"T{i:02d}" for i in range(teams)),
        home=zero_i,
        away=zero_i,
        home_flag=zero_f,
        cutoff=zero_f,
        season=zero_i,
        row_time=time,
        row_season=np.full(len(time), 2020, dtype=np.int64),
        row_team=team,
        row_opp=opp,
        row_home=sign,
        row_ortg=ortg,
        row_poss=row_poss,
        pace_time=zero_f,
        pace_season=zero_i,
        pace_home=zero_i,
        pace_away=zero_i,
        pace_poss40=zero_f,
    )
    return history, off, dfn, float(time[-1] + 3600.0)


def test_ninety_percent_intervals_cover_the_truth_when_the_model_is_right() -> None:
    """200 leagues of 10 teams from the model itself (one season, no decay, truth from the prior).

    Pooled over leagues and the 20 off/def columns that is 4000 intervals; at 90% the binomial SE
    of the coverage is sqrt(.9 * .1 / 4000) = 0.47 points. The 20 intervals of a league share one
    fit and one sigma-hat, so allow a design effect of about 4 (SE 0.95 points). sigma^2 is
    estimated with the effective-degrees-of-freedom correction, so it is unbiased for a ridge
    fit and only the Normal-versus-t width effect of order p_eff / (n - p_eff) remains, a few
    tenths of a point. 87-93% is then about +-3 SE around 90%.
    """
    sigma2, ridge = 90.0**2, 250.0  # prior sd of a rating = sqrt(σ²/ridge) = 5.7 points
    params = DecayParams(half_life_days=1e9, carry=1.0, ridge=ridge)
    z90 = float(stats.norm.ppf(0.95))
    rng = np.random.default_rng(20261007)
    covered = total = 0
    sigma2_hats = []
    for _ in range(200):
        history, off, dfn, cutoff = league_history(rng, 10, sigma2, ridge)
        (post,) = rating_posteriors(history, params, [(cutoff, 2020)])
        truth = np.concatenate([[np.nan, np.nan], off, dfn])[post.columns]
        rated = post.columns >= 2
        sd = np.sqrt(np.diag(post.cov))
        covered += int(np.sum((np.abs(post.mean - truth) < z90 * sd)[rated]))
        total += int(rated.sum())
        sigma2_hats.append(post.sigma2)
    assert total == 4000
    assert 0.87 < covered / total < 0.93
    assert np.mean(sigma2_hats) == pytest.approx(sigma2, rel=0.1)
