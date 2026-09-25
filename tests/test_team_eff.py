"""E3: M1 ratings (closed-form ridge, synthetic recovery), pace, distributions and t CRPS."""

import math

import numpy as np
import pandas as pd
import pytest
from scipy import integrate, stats

from eurohoops.eval.metrics import crps_normal, crps_student_t
from eurohoops.models.team_eff import (
    SECONDS_PER_DAY,
    T_DF_GRID,
    DecayedRidge,
    DecayParams,
    MarginModel,
    fit_margin_model,
    forecast,
    pace_fits,
    prepare_history,
    rating_design,
    rating_fits,
    rating_penalty,
)
from eurohoops.parse.games import conform

DAY = pd.Timedelta(days=1)


def league(
    ratings: dict[str, tuple[float, float, float]],
    seasons: list[int],
    *,
    mu: float = 110.0,
    home_edge: float = 2.0,
    noise: float = 0.0,
    seed: int = 1,
    unplayed_last_round: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Double round-robin ``games`` and ``team_games`` from known (off, def, pace) ratings.

    Each pair plays once per round (all pairs of a round on consecutive days). Points follow
    the M1 equations exactly (plus Normal noise per 100 possessions).
    """
    rng = np.random.default_rng(seed)
    teams = sorted(ratings)
    pairs = [(h, a) for h in teams for a in teams if h != a]
    games, rows = [], []
    for season in seasons:
        start = pd.Timestamp(f"{season}-10-01T18:00:00Z")
        for code, (home, away) in enumerate(pairs, start=1):
            rnd = (code - 1) // max(len(teams) // 2, 1) + 1
            tip = start + (rnd - 1) * 7 * DAY + (code % 3) * pd.Timedelta(hours=2)
            played = not (unplayed_last_round and season == seasons[-1] and code > len(pairs) - 3)
            poss = 70.0 + ratings[home][2] + ratings[away][2]
            ortg_h = mu + home_edge + ratings[home][0] - ratings[away][1] + noise * rng.normal()
            ortg_a = mu - home_edge + ratings[away][0] - ratings[home][1] + noise * rng.normal()
            hp, ap = round(poss * ortg_h / 100.0), round(poss * ortg_a / 100.0)
            ap = ap if ap != hp else ap + 1
            game_id = f"S{season}_{code}"
            games.append(
                {
                    "game_id": game_id,
                    "season": season,
                    "game_code": code,
                    "phase": "RS",
                    "round": rnd,
                    "round_label": str(rnd),
                    "tipoff_utc": tip,
                    "home": home,
                    "away": away,
                    "home_score": hp if played else None,
                    "away_score": ap if played else None,
                    "played": played,
                    "forfeit": False,
                    "neutral": False,
                    "confirmed_date": True,
                }
            )
            if played:
                for team, opp, pts, is_home in ((home, away, hp, True), (away, home, ap, False)):
                    rows.append(
                        {
                            "game_id": game_id,
                            "team": team,
                            "opponent": opp,
                            "home": is_home,
                            "points": pts,
                            "poss_game": poss,
                            "minutes": 40.0,
                        }
                    )
    return conform(pd.DataFrame(games)), pd.DataFrame(rows)


TOY = {"AAA": (4.0, 1.0, 0.0), "BBB": (0.0, -2.0, 0.0), "CCC": (-3.0, 1.0, 0.0)}


def closed_form(history, params: DecayParams, cutoff: float, season: int) -> np.ndarray:
    """(X'WX + P)^-1 X'Wy from the rows before ``cutoff``, with weights computed directly."""
    n = len(history.teams)
    use = history.row_time < cutoff
    x = np.zeros((int(use.sum()), 2 + 2 * n))
    rows = np.flatnonzero(use)
    for r, i in enumerate(rows):
        x[r, 0] = 1.0
        x[r, 1] = history.row_home[i]
        x[r, 2 + history.row_team[i]] = 1.0
        x[r, 2 + n + history.row_opp[i]] = -1.0
    age = (cutoff - history.row_time[rows]) / SECONDS_PER_DAY
    back = season - history.row_season[rows]
    w = history.row_poss[rows] * 0.5 ** (age / params.half_life_days) * params.carry**back
    a = x.T @ (w[:, None] * x) + np.diag(rating_penalty(n, params.ridge))
    return np.linalg.solve(a, x.T @ (w * history.row_ortg[rows]))


def test_three_team_toy_matches_the_closed_form_ridge() -> None:
    games, rows = league(TOY, [2020, 2021], noise=8.0, unplayed_last_round=True)
    history = prepare_history(games, rows)
    params = DecayParams(half_life_days=90.0, carry=0.6, ridge=300.0)
    home_ortg, away_ortg = rating_fits(history, params)
    last = len(games) - 1  # an unplayed game of the last round of 2021
    theta = closed_form(history, params, float(history.cutoff[last]), 2021)
    n = len(history.teams)
    h, a = history.home[last], history.away[last]
    expected_home = theta[0] + theta[1] + theta[2 + h] - theta[2 + n + a]
    expected_away = theta[0] - theta[1] + theta[2 + a] - theta[2 + n + h]
    assert abs(home_ortg[last] - expected_home) < 1e-9
    assert abs(away_ortg[last] - expected_away) < 1e-9
    # every earlier cutoff too (incremental decay across a season boundary)
    for i in range(len(games)):
        if not np.isnan(home_ortg[i]) and history.row_time[0] < history.cutoff[i]:
            t = closed_form(history, params, float(history.cutoff[i]), int(history.season[i]))
            want = t[0] + t[1] + t[2 + history.home[i]] - t[2 + n + history.away[i]]
            assert abs(home_ortg[i] - want) < 1e-9


def test_decayed_ridge_ignores_rows_before_its_first_advance() -> None:
    model = DecayedRidge(np.ones(2), DecayParams(10.0, 1.0, 1.0))
    one = np.ones((1, 1), dtype=np.int64)
    model.add(
        one,
        np.ones((1, 1)),
        target=np.ones(1),
        weight=np.ones(1),
        time=np.zeros(1),
        season=np.zeros(1, dtype=np.int64),
    )
    assert not model.rhs.any()


def test_a_synthetic_league_is_recovered() -> None:
    rng = np.random.default_rng(3)
    truth = {f"T{i:02d}": (rng.normal(0, 5), rng.normal(0, 5), rng.normal(0, 3)) for i in range(12)}
    games, rows = league(truth, [2020, 2021, 2022], noise=6.0, seed=4)
    history = prepare_history(games, rows)
    params = DecayParams(half_life_days=10_000.0, carry=1.0, ridge=50.0)
    model = DecayedRidge(rating_penalty(len(history.teams), params.ridge), params)
    model.advance(float(history.row_time[-1]) + 1.0, 2022)
    columns, values = rating_design(history)
    model.add(
        columns,
        values,
        target=history.row_ortg,
        weight=history.row_poss,
        time=history.row_time,
        season=history.row_season,
    )
    theta = model.solve()
    n = len(history.teams)
    off_true = np.array([truth[t][0] for t in history.teams])
    def_true = np.array([truth[t][1] for t in history.teams])
    assert np.corrcoef(theta[2 : 2 + n], off_true)[0, 1] > 0.95
    assert np.corrcoef(theta[2 + n :], def_true)[0, 1] > 0.95
    assert theta[1] == pytest.approx(2.0, abs=0.5)
    # Pace: late-season expected possessions track the true pace effects.
    pace = pace_fits(history, DecayParams(10_000.0, 1.0, 1.0))
    true_pace = 70.0 + np.array(
        [
            truth[history.teams[h]][2] + truth[history.teams[a]][2]
            for h, a in zip(history.home, history.away, strict=True)
        ]
    )
    late = history.season == 2022
    assert np.corrcoef(pace[late], true_pace[late])[0, 1] > 0.95


def test_forecast_combines_pace_and_ratings() -> None:
    games, rows = league(TOY, [2020, 2021])
    history = prepare_history(games, rows)
    params = DecayParams(60.0, 0.5, 200.0)
    result = forecast(history, params, DecayParams(60.0, 0.5, 5.0))
    home_ortg, away_ortg = rating_fits(history, params)
    pace = pace_fits(history, DecayParams(60.0, 0.5, 5.0))
    assert np.allclose(result.margin, pace * (home_ortg - away_ortg) / 100.0, equal_nan=True)
    assert np.allclose(result.total, pace * (home_ortg + away_ortg) / 100.0, equal_nan=True)
    # The very first round has nothing before it: no forecast at all.
    first = history.cutoff == history.cutoff.min()
    assert np.isnan(home_ortg[first]).all() and np.isnan(pace[first]).all()
    assert not np.isnan(home_ortg[~first]).any()


def test_prepare_history_needs_sorted_games() -> None:
    games, rows = league(TOY, [2020])
    with pytest.raises(ValueError, match="sorted"):
        prepare_history(games.iloc[::-1], rows)


@pytest.mark.parametrize("df", [1.5, 3.0, 7.0, 30.0])
@pytest.mark.parametrize("y", [-25.0, -3.0, 0.0, 4.5, 40.0])
def test_student_t_crps_matches_the_integral(df: float, y: float) -> None:
    mu, scale = 2.0, 11.0
    dist = stats.t(df, loc=mu, scale=scale)
    below, _ = integrate.quad(lambda x: dist.cdf(x) ** 2, -np.inf, y, epsabs=1e-12, limit=500)
    above, _ = integrate.quad(lambda x: (1 - dist.cdf(x)) ** 2, y, np.inf, epsabs=1e-12, limit=500)
    got = crps_student_t(np.array([mu]), scale, df, np.array([y]))[0]
    assert abs(got - (below + above)) < 1e-6


def test_student_t_crps_tends_to_normal_and_needs_df_above_one() -> None:
    y = np.array([-10.0, 0.0, 7.0])
    t = crps_student_t(np.zeros(3), 12.0, 1e6, y)
    assert np.allclose(t, crps_normal(np.zeros(3), 12.0, y), atol=1e-4)
    with pytest.raises(ValueError, match="df > 1"):
        crps_student_t(np.zeros(1), 1.0, 1.0, np.zeros(1))


def test_margin_models_fit_on_residuals() -> None:
    rng = np.random.default_rng(5)
    m = rng.normal(0, 8, 4000)
    pace = rng.uniform(64, 80, 4000)
    actual = m + 12.0 * rng.standard_t(6, 4000)
    const = fit_margin_model("normal_const", m, pace, actual)
    assert const.scale == pytest.approx(math.sqrt(np.mean((actual - m) ** 2)), abs=1e-6)
    assert const.ref_pace is None and const.df is None
    paced = fit_margin_model("normal_pace", m, pace, actual)
    assert paced.ref_pace == pytest.approx(float(np.mean(pace)), abs=1e-6)
    assert paced.scales(np.array([paced.ref_pace]))[0] == paced.scale
    t = fit_margin_model("student_t_const", m, pace, actual)
    assert t.df in T_DF_GRID
    assert 10.0 < t.scale < 14.0
    p = t.p_home(m, pace)
    assert ((p > 0) & (p < 1)).all()
    assert np.allclose(t.crps(m, pace, actual), crps_student_t(m, t.scale, t.df, actual))
    assert np.allclose(const.crps(m, pace, actual), crps_normal(m, const.scale, actual))
    tp = fit_margin_model("student_t_pace", m, pace, actual)
    assert tp.ref_pace is not None and tp.df in T_DF_GRID
    with pytest.raises(ValueError, match="unknown variant"):
        fit_margin_model("laplace", m, pace, actual)


def test_home_probability_is_clipped_and_symmetric() -> None:
    model = MarginModel("normal_const", 10.0)
    p = model.p_home(np.array([-500.0, 0.0, 500.0]), np.full(3, 70.0))
    assert p[0] > 0 and p[2] < 1 and p[1] == 0.5
