"""M1: possession-based team efficiency model (PLAN §5.2), fitted walk-forward by round.

Ratings (points per 100 possessions), a weighted ridge on team-game rows:

    100 · points / poss_game = μ + h · home + off[team] - def[opponent]

with home = +1 for the home team's row, -1 for the away team's, 0 at a neutral venue. Rows are
weighted by ``poss_game * 0.5^(age / half_life) * carry^(seasons back)``: exponential time decay
plus an extra pull toward the league mean at each season start (``carry`` < 1 down-weights
earlier seasons, so a new season's ratings start closer to 0). μ and h are unpenalised; each
off/def rating has ridge penalty ``ridge`` (in possessions of evidence).

Pace, the same way on game rows: ``poss_game · 40 / minutes = μ_p + pace[home] + pace[away]``,
weight = decay only (one per game), penalty ``ridge`` (in games).

Forecast for home H, away A with expected possessions P (per 40 minutes):

    ORtg_H = μ + h + off[H] - def[A],  ORtg_A = μ - h + off[A] - def[H]
    margin = P (ORtg_H - ORtg_A) / 100,  total = P (ORtg_H + ORtg_A) / 100

Walk-forward: every game is predicted from the fit on games that tipped off before the first
tip-off of its round (season, phase, round), so nothing from its round or later is used. The
decayed normal equations are kept incrementally, so each refit is one small linear solve.
"""

import math
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import optimize, special

from eurohoops.eval.metrics import crps_normal, crps_student_t
from eurohoops.models.elo import FloatArray

MINUTES_PER_GAME = 40.0
UNPENALISED = 1e-8  # keeps the μ/h block invertible before the first game; negligible after
SECONDS_PER_DAY = 86_400.0
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class DecayParams:
    half_life_days: float
    carry: float  # weight kept per season boundary (1 = no extra season-start shrinkage)
    ridge: float


class DecayedRidge:
    """Weighted ridge whose row weights decay with time and season, kept as normal equations.

    ``advance(t, season)`` rescales everything added so far to time ``t`` of ``season``;
    ``add`` then adds rows with their own age at ``t``. The penalty matrix ``penalty`` is fixed.
    ``solve`` works on the columns rows have touched so far (the rest stay 0), so a team that
    first appears later cannot change an earlier solution, not even in the last bit.
    """

    def __init__(self, penalty: FloatArray, params: DecayParams) -> None:
        self.penalty = penalty
        self.params = params
        size = len(penalty)
        self.gram = np.zeros((size, size))
        self.rhs = np.zeros(size)
        self.seen = np.zeros(size, dtype=bool)
        self.time = 0.0
        self.season: int | None = None

    def decay(self, age_days: FloatArray, seasons_back: IntArray) -> FloatArray:
        weights: FloatArray = np.power(0.5, age_days / self.params.half_life_days) * np.power(
            self.params.carry, seasons_back.astype(np.float64)
        )
        return weights

    def advance(self, time: float, season: int) -> None:
        if self.season is not None:
            factor = self.decay(
                np.array([(time - self.time) / SECONDS_PER_DAY]),
                np.array([season - self.season], dtype=np.int64),
            )[0]
            self.gram *= factor
            self.rhs *= factor
        self.time, self.season = time, season

    def add(
        self,
        columns: IntArray,
        values: FloatArray,
        *,
        target: FloatArray,
        weight: FloatArray,
        time: FloatArray,
        season: IntArray,
    ) -> None:
        """Rows with nonzeros ``values[i, k]`` in ``columns[i, k]`` (both shaped rows * k)."""
        if self.season is None or not len(target):
            return
        w = weight * self.decay((self.time - time) / SECONDS_PER_DAY, self.season - season)
        k = columns.shape[1]
        rows = np.repeat(columns, k, axis=1)
        cols = np.tile(columns, (1, k))
        products = np.repeat(values, k, axis=1) * np.tile(values, (1, k)) * w[:, None]
        np.add.at(self.gram, (rows, cols), products)
        np.add.at(self.rhs, columns, values * (w * target)[:, None])
        self.seen[columns] = True

    def solve(self) -> FloatArray:
        used = np.flatnonzero(self.seen)
        system = self.gram[np.ix_(used, used)] + np.diag(self.penalty[used])
        solution = np.zeros(len(self.rhs))
        solution[used] = np.linalg.solve(system, self.rhs[used])
        return solution


def rating_penalty(n_teams: int, ridge: float) -> FloatArray:
    """[μ, h, off * n, def * n]: μ and h (almost) free, every rating penalised by ``ridge``."""
    return np.concatenate([[UNPENALISED, UNPENALISED], np.full(2 * n_teams, ridge)])


def pace_penalty(n_teams: int, ridge: float) -> FloatArray:
    """[μ_p, pace * n]."""
    return np.concatenate([[UNPENALISED], np.full(n_teams, ridge)])


@dataclass(frozen=True)
class History:
    """Arrays of the games to forecast (sorted by tip-off) and of the rows that feed the fits."""

    teams: tuple[str, ...]
    # games (one entry per game to forecast)
    home: IntArray
    away: IntArray
    home_flag: FloatArray  # 1, or 0 at a neutral venue
    cutoff: FloatArray  # first tip-off of the game's round (epoch seconds)
    season: IntArray
    # rating rows (two per game with a box line), sorted by tip-off
    row_time: FloatArray
    row_season: IntArray
    row_team: IntArray
    row_opp: IntArray
    row_home: FloatArray  # +1, -1 or 0
    row_ortg: FloatArray
    row_poss: FloatArray
    # pace rows (one per game with a box line)
    pace_time: FloatArray
    pace_season: IntArray
    pace_home: IntArray
    pace_away: IntArray
    pace_poss40: FloatArray


def _epoch(times: pd.Series) -> FloatArray:
    values: FloatArray = times.dt.tz_convert("UTC").astype("int64").to_numpy(dtype=np.float64) / 1e9
    return values


def prepare_history(games: pd.DataFrame, team_games: pd.DataFrame) -> History:
    """``games``: one competition's ``games`` mart rows sorted by tip-off (all to forecast).

    ``team_games``: its ``team_games`` rows; only rated games (played, not forfeit) feed fits.
    """
    if not games["tipoff_utc"].is_monotonic_increasing:
        raise ValueError("games must be sorted by tipoff_utc")
    teams = tuple(sorted({*games["home"], *games["away"]}))
    index = {team: i for i, team in enumerate(teams)}
    tipoff = _epoch(games["tipoff_utc"])
    keys = list(zip(games["season"], games["phase"], games["round"], strict=True))
    first: dict[tuple[object, ...], float] = {}
    for key, t in zip(keys, tipoff, strict=True):
        first[key] = min(first.get(key, math.inf), t)
    rated = games[games["played"] & ~games["forfeit"]]
    info = rated[["game_id", "tipoff_utc", "season", "home", "neutral"]]
    rows = team_games.merge(info, on="game_id", suffixes=("", "_game"))
    rows = rows.sort_values(["tipoff_utc", "game_id", "home"], ascending=[True, True, False])
    home_sign = np.where(rows["neutral"], 0.0, np.where(rows["home"], 1.0, -1.0))
    per_game = rows.drop_duplicates("game_id")
    home_team = per_game["home_game"]
    away_team = np.where(per_game["team"] == home_team, per_game["opponent"], per_game["team"])
    return History(
        teams=teams,
        home=games["home"].map(index).to_numpy(dtype=np.int64),
        away=games["away"].map(index).to_numpy(dtype=np.int64),
        home_flag=np.where(games["neutral"], 0.0, 1.0),
        cutoff=np.array([first[key] for key in keys]),
        season=games["season"].to_numpy(dtype=np.int64),
        row_time=_epoch(rows["tipoff_utc"]),
        row_season=rows["season"].to_numpy(dtype=np.int64),
        row_team=rows["team"].map(index).to_numpy(dtype=np.int64),
        row_opp=rows["opponent"].map(index).to_numpy(dtype=np.int64),
        row_home=home_sign.astype(np.float64),
        row_ortg=(100.0 * rows["points"] / rows["poss_game"]).to_numpy(dtype=np.float64),
        row_poss=rows["poss_game"].to_numpy(dtype=np.float64),
        pace_time=_epoch(per_game["tipoff_utc"]),
        pace_season=per_game["season"].to_numpy(dtype=np.int64),
        pace_home=home_team.map(index).to_numpy(dtype=np.int64),
        pace_away=pd.Series(away_team).map(index).to_numpy(dtype=np.int64),
        pace_poss40=(per_game["poss_game"] * MINUTES_PER_GAME / per_game["minutes"]).to_numpy(
            dtype=np.float64
        ),
    )


@dataclass(frozen=True)
class Forecast:
    """Per game (aligned with ``History``): expected points of each side and possessions."""

    home_points: FloatArray
    away_points: FloatArray
    pace: FloatArray

    @property
    def margin(self) -> FloatArray:
        margin: FloatArray = self.home_points - self.away_points
        return margin

    @property
    def total(self) -> FloatArray:
        total: FloatArray = self.home_points + self.away_points
        return total


def _cutoffs(history: History) -> list[tuple[float, int, IntArray]]:
    """(cutoff time, season, indices of the games it forecasts) in time order."""
    order = np.lexsort((history.season, history.cutoff))
    out: list[tuple[float, int, IntArray]] = []
    for i in order:
        key = (float(history.cutoff[i]), int(history.season[i]))
        if out and (out[-1][0], out[-1][1]) == key:
            out[-1] = (key[0], key[1], np.append(out[-1][2], i))
        else:
            out.append((key[0], key[1], np.array([i], dtype=np.int64)))
    return out


def rating_design(history: History) -> tuple[IntArray, FloatArray]:
    """Nonzero columns and values of every rating row: [μ, h, off[team], def[opp]]."""
    n = len(history.teams)
    columns = np.stack(
        [
            np.zeros_like(history.row_team),
            np.ones_like(history.row_team),
            2 + history.row_team,
            2 + n + history.row_opp,
        ],
        axis=1,
    )
    ones = np.ones_like(history.row_home)
    values = np.stack([ones, history.row_home, ones, -ones], axis=1)
    return columns, values


class _Rows(NamedTuple):
    """Rows feeding one fit, sorted by ``time``: what ``DecayedRidge.add`` takes."""

    columns: IntArray
    design: FloatArray
    target: FloatArray
    weight: FloatArray
    time: FloatArray
    season: IntArray


def _walk[Payload](
    model: DecayedRidge, rows: _Rows, steps: Iterable[tuple[float, int, Payload]]
) -> Iterator[tuple[float, int, Payload, int]]:
    """Advance ``model`` through ``steps`` (time, season, payload) in time order, adding the rows
    that tipped off before each; yields the step and the number of rows added so far."""
    added = 0
    for time, season, payload in steps:
        model.advance(time, season)
        stop = int(np.searchsorted(rows.time, time, side="left"))
        if stop > added:
            span = slice(added, stop)
            model.add(
                rows.columns[span],
                rows.design[span],
                target=rows.target[span],
                weight=rows.weight[span],
                time=rows.time[span],
                season=rows.season[span],
            )
            added = stop
        yield time, season, payload, added


def _requested(cutoffs: Sequence[tuple[float, int]]) -> list[tuple[float, int, int]]:
    """(time, season, position in ``cutoffs``) in time order."""
    order = sorted(range(len(cutoffs)), key=lambda i: cutoffs[i])
    return [(cutoffs[i][0], cutoffs[i][1], i) for i in order]


def _rating_rows(history: History) -> _Rows:
    columns, values = rating_design(history)
    return _Rows(
        columns, values, history.row_ortg, history.row_poss, history.row_time, history.row_season
    )


def rating_fits(history: History, params: DecayParams) -> tuple[FloatArray, FloatArray]:
    """Pre-round ORtg of the home and away side of every game (NaN before any game)."""
    n = len(history.teams)
    model = DecayedRidge(rating_penalty(n, params.ridge), params)
    home_ortg = np.full(len(history.home), np.nan)
    away_ortg = np.full(len(history.home), np.nan)
    for _, _, games, added in _walk(model, _rating_rows(history), _cutoffs(history)):
        if not added:
            continue  # nothing tipped off yet: no forecast (NaN)
        theta = model.solve()
        mu, h = theta[0], theta[1]
        off, dfn = theta[2 : 2 + n], theta[2 + n :]
        hf = history.home_flag[games]
        home, away = history.home[games], history.away[games]
        home_ortg[games] = mu + h * hf + off[home] - dfn[away]
        away_ortg[games] = mu - h * hf + off[away] - dfn[home]
    return home_ortg, away_ortg


def _pace_rows(history: History) -> _Rows:
    columns = np.stack(
        [np.zeros_like(history.pace_home), 1 + history.pace_home, 1 + history.pace_away], axis=1
    )
    return _Rows(
        columns,
        np.ones(columns.shape, dtype=np.float64),
        history.pace_poss40,
        np.ones(len(history.pace_home)),
        history.pace_time,
        history.pace_season,
    )


def pace_fits(history: History, params: DecayParams) -> FloatArray:
    """Pre-round expected possessions (per 40 minutes) of every game."""
    n = len(history.teams)
    model = DecayedRidge(pace_penalty(n, params.ridge), params)
    pace = np.full(len(history.home), np.nan)
    for _, _, games, added in _walk(model, _pace_rows(history), _cutoffs(history)):
        if not added:
            continue
        theta = model.solve()
        pace[games] = theta[0] + theta[1 + history.home[games]] + theta[1 + history.away[games]]
    return pace


@dataclass(frozen=True)
class RatingPosterior:
    """M1's rating posterior at a cutoff (K-b): Gaussian over the seen columns of the design
    [μ, h, off · n, def · n] (``rating_design`` order).

    Each ORtg row has variance σ²/poss_i and the time/season decay d_i acts as a fractional
    likelihood power, so the posterior precision is (G + P)/σ² with G = Σ d_i·poss_i·x_i x_iᵀ
    (``DecayedRidge.gram``) and the ridge penalty P a prior in possessions of evidence. σ² is
    Σ d_i·poss_i·r_i² / Σ d_i with r_i = ORtg_i - x_i·θ̂ at the cutoff's solution θ̂, over the
    rows that tipped off before the cutoff. Before any row: no columns and σ² is NaN.
    """

    teams: tuple[str, ...]  # History.teams (all n, design order)
    columns: IntArray  # (k,) seen design columns, ascending
    labels: tuple[str, ...]  # (k,) "mu", "h", "off:<team>", "def:<team>"
    mean: FloatArray  # (k,) DecayedRidge.solve() at the cutoff, seen columns
    cov: FloatArray  # (k, k) sigma2 · inv(G + P) over the seen columns
    sigma2: float  # Σ d_i·poss_i·r_i² / Σ d_i over the rows used (d_i = decay weight at the cutoff)


def rating_posteriors(
    history: History, params: DecayParams, cutoffs: Sequence[tuple[float, int]]
) -> list[RatingPosterior]:
    """Posterior at each (cutoff epoch seconds, season), in one walk-forward pass: rows with
    row_time < cutoff, the model advanced to (cutoff, season) as rating_fits does.

    ``cutoffs`` may come in any order; the result is aligned with it.
    """
    teams = history.teams
    names = ["mu", "h", *(f"off:{t}" for t in teams), *(f"def:{t}" for t in teams)]
    model = DecayedRidge(rating_penalty(len(teams), params.ridge), params)
    rows = _rating_rows(history)
    out: list[RatingPosterior | None] = [None] * len(cutoffs)
    for time, season, position, added in _walk(model, rows, _requested(cutoffs)):
        columns = np.flatnonzero(model.seen).astype(np.int64)
        mean = model.solve()
        precision = model.gram[np.ix_(columns, columns)] + np.diag(model.penalty[columns])
        sigma2 = math.nan
        if added:
            span = slice(0, added)
            residual = rows.target[span] - (mean[rows.columns[span]] * rows.design[span]).sum(1)
            decay = model.decay(
                (time - rows.time[span]) / SECONDS_PER_DAY, season - rows.season[span]
            )
            sigma2 = float(np.sum(decay * rows.weight[span] * residual**2) / np.sum(decay))
        inverse = np.linalg.inv(precision)
        out[position] = RatingPosterior(
            teams=teams,
            columns=columns,
            labels=tuple(names[c] for c in columns),
            mean=mean[columns],
            cov=sigma2 * (inverse + inverse.T) / 2.0,
            sigma2=sigma2,
        )
    return [posterior for posterior in out if posterior is not None]


def pace_points(
    history: History, params: DecayParams, cutoffs: Sequence[tuple[float, int]]
) -> list[FloatArray]:
    """Pace solution [μ_p, pace · n] (1 + n,) at each cutoff, as pace_fits uses it (all 0 before
    any game), aligned with ``cutoffs``."""
    model = DecayedRidge(pace_penalty(len(history.teams), params.ridge), params)
    out: list[FloatArray | None] = [None] * len(cutoffs)
    for _, _, position, _ in _walk(model, _pace_rows(history), _requested(cutoffs)):
        out[position] = model.solve()
    return [point for point in out if point is not None]


def forecast(history: History, rating: DecayParams, pace: DecayParams) -> Forecast:
    home_ortg, away_ortg = rating_fits(history, rating)
    poss = pace_fits(history, pace)
    return Forecast(poss * home_ortg / 100.0, poss * away_ortg / 100.0, poss)


# --- Predictive distributions (E-e) -----------------------------------------------------------

VARIANTS = ("normal_const", "normal_pace", "student_t_const", "student_t_pace")
T_DF_GRID = (3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0)
P_CLIP = 1e-9  # keeps a log loss finite for a (practically impossible) z beyond ±8


@dataclass(frozen=True)
class MarginModel:
    """Margin ~ m + scale_g · e, e standard Normal or Student-t(df); scale_g = scale at a
    constant variance, or scale · sqrt(P_g / ref_pace) when it grows with expected pace."""

    variant: str
    scale: float
    df: float | None = None
    ref_pace: float | None = None

    def scales(self, pace: FloatArray) -> FloatArray:
        if self.ref_pace is None:
            return np.full(len(pace), self.scale)
        scaled: FloatArray = self.scale * np.sqrt(pace / self.ref_pace)
        return scaled

    def cdf(self, z: FloatArray) -> FloatArray:
        out: FloatArray = special.ndtr(z) if self.df is None else special.stdtr(self.df, z)
        return out

    def p_home(self, margin: FloatArray, pace: FloatArray) -> FloatArray:
        p: FloatArray = np.clip(self.cdf(margin / self.scales(pace)), P_CLIP, 1.0 - P_CLIP)
        return p

    def crps(self, margin: FloatArray, pace: FloatArray, actual: FloatArray) -> FloatArray:
        if self.df is None:
            return crps_normal(margin, self.scales(pace), actual)
        return crps_student_t(margin, self.scales(pace), self.df, actual)


def _t_log_likelihood(z: FloatArray, df: float) -> float:
    """Sum of standard-t log densities."""
    return float(
        np.sum(
            special.gammaln((df + 1.0) / 2.0)
            - special.gammaln(df / 2.0)
            - 0.5 * math.log(df * math.pi)
            - (df + 1.0) / 2.0 * np.log1p(z**2 / df)
        )
    )


def _t_scale(residual: FloatArray, relative: FloatArray, df: float) -> float:
    """Maximum-likelihood scale of residual / sqrt(relative) under a Student-t(df)."""
    standard = residual / np.sqrt(relative)
    rms = float(np.sqrt(np.mean(standard**2)))

    def negative(scale: float) -> float:
        return -_t_log_likelihood(standard / scale, df) + len(standard) * math.log(scale)

    found = optimize.minimize_scalar(
        negative, bounds=(rms / 10.0, rms * 2.0), method="bounded", options={"xatol": 1e-8}
    )
    return round(float(found.x), 6)


def _log_loss(p: FloatArray, home_won: FloatArray) -> float:
    return float(np.mean(-(home_won * np.log(p) + (1.0 - home_won) * np.log(1.0 - p))))


def fit_margin_model(
    variant: str, margin: FloatArray, pace: FloatArray, actual: FloatArray
) -> MarginModel:
    """Fit a declared variant on (tuning) forecasts ``margin``, ``pace`` and outcomes ``actual``.

    Scales by maximum likelihood; the t variants pick df from ``T_DF_GRID`` by log loss.
    """
    residual = actual - margin
    ref = round(float(np.mean(pace)), 6) if variant.endswith("_pace") else None
    relative = pace / ref if ref is not None else np.ones(len(pace))
    if variant.startswith("normal"):
        scale = round(float(np.sqrt(np.mean(residual**2 / relative))), 6)
        return MarginModel(variant, scale, None, ref)
    if not variant.startswith("student_t"):
        raise ValueError(f"unknown variant {variant!r}")
    home_won = (actual > 0).astype(np.float64)
    candidates = [
        MarginModel(variant, _t_scale(residual, relative, df), df, ref) for df in T_DF_GRID
    ]
    losses = [_log_loss(m.p_home(margin, pace), home_won) for m in candidates]
    return candidates[int(np.argmin(losses))]
