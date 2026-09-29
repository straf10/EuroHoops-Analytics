"""M3 box-only and PIR baselines (week 9-12 H-e): the same inputs RAPM sees, without a lineup.

Both baselines share one shape: a per-player, time-decayed, minutes-shrunk rate, combined into a
team value with the game's projected shares (``models.minutes``), then a small linear model
(ridge for the box-only regression, plain OLS for the PIR baseline) turns the team-value
difference into a margin. The decayed sums are kept incrementally and walked forward round by
round, exactly like ``models.team_eff.DecayedRidge`` (advance to a cutoff, add the rows that
tipped off before it, read off the state) — so no rate, shrinkage or feature used for a game can
depend on that game or a later one; only ``beta``/``h`` (box-only) and ``a``/``c`` (PIR) are
fitted once, on the tuning games only.

Box-only (H-e): 11 per-100-possession rates (PTS, 2PA, 3PA, FTA, OREB, DREB, AST, STL, BLK, TOV,
PF), each player's rate shrunk toward the decayed league rate by a minutes-based weight
``m_p / (m_p + k)``, features = shrunk rate - league rate (so an unseen player contributes 0).
Game feature ``z_g = P_g/100 * (sum_home f_p x_p - sum_away f_p x_p)`` (``f`` = projected share,
``P`` = expected possessions); margin = ``beta.z_g + h * P_g/100 * home_flag``. ``k``, the decay
half-life and the ridge penalty are chosen by leave-one-tuning-season-out CV of margin RMSE over
``BoxGrid``; ``beta``/``h`` are then fitted once more on every tuning game.

PIR (the naive baseline): the same decayed/shrunk construction on one "stat" (PIR), rate = per
40 minutes rather than per 100 possessions (``PIR_HALF_LIFE_DAYS``/``PIR_K_MINUTES``, fixed, not
grid-searched: ``pir_margins`` takes no grid). Team value = ``sum_p f_p (pir40_p - league) / 5``;
margin = ``a * home_flag + c * (team_home - team_away)``, ``(a, c)`` by OLS on tuning games.
"""

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from eurohoops.models.elo import FloatArray
from eurohoops.models.minutes import round_cutoffs

IntArray = npt.NDArray[np.int64]
SECONDS_PER_DAY = 86_400.0
UNPENALISED = 1e-8  # h (box-only) and the PIR OLS carry no ridge penalty

# H-e's 11 box-score inputs, in the order the task lists them (PTS, 2PA, 3PA, FTA, OREB, DREB,
# AST, STL, BLK, TOV, PF); `pf` here is fouls committed (ESAKE's "FOULS M").
STAT_COLUMNS = ("pts", "fg2a", "fg3a", "fta", "oreb", "dreb", "ast", "stl", "blk", "tov", "pf")

# PIR baseline: fixed decay/shrinkage (no grid parameter on `pir_margins`), the midpoints of
# `M3Grid`'s half-life and k ranges -- declared here, not tuned, because it is the *naive*
# baseline PIR-per-minute is compared against, not a fitted model.
PIR_HALF_LIFE_DAYS = 365.0
PIR_K_MINUTES = 250.0

EMPTY_IDX: IntArray = np.zeros(0, dtype=np.int64)
EMPTY_W: FloatArray = np.zeros(0, dtype=np.float64)


@dataclass(frozen=True)
class BoxGrid:
    """Box-only hyperparameters (H-e), all chosen on the tuning seasons by leave-one-tuning-
    season-out CV of margin RMSE. ``k`` is in minutes of shrinkage evidence, ``ridge`` is the
    ridge penalty on ``beta`` (``h`` is unpenalised)."""

    # Widened once on 2026-09-28 before the verdict (tuning only): the first tuning run chose
    # half-life 182 and ridge 300, both grid edges (reports/week9-12_progress.md, iteration 7).
    k: tuple[float, ...] = (100.0, 250.0, 500.0, 1000.0)
    half_life_days: tuple[float, ...] = (45.0, 91.0, 182.0, 365.0, 730.0, 1460.0)
    ridge: tuple[float, ...] = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0)


@dataclass(frozen=True)
class BaselineResult:
    margin: FloatArray  # one per game in `games`, aligned with it; NaN where `possessions` is
    params: dict[str, Any]  # JSON-ready (rounded to 6 decimals)


def _epoch(times: pd.Series) -> FloatArray:
    """Epoch seconds, whatever the timestamps' resolution (matches ``models.minutes._epoch``:
    ``.astype("int64")`` alone assumes nanoseconds, which pandas 3.x does not always use)."""
    epoch = pd.Timestamp(0, tz="UTC")
    seconds: FloatArray = (
        (times.dt.tz_convert("UTC") - epoch).dt.total_seconds().to_numpy(dtype=np.float64)
    )
    return seconds


def _round(value: float) -> float:
    return round(float(value), 6)


def _round_dict(values: dict[str, float]) -> dict[str, float]:
    return {name: _round(v) for name, v in values.items()}


class _Accumulator:
    """Decayed per-player sums of a set of stat columns, plus on-court possessions and minutes,
    and the same summed league-wide -- the walk-forward building block both baselines share.
    Mirrors ``models.team_eff.DecayedRidge``'s ``advance``/``add`` (rescale to a new time, then
    add rows with their own age), without the linear solve.
    """

    def __init__(self, n_players: int, n_stats: int, half_life_days: float) -> None:
        self.half_life_days = half_life_days
        self.stats = np.zeros((n_players, n_stats))
        self.poss = np.zeros(n_players)
        self.minutes = np.zeros(n_players)
        self.league_stats = np.zeros(n_stats)
        self.league_poss = 0.0
        self.league_minutes = 0.0
        self.time: float | None = None

    def _decay(self, age_days: FloatArray) -> FloatArray:
        weights: FloatArray = np.power(0.5, age_days / self.half_life_days)
        return weights

    def advance(self, time: float) -> None:
        if self.time is not None:
            factor = float(self._decay(np.array([time - self.time]) / SECONDS_PER_DAY)[0])
            self.stats *= factor
            self.poss *= factor
            self.minutes *= factor
            self.league_stats *= factor
            self.league_poss *= factor
            self.league_minutes *= factor
        self.time = time

    def add(
        self,
        players: IntArray,
        stats: FloatArray,
        poss: FloatArray,
        minutes: FloatArray,
        time: FloatArray,
    ) -> None:
        if self.time is None or not len(players):
            return
        w = self._decay((self.time - time) / SECONDS_PER_DAY)
        weighted = stats * w[:, None]
        np.add.at(self.stats, players, weighted)
        np.add.at(self.poss, players, poss * w)
        np.add.at(self.minutes, players, minutes * w)
        self.league_stats += weighted.sum(axis=0)
        self.league_poss += float((poss * w).sum())
        self.league_minutes += float((minutes * w).sum())


def _rate(sums: FloatArray, denom: FloatArray, scale: float) -> FloatArray:
    """``scale * sums / denom``, elementwise; 0 wherever ``denom`` is 0 (an unseen player).

    ``denom`` must already be broadcastable to ``sums``'s shape: for a per-player ``(n, k)``
    ``sums``, pass a ``(n, 1)`` ``denom`` (``player_denom[:, None]``), not a bare ``(n,)`` one --
    that silently broadcasts to an ``(n, n)`` result instead of raising.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = scale * sums / denom
    return np.where(denom > 0, rate, 0.0)


def _player_index(player_games: pd.DataFrame, shares: pd.DataFrame) -> dict[str, int]:
    """Every player id in either table, indexed alphabetically (deterministic across runs)."""
    ids = pd.concat([player_games["player_id"], shares["player_id"]], ignore_index=True)
    return {p: i for i, p in enumerate(sorted(ids.astype(str).unique()))}


@dataclass(frozen=True)
class _Rows:
    """Player-games rows joined to their game's tip-off, sorted by time (see ``_rows``)."""

    idx: IntArray  # player index (into a `_player_index` mapping)
    time: FloatArray  # tip-off, epoch seconds
    stats: FloatArray  # (n_rows, n_stats)
    poss: FloatArray
    minutes: FloatArray


def _rows(
    games: pd.DataFrame,
    player_games: pd.DataFrame,
    player_index: dict[str, int],
    stats: tuple[str, ...],
) -> _Rows:
    """Every player_games row of a rated game in ``games``, sorted by tip-off (ties broken by
    game id then player id)."""
    rated = games[games["played"] & ~games["forfeit"]][["game_id", "tipoff_utc"]]
    rows = player_games.merge(rated, on="game_id")
    rows = rows.sort_values(["tipoff_utc", "game_id", "player_id"]).reset_index(drop=True)
    idx: IntArray = np.asarray(rows["player_id"].astype(str).map(player_index), dtype=np.int64)
    time = _epoch(rows["tipoff_utc"])
    values = rows[list(stats)].to_numpy(dtype=np.float64)
    poss = rows["poss"].to_numpy(dtype=np.float64)
    minutes = rows["sec"].to_numpy(dtype=np.float64) / 60.0
    return _Rows(idx, time, values, poss, minutes)


def _shares_lookup(
    shares: pd.DataFrame, player_index: dict[str, int]
) -> dict[tuple[str, str], tuple[IntArray, FloatArray]]:
    out: dict[tuple[str, str], tuple[IntArray, FloatArray]] = {}
    for (game_id, side), rows in shares.groupby(["game_id", "side"]):
        idx: IntArray = np.asarray(rows["player_id"].astype(str).map(player_index), dtype=np.int64)
        weight = rows["share"].to_numpy(dtype=np.float64)
        out[(str(game_id), str(side))] = (idx, weight)
    return out


@dataclass(frozen=True)
class _RoundSnapshot:
    """One round's decayed state (half-life dependent only): the games it forecasts and, as of
    its cutoff, every player's rate and decayed-minutes weight, and the league rate."""

    games_idx: IntArray
    player_rate: FloatArray  # (n_players, n_stats)
    league_rate: FloatArray  # (n_stats,)
    weight: FloatArray  # (n_players,), decayed minutes


def _round_batches(games: pd.DataFrame) -> list[tuple[float, IntArray]]:
    """(cutoff time, indices into ``games``) in time order, one entry per distinct round.

    Mirrors ``models.team_eff._cutoffs`` (season is only a tie-breaker: two different seasons'
    rounds never share a cutoff in practice, but grouping by both is free and matches the
    pattern the rest of the codebase uses for this).
    """
    cutoff = round_cutoffs(games).to_numpy()
    season = games["season"].to_numpy()
    order = np.lexsort((season, cutoff))
    out: list[tuple[float, int, IntArray]] = []
    for i in order:
        key = (float(cutoff[i]), int(season[i]))
        if out and (out[-1][0], out[-1][1]) == key:
            out[-1] = (key[0], key[1], np.append(out[-1][2], i))
        else:
            out.append((key[0], key[1], np.array([i], dtype=np.int64)))
    return [(time, idx) for time, _, idx in out]


@dataclass(frozen=True)
class _RateSpec:
    """What a walk-forward pass rates a player on: possessions (box-only, per 100) or on-court
    minutes (PIR, per 40)."""

    denom: str  # "poss" or "minutes"
    scale: float


def _round_snapshots(
    games: pd.DataFrame, rows: _Rows, n_players: int, half_life_days: float, spec: _RateSpec
) -> list[_RoundSnapshot]:
    """Walk forward by round: one snapshot per round, computed from rows that tipped off before
    the round's cutoff only (never that round's or a later one's)."""
    n_stats = rows.stats.shape[1]
    acc = _Accumulator(n_players, n_stats, half_life_days)
    added = 0
    snapshots: list[_RoundSnapshot] = []
    for time, games_idx in _round_batches(games):
        acc.advance(time)
        stop = int(np.searchsorted(rows.time, time, side="left"))
        if stop > added:
            span = slice(added, stop)
            acc.add(
                rows.idx[span],
                rows.stats[span],
                rows.poss[span],
                rows.minutes[span],
                rows.time[span],
            )
            added = stop
        player_denom = acc.poss if spec.denom == "poss" else acc.minutes
        league_denom = acc.league_poss if spec.denom == "poss" else acc.league_minutes
        player_rate = _rate(acc.stats, player_denom[:, None], spec.scale)
        league_rate = _rate(acc.league_stats, np.array([league_denom]), spec.scale)
        snapshots.append(_RoundSnapshot(games_idx, player_rate, league_rate, acc.minutes.copy()))
    return snapshots


def _features(snapshot: _RoundSnapshot, k: float) -> FloatArray:
    """Shrunk rate minus league rate, per player (0 for a player never seen: ``weight`` is 0)."""
    weight = snapshot.weight[:, None]
    shrunk = weight / (weight + k) * (snapshot.player_rate - snapshot.league_rate)
    return np.where(weight > 0, shrunk, 0.0)


def _z_from_snapshots(
    snapshots: list[_RoundSnapshot],
    game_ids: npt.NDArray[np.object_],
    lookup: dict[tuple[str, str], tuple[IntArray, FloatArray]],
    k: float,
) -> FloatArray:
    n_stats = snapshots[0].league_rate.shape[0] if snapshots else 0
    z = np.zeros((len(game_ids), n_stats))
    for snapshot in snapshots:
        feature = _features(snapshot, k)
        for g in snapshot.games_idx:
            gid = str(game_ids[g])
            home_idx, home_w = lookup.get((gid, "home"), (EMPTY_IDX, EMPTY_W))
            away_idx, away_w = lookup.get((gid, "away"), (EMPTY_IDX, EMPTY_W))
            z[g] = home_w @ feature[home_idx] - away_w @ feature[away_idx]
    return z


def _ridge_fit(x: FloatArray, y: FloatArray, penalty: FloatArray) -> FloatArray:
    """Normal-equations ridge solve: ``inverse(X'X + diag(penalty)) . X'y``."""
    theta: FloatArray = np.linalg.solve(x.T @ x + np.diag(penalty), x.T @ y)
    return theta


def _rmse(x: FloatArray, y: FloatArray, theta: FloatArray) -> float:
    residual = x @ theta - y
    return float(np.sqrt(np.mean(residual**2)))


def _actual_margin(games: pd.DataFrame) -> FloatArray:
    home = games["home_score"].to_numpy(dtype="float64", na_value=np.nan)
    away = games["away_score"].to_numpy(dtype="float64", na_value=np.nan)
    margin: FloatArray = home - away
    return margin


def _home_flag(games: pd.DataFrame) -> FloatArray:
    flag: FloatArray = np.where(games["neutral"].to_numpy(), 0.0, 1.0)
    return flag


def box_only_margins(  # noqa: PLR0917 -- the harness calls this exact signature (week 9-12 H2)
    games: pd.DataFrame,
    player_games: pd.DataFrame,
    shares: pd.DataFrame,
    possessions: pd.Series,
    tuning: npt.NDArray[np.bool_],
    grid: BoxGrid,
) -> BaselineResult:
    """H-e's box-only baseline: a ridge regression on decayed, shrunk per-100 box rates."""
    game_ids = games["game_id"].astype(str).to_numpy()
    player_index = _player_index(player_games, shares)
    n_players = len(player_index)
    rows = _rows(games, player_games, player_index, STAT_COLUMNS)
    lookup = _shares_lookup(shares, player_index)
    p = possessions.reindex(games["game_id"]).to_numpy(dtype=np.float64)
    home_flag = _home_flag(games)
    actual = _actual_margin(games)
    season = games["season"].to_numpy()
    tuning_seasons = sorted({int(s) for s in season[tuning]})
    fittable = tuning & ~np.isnan(p) & ~np.isnan(actual)

    grid_results: list[dict[str, Any]] = []
    best: tuple[float, float, float, float] | None = None  # (cv_rmse, half_life, k, ridge)
    best_z: FloatArray | None = None
    for half_life in grid.half_life_days:
        snapshots = _round_snapshots(games, rows, n_players, half_life, _RateSpec("poss", 100.0))
        for k in grid.k:
            z = _z_from_snapshots(snapshots, game_ids, lookup, k)
            x = np.column_stack([z, p / 100.0 * home_flag])
            for ridge in grid.ridge:
                penalty = np.concatenate([np.full(len(STAT_COLUMNS), ridge), [UNPENALISED]])
                fold_rmse = []
                for held_out in tuning_seasons:
                    train = fittable & (season != held_out)
                    test = fittable & (season == held_out)
                    if not train.any() or not test.any():
                        continue
                    theta = _ridge_fit(x[train], actual[train], penalty)
                    fold_rmse.append(_rmse(x[test], actual[test], theta))
                cv_rmse = float(np.mean(fold_rmse)) if fold_rmse else math.nan
                grid_results.append(
                    {
                        "half_life_days": half_life,
                        "k": k,
                        "ridge": ridge,
                        "cv_rmse": _round(cv_rmse),
                    }
                )
                if not fold_rmse:
                    continue
                candidate = (cv_rmse, half_life, k, ridge)
                if best is None or candidate < best:
                    best = candidate
                    best_z = z

    if best is None or best_z is None:
        raise ValueError("no tuning season has both a training and a held-out fold")
    _, half_life, k, ridge = best
    penalty = np.concatenate([np.full(len(STAT_COLUMNS), ridge), [UNPENALISED]])
    x_full = np.column_stack([best_z, p / 100.0 * home_flag])
    theta = _ridge_fit(x_full[fittable], actual[fittable], penalty)
    beta, h = theta[:-1], float(theta[-1])
    margin = x_full @ theta

    params = {
        "half_life_days": half_life,
        "k": k,
        "ridge": ridge,
        "beta": _round_dict(dict(zip(STAT_COLUMNS, beta.tolist(), strict=True))),
        "h": _round(h),
        "cv_grid": grid_results,
    }
    return BaselineResult(margin.astype(np.float64), params)


# --- Public aliases (week 9-12 subagent F): the SPM prior (``models/spm.py``) reuses this
# module's decayed, shrunk per-100 rate machinery -- same half-life, same shrinkage shape, same
# numbers -- rather than a second implementation. The private names above are untouched (every
# existing call and test keeps working unchanged); these are the same objects under public names.
RoundSnapshot = _RoundSnapshot
RateSpec = _RateSpec
player_index = _player_index
rows_from_player_games = _rows
round_batches = _round_batches
round_snapshots = _round_snapshots
features = _features


def pir_margins(
    games: pd.DataFrame,
    player_games: pd.DataFrame,
    shares: pd.DataFrame,
    possessions: pd.Series,
    tuning: npt.NDArray[np.bool_],
) -> BaselineResult:
    """The naive baseline: player PIR per 40 minutes, decayed and shrunk the same way, combined
    with the projected shares into a team value, then ``margin = a*home_flag + c*team_diff`` by
    OLS on tuning games. Fixed ``PIR_HALF_LIFE_DAYS``/``PIR_K_MINUTES`` (no grid parameter here).
    """
    game_ids = games["game_id"].astype(str).to_numpy()
    player_index = _player_index(player_games, shares)
    n_players = len(player_index)
    rows = _rows(games, player_games, player_index, ("pir",))
    lookup = _shares_lookup(shares, player_index)
    snapshots = _round_snapshots(
        games, rows, n_players, PIR_HALF_LIFE_DAYS, _RateSpec("minutes", 40.0)
    )
    z = _z_from_snapshots(snapshots, game_ids, lookup, PIR_K_MINUTES)  # (n_games, 1)
    team_diff = z[:, 0] / 5.0
    home_flag = _home_flag(games)
    p = possessions.reindex(games["game_id"]).to_numpy(dtype=np.float64)
    actual = _actual_margin(games)
    fittable = tuning & ~np.isnan(p) & ~np.isnan(actual)

    x = np.column_stack([home_flag, team_diff])
    theta = _ridge_fit(x[fittable], actual[fittable], np.full(2, UNPENALISED))
    a, c = float(theta[0]), float(theta[1])
    margin = x @ theta
    margin = np.where(np.isnan(p), np.nan, margin)

    params = {
        "half_life_days": PIR_HALF_LIFE_DAYS,
        "k": PIR_K_MINUTES,
        "a": _round(a),
        "c": _round(c),
    }
    return BaselineResult(margin.astype(np.float64), params)
