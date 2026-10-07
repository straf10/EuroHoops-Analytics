"""Monte Carlo season simulator: the rest of a regular season, its table and the knockouts.

Every simulation draws one set of team strengths (``StrengthSampler``) and keeps it for the
whole season and the knockouts. Games are drawn as ``margin = round(pace·(ORtg_H - ORtg_A)/100
+ scale·e)`` with ``e`` Student-t (or Normal); a margin of 0 becomes +1 or -1 with probability
1/2 (overtime is not modelled and overtime points count for no tie-break, Art. 19.4). Each final
table is ranked as ``standings.rank`` does: by wins, and only the tied groups go through the
Art. 19 procedure (``standings.rank_by_wins``) with just the games of their own teams.
Knockouts follow the format's ``series`` (EuroLeague Art. 18.2-18.4, GBL as ``GBL_2026``).

Random draws come from one ``np.random.default_rng(seed)`` in a fixed order, so the output is a
pure function of the inputs and the seed: strengths, regular-season noise (sims, games),
regular-season coin flips (sims, games), then per knockout stage one noise array
(sims, series, games) in bracket order (play-in A and B, C, playoffs, Final Four semifinals,
final; GBL: quarterfinals, semifinals, final).
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from eurohoops.models.elo import FloatArray
from eurohoops.models.team_eff import IntArray
from eurohoops.standings import Format, Result, rank_by_wins

StrengthSampler = Callable[[int, np.random.Generator], tuple[FloatArray, FloatArray, FloatArray]]
# (n_sims, rng) -> (off (sims, teams), def_ (sims, teams), home (sims,)), teams in `teams` order.
# μ is folded into off: ORtg_H = off[H] + home·hf - def[A], ORtg_A = off[A] - home·hf - def[H]
# (hf = 0 at a neutral venue). One draw per simulation, kept for the whole season.

# Games of a series hosted by the higher seed (D5): games 1, 2, 5 of a best of 5, 1 and 3 of a
# best of 3; a single game is at the better-placed team.
_HIGHER_SEED_HOSTS = {
    1: (True,),
    3: (True, False, True),
    5: (True, True, False, False, True),
}
_EL_PLAY_IN = ("play-in", "playoffs", "final four")
_EL_PLAIN = ("playoffs", "final four")
_GBL = ("quarterfinals", "semifinals", "final")
_COLUMNS = ("home", "away", "neutral")


@dataclass(frozen=True)
class NoiseModel:
    scale: float
    df: float | None  # Student-t degrees of freedom; None = Normal


@dataclass(frozen=True)
class PaceModel:
    mu: float
    team: FloatArray  # (teams,), `teams` order; a game's pace = mu + team[H] + team[A]


@dataclass(frozen=True)
class SimOutput:
    teams: tuple[str, ...]
    n_sims: int
    seed: int
    rank_counts: IntArray  # (teams, len(teams)): [i, j] = sims where team i finished j + 1
    wins: FloatArray  # (teams,) mean final regular-season wins (played + simulated)
    p_direct: FloatArray  # (teams,) P(final place in fmt.playoffs_direct)
    p_play_in: FloatArray  # (teams,) P(place in fmt.play_in); zeros without a play-in
    p_top10: FloatArray  # (teams,) P(place <= 10)
    p_playoffs: FloatArray  # (teams,) P(in the playoff bracket: EL playoffs / GBL quarterfinals)
    p_semis: FloatArray  # (teams,) EL: P(Final Four); GBL: P(semifinals)
    p_final: FloatArray  # (teams,) P(reached the final)
    p_title: FloatArray  # (teams,) P(won the title)


def _draw_noise(noise: NoiseModel, rng: np.random.Generator, size: tuple[int, ...]) -> FloatArray:
    if noise.df is None:
        return rng.standard_normal(size)
    return rng.standard_t(noise.df, size=size)


def _play_regular_season(
    *,
    off: FloatArray,
    def_: FloatArray,
    home: FloatArray,
    home_idx: IntArray,
    away_idx: IntArray,
    neutral: np.ndarray,
    pace: PaceModel,
    noise: NoiseModel,
    rng: np.random.Generator,
) -> tuple[IntArray, IntArray]:
    """(home points, away points), both (sims, games), of the remaining fixtures."""
    n_sims, n_games = off.shape[0], len(home_idx)
    shocks = _draw_noise(noise, rng, (n_sims, n_games))
    coins = rng.random((n_sims, n_games))
    host_edge = home[:, None] * (~neutral).astype(np.float64)
    ortg_home = off[:, home_idx] + host_edge - def_[:, away_idx]
    ortg_away = off[:, away_idx] - host_edge - def_[:, home_idx]
    game_pace = pace.mu + pace.team[home_idx] + pace.team[away_idx]
    margin = np.rint(game_pace * (ortg_home - ortg_away) / 100 + noise.scale * shocks)
    margin = np.where(margin == 0, np.where(coins < 0.5, 1.0, -1.0), margin)
    total = game_pace * (ortg_home + ortg_away) / 100
    home_points = np.rint((total + margin) / 2).astype(np.int64)
    return home_points, home_points - margin.astype(np.int64)


def _order_tables(
    *,
    wins: IntArray,
    deducted: IntArray,
    teams: Sequence[str],
    played: Sequence[Result],
    home_idx: IntArray,
    away_idx: IntArray,
    home_points: IntArray,
    away_points: IntArray,
) -> IntArray:
    """Team indices best first, (sims, teams); equals ``standings.rank`` on each full table.

    ``wins`` (sims, teams) are before deductions. When every net win total is unique the order
    is by wins; otherwise each tied group is resolved by ``rank_by_wins`` from the games of its
    own teams (played and simulated).
    """
    n_teams = len(teams)
    net = wins - deducted
    order = np.argsort(-net, axis=1, kind="stable")
    ranked = np.take_along_axis(net, order, axis=1)
    tied_sims = np.flatnonzero((ranked[:, 1:] == ranked[:, :-1]).any(axis=1))
    if not len(tied_sims):
        return order
    played_of: list[set[int]] = [set() for _ in teams]
    fixtures_of: list[set[int]] = [set() for _ in teams]
    index = {team: i for i, team in enumerate(teams)}
    for k, r in enumerate(played):
        played_of[index[r.home]].add(k)
        played_of[index[r.away]].add(k)
    home_names = [teams[i] for i in home_idx]
    away_names = [teams[i] for i in away_idx]
    for g, (h, a) in enumerate(zip(home_idx.tolist(), away_idx.tolist(), strict=True)):
        fixtures_of[h].add(g)
        fixtures_of[a].add(g)
    sanctions = {teams[i]: int(d) for i, d in enumerate(deducted) if d}
    for s in tied_sims.tolist():
        hp, ap = home_points[s].tolist(), away_points[s].tolist()
        values = ranked[s].tolist()
        start = 0
        while start < n_teams:
            stop = start + 1
            while stop < n_teams and values[stop] == values[start]:
                stop += 1
            if stop - start > 1:
                group = order[s, start:stop].tolist()
                games_played = set().union(*(played_of[i] for i in group))
                games_sim = set().union(*(fixtures_of[i] for i in group))
                results = [played[k] for k in sorted(games_played)] + [
                    Result(home_names[g], away_names[g], hp[g], ap[g]) for g in sorted(games_sim)
                ]
                resolved = rank_by_wins(
                    {teams[i]: int(wins[s, i]) for i in group},
                    results,
                    {teams[i]: sanctions[teams[i]] for i in group if teams[i] in sanctions},
                )
                order[s, start:stop] = [index[team] for team in resolved]
            start = stop
    return order


@dataclass(frozen=True)
class _Court:
    """The drawn strengths of every simulation and the final tables: plays knockout series."""

    off: FloatArray
    def_: FloatArray
    home: FloatArray
    order: IntArray  # (sims, teams) team index by regular-season place
    pace: PaceModel
    noise: NoiseModel
    rng: np.random.Generator

    def team(self, places: IntArray) -> IntArray:
        """Team index at each regular-season place (0-based) of (sims, k) ``places``."""
        return np.take_along_axis(self.order, places, axis=1)

    def series(
        self, place_a: IntArray, place_b: IntArray, best_of: int, *, neutral: bool = False
    ) -> IntArray:
        """Winning place of each (sims, k) series; the better-placed side hosts games per D5.

        All ``best_of`` games are drawn (the winner is the first to a majority either way); a
        game is won by the host when its unrounded margin is positive.
        """
        high, low = np.minimum(place_a, place_b), np.maximum(place_a, place_b)
        team_high, team_low = self.team(high), self.team(low)
        rows = np.arange(len(self.home))[:, None]
        shocks = _draw_noise(self.noise, self.rng, (*high.shape, best_of))
        edge = 0.0 if neutral else 1.0
        high_wins = np.zeros(high.shape, dtype=np.int64)
        for game, high_hosts in enumerate(_HIGHER_SEED_HOSTS[best_of]):
            host, guest = (team_high, team_low) if high_hosts else (team_low, team_high)
            ortg_host = self.off[rows, host] + self.home[:, None] * edge - self.def_[rows, guest]
            ortg_guest = self.off[rows, guest] - self.home[:, None] * edge - self.def_[rows, host]
            game_pace = self.pace.mu + self.pace.team[host] + self.pace.team[guest]
            margin = game_pace * (ortg_host - ortg_guest) / 100
            host_wins = margin + self.noise.scale * shocks[:, :, game] > 0
            high_wins += host_wins if high_hosts else ~host_wins
        return np.where(high_wins > best_of // 2, high, low)


def _places(n_sims: int, *places: int) -> IntArray:
    """(sims, len(places)) array repeating the given 0-based places."""
    return np.broadcast_to(np.array(places, dtype=np.int64), (n_sims, len(places)))


def _knockouts(
    court: _Court, fmt: Format, kind: tuple[str, ...]
) -> tuple[IntArray, IntArray, IntArray, IntArray]:
    """Teams (sims, k) in the bracket, the last four, the finalists and the champion."""
    n_sims = len(court.home)
    best_of = {s.name: s.best_of for s in fmt.series}
    if kind == _GBL:
        seeds = _places(n_sims, *range(8))
        quarters = court.series(seeds[:, :4], seeds[:, [7, 6, 5, 4]], best_of["quarterfinals"])
        semis = court.series(
            quarters[:, [0, 1]], quarters[:, [3, 2]], best_of["semifinals"]
        )  # 1/8 v 4/5, 2/7 v 3/6
        final = court.series(semis[:, :1], semis[:, 1:], best_of["final"])
        return court.team(seeds), court.team(quarters), court.team(semis), court.team(final)
    if kind == _EL_PLAY_IN:
        ahead = court.series(  # play-in A (7 v 8) and B (9 v 10)
            _places(n_sims, 6, 8), _places(n_sims, 7, 9), best_of["play-in"]
        )
        winner_a, winner_b = ahead[:, :1], ahead[:, 1:]
        winner_c = court.series(13 - winner_a, winner_b, best_of["play-in"])  # 13 = 6 + 7
        seeds = np.concatenate([_places(n_sims, *range(6)), winner_a, winner_c], axis=1)
    else:
        seeds = _places(n_sims, *range(8))
    quarters = court.series(  # 1v8, 4v5, 3v6, 2v7
        seeds[:, [0, 3, 2, 1]], seeds[:, [7, 4, 5, 6]], best_of["playoffs"]
    )
    semis = court.series(
        quarters[:, [0, 3]], quarters[:, [1, 2]], best_of["final four"], neutral=True
    )
    final = court.series(semis[:, :1], semis[:, 1:], best_of["final four"], neutral=True)
    return court.team(seeds), court.team(quarters), court.team(semis), court.team(final)


def _knockout_kind(fmt: Format) -> tuple[str, ...]:
    kind = tuple(s.name for s in fmt.series)
    if kind not in (_EL_PLAY_IN, _EL_PLAIN, _GBL):
        raise ValueError(f"unsupported knockout structure {kind}")
    if kind == _EL_PLAY_IN:
        shape_ok = fmt.playoffs_direct == tuple(range(1, 7)) and fmt.play_in == (7, 8, 9, 10)
    else:
        shape_ok = fmt.playoffs_direct == tuple(range(1, 9)) and not fmt.play_in
    if not shape_ok:
        raise ValueError(f"{kind} needs playoffs_direct and play_in places of the bracket")
    for s in fmt.series:
        if s.best_of not in _HIGHER_SEED_HOSTS:
            raise ValueError(f"series {s.name!r}: best of {s.best_of} is not supported")
    return kind


def simulate(
    *,
    teams: Sequence[str],
    played: Sequence[Result],
    remaining: pd.DataFrame,
    sampler: StrengthSampler,
    pace: PaceModel,
    noise: NoiseModel,
    fmt: Format,
    n_sims: int,
    seed: int,
) -> SimOutput:
    """Regular season: margin = round(pace·(ORtg_H - ORtg_A)/100 + scale·e), e ~ t(df) or N(0,1);
    a 0 becomes ±1 with probability ½; home points = round((total + margin)/2) with
    total = pace·(ORtg_H + ORtg_A)/100, away points = home points - margin. Final table per
    sim: wins (played + simulated) minus fmt.deducted_wins(); if all unique, ordered by wins,
    else standings.rank_by_wins. Knockouts per fmt (D5) with the same draw.

    ``remaining`` needs the columns ``home``, ``away`` and ``neutral`` (source team codes); its
    row order fixes which random draw belongs to which game.
    """
    kind = _knockout_kind(fmt)
    teams = tuple(teams)
    n_teams = len(teams)
    index = {team: i for i, team in enumerate(teams)}
    if len(index) != n_teams or n_teams != fmt.teams:
        raise ValueError(f"expected {fmt.teams} distinct teams, got {n_teams}")
    missing = [c for c in _COLUMNS if c not in remaining.columns]
    if missing:
        raise ValueError(f"remaining fixtures lack columns {missing}")
    deductions = fmt.deducted_wins()
    unknown = (
        {r.home for r in played}
        | {r.away for r in played}
        | set(remaining["home"])
        | set(remaining["away"])
        | set(deductions)
    ) - set(index)
    if unknown:
        raise ValueError(f"teams not in `teams`: {sorted(unknown)}")
    if len(pace.team) != n_teams:
        raise ValueError("pace.team must have one entry per team")

    rng = np.random.default_rng(seed)
    off, def_, home = sampler(n_sims, rng)
    home_idx = np.asarray(remaining["home"].map(index), dtype=np.int64)
    away_idx = np.asarray(remaining["away"].map(index), dtype=np.int64)
    home_points, away_points = _play_regular_season(
        off=off,
        def_=def_,
        home=home,
        home_idx=home_idx,
        away_idx=away_idx,
        neutral=remaining["neutral"].to_numpy(dtype=bool),
        pace=pace,
        noise=noise,
        rng=rng,
    )

    won = np.zeros(n_teams)
    for r in played:
        won[index[r.winner]] += 1
    home_won = (home_points > away_points).astype(np.float64)
    home_onehot = np.zeros((len(home_idx), n_teams))
    away_onehot = np.zeros((len(home_idx), n_teams))
    home_onehot[np.arange(len(home_idx)), home_idx] = 1.0
    away_onehot[np.arange(len(home_idx)), away_idx] = 1.0
    wins = np.rint(won + home_won @ home_onehot + (1.0 - home_won) @ away_onehot).astype(np.int64)
    deducted = np.array([deductions.get(team, 0) for team in teams], dtype=np.int64)
    order = _order_tables(
        wins=wins,
        deducted=deducted,
        teams=teams,
        played=played,
        home_idx=home_idx,
        away_idx=away_idx,
        home_points=home_points,
        away_points=away_points,
    )

    place = np.empty_like(order)
    np.put_along_axis(place, order, np.broadcast_to(np.arange(n_teams), order.shape), axis=1)
    rank_counts = np.bincount(
        (np.arange(n_teams) * n_teams + place).ravel(), minlength=n_teams * n_teams
    ).reshape(n_teams, n_teams)

    court = _Court(off, def_, home, order, pace, noise, rng)
    bracket, last_four, finalists, champion = _knockouts(court, fmt, kind)

    def share(teams_in: IntArray) -> FloatArray:
        return np.bincount(teams_in.ravel(), minlength=n_teams) / n_sims

    def cut(places: tuple[int, ...]) -> FloatArray:
        return rank_counts[:, [p - 1 for p in places]].sum(axis=1) / n_sims

    return SimOutput(
        teams=teams,
        n_sims=n_sims,
        seed=seed,
        rank_counts=rank_counts,
        wins=wins.mean(axis=0),
        p_direct=cut(fmt.playoffs_direct),
        p_play_in=cut(fmt.play_in),
        p_top10=cut(tuple(range(1, min(10, n_teams) + 1))),
        p_playoffs=share(bracket),
        p_semis=share(last_four),
        p_final=share(finalists),
        p_title=share(champion),
    )
