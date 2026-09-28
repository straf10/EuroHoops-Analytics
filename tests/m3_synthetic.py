"""Synthetic EuroLeague-like seasons with known player RAPM ratings (week 9-12 H1).

Used by the recovery test (``test_rapm.py``), the leakage tests and the benchmark script
(``scripts/bench_m3_rapm.py``); kept general so subagents E and F (``rapm_dummy``, ``rapm_spm``)
can reuse it for their own tests instead of writing a second generator.

Each player keeps one true offensive and one true defensive rating for the player's whole
synthetic career (whatever teams or seasons he appears in), so a recovered RAPM rating can be
compared against a single number per player. Lineups are random every stint; a team's roster
turns over between seasons by ``turnover`` (a season's roster keeps ``1 - turnover`` of the
previous season's, at random, and fills the rest with brand-new players), so a player's spells
span at most a few consecutive seasons with one team, and some players change teams entirely
(dropped from a roster, reappear only if re-drawn -- not forced), like real free agency/trades.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import count

import numpy as np
import pandas as pd

from eurohoops.parse.games import conform
from eurohoops.parse.schemas import validated
from eurohoops.parse.stints_mart import STINT_COLUMNS
from eurohoops.parse.team_box import TEAM_GAMES_SCHEMA
from eurohoops.stats.box import PLAYER_GAMES_SCHEMA

STINT_SECONDS = 73  # ~2,400 s / 33 stints, close to EuroLeague's real average
GAME_SECONDS_NOMINAL = 2400.0  # 4 x 600 s quarters (synthetic games never go to overtime)
BASE_RATE = 100.0  # league-average points per 100 possessions
GAME_CHECKS = ("five_on_court", "seconds", "minutes", "points", "possessions")


@dataclass(frozen=True)
class SyntheticM3:
    games: pd.DataFrame
    stints: pd.DataFrame
    checks: pd.DataFrame  # stint_game_checks mart shape
    team_games: pd.DataFrame
    player_games: pd.DataFrame
    true_o: dict[str, float]
    true_d: dict[str, float]


def _rosters(
    rng: np.random.Generator,
    seasons: Sequence[int],
    teams: Sequence[str],
    roster: int,
    turnover: float,
) -> dict[tuple[str, int], list[str]]:
    counter = count(1)

    def new_player() -> str:
        return f"P{next(counter):06d}"

    rosters: dict[tuple[str, int], list[str]] = {}
    for i, season in enumerate(seasons):
        for team in teams:
            if i == 0:
                rosters[(team, season)] = [new_player() for _ in range(roster)]
                continue
            previous = rosters[(team, seasons[i - 1])]
            keep_n = max(1, round(roster * (1.0 - turnover)))
            keep = list(rng.choice(previous, size=min(keep_n, len(previous)), replace=False))
            rosters[(team, season)] = keep + [new_player() for _ in range(roster - len(keep))]
    return rosters


@dataclass
class _GameSim:
    """One simulated game's stints and the on-court bookkeeping its box rows need."""

    stint_rows: list[dict[str, object]] = field(default_factory=list)
    home_total: int = 0
    away_total: int = 0
    home_poss_total: int = 0
    away_poss_total: int = 0
    seconds_on_court: dict[str, float] = field(default_factory=dict)
    points_while_on: dict[str, float] = field(default_factory=dict)
    last_home_five: list[str] = field(default_factory=list)


def _simulate_stints(
    rng: np.random.Generator,
    *,
    game_id: str,
    season: int,
    home: str,
    away: str,
    home_roster: Sequence[str],
    away_roster: Sequence[str],
    stints_per_game: int,
    neutral: bool,
    home_effect: float,
    noise_sd: float,
    true_o: dict[str, float],
    true_d: dict[str, float],
) -> _GameSim:
    sim = _GameSim()
    home_effect_signed = 0.0 if neutral else home_effect
    for i in range(stints_per_game):
        start_s, end_s = i * STINT_SECONDS, (i + 1) * STINT_SECONDS
        home_five = sorted(rng.choice(home_roster, size=5, replace=False))
        away_five = sorted(rng.choice(away_roster, size=5, replace=False))
        sim.last_home_five = home_five
        home_poss = int(rng.integers(1, 4))
        away_poss = int(rng.integers(1, 4))
        home_rate = (
            BASE_RATE
            + home_effect_signed
            + sum(true_o[p] for p in home_five)
            - sum(true_d[p] for p in away_five)
            + rng.normal(0.0, noise_sd)
        )
        away_rate = (
            BASE_RATE
            - home_effect_signed
            + sum(true_o[p] for p in away_five)
            - sum(true_d[p] for p in home_five)
            + rng.normal(0.0, noise_sd)
        )
        home_points = max(0, round(home_rate * home_poss / 100.0))
        away_points = max(0, round(away_rate * away_poss / 100.0))
        sim.home_total += home_points
        sim.away_total += away_points
        sim.home_poss_total += home_poss
        sim.away_poss_total += away_poss
        duration = float(end_s - start_s)
        for p in home_five + away_five:
            sim.seconds_on_court[p] = sim.seconds_on_court.get(p, 0.0) + duration
        for p in home_five:
            sim.points_while_on[p] = sim.points_while_on.get(p, 0.0) + home_points
        for p in away_five:
            sim.points_while_on[p] = sim.points_while_on.get(p, 0.0) + away_points
        sim.stint_rows.append(
            {
                "game_id": game_id,
                "season": season,
                "period": 1,
                "start_s": start_s,
                "end_s": end_s,
                "home": home,
                "away": away,
                "home_players": list(home_five),
                "away_players": list(away_five),
                "home_points": home_points,
                "away_points": away_points,
                "home_poss": home_poss,
                "away_poss": away_poss,
            }
        )
    if sim.home_total == sim.away_total:  # basketball has no draws (GAMES_SCHEMA)
        sim.home_total += 1
        sim.stint_rows[-1]["home_points"] += 1
        for p in sim.last_home_five:
            sim.points_while_on[p] = sim.points_while_on.get(p, 0.0) + 1.0 / len(sim.last_home_five)
    return sim


def _team_game_rows(
    sim: _GameSim, *, season: int, game_id: str, home: str, away: str, neutral: bool
) -> list[dict[str, object]]:
    shared_poss = (sim.home_poss_total + sim.away_poss_total) / 2.0
    rows = []
    for team, opponent, own_points, own_poss, is_home in (
        (home, away, sim.home_total, sim.home_poss_total, True),
        (away, home, sim.away_total, sim.away_poss_total, False),
    ):
        rows.append(
            {
                "competition": "euroleague",
                "season": season,
                "game_id": game_id,
                "team": team,
                "opponent": opponent,
                "home": is_home and not neutral,
                "points": own_points,
                "fga": max(own_poss, 1),
                "fta": 10,
                "oreb": 8,
                "dreb": 20,
                "tov": 10,
                "minutes": 40.0,
                "poss_raw": float(own_poss),
                "poss_game": float(shared_poss),
                "source": "euroleague_box",
            }
        )
    return rows


def _player_game_rows(
    sim: _GameSim,
    *,
    season: int,
    game_id: str,
    home: str,
    away: str,
    home_roster: Sequence[str],
    away_roster: Sequence[str],
    neutral: bool,
) -> list[dict[str, object]]:
    rows = []
    for team, opponent, own_points, is_home, roster_list in (
        (home, away, sim.home_total, True, home_roster),
        (away, home, sim.away_total, False, away_roster),
    ):
        team_seconds = sum(sim.seconds_on_court.get(p, 0.0) for p in roster_list)
        for p in roster_list:
            sec = sim.seconds_on_court.get(p, 0.0)
            if sec <= 0:
                continue
            pts = round(sim.points_while_on.get(p, 0.0) * sec / max(team_seconds, 1.0))
            rows.append(
                {
                    "season": season,
                    "game_id": game_id,
                    "team": team,
                    "opponent": opponent,
                    "venue": "neutral" if neutral else ("home" if is_home else "away"),
                    "won": own_points > (sim.away_total if is_home else sim.home_total),
                    "player_id": p,
                    "player": p,
                    "dorsal": "0",
                    "starter": False,
                    "sec": int(sec),
                    "pts": int(pts),
                    "fg2m": 0,
                    "fg2a": 0,
                    "fg3m": 0,
                    "fg3a": 0,
                    "ftm": 0,
                    "fta": 0,
                    "oreb": 0,
                    "dreb": 0,
                    "ast": 0,
                    "stl": 0,
                    "tov": 0,
                    "blk": 0,
                    "blka": 0,
                    "pf": 0,
                    "fd": 0,
                    "pir": int(pts),
                    "pm": 0,
                    "poss": float(sec),
                    "game_sec": int(GAME_SECONDS_NOMINAL),
                }
            )
    return rows


def make_synthetic_m3(
    seasons: Sequence[int],
    *,
    teams: int = 18,
    games_per_season: int = 300,
    stints_per_game: int = 33,
    roster: int = 12,
    seed: int = 20261001,
    turnover: float = 0.25,
    checks_pass_rate: float = 1.0,
    home_effect: float = 4.0,
    rating_sd: float = 6.0,
    noise_sd: float = 9.0,
    neutral_every: int = 0,
) -> SyntheticM3:
    """A synthetic multi-season history with known per-player O/D ratings.

    ``checks_pass_rate``: the fraction of games whose ``stint_game_checks.passed`` is True (the
    rest are marked failed with a planted reason, to exercise the fit-vs-evaluate split); their
    stints are still generated and still byte-for-byte consistent (final score = stint sum), the
    failure is only in the checks table, exactly as the real mart never drops flagged games.
    """
    rng = np.random.default_rng(seed)
    team_codes = [f"T{i:02d}" for i in range(teams)]
    rosters = _rosters(rng, list(seasons), team_codes, roster, turnover)
    players = sorted({p for roster_list in rosters.values() for p in roster_list})
    true_o = {p: float(rng.normal(0.0, rating_sd)) for p in players}
    true_d = {p: float(rng.normal(0.0, rating_sd)) for p in players}

    games_rows: list[dict[str, object]] = []
    stint_rows: list[dict[str, object]] = []
    check_rows: list[dict[str, object]] = []
    team_game_rows: list[dict[str, object]] = []
    player_game_rows: list[dict[str, object]] = []

    per_round = max(2, teams // 2)
    rounds = max(1, math.ceil(games_per_season / per_round))
    game_no = 0
    for season in seasons:
        season_start = pd.Timestamp(f"{season}-10-01T18:00:00Z")
        played = 0
        for rnd in range(1, rounds + 1):
            for _ in range(per_round):
                if played >= games_per_season:
                    break
                home, away = rng.choice(team_codes, size=2, replace=False)
                game_id = f"SYN{season}_{game_no:06d}"
                tipoff = season_start + pd.Timedelta(days=(rnd - 1) * 7)
                neutral = bool(neutral_every) and game_no % neutral_every == 0
                home_roster, away_roster = rosters[(home, season)], rosters[(away, season)]
                sim = _simulate_stints(
                    rng,
                    game_id=game_id,
                    season=season,
                    home=home,
                    away=away,
                    home_roster=home_roster,
                    away_roster=away_roster,
                    stints_per_game=stints_per_game,
                    neutral=neutral,
                    home_effect=home_effect,
                    noise_sd=noise_sd,
                    true_o=true_o,
                    true_d=true_d,
                )
                stint_rows += sim.stint_rows
                games_rows.append(
                    {
                        "game_id": game_id,
                        "season": season,
                        "game_code": game_no + 1,
                        "phase": "RS",
                        "round": rnd,
                        "round_label": f"Round {rnd}",
                        "tipoff_utc": tipoff,
                        "home": home,
                        "away": away,
                        "home_score": sim.home_total,
                        "away_score": sim.away_total,
                        "played": True,
                        "forfeit": False,
                        "neutral": neutral,
                        "confirmed_date": True,
                    }
                )
                passed = bool(rng.random() < checks_pass_rate)
                check_rows.append(
                    {
                        "game_id": game_id,
                        "season": season,
                        **dict.fromkeys(GAME_CHECKS, True),
                        "passed": passed,
                        "reasons": "" if passed else "synthetic: planted failure",
                    }
                )
                team_game_rows += _team_game_rows(
                    sim, season=season, game_id=game_id, home=home, away=away, neutral=neutral
                )
                player_game_rows += _player_game_rows(
                    sim,
                    season=season,
                    game_id=game_id,
                    home=home,
                    away=away,
                    home_roster=home_roster,
                    away_roster=away_roster,
                    neutral=neutral,
                )
                game_no += 1
                played += 1

    games = conform(pd.DataFrame(games_rows))
    stints = pd.DataFrame(stint_rows, columns=list(STINT_COLUMNS)).astype(
        {"season": "int64", "period": "int64", "start_s": "int64", "end_s": "int64"}
    )
    checks = pd.DataFrame(check_rows).astype({"season": "int64"})
    team_columns = list(TEAM_GAMES_SCHEMA.columns)
    team_games = validated(pd.DataFrame(team_game_rows, columns=team_columns), TEAM_GAMES_SCHEMA)
    player_columns = list(PLAYER_GAMES_SCHEMA.columns)
    player_games = validated(
        pd.DataFrame(player_game_rows, columns=player_columns), PLAYER_GAMES_SCHEMA
    )
    return SyntheticM3(games, stints, checks, team_games, player_games, true_o, true_d)
