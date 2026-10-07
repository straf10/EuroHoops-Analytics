"""A small synthetic league for the M7 harness (week 14-16 K4).

Reused by ``test_m7_backtest.py`` and the leakage suite: ``build_inputs`` returns an ``M7Inputs``,
``small_spec`` the matching ``M7Backtest`` (200 simulations, its own season splits) and
``synthetic_formats`` the injectable ``formats`` callable of ``run_m7_backtest``.

Twelve teams play seasons 2015-2021. Each regular season is a double round robin of 22 weekly
rounds (6 games a round at 15-minute slots) with a hidden team effect, a home edge and normal
noise; then two playoff games (phase ``PO``) and three Final Four games (phase ``FF``, neutral)
among the four strongest teams, so both the EuroLeague (last ``FF`` game) and the GBL (last
``PO`` game) champion rules find a winner. 2015-2016 only warm M1 up. The EuroLeague-shaped
format has a play-in from 2019 (6 direct places, play-in 7-10) and 8 direct places before;
T03 starts 2018 with 4 points (2 wins) deducted. The GBL-shaped format has 8 direct places
and the quarterfinal / semifinal / final series.
"""

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eurohoops.config import M7Backtest
from eurohoops.eval.m7_backtest import M7Inputs
from eurohoops.parse.games import conform
from eurohoops.standings import Format, Series
from tests.conftest import make_team_games

TEAMS = tuple(f"T{i:02d}" for i in range(1, 13))
SEASONS = tuple(range(2015, 2022))
ROUNDS = 22
PLAY_IN_FROM = 2019
DEDUCTION = (2018, "T03", 4)  # (season, team, points)
HOME_EDGE = 3.0
TUNED_M1: dict[str, Any] = {
    "rating": {"half_life_days": 365.0, "carry": 1.0, "ridge": 100.0},
    "pace": {"half_life_days": 365.0, "carry": 1.0, "ridge": 2.0},
    "margin": {"variant": "student_t_const", "scale": 10.0, "df": 7.0, "ref_pace": None},
    "totals_sigma": 16.0,
}
TUNED_M1_GBL: dict[str, Any] = {
    **TUNED_M1,
    "margin": {"variant": "student_t_pace", "scale": 11.4, "df": 20.0, "ref_pace": 70.0},
}
ELO: dict[str, Any] = {
    "k": 20.0,
    "hca": 90.0,
    "reversion": 0.25,
    "margin_scale": 23.358237,
    "margin_sigma": 11.882097,
}


def small_spec() -> M7Backtest:
    return M7Backtest(
        report=Path("unused.json"),
        teams_report=Path("unused.csv"),
        tuning=(2017, 2018),
        validation=(2019,),
        test=(2020, 2021),
        n_sims=200,
        bootstrap_resamples=200,
    )


def synthetic_formats(competition: str, season: int) -> Format:
    """The 12-team format of ``competition`` ("euroleague" or "gbl") in ``season``."""
    deduction = (
        ((DEDUCTION[1], DEDUCTION[2]),)
        if season == DEDUCTION[0] and competition == "euroleague"
        else ()
    )
    if competition == "gbl":
        direct, play_in = tuple(range(1, 9)), ()
        series = (Series("quarterfinals", 3), Series("semifinals", 3), Series("final", 5))
    elif season >= PLAY_IN_FROM:
        direct, play_in = tuple(range(1, 7)), (7, 8, 9, 10)
        series = (Series("play-in", 1), Series("playoffs", 5), Series("final four", 1))
    else:
        direct, play_in = tuple(range(1, 9)), ()
        series = (Series("playoffs", 5), Series("final four", 1))
    return Format(
        competition=competition,
        season=f"{season}-{(season + 1) % 100:02d}",
        teams=len(TEAMS),
        regular_season_rounds=ROUNDS,
        playoffs_direct=direct,
        play_in=play_in,
        eliminated=tuple(range(1 + len(direct) + len(play_in), len(TEAMS) + 1)),
        relegated=(),
        series=series,
        sources=("synthetic",),
        points_deducted=deduction,
    )


def _legs(teams: tuple[str, ...]) -> list[list[tuple[str, str]]]:
    """A double round robin by the circle method: every team plays once per round."""
    rotation = list(teams)
    half = len(teams) // 2
    first = []
    for r in range(len(teams) - 1):
        pairs = [(rotation[i], rotation[-1 - i]) for i in range(half)]
        first.append(pairs if r % 2 == 0 else [(b, a) for a, b in pairs])
        rotation = [rotation[0], rotation[-1], *rotation[1:-1]]
    return first + [[(b, a) for a, b in pairs] for pairs in first]


def _game(
    rng: np.random.Generator,
    effect: dict[str, float],
    season: int,
    code: int,
    *,
    phase: str,
    round_: int,
    tipoff: pd.Timestamp,
    home: str,
    away: str,
    neutral: bool = False,
) -> dict[str, Any]:
    margin = (0.0 if neutral else HOME_EDGE) + effect[home] - effect[away] + rng.normal(0.0, 9.0)
    diff = round(margin) or 1
    base = 72 + int(rng.integers(0, 10))
    return {
        "game_id": f"E{season}_{code}",
        "season": season,
        "game_code": code,
        "phase": phase,
        "round": round_,
        "round_label": f"{phase} {round_}",
        "tipoff_utc": tipoff,
        "home": home,
        "away": away,
        "home_score": base + max(diff, 0),
        "away_score": base + max(-diff, 0),
        "played": True,
        "forfeit": False,
        "neutral": neutral,
        "confirmed_date": True,
    }


def _season(
    rng: np.random.Generator, season: int, effect: dict[str, float]
) -> list[dict[str, Any]]:
    start = pd.Timestamp(f"{season}-10-01T18:00:00Z")
    games: list[dict[str, Any]] = []
    for r, pairs in enumerate(_legs(TEAMS), start=1):
        for slot, (home, away) in enumerate(pairs):
            tipoff = start + pd.Timedelta(days=7 * (r - 1), minutes=15 * slot)
            games.append(
                _game(
                    rng,
                    effect,
                    season,
                    len(games) + 1,
                    phase="RS",
                    round_=r,
                    tipoff=tipoff,
                    home=home,
                    away=away,
                )
            )
    top = sorted(TEAMS, key=lambda t: -effect[t])[:4]
    later = start + pd.Timedelta(days=7 * ROUNDS)

    def play(phase: str, round_: int, day: int, home: str, away: str, *, neutral: bool) -> str:
        games.append(
            _game(
                rng,
                effect,
                season,
                len(games) + 1,
                phase=phase,
                round_=round_,
                tipoff=later + pd.Timedelta(days=day),
                home=home,
                away=away,
                neutral=neutral,
            )
        )
        last = games[-1]
        return str(last["home"] if last["home_score"] > last["away_score"] else last["away"])

    play("PO", 1, 0, top[0], top[3], neutral=False)
    play("PO", 1, 1, top[1], top[2], neutral=False)
    first = play("FF", 1, 7, top[0], top[3], neutral=True)
    second = play("FF", 1, 7, top[1], top[2], neutral=True)
    play("FF", 2, 9, first, second, neutral=True)
    return games


def build_inputs(competition: str = "euroleague", seed: int = 0) -> M7Inputs:
    """The synthetic league of the module docstring, deterministic in ``seed``."""
    rng = np.random.default_rng(seed)
    effect = {team: float(rng.normal(0.0, 4.0)) for team in TEAMS}
    games: list[dict[str, Any]] = []
    for season in SEASONS:
        games += _season(rng, season, effect)
    table = conform(pd.DataFrame(games)).sort_values("tipoff_utc").reset_index(drop=True)
    return M7Inputs(
        competition=competition,
        games=table,
        team_games=make_team_games(table),
        regulation={},
        tuned_m1=copy.deepcopy(TUNED_M1_GBL if competition == "gbl" else TUNED_M1),
        elo=dict(ELO),
    )
