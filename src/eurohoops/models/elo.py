"""M0: FiveThirtyEight-style Elo with MOV multiplier, home-court advantage and season reversion.

``replay`` is one chronological O(N) pass. Every game gets a pre-game rating difference computed
from games processed before it; only played games then update ratings, so a game's prediction
can never see its own result or anything later.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

INITIAL_RATING = 1500.0

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class EloParams:
    k: float
    hca: float
    reversion: float  # weight pulled toward INITIAL_RATING at each season start


@dataclass(frozen=True)
class GameArrays:
    """Columns of a tipoff-sorted games table, extracted once and reused across parameter sets."""

    season: list[int]
    home: list[str]
    away: list[str]
    neutral: list[bool]
    played: list[bool]  # played and rated: forfeits (20-0 by decision) do not update ratings
    margin: list[int]  # home - away; 0 for unplayed games (never read for them)


def prepare(games: pd.DataFrame) -> GameArrays:
    """``games`` must already be sorted by tip-off (``build_games_table`` guarantees it)."""
    if not games["tipoff_utc"].is_monotonic_increasing:
        raise ValueError("games must be sorted by tipoff_utc")
    margin = (games["home_score"] - games["away_score"]).fillna(0).astype("int64")
    return GameArrays(
        season=games["season"].tolist(),
        home=games["home"].tolist(),
        away=games["away"].tolist(),
        neutral=games["neutral"].tolist(),
        played=(games["played"] & ~games["forfeit"]).tolist(),
        margin=margin.tolist(),
    )


def win_probability(diff: float) -> float:
    """P(home win) for a rating difference that already includes home-court advantage."""
    return float(1.0 / (1.0 + 10.0 ** (-diff / 400.0)))


def mov_multiplier(mov: int, winner_diff: float) -> float:
    """538's margin-of-victory multiplier; ``winner_diff`` = winner - loser rating incl. HCA."""
    return float((mov + 3) ** 0.8 / (7.5 + 0.006 * winner_diff))


def replay(games: GameArrays, params: EloParams) -> FloatArray:
    """Return the pre-game rating difference (home - away + HCA) for every game."""
    return _run(games, params)[0]


def season_ratings(
    games: GameArrays, params: EloParams, season: int
) -> dict[str, tuple[float, float]]:
    """Each ``season`` team's (current rating, rating at that season's start after reversion)."""
    _, ratings, starts = _run(games, params)
    teams = {
        t
        for s, h, a in zip(games.season, games.home, games.away, strict=True)
        if s == season
        for t in (h, a)
    }
    start = starts.get(season, {})
    return {t: (ratings[t], start.get(t, INITIAL_RATING)) for t in sorted(teams)}


def _run(
    games: GameArrays, params: EloParams
) -> tuple[FloatArray, dict[str, float], dict[int, dict[str, float]]]:
    """One chronological pass: pre-game diffs, final ratings and a snapshot at each season start."""
    ratings: dict[str, float] = {}
    starts: dict[int, dict[str, float]] = {}
    diffs = np.empty(len(games.home), dtype=np.float64)
    current_season: int | None = None
    keep = 1.0 - params.reversion
    for i, (season, home, away, neutral, played, margin) in enumerate(
        zip(
            games.season,
            games.home,
            games.away,
            games.neutral,
            games.played,
            games.margin,
            strict=True,
        )
    ):
        if season != current_season:
            current_season = season
            for team, rating in ratings.items():
                ratings[team] = keep * rating + params.reversion * INITIAL_RATING
            starts[season] = dict(ratings)
        home_rating = ratings.setdefault(home, INITIAL_RATING)
        away_rating = ratings.setdefault(away, INITIAL_RATING)
        diff = home_rating - away_rating + (0.0 if neutral else params.hca)
        diffs[i] = diff
        if not played:
            continue
        home_won = margin > 0
        winner_diff = diff if home_won else -diff
        shift = (
            params.k
            * mov_multiplier(abs(margin), winner_diff)
            * ((1.0 if home_won else 0.0) - win_probability(diff))
        )
        ratings[home] = home_rating + shift
        ratings[away] = away_rating - shift
    return diffs, ratings, starts


def fit_margin_scale(diffs: FloatArray, margins: FloatArray) -> float:
    """Least-squares ``s`` in ``margin ~ diff / s`` (regression through the origin)."""
    return float(np.dot(diffs, diffs) / np.dot(diffs, margins))
