"""Projected-share variants for the M5 game predictor (week 14-16, J2).

Same shares as :mod:`eurohoops.models.minutes` (5 x player seconds / team seconds, capped at 1,
no renormalisation after the cap), projected from the team's earlier games under three rules:

- ``proj_hc``: H-c, the team's previous ``n_games`` games (``minutes.projected_shares``).
- ``proj_decay`` (D2): *all* the team's earlier rated games of the season with player rows,
  weighted ``0.5 ** (k / half_life_games)`` (k = 0 for the most recent game, 1 for the one before,
  ...): share = min(1, 5 x sum(w · sec) / sum(w · team_sec)). Before the team's first such game
  the whole previous season with that team counts, equally weighted (as in H-c).
  ``half_life_games=math.inf`` weights every game equally.
- ``proj_avail``: ``proj_decay``, then a player absent (no row, or 0 seconds) from each of the
  team's last ``absent_games`` earlier games of the season is dropped, and the others' shares are
  scaled so the team's share sum is unchanged, then capped at 1. Nobody is dropped before the
  team has ``absent_games`` earlier games this season (the previous-season fallback included),
  nor if that would drop every player.

Walk-forward safe like H-c: a game's projection reads only games that tipped off before its round
cutoff (the first tip-off of its ``(season, phase, round)``), never the game's own rows.
"""

import numpy as np
import pandas as pd

from eurohoops.models import minutes
from eurohoops.models.elo import FloatArray
from eurohoops.parse.schemas import validated

VARIANTS = ("proj_hc", "proj_decay", "proj_avail")


def _decay_weights(count: int, half_life_games: float) -> FloatArray:
    """Weights of ``count`` games in time order: the most recent has k = 0 and weight 1."""
    ages = np.arange(count - 1, -1, -1, dtype=np.float64)
    weights: FloatArray = 0.5 ** (ages / half_life_games)
    return weights


def _weighted_shares(rows: pd.DataFrame, denominator: float) -> dict[str, float]:
    """Share of each player with weighted seconds: ``rows`` carries ``player_id, sec, weight``."""
    by_player = (rows["sec"] * rows["weight"]).groupby(rows["player_id"]).sum().sort_index()
    return {
        str(p): min(1.0, minutes.ON_COURT * float(s) / denominator)
        for p, s in by_player.items()
        if s > 0
    }


def _played_in(rows: pd.DataFrame, recent: pd.Series) -> set[str]:
    """Players with seconds in at least one of the ``recent`` games (``rows``: the team's rows)."""
    on_court = rows[rows["game_id"].isin(recent) & (rows["sec"] > 0)]
    return {str(p) for p in on_court["player_id"]}


def _drop_absent(shares: dict[str, float], played: set[str]) -> dict[str, float]:
    """Drop the players who played in none of the recent games; rescale the rest to the old sum."""
    kept = {p: s for p, s in shares.items() if p in played}
    if not kept or len(kept) == len(shares):
        return shares
    scale = sum(shares.values()) / sum(kept.values())
    return {p: min(1.0, s * scale) for p, s in kept.items()}


def projected_shares_variant(
    variant: str,
    games: pd.DataFrame,
    player_games: pd.DataFrame,
    *,
    n_games: int = 5,
    half_life_games: float = 4.0,
    absent_games: int = 2,
) -> pd.DataFrame:
    """Projected shares of both sides of every game in ``games`` under ``variant``.

    ``variant``: ``proj_hc`` (``n_games`` window), ``proj_decay`` (``half_life_games``) or
    ``proj_avail`` (``proj_decay`` plus the ``absent_games`` rule); see the module docstring.
    ``player_games``: one row per player per game with ``game_id, team, player_id, sec``.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown projection variant {variant!r}; expected one of {VARIANTS}")
    if variant == "proj_hc":
        return minutes.projected_shares(games, player_games, n_games)
    team_games = minutes._team_games(games, player_games)
    rows_by = {key: frame for key, frame in player_games.groupby(["game_id", "team"])}
    by_team = {key: frame for key, frame in team_games.groupby(["team", "season"])}
    out: list[dict[str, object]] = []
    for game_id, season, home, away, cutoff in zip(
        games["game_id"],
        games["season"].astype(int),
        games["home"],
        games["away"],
        minutes.round_cutoffs(games),
        strict=True,
    ):
        for side, team in zip(minutes.SIDES, (str(home), str(away)), strict=True):
            played = by_team.get((team, season))
            earlier = played[played["time"] < cutoff] if played is not None else None
            this_season = earlier is not None and not earlier.empty
            if not this_season:
                earlier = by_team.get((team, season - 1))
            if earlier is None or earlier.empty:
                continue
            if this_season:
                weights = _decay_weights(len(earlier), half_life_games)
            else:
                weights = np.ones(len(earlier), dtype=np.float64)
            denominator = float(np.dot(weights, earlier["team_sec"].to_numpy(dtype=np.float64)))
            rows = pd.concat(
                [
                    rows_by[(g, team)].assign(weight=w)
                    for g, w in zip(earlier["game_id"], weights, strict=True)
                ]
            )
            shares = _weighted_shares(rows, denominator)
            if variant == "proj_avail" and this_season and len(earlier) >= absent_games:
                recent = earlier["game_id"].tail(absent_games)
                shares = _drop_absent(shares, _played_in(rows, recent))
            for player, share in shares.items():
                out.append(
                    {
                        "game_id": str(game_id),
                        "side": side,
                        "team": team,
                        "player_id": player,
                        "share": share,
                    }
                )
    return validated(
        pd.DataFrame(out, columns=list(minutes.SHARES_SCHEMA.columns)), minutes.SHARES_SCHEMA
    )
