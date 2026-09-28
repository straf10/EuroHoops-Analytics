"""Projected on-court shares and expected possessions for player-based margin forecasts (M3).

A player's *share* is the fraction of the game he is on court (seconds / game seconds), so a
team's shares add up to 5. Week 9-12 H-c:

- **Projected share** for a game: the player's seconds over the team's previous
  ``n_games`` games of the same season with player rows, all of which tipped off before the
  game's round cutoff (the first tip-off of its ``(season, phase, round)``), divided by those
  games' seconds. Before the team's first such game: its share over the whole previous season
  with that team; players who were not on the team then get nothing (0).
- **Oracle share**: the game's own seconds (labelled "oracle, not a forecast"; never gated).
- **Expected possessions** P (per 40 minutes): the mean of the two teams' season-to-date pace
  (``poss_game · 40 / minutes``) over games before the cutoff; a team without such games uses
  its previous-season mean, then the league mean of every earlier team-game.

Only rated games (played, not forfeit) feed any of these; every game in ``games`` gets a value.
"""

import math

import numpy as np
import pandas as pd
import pandera.pandas as pa

from eurohoops.models.elo import FloatArray
from eurohoops.parse.schemas import validated

MINUTES_PER_GAME = 40.0
SIDES = ("home", "away")

SHARES_SCHEMA = pa.DataFrameSchema(
    {
        "game_id": pa.Column(str),
        "side": pa.Column(str, pa.Check.isin(SIDES)),
        "team": pa.Column(str),
        "player_id": pa.Column(str),
        "share": pa.Column("float64", [pa.Check.ge(0.0), pa.Check.le(1.0 + 1e-9)]),
    },
    unique=["game_id", "side", "player_id"],
    strict=True,
)


def _epoch(times: pd.Series) -> FloatArray:
    """Epoch seconds, whatever the timestamps' resolution (s from literals, ns from DuckDB)."""
    epoch = pd.Timestamp(0, tz="UTC")
    seconds: FloatArray = (
        (times.dt.tz_convert("UTC") - epoch).dt.total_seconds().to_numpy(dtype=np.float64)
    )
    return seconds


def round_cutoffs(games: pd.DataFrame) -> pd.Series:
    """Epoch seconds of the first tip-off of each game's ``(season, phase, round)``."""
    tipoff = pd.Series(_epoch(games["tipoff_utc"]), index=games.index)
    first = tipoff.groupby([games["season"], games["phase"], games["round"]]).transform("min")
    return first.rename("cutoff")


def _rated(games: pd.DataFrame) -> pd.DataFrame:
    return games[games["played"] & ~games["forfeit"]]


def _team_games(games: pd.DataFrame, player_games: pd.DataFrame) -> pd.DataFrame:
    """One row per (rated game with player rows, team): season, tip-off, game seconds."""
    rated = _rated(games)[["game_id", "season", "tipoff_utc"]]
    seconds = player_games.groupby(["game_id", "team"], as_index=False)["game_sec"].first()
    out = seconds.merge(rated, on="game_id")
    out["time"] = _epoch(out["tipoff_utc"])
    ordered: pd.DataFrame = out.sort_values(["time", "game_id", "team"]).reset_index(drop=True)
    return ordered


def _shares(rows: pd.DataFrame, seconds: float) -> dict[str, float]:
    by_player = rows.groupby("player_id")["sec"].sum().sort_index()
    return {str(p): float(s) / seconds for p, s in by_player.items() if s > 0}


def projected_shares(
    games: pd.DataFrame, player_games: pd.DataFrame, n_games: int = 5
) -> pd.DataFrame:
    """Projected shares of both sides of every game in ``games`` (see the module docstring).

    ``player_games``: one row per player per game with ``game_id, team, player_id, sec,
    game_sec`` (the game's length in seconds).
    """
    team_games = _team_games(games, player_games)
    rows_by = {key: frame for key, frame in player_games.groupby(["game_id", "team"])}
    by_team = {key: frame for key, frame in team_games.groupby(["team", "season"])}
    out: list[dict[str, object]] = []
    for game_id, season, home, away, cutoff in zip(
        games["game_id"],
        games["season"].astype(int),
        games["home"],
        games["away"],
        round_cutoffs(games),
        strict=True,
    ):
        for side, team in zip(SIDES, (str(home), str(away)), strict=True):
            played = by_team.get((team, season))
            earlier = played[played["time"] < cutoff].tail(n_games) if played is not None else None
            if earlier is None or earlier.empty:
                earlier = by_team.get((team, season - 1))
            if earlier is None or earlier.empty:
                continue
            seconds = float(earlier["game_sec"].sum())
            rows = pd.concat([rows_by[(g, team)] for g in earlier["game_id"]])
            for player, share in _shares(rows, seconds).items():
                out.append(
                    {
                        "game_id": str(game_id),
                        "side": side,
                        "team": team,
                        "player_id": player,
                        "share": share,
                    }
                )
    return validated(pd.DataFrame(out, columns=list(SHARES_SCHEMA.columns)), SHARES_SCHEMA)


def oracle_shares(games: pd.DataFrame, player_games: pd.DataFrame) -> pd.DataFrame:
    """Each game's actual shares (oracle, not a forecast)."""
    sides = pd.concat(
        [
            games[["game_id", side]].rename(columns={side: "team"}).assign(side=side)
            for side in SIDES
        ]
    )
    rows = player_games[player_games["sec"] > 0].merge(sides, on=["game_id", "team"])
    rows = rows.assign(share=rows["sec"] / rows["game_sec"])
    frame = rows[list(SHARES_SCHEMA.columns)].sort_values(["game_id", "side", "player_id"])
    return validated(frame.reset_index(drop=True), SHARES_SCHEMA)


def expected_possessions(games: pd.DataFrame, team_games: pd.DataFrame) -> pd.Series:
    """Expected possessions per 40 minutes of every game in ``games`` (NaN with no history).

    ``team_games``: the ``team_games`` mart rows (``game_id, team, poss_game, minutes``).
    """
    rated = _rated(games)[["game_id", "season", "tipoff_utc"]]
    rows = team_games[["game_id", "team", "poss_game", "minutes"]].merge(rated, on="game_id")
    rows = rows.assign(
        pace=rows["poss_game"] * MINUTES_PER_GAME / rows["minutes"], time=_epoch(rows["tipoff_utc"])
    ).sort_values(["time", "game_id", "team"])
    times = rows["time"].to_numpy()
    paces = rows["pace"].to_numpy()
    by_team = {
        key: (frame["time"].to_numpy(), frame["pace"].to_numpy())
        for key, frame in rows.groupby(["team", "season"])
    }

    def team_pace(team: str, season: int, before: float) -> float:
        now = by_team.get((team, season))
        if now is not None and (now[0] < before).any():
            return float(now[1][now[0] < before].mean())
        last = by_team.get((team, season - 1))
        if last is not None:
            return float(last[1].mean())
        earlier = paces[times < before]
        return float(earlier.mean()) if len(earlier) else math.nan

    values = [
        (team_pace(str(home), season, cutoff) + team_pace(str(away), season, cutoff)) / 2.0
        for home, away, season, cutoff in zip(
            games["home"],
            games["away"],
            games["season"].astype(int),
            round_cutoffs(games),
            strict=True,
        )
    ]
    return pd.Series(values, index=games["game_id"].to_numpy(), name="possessions", dtype="float64")
