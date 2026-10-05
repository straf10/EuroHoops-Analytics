"""Rest features for game forecasts (M5): how long each team has rested, across both competitions.

For a team and a game with tip-off ``t`` (all times are tip-offs):

- **days_rest**: days (fractional) since the team's previous game in either competition, capped
  at 7; 7.0 when the team has no previous game at all.
- **short_rest**: the previous game was at most 2.5 days earlier (D1: no game in the data is
  within 1.5 days, so a back-to-back flag would never fire).
- **games_last_7d**: the team's games with tip-off in ``[t - 7 days, t)``.
- **other_comp_prev**: the team's previous game was in the other competition (when two previous
  games share the latest tip-off, the other competition's counts).

A previous game is a row with ``played`` True and a tip-off strictly before ``t``; forfeits count
when ``played`` is True, unplayed or postponed rows never do, and a game at the same tip-off is
not previous. Scores are never read, so nothing after ``t`` can reach a feature. The other
competition's team codes are mapped to ``games``' codes through ``club_map``; teams outside the
map have no games there. Every game in ``games`` gets both sides, played or not.
"""

from collections.abc import Mapping

import numpy as np
import pandas as pd
import pandera.pandas as pa

from eurohoops.models.elo import FloatArray
from eurohoops.parse.schemas import validated

SIDES = ("home", "away")
DAY = 86400.0
CAP_DAYS = 7.0
SHORT_REST_DAYS = 2.5

REST_SCHEMA = pa.DataFrameSchema(
    {
        "game_id": pa.Column(str),
        "side": pa.Column(str, pa.Check.isin(SIDES)),
        "team": pa.Column(str),
        "days_rest": pa.Column("float64", [pa.Check.gt(0.0), pa.Check.le(CAP_DAYS)]),
        "short_rest": pa.Column(bool),
        "games_last_7d": pa.Column("int64", pa.Check.ge(0)),
        "other_comp_prev": pa.Column(bool),
    },
    unique=["game_id", "side"],
    strict=True,
)


def _epoch(times: pd.Series) -> FloatArray:
    """Epoch seconds, whatever the timestamps' resolution (s from literals, ns from DuckDB)."""
    epoch = pd.Timestamp(0, tz="UTC")
    seconds: FloatArray = (
        (times.dt.tz_convert("UTC") - epoch).dt.total_seconds().to_numpy(dtype=np.float64)
    )
    return seconds


def _team_rows(games: pd.DataFrame, mapping: Mapping[str, str], is_other: bool) -> pd.DataFrame:
    """One row per (played game, team in ``mapping``): the team's code in ``games``, the tip-off
    in epoch seconds and whether the game belongs to the other competition."""
    played = games[games["played"].to_numpy(dtype=bool)]
    time = _epoch(played["tipoff_utc"])
    parts = [
        pd.DataFrame({"team": played[side].astype(str).map(mapping), "time": time})
        for side in SIDES
    ]
    rows = pd.concat(parts, ignore_index=True).dropna(subset=["team"])
    rows["other"] = is_other
    return rows


def rest_features(
    games: pd.DataFrame, other: pd.DataFrame | None, club_map: Mapping[str, str]
) -> pd.DataFrame:
    """Rest features of both sides of every game in ``games`` (see the module docstring).

    ``other``: the other competition's games in the same format (or ``None``); ``club_map`` maps
    its team codes to the same clubs' codes in ``games``.
    """
    identity = {str(team): str(team) for team in pd.concat([games["home"], games["away"]])}
    frames = [_team_rows(games, identity, is_other=False)]
    if other is not None:
        frames.append(_team_rows(other, club_map, is_other=True))
    history = pd.concat(frames, ignore_index=True).sort_values(["time", "other"], kind="stable")
    by_team = {
        str(team): (frame["time"].to_numpy(dtype=np.float64), frame["other"].to_numpy(dtype=bool))
        for team, frame in history.groupby("team")
    }
    empty = (np.empty(0, dtype=np.float64), np.empty(0, dtype=bool))
    out: list[dict[str, object]] = []
    for game_id, home, away, time in zip(
        games["game_id"], games["home"], games["away"], _epoch(games["tipoff_utc"]), strict=True
    ):
        for side, team in zip(SIDES, (str(home), str(away)), strict=True):
            times, from_other = by_team.get(team, empty)
            before = int(np.searchsorted(times, time, side="left"))
            since = np.inf if before == 0 else (time - times[before - 1]) / DAY
            out.append(
                {
                    "game_id": str(game_id),
                    "side": side,
                    "team": team,
                    "days_rest": min(CAP_DAYS, float(since)),
                    "short_rest": bool(since <= SHORT_REST_DAYS),
                    "games_last_7d": before
                    - int(np.searchsorted(times, time - CAP_DAYS * DAY, side="left")),
                    "other_comp_prev": bool(before > 0 and from_other[before - 1]),
                }
            )
    return validated(pd.DataFrame(out, columns=list(REST_SCHEMA.columns)), REST_SCHEMA)
