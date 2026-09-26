"""Cached EuroLeague box scores -> ``player_games`` and ``team_stat_games``.

One row per player who played (``Minutes`` is ``DNP`` otherwise) and one per team per game,
the team row read from the box score's ``totr`` totals. Possessions follow ``team_games``
(PLAN R1): each team's box estimate, the game's count the mean of both. A player is on court for
his share of his team's recorded seconds times five game counts: shares of the recorded total, not
of the clock, because a few old box scores record more minutes than were played (2010-11 has a
team with 240 minutes in a regulation game). Games without a usable box score go to ``missing``
with the reason.
"""

import json
from collections.abc import Hashable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pandera.pandas as pa

from eurohoops.ingest.cache import read_cached
from eurohoops.parse.possessions import box_possessions
from eurohoops.parse.schemas import schema_dtypes
from eurohoops.parse.team_box import euroleague_lines

# Output name -> box-score key. Minutes are parsed separately into seconds.
BOX_KEYS = {
    "pts": "Points",
    "fg2m": "FieldGoalsMade2",
    "fg2a": "FieldGoalsAttempted2",
    "fg3m": "FieldGoalsMade3",
    "fg3a": "FieldGoalsAttempted3",
    "ftm": "FreeThrowsMade",
    "fta": "FreeThrowsAttempted",
    "oreb": "OffensiveRebounds",
    "dreb": "DefensiveRebounds",
    "ast": "Assistances",
    "stl": "Steals",
    "tov": "Turnovers",
    "blk": "BlocksFavour",
    "blka": "BlocksAgainst",
    "pf": "FoulsCommited",
    "fd": "FoulsReceived",
    "pir": "Valuation",
}
STATS = tuple(BOX_KEYS)
NON_NEGATIVE = tuple(s for s in STATS if s != "pir")  # PIR (valuation) can be negative
VENUES = ("home", "away", "neutral")


def _stat_columns() -> dict[str, pa.Column]:
    return {
        s: pa.Column("int64", pa.Check.ge(0)) if s in NON_NEGATIVE else pa.Column("int64")
        for s in STATS
    }


_GAME_COLUMNS = {
    "season": pa.Column("int64"),
    "game_id": pa.Column(str),
    "team": pa.Column(str),
    "opponent": pa.Column(str),
    "venue": pa.Column(str, pa.Check.isin(VENUES)),
    "won": pa.Column(bool),
}

PLAYER_GAMES_SCHEMA = pa.DataFrameSchema(
    {
        **_GAME_COLUMNS,
        "player_id": pa.Column(str),
        "player": pa.Column(str),
        "dorsal": pa.Column(str),
        "starter": pa.Column(bool),
        "sec": pa.Column("int64", pa.Check.ge(0)),
        **_stat_columns(),
        "pm": pa.Column("int64"),
        "poss": pa.Column("float64", pa.Check.ge(0)),
        "game_sec": pa.Column("int64", pa.Check.gt(0)),
    },
    unique=["game_id", "player_id"],
    strict=True,
)

TEAM_STAT_GAMES_SCHEMA = pa.DataFrameSchema(
    {
        **_GAME_COLUMNS,
        **_stat_columns(),
        "poss": pa.Column("float64", pa.Check.gt(0)),
        "game_sec": pa.Column("int64", pa.Check.gt(0)),
    },
    unique=["game_id", "team"],
    strict=True,
)

MISSING_SCHEMA = pa.DataFrameSchema(
    {
        "season": pa.Column("int64"),
        "game_id": pa.Column(str, unique=True),
        "reason": pa.Column(str),
    },
    strict=True,
)


@dataclass(frozen=True)
class BoxGames:
    players: pd.DataFrame
    teams: pd.DataFrame
    missing: pd.DataFrame


def seconds(minutes: str) -> int | None:
    """Box ``Minutes`` ``mm:ss`` -> seconds; ``DNP`` (did not play) -> None."""
    text = minutes.strip()
    if ":" not in text:
        return None
    mins, secs = text.split(":")
    return int(mins) * 60 + int(secs)


def _stats(line: dict[str, Any]) -> dict[str, int]:
    return {name: int(line[key] or 0) for name, key in BOX_KEYS.items()}


def game_lines(
    box: dict[str, Any], game: dict[Hashable, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | str:
    """Player rows and the two team rows of one game (``game``: a games-table row); or why not."""
    parsed = euroleague_lines(box)
    if isinstance(parsed, str):
        return parsed
    counts, minutes = parsed
    teams = (str(game["home"]), str(game["away"]))
    if set(counts) != set(teams):
        return f"box score teams {sorted(counts)} are not the game's {list(teams)}"
    game_sec = round(minutes * 60)
    raw = {t: box_possessions(c["fga"], c["oreb"], c["tov"], c["fta"]) for t, c in counts.items()}
    game_poss = (raw[teams[0]] + raw[teams[1]]) / 2.0
    home_won = int(game["home_score"]) > int(game["away_score"])
    players: list[dict[str, Any]] = []
    team_rows: list[dict[str, Any]] = []
    for side in box["Stats"]:
        team = side["PlayersStats"][0]["Team"].strip()
        is_home = team == teams[0]
        base = {
            "season": int(game["season"]),
            "game_id": str(game["game_id"]),
            "team": team,
            "opponent": teams[1] if is_home else teams[0],
            "venue": "neutral" if game["neutral"] else ("home" if is_home else "away"),
            "won": home_won == is_home,
        }
        team_rows.append({**base, **_stats(side["totr"]), "poss": raw[team], "game_sec": game_sec})
        lines = [(line, seconds(str(line["Minutes"]))) for line in side["PlayersStats"]]
        on_court = sum(played or 0 for _, played in lines) or 1
        for line, played in lines:
            if played is None:
                continue
            players.append(
                {
                    **base,
                    "player_id": str(line["Player_ID"]).strip(),
                    "player": str(line["Player"]).strip(),
                    "dorsal": str(line["Dorsal"] or "").strip(),
                    "starter": bool(line["IsStarter"]),
                    "sec": played,
                    **_stats(line),
                    "pm": int(line["Plusminus"] or 0),
                    "poss": 5 * game_poss * played / on_court,
                    "game_sec": game_sec,
                }
            )
    return players, team_rows


def build_box_games(raw_dir: Path, games: pd.DataFrame) -> BoxGames:
    """Every played, non-forfeit game in ``games`` (a games table) with a cached box score."""
    players: list[dict[str, Any]] = []
    teams: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    rated = games[games["played"] & ~games["forfeit"]]
    for game in rated.to_dict("records"):
        path = raw_dir / "boxscore" / f"E{game['season']}" / f"{game['game_code']}.json.gz"
        found = (
            game_lines(json.loads(read_cached(path)), game)
            if path.exists()
            else "no cached box score (ingest --details)"
        )
        if isinstance(found, str):
            missing.append(
                {"season": int(game["season"]), "game_id": str(game["game_id"]), "reason": found}
            )
            continue
        players += found[0]
        teams += found[1]
    return BoxGames(
        _validated(
            pd.DataFrame(players, columns=list(PLAYER_GAMES_SCHEMA.columns)), PLAYER_GAMES_SCHEMA
        ),
        _validated(
            pd.DataFrame(teams, columns=list(TEAM_STAT_GAMES_SCHEMA.columns)),
            TEAM_STAT_GAMES_SCHEMA,
        ),
        _validated(pd.DataFrame(missing, columns=list(MISSING_SCHEMA.columns)), MISSING_SCHEMA),
    )


def _validated(frame: pd.DataFrame, schema: pa.DataFrameSchema) -> pd.DataFrame:
    return schema.validate(frame.astype(schema_dtypes(schema)))
