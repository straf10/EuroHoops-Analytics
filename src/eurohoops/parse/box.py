"""Cached GBL box-score HTML -> validated ``player_box`` and ``team_box`` staging tables."""

from pathlib import Path

import pandas as pd
import pandera.pandas as pa

from eurohoops.ingest.cache import read_cached
from eurohoops.parse.esake import BoxLine, parse_box_score

COUNTS = ("points", "fg2m", "fg2a", "fg3m", "fg3a", "ftm", "fta", "seconds")

PLAYER_BOX_SCHEMA = pa.DataFrameSchema(
    {
        "game_id": pa.Column(str),
        "team": pa.Column(str),
        "player_id": pa.Column(str),
        **{column: pa.Column("int64", pa.Check.ge(0)) for column in COUNTS},
    },
    unique=["game_id", "player_id"],
    strict=True,
)

TEAM_BOX_SCHEMA = pa.DataFrameSchema(
    {
        "game_id": pa.Column(str),
        "team": pa.Column(str),
        "total_points": pa.Column("int64", pa.Check.ge(0)),
    },
    unique=["game_id", "team"],
    strict=True,
)


def _player_row(game_id: str, team: str, line: BoxLine) -> dict[str, object]:
    return {
        "game_id": game_id,
        "team": team,
        "player_id": line.player_id,
        "points": line.points,
        "fg2m": line.fg2m,
        "fg2a": line.fg2a,
        "fg3m": line.fg3m,
        "fg3a": line.fg3a,
        "ftm": line.ftm,
        "fta": line.fta,
        "seconds": line.seconds,
    }


def build_box_tables(root: Path, games: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse every cached box score of a played, non-forfeit game in ``games``."""
    players: list[dict[str, object]] = []
    teams: list[dict[str, object]] = []
    rated = games[games["played"] & ~games["forfeit"]]
    for game_id, season, idgame, home, away in zip(
        rated["game_id"],
        rated["season"],
        rated["game_code"],
        rated["home"],
        rated["away"],
        strict=True,
    ):
        path = root / "boxscore" / str(season) / f"{idgame:08X}.html.gz"
        if not path.exists():
            continue
        boxes = parse_box_score(read_cached(path).decode())
        for team, box in zip((home, away), boxes, strict=True):
            teams.append({"game_id": game_id, "team": team, "total_points": box.total_points})
            players.extend(_player_row(game_id, team, line) for line in box.players)
    player_box = pd.DataFrame(players, columns=list(PLAYER_BOX_SCHEMA.columns))
    team_box = pd.DataFrame(teams, columns=list(TEAM_BOX_SCHEMA.columns))
    return (
        PLAYER_BOX_SCHEMA.validate(player_box.astype(dict.fromkeys(COUNTS, "int64"))),
        TEAM_BOX_SCHEMA.validate(team_box.astype({"total_points": "int64"})),
    )
