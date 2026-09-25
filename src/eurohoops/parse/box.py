"""Cached GBL box-score HTML -> validated ``player_box`` and ``team_box`` staging tables.

Official lines come from ESAKE (``source = "esake"``). For the 2018-19 and 2019-20 gaps ESAKE
cannot fill (decided 2026-09-25), lines rebuilt from the BasketHotel play-by-play are added
with ``source = "pbp"``; they never count toward an official-box claim:

- a game without an ESAKE box gets PBP team totals (``team_box`` rows);
- a team whose ESAKE player lines add up to less than its result gets the one missing
  player's PBP line, but only when exactly one PBP line with stats has no exact match among
  the ESAKE lines and adding it makes the result.

Every short game gets a row in the fill log: what was filled, or why not.
"""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pandera.pandas as pa

from eurohoops.ingest.cache import read_cached
from eurohoops.parse.esake import BoxLine, parse_box_score
from eurohoops.parse.gbl_pbp import PbpScoreError, parse_export, player_lines
from eurohoops.parse.schemas import schema_dtypes

COUNTS = ("points", "fg2m", "fg2a", "fg3m", "fg3a", "ftm", "fta", "seconds")
SIGNATURE = COUNTS[:-1]  # every counting stat except minutes, which ESAKE and PBP round alike
SOURCES = ("esake", "pbp")
FILL_SEASONS = (2018, 2019)
FILLS = ("team_totals_from_pbp", "missing_player_from_pbp", "not_filled")

PLAYER_BOX_SCHEMA = pa.DataFrameSchema(
    {
        "game_id": pa.Column(str),
        "team": pa.Column(str),
        "player_id": pa.Column(str),
        **{column: pa.Column("int64", pa.Check.ge(0)) for column in COUNTS},
        "source": pa.Column(str, pa.Check.isin(SOURCES)),
    },
    unique=["game_id", "player_id"],
    strict=True,
)

TEAM_BOX_SCHEMA = pa.DataFrameSchema(
    {
        "game_id": pa.Column(str),
        "team": pa.Column(str),
        "total_points": pa.Column("int64", pa.Check.ge(0)),
        "source": pa.Column(str, pa.Check.isin(SOURCES)),
    },
    unique=["game_id", "team", "source"],
    strict=True,
)

FILL_SCHEMA = pa.DataFrameSchema(
    {
        "game_id": pa.Column(str),
        "season": pa.Column("int64"),
        "team": pa.Column(str),
        "fill": pa.Column(str, pa.Check.isin(FILLS)),
        "detail": pa.Column(str),
    },
    unique=["game_id", "team"],
    strict=True,
)


@dataclass(frozen=True)
class BoxTables:
    player_box: pd.DataFrame
    team_box: pd.DataFrame
    fill: pd.DataFrame  # one row per team of a short game


def _player_row(game_id: str, team: str, line: BoxLine) -> dict[str, object]:
    return {
        "game_id": game_id,
        "team": team,
        "player_id": line.player_id,
        **{column: getattr(line, column) for column in COUNTS},
        "source": "esake",
    }


@dataclass(frozen=True)
class _Game:
    game_id: str
    season: int
    idgame: str
    teams: tuple[str, str]  # home, away
    scores: tuple[int, int]


def _pbp(root: Path, game: _Game) -> pd.DataFrame | str:
    """PBP player lines of the game, or why there are none."""
    path = root / "pbp" / str(game.season) / f"{game.idgame}.xlsx.gz"
    if not path.exists():
        return "no cached PBP export (run: eurohoops ingest --competition gbl --pbp)"
    try:
        events = parse_export(read_cached(path), game.game_id, *game.scores)
    except PbpScoreError as exc:
        return f"PBP not usable: {exc}"
    return player_lines(events)


def _missing_player(
    official: tuple[BoxLine, ...], pbp: pd.DataFrame, score: int
) -> dict[str, Any] | str:
    """The single PBP line absent from the ESAKE lines that closes the gap, or why not."""
    known = Counter(tuple(getattr(line, c) for c in SIGNATURE) for line in official)
    unmatched: list[dict[str, Any]] = []
    for record in pbp.to_dict("records"):
        line = {str(key): value for key, value in record.items()}
        signature = tuple(line[c] for c in SIGNATURE)
        if known[signature]:
            known[signature] -= 1
        elif any(signature):
            unmatched.append(line)
    gap = score - sum(line.points for line in official)
    if len(unmatched) != 1 or unmatched[0]["points"] != gap:
        found = ", ".join(f"#{u['number']} {u['player']} ({u['points']} pts)" for u in unmatched)
        return f"{len(unmatched)} unmatched PBP lines [{found}] for a {gap}-point gap"
    return unmatched[0]


def _fill(
    root: Path, game: _Game, boxes: tuple[tuple[BoxLine, ...], ...] | None
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """PBP player rows, PBP team rows and fill-log rows for one game's short teams."""
    short = [
        side
        for side in (0, 1)
        if boxes is None or sum(line.points for line in boxes[side]) != game.scores[side]
    ]
    if not short:
        return [], [], []
    log = [
        {"game_id": game.game_id, "season": game.season, "team": game.teams[side]} for side in short
    ]
    if game.season not in FILL_SEASONS:
        reason = "outside the 2018-19/2019-20 fill scope (user decision 2026-09-25)"
        return [], [], [{**row, "fill": "not_filled", "detail": reason} for row in log]
    pbp = _pbp(root, game)
    if isinstance(pbp, str):
        return [], [], [{**row, "fill": "not_filled", "detail": pbp} for row in log]
    players: list[dict[str, object]] = []
    teams: list[dict[str, object]] = []
    for row, side in zip(log, short, strict=True):
        lines = pbp[pbp["side"] == ("home", "away")[side]]
        points = int(lines["points"].sum())
        team, score = game.teams[side], game.scores[side]
        if points != score:
            row |= {"fill": "not_filled", "detail": f"PBP events sum to {points}, result {score}"}
        elif boxes is None:
            teams.append(
                {"game_id": game.game_id, "team": team, "total_points": points, "source": "pbp"}
            )
            row |= {"fill": "team_totals_from_pbp", "detail": f"{points} points from PBP events"}
        else:
            found = _missing_player(boxes[side], lines, score)
            if isinstance(found, str):
                row |= {"fill": "not_filled", "detail": found}
            else:
                player_id = f"pbp:{found['number']}:{found['player']}"
                players.append(
                    {
                        "game_id": game.game_id,
                        "team": team,
                        "player_id": player_id,
                        **{column: int(found[column]) for column in COUNTS},
                        "source": "pbp",
                    }
                )
                row |= {
                    "fill": "missing_player_from_pbp",
                    "detail": f"#{found['number']} {found['player']}: {found['points']} points",
                }
    return players, teams, log


def build_box_tables(root: Path, games: pd.DataFrame) -> BoxTables:
    """Parse every cached box score of a played, non-forfeit game in ``games``, then fill.

    Games without a cached or published box score get no ESAKE rows; the invariant report
    flags them as ``missing_box`` whatever the PBP fill adds.
    """
    players: list[dict[str, object]] = []
    teams: list[dict[str, object]] = []
    fills: list[dict[str, object]] = []
    rated = games[games["played"] & ~games["forfeit"]]
    for game_id, season, code, home, away, home_score, away_score in zip(
        rated["game_id"],
        rated["season"],
        rated["game_code"],
        rated["home"],
        rated["away"],
        rated["home_score"],
        rated["away_score"],
        strict=True,
    ):
        game = _Game(
            game_id, int(season), f"{code:08X}", (home, away), (int(home_score), int(away_score))
        )
        path = root / "boxscore" / str(season) / f"{game.idgame}.html.gz"
        parsed = parse_box_score(read_cached(path).decode()) if path.exists() else None
        boxes = None if parsed is None else (parsed[0].players, parsed[1].players)
        for team, box in zip(game.teams, parsed or (), strict=False):
            teams.append(
                {
                    "game_id": game_id,
                    "team": team,
                    "total_points": box.total_points,
                    "source": "esake",
                }
            )
            players.extend(_player_row(game_id, team, line) for line in box.players)
        pbp_players, pbp_teams, log = _fill(root, game, boxes)
        players += pbp_players
        teams += pbp_teams
        fills += log
    player_box = pd.DataFrame(players, columns=list(PLAYER_BOX_SCHEMA.columns))
    team_box = pd.DataFrame(teams, columns=list(TEAM_BOX_SCHEMA.columns))
    fill = pd.DataFrame(fills, columns=list(FILL_SCHEMA.columns))
    return BoxTables(
        _validated(player_box, PLAYER_BOX_SCHEMA),
        _validated(team_box, TEAM_BOX_SCHEMA),
        _validated(fill, FILL_SCHEMA),
    )


def _validated(frame: pd.DataFrame, schema: pa.DataFrameSchema) -> pd.DataFrame:
    """Cast to the schema's dtypes first: an empty frame's columns are ``object`` otherwise."""
    return schema.validate(frame.astype(schema_dtypes(schema)))
