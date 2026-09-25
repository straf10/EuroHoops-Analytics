"""Team box lines of both competitions -> ``team_games``: one row per team per played game.

Sources, read from the raw cache only (nothing is fetched here):

- EuroLeague: the box-score JSON's ``totr`` row (player lines plus the ``tmr`` team row of team
  rebounds and team turnovers).
- GBL: the ESAKE box score's totals row (``ΣΥΝΟΛΟ``, which includes the team row). Games whose
  ESAKE totals do not reproduce the result (the 2018-19/2019-20 gaps) are counted from the
  BasketHotel play-by-play instead (``source = "gbl_pbp"``) when its points reproduce it.

Possessions (PLAN R1) are ``FGA - OREB + TOV + 0.42 * FTA`` per team (``poss_raw``); the game's
count (``poss_game``) is the mean of both teams'. Every played, non-forfeit game either has two
rows or a row in the missing table with the reason; none is dropped silently.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pandera.pandas as pa

from eurohoops.ingest.cache import read_cached
from eurohoops.parse.esake import parse_box_score, parse_overtimes
from eurohoops.parse.gbl_pbp import team_counts
from eurohoops.parse.possessions import box_possessions
from eurohoops.parse.schemas import schema_dtypes
from eurohoops.parse.stints import overtimes

COMPETITIONS = ("euroleague", "gbl")
SOURCES = ("euroleague_box", "esake_box", "gbl_pbp")
COUNTS = ("points", "fga", "fta", "oreb", "dreb", "tov")
REGULATION_MINUTES = 40.0
OVERTIME_MINUTES = 5.0


def _two_rows_per_game(df: pd.DataFrame) -> pd.Series:
    return df.groupby("game_id")["team"].transform("size") == 2


def _game_possessions_are_the_mean(df: pd.DataFrame) -> pd.Series:
    mean = df.groupby("game_id")["poss_raw"].transform("mean")
    return (df["poss_game"] - mean).abs() < 1e-9


TEAM_GAMES_SCHEMA = pa.DataFrameSchema(
    {
        "competition": pa.Column(str, pa.Check.isin(COMPETITIONS)),
        "season": pa.Column("int64"),
        "game_id": pa.Column(str),
        "team": pa.Column(str),
        "opponent": pa.Column(str),
        "home": pa.Column(bool),
        **{column: pa.Column("int64", pa.Check.ge(0)) for column in COUNTS},
        "minutes": pa.Column("float64", pa.Check.ge(REGULATION_MINUTES)),
        "poss_raw": pa.Column("float64", pa.Check.gt(0)),
        "poss_game": pa.Column("float64", pa.Check.gt(0)),
        "source": pa.Column(str, pa.Check.isin(SOURCES)),
    },
    checks=[
        pa.Check(lambda df: df["team"] != df["opponent"], error="team equals opponent"),
        pa.Check(_two_rows_per_game, error="a game needs exactly two team rows"),
        pa.Check(_game_possessions_are_the_mean, error="poss_game must be the mean of poss_raw"),
    ],
    unique=["game_id", "team"],
    strict=True,
)

MISSING_SCHEMA = pa.DataFrameSchema(
    {
        "competition": pa.Column(str, pa.Check.isin(COMPETITIONS)),
        "season": pa.Column("int64"),
        "game_id": pa.Column(str, unique=True),
        "reason": pa.Column(str),
    },
    strict=True,
)


@dataclass(frozen=True)
class TeamGames:
    table: pd.DataFrame
    missing: pd.DataFrame


@dataclass(frozen=True)
class _Game:
    competition: str
    season: int
    game_id: str
    code: int
    teams: tuple[str, str]  # home, away
    scores: tuple[int, int]
    neutral: bool


Line = dict[str, int]


def euroleague_lines(box: dict[str, Any]) -> tuple[dict[str, Line], float] | str:
    """Team code -> counts from ``totr``, and the game's minutes; or why there are none."""
    if len(box.get("Stats") or []) != 2 or not all(s["PlayersStats"] for s in box["Stats"]):
        return "box score has no players (API placeholder)"
    lines = {}
    for side in box["Stats"]:
        totals = side["totr"]
        lines[side["PlayersStats"][0]["Team"].strip()] = {
            "points": int(totals["Points"]),
            "fga": int(totals["FieldGoalsAttempted2"]) + int(totals["FieldGoalsAttempted3"]),
            "fta": int(totals["FreeThrowsAttempted"]),
            "oreb": int(totals["OffensiveRebounds"]),
            "dreb": int(totals["DefensiveRebounds"]),
            "tov": int(totals["Turnovers"]),
        }
    minutes = REGULATION_MINUTES + OVERTIME_MINUTES * overtimes(box["ByQuarter"])
    return lines, minutes


def _euroleague(root: Path, game: _Game) -> tuple[list[Line], float, str] | str:
    path = root / "boxscore" / f"E{game.season}" / f"{game.code}.json.gz"
    if not path.exists():
        return "no cached box score (ingest --details)"
    parsed = euroleague_lines(json.loads(read_cached(path)))
    if isinstance(parsed, str):
        return parsed
    lines, minutes = parsed
    if set(lines) != set(game.teams):
        return f"box score teams {sorted(lines)} are not the game's {list(game.teams)}"
    return [lines[team] for team in game.teams], minutes, "euroleague_box"


def _esake(root: Path, game: _Game) -> tuple[list[Line], float] | str:
    path = root / "boxscore" / str(game.season) / f"{game.code:08X}.html.gz"
    if not path.exists():
        return "no cached ESAKE box score"
    html = read_cached(path).decode()
    boxes = parse_box_score(html)
    if boxes is None:
        return "ESAKE page has no box score"
    lines = [
        {
            "points": t.points,
            "fga": t.fg2a + t.fg3a,
            "fta": t.fta,
            "oreb": t.oreb,
            "dreb": t.dreb,
            "tov": t.tov,
        }
        for t in (box.totals for box in boxes)
    ]
    return lines, REGULATION_MINUTES + OVERTIME_MINUTES * parse_overtimes(html)


def _pbp(counts: dict[str, pd.DataFrame], game: _Game) -> list[Line] | str:
    rows = counts.get(game.game_id)
    if rows is None:
        return "no GBL play-by-play"
    by_side = {str(r["side"]): r for r in rows.to_dict("records")}
    if set(by_side) != {"home", "away"}:
        return "play-by-play without both teams"
    return [{c: int(by_side[side][c]) for c in COUNTS} for side in ("home", "away")]


def _gbl(
    root: Path, counts: dict[str, pd.DataFrame], periods: dict[str, int], game: _Game
) -> tuple[list[Line], float, str] | str:
    esake = _esake(root, game)
    if not isinstance(esake, str):
        lines, minutes = esake
        points = tuple(line["points"] for line in lines)
        if points == game.scores:
            return lines, minutes, "esake_box"
        esake = f"ESAKE totals {points[0]}-{points[1]} are not the result"
    pbp = _pbp(counts, game)
    if isinstance(pbp, str):
        return f"{esake}; {pbp}"
    points = tuple(line["points"] for line in pbp)
    if points != game.scores:
        return f"{esake}; play-by-play points {points[0]}-{points[1]} are not the result"
    extra = max(periods[game.game_id] - 4, 0)
    return pbp, REGULATION_MINUTES + OVERTIME_MINUTES * extra, "gbl_pbp"


def _games(games: pd.DataFrame, competition: str) -> list[_Game]:
    rated = games[games["played"] & ~games["forfeit"]]
    return [
        _Game(
            competition,
            int(season),
            str(game_id),
            int(code),
            (str(home), str(away)),
            (int(home_score), int(away_score)),
            bool(neutral),
        )
        for game_id, season, code, home, away, home_score, away_score, neutral in zip(
            rated["game_id"],
            rated["season"],
            rated["game_code"],
            rated["home"],
            rated["away"],
            rated["home_score"],
            rated["away_score"],
            rated["neutral"],
            strict=True,
        )
    ]


def _rows(game: _Game, lines: list[Line], minutes: float, source: str) -> list[dict[str, Any]]:
    raw = [box_possessions(line["fga"], line["oreb"], line["tov"], line["fta"]) for line in lines]
    game_poss = (raw[0] + raw[1]) / 2.0
    return [
        {
            "competition": game.competition,
            "season": game.season,
            "game_id": game.game_id,
            "team": game.teams[side],
            "opponent": game.teams[1 - side],
            "home": side == 0 and not game.neutral,
            **lines[side],
            "minutes": minutes,
            "poss_raw": raw[side],
            "poss_game": game_poss,
            "source": source,
        }
        for side in (0, 1)
    ]


def build_team_games(
    euroleague: pd.DataFrame,
    gbl: pd.DataFrame,
    raw_dirs: tuple[Path, Path],
    gbl_pbp: pd.DataFrame | None,
) -> TeamGames:
    """``team_games`` and the missing-game list of both competitions (games: marts tables)."""
    events = gbl_pbp if gbl_pbp is not None else pd.DataFrame(columns=["game_id", "side"])
    counts = {str(k): v for k, v in team_counts(events).groupby("game_id")} if len(events) else {}
    periods = (
        {str(k): int(v) for k, v in events.groupby("game_id")["period"].max().items()}
        if len(events)
        else {}
    )
    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for game in _games(euroleague, "euroleague") + _games(gbl, "gbl"):
        found = (
            _euroleague(raw_dirs[0], game)
            if game.competition == "euroleague"
            else _gbl(raw_dirs[1], counts, periods, game)
        )
        if isinstance(found, str):
            reason = found
        else:
            lines, minutes, source = found
            points = (lines[0]["points"], lines[1]["points"])
            if points == game.scores:
                rows += _rows(game, lines, minutes, source)
                continue
            reason = f"box points {points[0]}-{points[1]} are not the result"
        missing.append(
            {
                "competition": game.competition,
                "season": game.season,
                "game_id": game.game_id,
                "reason": reason,
            }
        )
    table = pd.DataFrame(rows, columns=list(TEAM_GAMES_SCHEMA.columns))
    gaps = pd.DataFrame(missing, columns=list(MISSING_SCHEMA.columns))
    return TeamGames(
        TEAM_GAMES_SCHEMA.validate(table.astype(schema_dtypes(TEAM_GAMES_SCHEMA))),
        MISSING_SCHEMA.validate(gaps.astype(schema_dtypes(MISSING_SCHEMA))),
    )
