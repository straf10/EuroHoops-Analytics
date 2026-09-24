"""DuckDB marts: staging Parquet -> ``data/marts/eurohoops.duckdb``, plus box-score invariants.

``backtest``, ``predict``, ``score`` and ``publish`` read games and teams from here, so every
command sees the same validated union of both competitions.
"""

from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from eurohoops.config import EUROLEAGUE, GBL, GBL_PLAYER_BOX, GBL_TEAM_BOX
from eurohoops.parse.games import TEAMS_SCHEMA, conform

MINUTES_PER_GAME = 200.0
MINUTES_PER_OVERTIME = 25.0
MINUTES_TOLERANCE = 1.0


def _sql(path: Path, **files: Path) -> str:
    """Fill ``{name}`` placeholders with quoted paths (all paths come from config, not users)."""
    return path.read_text(encoding="utf-8").format(
        **{name: "'" + file.as_posix() + "'" for name, file in files.items()}
    )


def build_marts(mart_path: Path, sql_dir: Path) -> bool:
    """(Re)build the marts; return whether GBL box-score tables were available and built."""
    mart_path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(mart_path)) as con:
        con.execute(
            _sql(
                sql_dir / "games.sql",
                euroleague_games=EUROLEAGUE.staging_games,
                gbl_games=GBL.staging_games,
                euroleague_teams=EUROLEAGUE.staging_teams,
                gbl_teams=GBL.staging_teams,
            )
        )
        has_box = GBL_PLAYER_BOX.exists() and GBL_TEAM_BOX.exists()
        if has_box:
            con.execute(
                _sql(sql_dir / "box.sql", gbl_player_box=GBL_PLAYER_BOX, gbl_team_box=GBL_TEAM_BOX)
            )
    return has_box


def _query(mart_path: Path, sql: str, competition: str) -> pd.DataFrame:
    with duckdb.connect(str(mart_path), read_only=True) as con:
        con.execute("SET TimeZone = 'UTC'")
        return con.execute(sql, [competition]).df()


def read_games(mart_path: Path, competition: str) -> pd.DataFrame:
    games = _query(mart_path, "SELECT * FROM games WHERE competition = ?", competition)
    return conform(games.drop(columns="competition"))


def read_teams(mart_path: Path, competition: str) -> pd.DataFrame:
    teams = _query(mart_path, "SELECT team, name FROM teams WHERE competition = ?", competition)
    return TEAMS_SCHEMA.validate(teams.astype(str))


def _minutes_ok(minutes: pd.Series) -> pd.Series:
    """Team minutes must be 200 plus a whole number of 25-minute overtimes (within 1 min)."""
    overtimes = ((minutes - MINUTES_PER_GAME) / MINUTES_PER_OVERTIME).round().clip(lower=0)
    expected = MINUTES_PER_GAME + MINUTES_PER_OVERTIME * overtimes
    return (minutes - expected).abs() <= MINUTES_TOLERANCE


def box_invariants(mart_path: Path) -> dict[str, Any]:
    """Per-season pass rates; failing games are listed with reasons, never dropped."""
    with duckdb.connect(str(mart_path), read_only=True) as con:
        sides = con.execute("SELECT * FROM box_checks").df()
    checks = pd.DataFrame(
        {
            "season": sides["season"],
            "game_id": sides["game_id"],
            "missing_box": sides["total_points"].isna(),
            "points_mismatch": (sides["player_points"] != sides["total_points"])
            | (sides["total_points"] != sides["score"]),
            "minutes_off": ~_minutes_ok(sides["minutes"].astype(float)),
            "bad_shot_lines": sides["bad_shot_lines"].fillna(0) > 0,
            "bad_point_lines": sides["bad_point_lines"].fillna(0) > 0,
        }
    )
    reasons = [c for c in checks.columns if c not in {"season", "game_id"}]
    per_game = checks.groupby(["season", "game_id"])[reasons].any()
    per_game["passed"] = ~per_game[reasons].any(axis=1)
    seasons: dict[str, Any] = {}
    for season, games in per_game.groupby(level="season"):
        failed = games[~games["passed"]]
        seasons[str(season)] = {
            "games": len(games),
            "passed": int(games["passed"].sum()),
            "pass_rate": round(float(games["passed"].mean()), 4),
            "failures_by_reason": {r: int(games[r].sum()) for r in reasons},
            "failed_games": {
                str(row["game_id"]): [r for r in reasons if row[r]]
                for row in failed.reset_index().to_dict("records")
            },
        }
    return {"tolerance_minutes": MINUTES_TOLERANCE, "seasons": seasons}
