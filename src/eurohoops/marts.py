"""DuckDB marts: staging Parquet -> ``data/marts/eurohoops.duckdb``, plus box-score invariants.

``backtest``, ``predict``, ``score`` and ``publish`` read games and teams from here, so every
command sees the same validated union of both competitions.
"""

from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from eurohoops.config import EUROLEAGUE, GBL, GBL_BOX_FILL, GBL_PLAYER_BOX, GBL_TEAM_BOX
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
        has_box = all(p.exists() for p in (GBL_PLAYER_BOX, GBL_TEAM_BOX, GBL_BOX_FILL))
        if has_box:
            con.execute(
                _sql(
                    sql_dir / "box.sql",
                    gbl_player_box=GBL_PLAYER_BOX,
                    gbl_team_box=GBL_TEAM_BOX,
                    gbl_box_fill=GBL_BOX_FILL,
                )
            )
    return has_box


def write_tables(mart_path: Path, tables: dict[str, pd.DataFrame]) -> None:
    """Create or replace mart tables from frames (names come from code, never from users)."""
    with duckdb.connect(str(mart_path)) as con:
        for name, frame in tables.items():
            con.register("frame", frame)
            con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM frame")
            con.unregister("frame")


def read_table(mart_path: Path, name: str, competition: str | None = None) -> pd.DataFrame | None:
    """A mart table (optionally one competition's rows) in its stored row order.

    None when the table does not exist (e.g. the stints mart was never built).
    """
    with duckdb.connect(str(mart_path), read_only=True) as con:
        tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
        if name not in tables:
            return None
        con.execute("SET TimeZone = 'UTC'")
        if competition is None:
            return con.execute(f"SELECT * FROM {name}").df()
        return con.execute(f"SELECT * FROM {name} WHERE competition = ?", [competition]).df()


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


def _fill_summary(fill: pd.DataFrame) -> dict[str, Any]:
    """Per season: teams of short games, how many were filled from PBP and whether they add up."""
    filled = fill[fill["fill"] != "not_filled"]
    return {
        str(season): {
            "teams": len(rows),
            "by_fill": {k: int(v) for k, v in rows["fill"].value_counts().sort_index().items()},
            "filled_points_match_result": int(
                (
                    filled.loc[filled["season"] == season, "filled_points"]
                    == filled.loc[filled["season"] == season, "score"]
                ).sum()
            ),
        }
        for season, rows in fill.groupby("season")
    }


def box_invariants(mart_path: Path) -> dict[str, Any]:
    """Per-season pass rates of the official box scores; failing games are listed, never dropped.

    ``pbp_fill`` reports the PBP fill separately: it never changes an official pass rate.
    """
    with duckdb.connect(str(mart_path), read_only=True) as con:
        sides = con.execute("SELECT * FROM box_checks").df()
        fill = con.execute("SELECT * FROM box_fill_checks ORDER BY game_id, team").df()
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
    return {
        "tolerance_minutes": MINUTES_TOLERANCE,
        "seasons": seasons,
        "pbp_fill": _fill_summary(fill),
    }


def refresh_box_gaps(mart_path: Path, gaps_csv: Path) -> None:
    """Add each listed game's PBP fill outcome to the gaps report (other columns untouched).

    The report is a curated list; without one there is nothing to refresh.
    """
    if not gaps_csv.exists():
        return
    gaps = pd.read_csv(gaps_csv, dtype=str, keep_default_na=False)
    with duckdb.connect(str(mart_path), read_only=True) as con:
        fill = con.execute("SELECT * FROM box_fill_checks ORDER BY game_id, team").df()
    by_game = {
        game_id: rows.to_dict("records") for game_id, rows in fill.groupby("game_id", sort=True)
    }
    outcome, detail, matches = [], [], []
    for game_id in gaps["game_id"]:
        rows = by_game.get(game_id, [])
        fills = sorted({str(r["fill"]) for r in rows})
        outcome.append("+".join(fills) if rows else "not_filled")
        detail.append(
            "; ".join(f"{r['team']}: {r['detail']}" for r in rows)
            if rows
            else "not short against the result in the current box tables"
        )
        done = [r for r in rows if r["fill"] != "not_filled"]
        matches.append(str(all(r["filled_points"] == r["score"] for r in done)) if done else "")
    gaps = gaps.assign(
        pbp_fill=outcome, pbp_fill_detail=detail, pbp_filled_points_match_result=matches
    )
    gaps.to_csv(gaps_csv, index=False, lineterminator="\n")
