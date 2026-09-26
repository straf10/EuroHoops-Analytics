"""EuroLeague shot table (M2): one row per field-goal attempt, from the cached ``Points`` feed.

Read from the raw cache only (``data/raw/euroleague/points``); nothing is fetched here. The feed
facts this relies on are checked over every cached game and recorded in ``docs/data/shots.md``:

- Field-goal codes: ``2FGM 2FGA 3FGM 3FGA`` every season, blocked attempts ``2FGAB``/``3FGAB``
  up to 2016-17, and ``LAYUPMD``/``LAYUPATT``/``DUNK`` for layups and dunks in 2008-09 to
  2014-15. ``FTM`` rows (made free throws, the only free throws in the feed) are not shots.
- ``POINTS_A``/``POINTS_B`` are the schedule's home/away score *after* the row, so the score
  before a shot is the after-score minus the shot's own points.
- ``MINUTE`` counts on through overtime (41-45 is the first); ``CONSOLE`` is the time left in
  the period, occasionally ``00:-1`` at the buzzer (read as 0).

Rows that cannot be used go to ``shots_excluded`` with a reason, never silently: unparseable
rows, missing coordinates (0, 0), and (2011-12 on, where the coordinates fit the FIBA line) a
2/3 label contradicting the geometry by more than ``LABEL_TOLERANCE_M``. Seasons before
``FIRST_VALIDATED_SEASON`` are kept, flagged ``validated_season = False`` and never used for
fitting or scoring (F-a).
"""

import json
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pandera.pandas as pa

from eurohoops.ingest.cache import read_cached
from eurohoops.parse.schemas import schema_dtypes
from eurohoops.parse.shots import (
    FIRST_VALIDATED_SEASON,
    FREE_THROW_SENTINEL,
    beyond_three_line,
    to_court_coords,
)
from eurohoops.parse.stints_mart import digest

COMPETITION = "euroleague"
LAST_SEASON = 2025  # M2 feeds no live use (F-i): the live season's shots stay out
LABEL_TOLERANCE_M = 0.15
# Field-goal codes -> shot value; made or missed comes from the row's POINTS.
SHOT_VALUE = {
    "2FGM": 2,
    "2FGA": 2,
    "2FGAB": 2,
    "LAYUPMD": 2,
    "LAYUPATT": 2,
    "DUNK": 2,
    "3FGM": 3,
    "3FGA": 3,
    "3FGAB": 3,
}
MADE_CODES = frozenset({"2FGM", "3FGM", "LAYUPMD", "DUNK"})
FREE_THROW = "FTM"
REASONS = ("unparseable", "zero_coordinates", "label_geometry")
QUARTER_MINUTES = 10
OVERTIME_MINUTES = 5
REGULATION_MINUTES = 40
# Reporting bands (F-f, F-e): by distance for 2s, split at 8 m for 3s.
BANDS = ("rim", "short", "mid", "long2", "three", "deep3")
RIM_M, SHORT_M, MID_M, DEEP3_M = 1.5, 3.0, 5.0, 8.0


def _one_value_per_code(df: pd.DataFrame) -> pd.Series:
    return (df["value"] == 2) | (df["value"] == 3)


SHOTS_SCHEMA = pa.DataFrameSchema(
    {
        "competition": pa.Column(str, pa.Check.eq(COMPETITION)),
        "season": pa.Column("int64"),
        "game_id": pa.Column(str),
        "event": pa.Column("int64"),
        "period": pa.Column("int64", pa.Check.ge(1)),
        "seconds_left": pa.Column("int64", pa.Check.in_range(0, QUARTER_MINUTES * 60)),
        "team": pa.Column(str),
        "opponent": pa.Column(str),
        "home": pa.Column(bool),
        "shooter": pa.Column(str),
        "made": pa.Column(bool),
        "value": pa.Column("int64", pa.Check.isin([2, 3])),
        "x": pa.Column("float64"),
        "y": pa.Column("float64"),
        "distance": pa.Column("float64", pa.Check.ge(0)),
        "angle": pa.Column("float64", pa.Check.in_range(0.0, 180.0)),
        "band": pa.Column(str, pa.Check.isin(BANDS)),
        "zone": pa.Column(str),
        "fastbreak": pa.Column(bool),
        "second_chance": pa.Column(bool),
        "points_off_turnover": pa.Column(bool),
        "margin_before": pa.Column("int64"),
        "validated_season": pa.Column(bool),
    },
    checks=[
        pa.Check(lambda df: df["team"] != df["opponent"], error="team equals opponent"),
        pa.Check(_one_value_per_code, error="value must be 2 or 3"),
    ],
    unique=["game_id", "event"],
    strict=True,
)

EXCLUDED_SCHEMA = pa.DataFrameSchema(
    {
        "competition": pa.Column(str, pa.Check.eq(COMPETITION)),
        "season": pa.Column("int64"),
        "game_id": pa.Column(str),
        "event": pa.Column("int64"),
        "team": pa.Column(str),
        "action": pa.Column(str),
        "points": pa.Column("int64"),
        "coord_x": pa.Column("float64", nullable=True),
        "coord_y": pa.Column("float64", nullable=True),
        "reason": pa.Column(str, pa.Check.isin(REASONS)),
        "detail": pa.Column(str),
    },
    unique=["game_id", "event"],
    strict=True,
)


@dataclass(frozen=True)
class Game:
    game_id: str
    season: int
    code: int
    home: str
    away: str
    neutral: bool


def period_of(minute: int) -> int:
    """``MINUTE`` 1-40 -> quarters 1-4; 41-45 -> 5 (first overtime), 46-50 -> 6, ..."""
    if minute <= REGULATION_MINUTES:
        return (max(minute, 1) - 1) // QUARTER_MINUTES + 1
    return 5 + (minute - REGULATION_MINUTES - 1) // OVERTIME_MINUTES


def seconds_left(console: str, period: int) -> int:
    """``CONSOLE`` ``mm:ss`` left in the period -> seconds, clipped to the period (``00:-1``: 0)."""
    minutes, seconds = console.split(":")
    length = (QUARTER_MINUTES if period <= 4 else OVERTIME_MINUTES) * 60
    return min(max(int(minutes) * 60 + int(seconds), 0), length)


def shot_band(distance: float, value: int) -> str:
    if value == 3:
        return "deep3" if distance >= DEEP3_M else "three"
    if distance < RIM_M:
        return "rim"
    if distance < SHORT_M:
        return "short"
    return "mid" if distance < MID_M else "long2"


def _flag(raw: Any) -> bool:
    return str(raw).strip() == "1"


class _Unparseable(ValueError):
    pass


def _parse(row: dict[str, Any], game: Game) -> dict[str, Any]:
    """One field-goal row -> shot fields; raises ``_Unparseable`` with the reason."""
    team = str(row.get("TEAM") or "").strip()
    if team not in (game.home, game.away):
        raise _Unparseable(f"team {team!r} is not in the game")
    action = str(row["ID_ACTION"]).strip()
    points = int(row["POINTS"])
    value = SHOT_VALUE[action]
    made = points > 0
    if points not in (0, value) or made != (action in MADE_CODES):
        raise _Unparseable(f"{action} with {points} points")
    after_home, after_away = row.get("POINTS_A"), row.get("POINTS_B")
    if after_home is None or after_away is None:
        raise _Unparseable("no score")
    minute, console = row.get("MINUTE"), str(row.get("CONSOLE") or "")
    if not isinstance(minute, int) or ":" not in console:
        raise _Unparseable(f"clock {minute!r} {console!r}")
    period = period_of(minute)
    own, opp = (after_home, after_away) if team == game.home else (after_away, after_home)
    coord_x, coord_y = row.get("COORD_X"), row.get("COORD_Y")
    if coord_x is None or coord_y is None:
        raise _Unparseable("no coordinates")
    if coord_x == FREE_THROW_SENTINEL and coord_y == FREE_THROW_SENTINEL:
        raise _Unparseable("free-throw sentinel (-1, -1) on a field goal")
    return {
        "team": team,
        "opponent": game.away if team == game.home else game.home,
        "home": team == game.home and not game.neutral,
        "shooter": str(row.get("ID_PLAYER") or "").strip(),
        "action": action,
        "points": points,
        "made": made,
        "value": value,
        "coord_x": float(coord_x),
        "coord_y": float(coord_y),
        "period": period,
        "seconds_left": seconds_left(console, period),
        "zone": str(row.get("ZONE") or "").strip(),
        "fastbreak": _flag(row.get("FASTBREAK")),
        "second_chance": _flag(row.get("SECOND_CHANCE")),
        "points_off_turnover": _flag(row.get("POINTS_OFF_TURNOVER")),
        "margin_before": int(own) - points - int(opp),
    }


def _excluded(game: Game, row: dict[str, Any], reason: str, detail: str) -> dict[str, Any]:
    x, y = row.get("COORD_X"), row.get("COORD_Y")
    return {
        "competition": COMPETITION,
        "season": game.season,
        "game_id": game.game_id,
        "event": int(row["NUM_ANOT"]),
        "team": str(row.get("TEAM") or "").strip(),
        "action": str(row["ID_ACTION"]).strip(),
        "points": int(row.get("POINTS") or 0),
        "coord_x": None if x is None else float(x),
        "coord_y": None if y is None else float(y),
        "reason": reason,
        "detail": detail,
    }


def game_shots(
    rows: list[dict[str, Any]], game: Game
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The game's field-goal attempts: (shots, excluded rows with their reason)."""
    shots: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for row in rows:
        action = str(row.get("ID_ACTION") or "").strip()
        if action == FREE_THROW:
            continue
        if action not in SHOT_VALUE:
            excluded.append(_excluded(game, row, "unparseable", f"unknown action {action!r}"))
            continue
        try:
            shot = _parse(row, game)
        except _Unparseable as exc:
            excluded.append(_excluded(game, row, "unparseable", str(exc)))
            continue
        if shot["coord_x"] == 0 and shot["coord_y"] == 0:
            excluded.append(_excluded(game, row, "zero_coordinates", "(0, 0)"))
            continue
        shot["event"] = int(row["NUM_ANOT"])
        shots.append(shot)
    return shots, excluded


def add_geometry(frame: pd.DataFrame) -> pd.DataFrame:
    x, y = to_court_coords(
        frame["coord_x"].to_numpy(dtype=np.float64), frame["coord_y"].to_numpy(dtype=np.float64)
    )
    distance = np.hypot(x, y)
    return frame.assign(
        x=x,
        y=y,
        distance=distance,
        angle=np.degrees(np.arctan2(np.abs(x), y)),
        band=[shot_band(d, v) for d, v in zip(distance, frame["value"], strict=True)],
    )


def label_contradicts_geometry(x: pd.Series, y: pd.Series, value: pd.Series) -> pd.Series:
    """A 3 inside the line moved in by the tolerance, or a 2 beyond it moved out."""
    xs, ys = x.to_numpy(dtype=np.float64), y.to_numpy(dtype=np.float64)
    inside = ~beyond_three_line(xs, ys, tolerance=LABEL_TOLERANCE_M)
    outside = beyond_three_line(xs, ys, tolerance=-LABEL_TOLERANCE_M)
    wrong = np.where(value.to_numpy() == 3, inside, outside)
    return pd.Series(wrong, index=x.index)


@dataclass(frozen=True)
class ShotTable:
    shots: pd.DataFrame
    excluded: pd.DataFrame
    not_cached: list[str]
    actions: dict[int, dict[str, int]]  # season -> raw ID_ACTION -> rows (the feed vocabulary)


def _games(games: pd.DataFrame) -> Iterator[Game]:
    rated = games[games["played"] & ~games["forfeit"] & (games["season"] <= LAST_SEASON)]
    rated = rated.sort_values(["season", "game_code"])
    for game_id, season, code, home, away, neutral in zip(
        rated["game_id"],
        rated["season"],
        rated["game_code"],
        rated["home"],
        rated["away"],
        rated["neutral"],
        strict=True,
    ):
        yield Game(str(game_id), int(season), int(code), str(home), str(away), bool(neutral))


def build_shot_table(raw_dir: Path, games: pd.DataFrame) -> ShotTable:
    """Shots and exclusions of every played game up to ``LAST_SEASON`` with a cached feed."""
    shot_rows: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    not_cached: list[str] = []
    actions: dict[int, Counter[str]] = {}
    for game in _games(games):
        path = raw_dir / "points" / f"E{game.season}" / f"{game.code}.json.gz"
        if not path.exists():
            not_cached.append(game.game_id)
            continue
        rows = json.loads(read_cached(path))["Rows"] or []
        actions.setdefault(game.season, Counter()).update(
            str(row.get("ID_ACTION") or "").strip() for row in rows
        )
        shots, dropped = game_shots(rows, game)
        for shot in shots:
            shot |= {"competition": COMPETITION, "season": game.season, "game_id": game.game_id}
        shot_rows += shots
        excluded += dropped
    frame = add_geometry(pd.DataFrame(shot_rows)) if shot_rows else pd.DataFrame()
    if len(frame):
        validated = frame["season"] >= FIRST_VALIDATED_SEASON
        wrong = validated & label_contradicts_geometry(frame["x"], frame["y"], frame["value"])
        for row in frame[wrong].to_dict("records"):
            excluded.append(
                {
                    "competition": COMPETITION,
                    "season": row["season"],
                    "game_id": row["game_id"],
                    "event": row["event"],
                    "team": row["team"],
                    "action": row["action"],
                    "points": row["points"],
                    "coord_x": row["coord_x"],
                    "coord_y": row["coord_y"],
                    "reason": "label_geometry",
                    "detail": f"{row['value']} at {row['distance']:.2f} m",
                }
            )
        frame = frame[~wrong].assign(validated_season=validated[~wrong])
    kept = pd.DataFrame(frame, columns=list(SHOTS_SCHEMA.columns))
    gone = pd.DataFrame(excluded, columns=list(EXCLUDED_SCHEMA.columns))
    kept = kept.astype(schema_dtypes(SHOTS_SCHEMA)).sort_values(["game_id", "event"])
    gone = gone.astype(schema_dtypes(EXCLUDED_SCHEMA)).sort_values(["game_id", "event"])
    return ShotTable(
        SHOTS_SCHEMA.validate(kept.reset_index(drop=True)),
        EXCLUDED_SCHEMA.validate(gone.reset_index(drop=True)),
        not_cached,
        {season: dict(sorted(counts.items())) for season, counts in sorted(actions.items())},
    )


def box_shooting(box: dict[str, Any]) -> dict[str, tuple[int, int]] | str:
    """Team -> (FGA, made-FG points) from the box score's ``totr``; or why there is none."""
    if len(box.get("Stats") or []) != 2 or not all(s["PlayersStats"] for s in box["Stats"]):
        return "box score has no players (API placeholder)"
    out = {}
    for side in box["Stats"]:
        totals = side["totr"]
        fga = int(totals["FieldGoalsAttempted2"]) + int(totals["FieldGoalsAttempted3"])
        points = 2 * int(totals["FieldGoalsMade2"]) + 3 * int(totals["FieldGoalsMade3"])
        out[side["PlayersStats"][0]["Team"].strip()] = (fga, points)
    return out


def feed_counts(table: ShotTable) -> pd.DataFrame:
    """Per team-game FGA and made-FG points of the whole feed (kept plus excluded rows)."""
    columns = ["season", "game_id", "team", "points"]
    kept = table.shots.assign(points=table.shots["made"] * table.shots["value"])
    rows = pd.concat([kept[columns], table.excluded[columns]], ignore_index=True)
    return (
        rows.groupby(["season", "game_id", "team"], as_index=False)
        .agg(fga=("points", "size"), fg_points=("points", "sum"))
        .astype({"fga": "int64", "fg_points": "int64"})
    )


def reconcile(table: ShotTable, raw_dir: Path, games: pd.DataFrame) -> pd.DataFrame:
    """Every validated-season team-game with a box score: feed vs box FGA and made-FG points.

    ``status`` is ``match``, ``mismatch`` or ``no_box`` (the API's empty placeholder).
    """
    counts = feed_counts(table)
    feed = {
        (str(g), str(t)): (int(n), int(p))
        for g, t, n, p in zip(
            counts["game_id"], counts["team"], counts["fga"], counts["fg_points"], strict=True
        )
    }
    rows = []
    for game in _games(games):
        if game.season < FIRST_VALIDATED_SEASON or game.game_id in table.not_cached:
            continue
        path = raw_dir / "boxscore" / f"E{game.season}" / f"{game.code}.json.gz"
        box = box_shooting(json.loads(read_cached(path))) if path.exists() else "no box score"
        for team in (game.home, game.away):
            fga, points = feed.get((game.game_id, team), (0, 0))
            row = {
                "season": game.season,
                "game_id": game.game_id,
                "team": team,
                "feed_fga": int(fga),
                "feed_fg_points": int(points),
                "box_fga": None,
                "box_fg_points": None,
                "status": "no_box",
            }
            if not isinstance(box, str) and team in box:
                row["box_fga"], row["box_fg_points"] = box[team]
                same = (row["feed_fga"], row["feed_fg_points"]) == box[team]
                row["status"] = "match" if same else "mismatch"
            rows.append(row)
    return pd.DataFrame(rows).astype({"box_fga": "Int64", "box_fg_points": "Int64"})


def shot_report(table: ShotTable, checks: pd.DataFrame) -> dict[str, Any]:
    """Per season: FGA, exclusions by reason and share, and the box reconciliation; JSON-ready."""
    seasons: dict[str, Any] = {}
    fga = pd.concat([table.shots["season"], table.excluded["season"]]).value_counts()
    for season in sorted(fga.index):
        gone = table.excluded[table.excluded["season"] == season]
        total = int(fga[season])
        block: dict[str, Any] = {
            "validated": bool(season >= FIRST_VALIDATED_SEASON),
            "actions": table.actions.get(int(season), {}),
            "fga": total,
            "kept": total - len(gone),
            "excluded_share": round(len(gone) / total, 6),
            "excluded_by_reason": {
                reason: {
                    "n": int((gone["reason"] == reason).sum()),
                    "share": round(float((gone["reason"] == reason).sum()) / total, 6),
                }
                for reason in REASONS
            },
        }
        rec = checks[checks["season"] == season]
        if len(rec):
            with_box = rec[rec["status"] != "no_box"]
            block["reconciliation"] = {
                "team_games": len(rec),
                "with_box": len(with_box),
                "match": int((with_box["status"] == "match").sum()),
                "match_rate": round(float((with_box["status"] == "match").mean()), 6),
                "mismatches": [
                    {
                        "game_id": r["game_id"],
                        "team": r["team"],
                        "fga_feed_minus_box": int(r["feed_fga"] - r["box_fga"]),
                        "fg_points_feed_minus_box": int(r["feed_fg_points"] - r["box_fg_points"]),
                    }
                    for r in with_box[with_box["status"] == "mismatch"].to_dict("records")
                ],
                "no_box": sorted({str(g) for g in rec.loc[rec["status"] == "no_box", "game_id"]}),
            }
        seasons[str(season)] = block
    validated = checks[checks["status"] != "no_box"]
    return {
        "first_validated_season": FIRST_VALIDATED_SEASON,
        "last_season": LAST_SEASON,
        "label_tolerance_m": LABEL_TOLERANCE_M,
        "shots": len(table.shots),
        "excluded": len(table.excluded),
        "not_cached": table.not_cached,
        "reconciliation_match_rate_validated": round(
            float((validated["status"] == "match").mean()), 6
        ),
        "seasons": seasons,
        "table_sha256": {
            "shots": digest(table.shots),
            "shots_excluded": digest(table.excluded),
        },
    }
