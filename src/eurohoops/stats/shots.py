"""Cached EuroLeague ``Points`` feed -> one row per field-goal attempt, plus hex bins.

Codes: ``2FGM 2FGA 3FGM 3FGA`` every season, blocked attempts ``2FGAB``/``3FGAB`` up to
2016-17, and ``LAYUPMD``/``LAYUPATT``/``DUNK`` in 2008-09 to 2014-15. ``FTM`` rows are free
throws, not shots. Rows at (0, 0) or at the free-throw sentinel (-1, -1) have no location; they
are counted in ``unplaced``, not kept.
Coordinates before 2011-12 fit neither three-point line (``parse/shots.py``); those seasons'
charts are flagged on the site, never hidden.

Distance bands are the M2 reporting bands (F-f, F-e): by distance for 2s, split at 8 m for 3s.
"""

import json
import math
from collections.abc import Hashable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pandera.pandas as pa

from eurohoops.ingest.cache import read_cached
from eurohoops.parse.schemas import schema_dtypes
from eurohoops.parse.shots import FIRST_VALIDATED_SEASON, FREE_THROW_SENTINEL, to_court_coords

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
BANDS = ("rim", "short", "mid", "long2", "three", "deep3")
RIM_M, SHORT_M, MID_M, DEEP3_M = 1.5, 3.0, 5.0, 8.0
HEX_RADIUS_M = 0.5  # centre to corner of a pointy-top hexagon
SQRT3 = math.sqrt(3.0)
NO_LOCATION = ((0, 0), (FREE_THROW_SENTINEL, FREE_THROW_SENTINEL))

SHOTS_SCHEMA = pa.DataFrameSchema(
    {
        "season": pa.Column("int64"),
        "game_id": pa.Column(str),
        "team": pa.Column(str),
        "opponent": pa.Column(str),
        "player_id": pa.Column(str),
        "value": pa.Column("int64", pa.Check.isin([2, 3])),
        "made": pa.Column(bool),
        "x": pa.Column("float64"),
        "y": pa.Column("float64"),
        "band": pa.Column(str, pa.Check.isin(BANDS)),
        "period": pa.Column("int64", pa.Check.ge(1)),
        "fastbreak": pa.Column(bool),
        "second_chance": pa.Column(bool),
    },
    strict=True,
)


@dataclass(frozen=True)
class Shots:
    table: pd.DataFrame
    coverage: pd.DataFrame  # per season: games with a cached feed, shots kept, shots unplaced


def bands(distance: pd.Series, value: pd.Series) -> pd.Series:
    """Distance band of each shot (metres from the basket, 2 or 3 points)."""
    three = value == 3
    choice = np.select(
        [
            three & (distance >= DEEP3_M),
            three,
            distance < RIM_M,
            distance < SHORT_M,
            distance < MID_M,
        ],
        ["deep3", "three", "rim", "short", "mid"],
        default="long2",
    )
    return pd.Series(choice, index=distance.index, dtype=str)


def period(minute: int) -> int:
    """``MINUTE`` 1-40 -> quarters 1-4; 41-45 -> 5 (first overtime), 46-50 -> 6, ..."""
    if minute <= 40:
        return (max(minute, 1) - 1) // 10 + 1
    return 5 + (minute - 41) // 5


def _flag(raw: Any) -> bool:
    return str(raw).strip() == "1"


def game_shots(feed: dict[str, Any], game: dict[Hashable, Any]) -> tuple[list[dict[str, Any]], int]:
    """Field-goal rows of one game and how many had no location."""
    teams = (str(game["home"]), str(game["away"]))
    rows: list[dict[str, Any]] = []
    unplaced = 0
    for row in feed.get("Rows") or []:
        value = SHOT_VALUE.get(str(row["ID_ACTION"]).strip())
        team = str(row["TEAM"] or "").strip()
        if value is None or team not in teams:
            continue
        cx, cy = row.get("COORD_X"), row.get("COORD_Y")
        if cx is None or cy is None or (cx, cy) in NO_LOCATION:
            unplaced += 1
            continue
        rows.append(
            {
                "season": int(game["season"]),
                "game_id": str(game["game_id"]),
                "team": team,
                "opponent": teams[1] if team == teams[0] else teams[0],
                "player_id": str(row["ID_PLAYER"] or "").strip(),
                "value": value,
                "made": int(row["POINTS"] or 0) > 0,
                "coord_x": float(cx),
                "coord_y": float(cy),
                "period": period(int(row["MINUTE"] or 1)),
                "fastbreak": _flag(row.get("FASTBREAK")),
                "second_chance": _flag(row.get("SECOND_CHANCE")),
            }
        )
    return rows, unplaced


RAW_COLUMNS = (
    *(c for c in SHOTS_SCHEMA.columns if c not in ("x", "y", "band")),
    "coord_x",
    "coord_y",
)


def _with_geometry(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw centimetres -> court metres and the distance band, in the schema's column order."""
    x, y = to_court_coords(raw["coord_x"].to_numpy(float), raw["coord_y"].to_numpy(float))
    frame = raw.drop(columns=["coord_x", "coord_y"]).assign(x=x, y=y)
    frame["band"] = bands(pd.Series(np.hypot(x, y), index=raw.index), frame["value"])
    return frame[list(SHOTS_SCHEMA.columns)]


def build_shots(raw_dir: Path, games: pd.DataFrame) -> Shots:
    """Every cached ``Points`` feed of a played, non-forfeit game in ``games``."""
    rows: list[dict[str, Any]] = []
    coverage: dict[int, dict[str, int]] = {}
    rated = games[games["played"] & ~games["forfeit"]]
    for game in rated.to_dict("records"):
        season = int(game["season"])
        seen = coverage.setdefault(season, {"games": 0, "shots": 0, "unplaced": 0})
        path = raw_dir / "points" / f"E{season}" / f"{game['game_code']}.json.gz"
        if not path.exists():
            continue
        shots, unplaced = game_shots(json.loads(read_cached(path)), game)
        seen["games"] += 1 if shots else 0
        seen["shots"] += len(shots)
        seen["unplaced"] += unplaced
        rows += shots
    table = _with_geometry(pd.DataFrame(rows, columns=list(RAW_COLUMNS)))
    cover = pd.DataFrame(
        [{"season": s, **c, "validated": s >= FIRST_VALIDATED_SEASON} for s, c in coverage.items()],
        columns=["season", "games", "shots", "unplaced", "validated"],
    )
    return Shots(SHOTS_SCHEMA.validate(table.astype(schema_dtypes(SHOTS_SCHEMA))), cover)


def hex_cells(
    x: pd.Series, y: pd.Series, radius: float = HEX_RADIUS_M
) -> tuple[pd.Series, pd.Series]:
    """Axial (q, r) of the pointy-top hexagon holding each point (cube rounding).

    The cell centre is ``(radius * sqrt(3) * (q + r / 2), radius * 1.5 * r)``.
    """
    fq = (SQRT3 / 3.0 * x - y / 3.0) / radius
    fr = (2.0 / 3.0 * y) / radius
    fs = -fq - fr
    q, r, s = fq.round(), fr.round(), fs.round()
    dq, dr, ds = (q - fq).abs(), (r - fr).abs(), (s - fs).abs()
    fix_q = (dq > dr) & (dq > ds)
    fix_r = ~fix_q & (dr > ds)
    q = q.where(~fix_q, -r - s)
    r = r.where(~fix_r, -q - s)
    return q.astype("int64"), r.astype("int64")


def hexbins(
    shots: pd.DataFrame, by: str | None = None, radius: float = HEX_RADIUS_M
) -> dict[str, list[list[int]]]:
    """``[q, r, attempts, makes, points]`` per occupied cell, sorted by (q, r), per ``by`` value.

    Without ``by`` every shot falls under the key ``""``.
    """
    if shots.empty:
        return {}
    q, r = hex_cells(shots["x"], shots["y"], radius)
    frame = pd.DataFrame(
        {
            "key": shots[by].astype(str) if by else "",
            "q": q,
            "r": r,
            "made": shots["made"].astype("int64"),
            "pts": shots["value"] * shots["made"],
        }
    )
    cells = frame.groupby(["key", "q", "r"]).agg(
        att=("made", "size"), made=("made", "sum"), pts=("pts", "sum")
    )
    out: dict[str, list[list[int]]] = {}
    for (key, qq, rr), row in zip(cells.index, cells.to_numpy(), strict=True):
        out.setdefault(str(key), []).append([int(qq), int(rr), *(int(v) for v in row)])
    return out


def band_table(shots: pd.DataFrame, by: str | None = None) -> dict[str, dict[str, list[int]]]:
    """Band -> ``[attempts, makes]`` (every band, zeros included) per ``by`` value (or ``""``)."""
    keys = shots[by].astype(str) if by else pd.Series("", index=shots.index)
    grouped = shots.groupby([keys, shots["band"]])["made"].agg(["size", "sum"])
    out: dict[str, dict[str, list[int]]] = {}
    for (key, b), (n, m) in zip(grouped.index, grouped.to_numpy(), strict=True):
        out.setdefault(str(key), {name: [0, 0] for name in BANDS})[str(b)] = [int(n), int(m)]
    return out


def band_counts(shots: pd.DataFrame) -> dict[str, list[int]]:
    """Band -> ``[attempts, makes]`` of all ``shots`` (every band, zeros included)."""
    return band_table(shots).get("", {name: [0, 0] for name in BANDS})
