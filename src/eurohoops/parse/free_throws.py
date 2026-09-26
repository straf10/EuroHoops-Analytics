"""Free-throw generation (F-e): FT trips and FT points per team-game from EuroLeague play-by-play.

The shot feed has no free-throw attempts on missed shots (it logs made free throws only), so
xPTS alone undervalues shots that draw fouls. The play-by-play has every free throw (``FTM``
made, ``FTA`` missed), but its fouls are untyped (``CM`` is any personal foul): a trip can be
tied to the shot that drew it only when it is an **and-one**, i.e. a one-shot trip whose last
field-goal row before it is a made field goal by the same team at the same clock. That shot's
``NUMBEROFPLAY`` is the feed's ``NUM_ANOT``, which gives the shot's distance band. Every other
trip (shooting fouls on misses, bonus free throws, technicals) stays at team level.

A trip is a run of free-throw rows of one team at one clock (period + ``MARKERTIME``), other
rows in between (substitutions, time-outs) do not split it. Expected FT points of a team-game:

    sum over bands b of FGA_b * and-one trips per FGA_b * points per and-one trip
    + FGA * other trips per FGA * points per other trip

with every rate a ratio of totals over the seasons it is fitted on (leave-one-season-out for
development seasons, all development seasons for validation and test).
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eurohoops.ingest.cache import read_cached
from eurohoops.parse.shot_table import BANDS, MADE_CODES, SHOT_VALUE, period_of
from eurohoops.parse.stints import QUARTERS

FT_CODES = frozenset({"FTM", "FTA"})
COUNT_COLUMNS = (
    "fga",
    *(f"fga_{b}" for b in BANDS),
    "fta",
    "ftm",
    "trips",
    "and_one_trips",
    *(f"and_one_trips_{b}" for b in BANDS),
    "and_one_ftm",
    "other_trips",
    "other_ftm",
)


@dataclass(frozen=True)
class Trip:
    team: str
    period: int
    clock: str
    fta: int
    ftm: int
    kind: str  # "and_one" or "other"
    shot_event: int | None  # NUMBEROFPLAY of the made field goal an and-one belongs to


def pbp_rows(pbp: dict[str, Any]) -> list[dict[str, Any]]:
    """Every PBP row in log order."""
    return [row for key in (*QUARTERS, "ExtraTime") for row in pbp.get(key) or []]


def trips(rows: Sequence[dict[str, Any]]) -> list[Trip]:
    """Free-throw trips of one game in log order, and-ones tied to their made field goal."""
    out: list[dict[str, Any]] = []
    last_fg: dict[str, Any] | None = None
    for row in rows:
        kind = str(row.get("PLAYTYPE") or "").strip()
        team = str(row.get("CODETEAM") or "").strip()
        if kind in SHOT_VALUE:
            last_fg = row
            continue
        if kind not in FT_CODES:
            continue
        key = (team, period_of(int(row["MINUTE"])), str(row.get("MARKERTIME") or ""))
        if out and out[-1]["key"] == key:
            out[-1]["fta"] += 1
            out[-1]["ftm"] += kind == "FTM"
            continue
        tied = (
            last_fg is not None
            and str(last_fg.get("PLAYTYPE") or "").strip() in MADE_CODES
            and str(last_fg.get("CODETEAM") or "").strip() == team
            and (period_of(int(last_fg["MINUTE"])), str(last_fg.get("MARKERTIME") or "")) == key[1:]
        )
        out.append(
            {
                "key": key,
                "fta": 1,
                "ftm": int(kind == "FTM"),
                "event": int(last_fg["NUMBEROFPLAY"]) if tied and last_fg is not None else None,
            }
        )
    return [
        Trip(
            t["key"][0],
            t["key"][1],
            t["key"][2],
            t["fta"],
            t["ftm"],
            "and_one" if t["event"] is not None and t["fta"] == 1 else "other",
            t["event"] if t["event"] is not None and t["fta"] == 1 else None,
        )
        for t in out
    ]


def build_ft_team_games(raw_dir: Path, shots: pd.DataFrame, excluded: pd.DataFrame) -> pd.DataFrame:
    """One row per validated-season team-game of the shot table: FGA (all, and per band of the
    kept shots), FT trips and FT points by kind. Games without cached PBP are left out."""
    band_of = {
        (str(g), int(e)): str(b)
        for g, e, b in zip(shots["game_id"], shots["event"], shots["band"], strict=True)
    }
    kept = shots[shots["validated_season"]]
    fga_band = kept.pivot_table(
        index=["season", "game_id", "team"],
        columns="band",
        values="event",
        aggfunc="size",
        fill_value=0,
    )
    fga_band = fga_band.reindex(columns=list(BANDS), fill_value=0).add_prefix("fga_")
    dropped = excluded[excluded["game_id"].isin(set(kept["game_id"]))]
    fga_all = pd.concat([kept[["game_id", "team"]], dropped[["game_id", "team"]]])
    fga = fga_all.groupby(["game_id", "team"]).size().rename("fga")
    frame = fga_band.reset_index()
    frame["fga"] = [
        fga.get((g, t), 0) for g, t in zip(frame["game_id"], frame["team"], strict=True)
    ]
    ft: dict[tuple[str, str], dict[str, int]] = {}
    has_pbp: set[str] = set()
    for game_id, season in sorted(set(zip(frame["game_id"], frame["season"], strict=True))):
        code = str(game_id).split("_")[1]
        path = raw_dir / "playbyplay" / f"E{season}" / f"{code}.json.gz"
        if not path.exists():
            continue
        has_pbp.add(str(game_id))
        for trip in trips(pbp_rows(json.loads(read_cached(path)))):
            tally = ft.setdefault((str(game_id), trip.team), {})
            band = band_of.get((str(game_id), trip.shot_event or -1))
            # an and-one tied to an excluded shot has no band: it counts as an other trip
            kind = "and_one" if trip.kind == "and_one" and band is not None else "other"
            names = ["fta", "ftm", "trips", f"{kind}_trips", f"{kind}_ftm"]
            if kind == "and_one":
                names.append(f"and_one_trips_{band}")
            for name in names:
                step = trip.fta if name == "fta" else trip.ftm if name.endswith("ftm") else 1
                tally[name] = tally.get(name, 0) + step
    frame = frame[frame["game_id"].isin(has_pbp)].reset_index(drop=True)
    for column in COUNT_COLUMNS:
        if column.startswith("fga"):
            continue
        frame[column] = [
            ft.get((g, t), {}).get(column, 0)
            for g, t in zip(frame["game_id"], frame["team"], strict=True)
        ]
    frame = frame[["season", "game_id", "team", *COUNT_COLUMNS]]
    return frame.astype({c: "int64" for c in ("season", *COUNT_COLUMNS)}).sort_values(
        ["game_id", "team"], ignore_index=True
    )


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def fit_rates(team_games: pd.DataFrame) -> dict[str, Any]:
    """Ratio-of-totals rates on the given team-games (0 where a denominator is empty)."""
    total = team_games.sum(numeric_only=True)
    return {
        "and_one_trips_per_fga": {
            b: _ratio(total[f"and_one_trips_{b}"], total[f"fga_{b}"]) for b in BANDS
        },
        "points_per_and_one_trip": _ratio(total["and_one_ftm"], total["and_one_trips"]),
        "other_trips_per_fga": _ratio(total["other_trips"], total["fga"]),
        "points_per_other_trip": _ratio(total["other_ftm"], total["other_trips"]),
    }


def expected_ft_points(team_games: pd.DataFrame, rates: dict[str, Any]) -> pd.Series:
    and_one = (
        sum(team_games[f"fga_{b}"] * rates["and_one_trips_per_fga"][b] for b in BANDS)
        * rates["points_per_and_one_trip"]
    )
    other = team_games["fga"] * rates["other_trips_per_fga"] * rates["points_per_other_trip"]
    return pd.Series(np.asarray(and_one + other, dtype=np.float64), index=team_games.index)


DEVELOPMENT_TOLERANCE = 0.1  # mean |actual - expected| FT points per team-game (F2 Done-when)
VALIDATION_TOLERANCE = 0.2


def _rounded(rates: dict[str, Any]) -> dict[str, Any]:
    return {
        k: {b: round(v, 6) for b, v in value.items()}
        if isinstance(value, dict)
        else round(value, 6)
        for k, value in rates.items()
    }


def ft_report(
    team_games: pd.DataFrame, development: Sequence[int], validation: Sequence[int]
) -> dict[str, Any]:
    """Rates on development, and per season the mean actual vs expected FT points per team-game
    (LOSO for development seasons); JSON-ready."""
    scored = team_games[team_games["season"].isin([*development, *validation])]
    expected = out_of_fold_expected(scored, development, validation)
    seasons: dict[str, Any] = {}
    for season, rows in scored.groupby("season"):
        gap = float((rows["ftm"] - expected[rows.index]).mean())
        tolerance = DEVELOPMENT_TOLERANCE if season in development else VALIDATION_TOLERANCE
        total = rows.sum(numeric_only=True)
        seasons[str(season)] = {
            "split": "development" if season in development else "validation",
            "team_games": len(rows),
            "ft_points_per_team_game": round(float(rows["ftm"].mean()), 4),
            "expected_per_team_game": round(float(expected[rows.index].mean()), 4),
            "mean_gap": round(gap, 4),
            "tolerance": tolerance,
            "within_tolerance": abs(gap) <= tolerance,
            "trips_per_fga": round(float(total["trips"] / total["fga"]), 6),
            "points_per_trip": round(float(total["ftm"] / total["trips"]), 6),
            "and_one_share_of_trips": round(float(total["and_one_trips"] / total["trips"]), 6),
        }
    dev = scored["season"].isin(development)
    return {
        "method": "ratio-of-totals rates; development seasons leave-one-season-out",
        "fouls_tied_to_shots": "and-ones only (a one-shot trip after the same team's made field "
        "goal at the same clock); other trips are team-level",
        "development": list(development),
        "validation": list(validation),
        "rates_development": _rounded(fit_rates(scored[dev])),
        "season_ft_points_sd_development": round(
            float(scored[dev].groupby("season")["ftm"].mean().std(ddof=1)), 4
        ),
        "seasons": seasons,
        "all_within_tolerance": all(s["within_tolerance"] for s in seasons.values()),
    }


def out_of_fold_expected(
    team_games: pd.DataFrame, development: Sequence[int], later: Sequence[int]
) -> pd.Series:
    """Expected FT points per team-game: development seasons with rates fitted on the other
    development seasons (LOSO), later seasons with rates fitted on all development seasons."""
    out = pd.Series(np.nan, index=team_games.index)
    dev = team_games["season"].isin(development)
    for season in development:
        mask = team_games["season"] == season
        rates = fit_rates(team_games[dev & ~mask])
        out[mask] = expected_ft_points(team_games[mask], rates)
    rates = fit_rates(team_games[dev])
    held = team_games["season"].isin(later)
    out[held] = expected_ft_points(team_games[held], rates)
    return out
