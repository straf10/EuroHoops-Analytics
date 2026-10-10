"""Load validation for the stats site: per-season coverage, format checks and special seasons.

The result is ``web/src/data/stats/validation.json``; its shape is pinned by the fixture
``tests/fixtures/validation.json``. Every season in the games table is covered, published or
not. Flags are plain English (the Methodology page shows them); a warn or error flag is also
echoed to stderr, so a problem is never silent.

Counting: ``games_scheduled`` is every row of the games table for the season (regular season,
playoffs and Final Four); ``games_played`` those with ``played`` set (forfeits included).
Box scores and shots are expected for played games that were not forfeited.
"""

import json
from collections.abc import Callable, Iterable
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, TypedDict

import pandas as pd
import pandera.pandas as pa
import typer

from eurohoops.config import SPECIAL_SEASONS
from eurohoops.parse.games import GAMES_SCHEMA
from eurohoops.stats.box import MISSING_SCHEMA, PLAYER_GAMES_SCHEMA, TEAM_STAT_GAMES_SCHEMA
from eurohoops.stats.export import Inputs, season_label, site_window
from eurohoops.stats.shots import SHOTS_SCHEMA

Severity = Literal["info", "warn", "error"]
VALIDATION_FILE = "validation.json"


class Flag(TypedDict):
    code: str  # stable snake_case
    severity: Severity
    message: str  # plain English a site visitor can read


class Special(TypedDict):
    kind: str
    note: str


class SeasonReport(TypedDict):
    season: int
    label: str
    published: bool  # in the site window
    games_scheduled: int
    games_played: int
    box_missing: int
    shots_games: int
    flags: list[Flag]
    special: Special | None


class Validation(TypedDict):
    built: str  # ISO-8601, e.g. 2026-10-10T08:41Z
    seasons: list[SeasonReport]  # ascending


def published_seasons(inputs: Inputs) -> list[int]:
    """The seasons the site publishes (the window ``build_payloads`` uses)."""
    return site_window(inputs.box.players["season"].unique())


def _plural(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


def _schema_problem(frame: pd.DataFrame, schema: pa.DataFrameSchema, season: int) -> str | None:
    """Why ``frame``'s rows of ``season`` break ``schema`` (missing columns first), or None."""
    missing = [c for c in [*schema.columns, "season"] if c not in frame.columns]
    if missing:
        return f"missing columns {', '.join(missing)}"
    try:
        schema.validate(frame[frame["season"] == season], lazy=True)
    except pa.errors.SchemaErrors as err:
        cases = err.failure_cases
        checks = sorted({f"{c}: {k}" for c, k in zip(cases["column"], cases["check"], strict=True)})
        return "; ".join(checks[:3])
    return None


def _schema_flags(inputs: Inputs, season: int, published: bool) -> list[Flag]:
    frames: list[tuple[str, pd.DataFrame, pa.DataFrameSchema]] = [
        ("games", inputs.games, GAMES_SCHEMA),
        ("player box line", inputs.box.players, PLAYER_GAMES_SCHEMA),
        ("team box line", inputs.box.teams, TEAM_STAT_GAMES_SCHEMA),
        ("missing box score", inputs.box.missing, MISSING_SCHEMA),
        ("shot", inputs.shots.table, SHOTS_SCHEMA),
    ]
    flags: list[Flag] = []
    for name, frame, schema in frames:
        problem = _schema_problem(frame, schema, season)
        if problem:
            flags.append(
                {
                    "code": "schema_mismatch",
                    "severity": "error" if published else "warn",
                    "message": f"The {name} data does not match its expected format ({problem}).",
                }
            )
    return flags


def _season_report(inputs: Inputs, season: int, published: bool) -> SeasonReport:
    games = inputs.games[inputs.games["season"] == season]
    scheduled, played = len(games), int(games["played"].sum())
    rated = int((games["played"] & ~games["forfeit"]).sum())
    box_missing = int((inputs.box.missing["season"] == season).sum())
    cover = inputs.shots.coverage[inputs.shots.coverage["season"] == season]
    shots_games, unplaced = int(cover["games"].sum()), int(cover["unplaced"].sum())
    special = SPECIAL_SEASONS.get(season)
    flags: list[Flag] = []

    def flag(code: str, severity: Severity, message: str) -> None:
        flags.append({"code": code, "severity": severity, "message": message})

    if published and played == 0:
        flag("no_games_played", "error", "No games have been played, so nothing can be shown.")
    unplayed = scheduled - played
    if unplayed > 0 and played > 0:
        what = _plural(unplayed, "scheduled game", "scheduled games")
        if season == inputs.live_season:
            flag("games_unplayed", "info", f"{what} still to be played.")
        else:  # a declared special season explains its own gap
            flag("games_unplayed", "info" if special else "warn", f"{what} never played.")
    if box_missing:
        who = _plural(box_missing, "played game has", "played games have")
        flag("box_missing", "warn", f"{who} no box score.")
    no_shots = max(rated - shots_games, 0)
    if no_shots:
        who = _plural(no_shots, "played game has", "played games have")
        flag("shots_missing", "warn", f"{who} no shot locations.")
    if unplaced:
        flag("shots_unplaced", "info", f"{unplaced:,} shots have no court position.")
    flags += _schema_flags(inputs, season, published)
    if special:
        flag("special_season", "info", special[1])
    return {
        "season": season,
        "label": season_label(season),
        "published": published,
        "games_scheduled": scheduled,
        "games_played": played,
        "box_missing": box_missing,
        "shots_games": shots_games,
        "flags": flags,
        "special": None if special is None else {"kind": special[0], "note": special[1]},
    }


def validate(inputs: Inputs, now: datetime, published: Iterable[int]) -> Validation:
    """Coverage, format and special-season report for every season in the games table."""
    window = set(published)
    seasons = sorted({int(s) for s in inputs.games["season"].unique()})
    return {
        "built": now.strftime("%Y-%m-%dT%H:%MZ"),
        "seasons": [_season_report(inputs, s, s in window) for s in seasons],
    }


def report_lines(report: Validation) -> list[tuple[str, bool]]:
    """The build report: a summary line per season, then each warn/error flag.

    Returns (line, is_warning) pairs; warnings carry the ``warning:`` prefix.
    """
    lines: list[tuple[str, bool]] = []
    for s in report["seasons"]:
        warned = [f for f in s["flags"] if f["severity"] != "info"]
        status = "published" if s["published"] else "not published"
        lines.append(
            (
                f"{s['label']}: {s['games_played']}/{s['games_scheduled']} games played, "
                f"{s['box_missing']} without box score, {s['shots_games']} with shots, "
                f"{status}, {len(warned)} warning(s)",
                False,
            )
        )
        lines += [
            (f"warning: {s['label']}: [{f['severity']}] {f['message']}", True) for f in warned
        ]
    return lines


def write_validation(
    out_dir: Path,
    inputs: Inputs,
    now: datetime,
    echo: Callable[..., Any] = typer.echo,
) -> Validation:
    """Write ``validation.json`` below ``out_dir`` and print the build report; exit non-zero
    (after writing) when any flag has severity error."""
    report = validate(inputs, now, published_seasons(inputs))
    out_dir.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, ensure_ascii=False, separators=(",", ":"))
    (out_dir / VALIDATION_FILE).write_text(text + "\n", encoding="utf-8", newline="\n")
    for line, is_warning in report_lines(report):
        echo(line, err=is_warning)
    if any(f["severity"] == "error" for s in report["seasons"] for f in s["flags"]):
        raise typer.Exit(code=1)
    return report
