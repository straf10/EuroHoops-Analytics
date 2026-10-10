"""Load validation for the stats site: per-season coverage, format checks and special seasons.

The result is ``web/src/data/stats/validation.json``; its shape is pinned by the fixture
``tests/fixtures/validation.json``.
"""

from typing import Literal, TypedDict

Severity = Literal["info", "warn", "error"]


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
