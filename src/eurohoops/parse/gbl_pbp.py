"""GBL play-by-play (BasketHotel xlsx export) -> a typed event table and per-player box lines.

The export has one sheet: ``Time | Score | <team> | <team> | Game Actions``. Time is elapsed
game time (``MM:SS``); the team column a sentence sits in says which team acted; Score is
filled on scoring rows only. Sentences are English templates (``(24) Frank BARTLEY made a free
throw (2 of 2)``). Only the templates below are classified; everything else is kept as
``other`` with its text, never guessed.

The two team columns are mapped to home/away by the sheet's final score, which must equal the
results page in one order (``PbpScoreError`` otherwise).
"""

import io
import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import openpyxl
import pandas as pd
import pandera.pandas as pa

from eurohoops.ingest.cache import read_cached
from eurohoops.parse.schemas import schema_dtypes

HEADER = ("Time", "Score")
ACTIONS = ("in", "out", "ft_made", "ft_missed", "fg_made", "fg_missed", "period", "other")
PLAYER = re.compile(r"^\((?P<number>\d+)\)\s+(?P<rest>.+)$")
VERB = re.compile(
    r"\s(?=(?:entered|left|made|missed|performed|perfomed|commited|committed|blocked|passed)\b)"
)
TEMPLATES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^entered the court$"), "in"),
    (re.compile(r"^left the court$"), "out"),
    (re.compile(r"^made a free throw\b"), "ft_made"),
    (re.compile(r"^missed a free throw\b"), "ft_missed"),
    (re.compile(r"^performed a (?P<value>[23]) points\b"), "fg_made"),
    (re.compile(r"^(?:missed|blocked while attempting) a (?P<value>[23]) points\b"), "fg_missed"),
)
PERIOD_START = re.compile(r"^Start of (?!game)")

PBP_SCHEMA = pa.DataFrameSchema(
    {
        "game_id": pa.Column(str),
        "seq": pa.Column("int64", pa.Check.ge(0)),
        "period": pa.Column("int64", pa.Check.ge(1)),
        "clock_s": pa.Column("int64", pa.Check.ge(0)),
        "side": pa.Column(str, pa.Check.isin(["home", "away", ""])),
        "number": pa.Column(str),
        "player": pa.Column(str),
        "action": pa.Column(str, pa.Check.isin(ACTIONS)),
        "value": pa.Column("int64", pa.Check.isin([0, 1, 2, 3])),  # shot value; 0 otherwise
        "home_score": pa.Column("Int64", nullable=True),
        "away_score": pa.Column("Int64", nullable=True),
        "text": pa.Column(str),
    },
    unique=["game_id", "seq"],
    strict=True,
)
log = logging.getLogger(__name__)
LINE_COUNTS = ("points", "fg2m", "fg2a", "fg3m", "fg3a", "ftm", "fta", "seconds")


class PbpFormatError(ValueError):
    """The export does not have the expected sheet layout."""


class PbpScoreError(ValueError):
    """The sheet's final score matches the results page in neither column order."""


@dataclass(frozen=True)
class RawRow:
    clock_s: int
    score: tuple[int, int] | None  # (first team column, second team column)
    column: int  # 0 first team, 1 second team, 2 game action
    text: str


def _clock(value: Any) -> int:
    minutes, seconds = str(value).split(":")
    return int(minutes) * 60 + int(seconds)


def read_export(xlsx: bytes) -> list[RawRow]:
    sheet = openpyxl.load_workbook(io.BytesIO(xlsx), read_only=True).worksheets[0]
    rows = sheet.iter_rows(values_only=True)
    header = next(rows, ())
    if tuple(header[:2]) != HEADER or len(header) != 5 or header[4] != "Game Actions":
        raise PbpFormatError(f"unexpected header {header}")
    out = []
    for time, score, first, second, action in rows:
        cells = [first, second, action]
        column = next((i for i, cell in enumerate(cells) if cell), None)
        if column is None:
            continue
        left, _, right = str(score or "").partition("-")
        out.append(
            RawRow(
                clock_s=_clock(time),
                score=(int(left), int(right)) if right else None,
                column=column,
                text=" ".join(str(cells[column]).split()),  # the export has double spaces
            )
        )
    return out


def classify(sentence: str) -> tuple[str, str, str, int]:
    """(number, player, action, shot value) of a team-column sentence."""
    match = PLAYER.match(sentence)
    if match is None:
        return "", "", "other", 0
    name, verb = _split_name(match["rest"])
    for pattern, action in TEMPLATES:
        found = pattern.match(verb)
        if found:
            value = int(found["value"]) if "value" in pattern.groupindex else 0
            return match["number"], name, action, 1 if action.startswith("ft") else value
    return match["number"], name, "other", 0


def _split_name(rest: str) -> tuple[str, str]:
    """``Frank BARTLEY made a free throw`` -> (``Frank BARTLEY``, ``made a free throw``)."""
    parts = VERB.split(rest, maxsplit=1)
    return (parts[0], parts[1]) if len(parts) == 2 else (rest, "")


def _sides(rows: list[RawRow], home_score: int, away_score: int) -> tuple[str, str]:
    final = next((row.score for row in reversed(rows) if row.score), None)
    if final == (home_score, away_score):
        return "home", "away"
    if final == (away_score, home_score):
        return "away", "home"
    raise PbpScoreError(f"sheet final {final}, result {home_score}-{away_score}")


def parse_export(xlsx: bytes, game_id: str, home_score: int, away_score: int) -> pd.DataFrame:
    rows = read_export(xlsx)
    first, second = _sides(rows, home_score, away_score)
    period = 0
    records = []
    for seq, row in enumerate(rows):
        if row.column == 2:
            period += bool(PERIOD_START.match(row.text))
            number, player, action, value, side = "", "", "period", 0, ""
        else:
            number, player, action, value = classify(row.text)
            side = (first, second)[row.column]
        scores = dict(zip((first, second), row.score or (None, None), strict=True))
        records.append(
            {
                "game_id": game_id,
                "seq": seq,
                "period": max(period, 1),
                "clock_s": row.clock_s,
                "side": side,
                "number": number,
                "player": player,
                "action": action,
                "value": value,
                "home_score": scores["home"],
                "away_score": scores["away"],
                "text": row.text,
            }
        )
    frame = pd.DataFrame(records, columns=list(PBP_SCHEMA.columns))
    return PBP_SCHEMA.validate(
        frame.astype({"home_score": "Int64", "away_score": "Int64", "seq": "int64"})
    )


def _on_court_seconds(events: pd.DataFrame, end: int) -> Iterator[tuple[tuple[str, str, str], int]]:
    """Seconds per (side, number, player) from IN/OUT rows; on court at ``end`` counts to it."""
    entered: dict[tuple[str, str, str], int] = {}
    for side, number, player, action, clock in zip(
        events["side"],
        events["number"],
        events["player"],
        events["action"],
        events["clock_s"],
        strict=True,
    ):
        key = (side, number, player)
        if action == "in":
            entered.setdefault(key, clock)
        elif action == "out" and key in entered:
            yield key, clock - entered.pop(key)
    for key, start in entered.items():
        yield key, end - start


def player_lines(events: pd.DataFrame) -> pd.DataFrame:
    """Box lines per player: points, 2P/3P/FT made and attempted, and seconds on court."""
    players = events[events["side"] != ""]
    lines: dict[tuple[str, str, str], dict[str, int]] = {}

    def line(key: tuple[str, str, str]) -> dict[str, int]:
        return lines.setdefault(key, dict.fromkeys(LINE_COUNTS, 0))

    for side, number, player, action, value in zip(
        players["side"],
        players["number"],
        players["player"],
        players["action"],
        players["value"],
        strict=True,
    ):
        key = (side, number, player)
        if action in {"fg_made", "fg_missed"}:
            line(key)[f"fg{value}a"] += 1
            if action == "fg_made":
                line(key)[f"fg{value}m"] += 1
                line(key)["points"] += value
        elif action in {"ft_made", "ft_missed"}:
            line(key)["fta"] += 1
            if action == "ft_made":
                line(key)["ftm"] += 1
                line(key)["points"] += 1
    # The game ends at its "End of game" row, usually after the last player event.
    for key, seconds in _on_court_seconds(players, int(events["clock_s"].max())):
        line(key)["seconds"] += seconds
    rows = [
        {"side": side, "number": number, "player": player, **counts}
        for (side, number, player), counts in sorted(lines.items())
        if number
    ]
    return pd.DataFrame(rows, columns=["side", "number", "player", *LINE_COUNTS])


def build_pbp_table(root: Path, games: pd.DataFrame) -> pd.DataFrame:
    """Every cached export of a played, non-forfeit game, as one validated event table.

    Exports whose final score matches the result in neither order are logged and left out.
    """
    frames = []
    rated = games[games["played"] & ~games["forfeit"]]
    for game_id, season, code, home_score, away_score in zip(
        rated["game_id"],
        rated["season"],
        rated["game_code"],
        rated["home_score"],
        rated["away_score"],
        strict=True,
    ):
        path = root / "pbp" / str(season) / f"{code:08X}.xlsx.gz"
        if not path.exists():
            continue
        try:
            frames.append(
                parse_export(read_cached(path), game_id, int(home_score), int(away_score))
            )
        except PbpScoreError as exc:
            log.warning("%s: %s", game_id, exc)
    if not frames:
        empty = pd.DataFrame(columns=list(PBP_SCHEMA.columns))
        return PBP_SCHEMA.validate(empty.astype(schema_dtypes(PBP_SCHEMA)))
    return PBP_SCHEMA.validate(pd.concat(frames, ignore_index=True))
