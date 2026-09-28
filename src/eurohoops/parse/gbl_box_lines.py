"""Cached ESAKE box-score HTML -> a ``player_games``-shaped frame (week 9-12 H2).

``parse/esake.py``'s ``BoxLine`` keeps only the fields the box invariant checks need (points,
shots, seconds); the staged ``player_box`` (``parse/box.py``) keeps the same. The box-only
baseline (H-e) needs every column ESAKE prints, so this module reads the remaining ones
(rebound split, assists, blocks, fouls, steals, turnovers, PIR) from the same tables, into the
column names ``stats/box.py`` uses for the EuroLeague ``player_games`` mart. It reuses
``parse/esake.py``'s regex and error class (not edited): ``ID_PLAYER``, ``EsakeParseError``,
``parse_overtimes``.

Column order after the player-name cell ("ΠΑΙΚΤΗΣ"), confirmed against
``tests/fixtures/esake/box_8FC479F6.html``: P, 2PM-A, 3PM-A, FTM-A, REBS, D.REBS, O.REBS, AST,
BLK, BLK-A, FOULS F, FOULS M, STL, TO, TIM.PL., RANK (17 cells with the name). FOULS M is fouls
committed (``pf``), FOULS F is fouls drawn, RANK is PIR: PIR = PTS + REB + AST + STL + BLK + FD
- missed FG - missed FT - TO - BLK-A - PF holds row by row (checked in
``tests/test_gbl_box_lines.py``). Team totals include the "ΟΜΑΔΙΚΑ - ΠΑΓΚΟΣ" team/bench row, so
player rows sum to the totals row for every column except rebounds and turnovers.

Home/away order follows ``parse/box.py``'s convention: the two tables are read in page order and
zipped directly with the game's ``(home, away)`` team codes from the games mart, with no
score-based re-ordering.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pandera.pandas as pa
from selectolax.lexbor import LexborHTMLParser, LexborNode

from eurohoops.ingest.cache import read_cached
from eurohoops.parse.esake import ID_PLAYER, EsakeParseError, parse_overtimes
from eurohoops.parse.schemas import validated

REGULATION_SECONDS = 2400
OVERTIME_SECONDS = 300
PLAYER_ROW_CELLS = 17

# Output column order (week 9-12 H2): the same names as `stats/box.py`'s `player_games`.
STATS = (
    "pts",
    "fg2m",
    "fg2a",
    "fg3m",
    "fg3a",
    "ftm",
    "fta",
    "oreb",
    "dreb",
    "ast",
    "stl",
    "blk",
    "tov",
    "pf",
)

PLAYER_GAMES_SCHEMA = pa.DataFrameSchema(
    {
        "season": pa.Column("int64"),
        "game_id": pa.Column(str),
        "team": pa.Column(str),
        "player_id": pa.Column(str),
        "sec": pa.Column("int64", pa.Check.ge(0)),
        "game_sec": pa.Column("int64", pa.Check.gt(0)),
        **{s: pa.Column("int64", pa.Check.ge(0)) for s in STATS},
        "pir": pa.Column("int64"),  # PIR (valuation) can be negative
        "poss": pa.Column("float64", pa.Check.ge(0)),
    },
    unique=["game_id", "player_id"],
    strict=True,
)


def _cell_text(node: LexborNode) -> str:
    return " ".join(node.text(separator=" ").split())


def _number(text: str) -> int:
    return 0 if text in {"", "-"} else int(text)


def _made_attempted(text: str) -> tuple[int, int]:
    made, _, attempted = text.partition("-")
    return _number(made.strip()), _number(attempted.strip())


def _seconds(text: str) -> int:
    if not text or text.startswith("-"):
        return 0
    hours, minutes, secs = (int(part) for part in text.split(":"))
    return hours * 3600 + minutes * 60 + secs


def _row_stats(cells: Sequence[str]) -> dict[str, int]:
    """Every stat one player row carries, including ``reb`` (total), ``blka`` and ``fd`` (the
    PIR-identity fields that the final schema does not keep: rebounds are already split into
    ``oreb``/``dreb``, and ``blka``/``fd`` are not part of the box-only feature set)."""
    fg2m, fg2a = _made_attempted(cells[2])
    fg3m, fg3a = _made_attempted(cells[3])
    ftm, fta = _made_attempted(cells[4])
    return {
        "pts": _number(cells[1]),
        "fg2m": fg2m,
        "fg2a": fg2a,
        "fg3m": fg3m,
        "fg3a": fg3a,
        "ftm": ftm,
        "fta": fta,
        "reb": _number(cells[5]),
        "dreb": _number(cells[6]),
        "oreb": _number(cells[7]),
        "ast": _number(cells[8]),
        "blk": _number(cells[9]),
        "blka": _number(cells[10]),
        "fd": _number(cells[11]),
        "pf": _number(cells[12]),
        "stl": _number(cells[13]),
        "tov": _number(cells[14]),
        "pir": _number(cells[16]),
    }


def _player_row(row: LexborNode) -> dict[str, Any] | None:
    """One player's full row (``player_id``, ``sec`` and every stat of ``_row_stats``); ``None``
    for a row without a player link (the totals and team/bench rows)."""
    link = row.css_first('a[href*="idplayer="]')
    if link is None:
        return None
    player = ID_PLAYER.search(link.attributes.get("href") or "")
    cells = [_cell_text(cell) for cell in row.css("td")]
    if player is None or len(cells) != PLAYER_ROW_CELLS:
        raise EsakeParseError(f"unexpected player row {cells}")
    return {"player_id": player.group(1), "sec": _seconds(cells[15]), **_row_stats(cells)}


def _team_tables(html: str) -> list[list[LexborNode]]:
    return [
        rows
        for rows in (table.css("tr") for table in LexborHTMLParser(html).css("table"))
        if any(_cell_text(row).startswith("ΠΑΙΚΤΗΣ") for row in rows)
    ]


def parse_box_lines(html: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | None:
    """Both teams' full player rows, in page order (home first); ``None`` if ESAKE has no box
    (some 2018-19 and 2019-20 game pages carry only the game header, per §3 finding 4)."""
    tables = _team_tables(html)
    if not tables:
        return None
    if len(tables) != 2:
        raise EsakeParseError(f"expected 2 team box scores, found {len(tables)}")
    parsed = [[line for row in rows if (line := _player_row(row)) is not None] for rows in tables]
    return parsed[0], parsed[1]


@dataclass(frozen=True)
class GblPlayerGames:
    table: pd.DataFrame
    skipped: int  # cached pages without a usable box-score table or team_games possessions


def _game_sec(html: str) -> int:
    return REGULATION_SECONDS + OVERTIME_SECONDS * parse_overtimes(html)


def build_gbl_player_games(
    raw_dir: Path, games: pd.DataFrame, team_games: pd.DataFrame
) -> GblPlayerGames:
    """Every rated (played, non-forfeit) GBL game in ``games`` with a cached box page.

    ``games``: the GBL games mart (``game_id, season, game_code, home, away, played, forfeit``).
    ``team_games``: the GBL rows of the ``team_games`` mart (``game_id, team, poss_game``), for
    each row's ``poss`` (5 * the game's possessions * ``sec`` / the team's recorded seconds, as
    ``stats/box.py`` computes it for EuroLeague). A page without a box-score table, or a
    (game_id, team) missing from ``team_games``, is skipped and counted, not raised on.
    """
    poss_game = team_games.drop_duplicates(["game_id", "team"]).set_index(["game_id", "team"])[
        "poss_game"
    ]
    rated = games[games["played"] & ~games["forfeit"]]
    rows: list[dict[str, Any]] = []
    skipped = 0
    for game_id, season, code, home, away in zip(
        rated["game_id"],
        rated["season"],
        rated["game_code"],
        rated["home"],
        rated["away"],
        strict=True,
    ):
        path = raw_dir / "boxscore" / str(int(season)) / f"{int(code):08X}.html.gz"
        if not path.exists():
            continue
        html = read_cached(path).decode()
        parsed = parse_box_lines(html)
        if parsed is None:
            skipped += 1
            continue
        game_sec = _game_sec(html)
        teams = (str(home), str(away))
        if any((game_id, team) not in poss_game.index for team in teams):
            skipped += 1
            continue
        for team, lines in zip(teams, parsed, strict=True):
            recorded = sum(line["sec"] for line in lines) or 1
            game_poss = float(poss_game[(game_id, team)])
            for line in lines:
                rows.append(
                    {
                        "season": int(season),
                        "game_id": str(game_id),
                        "team": team,
                        "player_id": line["player_id"],
                        "sec": line["sec"],
                        "game_sec": game_sec,
                        **{s: line[s] for s in STATS},
                        "pir": line["pir"],
                        "poss": 5.0 * game_poss * line["sec"] / recorded,
                    }
                )
    table = validated(
        pd.DataFrame(rows, columns=list(PLAYER_GAMES_SCHEMA.columns)), PLAYER_GAMES_SCHEMA
    )
    return GblPlayerGames(table, skipped)
