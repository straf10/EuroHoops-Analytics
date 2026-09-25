"""Parsers for ESAKE (Greek Basket League) server-rendered HTML.

Results pages list one round. Each game block shows the date as Athens local time without a
year ("Σαβ 4 Οκτ - 16:00"); the year comes from the preceding month header ("Οκτώβριος 2025").
A missing time means the date is not confirmed yet. Scores: "85 - 87" played, " - " unplayed,
"20 - " a forfeit awarded to the side with the number.
"""

import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime

from selectolax.lexbor import LexborHTMLParser, LexborNode

MONTHS = (
    "ΙΑΝΟΥΑΡΙΟΣ",
    "ΦΕΒΡΟΥΑΡΙΟΣ",
    "ΜΑΡΤΙΟΣ",
    "ΑΠΡΙΛΙΟΣ",
    "ΜΑΙΟΣ",
    "ΙΟΥΝΙΟΣ",
    "ΙΟΥΛΙΟΣ",
    "ΑΥΓΟΥΣΤΟΣ",
    "ΣΕΠΤΕΜΒΡΙΟΣ",
    "ΟΚΤΩΒΡΙΟΣ",
    "ΝΟΕΜΒΡΙΟΣ",
    "ΔΕΚΕΜΒΡΙΟΣ",
)
FORFEIT_POINTS = 20
OVERTIME_LABELS = ("ΟΤ", "OT")  # "ΟΤ1" is written with Greek letters
ROUND_OPTION = re.compile(r"new Option\('([^']*)', '([^']*)'\)")
GAME_DATE = re.compile(r"(\d{1,2})\s+(\S+?)(?:\s*-\s*(\d{1,2}):(\d{2}))?$")
TEAM_LOGO = re.compile(r"/esaketeam/([0-9A-F]{8})/")
ID_GAME = re.compile(r"idgame=([0-9A-F]{8})")
ID_PLAYER = re.compile(r"idplayer=([0-9A-F]{8})")


class EsakeParseError(ValueError):
    """The page does not have the structure this parser was written against."""


@dataclass(frozen=True)
class GblGame:
    idgame: str
    round_label: str
    tipoff_local: datetime  # naive, Europe/Athens
    confirmed_date: bool
    home: str
    away: str
    home_name: str
    away_name: str
    home_score: int | None
    away_score: int | None
    forfeit: bool


@dataclass(frozen=True)
class ResultsPage:
    round_codes: tuple[str, ...]  # the phase's round list; empty on old-format playoff pages
    games: tuple[GblGame, ...]


@dataclass(frozen=True)
class BoxLine:
    """The box-score fields the invariant checks use; the full row stays in the raw cache."""

    player_id: str
    points: int
    fg2m: int
    fg2a: int
    fg3m: int
    fg3a: int
    ftm: int
    fta: int
    seconds: int


@dataclass(frozen=True)
class TeamLine:
    """The team-level counts of the totals row or of the team/bench row (``ΟΜΑΔΙΚΑ - ΠΑΓΚΟΣ``).

    The totals row includes the team row: team rebounds and team turnovers are in both.
    """

    points: int
    fg2a: int
    fg3a: int
    fta: int
    oreb: int
    dreb: int
    tov: int


@dataclass(frozen=True)
class TeamBox:
    players: tuple[BoxLine, ...]
    total_points: int
    totals: TeamLine
    team_row: TeamLine


def _plain(text: str) -> str:
    """Upper-case without accents/diaeresis, so 'Μάϊος' and 'Μαϊ' compare equal."""
    decomposed = unicodedata.normalize("NFD", text.strip().upper())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def month_number(name: str) -> int:
    """1-12 for a Greek month name or abbreviation ('Οκτ', 'Ιουν', 'Μάϊος')."""
    plain = _plain(name).rstrip(".")
    matches = [i for i, month in enumerate(MONTHS, start=1) if month.startswith(plain)]
    if len(matches) != 1:
        raise EsakeParseError(f"unknown month {name!r}")
    return matches[0]


def _text(node: LexborNode | None) -> str:
    return " ".join((node.text(separator=" ") if node is not None else "").split())


def _scores(text: str) -> tuple[int | None, int | None, bool]:
    home, _, away = text.replace("\xa0", " ").partition("-")
    home, away = home.strip(), away.strip()
    if home.isdigit() and away.isdigit():
        return int(home), int(away), False
    if not home and not away:
        return None, None, False
    if home.isdigit() and not away and int(home) == FORFEIT_POINTS:
        return FORFEIT_POINTS, 0, True
    if away.isdigit() and not home and int(away) == FORFEIT_POINTS:
        return 0, FORFEIT_POINTS, True
    raise EsakeParseError(f"unrecognised score {text!r}")


def _game(block: LexborNode, year: int, header_month: int) -> GblGame:
    link = block.css_first('a[href*="idgame="]')
    idgame = ID_GAME.search(link.attributes.get("href") or "") if link is not None else None
    clock = block.css_first('img[src*="clock"]')
    when = GAME_DATE.search(_text(clock.parent) if clock is not None else "")
    score_box = block.css_first(".esake-program-game-final-score")
    if idgame is None or when is None or score_box is None:
        raise EsakeParseError(f"incomplete game block: {_text(block)[:120]!r}")
    month = month_number(when.group(2))
    if month != header_month:
        raise EsakeParseError(f"game month {month} under header month {header_month}")
    hour, minute = (int(when.group(3)), int(when.group(4))) if when.group(3) else (0, 0)
    teams = TEAM_LOGO.findall(score_box.html or "")
    names = [_text(span) for span in score_box.css("span")]
    if len(teams) != 2 or len(names) != 3:
        raise EsakeParseError(f"expected 2 teams and a score, got {teams} {names}")
    home_score, away_score, forfeit = _scores(names[1])
    return GblGame(
        idgame=idgame.group(1),
        round_label=_text(block.css_first("h5")),
        tipoff_local=datetime(year, month, int(when.group(1)), hour, minute),
        confirmed_date=when.group(3) is not None,
        home=teams[0],
        away=teams[1],
        home_name=names[0],
        away_name=names[2],
        home_score=home_score,
        away_score=away_score,
        forfeit=forfeit,
    )


def parse_results_page(html: str) -> ResultsPage:
    tree = LexborHTMLParser(html)
    games: list[GblGame] = []
    year = month = 0
    for node in tree.css("div.esake-program-series-title, div.esake-program-game"):
        if "esake-program-series-title" in (node.attributes.get("class") or ""):
            month_name, _, year_text = _text(node).rpartition(" ")
            year, month = int(year_text), month_number(month_name)
        elif year == 0:
            raise EsakeParseError("game block before any month header")
        else:
            games.append(_game(node, year, month))
    codes = tuple(code for _, code in ROUND_OPTION.findall(html))
    return ResultsPage(round_codes=codes, games=tuple(games))


def _made_attempted(text: str) -> tuple[int, int]:
    made, _, attempted = text.partition("-")
    return int(made), int(attempted)


def _number(text: str) -> int:
    return 0 if text in {"", "-"} else int(text)


def _seconds(text: str) -> int:
    if not text or text.startswith("-"):
        return 0
    hours, minutes, seconds = (int(part) for part in text.split(":"))
    return hours * 3600 + minutes * 60 + seconds


def _box_line(row: LexborNode) -> BoxLine:
    cells = [_text(cell) for cell in row.css("td")]
    link = row.css_first('a[href*="idplayer="]')
    player = ID_PLAYER.search(link.attributes.get("href") or "") if link is not None else None
    if player is None or len(cells) != 17:
        raise EsakeParseError(f"unexpected player row {cells}")
    fg2m, fg2a = _made_attempted(cells[2])
    fg3m, fg3a = _made_attempted(cells[3])
    ftm, fta = _made_attempted(cells[4])
    return BoxLine(
        player_id=player.group(1),
        points=_number(cells[1]),
        fg2m=fg2m,
        fg2a=fg2a,
        fg3m=fg3m,
        fg3a=fg3a,
        ftm=ftm,
        fta=fta,
        seconds=_seconds(cells[15]),
    )


def _team_line(cells: list[str]) -> TeamLine:
    """Totals or team row cells: ``P, 2PM-A, 3PM-A, FTM-A, REBS, D.REBS, O.REBS, ... TO``."""
    if len(cells) != 17:
        raise EsakeParseError(f"unexpected team row {cells}")
    attempted = [0 if cell in {"", "-"} else _made_attempted(cell)[1] for cell in cells[2:5]]
    return TeamLine(
        points=_number(cells[1]),
        fg2a=attempted[0],
        fg3a=attempted[1],
        fta=attempted[2],
        oreb=_number(cells[7]),
        dreb=_number(cells[6]),
        tov=_number(cells[14]),
    )


def _single_row(rows: list[LexborNode], label: str) -> list[str]:
    found = [row for row in rows if _text(row).startswith(label)]
    if len(found) != 1:
        raise EsakeParseError(f"box score table without a single {label} row")
    return [_text(cell) for cell in found[0].css("td, th")]


def _team_box(rows: list[LexborNode]) -> TeamBox:
    players = tuple(_box_line(row) for row in rows if row.css_first('a[href*="idplayer="]'))
    totals = _single_row(rows, "ΣΥΝΟΛΟ")
    team_row = _single_row(rows, "ΟΜΑΔΙΚΑ")
    return TeamBox(players, _number(totals[1]), _team_line(totals), _team_line(team_row))


def parse_box_score(html: str) -> tuple[TeamBox, TeamBox] | None:
    """Home and away box scores in page order (home first); None when ESAKE has none.

    Some 2018-19 and 2019-20 game pages carry only the game header and no stat tables.
    """
    tables = [
        rows
        for rows in (table.css("tr") for table in LexborHTMLParser(html).css("table"))
        if any(_text(row).startswith("ΠΑΙΚΤΗΣ") for row in rows)
    ]
    if not tables:
        return None
    if len(tables) != 2:
        raise EsakeParseError(f"expected 2 team box scores, found {len(tables)}")
    return _team_box(tables[0]), _team_box(tables[1])


def parse_overtimes(html: str) -> int:
    """Overtimes in which someone scored, from the score-by-period table (``ΟΤ1``, ``ΟΤ2``)."""
    overtimes = 0
    for row in LexborHTMLParser(html).css("tr"):
        cells = [_text(cell) for cell in row.css("td, th")]
        if len(cells) == 3 and _plain(cells[1]).startswith(OVERTIME_LABELS):
            overtimes += any(_number(cells[i]) for i in (0, 2))
    return overtimes
