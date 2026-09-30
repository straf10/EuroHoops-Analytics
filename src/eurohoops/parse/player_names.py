"""Player names per competition, season and team (weeks 12-14 I1), the matcher's input.

GBL: the cached ESAKE box pages give each player's name and jersey in the box-table cell,
``<a …idplayer=ID…>#<jersey><div…photo…></div><span>SURNAME</span> FIRST</a>``; ``##`` and
``#0`` mean no jersey (§3 finding 1 and D6 of reports/week12-14_progress.md). Team and seconds
come from the staged ``player_box`` rows of the same game. Names keep ESAKE's letters as
printed, Latin look-alikes
inside Greek words included (``entity.translit.fold`` normalises them). The 2018-20 PBP fill
keys its lines ``pbp:<jersey>:<First Last>``: their names come from the key.

EuroLeague: ``stats/box.py``'s player lines (``player`` = ``SURNAME, NAME``, ``dorsal``).

A row is one (competition, source_id, season, team): ``games`` counts lines with time on court,
``jersey`` is the season's most frequent jersey with that team (ties: the smallest).
"""

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pandera.pandas as pa
from selectolax.lexbor import LexborHTMLParser

from eurohoops.ingest.cache import read_cached
from eurohoops.parse.esake import ID_PLAYER
from eurohoops.parse.schemas import validated

COMPETITIONS = ("euroleague", "gbl")
PBP_KEY = re.compile(r"^pbp:(?P<jersey>[^:]*):(?P<name>.+)$")

PLAYER_NAMES_SCHEMA = pa.DataFrameSchema(
    {
        "competition": pa.Column(str, pa.Check.isin(COMPETITIONS)),
        "source_id": pa.Column(str),
        "season": pa.Column("int64"),
        "team": pa.Column(str),
        "jersey": pa.Column(str, nullable=True),
        "surname_raw": pa.Column(str),
        "first_raw": pa.Column(str),
        "games": pa.Column("int64", pa.Check.ge(0)),
        "minutes": pa.Column("float64", pa.Check.ge(0)),
    },
    unique=["competition", "source_id", "season", "team"],
    strict=True,
)

PBP_LINK_SCHEMA = pa.DataFrameSchema(
    {
        "competition": pa.Column(str, pa.Check.isin(COMPETITIONS)),
        "source_id": pa.Column(str),
        "linked_source_id": pa.Column(str, nullable=True),
        "method": pa.Column(str),
        "reason": pa.Column(str),
    },
    unique=["source_id"],
    strict=True,
)


@dataclass(frozen=True)
class NameCell:
    player_id: str
    jersey: str | None
    surname: str
    first: str


def _jersey(text: str) -> str | None:
    """ESAKE jersey text (``#17``, ``##``, ``#0``) -> the number; ``0`` is ESAKE's placeholder
    (313 GBL rows 2018-2025, up to 12 players of one team-season), so it counts as unknown."""
    value = text.strip().removeprefix("#").strip()
    return value if value and value not in {"#", "0"} else None


def parse_name_cells(html: str) -> list[NameCell]:
    """Every box-table player cell of an ESAKE game page, in page order."""
    cells = []
    for td in LexborHTMLParser(html).css("td.table-left"):
        link = td.css_first('a[href*="idplayer="]')
        span = link.css_first("span") if link is not None else None
        player = ID_PLAYER.search(link.attributes.get("href") or "") if link is not None else None
        if link is None or span is None or player is None:
            continue
        inner = link.html or ""
        head = inner.split(">", 1)[1].split("<", 1)[0]
        first = inner.rsplit("</span>", 1)[1].split("<", 1)[0]
        cells.append(
            NameCell(player.group(1), _jersey(head), span.text(strip=True), " ".join(first.split()))
        )
    return cells


def _mode(values: pd.Series) -> str | None:
    counts = values.dropna().value_counts()
    if counts.empty:
        return None
    top = counts[counts == counts.max()].index
    return str(min(str(v) for v in top))


def _aggregate(lines: pd.DataFrame, competition: str) -> pd.DataFrame:
    """Lines (source_id season team jersey surname_raw first_raw seconds) -> name rows."""
    keys = ["source_id", "season", "team"]
    grouped = lines.groupby(keys, sort=True)
    spelling = lines.sort_values([*keys, "surname_raw", "first_raw"]).drop_duplicates(keys)
    rows = grouped.agg(
        games=("seconds", lambda s: int((s > 0).sum())),
        minutes=("seconds", lambda s: round(float(s.sum()) / 60.0, 2)),
    ).reset_index()
    rows["jersey"] = grouped["jersey"].agg(_mode).to_numpy()
    rows = rows.merge(spelling[[*keys, "surname_raw", "first_raw"]], on=keys, how="left")
    rows.insert(0, "competition", competition)
    return validated(rows[list(PLAYER_NAMES_SCHEMA.columns)], PLAYER_NAMES_SCHEMA)


def _pbp_name(key: str) -> tuple[str | None, str, str]:
    """``pbp:<jersey>:<First Last>`` -> (jersey, surname, first name)."""
    match = PBP_KEY.match(key)
    if match is None:
        raise ValueError(f"not a pbp key: {key!r}")
    first, _, surname = match.group("name").partition(" ")
    if not surname:
        return _jersey(match.group("jersey")), first.upper(), ""
    return _jersey(match.group("jersey")), surname.upper(), first.upper()


def gbl_names(raw_dir: Path, player_box: pd.DataFrame) -> pd.DataFrame:
    """GBL name rows from the cached box pages and the staged ``player_box``
    (``game_id team player_id seconds``; ``game_id`` = ``GBL{season}_{idgame}``)."""
    cells: dict[tuple[str, str], NameCell] = {}
    for game_id in sorted(set(player_box["game_id"])):
        season, idgame = str(game_id).removeprefix("GBL").split("_")
        path = raw_dir / "boxscore" / season / f"{idgame}.html.gz"
        if path.exists():
            for cell in parse_name_cells(read_cached(path).decode("utf-8")):
                cells[(str(game_id), cell.player_id)] = cell
    rows: list[dict[str, Any]] = []
    for game_id, team, player_id, secs in zip(
        player_box["game_id"].astype(str),
        player_box["team"].astype(str),
        player_box["player_id"].astype(str),
        player_box["seconds"].astype(int),
        strict=True,
    ):
        found = cells.get((game_id, player_id))
        if found is not None:
            jersey, surname, first = found.jersey, found.surname, found.first
        elif player_id.startswith("pbp:"):
            jersey, surname, first = _pbp_name(player_id)
        else:
            raise ValueError(f"no name for {player_id} in {game_id}")
        rows.append(
            {
                "source_id": player_id,
                "season": int(game_id.removeprefix("GBL").split("_")[0]),
                "team": team,
                "jersey": jersey,
                "surname_raw": surname,
                "first_raw": first,
                "seconds": int(secs),
            }
        )
    return _aggregate(pd.DataFrame(rows), "gbl")


def euroleague_names(players: pd.DataFrame) -> pd.DataFrame:
    """EuroLeague name rows from ``stats/box.py`` player lines (``player_id player dorsal team
    season sec``); ``player`` is ``SURNAME, NAME`` (no comma: all surname)."""
    split = players["player"].str.partition(", ")
    lines = pd.DataFrame(
        {
            "source_id": players["player_id"].astype(str),
            "season": players["season"].astype("int64"),
            "team": players["team"].astype(str),
            "jersey": players["dorsal"].where(players["dorsal"].astype(str).str.len() > 0),
            "surname_raw": split[0].str.strip(),
            "first_raw": split[2].str.strip(),
            "seconds": players["sec"].astype("int64"),
        }
    )
    return _aggregate(lines, "euroleague")


def pbp_links(names: pd.DataFrame, same_name: Callable[[str, str], bool]) -> pd.DataFrame:
    """Each GBL ``pbp:`` key -> the one ESAKE id of its team-season with the key's jersey whose
    name ``same_name(key name, candidate name)`` accepts (names as ``SURNAME FIRST``); else
    unlinked, with the reason. ``same_name`` is injected so the rule uses the matcher's
    transliteration."""
    gbl = names[names["competition"] == "gbl"]
    official = gbl[~gbl["source_id"].str.startswith("pbp:")]
    rows = []
    for key in gbl[gbl["source_id"].str.startswith("pbp:")].itertuples(index=False):
        pool = official[
            (official["season"] == key.season)
            & (official["team"] == key.team)
            & (official["jersey"] == key.jersey)
        ]
        key_name = f"{key.surname_raw!s} {key.first_raw!s}"
        found = sorted(
            {
                str(c.source_id)
                for c in pool.itertuples(index=False)
                if same_name(key_name, f"{c.surname_raw!s} {c.first_raw!s}")
            }
        )
        linked = found[0] if len(found) == 1 else None
        jersey = str(key.jersey)
        if linked is not None:
            reason = f"jersey {jersey} and name, team-season {key.team!s} {key.season!s}"
        elif pd.isna(key.jersey):
            reason = "no jersey in the key"
        else:
            reason = f"{len(pool)} ids with jersey {jersey}, {len(found)} with the name"
        rows.append(
            {
                "competition": "gbl",
                "source_id": str(key.source_id),
                "linked_source_id": linked,
                "method": "pbp",
                "reason": reason,
            }
        )
    frame = pd.DataFrame(rows, columns=list(PBP_LINK_SCHEMA.columns))
    return validated(frame.drop_duplicates("source_id"), PBP_LINK_SCHEMA)
