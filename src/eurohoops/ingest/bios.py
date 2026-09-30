"""Player bios (birth date, country) for entity resolution (weeks 12-14 I-c), cached like every
raw page. Local backfill: the daily workflow never runs it.

Cache layout:
  data/raw/gbl/player/{idplayer}.html.gz           ESAKE player page; written once, never refetched
  data/raw/euroleague/people/E{season}.json.gz     every person of a EuroLeague season (one call)

Birth dates are a matching feature only: they stay in ``data/`` and never reach a committed file,
a report or the site.
"""

import json
import logging
import re
from collections.abc import Iterable
from datetime import date
from pathlib import Path

import pandas as pd
from selectolax.lexbor import LexborHTMLParser

from eurohoops.ingest.cache import read_cached, write_atomic
from eurohoops.ingest.http import Fetcher

ESAKE_PLAYER_URL = "https://www.esake.gr/el/action/EsakeplayerView?idplayer={idplayer}&mode=1"
EL_PEOPLE_URL = (
    "https://api-live.euroleague.net/v2/competitions/E/seasons/E{season}/people?limit=2000"
)
EL_PLAYER_TYPE = "J"  # people rows also hold coaches, staff and referees
ESAKE_ID = re.compile(r"^[0-9A-F]{8}$")
BIO_COLUMNS = ["competition", "source_id", "birth_date", "country"]

log = logging.getLogger(__name__)


def gbl_bio_path(gbl_raw: Path, idplayer: str) -> Path:
    return gbl_raw / "player" / f"{idplayer}.html.gz"


def el_people_path(el_raw: Path, season: int) -> Path:
    return el_raw / "people" / f"E{season}.json.gz"


def ingest_bios(
    gbl_fetcher: Fetcher,
    el_fetcher: Fetcher,
    *,
    gbl_raw: Path,
    el_raw: Path,
    gbl_ids: Iterable[str],
    el_seasons: Iterable[int],
) -> tuple[int, int]:
    """Cache the missing pages; return (ESAKE pages fetched, EuroLeague seasons fetched).

    ESAKE player pages are slow (~30 s each on 2026-09-30), so they are fetched in the order of
    ``gbl_ids``: a stopped run leaves the first ids cached, and a rerun picks up the rest.
    """
    el_fetched = 0
    for season in el_seasons:
        path = el_people_path(el_raw, season)
        if not path.exists():
            write_atomic(path, el_fetcher.get(EL_PEOPLE_URL.format(season=season)))
            el_fetched += 1
    gbl_fetched = 0
    for idplayer in dict.fromkeys(gbl_ids):
        if not ESAKE_ID.match(idplayer):  # `pbp:` keys have no ESAKE page
            continue
        path = gbl_bio_path(gbl_raw, idplayer)
        if not path.exists():
            write_atomic(path, gbl_fetcher.get(ESAKE_PLAYER_URL.format(idplayer=idplayer)))
            gbl_fetched += 1
            if gbl_fetched % 50 == 0:
                log.info("esake player pages: %d fetched", gbl_fetched)
    return gbl_fetched, el_fetched


def _day_month_year(text: str) -> date | None:
    match = re.fullmatch(r"(\d{2})-(\d{2})-(\d{4})", text.strip())
    if match is None:
        return None
    day, month, year = (int(g) for g in match.groups())
    return date(year, month, day)


def parse_esake_bio(html: str) -> tuple[date | None, str | None]:
    """Birth date and country from the player page's info table (``ΗΜ. ΓΕΝΝΗΣΗΣ``, ``ΧΩΡΑ``)."""
    fields: dict[str, str] = {}
    for row in LexborHTMLParser(html).css("tr"):
        cells = [c.text(strip=True) for c in row.css("td")]
        if len(cells) == 2 and cells[0] and cells[0] not in fields:
            fields[cells[0]] = cells[1]
    born = _day_month_year(fields.get("ΗΜ. ΓΕΝΝΗΣΗΣ", ""))
    return born, fields.get("ΧΩΡΑ") or None


def parse_el_people(payload: bytes) -> list[tuple[str, date | None, str | None]]:
    """(box-score player id, birth date, country) of each player row of a season's people."""
    rows = json.loads(payload)["data"]
    out = []
    for row in rows:
        if row.get("type") != EL_PLAYER_TYPE:
            continue
        person = row["person"]
        born = person.get("birthDate")
        country = (person.get("country") or {}).get("code")
        out.append(
            (
                "P" + str(person["code"]).strip(),
                date.fromisoformat(born[:10]) if born else None,
                country,
            )
        )
    return out


def build_bios(gbl_raw: Path, el_raw: Path) -> pd.DataFrame:
    """One row per cached source id (``BIO_COLUMNS``); the first non-empty birth date wins."""
    rows: list[tuple[str, str, date | None, str | None]] = []
    for path in sorted((gbl_raw / "player").glob("*.html.gz")):
        born, country = parse_esake_bio(read_cached(path).decode("utf-8", errors="replace"))
        rows.append(("gbl", path.name.split(".")[0], born, country))
    for path in sorted((el_raw / "people").glob("E*.json.gz")):
        rows.extend(
            ("euroleague", pid, born, country)
            for pid, born, country in parse_el_people(read_cached(path))
        )
    table = pd.DataFrame(rows, columns=BIO_COLUMNS)
    table["has_date"] = table["birth_date"].notna()
    table = table.sort_values(
        ["competition", "source_id", "has_date"], ascending=[True, True, False]
    )
    return (
        table.drop_duplicates(["competition", "source_id"])
        .drop(columns="has_date")
        .reset_index(drop=True)
    )
