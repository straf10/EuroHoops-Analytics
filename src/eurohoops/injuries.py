"""Forward EuroLeague injury recorder (BasketNews injury report): a public, append-only log.

One HTTP request per run. The raw HTML goes to the gitignored cache; the parsed table is
appended to ``injuries/euroleague_<season>.csv`` as one snapshot (a ``snapshot_utc`` stamp on
every row). A snapshot identical to the latest logged one (same page update time, same rows) is
not appended again. Any failure to fetch or parse is a warning and writes nothing: the daily
job never fails because of this step.
"""

import csv
import logging
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
from selectolax.lexbor import LexborHTMLParser, LexborNode

from eurohoops.ingest.cache import write_atomic
from eurohoops.logs import TIME_FORMAT, append_rows, require_terminated

REPORT_URL = "https://basketnews.com/news-212393-euroleague-injury-report-updated.html"

INJURY_COLUMNS = (
    "snapshot_utc",
    "page_updated_utc",
    "team",
    "team_name_bn",
    "player",
    "bn_player_id",
    "position",
    "status",
    "return_round",
    "comment",
)
# Everything but the snapshot stamp: what "the same snapshot" means.
CONTENT_COLUMNS = INJURY_COLUMNS[1:]

# BasketNews names that differ from the teams mart (sponsor changes); only codes that play in
# the season are ever matched, so a stale alias is harmless.
NAME_ALIASES = {"Fenerbahce Beko Istanbul": "ULK"}

_PLAYER_ID = re.compile(r"/players/(\d+)-")
_SPACES = re.compile(r"\s+")

log = logging.getLogger(__name__)


class InjuryReportError(RuntimeError):
    """The page could not be fetched or did not contain an injury table."""


@dataclass(frozen=True)
class InjuryPaths:
    log: Path  # public snapshots (append-only)
    raw_dir: Path  # gitignored raw HTML


def _text(node: LexborNode | None) -> str:
    return "" if node is None else _SPACES.sub(" ", node.text(separator=" ")).strip()


def page_updated(tree: LexborHTMLParser) -> str:
    """The article's last-update stamp (UTC), from the date element beside the title; else ''."""
    title = tree.css_first("h1.post-title")
    holder = None if title is None else title.parent
    stamp = None if holder is None else holder.css_first("div.date.times[data-time]")
    raw = "" if stamp is None else (stamp.attributes.get("data-time") or "")
    if not raw.isdigit():
        return ""
    return datetime.fromtimestamp(int(raw), tz=UTC).strftime(TIME_FORMAT)


def parse_report(html: bytes | str) -> tuple[str, list[dict[str, str]]]:
    """``(page_updated_utc, rows)`` with ``team`` left empty (mapped later).

    Raises ``InjuryReportError`` if the injury table is missing.
    """
    tree = LexborHTMLParser(html)
    table = tree.css_first("table.table_02")
    if table is None or table.css_first("tr.injury_reports__head") is None:
        raise InjuryReportError("injury table not found")
    updated = page_updated(tree)
    rows: list[dict[str, str]] = []
    team_name = ""
    for tr in table.css("tr"):
        classes = tr.attributes.get("class") or ""
        if "injury_reports__team-row" in classes:
            team_name = _text(tr.css_first("a"))
            continue
        cells = tr.css("td")
        link = tr.css_first("a[href*='/players/']")
        if "injury_reports__head" in classes or len(cells) < 5 or link is None:
            continue
        match = _PLAYER_ID.search(link.attributes.get("href") or "")
        rows.append(
            {
                "page_updated_utc": updated,
                "team": "",
                "team_name_bn": team_name,
                "player": _text(link),
                "bn_player_id": match.group(1) if match else "",
                "position": _text(cells[0]),
                "status": _text(cells[2].css_first("span.player-status") or cells[2]),
                "return_round": _text(cells[3]),
                "comment": _text(cells[4]),
            }
        )
    return updated, rows


def team_name_map(teams: pd.DataFrame, games: pd.DataFrame, season: int) -> dict[str, str]:
    """BasketNews team name -> source code, for the teams that play in ``season``."""
    in_season = games[games["season"] == season]
    playing = set(in_season["home"]) | set(in_season["away"])
    mapping = {
        str(t["name"]): str(t["team"]) for t in teams.to_dict("records") if t["team"] in playing
    }
    mapping.update({name: code for name, code in NAME_ALIASES.items() if code in playing})
    return mapping


def assign_teams(rows: list[dict[str, str]], names: Mapping[str, str]) -> list[dict[str, str]]:
    """Fill ``team`` from the name map; an unmapped name is a warning and stays empty."""
    for name in sorted({r["team_name_bn"] for r in rows if r["team_name_bn"] not in names}):
        log.warning("injuries: team %r is not in this season's teams; team left empty", name)
    return [{**r, "team": names.get(r["team_name_bn"], "")} for r in rows]


def _content(row: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(row[c]) for c in CONTENT_COLUMNS)


def latest_snapshot(path: Path) -> list[tuple[str, ...]]:
    """Content of the rows of the most recently logged snapshot, sorted (empty if no log)."""
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return []
    stamp = rows[-1]["snapshot_utc"]
    return sorted(_content(r) for r in rows if r["snapshot_utc"] == stamp)


def fetch_report(client: httpx.Client) -> bytes:
    try:
        response = client.get(REPORT_URL)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise InjuryReportError(f"fetch failed: {type(exc).__name__}") from None
    return response.content


def record_injuries(
    client: httpx.Client,
    names: Mapping[str, str],
    paths: InjuryPaths,
    clock: Callable[[], datetime],
) -> dict[str, Any]:
    """One request: cache the HTML, append a snapshot unless it repeats the latest one.

    A bad fetch or page never raises: it returns ``{"status": "failed", ...}`` and writes
    neither the raw cache nor the log. A log with a torn last row still raises ``ValueError``.
    """
    require_terminated(paths.log)
    taken = clock()
    try:
        html = fetch_report(client)
        updated, parsed = parse_report(html)
        if not parsed:
            raise InjuryReportError("injury table has no player rows")
    except InjuryReportError as exc:
        log.warning("injuries: %s; nothing recorded", exc)
        return {"status": "failed", "rows": 0, "teams": 0, "unmapped": []}
    write_atomic(paths.raw_dir / f"{taken:%Y%m%dT%H%M%SZ}.html.gz", html)
    rows = assign_teams(parsed, names)
    summary = {
        "rows": len(rows),
        "teams": len({r["team_name_bn"] for r in rows}),
        "page_updated_utc": updated,
        "unmapped": sorted({r["team_name_bn"] for r in rows if not r["team"]}),
    }
    if sorted(_content(r) for r in rows) == latest_snapshot(paths.log):
        log.info("injuries: snapshot unchanged since the last logged one; nothing appended")
        return {**summary, "status": "unchanged"}
    stamp = taken.strftime(TIME_FORMAT)
    append_rows(paths.log, INJURY_COLUMNS, [{"snapshot_utc": stamp, **r} for r in rows])
    return {**summary, "status": "appended"}
