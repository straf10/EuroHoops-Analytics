"""Forward odds recorder (EuroLeague, The Odds API): a market benchmark, never a betting product.

One API call per run. The raw response goes to the gitignored cache; only a derived consensus
row per upcoming game is appended to the public, append-only odds log: the median de-vigged
P(home), the median home spread and total, and how many bookmakers each came from.

The API key is sent as a query parameter, so no URL is ever logged or put in an error message.
Every call's quota headers are appended to a calls log, and no call is made while the last
known remaining quota is below ``MIN_REMAINING``.
"""

import csv
import json
import logging
import statistics
from collections.abc import Callable, Hashable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from eurohoops.ingest.cache import write_atomic

API_URL = "https://api.the-odds-api.com/v4/sports/basketball_euroleague/odds"
QUERY = {
    "regions": "eu",
    "markets": "h2h,spreads,totals",
    "oddsFormat": "decimal",
    "dateFormat": "iso",
}
MIN_REMAINING = 20
MATCH_WINDOW = pd.Timedelta(hours=48)  # API start time vs the schedule's tip-off
TIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

ODDS_COLUMNS = (
    "game_id",
    "season",
    "round",
    "tipoff_utc",
    "home",
    "away",
    "commence_utc",
    "fetched_at_utc",
    "bookmakers",
    "p_home",
    "spread_books",
    "spread_home",
    "total_books",
    "total",
)
CALL_COLUMNS = (
    "called_at_utc",
    "status",
    "cost",
    "requests_used",
    "requests_remaining",
    "events",
    "rows_written",
)

log = logging.getLogger(__name__)


class OddsApiError(RuntimeError):
    """The call failed or was refused; the message never contains the URL or the key."""


@dataclass(frozen=True)
class OddsPaths:
    log: Path  # public consensus rows (append-only)
    calls: Path  # public quota log, one row per API call
    teams: Path  # explicit API team name -> team code map
    raw_dir: Path  # gitignored raw responses


def api_key(env: Mapping[str, str], dotenv: Path) -> str | None:
    """``ODDS_API_KEY`` from the environment, else from a local ``.env`` file."""
    if env.get("ODDS_API_KEY"):
        return env["ODDS_API_KEY"]
    if not dotenv.exists():
        return None
    for line in dotenv.read_text(encoding="utf-8").splitlines():
        name, _, value = line.partition("=")
        if name.strip() == "ODDS_API_KEY" and value.strip():
            return value.strip().strip("\"'")
    return None


def read_team_map(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as fh:
        return {row["odds_api_name"]: row["team"] for row in csv.DictReader(fh)}


def last_remaining(calls: Path) -> int | None:
    """Remaining quota reported by the most recent call that returned the header."""
    if not calls.exists():
        return None
    with calls.open(newline="", encoding="utf-8") as fh:
        known = [row["requests_remaining"] for row in csv.DictReader(fh)]
    known = [value for value in known if value]
    return int(float(known[-1])) if known else None


def _append(path: Path, columns: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    is_new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, lineterminator="\n")
        if is_new:
            writer.writeheader()
        writer.writerows(rows)


def devig_two_way(home_price: float, away_price: float) -> float:
    """P(home) from decimal odds with the margin removed proportionally."""
    home, away = 1.0 / home_price, 1.0 / away_price
    return home / (home + away)


def _market(bookmaker: dict[str, Any], key: str) -> list[dict[str, Any]]:
    for market in bookmaker.get("markets", []):
        if market.get("key") == key:
            outcomes: list[dict[str, Any]] = market.get("outcomes", [])
            return outcomes
    return []


def consensus(event: dict[str, Any]) -> dict[str, Any]:
    """Median de-vigged P(home), home spread and total over the event's bookmakers.

    A bookmaker's moneyline counts only as an exact two-way home/away market (no draw).
    """
    home, away = event["home_team"], event["away_team"]
    p_home, spreads, totals = [], [], []
    for bookmaker in event.get("bookmakers", []):
        prices = {o["name"]: o["price"] for o in _market(bookmaker, "h2h")}
        if set(prices) == {home, away}:
            p_home.append(devig_two_way(prices[home], prices[away]))
        spreads += [o["point"] for o in _market(bookmaker, "spreads") if o["name"] == home]
        totals += [o["point"] for o in _market(bookmaker, "totals") if o["name"] == "Over"]
    return {
        "bookmakers": len(p_home),
        "p_home": f"{statistics.median(p_home):.4f}" if p_home else "",
        "spread_books": len(spreads),
        "spread_home": f"{statistics.median(spreads):g}" if spreads else "",
        "total_books": len(totals),
        "total": f"{statistics.median(totals):g}" if totals else "",
    }


def _match(
    games: pd.DataFrame, home: str, away: str, commence: pd.Timestamp
) -> dict[Hashable, Any] | None:
    """The unplayed scheduled game with these teams whose tip-off is nearest ``commence``."""
    candidates = games[(games["home"] == home) & (games["away"] == away) & ~games["played"]]
    candidates = candidates.assign(gap=(candidates["tipoff_utc"] - commence).abs())
    close = candidates[candidates["gap"] <= MATCH_WINDOW].sort_values("gap")
    return None if close.empty else close.to_dict("records")[0]


def consensus_rows(
    events: list[dict[str, Any]],
    team_map: Mapping[str, str],
    games: pd.DataFrame,
    fetched_at: datetime,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Rows for events that have not started, plus a reason for each event left unmatched."""
    rows, unmatched = [], []
    for event in events:
        commence = pd.Timestamp(event["commence_time"])
        label = f"{event['home_team']} v {event['away_team']} ({event['commence_time']})"
        if commence <= fetched_at:
            continue  # in play or finished: not a pre-game price
        home, away = team_map.get(event["home_team"]), team_map.get(event["away_team"])
        if home is None or away is None:
            unmatched.append(f"{label}: team name not in the map")
            continue
        game = _match(games, home, away, commence)
        if game is None:
            unmatched.append(f"{label}: no scheduled {home}-{away} game within 48 h")
            continue
        rows.append(
            {
                "game_id": game["game_id"],
                "season": int(game["season"]),
                "round": int(game["round"]),
                "tipoff_utc": game["tipoff_utc"].strftime(TIME_FORMAT),
                "home": home,
                "away": away,
                "commence_utc": commence.strftime(TIME_FORMAT),
                "fetched_at_utc": fetched_at.strftime(TIME_FORMAT),
                **consensus(event),
            }
        )
    return rows, unmatched


@contextmanager
def _quiet_http_logs() -> Iterator[None]:
    """httpx logs each request URL at INFO, and this URL carries the key: silence it here."""
    loggers = [logging.getLogger(name) for name in ("httpx", "httpcore")]
    levels = [logger.level for logger in loggers]
    for logger in loggers:
        logger.setLevel(logging.WARNING)
    try:
        yield
    finally:
        for logger, level in zip(loggers, levels, strict=True):
            logger.setLevel(level)


def _error_code(response: httpx.Response) -> str:
    """The API's machine-readable error code, if the body carries one."""
    try:
        body = response.json()
    except ValueError:
        return ""
    return str(body.get("error_code", "")) if isinstance(body, dict) else ""


def _header(response: httpx.Response, name: str) -> str:
    return str(response.headers.get(name, ""))


def record_odds(
    client: httpx.Client,
    key: str,
    games: pd.DataFrame,
    paths: OddsPaths,
    clock: Callable[[], datetime],
) -> dict[str, Any]:
    """Make one API call, cache it, append consensus rows; return a summary (no secrets)."""
    remaining = last_remaining(paths.calls)
    if remaining is not None and remaining < MIN_REMAINING:
        raise OddsApiError(f"refusing to call: {remaining} requests remaining (< {MIN_REMAINING})")
    fetched_at = clock()
    try:
        with _quiet_http_logs():
            response = client.get(API_URL, params={"apiKey": key, **QUERY})
    except httpx.HTTPError as exc:
        raise OddsApiError(f"transport error: {type(exc).__name__}") from None
    call = {
        "called_at_utc": fetched_at.strftime(TIME_FORMAT),
        "status": response.status_code,
        "cost": _header(response, "x-requests-last"),
        "requests_used": _header(response, "x-requests-used"),
        "requests_remaining": _header(response, "x-requests-remaining"),
        "events": "",
        "rows_written": 0,
    }
    log.info(
        "odds call: status %s, cost %s, used %s, remaining %s",
        call["status"],
        call["cost"] or "?",
        call["requests_used"] or "?",
        call["requests_remaining"] or "?",
    )
    if response.status_code != httpx.codes.OK:
        _append(paths.calls, CALL_COLUMNS, [call])
        raise OddsApiError(f"HTTP {response.status_code} {_error_code(response)}".strip())
    write_atomic(paths.raw_dir / f"{fetched_at:%Y%m%dT%H%M%SZ}.json.gz", response.content)
    events: list[dict[str, Any]] = json.loads(response.content)
    rows, unmatched = consensus_rows(events, read_team_map(paths.teams), games, fetched_at)
    _append(paths.log, ODDS_COLUMNS, rows)
    _append(paths.calls, CALL_COLUMNS, [{**call, "events": len(events), "rows_written": len(rows)}])
    for reason in unmatched:
        log.warning("unmatched odds event: %s", reason)
    return {**call, "events": len(events), "rows_written": len(rows), "unmatched": unmatched}
