"""Static predictions page (``site/index.html``) built from the logs, scorecards and results."""

from collections.abc import Hashable
from dataclasses import dataclass
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd

ATHENS = "Europe/Athens"
RECENT_RESULTS = 10

STYLE = (
    ":root{--bg:#fff;--fg:#1b1f24;--muted:#5b6470;--line:#e3e6ea;--accent:#0b5cad;"
    "--good:#1a7f37;--bad:#b42318}"
    "@media (prefers-color-scheme:dark){:root{--bg:#111418;--fg:#e8eaed;--muted:#9aa3ad;"
    "--line:#2a3038;--accent:#6aa9ff;--good:#4ac26b;--bad:#ff7b72}}"
    "*{box-sizing:border-box}"
    "body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}"
    "main{max-width:860px;margin:0 auto;padding:16px}h1{font-size:1.4rem;margin:.2em 0}"
    "h2{margin-top:1.6em}h3{font-size:1rem;color:var(--muted);margin:1.2em 0 .4em}"
    "p.note{color:var(--muted);font-size:.9rem}.scroll{overflow-x:auto}"
    "table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}"
    "th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}"
    "th{color:var(--muted);font-weight:600;font-size:.85rem}"
    "td.num,th.num{text-align:right;white-space:nowrap}.away{color:var(--muted)}"
    ".good{color:var(--good)}.bad{color:var(--bad)}a{color:var(--accent)}"
)


@dataclass(frozen=True)
class Section:
    title: str
    log_path: Path
    scorecard: dict[str, Any]
    games: pd.DataFrame
    names: dict[str, str]


def _athens(tipoff: pd.Timestamp) -> str:
    """Day on the first line, time on the second, so the column stays narrow on phones."""
    local = tipoff.tz_convert(ATHENS)
    return f"{local:%a %d %b}<br>{local:%H:%M}"


def _table(headers: list[str], rows: list[list[str]], numeric: set[int]) -> str:
    def cell(tag: str, i: int, value: str) -> str:
        cls = ' class="num"' if i in numeric else ""
        return f"<{tag}{cls}>{value}</{tag}>"

    head = "".join(cell("th", i, escape(h)) for i, h in enumerate(headers))
    body = "".join(
        "<tr>" + "".join(cell("td", i, v) for i, v in enumerate(r)) + "</tr>" for r in rows
    )
    return f'<div class="scroll"><table><tr>{head}</tr>{body}</table></div>'


def _logged(section: Section) -> pd.DataFrame:
    """Earliest (pre-registered) row per game, joined with the game's current state."""
    if not section.log_path.exists():
        return pd.DataFrame()
    log = pd.read_csv(section.log_path, dtype={"game_id": str})
    first = log.sort_values("predicted_at_utc").drop_duplicates("game_id", keep="first")
    cols = ["game_id", "tipoff_utc", "played", "forfeit", "home_score", "away_score"]
    return first.drop(columns="tipoff_utc").merge(section.games[cols], on="game_id")


def _section_html(section: Section, now: datetime) -> str:
    logged = _logged(section)
    records = logged.sort_values("tipoff_utc").to_dict("records") if len(logged) else []

    def teams(g: dict[Hashable, Any]) -> list[str]:
        home = escape(section.names.get(g["home"], g["home"]))
        away = escape(section.names.get(g["away"], g["away"]))
        return [_athens(g["tipoff_utc"]), f'{home}<br><span class="away">vs {away}</span>']

    parts = [f"<h2>{escape(section.title)}</h2>", "<h3>Upcoming</h3>"]
    upcoming = [
        [*teams(g), f"{g['p_home']:.0%}", f"{g['exp_margin']:+.1f}"]
        for g in records
        if not g["played"] and g["tipoff_utc"] > now
    ]
    if upcoming:
        headers = ["Athens time", "Home / away", "P(home)", "Margin"]
        parts.append(_table(headers, upcoming, {2, 3}))
    else:
        parts.append('<p class="note">No logged games in the next window.</p>')
    parts.append("<h3>Recent results</h3>")
    finished = [g for g in records if g["played"] and not g["forfeit"]][-RECENT_RESULTS:]
    card = section.scorecard
    hidden = set(card.get("games_not_provable", []))
    results = []
    for g in reversed(finished):
        hit = (g["p_home"] >= 0.5) == (g["home_score"] > g["away_score"])
        mark = f'<span class="{"good" if hit else "bad"}">{"hit" if hit else "miss"}</span>'
        if g["game_id"] in hidden:
            mark += "*"
        score = f"{g['home_score']}-{g['away_score']}"
        results.append([*teams(g), score, f"{g['p_home']:.0%}", mark])
    if results:
        headers = ["Athens time", "Home / away", "Score", "P(home)", "Pick"]
        parts.append(_table(headers, results, {2, 3}))
    else:
        parts.append('<p class="note">No logged game has finished yet.</p>')
    elo, b0 = card["elo"], card["b0"]

    def metric(m: dict[str, Any], key: str, fmt: str) -> str:
        return "n/a" if m[key] is None else format(m[key], fmt)

    rows = [
        [label, metric(elo, key, fmt), metric(b0, key, fmt)]
        for label, key, fmt in (
            ("Log loss (lower = better)", "log_loss", ".4f"),
            ("Brier score", "brier", ".4f"),
            ("Accuracy", "accuracy", ".1%"),
            ("Margin MAE (points)", "margin_mae", ".2f"),
        )
    ]
    parts.append(f"<h3>Scorecard: {elo['n']} finished games</h3>")
    parts.append(_table(["Metric", "Elo", "Baseline"], rows, {1, 2}))
    if hidden:
        parts.append(
            f'<p class="note">* Not scored: {len(hidden)} game{"s" * (len(hidden) > 1)} whose '
            "prediction was stamped before tip-off but reached the public log only after it.</p>"
        )
    return "".join(parts)


def render_page(sections: list[Section], now: datetime) -> str:
    body = "".join(_section_html(section, now) for section in sections)
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>EuroHoops predictions</title><style>{STYLE}</style></head><body><main>"
        "<h1>EuroHoops Analytics: live predictions</h1>"
        '<p class="note">Every forecast is committed to a public, append-only log before tip-off '
        "and scored afterwards against a home-win baseline. This is a model benchmark, not "
        'betting advice. <a href="https://github.com/straf10/EuroHoops-Analytics">Code and '
        "logs</a>.</p>"
        f"{body}"
        f'<p class="note">Generated {escape(now.strftime("%Y-%m-%d %H:%M UTC"))}.</p>'
        "</main></body></html>\n"
    )
