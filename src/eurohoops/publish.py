"""Site data (``web/src/data/site.json``) built from the logs, scorecards, reports and results.

The Astro front-end in ``web/`` renders this file into ``site/`` at build time; this module owns
every number the page shows, so the page never recomputes or reinterprets the pipeline.
"""

from collections.abc import Hashable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from eurohoops.eval.backtest import TunedModel
from eurohoops.models.elo import prepare, season_ratings

RECENT_RESULTS = 12
TIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


@dataclass(frozen=True)
class Section:
    key: str
    title: str
    log: pd.DataFrame  # the public prediction log, as read from its CSV
    scorecard: dict[str, Any]
    backtest: dict[str, Any]  # the live backtest report the log's parameters come from
    games: pd.DataFrame
    names: dict[str, str]
    model: TunedModel
    season: int
    replay_from: int


def _stamp(ts: pd.Timestamp) -> str:
    return ts.tz_convert("UTC").strftime(TIME_FORMAT)


def _logged(section: Section) -> list[dict[Hashable, Any]]:
    """Earliest (pre-registered) row per game, joined with the game's current state."""
    if section.log.empty:
        return []
    first = section.log.sort_values("predicted_at_utc").drop_duplicates("game_id", keep="first")
    cols = ["game_id", "tipoff_utc", "played", "forfeit", "home_score", "away_score"]
    joined = first.drop(columns="tipoff_utc").merge(section.games[cols], on="game_id")
    return joined.sort_values("tipoff_utc").to_dict("records")


def _team(section: Section, code: str) -> dict[str, str]:
    return {"code": code, "name": section.names.get(code, code)}


def _game(section: Section, g: dict[Hashable, Any]) -> dict[str, Any]:
    return {
        "game_id": g["game_id"],
        "round": int(g["round"]),
        "tipoff_utc": _stamp(g["tipoff_utc"]),
        "predicted_at_utc": g["predicted_at_utc"],
        "home": _team(section, g["home"]),
        "away": _team(section, g["away"]),
        "p_home": float(g["p_home"]),
        "exp_margin": float(g["exp_margin"]),
    }


def _next_tipoff(section: Section, now: datetime) -> str | None:
    games = section.games
    ahead = games[(games["season"] == section.season) & ~games["played"]]
    ahead = ahead[ahead["tipoff_utc"] > now]
    return None if ahead.empty else _stamp(ahead["tipoff_utc"].min())


def _ratings(section: Section) -> list[dict[str, Any]]:
    games = section.games[section.games["season"] >= section.replay_from]
    table = season_ratings(prepare(games), section.model.params, section.season)
    rows = [
        {**_team(section, code), "rating": round(now, 1), "change": round(now - start, 1)}
        for code, (now, start) in table.items()
    ]
    return sorted(rows, key=lambda r: -r["rating"])


def _backtest(report: dict[str, Any]) -> dict[str, Any]:
    tuned = report["tuned"]
    return {
        "tuning_seasons": report["seasons"]["tuning"],
        "test_seasons": report["seasons"]["test"],
        "params": {k: tuned[k] for k in ("k", "hca", "reversion")},
        "test": report["metrics"]["test"],
        "log_loss_diff": report.get("test_log_loss_diff_elo_minus_b0"),
    }


def section_data(section: Section, now: datetime) -> dict[str, Any]:
    logged = _logged(section)
    hidden = set(section.scorecard.get("games_not_provable", []))
    upcoming = [_game(section, g) for g in logged if not g["played"] and g["tipoff_utc"] > now]
    finished = [g for g in logged if g["played"] and not g["forfeit"]][-RECENT_RESULTS:]
    results = [
        {
            **_game(section, g),
            "home_score": int(g["home_score"]),
            "away_score": int(g["away_score"]),
            "hit": (g["p_home"] >= 0.5) == (g["home_score"] > g["away_score"]),
            "provable": g["game_id"] not in hidden,
        }
        for g in reversed(finished)
    ]
    card = section.scorecard
    return {
        "key": section.key,
        "title": section.title,
        "season": f"{section.season}-{(section.season + 1) % 100:02d}",
        "logged": len(logged),
        "next_tipoff_utc": _next_tipoff(section, now),
        "upcoming": upcoming,
        "results": results,
        "scorecard": {"elo": card["elo"], "b0": card["b0"], "not_provable": len(hidden)},
        "backtest": _backtest(section.backtest),
        "ratings": _ratings(section),
    }


def site_data(sections: list[Section], now: datetime) -> dict[str, Any]:
    return {
        "generated_at_utc": now.strftime(TIME_FORMAT),
        "competitions": [section_data(s, now) for s in sections],
    }
