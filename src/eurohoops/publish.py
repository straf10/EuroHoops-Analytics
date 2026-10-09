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
from eurohoops.logs import TIME_FORMAT
from eurohoops.models.elo import prepare, season_ratings
from eurohoops.titles import titles_payload

RECENT_RESULTS = 12

# Display-only team codes. The pipeline (marts, prediction logs, odds) keeps the source's codes;
# only the site shows these real-life abbreviations in their place.
DISPLAY_CODES: dict[str, dict[str, str]] = {
    "euroleague": {
        "BAS": "KBA",  # Baskonia
        "TEL": "MTA",  # Maccabi Tel Aviv
        "PAN": "PAO",  # Panathinaikos
        "ULK": "FBT",  # Fenerbahce
        "RED": "CZV",  # Crvena Zvezda
        "IST": "EFS",  # Anadolu Efes
        "PAM": "VBC",  # Valencia Basket
        "MUN": "BAY",  # Bayern Munich
        "PRS": "PBB",  # Paris Basketball
        "BES": "BJK",  # Besiktas
        "MAD": "RMB",  # Real Madrid
    },
    # ESAKE's team ids (stable across seasons)
    "gbl": {
        "00000001": "PAO",  # Panathinaikos
        "00000002": "OLY",  # Olympiacos
        "00000005": "ARI",  # Aris
        "0000000A": "KOL",  # Kolossos Rodou
        "0000000C": "PAOK",  # PAOK
        "0000000D": "PER",  # Peristeri
        "0000000F": "MAR",  # Maroussi
        "00000010": "AEK",  # AEK
        "00000011": "IRA",  # Iraklis
        "0041ADCB": "FAL",  # Vikos Falcons
        "2A25C696": "PRO",  # Promitheas Patras
        "3CA10C07": "DOX",  # Doxa Lefkadas
        "B742845D": "KAR",  # Karditsa
        "BB4B460F": "MYK",  # Mykonos
    },
}


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
    """The pre-registered row per game, joined with the game's current state.

    That is the earliest row stamped before its tip-off, as the scorecard counts it; a game with
    only rows stamped at or after tip-off keeps its earliest one, marked ``late``.
    """
    if section.log.empty:
        return []
    log = section.log.assign(
        late=pd.to_datetime(section.log["predicted_at_utc"], utc=True)
        >= pd.to_datetime(section.log["tipoff_utc"], utc=True)
    )
    first = log.sort_values(["late", "predicted_at_utc"]).drop_duplicates("game_id", keep="first")
    cols = ["game_id", "tipoff_utc", "played", "forfeit", "home_score", "away_score"]
    joined = first.drop(columns="tipoff_utc").merge(section.games[cols], on="game_id")
    return joined.sort_values("tipoff_utc").to_dict("records")


def _team(section: Section, code: str) -> dict[str, str]:
    shown = DISPLAY_CODES.get(section.key, {}).get(code, code)
    return {"code": shown, "name": section.names.get(code, code)}


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


def _next_tipoff(section: Section, now: datetime) -> tuple[str | None, bool]:
    """The next live-season tip-off with a confirmed time, and False.

    A game without a confirmed time sits at local midnight, so it is skipped; only when no game
    ahead has one is the first date returned, with True (the page shows "time TBC").
    """
    games = section.games
    ahead = games[(games["season"] == section.season) & ~games["played"]]
    ahead = ahead[ahead["tipoff_utc"] > now]
    confirmed = ahead[ahead["confirmed_date"]]
    if not confirmed.empty:
        return _stamp(confirmed["tipoff_utc"].min()), False
    return (None, False) if ahead.empty else (_stamp(ahead["tipoff_utc"].min()), True)


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


def _m1(card: dict[str, Any]) -> dict[str, Any] | None:
    """The one scorecard row of the team model: live M1 vs Elo log loss on the same games."""
    m1 = card.get("m1")
    if m1 is None:
        return None
    same = m1["same_games_as_elo"]
    return {"n": same["n"], "log_loss": same["m1_log_loss"], "elo_log_loss": same["elo_log_loss"]}


def section_data(section: Section, now: datetime) -> dict[str, Any]:
    logged = _logged(section)
    hidden = set(section.scorecard.get("games_not_provable", []))
    upcoming = [
        _game(section, g)
        for g in logged
        if not g["played"] and not g["late"] and g["tipoff_utc"] > now
    ]
    finished = [g for g in logged if g["played"] and not g["forfeit"]][-RECENT_RESULTS:]
    results = [
        {
            **_game(section, g),
            "home_score": int(g["home_score"]),
            "away_score": int(g["away_score"]),
            "hit": (g["p_home"] >= 0.5) == (g["home_score"] > g["away_score"]),
            "provable": g["game_id"] not in hidden and not g["late"],
            "late": bool(g["late"]),
        }
        for g in reversed(finished)
    ]
    card = section.scorecard
    next_tipoff, time_tbc = _next_tipoff(section, now)
    return {
        "key": section.key,
        "title": section.title,
        "season": f"{section.season}-{(section.season + 1) % 100:02d}",
        "logged": len(logged),
        "next_tipoff_utc": next_tipoff,
        "next_tipoff_time_tbc": time_tbc,
        "upcoming": upcoming,
        "results": results,
        "scorecard": {
            "elo": card["elo"],
            "b0": card["b0"],
            "not_provable": len(hidden),
            "late": sum(bool(g["late"]) for g in logged),
            "rolling": card["rolling"],
            "m1": _m1(card),
        },
        "backtest": _backtest(section.backtest),
        "ratings": _ratings(section),
    }


def site_data(sections: list[Section], now: datetime) -> dict[str, Any]:
    euroleague = next((s.games for s in sections if s.key == "euroleague"), None)
    return {
        "generated_at_utc": now.strftime(TIME_FORMAT),
        "competitions": [section_data(s, now) for s in sections],
        "el_titles": titles_payload(euroleague, DISPLAY_CODES["euroleague"]),
    }
