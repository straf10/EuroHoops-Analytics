import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from eurohoops.config import EUROLEAGUE, GBL, MART_PATH, SQL_DIR
from eurohoops.eval.backtest import TunedModel
from eurohoops.marts import build_marts
from eurohoops.models.elo import EloParams
from eurohoops.parse.games import conform, write_table
from eurohoops.parse.team_box import TEAM_GAMES_SCHEMA

FIXTURES = Path(__file__).parent / "fixtures"
REPO = Path(__file__).parent.parent
TEAMS = ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF")


def load_fixture(name: str) -> list[dict[str, Any]]:
    games: list[dict[str, Any]] = json.loads((FIXTURES / name).read_text(encoding="utf-8"))["data"]
    return games


def esake_fixture(name: str) -> str:
    return (FIXTURES / "esake" / name).read_text(encoding="utf-8")


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Disable throttle/backoff sleeps; record requested durations."""
    slept: list[float] = []
    monkeypatch.setattr(time, "sleep", slept.append)
    return slept


def make_games(seasons: dict[int, bool], seed: int = 7, gbl_like: bool = False) -> pd.DataFrame:
    """Synthetic double round-robin per season; ``seasons`` maps season -> played.

    ``gbl_like`` adds what GBL data has and EuroLeague data lacks: every 7th game is a 20-0
    forfeit and the last round of each season is a playoff round.
    """
    rng = np.random.default_rng(seed)
    strength = dict(zip(TEAMS, rng.normal(0, 6, len(TEAMS)), strict=True))
    prefix = "GBL" if gbl_like else "E"
    pairs = [(h, a) for h in TEAMS for a in TEAMS if h != a]
    rows = []
    for season, played in seasons.items():
        start = pd.Timestamp(f"{season}-10-01T18:00:00Z")
        for rnd, (home, away) in enumerate(pairs):
            code = rnd + 1
            margin = round(float(strength[home] - strength[away] + 3 + rng.normal(0, 11)))
            margin = margin if margin != 0 else 1
            forfeit = gbl_like and played and code % 7 == 0
            rows.append(
                {
                    "game_id": f"{prefix}{season}_{code}",
                    "season": season,
                    "game_code": code,
                    "phase": "PO" if gbl_like and rnd >= len(pairs) - 3 else "RS",
                    "round": rnd // 3 + 1,
                    "round_label": f"Round {rnd // 3 + 1}",
                    "tipoff_utc": start + pd.Timedelta(days=rnd // 3 * 7, minutes=15 * (rnd % 3)),
                    "home": home,
                    "away": away,
                    "home_score": (20 if forfeit else 80 + max(margin, 0)) if played else None,
                    "away_score": (0 if forfeit else 80 + max(-margin, 0)) if played else None,
                    "played": played,
                    "forfeit": forfeit,
                    "neutral": False,
                    "confirmed_date": True,
                }
            )
    return conform(pd.DataFrame(rows))


def teams_table(games: pd.DataFrame) -> pd.DataFrame:
    codes = sorted({*games["home"], *games["away"]})
    return pd.DataFrame({"team": codes, "name": [f"Team {c} & Co" for c in codes]}, dtype=str)


def write_pipeline(euroleague: pd.DataFrame, gbl: pd.DataFrame) -> None:
    """Stage both competitions and build the mart, as `ingest` + `build` would (CWD-relative)."""
    for comp, games in ((EUROLEAGUE, euroleague), (GBL, gbl)):
        write_table(games, comp.staging_games)
        write_table(teams_table(games), comp.staging_teams)
    build_marts(MART_PATH, REPO / SQL_DIR)


@pytest.fixture
def tuned() -> TunedModel:
    return TunedModel(
        params=EloParams(k=20.0, hca=90.0, reversion=0.25),
        margin_scale=25.0,
        b0_home_win_rate=0.6,
        b0_home_margin=3.0,
    )


def make_team_games(games: pd.DataFrame, seed: int = 11) -> pd.DataFrame:
    """``team_games`` rows for every rated game of ``make_games`` output (points = scores).

    FTA 20, OREB 10, TOV 12 per team, FGA 55-65, so poss_raw = FGA + 10.8 (65.8-75.8).
    """
    rng = np.random.default_rng(seed)
    rated = games[games["played"] & ~games["forfeit"]]
    rows = []
    for game in rated.to_dict("records"):
        fga = rng.integers(55, 66, size=2)
        raw = fga + 10.8
        competition = "gbl" if str(game["game_id"]).startswith("GBL") else "euroleague"
        for side, (team, opp, points) in enumerate(
            (
                (game["home"], game["away"], game["home_score"]),
                (game["away"], game["home"], game["away_score"]),
            )
        ):
            rows.append(
                {
                    "competition": competition,
                    "season": game["season"],
                    "game_id": game["game_id"],
                    "team": team,
                    "opponent": opp,
                    "home": side == 0 and not game["neutral"],
                    "points": int(points),
                    "fga": int(fga[side]),
                    "fta": 20,
                    "oreb": 10,
                    "dreb": 25,
                    "tov": 12,
                    "minutes": 40.0,
                    "poss_raw": float(raw[side]),
                    "poss_game": float(raw.mean()),
                    "source": "esake_box" if competition == "gbl" else "euroleague_box",
                }
            )
    return TEAM_GAMES_SCHEMA.validate(pd.DataFrame(rows))
