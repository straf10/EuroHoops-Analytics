import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from eurohoops.eval.backtest import TunedModel
from eurohoops.models.elo import EloParams
from eurohoops.parse.games import GAMES_SCHEMA

FIXTURES = Path(__file__).parent / "fixtures"
TEAMS = ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF")


def load_fixture(name: str) -> list[dict[str, Any]]:
    games: list[dict[str, Any]] = json.loads((FIXTURES / name).read_text())["data"]
    return games


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Disable throttle/backoff sleeps; record requested durations."""
    slept: list[float] = []
    monkeypatch.setattr(time, "sleep", slept.append)
    return slept


def make_games(seasons: dict[int, bool], seed: int = 7) -> pd.DataFrame:
    """Synthetic double round-robin per season; ``seasons`` maps season -> played."""
    rng = np.random.default_rng(seed)
    strength = dict(zip(TEAMS, rng.normal(0, 6, len(TEAMS)), strict=True))
    rows = []
    for season, played in seasons.items():
        start = pd.Timestamp(f"{season}-10-01T18:00:00Z")
        pairs = [(h, a) for h in TEAMS for a in TEAMS if h != a]
        for rnd, (home, away) in enumerate(pairs):
            code = rnd + 1
            margin = round(float(strength[home] - strength[away] + 3 + rng.normal(0, 11)))
            margin = margin if margin != 0 else 1
            rows.append(
                {
                    "game_id": f"E{season}_{code}",
                    "season": season,
                    "game_code": code,
                    "phase": "RS",
                    "round": rnd // 3 + 1,
                    "tipoff_utc": start + pd.Timedelta(days=rnd // 3 * 7, minutes=15 * (rnd % 3)),
                    "home": home,
                    "away": away,
                    "home_score": 80 + max(margin, 0) if played else None,
                    "away_score": 80 + max(-margin, 0) if played else None,
                    "played": played,
                    "neutral": False,
                    "confirmed_date": True,
                }
            )
    df = pd.DataFrame(rows).astype(
        {
            "home_score": "Int64",
            "away_score": "Int64",
            "tipoff_utc": "datetime64[ns, UTC]",
            "phase": str,
            "home": str,
            "away": str,
            "game_id": str,
        }
    )
    return GAMES_SCHEMA.validate(df.sort_values(["tipoff_utc", "game_code"], ignore_index=True))


@pytest.fixture
def tuned() -> TunedModel:
    return TunedModel(
        params=EloParams(k=20.0, hca=90.0, reversion=0.25),
        margin_scale=25.0,
        b0_home_win_rate=0.6,
        b0_home_margin=3.0,
    )
