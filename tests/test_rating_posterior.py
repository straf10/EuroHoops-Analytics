"""K1: M1's rating posterior (Gaussian over [μ, h, off, def]) and pace point, per cutoff."""

from pathlib import Path

import numpy as np

from eurohoops.models.team_eff import (
    DecayParams,
    History,
    forecast,
    pace_fits,
    prepare_history,
    rating_fits,
)
from tests.test_team_eff import league

FIXTURE = Path(__file__).parent / "fixtures" / "m1_fits_reference.npz"
RATING = DecayParams(half_life_days=90.0, carry=0.6, ridge=300.0)
PACE = DecayParams(half_life_days=60.0, carry=1.0, ridge=2.0)


def reference_history() -> History:
    """Four teams over 2020-2021 with noisy ratings; DDD only appears in 2021, one 2021 game is
    at a neutral venue and the last round of 2021 is unplayed (forecast only)."""
    ratings = {
        "AAA": (4.0, 1.0, 1.5),
        "BBB": (0.0, -2.0, -1.0),
        "CCC": (-3.0, 1.0, 0.5),
        "DDD": (1.0, 2.0, -0.5),
    }
    games, rows = league(ratings, [2020, 2021], noise=7.0, seed=11, unplayed_last_round=True)
    ddd = (games["home"] == "DDD") | (games["away"] == "DDD")
    early = games["season"] == 2020
    games = games[~(ddd & early)].reset_index(drop=True)
    rows = rows[rows["game_id"].isin(games["game_id"])]
    neutral = games.index[games["season"] == 2021][3]
    games.loc[neutral, "neutral"] = True
    return prepare_history(games, rows)


def test_m1_fits_are_byte_identical_to_the_reference_stored_before_the_refactor() -> None:
    history = reference_history()
    got = forecast(history, RATING, PACE)
    home, away = rating_fits(history, RATING)
    stored = np.load(FIXTURE)
    arrays = {
        "home_points": got.home_points,
        "away_points": got.away_points,
        "pace_forecast": got.pace,
        "home_ortg": home,
        "away_ortg": away,
        "pace_fits": pace_fits(history, PACE),
    }
    assert set(stored.files) == set(arrays)
    for name, array in arrays.items():
        assert np.array_equal(stored[name], array, equal_nan=True), name
