import numpy as np
import pytest

from eurohoops.models.elo import (
    EloParams,
    GameArrays,
    fit_margin_scale,
    mov_multiplier,
    prepare,
    replay,
    win_probability,
)
from tests.conftest import make_games

PARAMS = EloParams(k=20.0, hca=100.0, reversion=0.5)

# Hand-computed for a 1500 vs 1500 game, HCA 100, home wins by 10, K 20:
#   p      = 1 / (1 + 10^(-100/400)) = 1 / 1.5623413 = 0.6400650
#   mult   = (10 + 3)^0.8 / (7.5 + 0.006 * 100) = 7.783146 / 8.1 = 0.960882
#   shift  = 20 * 0.960882 * (1 - 0.6400650) = 6.917100
P_HOME = 0.6400650
MULT = 0.960882
SHIFT = 6.917100


def games(*rows: tuple[int, str, str, bool, bool, int]) -> GameArrays:
    season, home, away, neutral, played, margin = (list(col) for col in zip(*rows, strict=True))
    return GameArrays(
        season=season, home=home, away=away, neutral=neutral, played=played, margin=margin
    )


def test_win_probability_and_mov_multiplier_by_hand() -> None:
    assert win_probability(0.0) == 0.5
    assert win_probability(100.0) == pytest.approx(P_HOME, abs=1e-7)
    assert mov_multiplier(10, 100.0) == pytest.approx(MULT, abs=1e-6)


def test_single_game_update_by_hand() -> None:
    # A beats B at home by 10; then two neutral games reveal the new ratings against fresh C.
    diffs = replay(
        games(
            (2024, "A", "B", False, True, 10),
            (2024, "A", "C", True, False, 0),
            (2024, "C", "B", True, False, 0),
        ),
        PARAMS,
    )
    assert diffs[0] == 100.0
    assert diffs[1] == pytest.approx(SHIFT, abs=1e-5)  # A = 1500 + shift
    assert diffs[2] == pytest.approx(SHIFT, abs=1e-5)  # B = 1500 - shift


def test_ratings_are_zero_sum_per_game() -> None:
    # Whatever A gains, B loses: a neutral rematch differs by exactly twice the shift.
    diffs = replay(
        games((2024, "B", "A", False, True, -4), (2024, "A", "B", True, False, 0)), PARAMS
    )
    gain_a = diffs[1] / 2
    diffs_vs_fresh = replay(
        games(
            (2024, "B", "A", False, True, -4),
            (2024, "A", "C", True, False, 0),
            (2024, "C", "B", True, False, 0),
        ),
        PARAMS,
    )
    assert diffs_vs_fresh[1] == pytest.approx(gain_a)
    assert diffs_vs_fresh[2] == pytest.approx(gain_a)


def test_neutral_venue_has_no_home_court_advantage() -> None:
    diffs = replay(games((2024, "A", "B", True, True, 10)), PARAMS)
    assert diffs[0] == 0.0


def test_season_start_reversion() -> None:
    # After one game A = 1500 + shift, B = 1500 - shift; reversion 0.5 halves the gap.
    diffs = replay(
        games((2024, "A", "B", False, True, 10), (2025, "A", "B", True, False, 0)), PARAMS
    )
    assert diffs[1] == pytest.approx(2 * SHIFT * 0.5, abs=1e-5)


def test_unplayed_games_do_not_update_ratings() -> None:
    diffs = replay(
        games((2024, "A", "B", True, False, 0), (2024, "A", "B", True, False, 0)), PARAMS
    )
    assert list(diffs) == [0.0, 0.0]


def test_margin_scale_least_squares_through_origin() -> None:
    diffs = np.array([50.0, -100.0, 200.0])
    assert fit_margin_scale(diffs, diffs / 25.0) == pytest.approx(25.0)


def test_forfeits_do_not_update_ratings() -> None:
    games = make_games({2024: True}, gbl_like=True)
    assert games["forfeit"].any()
    without = games[~games["forfeit"]].reset_index(drop=True)
    diffs_all = replay(prepare(games), PARAMS)[~games["forfeit"].to_numpy()]
    assert np.array_equal(diffs_all, replay(prepare(without), PARAMS))
