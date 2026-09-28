import math

import pandas as pd
import pytest

from eurohoops.models.minutes import (
    expected_possessions,
    oracle_shares,
    projected_shares,
    round_cutoffs,
)


def game(game_id: str, season: int, *key: object) -> dict[str, object]:
    rnd, day, home, away = key
    return {
        "game_id": game_id,
        "season": season,
        "phase": "RS",
        "round": rnd,
        "tipoff_utc": pd.Timestamp(f"{day}T18:00:00Z"),
        "home": home,
        "away": away,
        "played": True,
        "forfeit": False,
        "neutral": False,
    }


@pytest.fixture
def games() -> pd.DataFrame:
    return pd.DataFrame(
        [
            game("G0", 2020, 1, "2020-10-01", "AAA", "BBB"),  # last season
            game("G1", 2021, 1, "2021-10-01", "AAA", "BBB"),
            game("G2", 2021, 2, "2021-10-08", "BBB", "AAA"),
            game("G3", 2021, 3, "2021-10-15", "AAA", "CCC"),
            game("G4", 2021, 3, "2021-10-16", "BBB", "DDD"),  # same round as G3, a day later
        ]
    )


def lines(game_id: str, team: str, seconds: dict[str, int], game_sec: int = 2400) -> list[dict]:
    return [
        {"game_id": game_id, "team": team, "player_id": p, "sec": s, "game_sec": game_sec}
        for p, s in seconds.items()
    ]


@pytest.fixture
def player_games() -> pd.DataFrame:
    """Each team-game's seconds add up to 5 x the game (12,000 s; 13,500 s with an overtime)."""
    full = {"f1": 2400, "f2": 2400, "f3": 2400}
    rows = [
        *lines("G0", "AAA", {"p1": 2400, "p2": 1200, "old": 1200, **full}),
        *lines("G1", "AAA", {"p1": 2000, "p2": 400, "f4": 2400, **full}),
        *lines(
            "G2",
            "AAA",
            {"p1": 1000, "p3": 1400, "f1": 2700, "f2": 2700, "f3": 2700, "f4": 3000},
            game_sec=2700,
        ),
        *lines("G3", "AAA", {"p1": 9, "p3": 9}),  # the game itself: never read before it
        *lines("G1", "BBB", {"q1": 12000}),  # a broken line: capped at the whole game
    ]
    return pd.DataFrame(rows)


def shares(frame: pd.DataFrame, game_id: str, side: str) -> dict[str, float]:
    rows = frame[(frame["game_id"] == game_id) & (frame["side"] == side)]
    return dict(zip(rows["player_id"], rows["share"], strict=True))


def some(got: dict[str, float], keys: tuple[str, ...]) -> dict[str, float]:
    return {k: got[k] for k in keys}


def test_round_cutoff_is_the_first_tipoff_of_the_round(games: pd.DataFrame) -> None:
    cutoff = round_cutoffs(games)
    assert cutoff[3] == cutoff[4] == pd.Timestamp("2021-10-15T18:00:00Z").timestamp()


def test_first_game_uses_last_seasons_shares(
    games: pd.DataFrame, player_games: pd.DataFrame
) -> None:
    got = shares(projected_shares(games, player_games), "G1", "home")
    assert got == pytest.approx({"p1": 1.0, "p2": 0.5, "old": 0.5, "f1": 1.0, "f2": 1.0, "f3": 1.0})
    assert sum(got.values()) == pytest.approx(5.0)


def test_hand_example_of_the_previous_games(
    games: pd.DataFrame, player_games: pd.DataFrame
) -> None:
    got = shares(projected_shares(games, player_games), "G3", "home")
    # G1 and G2: the team recorded 12,000 + 13,500 s; p1 played 2,000 + 1,000 s
    expected = {"p1": 5 * 3000 / 25500, "p2": 5 * 400 / 25500, "p3": 5 * 1400 / 25500}
    assert some(got, ("p1", "p2", "p3")) == pytest.approx(expected)
    assert got["f4"] == 1.0  # 5 x 5,400 / 25,500 > 1: capped at the whole game


def test_only_the_last_n_games_count(games: pd.DataFrame, player_games: pd.DataFrame) -> None:
    got = shares(projected_shares(games, player_games, n_games=1), "G3", "home")
    assert some(got, ("p1", "p3")) == pytest.approx({"p1": 5000 / 13500, "p3": 7000 / 13500})
    assert "p2" not in got


def test_a_team_without_history_gets_no_players(
    games: pd.DataFrame, player_games: pd.DataFrame
) -> None:
    frame = projected_shares(games, player_games)
    assert shares(frame, "G3", "away") == {}  # CCC: no games in 2020 or 2021


def test_editing_the_game_or_later_changes_no_projection(
    games: pd.DataFrame, player_games: pd.DataFrame
) -> None:
    before = projected_shares(games, player_games)
    edited = player_games.copy()
    edited.loc[edited["game_id"] == "G3", "sec"] = 2400
    edited = pd.concat([edited, pd.DataFrame(lines("G4", "BBB", {"q9": 2400}))])
    after = projected_shares(games, edited)
    pd.testing.assert_frame_equal(before, after)


def test_oracle_shares_are_the_games_own(games: pd.DataFrame, player_games: pd.DataFrame) -> None:
    got = shares(oracle_shares(games, player_games), "G2", "away")
    assert some(got, ("p1", "p3")) == pytest.approx({"p1": 5000 / 13500, "p3": 7000 / 13500})
    assert got["f4"] == 1.0  # 3,000 s of a 2,700 s game: capped
    assert shares(oracle_shares(games, player_games), "G1", "away") == {"q1": 1.0}


def test_expected_possessions(games: pd.DataFrame) -> None:
    team_games = pd.DataFrame(
        [
            {"game_id": "G0", "team": "AAA", "poss_game": 70.0, "minutes": 40.0},
            {"game_id": "G0", "team": "BBB", "poss_game": 70.0, "minutes": 40.0},
            {"game_id": "G1", "team": "AAA", "poss_game": 80.0, "minutes": 40.0},
            {"game_id": "G1", "team": "BBB", "poss_game": 80.0, "minutes": 40.0},
            {"game_id": "G2", "team": "AAA", "poss_game": 90.0, "minutes": 45.0},
            {"game_id": "G2", "team": "BBB", "poss_game": 90.0, "minutes": 45.0},
            {"game_id": "G3", "team": "AAA", "poss_game": 999.0, "minutes": 40.0},
        ]
    )
    p = expected_possessions(games, team_games)
    assert math.isnan(p["G0"])  # nothing earlier at all
    assert p["G1"] == pytest.approx(70.0)  # both teams: last season's mean
    assert p["G2"] == pytest.approx(80.0)
    # AAA: mean(80, 80) this season; CCC: no history -> mean of every earlier team-game row
    assert p["G3"] == pytest.approx((80.0 + 460.0 / 6.0) / 2.0)
