"""Projected-share variants (week 14-16, J2): decay, availability, walk-forward safety.

``tests/fixtures/m5/projected_shares_hc.csv`` is the output of ``minutes.projected_shares`` on
``fixture()`` below, generated once from the code as it stood before ``rotation.py`` was written
(``minutes.py`` is unchanged by J2). It is the single stored expected value in this file: it
guards against ``minutes.py`` changing behaviour under the variants built on it.
"""

import math
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from eurohoops.models import minutes
from eurohoops.models.rotation import projected_shares_variant

HC_FIXTURE = Path(__file__).parent / "fixtures" / "m5" / "projected_shares_hc.csv"
VARIANTS = ("proj_hc", "proj_decay", "proj_avail")
TEAMS = ("AAA", "BBB")
BASE_SECONDS = (2400, 2300, 2200, 2100, 1500, 1000, 800, 500, 300, 0)
LEAVER = 8  # plays the first two games of every season, then never


def game_row(game_id: str, season: int, rnd: int, day: str, *, home: str, away: str) -> dict:
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


def fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    """2 teams x 2 seasons; 2021 round 3 has two games (a day apart) to test the round cutoff."""
    schedule = [
        (2020, 1, "2020-10-01"),
        (2020, 2, "2020-10-08"),
        (2020, 3, "2020-10-15"),
        (2020, 4, "2020-10-22"),
        (2021, 1, "2021-10-01"),
        (2021, 2, "2021-10-08"),
        (2021, 3, "2021-10-15"),
        (2021, 3, "2021-10-16"),
        (2021, 4, "2021-10-22"),
        (2021, 5, "2021-10-29"),
    ]
    games, lines = [], []
    for index, (season, rnd, day) in enumerate(schedule):
        in_season = sum(1 for s, _, _ in schedule[:index] if s == season)
        home, away = TEAMS if index % 2 == 0 else TEAMS[::-1]
        game_id = f"G{index}"
        games.append(game_row(game_id, season, rnd, day, home=home, away=away))
        for team in TEAMS:
            for i, base in enumerate(BASE_SECONDS):
                sec = base + 60 * ((i * 7 + index * 3 + len(team)) % 5) * (base > 0)
                if i == LEAVER:
                    sec = 600 if in_season < 2 else 0
                lines.append(
                    {"game_id": game_id, "team": team, "player_id": f"{team}{i}", "sec": sec}
                )
    return pd.DataFrame(games), pd.DataFrame(lines)


def shares_of(frame: pd.DataFrame, game_id: str, team: str) -> dict[str, float]:
    rows = frame[(frame["game_id"] == game_id) & (frame["team"] == team)]
    return dict(zip(rows["player_id"], rows["share"], strict=True))


def world(team_games: list[dict[str, int]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One season of AAA v BBB, one round per game, AAA's seconds per game as given. The last
    game is the one projected. BBB just plays one man for 12,000 s (a capped broken line)."""
    games, lines = [], []
    for index, seconds in enumerate(team_games):
        game_id = f"W{index}"
        games.append(
            game_row(game_id, 2021, index + 1, f"2021-11-{index + 1:02d}", home="AAA", away="BBB")
        )
        lines.extend(
            {"game_id": game_id, "team": "AAA", "player_id": p, "sec": s}
            for p, s in seconds.items()
        )
        lines.append({"game_id": game_id, "team": "BBB", "player_id": "b", "sec": 12000})
    return pd.DataFrame(games), pd.DataFrame(lines)


def test_hc_is_unchanged_minutes_output() -> None:
    games, player_games = fixture()
    stored = pd.read_csv(
        HC_FIXTURE,
        dtype={"game_id": str, "side": str, "team": str, "player_id": str},
        float_precision="round_trip",
    )
    assert_frame_equal(minutes.projected_shares(games, player_games, 5), stored, check_exact=True)
    assert_frame_equal(projected_shares_variant("proj_hc", games, player_games), stored)
    wide = projected_shares_variant("proj_hc", games, player_games, n_games=2)
    assert_frame_equal(wide, minutes.projected_shares(games, player_games, 2))


def test_decay_with_infinite_half_life_is_all_games_equally_weighted() -> None:
    games, player_games = fixture()
    decay = projected_shares_variant("proj_decay", games, player_games, half_life_games=math.inf)
    assert_frame_equal(decay, minutes.projected_shares(games, player_games, n_games=10_000))
    # the fixture does include the previous-season fallback (2021 round 1) and a full season
    assert "G4" in set(decay["game_id"]) and "G9" in set(decay["game_id"])


def test_decay_hand_example() -> None:
    """3 games, 7 players, half-life 1 game: weights 0.25 (G0), 0.5 (G1), 1 (G2, most recent)."""
    full = 2400
    games, player_games = world(
        [
            {"a": full, "b": full, "c": full, "d": full, "e": full},
            {"a": full, "b": full, "c": full, "d": full, "f": 1200, "g": 1200},
            {"a": full, "b": full, "c": full, "f": full, "g": 1200, "e": 1200},
            {"a": 1},  # the game being projected
        ]
    )
    out = projected_shares_variant("proj_decay", games, player_games, half_life_games=1.0)
    got = shares_of(out, "W3", "AAA")
    # every game's team total is 5 x 2400 = 12000 s; weighted total = 12000 * (0.25 + 0.5 + 1)
    # = 21000. A player's share = 5 x (weighted seconds) / 21000.
    # a, b, c played 2400 in all three games: weighted 2400 * 1.75 = 4200 -> 5 x 4200 / 21000 = 1
    # d: G0 and G1 -> 2400 * (0.25 + 0.5) = 1800 -> 9000 / 21000 = 3/7
    # e: G0 (0.25) 2400 and G2 (1) 1200 -> 600 + 1200 = 1800 -> 3/7
    # f: G1 (0.5) 1200 and G2 (1) 2400 -> 600 + 2400 = 3000 -> 15000 / 21000 = 5/7
    # g: G1 (0.5) 1200 and G2 (1) 1200 -> 600 + 1200 = 1800 -> 3/7
    expected = {"a": 1.0, "b": 1.0, "c": 1.0, "d": 3 / 7, "e": 3 / 7, "f": 5 / 7, "g": 3 / 7}
    assert got == pytest.approx(expected, abs=1e-12)
    assert sum(got.values()) == pytest.approx(5.0)


def test_decay_falls_back_to_whole_previous_season_equally_weighted() -> None:
    games, player_games = fixture()
    decay = projected_shares_variant("proj_decay", games, player_games, half_life_games=1.0)
    hc = minutes.projected_shares(games, player_games, n_games=10_000)
    # G4 is 2021 round 1: no earlier 2021 game, so the half-life must not matter
    assert shares_of(decay, "G4", "AAA") == shares_of(hc, "G4", "AAA")
    # G5 has one earlier game this season: decay and H-c agree there too (a single weight)
    assert shares_of(decay, "G5", "AAA") == pytest.approx(shares_of(hc, "G5", "AAA"))
    # G6 has two: a short half-life moves the shares away from the equal-weight ones
    assert shares_of(decay, "G6", "AAA") != pytest.approx(shares_of(hc, "G6", "AAA"))


AVAIL_GAMES = [  # every game's 12,000 s; "gone" plays the first two games only
    {"star": 2000, "gone": 1200, "x": 2000, "y": 2000, "z": 2000, "p": 2000, "q": 800},
    {"star": 2000, "gone": 1200, "x": 2000, "y": 2000, "z": 2000, "p": 2000, "q": 800},
    {"star": 2000, "x": 2200, "y": 2200, "z": 2000, "p": 2000, "q": 1600},
    {"star": 2000, "x": 2200, "y": 2200, "z": 2000, "p": 2000, "q": 1600, "gone": 0},
    {"star": 1},  # the game being projected
]


def test_avail_drops_the_absent_player_and_rescales_to_the_same_sum() -> None:
    games, player_games = world(AVAIL_GAMES)
    decay = shares_of(projected_shares_variant("proj_decay", games, player_games), "W4", "AAA")
    avail = shares_of(projected_shares_variant("proj_avail", games, player_games), "W4", "AAA")
    assert "gone" in decay
    assert "gone" not in avail
    assert set(avail) == set(decay) - {"gone"}
    assert max(avail.values()) < 1.0  # no cap binds here
    assert sum(avail.values()) == pytest.approx(sum(decay.values()), abs=1e-9)
    scale = sum(decay.values()) / (sum(decay.values()) - decay["gone"])
    assert avail == pytest.approx({p: decay[p] * scale for p in avail})


def test_avail_cap_binds_and_the_team_sum_shrinks() -> None:
    star_all = [{**g, "star": 2400} for g in AVAIL_GAMES[:4]] + [AVAIL_GAMES[4]]
    games, player_games = world(star_all)
    decay = shares_of(projected_shares_variant("proj_decay", games, player_games), "W4", "AAA")
    avail = shares_of(projected_shares_variant("proj_avail", games, player_games), "W4", "AAA")
    # the star's share is just under 1 (5 x 2400 / 12,400); the rescaling would lift it above 1
    assert decay["star"] < 1.0
    assert decay["star"] * sum(decay.values()) / (sum(decay.values()) - decay["gone"]) > 1.0
    assert "gone" not in avail
    assert avail["star"] == 1.0
    assert all(share <= 1.0 for share in avail.values())
    assert sum(avail.values()) < sum(decay.values()) - 1e-4


def test_avail_keeps_everyone_when_nothing_qualifies_for_a_drop() -> None:
    games, player_games = world(AVAIL_GAMES)
    decay = projected_shares_variant("proj_decay", games, player_games)
    # the last 3 games include one "gone" played: nobody is absent from all of them
    wide = projected_shares_variant("proj_avail", games, player_games, absent_games=3)
    assert shares_of(wide, "W4", "AAA") == shares_of(decay, "W4", "AAA")
    # W1 has only one earlier game this season (< absent_games = 2): nobody dropped
    short = projected_shares_variant("proj_avail", games, player_games, absent_games=2)
    assert shares_of(short, "W1", "AAA") == shares_of(decay, "W1", "AAA")
    # a window longer than the history: nobody dropped anywhere
    longer = projected_shares_variant("proj_avail", games, player_games, absent_games=99)
    assert_frame_equal(longer, decay)
    # an empty window would drop every player: keep them all
    none = projected_shares_variant("proj_avail", games, player_games, absent_games=0)
    assert_frame_equal(none, decay)


def test_avail_on_fixture_drops_the_leaver() -> None:
    games, player_games = fixture()
    avail = projected_shares_variant("proj_avail", games, player_games)
    decay = projected_shares_variant("proj_decay", games, player_games)
    # 2021 game G8 (round 4): earlier games were played with the leaver only in the first two
    assert "AAA8" in shares_of(decay, "G8", "AAA")
    assert "AAA8" not in shares_of(avail, "G8", "AAA")
    # G6 follows only 2 earlier 2021 games, both with the leaver: nobody dropped
    assert "AAA8" in shares_of(avail, "G6", "AAA")


@pytest.mark.parametrize("variant", VARIANTS)
def test_shares_are_bounded(variant: str) -> None:
    games, player_games = fixture()
    out = projected_shares_variant(variant, games, player_games)
    assert not out.empty
    assert (out["share"] <= 1.0).all() and (out["share"] >= 0.0).all()
    totals = out.groupby(["game_id", "side"])["share"].sum()
    assert (totals <= 5.0 + 1e-9).all()


def test_unknown_variant_raises() -> None:
    games, player_games = fixture()
    with pytest.raises(ValueError, match="unknown projection variant"):
        projected_shares_variant("proj_nope", games, player_games)


def mutate(
    games: pd.DataFrame,
    player_games: pd.DataFrame,
    game_ids: list[str],
) -> pd.DataFrame:
    """Rewrite the seconds of ``game_ids`` (keeping them positive for the players who played)."""
    hit = player_games["game_id"].isin(game_ids)
    changed = player_games.copy()
    changed.loc[hit, "sec"] = changed.loc[hit, "sec"] * 3 + 777
    return changed


@pytest.mark.parametrize("variant", VARIANTS)
def test_nothing_at_or_after_the_round_cutoff_leaks(variant: str) -> None:
    games, player_games = fixture()
    cutoff = minutes.round_cutoffs(games)
    time = pd.Series(minutes._epoch(games["tipoff_utc"]), index=games.index)
    base = projected_shares_variant(variant, games, player_games)
    for index, game_id in games["game_id"].items():
        before = time < cutoff[index]
        late = games.loc[~before, "game_id"].tolist()  # the game itself and everything at/after
        for changed in (
            player_games[player_games["game_id"] != game_id],
            mutate(games, player_games, late),
        ):
            again = projected_shares_variant(variant, games, changed)
            for team in TEAMS:
                assert shares_of(again, str(game_id), team) == shares_of(base, str(game_id), team)


@pytest.mark.parametrize("variant", VARIANTS)
def test_altering_an_earlier_game_does_change_the_projection(variant: str) -> None:
    games, player_games = fixture()
    base = projected_shares_variant(variant, games, player_games)
    again = projected_shares_variant(variant, games, mutate(games, player_games, ["G5"]))
    # G7 (2021, round 3, second game) and G8 both look back at G5
    assert shares_of(again, "G8", "AAA") != shares_of(base, "G8", "AAA")
