import json

import pytest

from eurohoops.standings import EUROLEAGUE_2026, GBL_2026, Result, rank
from tests.conftest import REPO

OFFICIAL = json.loads((REPO / "tests/fixtures/standings.json").read_text(encoding="utf-8"))


def games(*rows: tuple[str, str, int, int]) -> list[Result]:
    return [Result(*row) for row in rows]


@pytest.mark.parametrize("season", sorted(OFFICIAL))
def test_matches_official_final_standings(season: str) -> None:
    """Real seasons with 2- to 5-team ties, an overtime-decided tie and a sanction."""
    table = OFFICIAL[season]
    results = [
        Result(home, away, hp, ap, (reg[0], reg[1]) if reg else None)
        for home, away, hp, ap, reg in table["results"]
    ]
    assert rank(results, table["deducted"]) == table["official"]


def test_overtime_points_do_not_break_ties() -> None:
    """EuroLeague 2022-23 is reproduced only when overtime points are left out (Art. 19.4)."""
    table = OFFICIAL["euroleague_2022-23"]
    final_only = [Result(h, a, hp, ap) for h, a, hp, ap, _ in table["results"]]
    assert rank(final_only, table["deducted"]) != table["official"]


def test_three_way_tie_broken_by_head_to_head_difference() -> None:
    # A, B, C are 4-2, each 2-2 among themselves and 2-0 against D.
    # Head-to-head difference: A +10 -5 -5 +10 = +10, B -10 +5 +10 -15 = -10, C 0.
    results = games(
        ("A", "B", 80, 70),
        ("B", "A", 75, 70),
        ("A", "C", 70, 75),
        ("C", "A", 70, 80),
        ("B", "C", 80, 70),
        ("C", "B", 85, 70),
        *[(t, "D", 90, 60) for t in "ABC"],
        *[("D", t, 60, 90) for t in "ABC"],
    )
    assert rank(results) == ["A", "C", "B", "D"]


def test_two_teams_left_tied_restart_with_their_own_games() -> None:
    # Each pair splits 1-1. Head-to-head difference among all three: A +5, B +5, C -10, so C
    # is third and A/B restart at 19.5.2 I with only their two games: B is +3 there.
    # A scored more points overall (325 v 308), so skipping the restart would put A first.
    results = games(
        ("A", "B", 80, 78),
        ("B", "A", 80, 75),
        ("A", "C", 90, 80),
        ("C", "A", 82, 80),
        ("B", "C", 80, 70),
        ("C", "B", 78, 70),
    )
    assert rank(results) == ["B", "A", "C"]


def test_teams_that_have_not_met_twice_use_overall_criteria() -> None:
    # All 1-1 after one meeting each (19.5.1): overall difference A +30, B 0, C -30.
    results = games(("A", "B", 70, 80), ("A", "C", 100, 60), ("C", "B", 80, 70))
    assert rank(results) == ["A", "B", "C"]


def test_sanctioned_team_is_last_among_teams_tied_with_it() -> None:
    # A 3-1 (beat B twice), B 2-2, C 1-3. With one win deducted A ties B and drops below it.
    results = games(
        ("A", "B", 80, 70),
        ("B", "A", 70, 80),
        ("A", "C", 80, 70),
        ("C", "A", 80, 70),
        ("B", "C", 80, 70),
        ("C", "B", 70, 80),
    )
    assert rank(results) == ["A", "B", "C"]
    assert rank(results, {"A": 1}) == ["B", "A", "C"]


def test_a_perfectly_symmetric_tie_stays_alphabetical() -> None:
    results = games(("B", "A", 80, 70), ("A", "B", 80, 70))
    assert rank(results) == ["A", "B"]


def test_2026_formats_are_consistent() -> None:
    for fmt in (EUROLEAGUE_2026, GBL_2026):
        positions = (*fmt.playoffs_direct, *fmt.play_in, *fmt.eliminated, *fmt.relegated)
        assert sorted(positions) == list(range(1, fmt.teams + 1))
        assert fmt.regular_season_rounds == 2 * (fmt.teams - 1)
        assert fmt.sources
    assert not EUROLEAGUE_2026.unverified
    assert GBL_2026.unverified
    assert EUROLEAGUE_2026.deducted_wins() == {}
    assert GBL_2026.deducted_wins() == {"00000002": 1, "00000001": 1}  # OLY, PAO: -2 points
