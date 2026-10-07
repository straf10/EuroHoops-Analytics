import json
from collections import Counter

import numpy as np
import pytest

from eurohoops.standings import Result, rank, rank_by_wins
from tests.conftest import REPO

OFFICIAL = json.loads((REPO / "tests/fixtures/standings.json").read_text(encoding="utf-8"))


def random_games(rng: np.random.Generator, teams: list[str]) -> list[Result]:
    """A double round robin with scores in a narrow range, so ties on wins and on every score
    criterion are common; some games carry a regulation score (overtime games)."""
    games = []
    for home in teams:
        for away in teams:
            if home == away:
                continue
            hp = int(rng.integers(70, 74))
            ap = hp + int(rng.choice([-3, -2, -1, 1, 2, 3]))
            regulation = None
            if rng.random() < 0.15:  # level after 40 minutes, then decided in overtime
                level = int(rng.integers(68, 72))
                regulation = (level, level)
                hp, ap = level + int(rng.integers(5, 9)), level + int(rng.integers(5, 9))
                if hp == ap:
                    hp += 1
            games.append(Result(home, away, hp, ap, regulation))
    return games


def random_deductions(rng: np.random.Generator, teams: list[str]) -> dict[str, int]:
    if rng.random() < 0.5:
        return {}
    chosen = rng.choice(teams, size=int(rng.integers(1, 3)), replace=False)
    return {str(t): int(rng.integers(1, 3)) for t in chosen}


def count_wins(teams: list[str], results: list[Result]) -> dict[str, int]:
    won = Counter(r.winner for r in results)
    return {team: won[team] for team in teams}


def test_matches_rank_on_1000_random_tables() -> None:
    """With only the games of the teams tied after deductions (and, for half of the tables,
    all games), rank_by_wins is rank on the full list: shuffled game order, many ties, regulation
    scores and deductions."""
    rng = np.random.default_rng(7)
    ties_seen = 0
    for table in range(1000):
        teams = [f"T{i}" for i in range(int(rng.integers(3, 8)))]
        results = random_games(rng, teams)
        results = [results[i] for i in rng.permutation(len(results))]
        deducted = random_deductions(rng, teams)
        wins = count_wins(teams, results)
        net = {t: wins[t] - deducted.get(t, 0) for t in teams}
        tied = {t for t in teams if sum(net[u] == net[t] for u in teams) > 1}
        ties_seen += bool(tied)
        subset = results if table % 2 else [r for r in results if r.home in tied or r.away in tied]
        assert rank_by_wins(wins, subset, deducted) == rank(results, deducted), table
    assert ties_seen > 500  # most tables really do exercise the tie-breaks


@pytest.mark.parametrize("season", sorted(OFFICIAL))
def test_reproduces_official_final_standings(season: str) -> None:
    table = OFFICIAL[season]
    results = [
        Result(home, away, hp, ap, (reg[0], reg[1]) if reg else None)
        for home, away, hp, ap, reg in table["results"]
    ]
    wins = count_wins(table["official"], results)
    assert rank_by_wins(wins, results, table["deducted"]) == table["official"]


def test_teams_without_a_game_are_still_ranked() -> None:
    results = [Result("A", "B", 80, 70), Result("B", "A", 75, 70)]
    assert rank_by_wins({"A": 1, "B": 1, "C": 2, "D": 0}, results) == ["C", "A", "B", "D"]


def test_sanctioned_team_is_last_among_teams_tied_with_it() -> None:
    # A 2-0 (beat B twice), B and C 1-1. With one win deducted all three are on one win: A,
    # sanctioned, is last; C is ahead of B on the overall score difference (0 v -20).
    results = [
        Result("A", "B", 80, 70),
        Result("B", "A", 70, 80),
        Result("B", "C", 80, 70),
        Result("C", "B", 70, 60),
    ]
    wins = {"A": 2, "B": 1, "C": 1}
    assert rank_by_wins(wins, results) == ["A", "C", "B"]
    assert rank_by_wins(wins, results, {"A": 1}) == ["C", "B", "A"]
