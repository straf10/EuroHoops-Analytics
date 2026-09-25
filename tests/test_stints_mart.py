"""E2: the EuroLeague stints mart (5-on-5 stints with points and possessions) and its checks."""

from pathlib import Path
from typing import Any

import pandas as pd

from eurohoops.parse.possessions import possession_ends
from eurohoops.parse.stints import box_facts, events
from eurohoops.parse.stints_mart import (
    GAME_CHECKS,
    build_game_stints,
    build_stints_mart,
    digest,
    game_stints,
    mart_report,
)
from tests.test_stints import game, write_game

TEAMS = ("AAA", "BBB")


def full_box(box: dict[str, Any], home_fga: int = 2, away_fga: int = 1) -> dict[str, Any]:
    """The stint test box plus the totals the possession check needs (made shots only)."""
    for side, fga in zip(box["Stats"], (home_fga, away_fga), strict=True):
        side["totr"] |= {
            "FieldGoalsAttempted2": fga,
            "FieldGoalsAttempted3": 0,
            "FreeThrowsAttempted": 1 if side is box["Stats"][0] else 0,
            "OffensiveRebounds": 0,
            "DefensiveRebounds": 0,
            "Turnovers": 0,
        }
    return box


def test_a_substitution_by_either_team_closes_the_stint() -> None:
    pbp, box = game()
    facts = box_facts(box)
    game_events = events(pbp)
    stints = build_game_stints(
        game_events, facts.starters, facts.overtimes, TEAMS, possession_ends(game_events, TEAMS)
    )
    q1 = [s for s in stints if s.period == 1]
    assert [(s.start, s.end) for s in q1] == [(0, 300), (300, 600)]
    assert [s.points for s in q1] == [(2, 0), (2, 3)]
    assert [s.possessions for s in q1] == [(1, 0), (1, 1)]
    assert "A1" in q1[0].players[0] and "A6" in q1[1].players[0]
    assert all(len(five) == 5 for s in stints for five in s.players)
    assert [s.period for s in stints] == [1, 1, 2, 3, 4]
    assert sum(s.points[0] for s in stints) == 5  # the Q3 free throw counts too
    assert sum(s.possessions[0] for s in stints) == 3  # 2 field goals + 1 one-shot trip


def test_game_checks_add_possessions_to_the_four_spike_checks() -> None:
    pbp, box = game()
    good = game_stints("E2024_1", 2024, pbp, full_box(box), TEAMS)
    assert all(not why for why in good.reasons.values())
    assert set(good.reasons) == set(GAME_CHECKS)
    pbp, box = game()
    off = game_stints("E2024_2", 2024, pbp, full_box(box, home_fga=12), TEAMS)
    assert off.reasons["possessions"] == ["AAA: 3 counted, 12.42 box formula"]
    pbp, box = game()
    other = game_stints("E2024_3", 2024, pbp, full_box(box), ("AAA", "CCC"))
    assert other.reasons["possessions"] == ["CCC: not in the box score"]
    empty = game_stints(
        "E2024_4", 2024, {}, {"ByQuarter": [], "Stats": [{"PlayersStats": []}] * 2}, TEAMS
    )
    assert empty.stints == []
    assert all(
        why == ["box score has no players (API placeholder)"] for why in empty.reasons.values()
    )


def games_frame(rows: list[tuple[str, int, int, int, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [r[0] for r in rows],
            "season": [r[1] for r in rows],
            "game_code": [r[2] for r in rows],
            "home": "AAA",
            "away": "BBB",
            "home_score": [r[3] for r in rows],
            "away_score": [r[4] for r in rows],
            "played": True,
            "forfeit": False,
        }
    )


def test_mart_lists_failures_and_is_reproducible(tmp_path: Path) -> None:
    pbp, box = game()
    write_game(tmp_path, 2015, 1, pbp, full_box(box))
    pbp, box = game(drop_in=True)
    write_game(tmp_path, 2015, 2, pbp, full_box(box))
    write_game(tmp_path, 2012, 3, pbp, full_box(box))
    pbp, box = game()
    write_game(tmp_path, 2010, 4, pbp, full_box(box))  # before 2011-12: not in the mart
    games = games_frame(
        [
            ("E2015_1", 2015, 1, 5, 3),
            ("E2015_2", 2015, 2, 5, 3),
            ("E2012_3", 2012, 3, 5, 3),
            ("E2010_4", 2010, 4, 5, 3),
            ("E2015_9", 2015, 9, 70, 60),  # not cached
        ]
    )
    mart = build_stints_mart(tmp_path, games)
    assert mart.not_cached == ["E2015_9"]
    assert mart.checks["game_id"].tolist() == ["E2015_1", "E2015_2", "E2012_3"]
    assert mart.checks["passed"].tolist() == [True, False, False]
    assert "five_on_court" in mart.checks.loc[1, "reasons"]
    team_games = pd.DataFrame(
        {
            "game_id": ["E2015_1", "E2015_1", "E2012_3", "E2012_3"],
            "team": ["AAA", "BBB", "AAA", "BBB"],
            "poss_raw": [2.44, 1.0, 2.44, 9.0],
            "fta": [1, 1, 2, 2],
        }
    )
    report = mart_report(mart, games, team_games)
    assert (report["pass_rate_2011_2014"], report["pass_rate_2015_on"]) == (0.0, 0.5)
    assert report["passing_games_where_stint_points_miss_the_final"] == []
    assert report["seasons"]["2015"]["failures_by_check"]["five_on_court"] == 1
    assert report["pbp_vs_box_possessions"]["2011_2014"]["within_2_share"] == 0.5
    assert report["pbp_vs_box_possessions"]["2015_on"]["mean_gap_pbp_minus_box"] == 0.28
    # Two team-games, gaps summing to 0.56 over 2 FTA: the weight rises by 0.28.
    assert report["pbp_vs_box_possessions"]["2015_on"]["ft_weight_matching_pbp"] == 0.70
    assert report["pbp_vs_box_possessions"]["all"]["team_games"] == 4
    moved = games.assign(home_score=games["home_score"] + 1)
    missed = mart_report(mart, moved, team_games)["passing_games_where_stint_points_miss_the_final"]
    assert missed == ["E2015_1"]
    again = build_stints_mart(tmp_path, games)
    assert digest(again.stints) == digest(mart.stints)
    assert report["table_sha256"] == mart_report(again, games, team_games)["table_sha256"]
