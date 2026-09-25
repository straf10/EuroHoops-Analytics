import gzip
import json
from pathlib import Path
from typing import Any

import pytest

from eurohoops.parse.stints import (
    CHECKS,
    SAMPLE_SEASONS,
    SAMPLE_SIZE,
    EmptyGameError,
    box_facts,
    build_stints,
    events,
    overtimes,
    sample_games,
    validate,
    validate_sample,
)

A = [f"A{i}" for i in range(1, 7)]  # A6 comes off the bench
B = [f"B{i}" for i in range(1, 6)]


def row(kind: str, team: str = "", player: str = "", clock: str = "", minute: int = 1) -> dict:
    return {
        "PLAYTYPE": kind.ljust(10) if kind == "2FGM" else kind,  # the API pads some types
        "CODETEAM": team.ljust(10),
        "PLAYER_ID": player.ljust(10),
        "MARKERTIME": clock,
        "MINUTE": minute,
    }


def game(drop_in: bool = False, overtime: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    """AAA 5-3 BBB: A1 -> A6 at 05:00 of Q1; A2 scores 2+2 (one before, one after), B1 hits a 3,
    A6 a free throw in Q3. With ``overtime``, a 5-minute OT in which A3 scores 2."""
    q1 = [
        row("BP"),
        row("2FGM", "AAA", "A2", "08:00"),
        row("OUT", "AAA", "A1", "05:00"),
        *([] if drop_in else [row("IN", "AAA", "A6", "05:00")]),
        row("3FGM", "BBB", "B1", "04:00"),
        row("LAYUPMD", "AAA", "A2", "03:00"),
        row("EP"),
    ]
    pbp: dict[str, Any] = {
        "FirstQuarter": q1,
        "SecondQuarter": [row("BP")],
        "ThirdQuarter": [row("FTM", "AAA", "A6", "02:00", 28)],
        "ForthQuarter": [row("BP")],
        "ExtraTime": [row("BP", minute=41), row("DUNK", "AAA", "A3", "01:00", 45)]
        if overtime
        else None,
    }
    total = 2700 if overtime else 2400

    def players(team: str, ids: list[str], starters: int) -> list[dict[str, Any]]:
        seconds = {"A1": 300, "A6": total - 300}
        return [
            {
                "Player_ID": pid.ljust(10),
                "Team": team,
                "IsStarter": int(i < starters),
                "Minutes": "{:02d}:{:02d}".format(*divmod(seconds.get(pid, total), 60)),
            }
            for i, pid in enumerate(ids)
        ]

    extra = {"Extra1": 2 if overtime else 0}  # the phantom Extra1 of some 2015-16 box scores
    box = {
        "ByQuarter": [{"Team": "A", **extra}, {"Team": "B", "Extra1": 0}],
        "Stats": [
            {"PlayersStats": players("AAA", A, 5), "totr": {"Points": 7 if overtime else 5}},
            {"PlayersStats": players("BBB", B, 5), "totr": {"Points": 3}},
        ],
    }
    return pbp, box


def stints_of(pbp: dict[str, Any], box: dict[str, Any]) -> tuple[list, Any]:
    facts = box_facts(box)
    return build_stints(events(pbp), facts.starters, facts.overtimes), facts


def test_stints_split_at_substitutions_and_period_ends() -> None:
    stints, facts = stints_of(*game())
    aaa = [s for s in stints if s.team == "AAA"]
    assert [(s.period, s.start, s.end) for s in aaa] == [
        (1, 0, 300),
        (1, 300, 600),
        (2, 600, 1200),
        (3, 1200, 1800),
        (4, 1800, 2400),
    ]
    assert aaa[0].players == frozenset(A[:5])
    assert aaa[1].players == frozenset([*A[1:5], "A6"])
    assert [(s.points_for, s.points_against) for s in aaa] == [
        (2, 0),
        (2, 3),
        (0, 0),
        (1, 0),
        (0, 0),
    ]
    assert len([s for s in stints if s.team == "BBB"]) == 4  # no BBB substitution
    assert validate(stints, facts) == {check: [] for check in CHECKS}


def test_a_missing_substitution_row_is_caught() -> None:
    stints, facts = stints_of(*game(drop_in=True))
    reasons = validate(stints, facts)
    assert reasons["five_on_court"] == ["AAA: 4 stints (2100 s) with [4] players"]
    assert reasons["minutes"] == ["AAA: 1 players off by > 60 s (worst A6 -2100 s)"]
    assert reasons["points"] == reasons["seconds"] == []


def test_points_must_add_up_to_the_final_score() -> None:
    pbp, box = game()
    box["Stats"][1]["totr"]["Points"] = 5
    assert validate(*stints_of(pbp, box))["points"] == ["BBB: 3 from stints, 5 final"]


def test_overtime_adds_five_minutes() -> None:
    stints, facts = stints_of(*game(overtime=True))
    assert facts.overtimes == 1
    last = [s for s in stints if s.team == "AAA"][-1]
    assert (last.period, last.start, last.end, last.points_for) == (5, 2400, 2700, 2)
    assert validate(stints, facts) == {check: [] for check in CHECKS}


def test_overtime_count_ignores_a_scoreless_placeholder() -> None:
    assert overtimes([{"Extra1": 0}, {"Extra1": 0}]) == 0
    assert overtimes([{"Extra1": 0, "Extra2": None}, {"Extra1": 3}]) == 1
    assert overtimes([{"Extra1": 9, "Extra2": 4}, {"Extra1": 9, "Extra2": 6}]) == 2
    assert overtimes([{"Quarter4": 20}, {"Quarter4": 22}]) == 0


def test_events_carry_period_and_game_clock() -> None:
    pbp, _ = game(overtime=True)
    evs = events(pbp)
    assert [(e.period, e.elapsed) for e in evs if e.kind == "OUT"] == [(1, 300)]
    assert [(e.period, e.elapsed) for e in evs if e.kind == "FTM"] == [(3, 1680)]
    assert [(e.period, e.elapsed) for e in evs if e.kind == "DUNK"] == [(5, 2640)]
    assert all(e.elapsed is None for e in evs if e.kind in {"BP", "EP"})
    assert {e.kind for e in evs} >= {"2FGM"}  # padding stripped


def test_an_empty_box_score_is_refused() -> None:
    with pytest.raises(EmptyGameError):
        box_facts({"ByQuarter": [], "Stats": [{"PlayersStats": []}, {"PlayersStats": []}]})


def write_game(raw: Path, season: int, code: int, pbp: Any, box: Any) -> None:
    for endpoint, payload in (("playbyplay", pbp), ("boxscore", box)):
        path = raw / endpoint / f"E{season}" / f"{code}.json.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(gzip.compress(json.dumps(payload).encode(), mtime=0))


def test_sample_is_seeded_and_spread_over_seasons(tmp_path: Path) -> None:
    for season in SAMPLE_SEASONS:
        for code in range(1, 21):
            for endpoint in ("playbyplay", "boxscore"):
                path = tmp_path / endpoint / f"E{season}" / f"{code}.json.gz"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
    (tmp_path / "boxscore" / "E2015" / "1.json.gz").unlink()  # PBP without a box: not eligible
    sample = sample_games(tmp_path)
    assert len(sample) == SAMPLE_SIZE == len(set(sample))
    per_season = [sum(s == season for s, _ in sample) for season in SAMPLE_SEASONS]
    assert min(per_season) >= SAMPLE_SIZE // len(SAMPLE_SEASONS)
    assert max(per_season) - min(per_season) <= 1
    assert (2015, 1) not in sample
    assert sample_games(tmp_path) == sample


def test_validate_sample_is_reproducible_and_lists_failures(tmp_path: Path) -> None:
    write_game(tmp_path, 2020, 1, *game())
    write_game(tmp_path, 2021, 7, *game(drop_in=True))
    write_game(tmp_path, 2022, 3, {}, {"ByQuarter": [], "Stats": [{"PlayersStats": []}] * 2})
    report = validate_sample(tmp_path)
    assert report["games"] == 3
    assert sorted(report["sample"]) == ["E2020_1", "E2021_7", "E2022_3"]
    assert report["pass_rate"] == {
        "five_on_court": 0.3333,
        "seconds": 0.6667,
        "minutes": 0.3333,
        "points": 0.6667,
    }
    assert report["all_checks_pass_rate"] == 0.3333
    assert set(report["failing_games"]) == {"E2021_7", "E2022_3"}
    assert report["failing_games"]["E2022_3"]["points"] == ["box score has no players"]
    assert validate_sample(tmp_path) == report
