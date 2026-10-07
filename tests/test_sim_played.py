"""Played results for M7: regulation scores of overtime games, forfeits counted as results."""

import gzip
import json
from pathlib import Path

import pandas as pd

from eurohoops.sim.played import regulation_score, regulation_scores, season_results
from eurohoops.standings import Result


def _box(home: dict[str, int], away: dict[str, int]) -> dict[str, object]:
    return {"EndOfQuarter": [{"Team": "H", **home}, {"Team": "A", **away}]}


REGULATION = _box(
    {"Quarter1": 20, "Quarter2": 40, "Quarter3": 60, "Quarter4": 80},
    {"Quarter1": 18, "Quarter2": 39, "Quarter3": 61, "Quarter4": 77},
)
OVERTIME = _box(
    {"Quarter1": 12, "Quarter2": 36, "Quarter3": 58, "Quarter4": 82, "Extra1": 95},
    {"Quarter1": 20, "Quarter2": 35, "Quarter3": 60, "Quarter4": 82, "Extra1": 92},
)


def test_regulation_score_only_for_overtime_games() -> None:
    assert regulation_score(REGULATION) is None
    assert regulation_score(OVERTIME) == (82, 82)
    # an "Extra" key present but empty (no overtime played) is not an overtime
    no_extra = _box({"Quarter4": 80, "Extra1": 0}, {"Quarter4": 77, "Extra1": 0})
    assert regulation_score(no_extra) is None


def _games() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["E2022_2", "E2022_1", "E2022_3", "E2022_4"],
            "season": [2022] * 4,
            "game_code": [2, 1, 3, 4],
            "tipoff_utc": pd.to_datetime(
                ["2022-10-07 18:00", "2022-10-06 18:00", "2022-10-08 18:00", "2022-10-09 18:00"],
                utc=True,
            ),
            "home": ["MCO", "PAN", "OLY", "BAR"],
            "away": ["IST", "MAD", "BAR", "OLY"],
            "home_score": [95, 68, 20, 0],
            "away_score": [92, 71, 0, 0],
            "played": [True, True, True, False],
            "forfeit": [False, False, True, False],
        }
    )


def test_regulation_scores_read_the_cached_boxes(tmp_path: Path) -> None:
    box_dir = tmp_path / "boxscore" / "E2022"
    box_dir.mkdir(parents=True)
    for code, box in ((1, REGULATION), (2, OVERTIME), (3, REGULATION)):
        (box_dir / f"{code}.json.gz").write_bytes(gzip.compress(json.dumps(box).encode()))
    # game 4 is unplayed and has no box: it must not be read
    assert regulation_scores(tmp_path, _games()) == {"E2022_2": (82, 82)}


def test_season_results_in_tipoff_order_with_forfeits() -> None:
    results = season_results(_games(), {"E2022_2": (82, 82)})
    assert results == [
        Result("PAN", "MAD", 68, 71),
        Result("MCO", "IST", 95, 92, (82, 82)),
        Result("OLY", "BAR", 20, 0),  # the forfeit counts as the awarded win
    ]
    assert season_results(_games())[1].regulation is None
