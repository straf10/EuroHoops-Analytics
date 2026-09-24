import math
from pathlib import Path

import pandas as pd
import pytest

from eurohoops.eval.backtest import TunedModel
from eurohoops.eval.scorecard import build_scorecard
from eurohoops.predict import LOG_COLUMNS

HEADER = ",".join(LOG_COLUMNS)
LOG = f"""{HEADER}
G1,2026,1,RS,2026-10-01T18:00:00Z,AAA,BBB,0.8,5.0,elo,v1,2026-10-01T08:00:00Z
G2,2026,1,RS,2026-10-01T18:00:00Z,CCC,DDD,0.4,-3.0,elo,v1,2026-10-01T08:00:00Z
G3,2026,1,RS,2026-10-01T18:00:00Z,EEE,FFF,0.9,9.0,elo,v1,2026-10-01T18:00:00Z
G2,2026,1,RS,2026-10-01T18:00:00Z,CCC,DDD,0.1,-9.0,elo,v2,2026-10-01T09:00:00Z
G4,2026,2,RS,2026-10-08T18:00:00Z,AAA,CCC,0.7,4.0,elo,v1,2026-10-01T08:00:00Z
"""


def results(played: bool) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["G1", "G2", "G3", "G4"],
            "home_score": pd.array([90, 82, 70, None] if played else [None] * 4, dtype="Int64"),
            "away_score": pd.array([80, 80, 60, None] if played else [None] * 4, dtype="Int64"),
            "played": [played, played, played, False],
            "neutral": [False] * 4,
        }
    )


def test_hand_computed_metrics_and_late_rows_excluded(tmp_path: Path, tuned: TunedModel) -> None:
    log = tmp_path / "log.csv"
    log.write_text(LOG)
    card = build_scorecard(log, results(played=True), tuned)
    # G3 is late (stamped at tip-off); G2 counts its earliest (v1) row; G4 is not played yet.
    # G1: p=.8, home +10 (exp 5).  G2: p=.4, home +2 (exp -3).
    assert card["rows_in_log"] == 5
    assert card["rows_excluded_late"] == 1
    elo = card["elo"]
    assert elo["n"] == 2
    assert elo["log_loss"] == pytest.approx(-(math.log(0.8) + math.log(0.4)) / 2, abs=1e-6)
    assert elo["brier"] == pytest.approx((0.2**2 + 0.6**2) / 2, abs=1e-6)
    assert elo["accuracy"] == 0.5
    assert elo["margin_mae"] == pytest.approx(5.0)
    # B0: p=.6, margin 3 for both home wins.
    b0 = card["b0"]
    assert b0["log_loss"] == pytest.approx(-math.log(0.6), abs=1e-6)
    assert b0["brier"] == pytest.approx(0.16, abs=1e-6)
    assert b0["accuracy"] == 1.0
    assert b0["margin_mae"] == pytest.approx(4.0)


def test_zero_completed_games(tmp_path: Path, tuned: TunedModel) -> None:
    log = tmp_path / "log.csv"
    log.write_text(LOG)
    card = build_scorecard(log, results(played=False), tuned)
    assert card["elo"] == {
        "n": 0,
        "log_loss": None,
        "brier": None,
        "accuracy": None,
        "margin_mae": None,
    }


def test_missing_log_is_an_empty_scorecard(tmp_path: Path, tuned: TunedModel) -> None:
    card = build_scorecard(tmp_path / "missing.csv", results(played=True), tuned)
    assert card["rows_in_log"] == 0
    assert card["b0"]["n"] == 0
