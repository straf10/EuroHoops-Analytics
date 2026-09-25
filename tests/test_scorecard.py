import math
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eurohoops.eval.backtest import TunedModel
from eurohoops.eval.metrics import crps_normal
from eurohoops.eval.scorecard import build_scorecard, public_at, rolling_log_loss
from eurohoops.odds import ODDS_COLUMNS
from eurohoops.predict import LOG_COLUMNS

HEADER = ",".join(LOG_COLUMNS)
NOW = datetime(2026, 10, 2, 12, tzinfo=UTC)
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
            "season": [2026] * 4,
            "tipoff_utc": pd.to_datetime(["2026-10-01T18:00:00Z"] * 3 + ["2026-10-08T18:00:00Z"]),
            "home_score": pd.array([90, 82, 70, None] if played else [None] * 4, dtype="Int64"),
            "away_score": pd.array([80, 80, 60, None] if played else [None] * 4, dtype="Int64"),
            "played": [played, played, played, False],
            "forfeit": [False] * 4,
            "neutral": [False] * 4,
        }
    )


def test_hand_computed_metrics_and_late_rows_excluded(tmp_path: Path, tuned: TunedModel) -> None:
    log = tmp_path / "log.csv"
    log.write_text(LOG)
    card = build_scorecard(log, results(played=True), tuned, NOW)
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
    assert elo["ece"] == pytest.approx((0.2 + 0.6) / 2)  # bins 8 and 4, both home wins
    # B0: p=.6, margin 3 for both home wins.
    b0 = card["b0"]
    assert b0["log_loss"] == pytest.approx(-math.log(0.6), abs=1e-6)
    assert b0["brier"] == pytest.approx(0.16, abs=1e-6)
    assert b0["accuracy"] == 1.0
    assert b0["margin_mae"] == pytest.approx(4.0)


def test_zero_completed_games(tmp_path: Path, tuned: TunedModel) -> None:
    log = tmp_path / "log.csv"
    log.write_text(LOG)
    card = build_scorecard(log, results(played=False), tuned, NOW)
    reliability = card["elo"].pop("reliability")
    assert card["elo"] == {
        "n": 0,
        "log_loss": None,
        "brier": None,
        "accuracy": None,
        "margin_mae": None,
        "ece": None,
        "margin_crps": None,
    }
    assert [b["n"] for b in reliability] == [0] * 10


def test_missing_log_is_an_empty_scorecard(tmp_path: Path, tuned: TunedModel) -> None:
    card = build_scorecard(tmp_path / "missing.csv", results(played=True), tuned, NOW)
    assert card["rows_in_log"] == 0
    assert card["b0"]["n"] == 0


def test_forfeits_are_not_scored(tmp_path: Path, tuned: TunedModel) -> None:
    log = tmp_path / "log.csv"
    log.write_text(LOG)
    games = results(played=True)
    games.loc[games["game_id"] == "G2", ["home_score", "away_score", "forfeit"]] = [20, 0, True]
    card = build_scorecard(log, games, tuned, NOW)
    assert card["elo"]["n"] == 1
    assert card["elo"]["log_loss"] == pytest.approx(-math.log(0.8), abs=1e-6)


def test_rows_pushed_after_tipoff_leave_the_headline(tmp_path: Path, tuned: TunedModel) -> None:
    log = tmp_path / "log.csv"
    log.write_text(LOG)
    pushes = (datetime(2026, 10, 1, 18, 30, tzinfo=UTC),)  # after G1/G2 tip-off, before G4
    card = build_scorecard(log, results(played=True), tuned, NOW, manual_pushes=pushes)
    assert card["rows_not_provable"] == 3  # G1, both G2 rows; G3 is late, not unprovable
    assert card["games_not_provable"] == ["G1", "G2"]
    assert card["elo"]["n"] == 0
    assert card["all_rows"]["elo"]["n"] == 2
    assert card["all_rows"]["elo"]["log_loss"] == pytest.approx(
        -(math.log(0.8) + math.log(0.4)) / 2, abs=1e-6
    )


def test_public_at_takes_the_first_push_at_or_after_the_stamp() -> None:
    stamps = pd.to_datetime(
        pd.Series(["2026-10-01T08:00:00Z", "2026-10-01T10:00:00Z", "2026-10-01T13:00:00Z"]),
        utc=True,
    )
    pushes = (datetime(2026, 10, 1, 12, tzinfo=UTC), datetime(2026, 10, 1, 10, tzinfo=UTC))
    assert public_at(stamps, pushes).dt.hour.tolist() == [10, 10, 13]


def test_totals_baseline_and_margin_crps(tmp_path: Path, tuned: TunedModel) -> None:
    log = tmp_path / "log.csv"
    log.write_text(LOG)
    model = replace(tuned, margin_sigma=12.0, totals_baseline={2026: 160.0})
    card = build_scorecard(log, results(played=True), model, NOW)
    # G1 total 170, G2 162 against a baseline of 160.
    assert card["totals"] == {"n": 2, "mae": 6.0}
    expected = crps_normal(np.array([5.0, -3.0]), 12.0, np.array([10.0, 2.0])).mean()
    assert card["elo"]["margin_crps"] == pytest.approx(expected, abs=1e-6)
    assert card["b0"]["margin_crps"] == pytest.approx(
        crps_normal(np.array([3.0, 3.0]), 12.0, np.array([10.0, 2.0])).mean(), abs=1e-6
    )
    assert build_scorecard(log, results(played=True), tuned, NOW)["totals"] == {"n": 0, "mae": None}


def scored_games(n: int, lost: tuple[int, ...] = (0,)) -> pd.DataFrame:
    """``n`` games in tip-off order: Elo says .8 and B0 .6 for home; home loses ``lost``."""
    margin = [-5.0 if i in lost else 5.0 for i in range(n)]
    return pd.DataFrame(
        {
            "game_id": [f"G{i:03d}" for i in range(n)],
            "tipoff_utc": pd.date_range("2026-10-01T18:00Z", periods=n, freq="h"),
            "p_home": 0.8,
            "p_b0": 0.6,
            "margin": margin,
        }
    )


@pytest.mark.parametrize(("n", "points"), [(49, 0), (50, 1), (51, 2)])
def test_rolling_window_starts_at_fifty_games(n: int, points: int) -> None:
    assert len(rolling_log_loss(scored_games(n))) == points


def test_rolling_window_values_and_the_oldest_game_dropping_out() -> None:
    series = rolling_log_loss(scored_games(51))
    # Point 1 covers games 0-49 (game 0 a home loss); point 2 covers games 1-50 (all wins).
    assert series[0]["n"] == 50
    assert series[0]["game_id"] == "G049"
    assert series[0]["elo"] == pytest.approx((-math.log(0.2) - 49 * math.log(0.8)) / 50, abs=1e-6)
    assert series[0]["b0"] == pytest.approx((-math.log(0.4) - 49 * math.log(0.6)) / 50, abs=1e-6)
    assert series[1]["elo"] == pytest.approx(-math.log(0.8), abs=1e-6)
    assert series[1]["tipoff_utc"] == "2026-10-03T20:00:00Z"


def log_of(games: pd.DataFrame, p_home: float) -> str:
    rows = [
        f"{g},2026,1,RS,{t:%Y-%m-%dT%H:%M:%SZ},AAA,BBB,{p_home},1.0,elo,v1,2026-09-30T08:00:00Z"
        for g, t in zip(games["game_id"], games["tipoff_utc"], strict=True)
    ]
    return "\n".join([HEADER, *rows]) + "\n"


def mart_games(n: int, played: bool = True) -> pd.DataFrame:
    tipoff = pd.date_range("2026-10-01T18:00Z", periods=n, freq="h")
    return pd.DataFrame(
        {
            "game_id": [f"G{i:03d}" for i in range(n)],
            "season": 2026,
            "tipoff_utc": tipoff,
            "home_score": pd.array([85 if played else None] * n, dtype="Int64"),
            "away_score": pd.array([80 if played else None] * n, dtype="Int64"),
            "played": played,
            "forfeit": False,
            "neutral": False,
        }
    )


@pytest.mark.parametrize(
    ("n", "p_home", "warned"), [(50, 0.3, True), (49, 0.3, False), (50, 0.9, False)]
)
def test_elo_worse_than_b0_warning_needs_fifty_games(
    tmp_path: Path, tuned: TunedModel, n: int, p_home: float, warned: bool
) -> None:
    games = mart_games(n)
    log = tmp_path / "log.csv"
    log.write_text(log_of(games, p_home))
    card = build_scorecard(log, games, tuned, datetime(2026, 10, 9, tzinfo=UTC))
    assert card["elo"]["n"] == n
    assert ("elo_worse_than_b0" in [w["code"] for w in card["warnings"]]) is warned


@pytest.mark.parametrize(("hours", "warned"), [(47, False), (49, True)])
def test_missing_result_48h_after_tipoff(
    tmp_path: Path, tuned: TunedModel, hours: int, warned: bool
) -> None:
    games = mart_games(1, played=False)
    log = tmp_path / "log.csv"
    log.write_text(log_of(games, 0.6))
    now = datetime(2026, 10, 1, 18, tzinfo=UTC) + timedelta(hours=hours)  # G000 tip-off
    card = build_scorecard(log, games, tuned, now)
    expected = {
        "code": "missing_result",
        "message": "1 logged game(s) have no result 48 h after tip-off",
        "games": ["G000"],
    }
    assert card["warnings"] == ([expected] if warned else [])


TIP = "2026-10-01T18:00:00Z"
ODDS = "\n".join(
    [
        ",".join(ODDS_COLUMNS),
        f"G1,2026,1,{TIP},AAA,BBB,{TIP},2026-10-01T08:00:00Z,5,0.7000,5,-4.5,5,160.5",
        f"G1,2026,1,{TIP},AAA,BBB,{TIP},2026-10-01T17:00:00Z,5,0.7500,5,-5.5,5,160.5",
        f"G1,2026,1,{TIP},AAA,BBB,{TIP},2026-10-01T19:00:00Z,5,0.9900,5,-9.5,5,160.5",
        f"G2,2026,1,{TIP},CCC,DDD,{TIP},2026-10-01T08:00:00Z,0,,2,1.5,2,158.5",
    ]
)


def test_market_column_uses_the_latest_pre_tipoff_snapshot(
    tmp_path: Path, tuned: TunedModel
) -> None:
    log, odds = tmp_path / "log.csv", tmp_path / "odds.csv"
    log.write_text(LOG)
    odds.write_text(ODDS)
    card = build_scorecard(log, results(played=True), tuned, NOW, odds_path=odds)
    # G1: the 17:00 snapshot (.75); the 19:00 one is after tip-off. G2 has no moneyline.
    assert card["market"] == {
        "n": 1,
        "log_loss": round(-math.log(0.75), 6),
        "elo_log_loss": round(-math.log(0.8), 6),
        "b0_log_loss": round(-math.log(0.6), 6),
    }
    assert card["elo"]["n"] == 2  # games without odds leave only the market column


def test_market_column_without_odds(tmp_path: Path, tuned: TunedModel) -> None:
    log = tmp_path / "log.csv"
    log.write_text(LOG)
    card = build_scorecard(log, results(played=True), tuned, NOW, odds_path=tmp_path / "none.csv")
    assert card["market"] == {"n": 0, "log_loss": None, "elo_log_loss": None, "b0_log_loss": None}
    assert build_scorecard(log, results(played=True), tuned, NOW)["market"] is None
