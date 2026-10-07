"""Live M5 shadow log (EuroLeague only): the frozen choice, the gate, the append-only log, the
walk-forward (no oracle roster, nothing at or after the round's cutoff) and its scorecard section.

The league is ``tests/m5_synthetic.py``'s; its last season (2022) is the "live" season: from round
9 on it is unplayed (no scores, no box lines), as a real season is before its rounds are played.
"""

import csv
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from functools import cache
from pathlib import Path

import pandas as pd
import pytest

from eurohoops.eval.m5_backtest import Choice, M5Inputs, PlayerPartFn, predict_games
from eurohoops.eval.scorecard import m1_section
from eurohoops.live_m5 import (
    M5_LOG_COLUMNS,
    LiveM5,
    live_spec,
    load_m5,
    predict_upcoming_m5,
)
from eurohoops.predict import LatePredictionError
from tests.m5_synthetic import build_inputs, fake_player_part, small_spec

SPEC = small_spec()
SEASON = 2022
FIRST_UNPLAYED_ROUND = 9  # tips off 2022-10-29T18:00Z (day 28 of the synthetic calendar)
NOW = datetime(2022, 10, 29, 8, 0, tzinfo=UTC)  # 10 h before round 9
WINDOW = timedelta(hours=36)  # round 10 is two days later: out
CHOICE = Choice("proj_hc", None, "core", 180.0, 10.0, None, False, "total_m1_rest")
ROUND_9_GAMES = 4


@cache
def full() -> M5Inputs:
    """The synthetic league with every game played: the oracle of the live season's future."""
    return build_inputs(0)


@cache
def live() -> M5Inputs:
    """The league as of ``NOW``: round 9 onwards unplayed, with no scores and no box lines."""
    inputs = full()
    games = inputs.games.copy()
    future = ((games["season"] == SEASON) & (games["round"] >= FIRST_UNPLAYED_ROUND)).to_numpy()
    games.loc[future, "played"] = False
    games.loc[future, ["home_score", "away_score"]] = pd.NA
    gone = set(games.loc[future, "game_id"])
    return replace(
        inputs,
        games=games,
        team_games=inputs.team_games[~inputs.team_games["game_id"].isin(gone)],
        player_games=inputs.player_games[~inputs.player_games["game_id"].isin(gone)],
    )


@pytest.fixture
def model() -> LiveM5:
    return LiveM5(CHOICE, "0.0.0+m5.test", gate_passed=True)


def builder(
    inputs: M5Inputs | None = None, calls: list[int] | None = None
) -> tuple[M5Inputs, PlayerPartFn]:
    if calls is not None:
        calls.append(1)
    return (live() if inputs is None else inputs), fake_player_part


def run(
    model: LiveM5,
    log: Path,
    *,
    inputs: M5Inputs | None = None,
    calls: list[int] | None = None,
    at: datetime = NOW,
) -> int:
    chosen = live() if inputs is None else inputs
    return predict_upcoming_m5(
        chosen.games,
        model,
        lambda: builder(chosen, calls),
        log_path=log,
        season=SEASON,
        window=WINDOW,
        clock=lambda: at,
        spec=SPEC,
    )


def rows(log: Path) -> list[dict[str, str]]:
    with log.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def forecast_columns(log: Path) -> list[dict[str, str]]:
    """The logged forecast, without the stamp (a rerun is stamped later)."""
    return [{k: v for k, v in r.items() if k != "predicted_at_utc"} for r in rows(log)]


def test_load_m5_reads_the_frozen_choice_and_gate(tmp_path: Path) -> None:
    assert load_m5(tmp_path / "none.json") is None
    report = {
        "model_version": "v1",
        "chosen": {"choice": CHOICE.__dict__},
        "gate": {"passed": True},
    }
    path = tmp_path / "backtest_m5.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    assert load_m5(path) == LiveM5(CHOICE, "v1", gate_passed=True)
    for gate in ({"passed": False}, {"passed": None}, {}):
        path.write_text(json.dumps({**report, "gate": gate}), encoding="utf-8")
        loaded = load_m5(path)
        assert loaded is not None and not loaded.gate_passed
    path.write_text(json.dumps({k: v for k, v in report.items() if k != "gate"}), encoding="utf-8")
    loaded = load_m5(path)
    assert loaded is not None and not loaded.gate_passed


def test_the_committed_report_loads() -> None:
    from eurohoops.config import M5  # noqa: PLC0415

    loaded = load_m5(M5.report)
    assert loaded is not None and loaded.choice.shares == "proj_hc"
    assert loaded.choice.form == "core" and not loaded.choice.platt


def test_a_failed_gate_logs_nothing_and_builds_nothing(model: LiveM5, tmp_path: Path) -> None:
    calls: list[int] = []
    log = tmp_path / "m5.csv"
    assert run(replace(model, gate_passed=False), log, calls=calls) == 0
    assert not log.exists() and not calls


def test_upcoming_games_are_logged_once_append_only(model: LiveM5, tmp_path: Path) -> None:
    calls: list[int] = []
    log = tmp_path / "m5.csv"
    assert run(model, log, calls=calls) == ROUND_9_GAMES
    logged = rows(log)
    assert tuple(logged[0]) == M5_LOG_COLUMNS
    assert {r["round"] for r in logged} == {str(FIRST_UNPLAYED_ROUND)}
    for row in logged:
        assert row["model"] == "m5" and row["model_version"] == model.version
        assert 0 < float(row["p_home"]) < 1
        assert float(row["exp_total"]) > 100 and float(row["margin_sigma"]) > 0
        assert float(row["total_sigma"]) > 0
        assert row["predicted_at_utc"] < row["tipoff_utc"]
    first = log.read_bytes()
    assert run(model, log, calls=calls) == 0  # same clock: nothing new, nothing rebuilt
    assert log.read_bytes() == first and len(calls) == 1
    # A later run adds only the games the window now reaches (round 10); round 9 is untouched.
    assert run(model, log, calls=calls, at=NOW + timedelta(days=2)) == ROUND_9_GAMES
    assert log.read_bytes().startswith(first)
    assert {r["round"] for r in rows(log)} == {"9", "10"}


def test_a_new_model_version_is_logged_beside_the_old_one(model: LiveM5, tmp_path: Path) -> None:
    log = tmp_path / "m5.csv"
    run(model, log)
    assert run(replace(model, version="0.0.0+m5.next"), log) == ROUND_9_GAMES
    assert len(rows(log)) == 2 * ROUND_9_GAMES


def test_the_log_is_the_backtests_walk_forward(model: LiveM5, tmp_path: Path) -> None:
    log = tmp_path / "m5.csv"
    run(model, log)
    frame = predict_games(
        live(), spec=live_spec(SPEC, SEASON), player_part=fake_player_part, choice=CHOICE
    ).set_index("game_id")
    for row in rows(log):
        f = frame.loc[row["game_id"]]
        assert float(row["p_home"]) == pytest.approx(f["p_home"], abs=5e-5)
        assert float(row["exp_margin"]) == pytest.approx(f["exp_margin"], abs=0.005)
        assert float(row["exp_total"]) == pytest.approx(f["exp_total"], abs=0.005)
        assert float(row["margin_sigma"]) == pytest.approx(f["margin_sigma"], abs=5e-5)


def test_no_oracle_roster_and_nothing_at_or_after_the_cutoff(model: LiveM5, tmp_path: Path) -> None:
    """The oracle league has the round's own box lines and every later result; the live forecast
    must equal the one made from the league as of the round's first tip-off."""
    cutoff_free = tmp_path / "live.csv"
    oracle = tmp_path / "oracle.csv"
    run(model, cutoff_free)
    # The same unplayed games (still upcoming), but with every later box line and result present.
    future = full()
    peeking = replace(live(), player_games=future.player_games, team_games=future.team_games)
    run(model, oracle, inputs=peeking)
    assert forecast_columns(oracle) == forecast_columns(cutoff_free)


def test_later_results_do_not_move_the_forecast(model: LiveM5, tmp_path: Path) -> None:
    """Rounds after the upcoming one played with wild scores: the round's forecast is unchanged."""
    base, other = tmp_path / "base.csv", tmp_path / "other.csv"
    run(model, base)
    inputs = live()
    games = inputs.games.copy()
    later = ((games["season"] == SEASON) & (games["round"] > FIRST_UNPLAYED_ROUND)).to_numpy()
    games.loc[later, "played"] = True
    games.loc[later, "home_score"] = 150
    games.loc[later, "away_score"] = 60
    full_later = full().team_games
    played_ids = set(games.loc[later, "game_id"])
    shaken = replace(
        inputs,
        games=games,
        team_games=pd.concat(
            [inputs.team_games, full_later[full_later["game_id"].isin(played_ids)]]
        ),
        player_games=pd.concat(
            [
                inputs.player_games,
                full().player_games[full().player_games["game_id"].isin(played_ids)],
            ]
        ),
    )
    run(model, other, inputs=shaken)
    assert forecast_columns(other) == forecast_columns(base)


def test_late_stamps_are_refused_and_nothing_is_written(model: LiveM5, tmp_path: Path) -> None:
    stamps = iter([NOW, NOW + timedelta(hours=11)])
    log = tmp_path / "m5.csv"
    with pytest.raises(LatePredictionError):
        predict_upcoming_m5(
            live().games,
            model,
            builder,
            log_path=log,
            season=SEASON,
            window=WINDOW,
            clock=lambda: next(stamps),
            spec=SPEC,
        )
    assert not log.exists()


def test_the_live_season_joins_the_frame_but_not_the_tuning_seasons() -> None:
    spec = live_spec(small_spec(), 2023)
    assert spec.test == (2021, 2022, 2023) and spec.tuning == small_spec().tuning
    assert live_spec(small_spec(), 2022) == small_spec()


def test_the_scorecard_section_scores_the_m5_log(tmp_path: Path) -> None:
    games = full().games
    played = games[games["season"] == SEASON].head(30)
    log = pd.DataFrame(
        {
            "game_id": played["game_id"],
            "season": played["season"],
            "round": played["round"],
            "phase": played["phase"],
            "tipoff_utc": played["tipoff_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "home": played["home"],
            "away": played["away"],
            "p_home": 0.6,
            "exp_margin": 3.0,
            "exp_total": 160.0,
            "margin_sigma": 11.0,
            "margin_df": 7.0,
            "total_sigma": 15.0,
            "model": "m5",
            "model_version": "v",
            "predicted_at_utc": (played["tipoff_utc"] - pd.Timedelta(hours=2)).dt.strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
        }
    )
    path = tmp_path / "m5.csv"
    log.to_csv(path, index=False)
    elo = log[["game_id"]].iloc[:10].assign(p_home=0.55)
    section = m1_section(path, games, elo, window=20, name="m5")
    assert section["m5"]["n"] == 30 and "m1" not in section
    assert section["same_games_as_elo"]["n"] == 10
    assert section["same_games_as_elo"]["m5_log_loss"] is not None
    assert all("m5" in point for point in section["rolling"]["series"])
