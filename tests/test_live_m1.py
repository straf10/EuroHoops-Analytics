"""E7: live M1 log (append-only, separate from Elo) and its scorecard section."""

import csv
import json
import logging
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from eurohoops.eval.m1_backtest import run_m1_backtest
from eurohoops.eval.scorecard import m1_section
from eurohoops.live_m1 import (
    M1_LOG_COLUMNS,
    LiveM1,
    load_m1,
    m1_forecasts,
    predict_upcoming_m1,
)
from eurohoops.predict import LatePredictionError
from tests.conftest import make_games, make_team_games
from tests.test_m1_backtest import SPEC

NOW = datetime(2026, 10, 1, 8, 0, tzinfo=UTC)  # 10 h before the 2026 opener
WINDOW = timedelta(hours=36)


@pytest.fixture(scope="module")
def history() -> tuple[pd.DataFrame, pd.DataFrame]:
    games = make_games({s: True for s in range(2016, 2026)} | {2026: False})
    return games, make_team_games(games)


@pytest.fixture(scope="module")
def model(
    history: tuple[pd.DataFrame, pd.DataFrame], tmp_path_factory: pytest.TempPathFactory
) -> LiveM1:
    path = tmp_path_factory.mktemp("m1") / "backtest_m1.json"
    report = run_m1_backtest(*history, replace(SPEC, test=(2022, 2023)))
    path.write_text(json.dumps(report), encoding="utf-8")
    loaded = load_m1(path)
    assert loaded is not None
    assert loaded.version == report["model_version"]
    return replace(loaded, gate_passed=True)


def run(history: tuple[pd.DataFrame, pd.DataFrame], model: LiveM1, log: Path, at: datetime) -> int:
    return predict_upcoming_m1(
        *history,
        model,
        log_path=log,
        season=2026,
        replay_from=2016,
        window=WINDOW,
        clock=lambda: at,
    )


def rows(log: Path) -> list[dict[str, str]]:
    with log.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_missing_report_means_no_live_m1(tmp_path: Path) -> None:
    assert load_m1(tmp_path / "none.json") is None


def test_dry_run_logs_upcoming_games_once(
    history: tuple[pd.DataFrame, pd.DataFrame], model: LiveM1, tmp_path: Path
) -> None:
    elo_log = tmp_path / "elo.csv"
    elo_log.write_bytes(b"game_id,model_version\nE2025_1,v\n")
    before = elo_log.read_bytes()
    log = tmp_path / "m1.csv"
    assert run(history, model, log, NOW) == 3
    logged = rows(log)
    assert tuple(logged[0]) == M1_LOG_COLUMNS
    assert [r["game_id"] for r in logged] == ["E2026_1", "E2026_2", "E2026_3"]
    for row in logged:
        assert row["model"] == "m1" and row["model_version"] == model.version
        assert 0 < float(row["p_home"]) < 1
        assert float(row["exp_total"]) > 100 and float(row["margin_sigma"]) > 0
        assert row["predicted_at_utc"] < row["tipoff_utc"]
    first = log.read_bytes()
    assert run(history, model, log, NOW) == 0  # same clock: nothing new
    assert log.read_bytes() == first
    assert elo_log.read_bytes() == before


def test_forecasts_match_the_walk_forward(
    history: tuple[pd.DataFrame, pd.DataFrame], model: LiveM1, tmp_path: Path
) -> None:
    log = tmp_path / "m1.csv"
    run(history, model, log, NOW)
    fc = m1_forecasts(*history, model, 2016)
    index = {g: i for i, g in enumerate(fc.games["game_id"])}
    for row in rows(log):
        i = index[row["game_id"]]
        assert float(row["exp_margin"]) == pytest.approx(fc.margin[i], abs=0.005)
        assert float(row["p_home"]) == pytest.approx(fc.p_home[i], abs=5e-5)


def test_a_failed_gate_logs_nothing(
    history: tuple[pd.DataFrame, pd.DataFrame], model: LiveM1, tmp_path: Path
) -> None:
    assert run(history, replace(model, gate_passed=False), tmp_path / "m1.csv", NOW) == 0
    assert not (tmp_path / "m1.csv").exists()


def test_late_stamps_are_refused(
    history: tuple[pd.DataFrame, pd.DataFrame], model: LiveM1, tmp_path: Path
) -> None:
    stamps = iter([NOW, NOW + timedelta(hours=11)])
    with pytest.raises(LatePredictionError):
        predict_upcoming_m1(
            *history,
            model,
            log_path=tmp_path / "m1.csv",
            season=2026,
            replay_from=2016,
            window=WINDOW,
            clock=lambda: next(stamps),
        )


def test_games_without_box_lines_have_no_forecast(
    history: tuple[pd.DataFrame, pd.DataFrame],
    model: LiveM1,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    games, team_games = history
    with caplog.at_level(logging.WARNING):
        added = predict_upcoming_m1(
            games,
            team_games.iloc[:0],
            model,
            log_path=tmp_path / "m1.csv",
            season=2026,
            replay_from=2016,
            window=WINDOW,
            clock=lambda: NOW,
        )
    assert added == 0
    assert "no forecast" in caplog.text


def test_scorecard_section_scores_the_m1_log(
    history: tuple[pd.DataFrame, pd.DataFrame], tmp_path: Path
) -> None:
    games, _ = history
    played = games[games["season"].isin([2024, 2025])].head(60)
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
            "margin_df": [7.0 if i % 2 else None for i in range(len(played))],
            "total_sigma": 15.0,
            "model": "m1",
            "model_version": "v",
            "predicted_at_utc": (played["tipoff_utc"] - pd.Timedelta(hours=2)).dt.strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
        }
    )
    late = log.iloc[:1].assign(predicted_at_utc="2030-01-01T00:00:00Z")
    path = tmp_path / "m1.csv"
    pd.concat([log, late]).to_csv(path, index=False)
    elo = log[["game_id"]].iloc[:10].assign(p_home=0.55)
    section = m1_section(path, games, elo, window=50)
    assert section["rows_in_log"] == 61 and section["rows_excluded_late"] == 1
    assert section["m1"]["n"] == 60
    assert section["m1"]["margin_crps"] > 0 and section["m1"]["totals_crps"] > 0
    assert section["same_games_as_elo"]["n"] == 10
    assert len(section["rolling"]["series"]) == 11
    empty = m1_section(tmp_path / "none.csv", games, elo)
    assert empty["m1"]["n"] == 0 and empty["same_games_as_elo"]["m1_log_loss"] is None
