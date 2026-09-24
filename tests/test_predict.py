import csv
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from eurohoops.eval.backtest import TunedModel
from eurohoops.predict import LOG_COLUMNS, LatePredictionError, predict_upcoming
from tests.conftest import make_games

NOW = datetime(2026, 10, 1, 8, 0, tzinfo=UTC)  # 10 h before round 1 (18:00, 18:15, 18:30)
WINDOW = timedelta(hours=36)


@pytest.fixture
def games() -> pd.DataFrame:
    return make_games({2024: True, 2025: True, 2026: False})


def fixed(t: datetime) -> Callable[[], datetime]:
    return lambda: t


def run(games: pd.DataFrame, model: TunedModel, log: Path, clock: Callable[[], datetime]) -> int:
    return predict_upcoming(
        games, model, log_path=log, season=2026, replay_from=2024, window=WINDOW, clock=clock
    )


def read_rows(log: Path) -> list[dict[str, str]]:
    with log.open(newline="") as fh:
        return list(csv.DictReader(fh))


def test_logs_upcoming_games_in_window(
    games: pd.DataFrame, tuned: TunedModel, tmp_path: Path
) -> None:
    log = tmp_path / "log.csv"
    assert run(games, tuned, log, fixed(NOW)) == 3
    rows = read_rows(log)
    assert tuple(rows[0]) == LOG_COLUMNS
    assert [r["game_id"] for r in rows] == ["E2026_1", "E2026_2", "E2026_3"]
    for row in rows:
        assert row["predicted_at_utc"] == "2026-10-01T08:00:00Z"
        assert row["predicted_at_utc"] < row["tipoff_utc"]
        assert 0 < float(row["p_home"]) < 1
        assert row["model_version"] == tuned.version()


def test_second_run_adds_zero_rows_and_never_changes_existing_bytes(
    games: pd.DataFrame, tuned: TunedModel, tmp_path: Path
) -> None:
    log = tmp_path / "log.csv"
    run(games, tuned, log, fixed(NOW))
    prefix = log.read_bytes()
    assert run(games, tuned, log, fixed(NOW + timedelta(hours=1))) == 0
    assert log.read_bytes() == prefix
    # A later run picks up the next round and only appends.
    assert run(games, tuned, log, fixed(NOW + timedelta(days=6))) == 3
    assert log.read_bytes().startswith(prefix)


def test_new_model_version_appends_without_touching_old_rows(
    games: pd.DataFrame, tuned: TunedModel, tmp_path: Path
) -> None:
    log = tmp_path / "log.csv"
    run(games, tuned, log, fixed(NOW))
    prefix = log.read_bytes()
    retuned = TunedModel(tuned.params, 30.0, tuned.b0_home_win_rate, tuned.b0_home_margin)
    assert run(games, retuned, log, fixed(NOW)) == 3
    assert log.read_bytes().startswith(prefix)


def test_refuses_when_stamp_is_not_before_tipoff(
    games: pd.DataFrame, tuned: TunedModel, tmp_path: Path
) -> None:
    log = tmp_path / "log.csv"
    log.write_text(",".join(LOG_COLUMNS) + "\n")
    before = log.read_bytes()
    # Selection happens at NOW, but the stamp is taken exactly at the first tip-off.
    ticks: Iterator[datetime] = iter([NOW, datetime(2026, 10, 1, 18, 0, tzinfo=UTC)])
    with pytest.raises(LatePredictionError):
        run(games, tuned, log, lambda: next(ticks))
    assert log.read_bytes() == before


def test_skips_played_unconfirmed_and_out_of_window_games(
    games: pd.DataFrame, tuned: TunedModel, tmp_path: Path
) -> None:
    unconfirmed = games.assign(
        confirmed_date=~((games["season"] == 2026) & (games["game_code"] == 1))
    )
    log = tmp_path / "log.csv"
    assert run(unconfirmed, tuned, log, fixed(NOW)) == 2
    assert run(games, tuned, tmp_path / "late.csv", fixed(NOW + timedelta(hours=11))) == 0


def test_history_before_replay_start_cannot_shift_live_predictions(
    games: pd.DataFrame, tuned: TunedModel, tmp_path: Path
) -> None:
    older = make_games({2021: True, 2022: True}, seed=99)
    with_history = pd.concat([older, games], ignore_index=True)
    run(games, tuned, tmp_path / "a.csv", fixed(NOW))
    run(with_history, tuned, tmp_path / "b.csv", fixed(NOW))
    assert (tmp_path / "a.csv").read_bytes() == (tmp_path / "b.csv").read_bytes()
