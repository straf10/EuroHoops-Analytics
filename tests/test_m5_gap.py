"""Week 14-16 J8: the projected-vs-oracle gap in the committed M5 reports equals a recomputation
from the committed per-game predictions (log-loss difference mean and margin-RMSE difference).
The ``missed_top3`` subset needs the shares frames, so only its size bound is checked here."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).parent.parent
REPORTS = [
    (REPO / "reports" / "backtest_m5.json", REPO / "reports" / "backtest_m5_games.csv"),
    (REPO / "reports" / "backtest_m5_gbl.json", REPO / "reports" / "backtest_m5_gbl_games.csv"),
]


def _log_loss(p: np.ndarray, won: np.ndarray) -> np.ndarray:
    return -(won * np.log(p) + (1.0 - won) * np.log(1.0 - p))


@pytest.mark.parametrize(("report_path", "games_path"), REPORTS)
def test_the_gap_block_equals_a_recomputation(report_path: Path, games_path: Path) -> None:
    if not report_path.is_file() or not games_path.is_file():
        pytest.skip(f"{report_path.name} not committed yet")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    frame = pd.read_csv(games_path, dtype={"game_id": str})
    assert set(report["gap"]) >= {"validation"}
    for split, block in report["gap"].items():
        rows = frame[frame["split"] == split]
        projected = rows[rows["model"] == "m5"].set_index("game_id").sort_index()
        oracle = rows[rows["model"] == "m5_oracle"].set_index("game_id").sort_index()
        both = projected.index.intersection(oracle.index)
        projected, oracle = projected.loc[both], oracle.loc[both]
        assert block["n"] == len(both)
        won = (projected["actual_margin"] > 0).to_numpy(dtype=float)
        loss = _log_loss(projected["p_home"].to_numpy(), won) - _log_loss(
            oracle["p_home"].to_numpy(), won
        )
        # the CSV holds 6-decimal p_home, so the mean agrees to a few 1e-6
        assert abs(loss.mean() - block["log_loss_projected_minus_oracle"]["mean"]) < 2e-5
        actual = projected["actual_margin"].to_numpy()
        rmse = np.sqrt(np.mean((actual - projected["exp_margin"].to_numpy()) ** 2)) - np.sqrt(
            np.mean((actual - oracle["exp_margin"].to_numpy()) ** 2)
        )
        assert abs(rmse - block["rmse_projected_minus_oracle"]["mean"]) < 2e-5
        assert 0 < block["missed_top3"]["n"] < block["n"]
