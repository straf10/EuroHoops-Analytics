import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eurohoops.config import Backtest, Grid
from eurohoops.eval.backtest import (
    grid_edges,
    load_tuned_model,
    run_backtest,
    totals_baseline,
    totals_scores,
)
from eurohoops.models.elo import EloParams, prepare, replay
from tests.conftest import make_games

GRID = Grid(k=(10.0, 20.0, 30.0), hca=(0.0, 50.0, 100.0), reversion=(0.0, 0.25, 0.5))


@pytest.mark.parametrize(
    ("best", "edges"),
    [
        (EloParams(20.0, 50.0, 0.25), []),
        (EloParams(20.0, 0.0, 0.0), []),  # a lower bound of 0 is natural, not an edge
        (EloParams(10.0, 100.0, 0.5), ["k", "hca", "reversion"]),
        (EloParams(30.0, 50.0, 0.25), ["k"]),
    ],
)
def test_grid_edges(best: EloParams, edges: list[str]) -> None:
    assert grid_edges(GRID, best) == edges


SPEC = Backtest(
    Path("unused.json"),
    warmup=(2019,),
    tuning=(2020, 2021),
    test=(2022, 2023),
    grid=Grid(k=(20.0, 30.0), hca=(50.0, 90.0), reversion=(0.25,)),
)


@pytest.fixture
def seasons() -> pd.DataFrame:
    return make_games({s: True for s in range(2017, 2024)} | {2024: False}, gbl_like=True)


def test_sigma_is_the_tuning_residual_rms(seasons: pd.DataFrame) -> None:
    report = run_backtest(seasons, SPEC)
    rated = seasons[seasons["played"] & ~seasons["forfeit"] & (seasons["season"] >= 2019)]
    diffs = replay(
        prepare(rated), EloParams(**{k: report["tuned"][k] for k in ("k", "hca", "reversion")})
    )
    tuning = rated["season"].isin(SPEC.tuning).to_numpy()
    residual = (rated["home_score"] - rated["away_score"]).to_numpy(dtype=float)[tuning] - (
        diffs[tuning] / report["tuned"]["margin_scale"]
    )
    assert report["tuned"]["margin_sigma"] == pytest.approx(np.sqrt(np.mean(residual**2)), abs=1e-6)
    assert report["metrics"]["test"]["elo"]["margin_crps"] > 0


def test_totals_baseline_is_the_mean_of_the_two_previous_seasons() -> None:
    games = pd.DataFrame(
        {
            "season": [2020, 2020, 2021, 2021, 2021, 2022],
            "home_score": [80, 90, 70, 20, 100, 999],
            "away_score": [70, 80, 60, 0, 90, 999],
            "played": [True] * 6,
            "forfeit": [False, False, False, True, False, False],
        }
    )
    # 2020: 150, 170; 2021: 130, 190 (the 20-0 forfeit is left out); 2022 is not read.
    assert totals_baseline(games, 2022) == pytest.approx(160.0)
    assert totals_baseline(games, 2021) is None  # 2019 missing: no partial baseline


def test_totals_scores_skip_seasons_without_a_baseline() -> None:
    games = pd.DataFrame(
        {"season": [2022, 2022, 2023], "home_score": [90, 80, 85], "away_score": [80, 60, 80]}
    )
    assert totals_scores(games, {2022: 160.0, 2023: None}) == {"n": 2, "mae": 15.0}
    assert totals_scores(games.iloc[:0], {}) == {"n": 0, "mae": None}


def test_load_tuned_model_reads_sigma_and_totals(tmp_path: Path, seasons: pd.DataFrame) -> None:
    report = run_backtest(seasons, SPEC)
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    model = load_tuned_model(path)
    assert model.margin_sigma == report["tuned"]["margin_sigma"]
    assert model.totals_baseline[2024] == report["totals"]["baseline_by_season"]["2024"]
    assert model.version() == report["model_version"]
