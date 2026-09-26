"""F4/F5: the Optuna study is reproducible, and the M2 backtest writes every field of its report
(gate block included) identically on two runs, on small synthetic shots."""

import json
from typing import Any

import numpy as np
import pandas as pd
import pytest

from eurohoops.config import M2Seasons
from eurohoops.eval import m2_backtest
from eurohoops.eval.shot_metrics import calibrated
from tests.test_xpts import synthetic_shots

pytest.importorskip("lightgbm")
pytest.importorskip("optuna")

SEASONS = M2Seasons(development=(2011, 2012, 2013), validation=(2014,), test=(2015,))
TINY = {
    "num_leaves": 7,
    "learning_rate": 0.1,
    "n_estimators": 20,
    "min_child_samples": 40,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "lambda_l2": 1.0,
}


def labelled(n: int, seed: int, seasons: tuple[int, ...]) -> pd.DataFrame:
    shots = synthetic_shots(n, seed)
    rng = np.random.default_rng(seed + 1)
    shots["season"] = rng.choice(seasons, n)
    shots["game_id"] = [f"E{s}_{i % 40}" for i, s in enumerate(shots["season"])]
    shots["event"] = np.arange(n)
    p = 1.0 / (1.0 + np.exp(-(0.9 - 0.25 * shots["distance"] + 0.3 * shots["fastbreak"])))
    shots["made"] = rng.random(n) < p
    shots["band"] = np.where(shots["value"] == 3, "three", "rim")
    shots["validated_season"] = True
    return shots


def test_a_five_trial_study_on_two_seasons_is_reproducible() -> None:
    """F4 Done-when: the same seed gives identical trial values and best parameters."""
    shots = labelled(3_000, 1, (2011, 2012))
    first = m2_backtest.search(shots, (2011, 2012), n_trials=5)
    second = m2_backtest.search(shots, (2011, 2012), n_trials=5)
    assert [t["value"] for t in first["trials"]] == [t["value"] for t in second["trials"]]
    assert first["best_params"] == second["best_params"]
    assert len(first["trials"]) == 5
    assert first["seed"] == 20261001


def _study() -> dict[str, Any]:
    return {
        "sampler": "TPESampler",
        "seed": 20261001,
        "n_trials": 0,
        "best_trial": 0,
        "best_value": 0.0,
        "best_params": TINY,
        "trials": [],
    }


def _run(monkeypatch: pytest.MonkeyPatch, score_test: bool) -> tuple[dict[str, Any], pd.DataFrame]:
    monkeypatch.setattr(m2_backtest, "SEEDS", (20261001, 20261002))
    monkeypatch.setattr(m2_backtest, "SPLINE_KNOTS", (4,))
    monkeypatch.setattr(m2_backtest, "SPLINE_L2", (1e-4, 1e-3))
    monkeypatch.setattr(m2_backtest, "BOOTSTRAP_RESAMPLES", 50)
    shots = labelled(6_000, 2, (2011, 2012, 2013, 2014, 2015))
    return m2_backtest.run_m2_backtest(shots, SEASONS, _study(), score_test, lambda _: None)


def test_report_has_every_field_and_two_runs_are_identical(monkeypatch: pytest.MonkeyPatch) -> None:
    report, xpts = _run(monkeypatch, score_test=False)
    again, _ = _run(monkeypatch, score_test=False)
    assert json.dumps(report, sort_keys=True) == json.dumps(again, sort_keys=True)
    for variant in m2_backtest.VARIANTS:
        block = report["variants"][variant]
        for split in ("cv", "validation"):
            assert {"log_loss", "brier", "ece", "reliability", "by_band", "by_type"} <= set(
                block[split]
            )
            assert set(block[split]["by_type"]) == {"2pt", "3pt"}
        assert block["test"] is None and not block["post_hoc"]
        assert set(block["cv_per_season"]) == {"2011", "2012", "2013"}
    gate = report["gate"]
    assert {"calibrated", "beats_baseline", "passed"} <= set(gate)
    assert gate["passed"] == (gate["calibrated"] and gate["beats_baseline"])
    assert gate["calibrated"] == calibrated(report["variants"][gate["chosen"]]["validation"])
    assert gate["challenger"] in m2_backtest.CHALLENGERS
    assert gate["baseline"] in m2_backtest.BASELINES
    assert len(gate["log_loss_challenger_minus_baseline"]["ci95"]) == 2
    assert set(gate["single_seed_would_flip"]) == {"20261001", "20261002"}
    assert report["test_gate"] is None and not report["test_scored"]
    assert set(xpts["split"]) == {"development", "validation"}
    assert xpts["p_make"].between(0, 1).all()


def test_scoring_the_test_seasons_adds_test_numbers(monkeypatch: pytest.MonkeyPatch) -> None:
    report, xpts = _run(monkeypatch, score_test=True)
    assert report["test_scored"]
    assert report["variants"]["spline"]["test"]["n"] == report["shots"]["test"] > 0
    assert report["test_gate"]["split"] == "test"
    assert set(xpts["split"]) == {"development", "validation", "test"}
