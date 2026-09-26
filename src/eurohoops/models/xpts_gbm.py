"""M2 challenger (F4): LightGBM on the same shot context as the spline baseline, plus isotonic
calibration. LightGBM and Optuna are dev dependencies (F-j), imported only when M2 runs.

Determinism: ``deterministic=True``, ``force_row_wise=True``, a fixed ``num_threads`` and
``seed``; the Optuna study uses ``TPESampler(seed=20261001)`` with ``n_jobs=1``.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from eurohoops.models.elo import FloatArray
from eurohoops.models.xpts import has_identity_column
from eurohoops.parse.shot_table import OUTCOME_CODED_FLAGS

FEATURES = (
    "distance",
    "angle",
    "zone_code",
    "three",
    "seconds_left",
    "period",
    "margin_before",
    "home",
    "season",
)
CATEGORICAL = ("zone_code",)
ZONES = tuple("ABCDEFGHIJ")
NUM_THREADS = 6  # physical cores of the reference machine; fixed for determinism
FIXED_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "deterministic": True,
    "force_row_wise": True,
    "num_threads": NUM_THREADS,
    "verbosity": -1,
    "bagging_freq": 1,
    "max_bin": 255,
}
STUDY_SEED = 20261001
N_TRIALS = 60


def features(shots: pd.DataFrame) -> pd.DataFrame:
    """The challenger's feature frame (F-b without the outcome-coded flags): no identity column;
    period caps at 5 (OT)."""
    zone = shots["zone"].astype(str)
    return pd.DataFrame(
        {
            "distance": shots["distance"].to_numpy(dtype=np.float64),
            "angle": shots["angle"].to_numpy(dtype=np.float64),
            "zone_code": np.array([ZONES.index(z) if z in ZONES else -1 for z in zone]),
            "three": (shots["value"].to_numpy() == 3).astype(np.float64),
            "seconds_left": shots["seconds_left"].to_numpy(dtype=np.float64),
            "period": np.minimum(shots["period"].to_numpy(), 5).astype(np.float64),
            "margin_before": shots["margin_before"].to_numpy(dtype=np.float64),
            "home": shots["home"].to_numpy(dtype=np.float64),
            "season": shots["season"].to_numpy(dtype=np.float64),
        },
        columns=list(FEATURES),
    )


if has_identity_column(FEATURES) or set(FEATURES) & set(OUTCOME_CODED_FLAGS):
    raise ValueError(f"identity or outcome-coded column in the M2 features: {FEATURES}")


def search_space(trial: Any) -> dict[str, Any]:
    """The declared Optuna search space (reports/week7-10_progress.md)."""
    return {
        "num_leaves": trial.suggest_int("num_leaves", 8, 128, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 2000, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 100.0, log=True),
    }


@dataclass(frozen=True)
class GbmModel:
    booster: Any

    def predict(self, shots: pd.DataFrame) -> FloatArray:
        out: FloatArray = np.asarray(self.booster.predict(features(shots)), dtype=np.float64)
        return out


def fit_gbm(shots: pd.DataFrame, params: dict[str, Any], seed: int) -> GbmModel:
    import lightgbm as lgb  # noqa: PLC0415 - dev dependency (F-j)

    rounds = int(params["n_estimators"])
    booster_params = {
        **FIXED_PARAMS,
        **{k: v for k, v in params.items() if k != "n_estimators"},
        "seed": seed,
    }
    data = lgb.Dataset(
        features(shots),
        label=shots["made"].to_numpy(dtype=np.float64),
        categorical_feature=list(CATEGORICAL),
        free_raw_data=True,
    )
    return GbmModel(lgb.train(booster_params, data, num_boost_round=rounds))


Objective = Callable[[dict[str, Any]], float]


def run_study(
    objective: Objective,
    n_trials: int,
    seed: int = STUDY_SEED,
    progress: Callable[[str], None] | None = None,
) -> Any:
    """TPE study minimising ``objective(params)``; returns the Optuna study. ``progress`` gets
    one line per finished trial (its number, value and the best so far)."""
    import optuna  # noqa: PLC0415 - dev dependency (F-j)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))

    def report(study: Any, trial: Any) -> None:
        if progress is not None:
            progress(
                f"optuna trial {trial.number + 1}/{n_trials}: {trial.value:.6f} "
                f"(best {study.best_value:.6f}, trial {study.best_trial.number})"
            )

    study.optimize(
        lambda trial: objective(search_space(trial)),
        n_trials=n_trials,
        n_jobs=1,
        callbacks=[report],
    )
    return study
