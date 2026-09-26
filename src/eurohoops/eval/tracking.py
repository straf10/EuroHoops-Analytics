"""MLflow tracking of M1 backtests (E-f): local store under ``mlruns/``, no server, no registry.

MLflow is a dev dependency and is imported only here, only when a backtest runs; without it
(``uv sync --no-dev``, as in the daily workflow) tracking is skipped with a warning. The live
predictor never reads MLflow: it reads the frozen parameters from the committed report JSON.

One parent run per backtest (all hyperparameters, the data snapshot hash, the git commit and
dirty flag, every numeric value of the report as a metric, the report JSON as an artifact) and
one nested child run per declared variant. The store is SQLite (``mlruns/mlflow.db``, artifacts
in ``mlruns/artifacts``): MLflow 3.16 refuses the plain file store unless opted back in.
"""

import hashlib
import json
import logging
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

EXPERIMENT = "m1-backtest"
TRACKING_DIR = Path("mlruns")
METRIC_NAME = re.compile(r"[^A-Za-z0-9_\-. /]")

log = logging.getLogger(__name__)


def default_tracking_uri(root: Path = TRACKING_DIR) -> str:
    return "sqlite:///" + (root.absolute() / "mlflow.db").as_posix()


def data_hash(games: pd.DataFrame, team_games: pd.DataFrame) -> str:
    """sha256 of the game rows and team-game rows a backtest reads, sorted by key."""
    digest = hashlib.sha256()
    for frame, keys in ((games, ["game_id"]), (team_games, ["game_id", "team"])):
        ordered = frame.sort_values(keys).reset_index(drop=True)
        digest.update(ordered.to_csv(index=False, lineterminator="\n").encode())
    return digest.hexdigest()


def git_state(cwd: Path | None = None) -> tuple[str, bool]:
    """(HEAD commit, dirty flag); ("unknown", True) outside a git checkout."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True
    return commit, bool(status.strip())


def flatten_metrics(value: Any, prefix: str = "") -> dict[str, float]:
    """Every numeric leaf (bools and None excluded) as ``a.b.0.c`` -> value."""
    out: dict[str, float] = {}
    if isinstance(value, dict):
        items = list(value.items())
    elif isinstance(value, list):
        items = list(enumerate(value))
    else:
        items = []
    for key, item in items:
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, (dict, list)):
            out |= flatten_metrics(item, name)
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            out[METRIC_NAME.sub("_", name)] = float(item)
    return out


def _params(report: dict[str, Any]) -> dict[str, Any]:
    tuned = report["tuned"]
    return {
        "model_version": report["model_version"],
        "variant": tuned["variant"],
        **{f"rating.{k}": v for k, v in tuned["rating"].items()},
        **{f"pace.{k}": v for k, v in tuned["pace"].items()},
        **{f"margin.{k}": v for k, v in tuned["margin"].items()},
        "totals_sigma": tuned["totals_sigma"],
        **{f"elo.{k}": v for k, v in report["comparison_elo"].items()},
        **{f"seasons.{k}": ",".join(map(str, v)) for k, v in report["seasons"].items()},
        "test_scored": report["test_scored"],
    }


def log_backtest(
    report: dict[str, Any], competition: str, snapshot: str, tracking_uri: str
) -> str | None:
    """Log one backtest; return the parent run id, or None when MLflow is not installed."""
    try:
        import mlflow  # noqa: PLC0415 - optional dev dependency, only on the backtest path
    except ImportError:
        log.warning("MLflow is not installed (dev dependency): backtest not tracked")
        return None
    commit, dirty = git_state()
    shared = {"competition": competition, "data_sha256": snapshot, "git_commit": commit}
    shared["git_dirty"] = str(dirty).lower()
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT)
    if experiment is None:
        artifacts = Path(tracking_uri.removeprefix("sqlite:///")).parent / "artifacts"
        experiment_id = client.create_experiment(EXPERIMENT, artifact_location=artifacts.as_uri())
    else:
        experiment_id = experiment.experiment_id
    name = f"{competition} {report['model_version']}"
    with mlflow.start_run(experiment_id=experiment_id, run_name=name) as parent:
        mlflow.log_params({**shared, **_params(report)})
        mlflow.set_tags(shared)
        mlflow.log_metrics(flatten_metrics(report))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"backtest_m1_{competition}.json"
            path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            mlflow.log_artifact(str(path))
        for variant, block in report["variants"].items():
            with mlflow.start_run(
                experiment_id=experiment_id, run_name=f"{name} {variant}", nested=True
            ):
                mlflow.log_params(
                    {
                        **shared,
                        **_params(report),
                        "variant": variant,
                        "gate_variant": report["tuned"]["variant"],
                        **{k: block[k] for k in ("df", "scale", "ref_pace", "post_hoc")},
                    }
                )
                mlflow.set_tags(shared)
                mlflow.log_metrics(flatten_metrics(block))
    run_id: str = parent.info.run_id
    return run_id


EXPERIMENT_M2 = "m2-backtest"


def _experiment(client: Any, name: str, tracking_uri: str) -> str:
    experiment = client.get_experiment_by_name(name)
    if experiment is not None:
        return str(experiment.experiment_id)
    artifacts = Path(tracking_uri.removeprefix("sqlite:///")).parent / "artifacts"
    return str(client.create_experiment(name, artifact_location=artifacts.as_uri()))


def log_m2_backtest(report: dict[str, Any], tracking_uri: str) -> str | None:
    """Log one M2 backtest (F9): a parent run (params, data hash, commit, every numeric report
    value, the report JSON), one child per declared variant and one for the Optuna study with
    every trial's value as a stepped metric. None when MLflow is not installed."""
    try:
        import mlflow  # noqa: PLC0415 - optional dev dependency, only on the backtest path
    except ImportError:
        log.warning("MLflow is not installed (dev dependency): backtest not tracked")
        return None
    commit, dirty = git_state()
    shared = {
        "competition": "euroleague",
        "data_sha256": report["data_sha256"],
        "git_commit": commit,
        "git_dirty": str(dirty).lower(),
        "model_version": report["model_version"],
    }
    params = {
        **shared,
        **{f"spline.{k}": v for k, v in report["spline_chosen"].items()},
        **{f"lgbm.{k}": v for k, v in report["lightgbm_params"].items()},
        **{f"seasons.{k}": ",".join(map(str, v)) for k, v in report["seasons"].items()},
        "seeds": ",".join(map(str, report["seed_robustness"]["seeds"])),
        "gate_chosen": report["gate"]["chosen"],
        "test_scored": report["test_scored"],
    }
    mlflow.set_tracking_uri(tracking_uri)
    experiment_id = _experiment(mlflow.MlflowClient(), EXPERIMENT_M2, tracking_uri)
    name = f"m2 {report['model_version']}"
    with mlflow.start_run(experiment_id=experiment_id, run_name=name) as parent:
        mlflow.log_params(params)
        mlflow.set_tags(shared)
        mlflow.log_metrics(flatten_metrics({k: v for k, v in report.items() if k != "variants"}))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "backtest_m2.json"
            path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            mlflow.log_artifact(str(path))
        for variant, block in report["variants"].items():
            with mlflow.start_run(
                experiment_id=experiment_id, run_name=f"{name} {variant}", nested=True
            ):
                mlflow.log_params({**params, "variant": variant, "post_hoc": block["post_hoc"]})
                mlflow.set_tags(shared)
                mlflow.log_metrics(flatten_metrics(block))
        study = report["optuna_study"]
        with mlflow.start_run(
            experiment_id=experiment_id, run_name=f"{name} optuna-study", nested=True
        ):
            mlflow.log_params(
                {
                    **shared,
                    "sampler": study["sampler"],
                    "study_seed": study["seed"],
                    "n_trials": study["n_trials"],
                    **{f"best.{k}": v for k, v in study["best_params"].items()},
                }
            )
            mlflow.set_tags(shared)
            mlflow.log_metric("best_value", study["best_value"])
            for trial in study["trials"]:
                mlflow.log_metric("trial_value", trial["value"], step=trial["number"])
    run_id: str = parent.info.run_id
    return run_id
