"""Checklist item 28: the last M2 backtest in the local MLflow store."""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import mlflow

from eurohoops.eval.tracking import EXPERIMENT_M2, default_tracking_uri

COMMIT_LENGTH = 40
mlflow.set_tracking_uri(default_tracking_uri())
report = json.loads(Path("reports/backtest_m2.json").read_text(encoding="utf-8"))
runs = mlflow.search_runs(experiment_names=[EXPERIMENT_M2], order_by=["start_time DESC"])
parent = runs[runs["tags.mlflow.parentRunId"].isna()].iloc[0]
children = runs[runs["tags.mlflow.parentRunId"] == parent["run_id"]]
print(
    f"parent {parent['run_id']} ({parent['tags.mlflow.runName']}): data_sha256 "
    f"{parent['params.data_sha256'][:12]}..., commit {parent['params.git_commit'][:10]} "
    f"dirty={parent['params.git_dirty']}"
)
for name in sorted(children["tags.mlflow.runName"]):
    print("  child", name)
study = children[children["tags.mlflow.runName"].str.endswith("optuna-study")]
client = mlflow.MlflowClient()
trials = client.get_metric_history(study.iloc[0]["run_id"], "trial_value") if len(study) else []
print(f"  optuna-study: {len(trials)} trial values (report {report['optuna_study']['n_trials']})")
ok = (
    parent["params.data_sha256"] == report["data_sha256"]
    and len(parent["params.git_commit"]) == COMMIT_LENGTH
    and len(children) == len(report["variants"]) + 1
    and len(trials) == report["optuna_study"]["n_trials"]
)
sys.exit(0 if ok else 1)
