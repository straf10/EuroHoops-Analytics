"""Checklist item 16: the last M1 backtest of each competition in the local MLflow store."""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import mlflow

from eurohoops.eval.tracking import default_tracking_uri

COMMIT_LENGTH = 40
mlflow.set_tracking_uri(default_tracking_uri())
runs = mlflow.search_runs(experiment_names=["m1-backtest"], order_by=["start_time DESC"])
metric_cols = [c for c in runs.columns if c.startswith("metrics.")]
param_cols = [c for c in runs.columns if c.startswith("params.")]


def counts(row: "dict[str, object]") -> tuple[int, int]:
    """(metrics, params) that are set on a run row (NaN != NaN marks a missing metric)."""
    metrics = sum(row[c] == row[c] for c in metric_cols)
    params = sum(row[c] is not None and row[c] == row[c] for c in param_cols)
    return metrics, params


ok = True
for comp, path in (
    ("euroleague", "reports/backtest_m1.json"),
    ("gbl", "reports/backtest_m1_gbl.json"),
):
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    parents = runs[(runs["params.competition"] == comp) & runs["tags.mlflow.parentRunId"].isna()]
    parent = parents.iloc[0]
    children = runs[runs["tags.mlflow.parentRunId"] == parent["run_id"]]
    pm, pp = counts(parent)
    print(
        f"{comp}: parent {parent['run_id']} ({parent['tags.mlflow.runName']}): {pp} params, "
        f"{pm} metrics, data_sha256 {parent['params.data_sha256'][:12]}…, "
        f"commit {parent['params.git_commit'][:10]} dirty={parent['params.git_dirty']}"
    )
    for _, child in children.sort_values("params.variant").iterrows():
        cm, cp = counts(child)
        print(
            f"  child {child['run_id']} {child['params.variant']}: {cp} params, {cm} metrics, "
            f"validation log loss {child['metrics.validation.log_loss']:.6f}"
        )
        ok &= cm > 0 and cp > 0 and len(child["params.git_commit"]) == COMMIT_LENGTH
    ok &= len(children) == len(report["variants"]) and pm > 0 and pp > 0
    ok &= parent["params.data_sha256"] == report["data_sha256"]
sys.exit(0 if ok else 1)
