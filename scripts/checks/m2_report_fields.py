"""Checklist item 24: every declared variant has every metric on every scored split, and the
report carries its machine-readable gate block."""

import json
import sys
from pathlib import Path

report = json.loads(Path("reports/backtest_m2.json").read_text(encoding="utf-8"))
splits = ("cv", "validation", "test") if report["test_scored"] else ("cv", "validation")
missing = [
    (variant, split, key)
    for variant in report["declared_variants"]
    for split in splits
    for key in ("log_loss", "brier", "ece", "reliability", "by_band", "by_type", "by_context")
    if report["variants"][variant][split].get(key) is None
]
gate = report["gate"]
print(
    f"gate: {gate['challenger']} vs {gate['baseline']}, log loss diff "
    f"{gate['log_loss_challenger_minus_baseline']['mean']} "
    f"{gate['log_loss_challenger_minus_baseline']['ci95']}; calibrated {gate['calibrated']}, "
    f"beats_baseline {gate['beats_baseline']}, passed {gate['passed']}"
)
print("missing fields:", missing)
sys.exit(0 if not missing and {"calibrated", "beats_baseline", "passed"} <= set(gate) else 1)
