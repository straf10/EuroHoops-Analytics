"""Checklist item 31 (weeks 7-10b G5): the season-level variants in reports/backtest_m2.json.

- every declared field is present on every scored split, and everything is labelled post-hoc;
- the two backtest runs of item 24 (``$SCRATCH/m2_run{1,2}.json``) agree on the level block;
- declaration < first run: the progress file names ``LEVEL_DECLARATION <sha>`` and
  ``LEVEL_RUN <sha>``; the declaration's report has no level block, the run commit's has one,
  and the declaration is an ancestor of the run commit;
- the weeks 7-10 numbers did not move: the report without its level block equals the report
  at the declaration commit.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

FIELDS = ("log_loss", "brier", "ece", "reliability", "by_band", "by_type")


def named(kind: str, text: str) -> str | None:
    found = re.findall(rf"^{kind} ([0-9a-f]{{7,40}})\b", text, flags=re.MULTILINE)
    return found[-1] if found else None


def report_at(commit: str) -> dict[str, Any] | None:
    shown = subprocess.run(
        ["git", "show", f"{commit}:reports/backtest_m2.json"],
        capture_output=True,
        text=True,
        check=False,
    )
    return json.loads(shown.stdout) if shown.returncode == 0 else None


report = json.loads(Path("reports/backtest_m2.json").read_text(encoding="utf-8"))
levels = report.get("level_variants")
ok = levels is not None
if levels is None:
    print("no level_variants in reports/backtest_m2.json")
    sys.exit(1)
splits = ("cv", "validation", "test") if report["test_scored"] else ("cv", "validation")
missing = []
for name, block in levels["variants"].items():
    for split in splits:
        missing += [(name, split, f) for f in FIELDS if block[split].get(f) is None]
        missing += [
            (name, split, f"minus_base.{f}")
            for f in ("log_loss", "brier", "ece")
            if block["minus_base"][split] is None or block["minus_base"][split].get(f) is None
        ]
    if not block["post_hoc"] or block["label"] != "post-hoc, not a clean hold-out":
        missing.append((name, "label", "post_hoc"))
    large = block["calibration_in_the_large"]["level"]
    print(
        f"{name}: CV log loss {block['cv']['log_loss']}, validation ECE "
        f"{block['validation']['ece']} (meets F-f: {block['meets_f_f']['validation']}), "
        f"calibration in the large {min(large.values())}-{max(large.values())}"
    )
ok &= not missing and levels["post_hoc"] is True
print("missing or unlabelled fields:", missing)
print("G-g condition:", levels["g_g_condition"])

scratch = Path(os.environ.get("SCRATCH", "."))
runs = [scratch / f"m2_run{i}.json" for i in (1, 2)]
if all(r.exists() for r in runs):
    blocks = [json.loads(r.read_text(encoding="utf-8"))["level_variants"] for r in runs]
    same = blocks[0] == blocks[1] == levels
    print("item 24's two runs: level blocks identical and equal to the report:", same)
    ok &= same
else:
    print("item 24's runs not found in $SCRATCH (run item 24 first)")
    ok = False

text = Path("reports/week7-10b_progress.md").read_text(encoding="utf-8")
declaration, first_run = named("LEVEL_DECLARATION", text), named("LEVEL_RUN", text)
print(f"declaration {declaration}, first run {first_run}")
if declaration is None or first_run is None:
    ok = False
else:
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", declaration, first_run], check=False
    )
    at_declaration, at_run = report_at(declaration), report_at(first_run)
    order = (
        ancestor.returncode == 0
        and declaration != first_run
        and at_declaration is not None
        and "level_variants" not in at_declaration
        and at_run is not None
        and "level_variants" in at_run
    )
    print("declaration < first run:", "yes" if order else "NO")
    unchanged = (
        at_declaration is not None
        and {k: v for k, v in report.items() if k != "level_variants"} == at_declaration
    )
    print("weeks 7-10 numbers unchanged since the declaration:", "yes" if unchanged else "NO")
    ok &= order and unchanged
sys.exit(0 if ok else 1)
