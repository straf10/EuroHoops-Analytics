"""Checklist item 24: the M2 declaration, verdict and test-run commits are in that order.

The progress file names them on lines ``DECLARATION <sha>``, ``VERDICT <sha>`` and
``TEST <sha>``. The declaration commit has no M2 report (so no validation number existed), the
verdict commit's report has not scored the test seasons, and each commit is an ancestor of the
next and of HEAD.
"""

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


def named(kind: str, text: str) -> str | None:
    found = re.findall(rf"^{kind} ([0-9a-f]{{7,40}})\b", text, flags=re.MULTILINE)
    return found[-1] if found else None


def ancestor(a: str, b: str) -> bool:
    run = subprocess.run(["git", "merge-base", "--is-ancestor", a, b], check=False)
    return run.returncode == 0 and a != b


def report_at(commit: str) -> dict[str, Any] | None:
    shown = subprocess.run(
        ["git", "show", f"{commit}:reports/backtest_m2.json"],
        capture_output=True,
        text=True,
        check=False,
    )
    return json.loads(shown.stdout) if shown.returncode == 0 else None


text = Path("reports/week7-10_progress.md").read_text(encoding="utf-8")
declaration, verdict, test = (named(k, text) for k in ("DECLARATION", "VERDICT", "TEST"))
print(f"declaration {declaration}, verdict {verdict}, test {test}")
ok = declaration is not None and verdict is not None
if declaration is not None and verdict is not None:
    ok &= ancestor(declaration, verdict) and report_at(declaration) is None
    at_verdict = report_at(verdict)
    ok &= at_verdict is not None and not at_verdict["test_scored"]
    current = json.loads(Path("reports/backtest_m2.json").read_text(encoding="utf-8"))
    if current["test_scored"]:
        in_history = subprocess.run(
            ["git", "merge-base", "--is-ancestor", str(test), "HEAD"], check=False
        )
        ok &= test is not None and ancestor(verdict, test) and in_history.returncode == 0
print("declaration < verdict < test:", "yes" if ok else "NO")
sys.exit(0 if ok else 1)
