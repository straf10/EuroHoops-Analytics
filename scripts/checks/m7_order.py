"""Week 14-16 K4 (order part): M7 verdict commit < validation scored < test scored.

The progress file names the commits on lines ``VERDICT <sha>``, ``VALIDATION <sha>`` and
``TEST <sha>``. At the verdict commit no validation number exists (the committed report, if
any, has ``validation_scored`` false); the validation commit's report has scored validation but
not test; each named commit is an ancestor of the next and of HEAD. ``TEST`` is required only
once the working report has ``test_scored``.
"""

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

PROGRESS = Path("reports/m7_progress.md")
REPORT = "reports/backtest_m7.json"


def named(kind: str, text: str) -> str | None:
    found = re.findall(rf"^{kind} ([0-9a-f]{{7,40}})\b", text, flags=re.MULTILINE)
    return found[-1] if found else None


def ancestor(a: str, b: str) -> bool:
    run = subprocess.run(["git", "merge-base", "--is-ancestor", a, b], check=False)
    return run.returncode == 0 and a != b


def in_history(commit: str) -> bool:
    """HEAD itself or one of its ancestors."""
    run = subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"], check=False)
    return run.returncode == 0


def report_at(commit: str) -> dict[str, Any] | None:
    shown = subprocess.run(
        ["git", "show", f"{commit}:{REPORT}"], capture_output=True, text=True, check=False
    )
    return json.loads(shown.stdout) if shown.returncode == 0 else None


text = PROGRESS.read_text(encoding="utf-8")
verdict, validation, test = (named(k, text) for k in ("VERDICT", "VALIDATION", "TEST"))
print(f"verdict {verdict}, validation {validation}, test {test}")
ok = verdict is not None and validation is not None
if verdict is not None and validation is not None:
    at_verdict = report_at(verdict)
    ok &= at_verdict is None or not at_verdict.get("validation_scored")
    at_validation = report_at(validation)
    ok &= at_validation is not None and bool(at_validation.get("validation_scored"))
    ok &= at_validation is not None and not at_validation.get("test_scored")
    ok &= ancestor(verdict, validation) and in_history(validation)
    current = json.loads(Path(REPORT).read_text(encoding="utf-8"))
    if current.get("test_scored"):
        ok &= test is not None and ancestor(validation, test) and in_history(test)
print("verdict < validation < test:", "yes" if ok else "NO")
sys.exit(0 if ok else 1)
