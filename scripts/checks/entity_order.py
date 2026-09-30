"""Checklist item 38 (order part): matcher commit < labels commit < first override from the
labels (I-g).

The progress file names the matcher commit on a ``MATCHER <sha>`` line. The labels commit is the
first commit that adds ``entity/labels.csv``; the first label override is the first commit whose
``entity/overrides.csv`` has a row whose reason starts with ``label:`` (none yet is fine). Until
the owner provides the labels, the item is BLOCKED (exit 1) and says so.
"""

import re
import subprocess
import sys
from pathlib import Path

PROGRESS = Path("reports/week12-14_progress.md")
LABELS = "entity/labels.csv"
OVERRIDES = "entity/overrides.csv"


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=False).stdout


def ancestor(a: str, b: str) -> bool:
    run = subprocess.run(["git", "merge-base", "--is-ancestor", a, b], check=False)
    return run.returncode == 0 and a != b


found = re.findall(r"^MATCHER ([0-9a-f]{7,40})\b", PROGRESS.read_text(encoding="utf-8"), re.M)
matcher = found[-1] if found else None
labels = git("log", "--diff-filter=A", "--format=%h", "--reverse", "--", LABELS).split()
print(f"matcher {matcher}, labels {labels[0] if labels else None}")
if matcher is None:
    print("no MATCHER line in the progress file")
    sys.exit(1)
if not labels:
    print("BLOCKED: entity/labels.csv not provided by the owner yet (I5)")
    sys.exit(1)
ok = ancestor(matcher, labels[0])
label_override = None
for commit in git("log", "--format=%h", "--reverse", "--", OVERRIDES).split():
    shown = git("show", f"{commit}:{OVERRIDES}")
    if any("label:" in line for line in shown.splitlines()[1:]):
        label_override = commit
        break
if label_override is not None:
    ok &= ancestor(labels[0], label_override)
print(f"first label override {label_override}")
print("matcher < labels < label overrides:", "yes" if ok else "NO")
sys.exit(0 if ok else 1)
