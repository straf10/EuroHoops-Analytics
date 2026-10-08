"""Checklist item 57: live M6 (weeks 16-18 L9) on the real marts.

`eurohoops project --dry-run` twice: both runs print the same sha256 for each report (the build
is deterministic) and leave predictions/, reports/ and entity/ untouched; the committed reports
(m6_projections, m6_board, m6_similarity, sim_ungated_*) hold no age and no birth date
(`live_m6.personal_fields`); `daily.yml` runs `eurohoops project` iff the EuroLeague M6 gate
passed (L-l) and `sim-ungated` for both competitions (D1); actionlint is clean. Says whether the
committed reports equal today's rebuild (they differ once new games are in the marts; reported,
not failed). Prints `RUNTIME project N` (the slower run).
"""

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

from eurohoops.config import M6, M6_BOARD, M6_PROJECTIONS, M6_SIMILARITY, SIM_UNGATED
from eurohoops.live_m6 import personal_fields

SHA = re.compile(r"^(?P<path>\S+): sha256 (?P<sha>[0-9a-f]{64})$")
RUNTIME = re.compile(r"^RUNTIME project: (?P<s>\d+) s$")


def tree_state() -> str:
    """Hash of the status and diff of predictions/, reports/ and entity/."""
    paths = ["predictions", "reports", "entity"]
    out = subprocess.run(
        ["git", "status", "--porcelain", "--", *paths], capture_output=True, text=True, check=True
    ).stdout
    diff = subprocess.run(
        ["git", "diff", "--", *paths], capture_output=True, text=True, check=True
    ).stdout
    return hashlib.sha256((out + diff).encode()).hexdigest()


def dry_run() -> tuple[dict[str, str], int]:
    out = subprocess.run(
        ["uv", "run", "eurohoops", "project", "--dry-run"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    shas, runtime = {}, -1
    for line in out.splitlines():
        if m := SHA.match(line.strip()):
            shas[Path(m["path"]).as_posix()] = m["sha"]
        if m := RUNTIME.match(line.strip()):
            runtime = int(m["s"])
    return shas, runtime


ok = True
before = tree_state()
first, t1 = dry_run()
second, t2 = dry_run()
same = bool(first) and first == second
print(f"two dry runs, same sha256 for {len(first)} reports: {same}")
untouched = tree_state() == before
print(f"predictions/, reports/ and entity/ untouched: {untouched}")
ok &= same and untouched and len(first) == 3

for path in (M6_PROJECTIONS, M6_BOARD, M6_SIMILARITY, *SIM_UNGATED.values()):
    found = personal_fields(json.loads(path.read_text(encoding="utf-8")))
    print(f"{path.as_posix()}: age or birth-date fields {found[:3] or 'none'}")
    ok &= not found
for path in (M6_PROJECTIONS, M6_BOARD, M6_SIMILARITY):
    committed = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"{path.as_posix()} equals today's rebuild: {committed == first.get(path.as_posix())}")

passed = bool(json.loads(M6.report.read_text(encoding="utf-8"))["gate"]["passed"])
workflow = Path(".github/workflows/daily.yml").read_text(encoding="utf-8")
scheduled = "eurohoops project" in workflow
ungated = all(
    line in workflow
    for line in ("eurohoops sim-ungated\n", "eurohoops sim-ungated --competition gbl\n")
)
print(f"daily.yml runs project: {scheduled} (gate passed: {passed}); sim-ungated both: {ungated}")
ok &= scheduled == passed and ungated
lint = subprocess.run(
    ["uvx", "--from", "actionlint-py", "actionlint", ".github/workflows/daily.yml"], check=False
)
print(f"actionlint daily.yml: {'clean' if lint.returncode == 0 else 'FAIL'}")
ok &= lint.returncode == 0
print(f"RUNTIME project {max(t1, t2)}")
print("m6 live:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
