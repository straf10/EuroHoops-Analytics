"""Checklist item 23: the recorded M2 runtimes are within budget.

The progress files record them on lines ``RUNTIME backtest <seconds> s`` and
``RUNTIME study <seconds> s``; for each kind the last line of the newest progress file that has
one counts (weeks 7-10b before weeks 7-10). Budgets: backtest 60 minutes since the user's
decision of 2026-09-26 (weeks 7-10b decision 3; it was 40), study 2 hours.
"""

import re
import sys
from pathlib import Path

LIMITS = {"backtest": 60 * 60, "study": 2 * 60 * 60}
FILES = (Path("reports/week7-10b_progress.md"), Path("reports/week7-10_progress.md"))
ok = True
for kind, limit in LIMITS.items():
    found = []
    for path in FILES:
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        found = re.findall(rf"^RUNTIME {kind} (\d+) s", text, flags=re.MULTILINE)
        if found:
            break
    if not found:
        print(f"no recorded {kind} runtime")
        ok = False
        continue
    seconds = int(found[-1])
    print(f"recorded {kind} runtime {seconds} s (limit {limit} s, from {path})")
    ok &= seconds < limit
sys.exit(0 if ok else 1)
