"""Checklist item 23: the recorded M2 runtimes are within budget.

The progress file records them on lines ``RUNTIME backtest <seconds> s`` and
``RUNTIME study <seconds> s``; the last line of each kind counts.
"""

import re
import sys
from pathlib import Path

LIMITS = {"backtest": 40 * 60, "study": 2 * 60 * 60}
text = Path("reports/week7-10_progress.md").read_text(encoding="utf-8")
ok = True
for kind, limit in LIMITS.items():
    found = re.findall(rf"^RUNTIME {kind} (\d+) s", text, flags=re.MULTILINE)
    if not found:
        print(f"no recorded {kind} runtime")
        ok = False
        continue
    seconds = int(found[-1])
    print(f"recorded {kind} runtime {seconds} s (limit {limit} s)")
    ok &= seconds < limit
sys.exit(0 if ok else 1)
