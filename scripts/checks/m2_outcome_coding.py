"""Checklist item 32 (weeks 7-10b G1): no level of either M2 feature builder is outcome-coded on
the development shots (make rate >= 0.99, or <= 0.01, among >= 100 shots where it is set).

Guard: the feed's three dropped flags, audited the same way as if they were features, must be
caught (proves the audit sees a real outcome tag in this data).
"""

import sys
from pathlib import Path

import numpy as np

from eurohoops.config import M2_SEASONS
from eurohoops.marts import read_table
from eurohoops.models.feature_audit import builder_columns, levels
from eurohoops.parse.shot_table import OUTCOME_CODED_FLAGS

shots = read_table(Path("data/marts/eurohoops.duckdb"), "shots")
if shots is None:
    print("no shots mart; run: eurohoops shots")
    sys.exit(1)
dev = shots[shots["validated_season"] & shots["season"].isin(M2_SEASONS.development)]
dev = dev.reset_index(drop=True)
made = dev["made"].to_numpy(dtype=np.float64)
found = levels(builder_columns(dev), made)
ok = True
for builder in ("spline", "lgbm"):
    mine = [lv for lv in found if lv.builder == builder]
    rates = [lv.make_rate for lv in mine if lv.n >= 100]
    print(
        f"{builder}: {len({lv.column for lv in mine})} flag/level columns, {len(mine)} levels, "
        f"make rates {min(rates):.4f}-{max(rates):.4f} on {len(dev)} development shots"
    )
    for lv in mine:
        if lv.outcome_coded:
            ok = False
            print(
                f"  OUTCOME-CODED {lv.column} = {lv.level:g}: {lv.n} shots, "
                f"make rate {lv.make_rate:.4f}"
            )
feed = levels({("feed", f): dev[f].to_numpy(dtype=np.float64) for f in OUTCOME_CODED_FLAGS}, made)
for lv in feed:
    print(
        f"guard: feed flag {lv.column} audited as a feature: {lv.n} shots, make rate "
        f"{lv.make_rate:.4f} -> {'caught' if lv.outcome_coded else 'NOT caught'}"
    )
ok &= len(feed) == len(OUTCOME_CODED_FLAGS) and all(lv.outcome_coded for lv in feed)
print("no outcome-coded M2 feature level" if ok else "FAIL")
sys.exit(0 if ok else 1)
