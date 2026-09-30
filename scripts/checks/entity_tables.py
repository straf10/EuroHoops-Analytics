"""Checklist items 38-39 helper: sha256 of the ``player_names`` and ``player_xwalk`` marts (sorted
CSV), so two ``eurohoops entity`` builds can be compared."""

import hashlib

from eurohoops.config import MART_PATH
from eurohoops.marts import read_table

for table in ("player_names", "player_xwalk"):
    frame = read_table(MART_PATH, table)
    if frame is None:
        raise SystemExit(f"{table} missing; run: eurohoops entity")
    frame = frame.sort_values(list(frame.columns)).reset_index(drop=True)
    digest = hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()
    print(table, len(frame), digest)
