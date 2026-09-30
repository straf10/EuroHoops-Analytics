"""Checklist item 39: crosswalk invariants on the real marts.

Every GBL ``player_box`` id (2018-2025) and every EuroLeague box id of a rated 2007-2025 game is
in ``player_xwalk`` exactly once; the schema holds; every ``match`` override links its two ids
into one person and every ``no_match`` one leaves them apart (Yurtseven: one person with two
EuroLeague ids; the two Marko Simonovic: two people).
"""

import sys

import pandas as pd

from eurohoops.config import ENTITY_OVERRIDES, ENTITY_SEASONS, GBL_PLAYER_BOX, MART_PATH
from eurohoops.entity.pipeline import read_overrides
from eurohoops.entity.xwalk import XWALK_SCHEMA
from eurohoops.marts import read_table

xwalk = read_table(MART_PATH, "player_xwalk")
names = read_table(MART_PATH, "player_names")
if xwalk is None or names is None:
    sys.exit("player_xwalk/player_names missing; run: eurohoops entity")
XWALK_SCHEMA.validate(xwalk)
ok = True
keys = list(zip(xwalk["competition"], xwalk["source_id"], strict=True))
dupes = len(keys) - len(set(keys))
print(f"crosswalk rows {len(xwalk)}, persons {xwalk['person_id'].nunique()}, duplicate ids {dupes}")
ok &= dupes == 0
box = pd.read_parquet(GBL_PLAYER_BOX, columns=["game_id", "player_id"])
box = box[box["game_id"].str.slice(3, 7).astype(int).isin(ENTITY_SEASONS)]
gbl_missing = set(box["player_id"]) - {s for c, s in keys if c == "gbl"}
el_ids = set(names.loc[names["competition"] == "euroleague", "source_id"])
el_missing = el_ids - {s for c, s in keys if c == "euroleague"}
print(f"GBL player_box ids without a person: {len(gbl_missing)}; EL box ids: {len(el_missing)}")
ok &= not gbl_missing and not el_missing
person = dict(zip(keys, xwalk["person_id"], strict=True))
for row in read_overrides(ENTITY_OVERRIDES).itertuples(index=False):
    a = person.get((row.competition_a, row.source_a))
    b = person.get((row.competition_b, row.source_b))
    held = a is not None and b is not None and (a == b) == (row.decision == "match")
    print(f"override {row.source_a} {row.decision} {row.source_b}: {'held' if held else 'BROKEN'}")
    ok &= held
print("crosswalk invariants:", "hold" if ok else "FAIL")
sys.exit(0 if ok else 1)
