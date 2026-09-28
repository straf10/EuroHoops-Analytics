"""Code built ahead of the pipeline step that will use it; vulture reads this file as usage.

Remove an entry as soon as the pipeline uses the name.
- standings.py: season simulator, PLAN §5.8
- config.py M3_GBL/M3_PLAYERS_REPORT/GBL_STINTS_REPORT and M3Grid.dummy_minutes: read by later
  M3 deliverables (H6 dummy variant, H8 GBL transfer, H7 player report), not H1.
"""

from eurohoops.config import GBL_STINTS_REPORT, M3_GBL, M3_PLAYERS_REPORT, M3Grid
from eurohoops.models.rapm import ModelColumns, build_minutes_rows, fit_decayed_minutes
from eurohoops.standings import EUROLEAGUE_2026, GBL_2026, Format, Series, rank

Series.best_of
Format.regular_season_rounds
Format.playoffs_direct
Format.play_in
Format.eliminated
Format.relegated
Format.series
Format.sources
Format.unverified
EUROLEAGUE_2026
GBL_2026
rank

# week 9-12 subagent A
M3Grid.dummy_minutes  # H-f: the rapm_dummy threshold grid, subagent E's variant, not H1's
M3_GBL  # H8: GBL transfer report spec, subagent C/orchestrator
M3_PLAYERS_REPORT  # H7: per-player ratings report, the orchestrator
GBL_STINTS_REPORT  # H3: GBL stints mart report, subagent C
ModelColumns.M  # the general sparse aggregation matrix, exposed for D's posterior (rapm.py)
build_minutes_rows  # D6: decayed on-court minutes per spell, exposed for E's rapm_dummy
fit_decayed_minutes  # D6: convenience snapshot on top of build_minutes_rows, same hook
