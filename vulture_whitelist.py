"""Code built ahead of the pipeline step that will use it; vulture reads this file as usage.

Remove an entry as soon as the pipeline uses the name.
- parse/shots.py: shot-quality (xPTS) model, PLAN §5.3
- standings.py: season simulator, PLAN §5.8
"""

from eurohoops.parse import shots
from eurohoops.standings import EUROLEAGUE_2026, GBL_2026, Format, Series, rank

shots.FIRST_VALIDATED_SEASON
shots.to_court_coords
shots.beyond_three_line
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
