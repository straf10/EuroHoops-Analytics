"""Code built ahead of the pipeline step that will use it; vulture reads this file as usage.

Remove an entry as soon as the pipeline uses the name.
- standings.py: season simulator, PLAN §5.8
- config.py M3 specs and models/minutes.py: M3 backtest (week 9-12 H1-H8), wired in wave 1
"""

from eurohoops.config import GBL_STINTS_REPORT, M3, M3_GBL, M3_PLAYERS_REPORT, M3Backtest, M3Grid
from eurohoops.models.minutes import expected_possessions, oracle_shares, projected_shares
from eurohoops.models.rapm_posterior import Posterior, noise_variance, posterior, ridge_solution
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

# week 9-12 (M3)
M3Grid.dummy_minutes
M3Backtest.projection_games
M3Backtest.bootstrap_resamples
M3Backtest.bootstrap_seed
M3
M3_GBL
M3_PLAYERS_REPORT
GBL_STINTS_REPORT
projected_shares
oracle_shares
expected_possessions

# week 9-12 subagent D (H4: models/rapm_posterior.py, the closed-form Bayesian RAPM posterior).
# `posterior_from_normal_equations` (the function subagent A's walk-forward fit calls once
# merged) is already called from `posterior()` within the same module, so vulture sees it as
# used; these are its other public entry points, not yet called from src until A wires them in.
Posterior.interval
posterior
ridge_solution
noise_variance
