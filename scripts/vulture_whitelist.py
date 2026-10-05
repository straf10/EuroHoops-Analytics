"""Code built ahead of the pipeline step that will use it; vulture reads this file as usage.

Remove an entry as soon as the pipeline uses the name.
- standings.py: season simulator, PLAN §5.8
- config.py GBL_STINTS_REPORT and M3Grid.dummy_minutes: read by later M3 deliverables
  (H6 dummy variant), not H1.
"""

from eurohoops.config import GBL_STINTS_REPORT, M3, M3Backtest, M3Grid
from eurohoops.models.box_impact import box_only_margins, pir_margins
from eurohoops.models.m5 import (
    ResidualFit,
    apply_platt,
    blend_by_season,
    fit_blend,
    fit_platt,
    fit_residual,
    p_cover,
    p_over,
)
from eurohoops.models.minutes import expected_possessions, oracle_shares, projected_shares
from eurohoops.models.rapm import ModelColumns, build_minutes_rows, fit_decayed_minutes
from eurohoops.models.rapm_posterior import Posterior, noise_variance, posterior, ridge_solution
from eurohoops.models.rest import rest_features
from eurohoops.models.translation import StatFit
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

# week 9-12 subagent B (H2): box-only/PIR baselines and the GBL box-line parser, called by
# eval/m3_backtest.py (subagent A/orchestrator, H5/H7/H8) once the harness is wired up.
box_only_margins
pir_margins

# week 9-12 subagent A
M3Grid.dummy_minutes  # H-f: the rapm_dummy threshold grid, subagent E's variant, not H1's
GBL_STINTS_REPORT  # H3: GBL stints mart report, subagent C
ModelColumns.M  # the general sparse aggregation matrix, exposed for D's posterior (rapm.py)
build_minutes_rows  # D6: decayed on-court minutes per spell, exposed for E's rapm_dummy
fit_decayed_minutes  # D6: convenience snapshot on top of build_minutes_rows, same hook

# weeks 12-14 I7: interval fields reach reports/m4_translation.json through fits_report's asdict
StatFit.delta_lo90
StatFit.delta_hi90

# week 14-16 J1: called by eval/m5_backtest.py (J4, wave 2)
rest_features

# week 14-16 J3: called by eval/m5_backtest.py (J4, wave 2)
ResidualFit
fit_residual
fit_blend
blend_by_season
fit_platt
apply_platt
p_over
p_cover
