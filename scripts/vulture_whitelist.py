"""Code built ahead of the pipeline step that will use it; vulture reads this file as usage.

Remove an entry as soon as the pipeline uses the name.
- standings.py: season simulator, PLAN §5.8
- config.py GBL_STINTS_REPORT and M3Grid.dummy_minutes: read by later M3 deliverables
  (H6 dummy variant), not H1.
"""

from eurohoops.config import (
    GBL_STINTS_REPORT,
    M3,
    M6,
    M6_BOARD,
    M6_GBL,
    M6_PROJECTIONS,
    M6_SIMILARITY,
    SIM_UNGATED,
    M3Backtest,
    M3Grid,
    M6Backtest,
)
from eurohoops.eval.m5_backtest import predict_games
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
from eurohoops.models.player_seasons import (
    PROJECTED_STATS,
    build_player_seasons,
    player_ages,
    rate_table,
)
from eurohoops.models.rapm import ModelColumns, build_minutes_rows, fit_decayed_minutes
from eurohoops.models.rapm_posterior import Posterior, noise_variance, posterior, ridge_solution
from eurohoops.models.rest import rest_features
from eurohoops.models.rotation import projected_shares_variant
from eurohoops.models.similarity import (
    SIMILAR_SCHEMA,
    SelfRetrieval,
    embed,
    neighbours,
    raw_features,
    self_retrieval_rate,
)
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

# week 14-16 J2: called by eval/m5_backtest.py (J4, wave 2)
projected_shares_variant

# week 14-16 J4: predict_games is the public per-choice entry point the leakage suite (J5) and
# the J8 gap recomputation test call; src does not call it
predict_games

# weeks 16-18 L0: the M6 spec, report paths and the shared player-season frame, read by L1-L4
# (wave 1), the L6 harness, `eurohoops project` (L9) and `sim-ungated` (D1)
M6Backtest.players_report
M6Backtest.min_poss
M6Backtest.half_lives
M6Backtest.coverage_band
M6
M6_GBL
M6_PROJECTIONS
M6_BOARD
M6_SIMILARITY
SIM_UNGATED
build_player_seasons
player_ages
rate_table
PROJECTED_STATS

# weeks 16-18 D: the "plays like" search, called by the L9 `eurohoops project` command and the
# L6 harness (later wave); self_retrieval_rate/SelfRetrieval.z are the embedding's own check
SIMILAR_SCHEMA
embed
neighbours
raw_features
self_retrieval_rate
SelfRetrieval.z

# weeks 16-18 C: the over/under board (L3) is called by the report writer and the L9 live run
from eurohoops.models.board import (  # noqa: E402
    RetentionBacktest,
    board,
    fg3_observations,
    on_off_observations,
    retention_backtest,
    shot_making_observations,
    stability,
)

board
fg3_observations
on_off_observations
retention_backtest
shot_making_observations
stability
RetentionBacktest.contrast_z  # read by the report writer and the board test

# weeks 16-18 B: the aging curve, called by the projection (L1) and the L6 harness (wave 2)
from eurohoops.models.aging import MIN_AGE_PAIRS, aging_curve, apply  # noqa: E402

MIN_AGE_PAIRS
aging_curve
apply

# weeks 16-18 A: L1 projections, called by the L6 harness and `eurohoops project` (L9), wave 2
from eurohoops.models.projection import (  # noqa: E402
    FLAGS,
    IMPACT_SCHEMA,
    VARIANTS,
    AgeAdjust,
    ProjectionParams,
    Translation,
    fit_drift,
    project,
    variant_params,
)

FLAGS
IMPACT_SCHEMA
VARIANTS
AgeAdjust
ProjectionParams
Translation
fit_drift
project
variant_params

# weeks 16-18 E: the read-only API; FastAPI calls the response classes' attributes and uvicorn
# calls the factory named in the Dockerfile (`uvicorn --factory eurohoops.api.app:local_app`)
from eurohoops.api.app import CompactJSON, IndentedJSON, local_app  # noqa: E402

CompactJSON.media_type
CompactJSON.render
IndentedJSON.media_type
IndentedJSON.render
local_app

# weeks 16-18 H (L10): route_for is the inverse of file_for, read by the contract check
# (scripts/checks/api_export.py, tests/test_api_contract.py); write_stats is the pre-L10 writer,
# kept as the reference the byte-identical check compares the API export against
from eurohoops.api.export import route_for  # noqa: E402
from eurohoops.stats.export import write_stats  # noqa: E402

route_for
write_stats
