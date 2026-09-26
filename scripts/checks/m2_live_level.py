"""Pre-registered live evaluation of the season-level variant (weeks 7-10b G5, rule declared in
reports/week7-10b_progress.md under LEVEL_DECLARATION). NOT part of the checklist: run it once,
after the last game of 2026-27, with ``--after-season``.

2026-27 is the only clean hold-out for ``lgbm_level``: validation and test were seen before it
was declared. The rule, fixed before any 2026-27 number exists:
- 2026-27 shots from the cache with the F1 rules (the shot table, live season included);
- ``lgbm`` = the 5 seeds refitted on 2011-12 -> 2025-26 with the stored Optuna parameters;
- ``lgbm_level``: offsets from 2026-27 games that tipped off before the shot's game, prior = the
  full-season shift of 2025-26 under the 5 seeds fitted on 2011-12 -> 2024-25, k = the report's
  validation/test k (``level_variants.variants.lgbm_level.shrinkage.later``);
- both scored on every 2026-27 shot: F-f (ECE <= 0.010, 20 equal-count bins, every bin with >=
  500 shots within +-0.02), log loss, Brier, and lgbm_level - lgbm log loss with the game-level
  bootstrap 95% CI (1,000 resamples, seed 20261001).
- ``lgbm_level`` is calibrated on 2026-27 iff it meets F-f; it improves on ``lgbm`` iff the CI's
  upper bound is below 0.
Writes reports/m2_live_level_2026.json.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from eurohoops.config import EUROLEAGUE
from eurohoops.eval.m2_backtest import BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, SEEDS
from eurohoops.eval.shot_metrics import calibrated, cluster_bootstrap, scores, shot_log_loss
from eurohoops.marts import read_games, read_table
from eurohoops.models.season_level import Shrinkage, logit, mle_shift, season_offsets, shifted
from eurohoops.models.xpts_gbm import fit_gbm
from eurohoops.parse import shot_table

LIVE = 2026
MART = Path("data/marts/eurohoops.duckdb")
OUT = Path("reports/m2_live_level_2026.json")

if "--after-season" not in sys.argv:
    print(__doc__)
    sys.exit(2)
games = read_games(MART, EUROLEAGUE.name)
season = games[games["season"] == LIVE]
if season.empty or not season["played"].all():
    print(f"{LIVE}-27 is not finished ({int((~season['played']).sum())} games unplayed): not run")
    sys.exit(2)

report = json.loads(Path("reports/backtest_m2.json").read_text(encoding="utf-8"))
params = report["lightgbm_params"]
stored = report["level_variants"]["variants"]["lgbm_level"]["shrinkage"]["later"]
shrink = Shrinkage(
    math.inf if stored["k"] is None else float(stored["k"]),
    float(stored["tau2"]),
    float(stored["v_bar"]),
    int(stored["consecutive_pairs"]),
)

history = read_table(MART, "shots")
assert history is not None
history = history[history["validated_season"]]
shot_table.LAST_SEASON = LIVE  # the F1 rules, with the live season let in (this script only)
live = shot_table.build_shot_table(EUROLEAGUE.raw_dir, season).shots
live = live[live["season"] == LIVE].reset_index(drop=True)


def seed_mean(train: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    return np.mean([fit_gbm(train, params, seed).predict(target) for seed in SEEDS], axis=0)


p_live = seed_mean(history[history["season"] <= LIVE - 1], live)
last = history[history["season"] == LIVE - 1].reset_index(drop=True)
p_last = seed_mean(history[history["season"] <= LIVE - 2], last)
prior = mle_shift(logit(p_last), last["made"].to_numpy(dtype=np.float64))

y = live["made"].to_numpy(dtype=np.float64)
ticks = pd.to_datetime(season.set_index("game_id")["tipoff_utc"], utc=True)
tip = np.asarray(live["game_id"].map(ticks.dt.tz_convert(None).astype("int64")), dtype=np.int64)
p_level = shifted(p_live, season_offsets(p_live, y, tip, prior, shrink))

games_of = live["game_id"].to_numpy()
mean, low, high = cluster_bootstrap(
    shot_log_loss(p_level, y) - shot_log_loss(p_live, y),
    games_of,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
)
result = {
    "season": LIVE,
    "shots": len(live),
    "prior": round(prior, 6),
    "shrinkage": stored,
    "lgbm": scores(p_live, y),
    "lgbm_level": scores(p_level, y),
    "lgbm_meets_f_f": calibrated(scores(p_live, y)),
    "lgbm_level_meets_f_f": calibrated(scores(p_level, y)),
    "level_minus_lgbm_log_loss": {"mean": round(mean, 6), "ci95": [round(low, 6), round(high, 6)]},
    "level_improves": bool(high < 0.0),
}
OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(
    f"{LIVE}-27, {len(live)} shots: lgbm ECE {result['lgbm']['ece']}, lgbm_level ECE "
    f"{result['lgbm_level']['ece']}; lgbm_level meets F-f: {result['lgbm_level_meets_f_f']}; "
    f"level - lgbm log loss {result['level_minus_lgbm_log_loss']}; improves: "
    f"{result['level_improves']}; wrote {OUT}"
)
