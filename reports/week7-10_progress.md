# Weeks 7–10 progress log (M2 shot model)

Decisions (user, 2026-09-26): F-a … F-l all defaults (see `prompts/week-7-10.md` §0, commit 1e35d0d).
Branch `week-7-10` (from `main` at f9fcbc9), not pushed.

Format: `iteration N | deliverable | checks run | result | commit`

## Facts verified at start (2026-09-26, whole cache; recorded in docs/data/shots.md with F1)
- Shot feed action codes: `2FGM 2FGA 3FGM 3FGA FTM` every season; blocked attempts `2FGAB`/`3FGAB`
  up to 2016-17 only; 2008-09 → 2014-15 made layups/dunks are `LAYUPMD`/`DUNK` and missed layups
  `LAYUPATT` (as in PBP). No missed free throws (`FTA` never appears).
- `POINTS_A`/`POINTS_B` are the **schedule home / away** score **after** the row, in every
  season (all but 2–22 rows a season match the running sum of the feed exactly).
- `CONSOLE` is the time left in the period (`mm:ss`), `MINUTE` counts on through overtime
  (41–45 first OT, 46–50 second, …); they agree on all but 1–3 rows a season.
- `NUM_ANOT` equals the PBP `NUMBEROFPLAY` of the same action (join key shot ↔ PBP).
- Per team-game FGA and made-FG points of the feed equal the box score (`totr`) in every
  2011-12 → 2025-26 team-game but one (E2017_14 KHI: +15 FGA, +18 points) and E2018 one game
  with an empty box score.
- PBP fouls are not typed (`CM` = any personal foul; `RV` = foul drawn): a free-throw trip can
  be tied to a shot only when a made field goal by the same team at the same clock precedes it
  (and-one). Missed-shot shooting fouls leave no shot row.
- League FT points per team-game, 2011-12 → 2022-23: 14.23, 13.22, 12.59, 13.69, 13.71, 13.51,
  14.51, 13.70, 13.69, 13.40, 12.30, 14.44 (season-to-season sd ≈ 0.6 points).

## Decisions not covered by the task file
- (iteration 1) **`eurohoops shots` and `eurohoops free-throws` are local commands** (like
  `possessions`), not part of `build`: the daily workflow has no full shot/PBP cache and M2 has
  no live use. The shot table stops at 2025-26 (`LAST_SEASON`); the live season stays out.
- (iteration 1) **(−1, −1) on a field goal** (78 rows, the free-throw sentinel) is excluded as
  `unparseable`; the `label_geometry` exclusion is applied from 2011-12 only (2007-10 fit no line).
- (iteration 1) **Blocked (`xFGAB`) and layup/dunk codes are not features**: they encode the
  outcome (a blocked shot is always missed, `DUNK`/`LAYUPMD` are made only).
- (iteration 1) **Reports drifted by live games during the checklist** (`live_scorecard.json`,
  `possessions.json`, `stints_mart.json` gained 2026 games E2026_8..10 from the local marts):
  reverted, never committed from this branch; they are the daily workflow's.
- (iteration 2) **F2 / checklist item 22 is red for a structural reason, not a bug.** League FT
  points per team-game vary between development seasons with sd 0.66 (12.30 in 2021-22 to 14.51
  in 2017-18); rates fitted on the other seasons (LOSO, as F2 specifies) cannot know a season's
  level, so the mean gap is ±0.9-1.3 in 2011, 2013, 2017, 2021, 2022 (validation −0.60). No
  shot-profile model fitted out of season can reach ±0.1. I have **not** widened the tolerance
  (that would make the check vacuous) and have not switched to in-season rates (that breaks
  the LOSO rule). Item 22 stays red and is reported as such; open question for the user.

## Loop log
iteration 1 | F1 shot table (marts shots/shots_excluded, reports/shots.json, docs/data/shots.md) | full §6 1-22 (one run, no edits) | 1-21 PASS; 22 FAIL (F2, see decision) — F1 done: feed = box 99.99%, exclusions ≤ 0.53%/season, two builds identical | see F1+F2 commit
iteration 2 | F2 free-throw generation (ft_team_games, reports/free_throws.json) | same run | code + tests green; item 22 FAIL (structural, diagnosed above) | see F1+F2 commit

## Pre-declared M2 variants and search space (committed before any validation run, 2026-09-26)
No M2 model has been scored on validation (2023-24) or test (2024-25, 2025-26) when this is
committed. Development = 2011-12 → 2022-23, LOSO CV (12 folds). Every choice below is made on
pooled development out-of-fold log loss only; all four variants are reported on validation.

Features (F-b), both families: distance, |angle|, `ZONE`, 2 vs 3, `FASTBREAK`,
`SECOND_CHANCE`, `POINTS_OFF_TURNOVER`, seconds left in the period, period (1-4, OT as one
level), pre-shot margin from the shooter's side, home, season (numeric). No identity column.

Variants (4 of the allowed 6):
1. `spline`: logistic regression, natural cubic splines in distance separately for 2s and 3s
   (knots at development quantiles 2.5%…97.5%, evenly spaced in probability), a 3-pt indicator,
   linear terms for the rest (zone levels with ≥ 1,000 training shots; margin clipped at ±40),
   whitened design, L2 on the whitened coefficients (mean log-loss scale). Grid: knots
   {4, 6, 8} × L2 {1e-5, 1e-4, 1e-3} = 9, chosen by pooled LOSO CV log loss (ties: fewer knots,
   smaller L2).
2. `spline_iso`: variant 1 + isotonic calibration (see below).
3. `lgbm`: LightGBM, `objective=binary`, `deterministic`, `force_row_wise`, `num_threads=6`,
   `bagging_freq=1`, `max_bin=255`, zone as a categorical feature; no early stopping.
   Optuna TPE (seed 20261001, `n_jobs=1`, 60 trials), objective = pooled LOSO CV log loss on
   development seasons with LightGBM seed 20261001. Search space:
   - `num_leaves` int 8-128 (log)
   - `learning_rate` float 0.01-0.2 (log)
   - `n_estimators` int 100-1000, step 50
   - `min_child_samples` int 20-2000 (log)
   - `feature_fraction` float 0.5-1.0
   - `bagging_fraction` float 0.5-1.0
   - `lambda_l2` float 1e-3-100 (log)
   The best parameters are stored in `reports/backtest_m2.json` (`optuna_study`) and reused.
4. `lgbm_iso`: variant 3 + isotonic calibration.

Isotonic calibration (PAV, linear interpolation between fitted points): for development
season s the calibrator is fitted on predictions for every other development season t from
models trained without s and t (66 leave-two-out fits per configuration and seed); validation
and test are calibrated on the development LOSO predictions. No calibrator sees the season it
calibrates.

Seeds (F-l): LightGBM variants use the mean prediction of seeds 20261001-20261005 in every
fold; per-seed CV and validation log loss (mean, sd) are reported, with whether any single seed
would flip the gate. The study is not repeated per seed. Challenger/baseline choice: the
variant with the lower pooled CV log loss in each family (LightGBM on the 5-seed mean).

Gate: challenger − baseline validation log loss, game-level cluster bootstrap 95% CI (1,000
resamples, seed 20261001); also Brier and ECE differences with CIs. `beats_baseline` = mean
difference < 0; the chosen M2 = challenger if it beats the baseline, else the baseline;
`calibrated` = F-f on the chosen M2's validation predictions (ECE ≤ 0.010 with 20 equal-count
bins, every bin with ≥ 500 shots within ±0.02); `passed` = both.

F7 (fixed now, before any player number exists): sampling variance of a player-season's raw
shot-making (100 × mean(actual − xPTS)) is its game-level bootstrap variance (1,000 resamples);
prior N(0, τ²) with τ² = var(raw) − mean(sampling variance) over development player-seasons
with ≥ 100 FGA (floored at 0); 90% posterior intervals. Year-to-year: Pearson r of shrunk
shot-making, players with ≥ 200 FGA in consecutive development seasons, 90% CI by bootstrap
over player pairs (the unit of that correlation; recorded as a decision: F-g's game-level
bootstrap does not apply across seasons). Split-half: Pearson r of raw shot-making on odd vs
even games (by tip-off) in development player-seasons with ≥ 200 FGA, not Spearman-Brown
corrected. Verdict per F-k.
