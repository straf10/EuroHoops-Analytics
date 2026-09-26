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

DECLARATION 87bb599

## First M2 run was invalid: outcome leakage in three declared features (2026-09-26)
Run: `backtest --model m2 --search` at 87bb599, 09:05Z-10:20Z (4,524 s for study + backtest
together; the split was not logged). MLflow parent run b94decee1ad343dc97a9e441ae606c5b. The
report was **not** committed; its numbers, kept here so nothing is hidden:
- study best CV log loss 0.55304 (trial 50); spline best knots 8, L2 1e-5 (CV 0.55519)
- validation log loss: spline 0.519354, spline_iso 0.518609, lgbm 0.515908, lgbm_iso 0.515932;
  gate lgbm − spline_iso −0.00270 [−0.00346, −0.00199]; lgbm ECE 0.0098 but 4 bins outside
  ±0.02 → calibrated False, gate FAILED.

**Diagnosis.** In the validation reliability table ~6,000 shots had P(make) ≈ 0.999 and all
went in, and the per-season CV log loss fell from 0.63 (2011-14) to 0.51 (2015+). Cause: the
feed sets `FASTBREAK`, `SECOND_CHANCE` and `POINTS_OFF_TURNOVER` **only on made shots** from
2015-16 on (99.8-100% of flagged shots are makes in every season 2015-2025; flags almost never
set before 2014-15). They are scoring tags ("fast-break points", "second-chance points",
"points off turnovers"), i.e. the outcome, not shot context. F-b listed them as features on
the assumption they were context; that assumption is false. The fact is visible in the
development seasons alone (2015-2022); it is now in `reports/shots.json`
(`outcome_coded_flags`), `docs/data/shots.md` and a test.

**Fix (not a choice made on validation numbers):** the three flags are removed from both
model families (a guard raises if they return); the report's per-flag breakdowns become
shot-context breakdowns (home, last 24 s of a period, overtime). Everything else in the
declaration above stands. Because F4's features changed, the Optuna study is re-run (allowed:
"runs only when F4 changes"). The re-run's validation numbers are the second look at
validation; the only change between the two is the removal of the leaking features.
Decision recorded for the user: this departs from the accepted F-b default.

## Amended declaration (before the second run)
Features (both families): distance, |angle|, `ZONE`, 2 vs 3, seconds left in the period,
period (OT as one level), pre-shot margin, home, season (numeric). Variants, grids, search
space, isotonic nesting, seeds and gate: unchanged from the declaration above.

DECLARATION afbc08e

## GATE VERDICT (validation 2023-24; committed before any test-season run)
Second run (the first valid one): `backtest --model m2 --search` at afbc08e, 10:25Z-12:13Z.
MLflow parent run a6d1ae891d434fe5aa410b2ea58210dd.
RUNTIME study 3508 s
RUNTIME backtest 2951 s

- Optuna: best pooled LOSO CV log loss 0.629734 (trial 57 of 60): num_leaves 15, learning_rate
  0.0183, n_estimators 900, min_child_samples 178, feature_fraction 0.955, bagging_fraction
  0.783, lambda_l2 0.140. Spline grid best: 8 knots, L2 1e-4 (CV 0.631577).
- CV log loss: spline 0.631577, spline_iso 0.631513, lgbm 0.629699, lgbm_iso 0.629867 → challenger
  `lgbm`, baseline `spline_iso` (both by CV).
- Validation log loss: spline 0.635097, spline_iso 0.635286, lgbm 0.633267, lgbm_iso 0.633333.
- **Beats baseline: YES.** lgbm − spline_iso = −0.002019, game-level 95% CI [−0.002910,
  −0.001255]; Brier −0.000756 [−0.001027, −0.000509]; ECE +0.001243 [−0.002595, +0.005604].
  5 seeds: validation log loss 0.633300 ± 0.000049 (sd); no single seed flips the verdict.
- **Calibrated: NO.** Chosen M2 = lgbm: validation ECE 0.010438 > 0.010, and 2 of 20 bins with
  ≥ 500 shots fall outside ±0.02 (P 0.401: +0.0338; P 0.694: −0.0236).
- **Gate: FAILED** (exit-gate item 1). Applied literally; no re-run to make it pass. For
  context only (not a choice): the spline's validation ECE is 0.0092, but it is not the chosen
  model because it loses on log loss, and F-f applies to the chosen M2.
- Hypothesis for the calibration miss: the failing bins sit in the 0.40 and 0.69 regions, and
  development CV per season shows season-level shifts of league shooting (as with FT points in
  F2); a model fitted on 2011-2022 cannot know the 2023-24 level exactly. Isotonic calibration
  (fitted on other seasons) does not fix it (lgbm_iso ECE 0.0116).
- **Runtime budget: RED.** The backtest (5 seeds, nested isotonic: 395 LightGBM fits of 900
  trees) takes 2,951 s > 2,400 s. Feature building is ~0.5 s of each ~6 s fit, so no
  number-preserving speed-up reaches 40 min; meeting it needs a change to the declared
  computation (fewer threads/seeds or no nested isotonic for LightGBM), which I did not make.

VERDICT 84308f9

## TEST RUN (once, after verdict commit 84308f9): 2026-09-26 12:14Z-13:07Z
`backtest --model m2 --score-test`; MLflow parent run d5ca70bafbbc411791aff2c649bb2e65.
RUNTIME backtest 3130 s (with the test seasons; over the 2,400 s budget)
Every development and validation number is identical to the verdict report (checked).
- Test (2024-25 + 2025-26, 93,119 shots) log loss: spline 0.635433, spline_iso 0.635587, lgbm
  0.633467, lgbm_iso 0.633374.
- lgbm − spline_iso: −0.002119 [−0.002601, −0.001658]; Brier −0.000842 [−0.001034, −0.000654];
  ECE +0.000633 [−0.003120, +0.003892]; no seed flips. lgbm test ECE 0.012626 → not
  calibrated on test either. Same verdict as validation: beats the baseline, fails F-f.
iteration 3 | F3 spline, F4 LightGBM + Optuna, F5 evaluation/gate/test | fast gate 1-6, 14, 19 before each commit; F3/F4 tests; study 3508 s; backtest 2951 s (RED > 2400 s) | declared → invalid run (leak) → amended declaration afbc08e → verdict 84308f9 (FAILED: not calibrated) → test run | see TEST line
TEST 6b2a5c1

## F8 chart subjects (named before drawing, 2026-09-26)
Validation season 2023-24, xPTS from the development-fitted chosen M2 (lgbm). Teams (source
codes): PAN, OLY, MAD (shown as PAO, OLY, RMB where `DISPLAY_CODES` renames them). Players: the
three with the most 2023-24 FGA in the shot table: P011948 HOWARD, MARKUS (589), P005985
JAMES, MIKE (541), P012774 NUNN, KENDRICK (471).
Charts, each opened with the Read tool after the redraw (titles on two lines, grey hex edges so
a white "as expected" hexagon is visible; the first draw clipped long titles and hid white hexes):
- xpts_surface_2023.png: court lines in place (arc, corners, key, restricted area); legend and title readable; sequential scale (not a residual chart); rim ≈ 1.8, threes ≈ 1.1-1.3, long 2s ≈ 0.7.
- team_PAO_2023.png: lines in place; readable; scale centred at 0 (TwoSlopeNorm ±0.6); +2.9 per 100.
- team_OLY_2023.png: lines in place; readable; centred at 0; −1.6 per 100.
- team_RMB_2023.png: lines in place; readable; centred at 0; +7.9 per 100.
- player_P011948_2023.png (Howard): lines in place; readable; centred at 0; sparse (≥ 5 shots per hexagon); +13.7 per 100.
- player_P005985_2023.png (James): lines in place; readable; centred at 0; +5.1 per 100.
- player_P012774_2023.png (Nunn): lines in place; readable; centred at 0; few hexagons reach 5 shots; +2.7 per 100.
iteration 4 | F6 team shot quality + F7 player shot-making + F8 charts | fast gate 1-6 (385 tests, 95.6%); F6 calibration in the large RED (ratio 0.977-1.020, 5/12 development seasons within 0.5%); F7 verdict "stable enough to show" (y2y r 0.396 [0.311, 0.481], split-half 0.315) = F-k rule (test); F8 7 charts opened and logged | see F6-F8 commit
- (F9) **Deliberate fold-rule breaks, each shown once and reverted** (tests/test_m2_leakage.py):
  (a) LOSO training set includes the scored season -> own-season test FAILED [spline, lgbm];
  (b) the development-wide fit also sees validation/test shots -> development-fitted test FAILED [spline, lgbm];
  (c) the isotonic calibrator for season s fitted on all development out-of-fold values (incl. s) -> own-season test FAILED [spline, lgbm].
  After each revert: 5/5 pass. Guards in the same tests prove the edits move what they should.
- (F10) Model card test catches an edited table value (0.633267 -> 0.633268: 2 tests failed; reverted, 3 pass).
iteration 5 | F9 MLflow + leakage, F10 model card | leakage 5/5 (+3 deliberate breaks shown); card test 3/3 (+ mutation shown) | green | see F10 commit

- (iteration 6) **Stopped final run** (13:19Z-16:50Z): CPU contention from two `astro dev` servers (UI worktree) made numpy/BLAS threads spin; M1 backtest 3,298 s vs 224 s quiet, item 13 FAIL on runtime, item 15 cut off by the stop. Servers stopped with the user's approval; restarted on a quiet machine.
- (iteration 6) **Number-preserving speed-up:** LightGBM `zone_code` built with a vectorised map instead of a per-shot Python loop (0.51 s -> 0.02 s per call, ~1,100 calls per backtest); codes verified identical on all development shots, item 24 checks the report is byte-identical.
iteration 6 | final full §6 run (349acb9, quiet machine, no edits between) | items 1-30 top to bottom | 26 PASS, 4 FAIL (22 FT reconciliation, 23 recorded runtime, 24 runtime 3,072 s, 25 calibration in the large); stop condition NOT met -> closeout INCOMPLETE | see closeout commit
