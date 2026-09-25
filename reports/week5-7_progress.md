# Weeks 5–7 progress log

Decisions (user, 2026-09-25): E-a … E-g all defaults (see `prompts/week-5-7.md` §0).
Branch `week-5-7` (from `main` at 55f0791), not pushed.

Format: `iteration N | deliverable | checks run | result | commit`

## Baseline (before any change, 2026-09-25)
Local gate 1-6 green on the branch start (224 tests, coverage 98.75%).

## Facts verified at start (recorded in docs/data/possessions.md with E1)
- EL box JSON: every season 2007-2026, `totr` (team totals) = sum of player lines + `tmr`
  (team row: team rebounds and team turnovers) for FGA2/FGA3/FTA/OREB/DREB/TOV/points, in
  every cached box score (5,008 games; E2018_21 has no players).
- ESAKE box HTML: `ΣΥΝΟΛΟ` = players + `ΟΜΑΔΙΚΑ - ΠΑΓΚΟΣ` row for REBS/D.REBS/O.REBS/TO, every
  cached game 2018-2025; the team row is always 0 in 2018-19..2020-21 and mostly non-zero from
  2021-22 on (ESAKE started recording team rebounds/turnovers during 2021-22).
- GBL PBP (2018-20) has rebound (player + team) and turnover templates.

## Decisions not covered by the task file
- (iteration 1) **Checklist item 7 runs in a clean copy of `web/`** (scratchpad): the user has
  `astro preview --port 4321` and `astro dev` running (started 16:37 and 17:03 local), which
  lock `web/node_modules/**.node` files, so `npm ci` in place fails with EPERM (and its
  partial delete was repaired with `npm install`, lockfile untouched). The copy runs the same
  `cp tests/fixtures/site.json src/data/site.json && npm ci && npm run build`. I did not stop
  the user's servers.
- (iteration 1) **Intermediate iterations:** §6 items 12-17 belong to deliverables not built
  yet; they are reported as "not built yet", every other item must be green before a commit.
  The stop condition still requires all 20 in one run.
- (iteration 1) **One pandera contract for both competitions' team lines**
  (`TEAM_GAMES_SCHEMA`, with a `source` column): both sources feed one table with the same
  columns, so one contract (two rows per game, poss_game = mean, counts ≥ 0) covers each.
- (iteration 1) **EL possession agreement 89.0% < 90%** on the sample: explained in
  `docs/data/possessions.md` (0.44 FT weight vs the EuroLeague-implied 0.42, which gives 94%).
  `team_games` keeps R1's 0.44 as the task specifies.
- (iteration 1) **GBL source order:** ESAKE totals when they reproduce the result, else the PBP
  counts when those do, else listed missing. 2018-19..2020-21 ESAKE totals lack team rebounds
  and team turnovers (documented bias ≈ +0.8 possessions per team).
- (iteration 1) New commands: `build` also writes `team_games` + `team_games_missing`;
  `eurohoops possessions` writes `reports/possessions.json` (kept out of `build` so the daily
  workflow, which has no EuroLeague box cache, never rewrites the report).

## Loop log
iteration 1 | E1 team_games possession mart | §6 1-11, 18, 19 (12-17 not built yet; 7 in a clean web/ copy) | green; 6378/6387 games with rows, 9 listed missing, 0 points mismatches, EL sample 89.0% within 2 (explained) | be1c18e
iteration 2 | E2 stints mart | §6 1-12, 18, 19 (13-17 not built yet) | green; 2011-14 95.9%, 2015+ 98.9%, points=final for all passing games, two builds identical (tables + report) | 3dda083

## Pre-declared M1 variants (committed before any M1 run on real data, 2026-09-25)
No M1 code has run on real data when this is committed. The gate uses the declared variant
with the best **tuning** log loss; every declared variant is reported on validation.

Shared by all variants (hyperparameters chosen on the tuning seasons only):
- Ratings: weighted ridge on team-game rows, `100·points/poss_game = μ + h·home + off[team] −
  def[opp]` (home = +1 home row, −1 away row, 0 neutral), weight = poss_game × 0.5^(age/half-life)
  × carry^(seasons back); μ and h unpenalised, off/def penalised by `ridge` (in possessions).
  Grid (by tuning log loss, Normal margin with tuning-RMS σ): half_life_days {60, 120, 240,
  480} × carry {0.25, 0.5, 0.75, 1.0} × ridge {250, 500, 1000, 2000, 4000} = 80.
- Pace: ridge on game rows, `poss_game·40/minutes = μ_p + pace[home] + pace[away]`, weight =
  decay as above; grid (by tuning MSE of possessions per 40) half_life_days {60, 120, 240, 480}
  × carry {0.25, 0.5, 0.75, 1.0} × ridge_games {2, 5, 10, 20} = 64.
- Refit before each round's first tip-off on games that tipped off before it.
- Totals: Normal(expected total, σ_T), σ_T = tuning RMS.

Margin distribution variants (E-e), 4 of the allowed 6:
1. `normal_const`: Normal(m, σ), σ = tuning RMS residual.
2. `normal_pace`: Normal(m, σ0·sqrt(P/P̄)), σ0 by maximum likelihood on tuning, P̄ = mean expected
   pace on tuning.
3. `student_t_const`: m + s·T_df; df from {3, 4, 5, 7, 10, 15, 20, 30, 50} by tuning log loss,
   s by maximum likelihood on tuning for each df.
4. `student_t_pace`: as 3 with s·sqrt(P/P̄).

Comparison Elo (E-b): the existing grid (EL `DEFAULT_GRID`; GBL its wide grid) re-tuned on the
M1 tuning seasons, replayed from the first warm-up season; margin σ = tuning RMS; totals =
the existing 2-season baseline with σ_T = tuning RMS. B0: home-win rate and margin of the
non-neutral games up to the last tuning season.
iteration 3 | E3 M1 model | §6 1-12, 13 (E3 tests 28 passed; full EL M1 backtest 122 s < 600 s), 18, 19 (14-17 not built yet) | green | 0341eb4

## GATE VERDICT (validation 2023-24; committed before any test-season run)
First validation run: 2026-09-25, after the variant declaration (f921181). Variant = the
declared variant with the best tuning log loss. Diffs are M1 − comparison Elo, paired
bootstrap 95% CI (1000 resamples, seed 20260924).

**EuroLeague: PASS** (variant `student_t_const`, df 7, s 10.41)
- log loss 0.5872 vs 0.5885: −0.0013 [−0.0084, +0.0064] (CI spans 0: not significant)
- Brier 0.2012 vs 0.2014: −0.0003 [−0.0034, +0.0032]; ECE 0.043 vs 0.057
- margin CRPS −0.026 [−0.092, +0.045]; totals CRPS −1.007 [−1.385, −0.606] (Elo has no
  totals model: it uses the 2-season baseline)
- rating grid best on edge: half-life 480 d (max) and ridge 250 (min); pace: half-life 60 d,
  ridge 2 (both min). Declared grids are kept; reported as a limitation.

**GBL: PASS** (variant `student_t_pace`, df 7, s 10.65 at P̄ 73.55)
- log loss 0.4328 vs 0.4362: −0.0035 [−0.0356, +0.0242] (not significant; n = 163)
- Brier 0.1393 vs 0.1433; ECE 0.060 vs 0.095
- For GBL the spread and total errors are the more informative metrics (PLAN §5: many
  lopsided games): margin CRPS −0.172 [−0.419, +0.072] (not significant), margin MAE 9.55 vs
  9.90; totals CRPS −0.835 [−1.295, −0.388] (significant, vs the baseline).
- rating grid best on edge: ridge 250 (min).

Both competitions pass the pre-registered rule (point estimate below Elo), but neither
log-loss gain is distinguishable from zero. Per E-g, M1 goes live for both.
iteration 4 | E4 (part 1) backtest + validation gate + verdict | §6 1-12, 13, 18, 19 (run before the commit, same tree) | green; verdict EL PASS, GBL PASS | 37cf85f
TEST RUN (once, after verdict commit 37cf85f): 2026-09-25T15:09Z
iteration 5 | E4 (part 2) test seasons scored once | check 15 (order declaration f921181 < verdict 37cf85f < test fc7de59; two runs byte-identical, both comps), tests | green; test EL -0.0025 [-0.0077,+0.0027], GBL +0.0076 [-0.0232,+0.0360] | fc7de59
