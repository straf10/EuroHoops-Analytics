# Weeks 5–7 close-out: team model M1

Branch `week-5-7` (from `main` 55f0791), not pushed. Decisions E-a … E-g: all defaults
(user, 2026-09-25). Loop log and every decision: `reports/week5-7_progress.md`.
Model card: `docs/models/m1.md`.

**Status: PASS.** All eight deliverables are done and the full §6 checklist passed top to
bottom in one run (commit af1ca1c, clean tree before and after, 16:18–16:29 UTC; output below).
9 loop iterations of the 30 budgeted.

## Checklist (single consecutive run on af1ca1c)
| # | Check | Result |
|---|---|---|
| 1 | `uv sync --frozen` | PASS |
| 2 | ruff check | PASS |
| 3 | ruff format --check | PASS |
| 4 | mypy src (strict) | PASS |
| 5 | pytest + coverage | PASS (329 tests, 99.13%) |
| 6 | vulture | PASS |
| 7 | web build from the fixture | PASS (in a clean copy of `web/`, see decisions) |
| 8 | Elo backtests reproduce every old value | PASS |
| 9 | predictions append-only, Elo model_version unchanged | PASS (0 changed lines; 425e6393 / df05260c) |
| 10 | stints sample byte-identical to main | PASS (50/50) |
| 11 | E1 team_games coverage, points, possessions | PASS (6378/6387 games, 9 listed; 0 mismatches; sample 89.0%, explained) |
| 12 | E2 stints mart | PASS (2011-14 95.87%, 2015+ 98.94%; two builds identical) |
| 13 | E3 unit tests + EL runtime | PASS (28 tests; full EL M1 backtest 111 s < 600 s) |
| 14 | E6 leakage tests | PASS (26) |
| 15 | M1 reports, order, reproducibility | PASS (declaration f921181 < verdict 37cf85f < test fc7de59; two runs identical) |
| 16 | MLflow parent + children | PASS (query output below) |
| 17 | E7 live dry run | PASS (both competitions went live) |
| 18 | build + score + publish twice, tree unchanged | PASS |
| 19 | actionlint (both workflows) | PASS |
| 20 | screenshots 1440/390, light/dark | PASS (0 px overflow; `reports/screenshots/scorecard_m1_*.png`) |

Nothing BLOCKED, nothing SKIPPED.

## Gate verdict (validation 2023-24; M1 minus comparison Elo, paired bootstrap 95% CI)
Committed in 37cf85f before any test-season run.

| | EuroLeague (`student_t_const`, df 7) | GBL (`student_t_pace`, df 7) |
|---|---|---|
| Games | 331 | 163 |
| Log loss M1 / Elo | 0.5872 / 0.5885 | 0.4328 / 0.4362 |
| Log loss diff | −0.0013 [−0.0084, +0.0064] | −0.0035 [−0.0356, +0.0242] |
| Brier M1 / Elo | 0.2012 / 0.2014, diff −0.0003 [−0.0034, +0.0032] | 0.1393 / 0.1433, diff −0.0040 [−0.0138, +0.0057] |
| ECE M1 / Elo | 0.043 / 0.057 | 0.060 / 0.095 |
| Margin CRPS diff | −0.026 [−0.092, +0.045] | −0.172 [−0.419, +0.072] |
| Totals CRPS diff | −1.007 [−1.385, −0.606] | −0.835 [−1.295, −0.388] |
| **Verdict** | **PASS** (not significant) | **PASS** (not significant) |

For the GBL, spread and total errors are the more informative metrics (PLAN §5): margin CRPS
improves but not significantly; totals improve significantly, but only against the Elo side's
two-season-mean totals baseline (Elo has no totals model).

## Test (2024-25 + 2025-26, scored once after the verdict)
| | EuroLeague (n 732) | GBL (n 340) |
|---|---|---|
| Log loss M1 / Elo | 0.6220 / 0.6246 | 0.4875 / 0.4799 |
| Log loss diff | −0.0025 [−0.0077, +0.0027] | +0.0076 [−0.0232, +0.0360] |
| Margin CRPS diff | −0.021 [−0.064, +0.027] | −0.182 [−0.429, +0.093] |
| Totals CRPS diff | −0.839 [−1.125, −0.574] | −0.502 [−0.841, −0.180] |

On the GBL test seasons M1 is worse than Elo in log loss (within noise).

## Variants
Four declared before the first validation run (normal_const, normal_pace, student_t_const,
student_t_pace); all four reported on validation and test in the reports and the model card.
None post-hoc. The rating/pace grids (80 and 64 points) were also declared up front. EuroLeague
rating best is on the grid edge (half-life 480 d max, ridge 250 min); GBL rating ridge 250 min.

## Possessions and stints
- EuroLeague 50-game sample, PBP count vs box formula within ±2: 89.0% of team-games (80.0% of
  games), mean gap −0.34. Target 90% missed by one team-game; explained in
  `docs/data/possessions.md` (0.44 FT weight vs the EuroLeague-implied 0.421 → 94.0%).
  Population: 87.3% (2011-14), 89.3% (2015+).
- Stints mart: 4,246 games, 141,625 stints; pass rate 95.87% (2011-14, ≥ 95%) and 98.94%
  (2015+, ≥ 98.5%); stint points = final in every passing game; two builds identical.

## Heteroscedasticity (R4, tuning seasons)
No evidence that margin variance grows with expected pace or the rating gap: EuroLeague
p = 0.248 (pace), 0.264 (|gap|), LR test of a pace-scaled σ p = 0.284; GBL 0.542, 0.407, 0.613.
Heavier tails (Student-t, df 7) help slightly; pace scaling does not.

## MLflow run ids (the final run's backtests, `mlruns/mlflow.db`)
- EuroLeague parent `b284b7618e6d4f12834542de77a8cb36`; children normal_const
  `4e1426b284f64d069b34888a11b64c81`, normal_pace `256897f8155747f091afcdbc959c533f`,
  student_t_const `d3af94bf907a4ece8d4c67390c10a7c7`, student_t_pace `1383bff19ea6473eb266f893312d2ab9`.
- GBL parent `7034c944b8cd40d488be5427b9669207`; children normal_const
  `3f591627b1a5466f97bacfcb41b7ddfc`, normal_pace `f2705bb1aaea463b969314cb4ab63e9d`,
  student_t_const `db55f66d1ade46c6b7b103e28a6638bc`, student_t_pace `3830d3a4447847f3a1c0d1c5918228ed`.

## Decisions the task file did not cover
1. §6 item 7 runs in a clean copy of `web/` (scratchpad): the user's `astro preview`/`astro dev`
   processes lock `web/node_modules`, so `npm ci` in place fails (EPERM). Its partial delete was
   repaired with `npm install` (lockfile untouched). The servers were not stopped.
2. One pandera contract (`TEAM_GAMES_SCHEMA`, with a `source` column) for both competitions'
   team lines.
3. GBL source order: ESAKE totals if they reproduce the result, else the 2018-20 PBP counts if
   those do, else listed missing. ESAKE 2018-19 → 2020-21 totals lack team rebounds/turnovers
   (≈ +0.8 possessions per team), documented.
4. `eurohoops possessions` and `eurohoops stints --mart` write their reports outside `build`,
   so the daily workflow never rewrites them.
5. The spike's minutes check broke ties in hash order (non-deterministic reasons); it now
   iterates players sorted. The 50-game sample report is unchanged.
6. MLflow store: SQLite under `mlruns/` (MLflow 3.16.1 refuses the plain file store without
   `MLFLOW_ALLOW_FILE_STORE=true`). Local, gitignored, no server, no registry.
7. Re-running `backtest --model m1 --score-test` for reproducibility/MLflow recomputes the same
   deterministic test numbers from tuning-only parameters; no decision used them.
8. `DecayedRidge.solve` solves only the columns seen so far (bit-identical forecasts when a new
   team appears later). This moved the GBL report in the 6th decimal after the verdict (margin
   scale 10.650608 → 10.650607; model_version 49583fa7 → 3f4861e2); gate and test numbers are
   unchanged at 4 dp.
9. Live M1 fits from the live Elo's first warm-up season (EL 2023-24, GBL 2018-19), the seasons
   the daily marts hold, not from 2007.
10. Daily workflow: a new `actions/cache` for `data/raw/euroleague`, then `ingest --details` and
    `ingest --competition gbl --details --pbp --seasons 2018 2019` before `build`. The first CI
    run fetches ≈ 3,200 EuroLeague API calls, ≈ 1,340 ESAKE pages and ≈ 680 BasketHotel calls
    (throttled; about 1.5 h); later runs fetch only new games.
11. The site shows one M1 row in the live scorecard block (M1 vs Elo log loss on the same games);
    screenshots use the fixture data because the real M1 log starts with the next daily run.
12. Added dependencies: `scipy` (runtime: Student-t CDF/optimiser), `scipy-stubs` and `mlflow`
    (dev).

## Open questions for the user
1. **First daily run with box scores:** it will take ≈ 1.5 h and ≈ 5,200 throttled requests to
   fill the CI caches (then only new games). OK to push as is, or run the first fill manually
   (`workflow_dispatch`) at a quiet time?
2. **M1 live despite non-significant gains:** both competitions pass the pre-registered rule,
   but no CI excludes zero and the GBL test goes the other way. Keep M1 live for 2026-27 (the
   default E-g), or treat it as shadow-only for the GBL?
3. **Grid edges:** several tuned values sit on the declared grid edge. Widen the grids for the
   next model (post-hoc, reported as such)?
4. **FT weight:** keep R1's 0.44, or switch to a EuroLeague-estimated weight (≈ 0.42) in a
   future model version (it would change `team_games` possessions)?
5. The web dev/preview servers hold `web/node_modules`; the checklist ran `npm ci` in a copy.
   Stop them before the next session if an in-place `npm ci` is wanted.
6. The §6 runner scripts live in the session scratchpad; should they be committed (e.g.
   `scripts/checklist.sh`) for the next weeks?

## The single consecutive §6 run (output, noise lines removed)
```
HEAD af1ca1c, tree clean: 0 changes, start 16:18Z
===== 1 uv sync --frozen =====
Checked 133 packages in 11ms
PASS
===== 2 ruff check =====
All checks passed!
PASS
===== 3 ruff format --check =====
89 files already formatted
PASS
===== 4 mypy src =====
Success: no issues found in 35 source files
PASS
===== 5 pytest + coverage =====
27 files skipped due to complete coverage.
Required test coverage of 85% reached. Total coverage: 99.13%
329 passed, 17 warnings in 53.04s
PASS
===== 6 vulture =====
PASS
===== 7 web build from fixture (clean copy of web/) =====
19:20:00 [build] 1 page(s) built in 1.28s
19:20:00 [build] Complete!
built 2 entries into site/
PASS
===== 8 Elo backtests reproduce every old value =====
5 passed in 0.06s
PASS
===== 9 predictions append-only, Elo model_version unchanged =====
removed/changed lines vs main: 0
reports/backtest_elo.json main=0.2.0+425e6393 now=0.2.0+425e6393
reports/backtest_elo_gbl.json main=0.2.0+df05260c now=0.2.0+df05260c
PASS
===== 10 stints sample byte-identical to main =====
50 games: all checks 100% (five_on_court 100%, seconds 100%, minutes 100%, points 100%); wrote reports\stint_validation.json
identical
PASS
===== 11 E1 team_games =====
9 games missing, 0 points mismatches; EuroLeague sample within 2: 89.0% of teams; wrote reports\possessions.json
rated games: 6387 with rows: 6378 missing (listed with reasons): 9
points = games scores for every row
EL sample: 50 games, within ±2: 89.0% of team-games, 80.0% of games; mean gap -0.342; FT weight matching PBP 0.4213 -> 94.0%
below 90%: explanation in docs/data/possessions.md: True
PASS
===== 12 =====
4246 games, 141625 stints; pass rate 2011-14 95.9%, 2015+ 98.9%; wrote reports\stints_mart.json
stints 534cd0cd811c7da77e1eefa45d534f2b09663d1a80b134260dae0c1a60675125
stint_game_checks 6ae4713d08bcce12a35385e7733cc2cb283532c2b201c3760dd06be6c8371810
two builds: stored tables identical
two builds: report byte-identical
2011-14 0.9587 (>= 0.95); 2015+ 0.9894 (>= 0.985)
passing games where stint points miss the final: []
per season: {'2011': 0.9734, '2012': 0.9526, '2013': 0.9605, '2014': 0.9522, '2015': 0.984, '2016': 0.9846, '2017': 0.9769, '2018': 0.9846, '2019': 0.9881, '2020': 0.9848, '2021': 0.9967, '2022': 0.9909, '2023': 0.997, '2024': 1.0, '2025': 0.99, '2026': 1.0}
PASS
===== 13 =====
28 passed in 1.59s
full EuroLeague M1 backtest (--score-test): 111 s (limit 600 s)
PASS
===== 14 =====
============================= 26 passed in 4.70s ==============================
PASS
===== 15 =====
declared f921181, verdict 37cf85f, first test scores committed fc7de59
order: declaration < verdict < test run
reports/backtest_m1.json: two runs byte-identical
reports/backtest_m1.json: equals the committed report
  gate student_t_const passed -0.001275 [-0.008356, 0.006426]
reports/backtest_m1_gbl.json: two runs byte-identical
reports/backtest_m1_gbl.json: equals the committed report
  gate student_t_pace passed -0.003487 [-0.035585, 0.024242]
PASS
===== 16 =====
mlruns/ is gitignored
euroleague: parent b284b7618e6d4f12834542de77a8cb36 (euroleague 0.2.0+m1.0cff42bc): 30 params, 823 metrics, data_sha256 62ddab91158f…, commit af1ca1c99d dirty=false
  child 4e1426b284f64d069b34888a11b64c81 normal_const: 35 params, 36 metrics, validation log loss 0.589192, hash 62ddab91158f…, commit af1ca1c99d
  child 256897f8155747f091afcdbc959c533f normal_pace: 35 params, 37 metrics, validation log loss 0.589293, hash 62ddab91158f…, commit af1ca1c99d
  child d3af94bf907a4ece8d4c67390c10a7c7 student_t_const: 35 params, 37 metrics, validation log loss 0.587234, hash 62ddab91158f…, commit af1ca1c99d
  child 1383bff19ea6473eb266f893312d2ab9 student_t_pace: 35 params, 38 metrics, validation log loss 0.587315, hash 62ddab91158f…, commit af1ca1c99d
gbl: parent 7034c944b8cd40d488be5427b9669207 (gbl 0.2.0+m1.3f4861e2): 30 params, 810 metrics, data_sha256 082d18568484…, commit af1ca1c99d dirty=false
  child 3f591627b1a5466f97bacfcb41b7ddfc normal_const: 35 params, 36 metrics, validation log loss 0.436988, hash 082d18568484…, commit af1ca1c99d
  child f2705bb1aaea463b969314cb4ab63e9d normal_pace: 35 params, 37 metrics, validation log loss 0.436287, hash 082d18568484…, commit af1ca1c99d
  child db55f66d1ade46c6b7b103e28a6638bc student_t_const: 35 params, 37 metrics, validation log loss 0.433360, hash 082d18568484…, commit af1ca1c99d
  child 3830d3a4447847f3a1c0d1c5918228ed student_t_pace: 35 params, 38 metrics, validation log loss 0.432759, hash 082d18568484…, commit af1ca1c99d
no-dev env: mlflow not installed
WARNING eurohoops.eval.tracking: MLflow is not installed (dev dependency): backtest not tracked
no-dev backtest: same report, tracking skipped with a warning
PASS
===== 17 =====
euroleague: clock 2026-09-25T07:00Z, first run 3 M1 rows, second run 0; log unchanged by run 2: True; Elo log byte-identical: True
    game_id,season,round,phase,tipoff_utc,home,away,p_home,exp_margin,exp_total,margin_sigma,margin_df,total_sigma,model,model_version,predicted_at_utc
    E2026_8,2026,1,RS,2026-09-25T17:00:00Z,BES,PAM,0.5066,0.18,171.46,10.4051,7,16.5296,m1,0.2.0+m1.0cff42bc,2026-09-25T07:00:00Z
    E2026_9,2026,1,RS,2026-09-25T17:45:00Z,ULK,VIR,0.7925,9.01,163.04,10.4051,7,16.5296,m1,0.2.0+m1.0cff42bc,2026-09-25T07:00:00Z
gbl: clock 2026-09-16T23:15Z, first run 1 M1 rows, second run 0; log unchanged by run 2: True; Elo log byte-identical: True
    game_id,season,round,phase,tipoff_utc,home,away,p_home,exp_margin,exp_total,margin_sigma,margin_df,total_sigma,model,model_version,predicted_at_utc
    GBL2026_3690EC22,2026,26,RS,2026-09-17T09:15:00Z,BB4B460F,2A25C696,0.5682,1.94,170.75,10.9069,7,16.8813,m1,0.2.0+m1.3f4861e2,2026-09-16T23:15:00Z
PASS
===== 18 build+score+publish twice leaves the tree unchanged =====
unchanged on the second run
PASS
===== 19 actionlint =====
clean
PASS
===== 20 screenshots =====
light 1440: horizontal overflow 0 px; M1 row visible: True; text: ['Team', 'model', '(M1)', 'log', 'loss', 'same', '60', 'games', 'as', 'Elo,', 'lower', 'is', 'better', '0.670', 'Elo', '0.665']
light 390: horizontal overflow 0 px; M1 row visible: True; text: ['Team', 'model', '(M1)', 'log', 'loss', 'same', '60', 'games', 'as', 'Elo,', 'lower', 'is', 'better', '0.670', 'Elo', '0.665']
dark 1440: horizontal overflow 0 px; M1 row visible: True; text: ['Team', 'model', '(M1)', 'log', 'loss', 'same', '60', 'games', 'as', 'Elo,', 'lower', 'is', 'better', '0.670', 'Elo', '0.665']
dark 390: horizontal overflow 0 px; M1 row visible: True; text: ['Team', 'model', '(M1)', 'log', 'loss', 'same', '60', 'games', 'as', 'Elo,', 'lower', 'is', 'better', '0.670', 'Elo', '0.665']
PASS
FAILS: 0
end 16:29Z; tree after: 
```
