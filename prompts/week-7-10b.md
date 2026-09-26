# Agent Task: EuroHoops Analytics, Weeks 7–10b: M2 follow-up (flags, FT check, runtime, season level)

## Role & goal
You continue the weeks 7–10 shot model M2 on branch `week-7-10`. Read first, in this order:
`reports/week7-10_closeout.md` (status INCOMPLETE, 4 red checks, gate failed on calibration),
`reports/week7-10_progress.md`, `prompts/week-7-10.md` (the original task: its rules still
apply unless this file changes them), `docs/models/m2.md`, `docs/data/shots.md`, `PLAN.md` §5.3.

**Goal, in one sentence:** make the four user decisions below real: the outcome-coded flags gone
for good, a free-throw check that tests what the model can know, an M2 backtest that is faster
and within a 60-minute budget, and a declared per-season-level variant that targets M2's
calibration failure. Then run the full checklist until it is green or every remaining red item
is justified in writing.

**User decisions (2026-09-26, final; do not re-ask):**
1. **Drop** `FASTBREAK`, `SECOND_CHANCE`, `POINTS_OFF_TURNOVER` as M2 features permanently
   (they are set only on made shots from 2015-16). Rebuilding them from play-by-play is **out of
   scope** here.
2. **Re-specify the F2 free-throw check:** test the within-season *shares* (by distance band and
   by team), which season-level drift does not affect. Report the season-level gap as a
   documented limitation, not a gate.
3. **Relax the M2 runtime budget to 60 minutes** for `eurohoops backtest --model m2
   --score-test` (5 seeds, nested isotonic kept), **and** analyse the code to cut time, cost and
   computational complexity.
4. **Add a declared per-season-level variant** (the season's league shooting level estimated
   only from games before the shot) for calibration and calibration in the large.

## How you work: the loop
Repeat until the stop condition holds. Do not stop early; do not call a partial run a success.

1. Read `reports/week7-10b_progress.md` and `git log --oneline -15` (iteration 1: create the
   progress file with §0's answers). Pick the first unfinished deliverable (§4, G1 → G6).
2. Implement it, with tests.
3. Verify, in this order:
   - the deliverable's **Done when** checks, each an automated test or a script under
     `scripts/checks/`, never by eye (charts: open each PNG with the Read tool and log it);
   - the fast gate every iteration: `SKIP="7 8 9 10 11 12 13 15 16 17 18 20 21 22 23 24 25 26
     27 28 29 30" bash scripts/checklist.sh`, plus every §6 item your change touched;
   - the **whole** §6 checklist when a deliverable is finished, before you mark it done.
4. If anything is red: find the root cause, fix it, go back to 3. Never weaken, skip, `xfail`
   or delete a check; never raise a tolerance without a written justification in the progress
   file that does **not** use the observed failing numbers as the reason.
5. Green → small commit (attribution footer from your environment) → append
   `iteration N | deliverable | checks run | result | commit` to the progress file.
6. **Stop condition:** G1–G6 done **and** the full §6 checklist passes top to bottom in one run
   with no edits in between, or every red item is one this file explicitly allows to stay red
   (only §4 G5's gate outcome may) with its evidence. Paste that run's output into the report.

Guards:
- **Same failure three times** (same check, same error): stop patching; write the diagnosis in
  the progress file; re-read code and data; fix the cause.
- **Iteration budget: 25.** At the end of iteration 25 stop anyway; write the report with status
  `INCOMPLETE`, each red check with its last error, and what you would try next.
- **Quiet machine for timed checks.** Before any full checklist run or timed backtest, check CPU
  load (`(Get-CimInstance Win32_Processor).LoadPercentage` in PowerShell) and list `node`/`python`
  processes you did not start. If other work is running, say so in the progress file and wait or
  ask the user; never kill processes you did not start. (Last time two `astro dev` servers made
  numpy/BLAS threads spin and the M1 backtest took 3,298 s instead of 224 s.)
- **DuckDB is single-writer.** Never run two commands that write `data/marts/eurohoops.duckdb`
  at once (including a background run plus a foreground one).
- **Resumability:** progress file + commits are your memory; after any interruption restart at
  step 1.
- **Heredocs:** Bash heredocs holding Python or Markdown have broken before in this environment.
  Write files with the Write/Edit tools.

## 0. Sub-decisions (defaults in bold; confirm with the user once, before starting, and record)
| # | Question | Default |
|---|---|---|
| G-a | Scope of the flag removal (G1) | **Code already dropped them (afbc08e). G1 makes it permanent: the amended F-b recorded in `prompts/week-7-10.md` §0 as "amended 2026-09-26 by the user", a data test that fails if any M2 feature has a make rate ≥ 0.99 when set (so no future outcome-coded feature slips in), docs updated.** |
| G-b | F2 share statistics (G2) | **Per held-out season (development LOSO, validation fitted on all development): (i) band shares: the share of and-one FT points per distance band, expected vs actual; (ii) team shares: each team's share of the league's FT points, expected vs actual, summarised by the mean absolute share gap and the Pearson r across team-seasons. Tolerance for each statistic = 2 × its game-level bootstrap standard error (1,000 resamples, seed 20261001), computed on the same season, fixed in code before the check runs. The season-level mean gap stays in the report as a limitation.** |
| G-c | Runtime budget (G3, G4) | **`backtest --model m2 --score-test` < 60 min on a quiet machine (5 seeds, nested isotonic kept, all declared variants incl. G5's). The Optuna study keeps its 2 h budget and is re-run only if F4's features change.** |
| G-d | Optimisation rule (G3) | **Two classes. (A) Number-preserving: allowed freely; the committed `reports/backtest_m2.json`, `m2_teams.json`, `m2_players.json` must stay byte-identical (proved by checklist items 24–25). (B) Number-changing (threads, trees, seeds, calibration design, Dataset reuse that changes binning): only as a written, declared change committed before its run, with old and new numbers side by side; never chosen by validation or test numbers.** |
| G-e | Season-level variant (G5) | **`lgbm_level` (and `spline_level`): the chosen base model's logit plus a season offset. For a shot in game g of season s, the offset is estimated only from shots of season s in games that tipped off before g, shrunk toward the previous season's final offset (empirical-Bayes weight from development seasons: offset = w·(season-to-date log-odds residual) + (1 − w)·(previous season's offset), w = n/(n + k), k fitted on development seasons only). The first game of a season uses the previous season's offset. Residuals come from out-of-fold base predictions, so the offset never uses the shot's own game or later games.** |
| G-f | Evaluation of the level variant (G5) | **Declared before its first run. Validation 2023-24 and test 2024-25/2025-26 have already been seen, so every G5 number on them is labelled "post-hoc, not a clean hold-out". The clean evaluation is the live 2026-27 season: G5 writes a scoring script and a pre-registered rule (F-f on 2026-27 shots, run after the season), but runs nothing on 2026-27 now. The weeks 7–10 verdict (84308f9) is not changed or re-labelled.** |
| G-g | Which M2 feeds F6/F7 afterwards | **The weeks 7–10 chosen M2 (`lgbm`) stays the reported M2. If `lgbm_level` meets F-f on validation and beats `lgbm` on development LOSO log loss, F6/F7 are additionally computed with it and reported side by side, labelled post-hoc. Nothing replaces the committed weeks 7–10 numbers.** |

## 1. Hard constraints (all weeks 0–10 constraints still apply)
- **Scope = this file.** No PBP-derived context features, no new data, no site changes, no live
  use of M2, no RAPM.
- **Live logs are sacred:** `predictions/*.csv` and `odds/*.csv` byte-identical; `git diff
  origin/main -- predictions/ odds/` shows only added lines (none expected here).
- **Existing reports don't move**, except where a deliverable here says so: M1, Elo,
  possessions, stints, `shots.json`, `backtest_m2.json` (numbers), `m2_teams.json`,
  `m2_players.json` keep every value. New numbers go in new fields or new files.
- **Out-of-fold only** for every number computed on a shot, now including the season offset
  (§0 G-e): a leakage test proves that editing any shot in game g or later changes no offset used
  for game g.
- **No dead code;** code ahead of its step in `vulture_whitelist.py` with a reason.
- **Do not change any value in `PLAN.md`.** Report factual errors instead.
- **Git:** stay on `week-7-10`, small commits. **Do not push.** Do not merge.

## 2. Verified facts (do not re-discover; do verify in tests)
- Timings on a quiet machine (12 logical cores): LightGBM fit, 900 trees, ~330k shots, 6 threads
  ≈ 5–10 s; prediction 0.45 s; spline fit (8 knots) 2.2 s; isotonic fit (350k) 0.4 s; ECE-diff
  bootstrap (1,000) 13.6 s per split. `features()` was 0.53 s per call (zone loop), now 0.02 s
  (349acb9); that saved only ~1 min, so LightGBM fitting dominates.
- Backtest structure per LightGBM seed: 12 LOSO fits + 66 leave-two-out fits (nested isotonic) +
  1 development fit = 79 fits; × 5 seeds = 395 fits. Spline: 9 grid configs × 12 LOSO = 108 fits,
  then 67 fits for the chosen config. The nested isotonic is O(S²) fits in the number of
  development seasons S = 12.
- Measured runtimes: study 3,508 s; backtest 2,951 s (validation) and 3,072–3,130 s (with test).
  M1 backtest 224–243 s.
- F2 LOSO mean gaps per development season: +0.97 (2011), −0.13, −0.95, −0.12, +0.20, −0.23,
  +0.89, −0.01, −0.10, −0.01, −1.28 (2021), +0.95 (2022); validation −0.60. Between-season sd of
  FT points per team-game 0.66.
- Calibration in the large (xPTS / actual FG points), development: 0.977–1.020, only 5 of 12
  seasons within 0.5%. `lgbm` validation ECE 0.0104, test 0.0126; failing bins at P ≈ 0.40 and
  0.69; per-band ECE up to 0.0495 (deep threes).
- `shot_xpts` mart holds the chosen M2's out-of-fold P(make) for development, validation and
  test. `ft_team_games` holds FT trips and points per team-game.
- Checklist item 24 currently stops after its first run when that run is over budget, so
  reproducibility is never re-checked in that case (see G4).

## 3. Unverified (verify, then record in the progress file and a test)
- Whether LightGBM with `deterministic=True` and fixed `num_threads` gives byte-identical
  predictions when several fits run **in parallel processes** (each with the same `num_threads`).
- Whether computing `features()` once per season and slicing (instead of per fit) changes any
  prediction (it should not).
- How much of the backtest's wall time each phase takes (spline grid, spline pairs, each seed's
  LOSO, pairs and development fit, calibration, metrics, bootstrap, MLflow): profile it.

## 4. Deliverables

### G1: The flag removal made permanent (decision 1)
- Record the amended F-b in `prompts/week-7-10.md` §0 (a line under the table, dated, "amended
  by the user"); update `docs/models/m2.md` and `docs/data/shots.md` wording if needed.
- A data test over `reports/shots.json` (or the shot table) and both feature builders: no M2
  feature, flag or one-hot level may have a make rate ≥ 0.99 among the development shots where
  it is set (catches any future outcome-coded feature).
- **Done when:** the test passes, fails when one of the three flags is put back in either feature
  builder (shown once in the progress file, then reverted), and the docs say it plainly.

### G2: The free-throw check re-specified (decision 2)
- Implement §0 G-b in `parse/free_throws.py` + `reports/free_throws.json` (new fields; the old
  per-season mean gaps stay, labelled "season level, not gated").
- Replace checklist item 22's criterion with the share checks; the old ±0.1 level check stays in
  the report and the model card as a limitation.
- **Done when:** the share checks pass on every development season and on validation, each
  tolerance is 2 × its bootstrap SE computed by code committed before the first run, a unit
  test shows each check fails on synthetic data with a planted share error of 3 SE, and the
  model card's F2 section and number table are updated (card test green).

### G3: Runtime analysis and optimisation (decision 3)
- Profile a full `backtest --model m2 --score-test` (cProfile or timed sections) and write
  `docs/models/m2_runtime.md`: a table of phases with wall time, share of total, number of
  fits, and big-O in S (seasons), n (shots), T (trees), K (seeds); what dominates and why.
- Implement every class-A (number-preserving) optimisation you find. Candidates to evaluate
  (measure, don't assume): feature matrices built once per season and sliced; the development
  fit reused where two code paths refit the same data; seeds or leave-two-out pairs fitted in
  parallel **processes** with the same `num_threads` each (only if §3 shows byte-identical
  predictions); vectorised ECE bootstrap; spline design built once per fold instead of per
  predict; avoiding repeated `pd.concat`/copies in `Folds`.
- For class-B ideas (fewer threads or trees, fewer seeds for the pairs, a cheaper calibration
  design such as cross-fitting within the training seasons), write the estimated saving and the
  statistical cost in the runtime doc; implement none unless declared per §0 G-d.
- Relax the budget in `scripts/checks/m2_reports.sh`, `scripts/checks/m2_runtimes.py` and
  `m2_models.sh` to 3,600 s for the backtest (study stays 7,200 s), with a comment citing the
  user's decision.
- **Done when:** `docs/models/m2_runtime.md` exists with before/after times for each
  optimisation; `reports/backtest_m2.json`, `m2_teams.json`, `m2_players.json` are
  byte-identical to the committed ones after every class-A change (checklist items 24–25); the
  quiet-machine backtest with `--score-test` is < 3,600 s (recorded as a `RUNTIME backtest
  <s> s` line in the progress file); and the M1 backtest (items 13, 15) is unaffected.

### G4: Checklist item 24 always checks reproducibility
- Change `scripts/checks/m2_reports.sh` so both runs always happen and the item fails if
  **either** the runtime is over budget **or** the two runs differ **or** the report differs from
  the committed one; print all three results. This makes the check stricter, not weaker.
- **Done when:** a dry run shows all three lines, and a deliberately slowed first run (e.g. a
  temporary budget of 1 s, shown once, then reverted) still reports the reproducibility result.

### G5: Declared per-season-level variant (decision 4)
- Declare `lgbm_level` and `spline_level` per §0 G-e and G-f in the progress file and commit
  **before** their first run on any split. Implement in `models/` + `eval/m2_backtest.py` as new
  report fields (e.g. `level_variants`), leaving the committed variants' numbers untouched.
- Leakage tests: editing any shot of game g or later changes no offset used for game g; editing
  another season's shots changes the offset only through the declared prior; guards prove the
  edits reach what they should; each test fails once when the rule is broken on purpose (shown,
  then reverted).
- Report on development (LOSO), validation and test, every number labelled post-hoc: log loss,
  Brier, ECE + reliability (overall, per band, 2s vs 3s), calibration in the large per season,
  and the difference vs `lgbm` with the game-level bootstrap CI.
- Write `scripts/checks/m2_live_level.py` (not in the checklist): the pre-registered scoring of
  both `lgbm` and `lgbm_level` on 2026-27 shots, with F-f as the rule, to run after the season.
- **Done when:** the declaration commit precedes the first run (a script checks it), the
  leakage tests are green, the report has every field above, two runs are byte-identical, and
  the model card has a "Season-level variant (post-hoc)" section whose numbers the card test
  checks. **Either outcome is accepted:** if `lgbm_level` still fails F-f, say so and give a
  hypothesis; do not iterate variants to make it pass.

### G6: Model card, closeout and memory
- Update `docs/models/m2.md` (F2 shares, runtime summary, the level variant), keep every weeks
  7–10 number, and extend `tests/test_model_card_m2.py` if new tables are added.
- Write `reports/week7-10b_closeout.md` (§7) and append a session entry to `memory.md`.

## 5. Out of scope (do not build)
PBP-derived context features, new data sources, RAPM, site or UI changes, live use of M2,
changing the weeks 7–10 verdict, pushing or merging.

## 6. Verification checklist (run top to bottom; all must pass in one run)
Items 1–30 of `scripts/checklist.sh` with these changes:
- **22:** the G2 share checks (band and team shares within 2 × bootstrap SE, every development
  season and validation).
- **23 / 24:** backtest budget 3,600 s (study 7,200 s); item 24 per G4.
- **25:** calibration in the large for the weeks 7–10 chosen M2 stays as it is (it may stay
  red: the committed M2 is not changed); add item **31**: the G5 report fields present, labelled
  post-hoc, two runs identical, leakage tests green, declaration < first run in `git log`.
- Add item **32:** the G1 outcome-coding data test.

Item 25 and the G5 gate outcome are the only items allowed to stay red, and only with the
evidence written in the report.

## 7. Final report (`reports/week7-10b_closeout.md`)
- The checklist with PASS/FAIL/BLOCKED/SKIPPED per item and the full output of the single
  consecutive run.
- G2: the share statistics per season with their tolerances; the season-level gaps as a
  limitation.
- G3: the runtime table (before/after, per phase, big-O), which optimisations were class A and
  B, and the final runtime.
- G5: the level variant's results (post-hoc), whether it meets F-f, calibration in the large per
  season, and the pre-registered 2026-27 rule.
- Every decision this file did not cover; open questions for the user; then the `memory.md`
  entry.
