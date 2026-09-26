# Agent Task: EuroHoops Analytics, Weeks 7–10: Shot Model M2 (xPTS)

## Role & goal
You are building the **weeks 7–10 deliverables** of `PLAN.md` (read §2 R9–R11, §5 "Evaluation protocol" and §5.3, §7, §8 row "7–10"). You build on the finished weeks 0–7 code (`reports/week3_closeout.md` §7a, `reports/week5-7_closeout.md`, `reports/m1-v2_progress.md`, `docs/models/m1.md`).

**Goal, in one sentence:** a calibrated shot-quality model (M2) that gives every EuroLeague field-goal attempt since 2011-12 an expected value (xPTS), plus the free throws the shot context generates. From it come team shot quality (offence and defence allowed) and a shrunk, stability-checked player shot-making measure. The model is judged on seasons it was never tuned on.

**Exit gate (PLAN §8, row 7–10).** All four must hold:
1. **Calibrated:** the chosen M2 meets F-f on validation (ECE ≤ 0.010 and every reliability bin with ≥ 500 shots within ±0.02).
2. **Beats the baseline:** its validation log loss (5-seed mean, F-l) is below the spline baseline's, with the game-level bootstrap CI (F-g) reported. **If it does not, that is an accepted outcome:** the spline model becomes M2, and the report shows every variant, where the challenger loses (distance bands, 2s vs 3s, context flags) and a hypothesis. Hiding or re-running until it wins is not accepted.
3. **Stability decided:** the F7 rule (F-k) has been applied and its verdict recorded.
4. **One clean run:** every check in §6 passes in one consecutive run.

## How you work: the loop
Repeat until the stop condition holds. Do not stop early, and do not call a partial run a success.

1. Read `reports/week7-10_progress.md` and `git log --oneline -15`. (Iteration 1: create the progress file with the recorded decisions from §0.) Pick the first unfinished deliverable (§4, in order F1 → F10).
2. Implement it, with tests.
3. Verify, in this order:
   - the deliverable's own **Done when** checks (§4), each as an automated test or a script under `scripts/checks/`, never by eye;
   - the fast gate every iteration: §6 items 1–6, 14 and 19 (`SKIP="7 8 9 10 11 12 13 15 16 17 18 20 21 22 23 24 25 26 27 28 29 30" bash scripts/checklist.sh`), plus any §6 item whose code or data your change touched;
   - the **whole** §6 checklist when a deliverable is finished, before you mark it done.
4. If anything is red, find the root cause, fix it and go back to step 3.
   - Never weaken, skip, `xfail` or delete a check to make it pass.
   - Never raise a tolerance unless you justify it in writing in the progress file.
5. If everything is green, make a small commit (attribution footer from your environment). Then append one line to `reports/week7-10_progress.md`: `iteration N | deliverable | checks run | result | commit`.
6. **Stop condition:** all deliverables are done **and** the full §6 checklist passes top to bottom in a single run with no edits in between. Paste that run's output into the final report (§7).

Guards:
- **Same failure three times** (same check, same error): stop patching. Write the diagnosis in the progress file, re-read the relevant code and data, then fix the cause.
- **External blocker** (a source down or refusing, quota exhausted): mark the check `BLOCKED` and save the evidence in the progress file, then carry on with the other deliverables. Never report a blocked check as passed.
- **Iteration budget: 35.** If iteration 35 ends without the stop condition, stop anyway. Write the final report with status `INCOMPLETE`, list each red check with its last error, and say what you would try next.
- **Runtime budget:** `eurohoops backtest --model m2` (LOSO CV + validation, reusing the stored Optuna result) must finish in < 40 min locally (5 seeds, F-l); the Optuna study (`--search`) runs only when F4 changes and must finish in < 2 h. Record both times in the progress file. Over budget is a red check.
- **Test seasons after a bug:** if you find a bug after the test seasons were scored, fix it, re-score, and record in the progress file what changed, both numbers, and why. The test result is then labelled "re-scored after a fix". Never pick between versions on test numbers.
- **Resumability:** the progress file and the commits are your memory. A usage-limit account switch or a restart can interrupt you at any time. After any interruption, start again at loop step 1.

## 0. Decisions (defaults in bold; confirm with the user before starting, then record the answers here)
| # | Question | Default |
|---|---|---|
| F-a | Seasons | **Shots from 2011-12 (the first season whose coordinates fit the FIBA line). Development 2011-12 → 2022-23 with leave-one-season-out (LOSO) CV (PLAN §5.3); validation 2023-24; test 2024-25 + 2025-26, scored once after the verdict. 2007-10 shots are kept in the table, flagged, and never used for fitting or scoring.** |
| F-b | Features | **Distance, angle, `ZONE`, 2 vs 3, `FASTBREAK`, `SECOND_CHANCE`, `POINTS_OFF_TURNOVER`, seconds left in the period, period (OT as one level), score margin from the shooter's side before the shot, home/away, and season as a numeric trend. No shooter, team or opponent identity (R9: quality ≠ skill).** |
| F-c | Model families (pre-declare ≤ 6 variants) | **Baseline = logistic regression with natural cubic splines on distance (knots on development quantiles) × shot type, plus the linear features. Challenger = LightGBM. Each with and without isotonic calibration fitted on out-of-fold predictions. The gate compares the best-on-CV challenger with the best-on-CV baseline.** |
| F-d | Hyperparameter search | **Spline baseline: a small declared grid (knots 4/6/8, L2 3 values). LightGBM: Optuna TPE, seed 20261001, 60 trials, declared search space, objective = mean LOSO CV log loss on development seasons only. Log the study to MLflow. The best parameters are written into the committed report JSON (single source of truth), never re-searched by the pipeline.** |
| F-e | Free throws ("total shot value") | **The shot feed has no shooting fouls on misses, so xPTS alone undervalues rim pressure. Model the free-throw points a team's shots generate at team-game level from PBP (FT trips per FGA, and FT points per trip), split by shot-distance band only where PBP ties a foul to the preceding shot. Report total shot value = xPTS + expected FT points. If PBP cannot tie fouls to shots, report team-level expected FT points and say so.** |
| F-f | Calibration target | **Validation ECE ≤ 0.010 (20 equal-count bins, shot level) for the chosen model, and the reliability curve inside ±0.02 in every bin with ≥ 500 shots. Also reported per distance band and for 2s vs 3s.** |
| F-g | Uncertainty unit | **Shots within a game are correlated: every CI (model differences, team shot quality, player shot-making) uses a game-level (cluster) bootstrap, 1,000 resamples, seed 20261001.** |
| F-h | Player shot-making | **Σ(actual − xPTS) per 100 shots, using out-of-fold xPTS only (never a model that saw the shot). Empirical-Bayes shrinkage (R11) toward 0 with the prior variance estimated on development seasons; 90% intervals. Shown only for ≥ 100 FGA in a season.** |
| F-i | Outputs and the site | **Marts tables `shots` and `shot_xpts`, reports JSON, a model card and static PNG shot charts in `docs/models/m2/`. No site change (PLAN §9: UI only after week 16 except predictions + performance). M2 feeds no live prediction.** |
| F-j | Dependencies | **`lightgbm`, `optuna` and `matplotlib` in the dev group (like MLflow): M2 is a local, monthly job, not part of the daily workflow. The daily workflow and CI keep working with `--no-dev`.** |
| F-k | Stability rule for showing shot-making (decided before F7 runs) | **Shot-making is "stable enough to show" only if its year-to-year correlation (players with ≥ 200 FGA in consecutive seasons, development seasons) has a 90% CI lower bound ≥ 0.20 and the split-half correlation is ≥ 0.30. Otherwise the card says "not stable enough" and the player table stays in the report only.** |
| F-l | Seed robustness (not repeated LOSO: its folds are fixed, a repeat gives identical numbers) | **The chosen LightGBM is refit with 5 seeds (20261001–20261005; row and feature subsampling on, as tuned). Report the mean and sd of LOSO CV and validation log loss across seeds. The gate uses the 5-seed mean prediction's log loss; the verdict also states whether any single seed would flip it. The Optuna study is not repeated per seed.** |

## 1. Hard constraints (all weeks 0–7 constraints still apply)
- **Scope = this file only.** No RAPM, no player impact beyond shot-making, no league translation, no GBL shot model (the GBL has no coordinates), no site pages, no live use of M2. Build nothing "for later".
- **The live logs are sacred.** Existing rows in `predictions/*.csv` and `odds/*.csv` stay byte-identical; `git diff origin/main -- predictions/ odds/` shows only added lines. No `model_version` of Elo or M1 changes.
- **Existing reports don't move.** Every M1, Elo, possessions and stints report keeps every value.
- **Validation discipline:** every choice on development seasons (LOSO CV) only; pre-declare the variants and the Optuna search space in the progress file and commit **before** the first validation run; report every declared variant on validation; anything added later is post-hoc and labelled so. **Test seasons are touched once**, after the verdict commit.
- **Out-of-fold only:** any number computed on a shot (residual, shot-making, team shot quality) uses a model that did not train on that shot's season.
- **No identity features** (F-b). A test proves the design matrix has no player, team or opponent column.
- **No dead code;** code ahead of its step goes in `vulture_whitelist.py` with a reason. `parse/shots.py` is already whitelisted: remove its entries once F1 uses it.
- **Politeness and data rights:** the shot cache (`data/raw/euroleague/points`) is complete for 2007-08 → 2025-26; do not re-fetch it. No raw JSON in git, only small trimmed fixtures.
- **Team codes:** source codes everywhere; display renames only in `publish.DISPLAY_CODES`.
- **Do not change any value in `PLAN.md`.** Report factual errors instead.
- **Git:** branch `week-7-10`, small commits. **Do not push.**

## 2. Verified facts (do not re-discover; do verify in tests)
- Raw shots: `data/raw/euroleague/points/E{season}/{game_code}.json.gz`, a dict with `Rows`, every game of 2007-08 → 2025-26 cached (2026-27: games so far). Row fields: `NUM_ANOT, TEAM, ID_PLAYER, PLAYER, ID_ACTION, ACTION, POINTS, COORD_X, COORD_Y, ZONE, FASTBREAK, SECOND_CHANCE, POINTS_OFF_TURNOVER, MINUTE, CONSOLE, POINTS_A, POINTS_B, UTC`.
- **Codes are space-padded:** `TEAM` ("PAN       "), `ID_PLAYER`, and a few `ID_ACTION` values ("2FGA      "). Strip them.
- 2024-25 action counts: 2FGM 13,556, 2FGA 10,832, 3FGM 6,203, 3FGA 10,942, FTM 9,660; **no missed free throws** in this feed (only `FTM`). Free throws carry coordinates (−1, −1).
- Coordinates (`parse/shots.py`): centimetres from the basket, mirrored onto one basket; `to_court_coords` → metres, `beyond_three_line` tests the FIBA line. 2011-12 onward: 99.63–99.94% of 3s beyond the line and 99.65–99.96% of 2s inside (±0.15 m). 2007-10 fits neither 6.75 m nor 6.25 m. About 0.1% of shots sit at (0, 0) (missing coordinates). The side meaning of x > 0 is unverified and irrelevant to distance and |angle|.
- About 41,500 FGA per season recently; roughly 500k FGA in 2011-12 → 2025-26.
- The marts hold `games`, `team_games` (FGA, FTA, OREB, TOV, possessions per team-game; FT weight 0.42), `stints` (2011+), `team_seasons`. Reuse `eval/metrics.py` (log loss, Brier, ECE + reliability, `paired_bootstrap_ci`), `eval/tracking.py` (MLflow, data hash) and the `reports/*.json` + number-checked model card pattern of `docs/models/m1.md`.
- `scripts/checklist.sh` runs the 20 existing checks; extend it (§6), do not fork it.
- The Odds API historical endpoint is paid-only (probe 2026-09-26: `HISTORICAL_UNAVAILABLE_ON_FREE_USAGE_PLAN`), irrelevant to M2.
- Bash heredocs holding Python source have corrupted `\n` escapes before. Write code with the Write/Edit tools.

## 3. Unverified (verify, then record in `docs/data/shots.md` and a test)
- Whether `POINTS_A`/`POINTS_B` are the score **after** the action, and which one is the home team, in every season (needed for the pre-shot margin).
- `MINUTE`/`CONSOLE` semantics in overtime and across seasons (seconds left in the period).
- The action-code vocabulary per season (do pre-2015 layups/dunks appear as `LAYUPMD`/`DUNK` here as they do in PBP? are blocked shots `2FGAB`-style codes?).
- Whether per team-game FGA and made-FG points in the shot feed equal `team_games` FGA and the box points from field goals.
- Whether PBP ties a shooting foul (and its free-throw trip) to the shot that drew it, and to a location (F-e).
- The share of shots at (0, 0) per season, and of 2/3 labels contradicting the geometry beyond ±0.15 m.

## 4. Deliverables

### F1: Shot table (`parse/shot_table.py`, marts `shots`)
- From the cache only: one row per FGA with competition, season, game_id, period, seconds left in the period, team, opponent, home flag, shooter id (for F7 only), made, value (2/3), x/y metres, distance, |angle|, `ZONE`, the three context flags, pre-shot score margin from the shooter's side, `validated_season` flag. Pandera contract, trimmed fixture.
- Exclusions kept in a `shots_excluded` table with the reason: (0, 0) coordinates, label contradicting the geometry beyond ±0.15 m, unparseable rows.
- **Done when:**
  - per team-game FGA and made-FG points equal `team_games`/box for ≥ 99% of 2011+ team-games. Below that, each mismatch class has a count, a cause in `docs/data/shots.md` and a test on a real example, and the unexplained remainder is ≤ 0.5% of team-games;
  - excluded shots are ≤ 1% of FGA in every validated season (report the share per season and reason);
  - every §3 item has an answer in `docs/data/shots.md` and a test;
  - two builds give identical tables.

### F2: Free-throw generation (F-e)
- From PBP: FT trips and FT points per team-game, tied to the preceding shot's distance band where the data allow.
- **Done when:** with rates fitted leave-one-season-out, expected FT points reconcile with actual FT points in **every** development season (mean gap within ±0.1 point per team-game) and on validation (within ±0.2); the method, and whether fouls could be tied to shots, are in `docs/data/shots.md` with a test on a real game.

### F3: Spline baseline (`models/xpts.py`)
- Logistic regression with natural cubic splines on distance × shot type plus the linear features (F-b, F-c), fitted with numpy/scipy (no new runtime dependency).
- **Done when:** on synthetic shots from a known logistic curve the fit recovers the coefficients (max abs error < 0.02 at n = 200k); a test proves no identity columns; mypy strict clean.

### F4: LightGBM challenger, calibration and the Optuna search
- LightGBM with the declared Optuna study (F-d); isotonic calibration on out-of-fold predictions (never on the fold being scored).
- Determinism: Optuna `TPESampler(seed=20261001)`, `n_jobs=1`; LightGBM `deterministic=True`, `force_row_wise=True`, fixed `num_threads` and `seed`. The study runs only with `backtest --model m2 --search`; otherwise the stored best parameters are used.
- Seed check (F-l): refit the chosen configuration with the 5 declared seeds; store per-seed and mean metrics in the report.
- **Done when:** a 5-trial study on a 2-season subset run twice gives identical trial values and best parameters (a test); the full study's trials are in MLflow and its best parameters in the report JSON.

### F5: Evaluation, gate and test
- `eurohoops backtest --model m2` writes `reports/backtest_m2.json`: per variant, LOSO CV and validation log loss, Brier, ECE + reliability (overall, per distance band, 2s vs 3s); the gate comparison with the game-level bootstrap CI (F-g).
- The verdict applies exit-gate items 1 and 2 literally (F-f thresholds, log loss vs the spline baseline). On a fail, add the analysis the exit gate asks for.
- Commit the verdict in the progress file, **then** score the test seasons once.
- **Done when:** the report has every field above plus a machine-readable `gate` block (`calibrated`, `beats_baseline`, `passed`), the declaration and verdict commits precede the test run in `git log` (a script checks it), and two runs give identical JSON.

### F6: Team shot quality and total shot value
- Per team-season (and per team-game in the marts): xPTS per shot for offence and allowed on defence, expected FT points (F2), total shot value, actual minus expected, with game-level bootstrap intervals. Out-of-fold xPTS only.
- **Done when:** league-season sums of xPTS equal actual FG points within 0.5% on every development season (calibration in the large), and the out-of-fold rule has a leakage test.

### F7: Player shot-making and stability (R10, R11)
- Shot-making per 100 shots with EB shrinkage and 90% intervals (F-h).
- Stability: year-to-year correlation (players with ≥ 200 FGA in consecutive seasons) of shrunk shot-making vs raw eFG% and vs raw FG% on the same players; plus within-season split-half (odd/even games). Report discrimination (spread across players) too.
- **Done when:** the numbers are in `reports/m2_players.json` with CIs, and a `stability.verdict` field equals what the F-k rule gives on those numbers (a test recomputes it).

### F8: Shot charts
- Static PNGs in `docs/models/m2/`: league xPTS surface (validation season), and actual-minus-expected hex charts for three teams and three players (named in the progress file before drawing). Court drawn to FIBA dimensions from `parse/shots.py` constants.
- **Done when:** a test checks the court geometry used for drawing equals the `shots.py` constants; you open every PNG (Read tool) and log one line per image in the progress file (court lines in place, legend and labels readable, colour scale centred at 0 for actual − expected); the images are referenced from the model card.

### F9: MLflow and leakage
- Experiment `m2-backtest`: parent run per backtest, one child per variant and the Optuna study; params, data hash, commit, metrics, report artifact (reuse `eval/tracking.py`).
- Leakage tests: (1) changing any shot of season s changes no out-of-fold xPTS of season s; (2) editing validation or test shots changes no development-fitted value; (3) a guard proves the edits do change what they should.
- **Done when:** both are green and each leakage test fails once when the fold rule is broken on purpose (shown in the progress file, then reverted).

### F10: Model card `docs/models/m2.md`
- Data and exclusions, features and why no identity, the variants and the search, calibration, results with CIs, free-throw method, team shot quality, player shot-making and stability, limitations.
- **Done when:** every number matches the reports (extend `tests/test_model_card.py` or add a sibling test).

## 5. Out of scope (do not build)
RAPM or any other player impact, league translation, GBL shot data, M1 changes, live use of M2, site pages or site changes, FastAPI, Docker, the MLflow registry, betting language.

## 6. Verification checklist (run top to bottom; all must pass in one run)
Items 1–20: `bash scripts/checklist.sh` as it is (lint, types, tests + coverage, vulture, web build, Elo/M1/stints/possessions regressions, live dry run, build/score/publish idempotence, actionlint, screenshots). Item 9's base is `origin/main`.

Add, in `scripts/checklist.sh` and `scripts/checks/`:

21. F1: shot reconciliation rate and exclusion shares; two builds identical.
22. F2: FT reconciliation within ±0.1 point per team-game.
23. F3/F4 unit tests; the 5-trial Optuna reproducibility test; recorded backtest and study runtimes within budget.
24. F5: `reports/backtest_m2.json` complete with its `gate` block; declaration < verdict < test in `git log`; two runs identical.
25. F6: calibration in the large within 0.5% per development season.
26. F7: `reports/m2_players.json` present with CIs; the stability verdict matches the F-k rule.
27. F8: the charts exist and the geometry test passes.
28. F9: MLflow parent + children with params, hash and commit; leakage tests pass.
29. F10: model card number check passes.
30. `uv sync --frozen --no-dev` then `eurohoops build && eurohoops predict` still work without the dev group.

## 7. Final report (`reports/week7-10_closeout.md`)
Include:
- the checklist with PASS, BLOCKED or SKIPPED per item, and the output of the single consecutive §6 run
- the gate verdict (chosen variant vs spline baseline on validation: log loss, Brier, ECE with game-level CIs), then the test numbers
- every variant and the Optuna study summary; which items were post-hoc
- the F-e free-throw method and its limits
- the stability finding, stated plainly
- MLflow run ids
- every decision this file did not cover
- open questions for the user

Then append a session entry to `memory.md`.
