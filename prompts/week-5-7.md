# Agent Task: EuroHoops Analytics, Weeks 5–7: Team Model M1

## Role & goal
You are building the **weeks 5–7 deliverables** of `PLAN.md` (read §2 R1–R4, §4.4, §5 "Evaluation protocol" and §5.2, §7, §8 row "5–7"). You build on the finished weeks 0–5 code (`prompts/week-3-5.md`, `reports/week3-5_closeout.md`, `docs/spikes/stints.md`, `docs/spikes/gbl-pbp.md`).

**Goal, in one sentence:** a possession-based team model (M1) that rates every team's offence, defence and pace, forecasts a full margin and total distribution for every game, and is compared with Elo on validation seasons it was never tuned on. The comparison is walk-forward, leak-free and tracked in MLflow. M1 goes live only if it wins.

**Exit gate (PLAN §8, row 5–7):** M1 beats Elo on validation log loss per competition, **or** the report says why not, with evidence. Every check in §6 must also pass in **one consecutive run**.

## How you work: the loop
Repeat until the stop condition holds. Do not stop early, and do not call a partial run a success.

1. Read `reports/week5-7_progress.md` and `git log --oneline -15`. Pick the first unfinished deliverable (§4, in order E1 → E8).
2. Implement it, with tests.
3. Run that deliverable's own checks (listed under it as **Done when**), then the **whole** §6 checklist.
4. If anything is red, find the root cause, fix it and go back to step 3.
   - Never weaken, skip, `xfail` or delete a check to make it pass.
   - Never raise a tolerance unless you justify it in writing in the progress file.
5. If everything is green, make a small commit (attribution footer from your environment). Then append one line to `reports/week5-7_progress.md`: `iteration N | deliverable | checks run | result | commit`.
6. **Stop condition:** all deliverables are done **and** the full §6 checklist passes top to bottom in a single run with no edits in between. Paste that run's output into the final report (§7).

Guards:
- **Same failure three times** (same check, same error): stop patching. Write the diagnosis in the progress file, re-read the relevant code and data, then fix the cause.
- **External blocker** (a source down or refusing, quota exhausted): mark the check `BLOCKED` and save the evidence in the progress file, then carry on with the other deliverables. Never report a blocked check as passed.
- **Iteration budget: 30.** If iteration 30 ends without the stop condition, stop anyway. Write the final report with status `INCOMPLETE`, list each red check with its last error, and say what you would try next.
- **Resumability:** the progress file and the commits are your memory. A usage-limit account switch (`cswap auto`, threshold 90%) or a restart can interrupt you at any time. After any interruption, start again at loop step 1.

## 0. Decisions (defaults in bold; confirm with the user before starting, then record the answers here)
| # | Question | Default |
|---|---|---|
| E-a | Seasons per competition (tuning / validation / test) | **EL: warm-up 2007–2014, tuning 2015–2022, validation 2023, test 2024+2025. GBL: warm-up 2018–2020, tuning 2021–2022, validation 2023, test 2024+2025.** (2023 = 2023-24.) Training always uses everything before the prediction point. |
| E-b | Elo in the gate comparison | **A comparison-only Elo, re-tuned on the M1 tuning seasons with the existing grid, so neither model has seen validation. The live Elo, its reports and its `model_version` stay untouched.** |
| E-c | Fitting level | **Game-level: one row per team-game, points per 100 possessions. Stints are built and validated (E2) but feed no model before M3.** |
| E-d | GBL possessions | **From the cached ESAKE box HTML, which already has `O.REBS`, `D.REBS` and `TO` columns. No new ESAKE fetches. The 2018-20 games filled from PBP get possessions from their PBP, or are excluded with a stated reason.** |
| E-e | Predictive distribution | **Normal and Student-t margins (df tuned on tuning seasons), with σ either constant or scaling with expected pace. Pick on tuning log loss; report all variants on validation. Totals: Normal, σ fitted on tuning.** |
| E-f | MLflow | **Tracking only, local file store `mlruns/` (gitignored), MLflow in the dev dependency group, imported only by the backtest path. No registry, no server. The live predictor reads frozen params from the committed report JSON, as Elo does.** |
| E-g | M1 going live | **Only for a competition where M1 passes the gate. Separate append-only logs `predictions/{euroleague,gbl}_m1_2026-27.csv` (Elo logs untouched), an M1 section in the scorecards, and one M1 row in the site scorecard. No new site pages.** |

**Answered by the user, 2026-09-25: all defaults (E-a to E-g).**

## 1. Hard constraints (all weeks 0–5 constraints still apply)
- **Scope = this file only.** No xPTS, no RAPM, no player ratings, no entity resolution, no FastAPI, no Docker, no state-space model (M1 v2), no blending with Elo. Build nothing "for later".
- **The live logs are sacred.** Existing rows in `predictions/*_2026-27.csv` stay byte-identical. The Elo `model_version` of both competitions must not change. `git diff main -- predictions/` shows only added lines or new files.
- **Existing reports don't move.** `reports/backtest_elo*.json` and `reports/live_scorecard*.json` keep every existing value. New fields are additions only.
- **Validation discipline:**
  - Choose every hyperparameter on the tuning seasons only.
  - Before the first validation run, pre-declare **at most 6 candidate variants** in the progress file and commit it.
  - Report every declared variant on validation, not just the winner. The gate uses the variant with the best **tuning** log loss.
  - Any variant added after the first validation run must be reported as post-hoc.
- **Test seasons are touched once.** Score the test seasons only after the gate verdict is committed. The progress file shows the verdict's commit before the test run.
- **Leak-free:** to predict round *r*, fit only on games whose tip-off is before round *r*'s first tip-off (PLAN §5).
- **No dead code** (weeks 0–1 definition). Code ahead of its pipeline step goes in `vulture_whitelist.py` with a one-line reason.
- **Politeness:** ESAKE ≥ 2 s between requests, gzip raw cache, never re-fetch what is cached. The EuroLeague API is unofficial: cache everything.
- **Team codes:** source codes everywhere (CLAUDE.md). Display renames only in `publish.DISPLAY_CODES`.
- **Data rights:** no raw HTML/JSON/xlsx in git, only small trimmed fixtures.
- **Website:** any change follows `DESIGN.md`. The page stays a model benchmark, not betting advice.
- **Do not change any value in `PLAN.md`.** Report factual errors instead.
- **Git:** branch `week-5-7`, small commits. **Do not push.**

## 2. Verified facts (do not re-discover; do verify in tests)
- The marts (`data/marts/eurohoops.duckdb`) hold `games` (both competitions), `teams`, and GBL-only `player_box`/`team_box`/`box_fill`. The GBL `player_box` has points, FG2/FG3/FT made/attempted and seconds only. **There is no EuroLeague box table in the marts yet.**
- The raw cache (`data/raw/`) holds EuroLeague box, PBP and shots for 2007–2026 and GBL ESAKE box HTML for 2018–2025. The ESAKE box table columns are `P, 2PM-A, 3PM-A, FTM-A, REBS, D.REBS, O.REBS, AST, BLK, BLK-A, FOULS F, FOULS M, STL, TO, TIM.PL., RANK`. It also has a team row (`ΟΜΑΔΙΚΑ`) and a totals row (`ΣΥΝΟΛΟ`). `parse/esake.parse_box_score` reads only some of these.
- `parse/stints.py` rebuilds EuroLeague lineups, and `eurohoops stints` validates a 50-game sample (seed 20260925): 50/50 pass. Across the full population, 99.0% of 2015+ games pass, 95.9% of 2011–14, and 3.0% of 2007–10 (whole-minute substitution clocks). `docs/spikes/stints.md` records the quirks: phantom overtime, `LAYUPMD`/`DUNK` before 2015, 4 or 6 players on court, empty game E2018_21.
- The harness (`eval/metrics.py`, `eval/backtest.py`) has log loss, Brier, ECE + reliability, `crps_normal`, the totals baseline, `paired_bootstrap_ci` and the walk-forward replay. Reuse it rather than duplicating it.
- `config.py`: EL live backtest splits are (2023,)/(2024,)/(2025,), with a history backtest over 2007–14/2015–23/2024–25. GBL live splits are 2018–21/2022–23/2024–25, frozen at K40/HCA130/rev0.25. These define the **live Elo** and must not change (see E-b).
- **Trap:** plain `eurohoops ingest` rewrites EuroLeague staging with the default seasons 2023–26 and drops the history locally. Restore with `--seasons 2007 … 2026`.
- The daily workflow runs `ingest` without `--details`, so live-season box scores are not cached daily yet.
- Bash heredocs holding Python source have turned `\n` escapes into real newlines before. Write code with the Write/Edit tools.

## 3. Unverified (verify, then record in `docs/` and a test)
- Whether EuroLeague box scores carry team rebounds and team turnovers separately from player totals, and whether they exist for every season from 2007 on.
- How close box-formula possessions are to PBP-counted possessions on the 50-game stint sample (PLAN §4.4 wants ±2 per team per game).
- Whether ESAKE `TO` includes team turnovers, and whether the 2018-20 PBP-filled games have rebounds and turnovers.
- Whether margin variance depends on expected pace or on the rating gap (R4 says the evidence for heteroscedasticity is weak; test it and report the result).

## 4. Deliverables

### E1: Team-game possession mart (both competitions)
- Parse full team box lines from the **cache only**: EL box JSON, and ESAKE HTML (extend `parse_box_score` with OREB, DREB, TOV, the team row, FGA/FTA). Add a pandera contract for each and update `tests/fixtures/`.
- Build a `team_games` table in the marts: one row per team per game, with competition, season, game_id, team, opponent, home flag, points, FGA, FTA, OREB, DREB, TOV, possessions and minutes.
- Possessions (R1): `FGA − OREB + TOV + 0.44·FTA` per team, and the game possession count = mean of both teams. Store both the raw and the game values.
- For EL, also count possessions from PBP (possession ends at a made FG, a defensive rebound, a turnover, or the last FT of a trip) on the 50-game stint sample.
- **Done when:**
  - `team_games` covers every played game of both competitions or lists each missing one with a reason.
  - A hand-computed test on one real box score gives the exact possession number.
  - On the 50-game EL sample, PBP and box possessions agree within ±2 per team in ≥ 90% of games, **or** the gap is explained in `docs/data/possessions.md`.
  - Points in `team_games` equal the `games` scores for every game.

### E2: Stints mart (EuroLeague, 2011-12 onward)
- Build a `stints` table from `parse/stints.py`: game_id, period, start/end seconds, the 5+5 player ids, points for/against and possessions for/against (from the E1 PBP possession counter).
- Build a `stint_game_checks` table: the four checks from the spike plus a possessions check, per game. Store failing games with their reason; never drop them silently.
- **Done when:**
  - Pass rates per season are in `reports/stints_mart.json`. The 2015+ rate is ≥ 98.5% and 2011–14 ≥ 95%; if either falls below, that is a regression from the spike, so find it.
  - Stint points sum to the final score for every passing game.
  - Two builds give byte-identical tables.

### E3: M1 model (`models/team_eff.py`)
- Adjusted efficiency: weighted ridge on team-game rows, `ORtg = μ + home·h + off[team] − def[opp]`, weighted by possessions × exponential time decay (half-life tuned). Ratings shrink toward the league mean at season start, with the amount tuned.
- Pace: `poss = μ_p + pace[team] + pace[opp]`, the same way.
- Forecast per game: expected points per team → expected margin and total. Margin distribution per E-e; P(home) = P(margin > 0). Totals distribution per E-e.
- Walk-forward: refit before each round's first tip-off, using only earlier games. It must be fast enough that a full EL backtest runs in under 10 minutes locally; measure and record it.
- **Done when:**
  - A 3-team toy season gives ratings equal to the closed-form ridge solution within 1e-9.
  - A synthetic league generated from known ratings is recovered with correlation > 0.95.
  - Student-t CRPS matches a numerical integral within 1e-6 (add `crps_student_t` to `metrics.py`).
  - mypy strict is clean.

### E4: Backtest, validation gate and test
- `eurohoops backtest --model m1 [--competition gbl]` writes `reports/backtest_m1.json` / `backtest_m1_gbl.json`. Per split (tuning, validation, and test once, after the gate), it reports log loss, Brier, accuracy, ECE + reliability, margin MAE + CRPS, totals MAE + CRPS, for M1, the comparison Elo (E-b) and B0.
- Pre-declare the variants (constraint above), tune on tuning only, then run validation.
- Gate per competition: M1 log loss < comparison-Elo log loss on validation. Report the paired bootstrap 95% CI of the per-game difference, plus the same for margin and totals CRPS.
- For GBL, the spread and total errors are the more informative metrics (PLAN §5). Say so in the verdict.
- Commit the verdict (pass or fail per competition) in the progress file, **then** score the test seasons once and add them to the report.
- If M1 fails on a competition, the report must show which variants were tried, where M1 loses (reliability bins, lopsided games, early-season rounds) and a hypothesis. Failing is an accepted outcome; hiding it is not.
- **Done when:** both reports exist with every field above, the verdict commit precedes the test run in `git log`, and `eurohoops backtest --model m1` is reproducible (two runs, identical JSON).

### E5: MLflow tracking
- Every `backtest --model m1` run logs:
  - params (all hyperparameters and the variant name)
  - a data snapshot hash (sha256 of the sorted game rows used)
  - the git commit and a dirty flag
  - every metric in the report
  - the report JSON as an artifact
- One run per variant, nested under one parent run per backtest.
- Tests use a temporary tracking URI and never write to `mlruns/`. CI and the daily workflow work without MLflow installed (`--no-dev`).
- **Done when:** a query of the local store (`mlflow.search_runs`) returns, for the last backtest, one parent run and N children with non-empty params, metrics, hash and commit. `mlruns/` is gitignored.

### E6: Leakage tests for M1 (extend `tests/test_leakage.py`)
- M1 forecasts for every game before time T are bit-identical when any game at or after T is added, deleted or altered (score, box line, possessions).
- A game's own possession row never feeds its forecast.
- The tuning-fitted values (hyperparameters, σ, df) are unchanged by any edit to validation or test games.
- A guard test proves the same edits do change the later forecasts, so the tests can fail.
- **Done when:** the tests pass for EL-like and GBL-like synthetic histories, and each one fails when you deliberately break the cutoff (show that once in the progress file, then revert).

### E7: Live M1 (only for a competition that passed the gate; skip with a note otherwise)
- The daily workflow caches live-season box scores (`ingest --details` for EL, the box for GBL; only new games are fetched) before `build`. `tests/test_workflow.py` asserts the step order.
- `predict` also appends M1 rows (P(home), expected margin, expected total, σ, model_version, predicted_at, tipoff_utc) to `predictions/{comp}_m1_2026-27.csv`, append-only, only for games not already logged at that version.
- `score` adds an M1 section and its rolling-50 line to the scorecard JSON. `publish` adds one M1 row to the site scorecard, and `tests/fixtures/site.json` is regenerated.
- **Done when:** a dry run on a fixed clock with a temporary log writes M1 rows and leaves the Elo log byte-identical. A second run on the same clock writes nothing.

### E8: Model card
- `docs/models/m1.md` covers:
  - data and the excluded share
  - the possession definition and its validation
  - the model equations
  - the chosen variant and why
  - validation and test results per competition, with CIs
  - calibration
  - the heteroscedasticity finding
  - limitations and what failed
- **Done when:** every number in the card matches the report JSON; a test reads both and compares them.

## 5. Out of scope (do not build)
xPTS, RAPM or any player-level model, GBL stints, the state-space model (M1 v2), blending M1 with Elo, the MLflow registry or server, FastAPI, Docker, new site pages, betting language.

## 6. Verification checklist (run top to bottom; all must pass in one run)
Local gate, identical to `.github/workflows/ci.yml`:
1. `uv sync --frozen`
2. `uv run ruff check .`
3. `uv run ruff format --check .`
4. `uv run mypy src`
5. `uv run pytest -q --cov=eurohoops --cov-fail-under=85`
6. `uv run vulture src vulture_whitelist.py --min-confidence 60`
7. `cd web && npm ci && npm run build` (with `tests/fixtures/site.json` copied to `web/src/data/site.json`).

No regressions:

8. `uv run eurohoops backtest` and `--competition gbl` reproduce every pre-existing value in the three Elo backtest reports (the existing old-vs-new test in `tests/test_reports.py` passes).
9. `git diff main -- predictions/` shows only added lines or new files, and both Elo `model_version` values are unchanged.
10. `uv run eurohoops stints` still gives the 50/50 sample report, byte-identical to main.

M1 correctness:

11. E1 checks: `team_games` coverage, points = `games` scores, the possession agreement rate on the sample (the number, or the documented explanation).
12. E2 checks: `reports/stints_mart.json` pass rates at or above the thresholds, and two builds identical.
13. E3 unit tests (closed-form ridge, synthetic recovery, Student-t CRPS) and the recorded EL backtest runtime < 10 min.
14. E6 leakage tests pass.
15. `reports/backtest_m1.json` and `backtest_m1_gbl.json` exist with validation and test splits; the pre-declared variant list and the verdict commit precede the test run in `git log`; two backtest runs give identical JSON.
16. MLflow: the last backtest's parent run and child runs have params, metrics, data hash and commit (query output pasted).
17. E7: if any competition went live, the dry run writes M1 rows and leaves the Elo logs byte-identical; the second run writes nothing. Otherwise, a `SKIPPED (gate failed)` line with the reason.
18. `uv run eurohoops build && uv run eurohoops score && uv run eurohoops publish`, run twice, leave the tree unchanged the second time.
19. `actionlint` passes on both workflows.
20. If the site changed: screenshots at 1440 px and 390 px, light and dark, with no horizontal page scroll.

## 7. Final report (`reports/week5-7_closeout.md`)
Include:
- the checklist with pass, BLOCKED or SKIPPED per item, and the output of the single consecutive §6 run
- the **gate verdict per competition** (M1 vs comparison Elo on validation: log loss, Brier, ECE, margin and totals CRPS, with bootstrap CIs), then the test numbers
- every variant tried, and which were post-hoc
- the possession validation result and the stints mart pass rates
- the heteroscedasticity finding
- MLflow run ids
- every decision this file did not cover
- open questions for the user

Then append a session entry to `memory.md`.
