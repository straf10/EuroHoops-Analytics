# Agent Task: EuroHoops Analytics, Weeks 9–12: player impact M3 (RAPM → Bayesian RAPM → box-score prior → GBL transfer)

## Role & goal
You are the **orchestrator** for M3. You plan, delegate parallel subtasks to Sonnet subagents,
verify everything they hand back yourself, run every real-data command, and own the result.
Read first, in this order: `docs/PLAN.md` §5.4 and §8 (row 9–12), `reports/week7-10b_closeout.md`,
`docs/models/m1.md`, `docs/spikes/stints.md`, `docs/spikes/gbl-pbp.md`, `docs/CONTEXT.md`,
`src/eurohoops/parse/stints.py`, `src/eurohoops/parse/stints_mart.py`, `src/eurohoops/config.py`,
`src/eurohoops/eval/m1_backtest.py` (the pattern to copy for walk-forward evaluation),
`scripts/checklist.sh`.

**Goal, in one sentence:** build a player-impact model for EuroLeague (RAPM, then a Bayesian
RAPM with intervals, then a box-score prior "SPM"), prove on held-out seasons that it predicts
future game margins better than a box-score-only model, and transfer it to the GBL through the
box-score prior, with every number reproducible and every check green.

**Exit gate (PLAN §8, row 9–12):** stints validated; RAPM (+dummy), Bayesian RAPM, SPM prior and
GBL transfer built; **the chosen RAPM variant beats box-only on future-margin prediction on
validation** (rule in §0 H-d).

## How you work: the loop
Repeat until the stop condition holds. Do not stop early; do not call a partial run a success.

1. Read `reports/week9-12_progress.md` and `git log --oneline -15` (iteration 1: create the
   branch `week-9-12` from `main` and the progress file with §0's answers). Pick the next
   unfinished work: a wave of parallel subtasks (§5) or a deliverable you do yourself.
2. Delegate or implement it, with tests.
3. **Verify, never by trust.** When a subagent reports done, you re-run its Done-when checks
   yourself in the main tree after merging its branch; a subagent's "all green" is not
   evidence. Order:
   - the deliverable's **Done when** checks (§4), each an automated test or a script under
     `scripts/checks/`, never by eye (charts: open each PNG with the Read tool and log it);
   - the fast gate every iteration: `SKIP="7 8 10 12 13 15 16 18 20 21 22 23 24 25 26 30 31"
     bash scripts/checklist.sh`, plus every §7 item the change touched;
   - the **whole** §7 checklist when a deliverable is finished, before you mark it done.
4. If anything is red: find the root cause, fix it (or send the subagent back with the exact
   failing output), go back to 3. Never weaken, skip, `xfail` or delete a check; never raise a
   tolerance without a written justification in the progress file that does **not** use the
   observed failing numbers as the reason.
5. Green → small commit → append `iteration N | deliverable | checks run | result | commit` to
   the progress file.
6. **Stop condition:** H0–H9 done **and** the full §7 checklist passes top to bottom in one run
   with no edits in between, or every red item is one this file explicitly allows to stay red
   (only items 22 and 25, the known M2 limitations, and the H-d gate *outcome*) with its
   evidence. Paste that run's output into the closeout.

Guards:
- **Same failure three times** (same check, same error): stop patching; write the diagnosis in
  the progress file; re-read code and data; fix the cause.
- **Iteration budget: 30.** At the end of iteration 30 stop anyway; write the closeout with
  status `INCOMPLETE`, each red check with its last error, and what you would try next.
- **Quiet machine for timed checks.** Before any full checklist run or timed backtest, check CPU
  load (`(Get-CimInstance Win32_Processor).LoadPercentage` in PowerShell) and list `node`/`python`
  processes you did not start. If other work is running, record it and wait or ask the user;
  never kill processes you did not start. No subagent may be running during a timed run.
- **DuckDB is single-writer.** Never run two commands that write `data/marts/eurohoops.duckdb`
  at once. Only you (the orchestrator) run commands that touch `data/`.
- **Resumability:** progress file + commits are your memory; after any interruption restart at
  step 1.
- **Heredocs:** Bash heredocs holding Python or Markdown have broken before in this environment.
  Write files with the Write/Edit tools.

## 0. Sub-decisions (defaults in bold; confirm with the user once, before starting, and record)
| # | Question | Default |
|---|---|---|
| H-a | Seasons | **Same as M1: warm-up 2011–2014 (stints exist from 2011; 2007–10 excluded, whole-minute sub clocks), tuning 2015–2022, validation 2023, test 2024–2025 (scored once, after the verdict commit). 2026-27 is not touched.** |
| H-b | RAPM target and design | **One row per stint side (offense team's five +1, defense five −1 in separate O and D blocks → O-RAPM and D-RAPM), target = points per 100 possessions of that side, weight = possessions, plus a home-court column and an intercept. Stint sides with 0 possessions dropped. Ridge penalty λ (one for O, one for D, or shared: decide on tuning) and an exponential time decay half-life, both chosen on tuning seasons only by future-margin RMSE (R6 protocol), never on validation or test.** |
| H-c | Future-margin protocol | **Walk forward by round inside each evaluated season: ratings fitted on all stints of games that tipped off before the round's first tip-off (earlier seasons included, decayed), then each game of the round is predicted: margin = home court + Σ_home w_p·r_p − Σ_away w_p·r_p, r_p = O-RAPM + D-RAPM per 100, w_p = the player's projected share of the team's minutes × team possessions / 100 (possessions from the team's season-to-date pace). Projected minutes = the player's minutes share over the team's previous 5 games this season (first round: last season's share, 0 for new players). A second "oracle minutes" variant (the game's actual minutes) is reported, labelled "oracle, not a forecast", and never used for the gate.** |
| H-d | Gate rule | **Primary metric: validation margin RMSE (plus MAE and the log loss of P(home win) from a normal margin with σ fitted on tuning). The chosen RAPM variant beats box-only iff the game-level bootstrap 95% CI (1,000 resamples, seed 20261001) of RMSE(RAPM) − RMSE(box-only) has upper bound < 0. Also reported, not gated: vs M1 (team-only) and vs PIR. The variant is chosen on tuning only, and the choice is committed (verdict commit) before validation is scored; test is scored once after that.** |
| H-e | Box-only baseline | **A box-score regression on the same inputs RAPM sees: per-player per-100 rates (PTS, 2PA, 3PA, FTA, OREB, DREB, AST, STL, BLK, TOV, PF) from games before the round, shrunk toward the league mean with a minutes-based weight, combined linearly (ridge, fitted on tuning seasons with the future-margin target), same projected minutes as H-c. PIR per minute with the same minutes is a second, naive baseline.** |
| H-f | Dummy RAPM (R8) | **Players under a minutes threshold in the fitting window share one "replacement" column per team-season instead of their own column. Threshold chosen on tuning from {0 (off), 50, 100, 200, 400} minutes. Declared as a variant `rapm_dummy`; kept only if it wins on tuning.** |
| H-g | Bayesian RAPM | **Closed-form conjugate Gaussian (no PyMC): prior r ~ N(m, τ²I), noise variance from the tuning residuals, posterior mean = the ridge solution, posterior covariance = (XᵀWX/σ² + I/τ²)⁻¹ (dense inverse is fine at ~2,000 players; use sparse/CG if not). 90% intervals per player. PyMC only if the user asks for it.** |
| H-h | SPM prior | **Regress multi-season RAPM (ratings from H-b on 2011–2022, fitted without the season being predicted: leave-one-season-out) on per-100 box rates weighted by minutes. `rapm_spm` = ridge toward the SPM prediction instead of 0. Declared variant, same protocol and gate as the others.** |
| H-i | GBL | **Build GBL stints from `gbl_pbp` (subs exist; `docs/spikes/gbl-pbp.md`) with the same validation checks as EuroLeague (5 on court, stint points = final score, possessions ±2). Evaluate on GBL validation/test seasons of `config.GBL.m1`: SPM-transfer (the EuroLeague SPM applied to GBL box rates) vs GBL box-only vs PIR, same H-c protocol. GBL RAPM only if GBL stints pass on ≥ 95% of games; otherwise document why and report SPM-transfer alone.** |
| H-j | Runtime budget | **`eurohoops backtest --model m3` (EuroLeague, tuning + validation + all variants) < 30 min on a quiet machine; recorded as a `RUNTIME` line and checked by item 36.** |

## 1. Hard constraints
- **Scope = this file.** No site changes, no live 2026-27 M3 predictions, no M4/M5/M6 work, no
  change to M1, M2 or Elo.
- **Live logs are sacred:** `predictions/*.csv` and `odds/*.csv` byte-identical; `git diff
  origin/main -- predictions/ odds/` shows no changes from you (the daily bot may append rows).
- **Existing reports don't move:** Elo, M1, M2, possessions, stints, shots, free throws keep
  every value (items 8, 15, 24 prove it). New numbers go in new files: `reports/backtest_m3.json`,
  `reports/m3_players.json`, `reports/gbl_stints.json`, `reports/backtest_m3_gbl.json`.
- **Walk-forward only:** no rating, λ, decay, shrinkage, SPM coefficient or σ used to predict a
  game may depend on that game or any later one. Leakage tests prove it (item 34).
- **Code rules (CLAUDE.md):** source team codes everywhere; pandera schema + `tests/fixtures/`
  for every new table; code ahead of its step whitelisted in `scripts/vulture_whitelist.py`
  with a reason; no dead code; `uv run eurohoops <cmd>` entry points in `cli.py` (or
  `research.py` for research-only commands).
- **Git:** branch `week-9-12`, small commits. **Commit messages carry no `Co-Authored-By` line
  and no other AI attribution** (the owner's rule overrides any default footer; tell every
  subagent the same). Do not push. Do not merge to `main`.
- **Do not change any value in `PLAN.md`.** Report factual errors instead.
- New dependencies only with the user's approval (none expected: numpy, scipy, pandas, duckdb
  cover H-b to H-h).

## 2. Verified facts (do not re-discover; do verify in tests)
- Stints mart (`reports/stints_mart.json`): EuroLeague 2011-12 on, 4,246 games, 141,625 stints;
  game check pass rate 95.9% (2011–14) and 98.9% (2015 on). Games failing a check are flagged in
  `stint_game_checks`, not dropped: exclude flagged games from fitting, keep them in evaluation.
- `GameStint` (`parse/stints.py`): `period`, `start`, `end`, `players` = (home five, away five)
  as sorted player ids, `points` (home, away), `possessions` (home, away).
- Stint sample validation (`reports/stint_validation.json`, 50 games 2015–2026): 5 on court,
  seconds, minutes and points checks pass in 100%.
- GBL play-by-play has substitutions ("entered/left the court"); replaying them left exactly 5
  per team at every clock step in the spike game (0 violations in 517 events). GBL stints are
  **not built yet**.
- M1 walk-forward backtest runs in 224–243 s on a quiet machine; the M2 backtest ~46–51 min.
  M1 is the team-only comparison (`reports/backtest_m1.json`).
- 12 logical cores; `FitPool` (spawn context) exists for parallel fits.

## 3. Unverified (verify first, record in the progress file and a test)
- How many distinct EuroLeague players have stints 2011–2025, and the minutes distribution
  (drives H-f thresholds and whether a dense posterior inverse fits in memory).
- Whether player ids are stable across seasons in the stints mart (same player, same id).
- Whether per-player box minutes and the stints' on-court seconds agree (sample of games, within
  60 s per player).
- Whether GBL box scores give every H-e rate (OREB/DREB split, PF) for every GBL season used.

## 4. Deliverables (each with its **Done when**)

### H0: Branch, progress file, decisions
Branch `week-9-12`; `reports/week9-12_progress.md` with §0 answers and §3 findings.
**Done when:** file committed; §3 facts each have a test.

### H1: Design matrix and walk-forward harness (wave 1, subagent A)
`models/rapm.py` (sparse design from stints, time-decay weights, ridge solve) and
`eval/m3_backtest.py` (round-by-round walk forward per H-c, projected minutes, metrics,
bootstrap CIs, report writer, MLflow like M1).
**Done when:** (1) synthetic recovery test: simulated seasons with known player ratings and
random lineups; RAPM ratings correlate ≥ 0.9 with the truth at 30k stints and the error shrinks
as stints grow; (2) a hand-computed 3-stint example matches the solver to 1e-9; (3) leakage
tests (item 34); (4) two runs on the fixture produce byte-identical reports.

### H2: Box-only and PIR baselines, projected minutes (wave 1, subagent B)
`models/box_impact.py` per H-e; projected minutes module shared with H1.
**Done when:** unit tests on fixtures; leakage test (editing a game's box score changes no
prediction for that game or earlier); the minutes projection is tested on a hand example.

### H3: GBL stints (wave 1, subagent C)
`parse/gbl_stints.py` + mart table + `reports/gbl_stints.json` with pass rates per check and
season, mirroring the EuroLeague stints mart.
**Done when:** pandera schema + fixtures; the three checks fail on a planted dropped
substitution; on real data (you run it) pass rates are reported per season and two builds are
byte-identical.

### H4: Bayesian posterior (wave 1, subagent D)
Posterior covariance and 90% intervals per H-g in `models/rapm.py` (coordinate with A through
you: D writes a function taking X, W, y, λ and returning mean + sd; A calls it).
**Done when:** simulation coverage test: over 200 synthetic seasons the 90% intervals cover the
true rating 87–93% of the time; the posterior mean equals the ridge solution to 1e-9.

### H5: Tuning and variant choice (you, real data)
Run the tuning grid (λ, decay, H-f thresholds) on tuning seasons; choose the variant; write the
choice and its tuning numbers to the progress file and commit it (**verdict commit**) before
scoring validation.
**Done when:** `git log` shows the verdict commit before any validation number exists (item 35).

### H6: Dummy RAPM and SPM prior (wave 2, subagents E and F, after H1 is merged)
E: `rapm_dummy` per H-f. F: SPM per H-h (leave-one-season-out fits, coefficients reported) and
`rapm_spm`.
**Done when:** each has unit tests and leakage tests; SPM LOSO: the SPM for season s never uses
season s's RAPM (test with a planted edit).

### H7: Validation, then test (you)
Score validation per H-d; then test once. Write `reports/backtest_m3.json` with a `gate` block
(rule, metric, CI, verdict) and `reports/m3_players.json` (per player-season: O/D/total rating,
90% interval, minutes).
**Done when:** items 33–37 green; the gate block states PASS or FAIL honestly.

### H8: GBL transfer (you, after H3 and H6)
Per H-i; `reports/backtest_m3_gbl.json`.
**Done when:** same protocol checks as H7 on GBL seasons; GBL RAPM included only if H3's pass
rate rule holds.

### H9: Model card, closeout, memory
`docs/models/m3.md` (method, data, choices, gate result, limitations, every number from the
reports); `reports/week9-12_closeout.md` (§8); a `memory.md` entry.
**Done when:** item 37 (card numbers equal report numbers) green; full §7 run pasted.

## 5. Subagents: how to delegate
Use the Agent tool with **`model: "sonnet"`** and **`isolation: "worktree"`** for every
subagent. Waves:

| Wave | Parallel subagents | Starts when |
|---|---|---|
| 1 | A (H1 harness), B (H2 baselines), C (H3 GBL stints), D (H4 posterior) | H0 done |
| 2 | E (H6 dummy), F (H6 SPM) | A and D merged and verified |

Each subagent prompt must be self-contained and include:
- the deliverable text from §4, the relevant §0 rows, §1 constraints, and the **Done when**
  checks as the definition of finished;
- the file boundaries it owns (no two subagents edit the same file; shared interfaces, like
  D's posterior function signature, are fixed by you in the prompt);
- "`data/` is not in your worktree and you must not read or write the main tree's `data/` or
  DuckDB file: build and test on `tests/fixtures/` and synthetic data only";
- "loop: run `uv run ruff check . && uv run ruff format --check . && uv run mypy src && uv run
  pytest -q` and your deliverable's tests; if anything is red, fix the cause and rerun; repeat
  until all green; never skip, xfail or weaken a test";
- "commit on your worktree branch with **no Co-Authored-By or AI attribution line**";
- "report: files changed, tests added, the exact commands you ran and their final output".

On return: read the diff, merge the worktree branch into `week-9-12`, re-run its Done-when
checks and the fast gate yourself. Red → send the same subagent (SendMessage) the failing output;
after two failed rounds on the same issue, fix it yourself. You never run a real-data command
while a subagent is running.

## 6. Out of scope (do not build)
Site pages or player cards, live M3 predictions, M4 league translation factors, M5 roster-aware
predictor, M6 projections, PyMC, any change to M1/M2/Elo numbers.

## 7. Verification checklist (run top to bottom; all must pass in one run)
Items 1–32 of `scripts/checklist.sh` unchanged (22 and 25 may stay red: known M2 limitations,
see `reports/week7-10b_closeout.md`), plus new items:
- **33** M3 unit tests: synthetic recovery (H1), hand example, posterior equals ridge, interval
  coverage 87–93% (H4).
- **34** leakage: editing any stint or box row of game g or later changes no rating, weight,
  SPM coefficient or prediction used for game g (RAPM, dummy, SPM, box-only, GBL).
- **35** `backtest_m3.json` complete with its gate block; in `git log`: verdict commit <
  validation scored < test scored; two runs byte-identical and equal to the committed report.
- **36** runtime: recorded `RUNTIME` of `backtest --model m3` < 1,800 s (H-j).
- **37** `docs/models/m3.md` numbers match the reports; `gbl_stints.json` pass rates present,
  two GBL stint builds identical.

## 8. Closeout (`reports/week9-12_closeout.md`)
- Status (COMPLETE / INCOMPLETE), exit-gate table, the checklist with PASS/FAIL/BLOCKED/SKIPPED
  per item and the full output of the single consecutive run.
- Tuning results and the chosen variant; validation and test tables (RAPM variants, box-only,
  PIR, M1) with CIs; the oracle-minutes variant labelled.
- Subagent log: which wave, what each delivered, how many rounds it needed, what you fixed
  yourself.
- Every decision this file did not cover; open questions for the user; then the `memory.md`
  entry.
