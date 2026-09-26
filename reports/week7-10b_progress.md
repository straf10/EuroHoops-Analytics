# Weeks 7–10b progress log (M2 follow-up: flags, FT shares, runtime, season level)

Decisions (user, 2026-09-26): user decisions 1–4 and sub-decisions G-a … G-g, all defaults
(`prompts/week-7-10b.md` §0, commit c2d850a). Branch `week-7-10`, not pushed.

Format: `iteration N | deliverable | checks run | result | commit`

## Machine state
- 2026-09-26 ~21:50 local: CPU load 3%. Not started by me: `astro dev --port 4321` from
  `C:\Python\eurohoops-ui\web` (node 5520, 11732, started 21:46) and `cswap.exe auto` (python
  14084, 14376). Not killed. Before any timed run I check again and ask the user if the dev
  server is still up (it made BLAS threads spin last time).

- 2026-09-26 (asked before any timed run): **the user chose "run anyway"** with the dev server
  up. Every runtime below is measured with that `astro dev` server running unless stated.

## G2 pre-run finding and the user's decision (before any real-data share number existed)
Simulation on synthetic seasons only (the model right by construction; 200 seasons, ~30 games
per team; `tests/test_free_throws.synthetic_season`): the declared rule "every statistic within
2 SE in every held-out season" fails at least one of the 6 band checks in 27% of seasons (per
band 3.5–10%; the rarest band highest), the team mean-|gap| check in 0%, the r check in 1%;
P(all 13 held-out seasons pass) ≈ 0.015. Cause: ~104 checks at a ~4.6% two-sided false-alarm
rate each. Asked the user; **answer (2026-09-26): count rule.** 2 SE per statistic stays and
each statistic is still flagged; the gate passes when the number of flagged band checks over
all held-out seasons is ≤ the 95th percentile of Binomial(n band checks, 0.0455) (≈ 7 of 78),
and every season's two team checks pass. Fixed in code before the first real-data run.

## Decisions not covered by the task file
- (G2) **Team share = share of per-game rates** (a team's FT points per game over the sum of all
  teams' per-game rates), so playoff games do not inflate a team's share; equal to the plain
  share when every team plays the same number of games.
- (G2) **"2 × SE" for the two team summaries** (G-b does not say what the target is): mean
  |share gap| ≤ 2 × the RMS of the teams' bootstrap gap SEs (the noise scale of one team's gap;
  a right model sits near 0.8 of it); Pearson r ≥ 2 × its bootstrap SE (expected shares track
  actual ones beyond noise). The band gaps have target 0: |gap| ≤ 2 SE.
- (G2) Actual and-one points per band needed new mart columns `and_one_ftm_{band}` in
  `ft_team_games`.
- (G1) **The audit also fails a missed-only level** (make rate ≤ 0.01), not only ≥ 0.99: a
  blocked-shot code is as outcome-coded as a scoring tag. Stricter than asked, not weaker.
- (G1) **A level counts only with ≥ 100 development shots** (`MIN_SET_SHOTS`): a handful of
  shots cannot tell a tag from chance. **Continuous columns** (> 32 distinct values: distance,
  spline basis, clock, margin) are not audited; a flag or one-hot level has ≤ 32 values.
  Fixed before the audit first ran on real data.
- (G1) **The audit runs at the start of every M2 backtest** too (raises on a hit), so it is live
  code, not only a check; it changes no number.
- (sequencing) **Full §6 runs are batched:** items 23/24 run the M2 backtest twice (~100 min
  today) and are red on runtime until G3/G4. I mark G1 and G2 done only after a full run; that
  first full run comes once G3/G4 are in. Each deliverable still gets the fast gate plus every
  item it touched.

## G1 evidence (2026-09-26)
Item 32 on the development shots (385,604): spline 15 flag/level columns (26 levels, make rates
0.3613–0.5754); LightGBM 5 columns (29 levels, 0.0256–0.9378; the extremes are zone J beyond
half court and zone A at the rim, real geometry). Guard: the three feed flags audited as
features are caught (fastbreak 0.9997, second_chance 0.9992, points_off_turnover 0.9997).

Deliberate breaks, each reverted right after (tree clean, check green again):
- `fastbreak` put back in the LightGBM builder under its own name → the name guard raises at
  import (`ValueError: identity or outcome-coded column in the M2 features: (..., 'fastbreak')`).
- the same column renamed `transition` (the name guard cannot see it) → item 32:
  `OUTCOME-CODED transition = 1: 11537 shots, make rate 0.9997…` → FAIL, exit 1.
- `second_chance` added to the spline builder as `putback` → item 32:
  `OUTCOME-CODED putback = 1: 17459 shots, make rate 0.9992…` → FAIL, exit 1.

## G2 result (first real-data run, code committed first at 1b09043)
`bash scripts/checks/free_throws.sh` → **FAIL** (item 22). Band flags 12 of 78 (allowed 7):
2013 deep3; 2014 long2, deep3; 2015 three; 2019 three; 2020 mid, three; 2021 rim; 2022 rim,
mid; 2023 rim, short. Team mean |gap| within 2 × noise in all 13 seasons (e.g. 2017 0.00783 ≤
0.00890). Team r below its 2-SE bar in all 13 (−0.421 in 2017 to 0.298 in 2012).

**Diagnosis (checked by hand for 2017 and 2021, not a bug).** Recomputed the team shares
directly from `ft_team_games`: same r (2017 −0.421, 2021 −0.036). Expected team shares barely
vary (sd 0.0025 vs actual 0.0075 in 2017; 0.0018 vs 0.0064 in 2021), and FT points per game
correlate *negatively* with FGA per game across teams (2017 r = −0.44): a shooting foul on a
miss ends a possession without an FGA, so FT trips and FGA compete. The F-e model's "other
trips per FGA × FGA" assumes the opposite. It has no team foul-drawing term, so it cannot
reproduce team shares; the band flags come from band and-one rates that move from season to
season (e.g. 2021 rim share 0.522 actual vs 0.576 expected on 742 and-one points).
**Not changed:** the F-e model is not a G2 deliverable, and changing the check after seeing
this would be tuning it to pass. Item 22 stays red; it is not on the list of items allowed to
stay red. Next step (for the user): a team-level FT model (trips per possession plus a shrunk
team foul-drawing rate) declared as a new variant.

## §3 unverified facts, now verified (G3)
- **Parallel processes, same `num_threads`:** 6 LightGBM fits (LOSO folds 2011-2016, seed
  20261001, stored parameters, real development shots) run 1, 2 and 3 at a time in separate
  processes give **byte-identical predictions** (same SHA-256 of every prediction array in all
  three modes). Throughput: 1 process 6.5 s per fit (incl. loading), 2 processes 4.4 s,
  3 processes 4.6 s (18 threads on 12 logical cores oversubscribe). Test:
  `test_parallel_fits_equal_sequential_fits`. Measured with the `astro dev` server up.
- Timings per phase: see `docs/models/m2_runtime.md` (from the TIMING lines of the runs below).

## G3 runs
- **Before** (HEAD 8a58e9d: sequential fits; code = weeks 7-10 + the G1 audit + timing lines),
  `backtest --model m2 --score-test`, 19:08Z-19:57Z: **2,921 s wall** (backtest 2,909 s). The
  report and the marts' `shot_xpts` came out byte-identical to the committed ones (tree clean),
  so the refactor and the audit change no number. Machine: `astro dev` (node 5520) was busy
  the whole time (its CPU time rose from 15 s to 3,661 s), and my own test runs overlapped
  ~2 min of seed 1. MLflow parent fa928c5ebd6e4ba6a904ce6d8ad08f3e.
  RUNTIME backtest 2921 s
- **After** (HEAD b0b45e3: LightGBM in 2 processes; + the G5 level variants), same command,
  19:58Z-20:44Z: **2,743 s wall** (backtest 2,728 s, of which the new level variants 169 s).
  Like for like without G5: 2,559 s vs 2,909 s (−350 s, −12%). LightGBM seeds 357, 357, 451,
  448, 459 s vs 486-493 s: seeds 1-2 −28%; seeds 3-5 only −8% while `astro dev` competed.
  **Every weeks 7-10 field of the report is byte-identical** (the report minus
  `level_variants`, dumped the same way, equals the committed file byte for byte);
  `m2_teams`/`m2_players` checked by item 25. MLflow parent edee8ed94dcd4cd09eb110dfa3d66a34.
  This run is the first G5 run on any split (declaration 1de203c came before it).
  RUNTIME backtest 2743 s

- 2026-09-27: the UI dev server (node 5520) is at ~1.6 cores continuously (CPU time 3,661 s →
  13,050 s during the verification run; machine at 100%). Asked the user; they are working in
  the UI worktree and **cannot stop it**. All remaining timed runs are measured with it up.

- **After 2** (HEAD 2068902: + spline in the pool, the chosen spline reuses the grid's LOSO
  fits, zone codes once), 20:44Z-21:31Z: **2,774 s wall** with the UI server at ~1.6 cores
  (machine 100%). Spline 252 s (was 368), LightGBM 2,249 s (was 2,447; seeds slowed 393 → 529 s
  as the load grew). Weeks 7-10 fields byte-identical to the declaration commit's report; the
  level block identical to After 1 (plus the new `mean_offset`). MLflow parent
  57f0e20f166d44cf904e0426a05b186e.
  RUNTIME backtest 2774 s

## G5 leakage tests: deliberate breaks, each shown once and reverted (tree clean after)
- (a) offset uses its own tip-off group (`n = int(b)`): FAILED
  `test_an_offset_uses_only_games_that_tipped_off_earlier`,
  `test_no_level_offset_uses_its_own_game_or_a_later_one[spline, lgbm]`.
- (b) development prior from the previous season's LOSO prediction (that model trained on
  season s): FAILED `test_no_level_offset_uses_its_own_game_or_a_later_one[spline, lgbm]`
  ("Mismatched elements: 348 / 348").
- (c) prior from two seasons back: FAILED
  `test_other_seasons_reach_an_offset_only_through_the_prior_and_k` (2011 reached 2013 with k
  held). After each revert: 13/13 green. Guards in the same tests prove the edits reach later
  games, the next season (prior) and k.

## G5 result (post-hoc, not a clean hold-out)
- `lgbm_level`: CV log loss 0.629697 (lgbm 0.629699); validation log loss 0.633309, ECE
  0.01022 → **does not meet F-f** (target 0.010); test ECE 0.00947 but a bin outside ±0.02 →
  F-f false on test too. Level − lgbm validation log loss +0.000042 [−0.000049, +0.000139]:
  no gain. k 2,312 shots (τ² 0.00196): after ~2,300 shots (≈ 15 games) the season's own level
  gets half the weight.
- Calibration in the large improves: `lgbm_level` 0.9835-1.0134 over development (base
  0.9766-1.0198); 2023 0.9893, 2024 0.9934, 2025 1.0030 (base 0.9945, 0.9714, 0.9874).
- `spline_level`: validation ECE 0.009695 → meets F-f on validation (not on test, 0.01198);
  log loss vs spline +0.000009 [−0.000228, +0.000234].
- G-g condition: meets F-f on validation False, CV below lgbm True → **does not hold**; F6/F7
  are not recomputed with `lgbm_level`. Hypothesis: the season offset fixes the league-wide
  level (calibration in the large) but F-f's failing bins sit at P ≈ 0.40 and 0.69, a shape
  error by distance band (deep 3s, long 2s), which one additive offset per season cannot fix.

## Declared season-level variants (G5; §0 G-e, G-f) — committed before any run on any split
No `lgbm_level` or `spline_level` number exists on any split, development, validation or test,
when this is committed. Validation 2023-24 and test 2024-25/2025-26 have been seen by the weeks
7–10 run, so **every G5 number on any split is post-hoc, not a clean hold-out**; the clean
evaluation is the live 2026-27 season (rule below). The weeks 7–10 verdict (84308f9) and every
committed number stay as they are.

Bases (unchanged, from the same backtest run): `lgbm` = the 5-seed mean LightGBM with the stored
Optuna parameters (the weeks 7–10 chosen M2); `spline` = the grid-chosen spline. For a shot i in
game g of season s:

    p_level = sigmoid(logit(p_base_i) + o_s(g)),   p_base clipped to [1e-6, 1 - 1e-6]
    o_s(g)  = w * d_s(<g) + (1 - w) * prior_s,     w = n / (n + k)

- n = season-s shots in games that tipped off **strictly before** g (`games.tipoff_utc`);
  d_s(<g) = the maximum-likelihood intercept shift on those shots: the d solving
  Σ (y − sigmoid(logit p_base + d)) = 0 (Newton). n = 0 (the season's first tip-off) → w = 0,
  o = prior.
- prior_s = the previous season's final offset = the same MLE shift over **all** shots of season
  s − 1, with base predictions for s − 1 from a model trained on **neither s nor s − 1**:
  development s → the leave-two-out model (s, s − 1) (the pair fits the nested isotonic already
  makes; seed mean for LightGBM); validation 2023 → 2022's LOSO prediction (fitted on
  development without 2022, never on 2023); test 2024 → 2023's development-fitted prediction;
  test 2025 → 2024's development-fitted prediction. First development season 2011: prior 0
  (2010-11 is not a validated season).
- k (empirical Bayes, development seasons only): for fitting set F, D_t = full-season MLE shift of
  season t, I_t = Σ p(1 − p) over t's shots (so Var D_t ≈ 1/I_t); over consecutive pairs
  (t − 1, t) both in F: τ² = mean((D_t − D_{t−1})²) − mean(1/I_t + 1/I_{t−1}); v̄ = mean p(1 − p)
  over F's shots; k = 1 / (v̄ τ²) (τ² ≤ 0 → w = 0, prior only). F and its predictions:
  development season s → the development seasons other than s, D_t from the pair model (s, t);
  validation and test → all development seasons, D_t from their LOSO predictions. So no offset
  used for a season-s game depends on a season-s shot of that game or a later one, nor (for
  development) on any season-s shot through the prior or k.

Evaluation (every number labelled `post_hoc: true`, "post-hoc, not a clean hold-out"),
development LOSO, validation and test: log loss, Brier, ECE + reliability (overall, per band,
2s vs 3s, contexts) as for the other variants; calibration in the large per season (Σ xPTS /
Σ actual FG points); `lgbm_level − lgbm` and `spline_level − spline`: log loss and Brier with
the game-level bootstrap 95% CI (1,000, seed 20261001) and the ECE difference with its CI; F-f
on validation for both (and on test, reported). G-g: F6/F7 are also computed with `lgbm_level`,
side by side and post-hoc, only if it meets F-f on validation **and** its development LOSO log
loss is below `lgbm`'s. Either outcome is accepted; no second variant is tried to make it pass.

Pre-registered live rule (2026-27, `scripts/checks/m2_live_level.py`, run only after the
season's last game): 2026-27 shots built from the cache with the F1 rules; `lgbm` = the 5 seeds
refitted on 2011-12 → 2025-26 with the stored parameters; `lgbm_level` offsets from 2026-27 games
before g, prior = the full-season shift of 2025-26 under the 5 seeds fitted on 2011-12 →
2024-25, k = the report's validation/test k. Both scored on every 2026-27 shot: F-f (ECE ≤
0.010 with 20 equal-count bins, every bin with ≥ 500 shots within ±0.02), log loss, Brier, and
`lgbm_level − lgbm` log loss with the game-level bootstrap 95% CI. Rule: `lgbm_level` is
calibrated on 2026-27 iff it meets F-f; it improves on `lgbm` iff that CI's upper bound < 0.

LEVEL_DECLARATION 1de203c
LEVEL_RUN 327d0a3

## Loop log
iteration 1 | G1 flags dropped for good + outcome-coding audit (item 32) | tests/test_feature_audit.py (6), fast gate 1-6, 14, 19, 32 (green after a format fix), items 23/28/29 tests | green; full §6 pending (see sequencing) | 62b9526
iteration 2 | G2 F2 share checks (code 1b09043 before the run) | tests/test_free_throws.py 11 (3-SE planted errors fail), item 22 real run, card test | item 22 FAIL (12/78 band flags > 7; team r below bar in 13/13) - diagnosed, not tuned; card updated | 8cc5640
iteration 3 | G3 runtime (profile, class A: LightGBM+spline pool, LOSO reuse, zone codes once) + G5 level variants | before 2,921 s, after-1 2,743 s, after-2 2,774 s (UI server ~1.6 cores); weeks 7-10 bytes identical each time; level leakage tests; card test | G3 under budget; G5 lgbm_level misses F-f (post-hoc) | 5328d64

## Stopped by the user (2026-09-27, ~21:45Z): resume here
The user stopped all runs for the day. The G4 demo (item 24 with a temporary 1 s budget) was
killed during its first backtest and reverted (`limit=3600`; tree clean; report untouched).
State:
- G1 done. G2 done, item 22 FAILS (diagnosed, not tuned). G3 done (under budget, bytes
  identical). G4 code done; **demo not shown**. G5 done (lgbm_level misses F-f, post-hoc);
  leakage breaks shown.
- **Not done:** G4 demo; the full §6 checklist in one run (items 1-32); the closeout
  `reports/week7-10b_closeout.md`; G1/G2 marked done only after that full run.
- Next session: decide with the user whether to run the full checklist (~3 h; its item 24 runs
  both backtests and prints all three G4 lines) or close out from the per-item evidence here.
