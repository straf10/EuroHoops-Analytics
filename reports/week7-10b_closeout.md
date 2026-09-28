# Weeks 7–10b closeout: M2 follow-up (flags, FT check, runtime, season level)

**Status: CLOSED WITHOUT THE SINGLE CONSECUTIVE RUN (owner's decision, 2026-09-28).** G1–G5 are
built, tested and committed; G6 is this file. The full §6 checklist was **not** run top to
bottom in one go: the owner judged the ~3 h run not worth it, because every item already has its
own evidence (below) and the run could not change any outcome (item 22 fails by design of the
F-e model, not by a bug). The long items are evidenced by the runs recorded in
`reports/week7-10b_progress.md`; the short items were re-run on 2026-09-28 (fast run below).
Nothing was weakened, skipped or re-run to make it pass.

## What the four user decisions (2026-09-26) became
| Decision | Deliverable | Result |
|---|---|---|
| 1. Drop the outcome-coded flags for good | G1 | **Done.** Name guard at import + data audit (item 32, also run at the start of every M2 backtest): any feature level with make rate ≥ 0.99 or ≤ 0.01 on ≥ 100 development shots fails. Shown to catch `fastbreak`, and the renamed `transition` (0.9997) and `putback` (0.9992). Rebuilding the context from play-by-play (misses included) stays on the backlog (owner, 2026-09-28). |
| 2. FT check on shares, not levels | G2 | **Done, check red (item 22).** Band flags 12 of 78 (allowed 7); team mean \|gap\| within 2 × noise in all 13 seasons; team r below its bar in all 13. Diagnosed, not a bug (see G2). |
| 3. Budget 60 min + cut runtime | G3, G4 | **Done.** Class A speed-ups only; 2,921 → 2,743 / 2,774 s under a busy machine (−12% like for like), every weeks 7–10 field byte-identical. Item 24 now always runs both backtests and prints all three results. |
| 4. Declared per-season-level variant | G5 | **Done, variant fails F-f.** Declared (1de203c) before any run; all its numbers post-hoc; clean evaluation pre-registered for the live 2026-27 season. |

## G2: the share statistics
`reports/free_throws.json` (`share_check`): 1,000 game-level bootstrap resamples, seed 20261001,
tolerance 2 × SE per statistic; band flags gated by the count rule (≤ 95th percentile of
Binomial(78, 0.0455) = 7).

- **Band flags (12):** 2013 deep3; 2014 long2, deep3; 2015 three; 2019 three; 2020 mid, three;
  2021 rim; 2022 rim, mid; 2023 rim, short.
- **Team shares:** mean |gap| within its bar in all 13 held-out seasons (e.g. 2017 0.00783 ≤
  0.00890); Pearson r below its 2-SE bar in all 13 (−0.421 in 2017 to 0.298 in 2012).
- **Why (checked by hand for 2017 and 2021):** expected team shares barely vary (sd 0.0025 vs
  actual 0.0075 in 2017), and FT points per game correlate *negatively* with FGA per game across
  teams (2017 r = −0.44): a shooting foul on a miss ends a possession without an FGA. The F-e
  model ("other trips per FGA × FGA") has no team foul-drawing term, so it cannot reproduce team
  shares. Band flags follow band and-one rates that move between seasons.
- **Limitation, not gated:** season-level mean gap of FT points per team-game, LOSO: +0.97
  (2011), −0.13, −0.95, −0.12, +0.20, −0.23, +0.89, −0.01, −0.10, −0.01, −1.28 (2021), +0.95
  (2022); validation −0.60. Between-season sd 0.66, so a ±0.1 level target was never reachable.
- **Next (owner's call):** a team-level FT model (trips per possession + a shrunk team
  foul-drawing rate), declared as a new variant. Not done here: changing the check after seeing
  the result would be tuning it to pass.

## G3: runtime
Full table, per-phase costs and big-O in `docs/models/m2_runtime.md`. Summary:

| Run | Code | Wall s |
|---|---|---|
| Weeks 7–10 final checklist (quiet machine) | 349acb9 | 3,072 |
| Before (sequential) | 8a58e9d | 2,921 |
| After 1 (LightGBM pool + G5 variants, 169 s) | b0b45e3 | 2,743 |
| After 2 (+ spline pool, LOSO reuse, zone codes once) | 2068902 | **2,774** |

All three G3 runs had a UI dev server using ~1.6 cores, so the times are pessimistic. Class A
only (number-preserving); class B ideas (e.g. dropping seeds 2–5's pair fits, −55%) are listed
with their statistical cost and **not** implemented. Final runtime 2,774 s < 3,600 s budget.
The pair fits of the nested isotonic calibration (O(S²) in development seasons, × 5 seeds) are
69% of the run.

## G5: season-level variants (post-hoc, not a clean hold-out)
Numbers after 3e20fa9 (offsets ignore games that tipped off < 3 h before, i.e. still in play):

| | CV log loss | Validation log loss | Validation ECE | Test ECE | F-f (val / test) |
|---|---|---|---|---|---|
| `lgbm` (reported M2) | 0.629699 | 0.633267 | 0.010438 | 0.012626 | fails / fails |
| `lgbm_level` | 0.629688 | 0.633323 | 0.010049 | 0.009980 | fails / fails |
| `spline` | 0.631577 | 0.635097 | 0.009219 | 0.011926 | — |
| `spline_level` | 0.631511 | 0.635116 | 0.009796 | 0.012512 | fails / fails |

(`spline_level` meets the ECE ≤ 0.010 part on validation, but a bin with ≥ 500 shots is
outside ±0.02, so F-f fails.)

- `lgbm_level − lgbm` validation log loss +0.000056, 95% CI [−0.000037, +0.000153]: no gain.
  k = 1,542–3,168 shots by development fold, 2,312 for validation/test (≈ 15 games until the
  season's own level gets half the weight).
- Calibration in the large improves: `lgbm_level` 0.9835–1.0134 over development (base
  0.9766–1.0198).
- G-g condition (meets F-f on validation **and** CV below `lgbm`) does not hold: team and player
  numbers stay on `lgbm`.
- Hypothesis: the offset fixes the league level, but F-f fails on a shape error by distance band
  (bins at P ≈ 0.40 and 0.69, deep threes and long twos), which one additive offset per season
  cannot fix.
- Leakage: three deliberate breaks (own tip-off group, prior trained on s, prior two seasons
  back) each turned the leakage tests red, and were reverted.
- **Pre-registered live rule** (`scripts/checks/m2_live_level.py`, run once after the 2026-27
  season): `lgbm_level` is calibrated on 2026-27 iff it meets F-f on every 2026-27 shot; it
  improves on `lgbm` iff the game-level bootstrap 95% CI of the log loss difference has upper
  bound < 0.

## Checklist
Fast run, 2026-09-28 19:08Z–19:16Z, HEAD 77a23cd (main), `SKIP="7 13 15 16 18 20 23 24 30 31"
bash scripts/checklist.sh`: **FAILS: 2** (22, 25; both expected). The run regenerated
`reports/possessions.json` and `reports/stints_mart.json` only because three new 2026 games
(E2026_8–10) have no cached box score locally; those two files were restored, not committed.

| # | Check | Result | Evidence |
|---|---|---|---|
| 1–6 | sync, ruff, format, mypy, pytest + coverage, vulture | PASS | fast run |
| 7 | web build from the fixture | not re-run | last PASS: week 7–10 final run (349acb9); site builds daily in CI |
| 8–12 | Elo reproduce, predictions append-only, stint sample, team_games, stints mart | PASS | fast run |
| 13 | M1 unit tests + runtime | not re-run | last PASS 349acb9; M1 code unchanged since |
| 14 | leakage tests | PASS | fast run |
| 15–16 | M1 reports, MLflow | not re-run | last PASS 349acb9 |
| 17 | live M1 dry run | PASS | fast run |
| 18 | build + score + publish twice | not re-run | last PASS 349acb9 |
| 19 | actionlint | PASS | fast run |
| 20 | screenshots | not re-run | last PASS 349acb9; site re-checked in the perf passes |
| 21 | F1 shots | PASS | fast run |
| 22 | F2 FT shares | **FAIL** (expected) | fast run: same 12 band flags, team r below bar in all seasons (G2) |
| 23 | F3/F4 runtimes | not re-run | recorded 2,774 s < 3,600 s (progress, After 2) |
| 24 | F5 report, two runs identical | not re-run | G3 runs: weeks 7–10 fields byte-identical each time; 3e20fa9 re-run changed only `level_variants` |
| 25 | F6 calibration in the large | **FAIL** (allowed) | fast run: worst 2013 0.9766; team/player reports equal the committed ones |
| 26–29 | F7 players, F8 charts, F9 MLflow + leakage, F10 card | PASS | fast run |
| 30 | no-dev build + predict | not re-run | daily CI runs predict every day |
| 31 | G5 level variants | not re-run | progress: fields, labels, leakage breaks; declaration 1de203c < run 327d0a3 |
| 32 | G1 outcome coding | PASS | fast run |

Item 22 is not on §6's list of items allowed to stay red; it stays red by the owner's decision
(2026-09-28) with the diagnosis above.

## Owner decisions, 2026-09-28
- Skip the single consecutive run; close from per-item evidence.
- F2: the season-level target is replaced by the share checks (already so since 2026-09-26);
  item 22 stays red with the diagnosis above.
- Outcome-coded flags: removal permanent; PBP rebuild on the backlog.
- Budget: 60 min (already so since 2026-09-26).
- xPTS used downstream with a visible season-level caveat; `lgbm` stays the reported M2;
  `lgbm_level` is judged on 2026-27 by the pre-registered rule.

## Open questions for the owner
1. Declare the team-level FT model (G2 next step), or leave item 22 red as a documented
   limitation?
2. Run the full §6 checklist once overnight for the formal record?
