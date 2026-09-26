# M2 backtest runtime: where the time goes, and what was cut

*Weeks 7–10b G3 (user decision 3, 2026-09-26: budget relaxed from 40 to 60 minutes for
`eurohoops backtest --model m2 --score-test`, 5 seeds, nested isotonic kept, all declared
variants including the G5 level variants; the Optuna study keeps its 2-hour budget).*
Measured from the backtest's `TIMING` lines (`eval/m2_backtest.py`, `Lap`). Machine: 6 cores /
12 logical; **every run below had a busy `astro dev` server from another worktree** (the user
chose to run anyway), so absolute times are pessimistic and seed-to-seed times vary by up to
~25%.

Notation: S = development seasons (12), n = development shots (385,604), T = trees (900),
F = features (9), K = LightGBM seeds (5), G = spline grid configurations (9), B = bootstrap
resamples (1,000), p = spline design columns (~30), I = IRLS iterations (~10).

## Before: phases of one run (HEAD 8a58e9d, sequential fits)

| Phase | Wall s | Share | Fits | Cost |
|---|---|---|---|---|
| Split + outcome-coding audit | 1 | 0.0% | 0 | O(n F) |
| Spline grid (9 configs × 12 LOSO) | 199 | 6.8% | 108 | O(G S · I n p²) |
| Spline chosen: LOSO 27, pairs 140, later 3 | 169 | 5.8% | 79 | O(S² · I n p²) |
| LightGBM, 5 seeds: LOSO ~76, pairs ~405, later ~8 each | 2,447 | 83.8% | 395 | O(K S² · T n F / threads) |
| Isotonic calibration (spline, seed mean) | 11 | 0.4% | — | O(S · n log n) |
| Per-seed isotonic calibration and scores | 28 | 1.0% | — | O(K S · n log n) |
| Variant scores (overall, bands, types, contexts) | 6 | 0.2% | — | O(n log n) |
| Gate on validation (bootstraps) | 13 | 0.4% | — | O(B n_val log n_val) |
| Gate on test (bootstraps) | 31 | 1.1% | — | O(B n_test log n_test) |
| xPTS table + report (data hash) | 4 | 0.1% | — | O(n) |
| Report + mart write, MLflow | 9 | 0.3% | — | — |
| **Total (wall, incl. start-up)** | **2,921** | | **582** | |

**What dominates and why.** The 66 leave-two-out LightGBM fits per seed are 69% of the whole
run (~2,027 s): the nested isotonic calibration needs, for every development season s, models
trained without s *and* each other season t, so its cost grows with S² (66 = 12·11/2 fits
against 12 for plain LOSO), and it is repeated for each of the K = 5 seeds. Each fit is a full
900-tree model on ~300k shots, O(T n F) in histogram work, about 6 s on 6 threads. Everything
that is not a model fit is under 4% of the run.

## Optimisations

### Class A (number-preserving; every weeks 7–10 field byte-identical, items 24–25)
| Change | Evidence it preserves numbers | Measured effect |
|---|---|---|
| LightGBM fits in 2 worker processes × 6 threads (`FitPool`) | §3 check: identical SHA-256 of every prediction, 1/2/3 processes; `test_parallel_fits_equal_sequential_fits`; after-run report byte-identical | seeds 486–493 s → 357 s on a less busy stretch, 448–459 s while `astro dev` competed; LightGBM 2,447 → 2,072 s (−15%) |
| Spline fits in the same pool | real data, 13 fits: identical SHA-256; 34.1 → 23.7 s | see "After 2" |
| Chosen spline reuses the grid's 12 LOSO fits (same fitter, same data) | same predictions by construction | −12 of 79 fits (~27 s sequential) |
| `zone_code` computed once per backtest, not on every `features()` call (0.040 of 0.049 s per call, ~1,100 calls); dropped before the data hash | same codes; report byte-identical | ~45 s of CPU |
| Refactor into helpers + phase timings | report byte-identical at 8a58e9d | — |

Evaluated, not worth changing: `pd.concat`/copies in `Folds` (the `iloc` row selection per fit
is < 0.1 s against ~6 s of training); a second process pool for the bootstraps (< 2%).

### Class B (would change numbers; **none implemented**, each needs a declared change first)
| Idea | Estimated saving | Statistical cost |
|---|---|---|
| Seeds 2–5 without their own pair fits; the seed mean calibrated with seed 1's pairs | ~1,600 s (−55%) | `lgbm_iso` calibrated on one seed's predictions but applied to the 5-seed mean (spread mismatch); per-seed isotonic numbers (F-l) for seeds 2–5 lost |
| Cross-fitting (3 groups of training seasons) instead of leave-two-out | ~900 s (−30%) | calibrators learned from models trained on ~8 instead of 10 seasons: noisier `_iso` variants |
| Fewer trees at a higher learning rate (e.g. 450 at 2×) | ~1,000 s (−35%) | a different model: every LightGBM number moves; the Optuna study (2 h) should be re-run |
| 1 process × 12 threads, or other thread counts | unknown, likely small (2 × 6 already fills 12 logical cores) | LightGBM sums histograms in another order: last-digit changes |
| Reuse one LightGBM `Dataset` across seeds | ~1 s per fit at most | bin boundaries depend on the seed (`data_random_seed`) |
| ECE bootstrap with games as weights, one sort (as the new G5 fields use) | ~35 s | tie and bin-edge handling differs from the committed CIs in the last digits |

Not recommended now: the budget is met without them, and each breaks the byte-for-byte
reproduction of the committed weeks 7–10 numbers.

## Runs
| Run | Code | What it includes | Wall s |
|---|---|---|---|
| Weeks 7–10 test run (quiet machine) | 6b2a5c1 | 4 variants | 3,130 |
| Weeks 7–10 final checklist (quiet machine) | 349acb9 | 4 variants | 3,072 |
| Before | 8a58e9d | 4 variants, sequential | 2,921 |
| After 1 | b0b45e3 | + LightGBM pool, + G5 level variants (169 s) | 2,743 |
| After 2 | 2068902 | + spline pool, LOSO reuse, zone codes once | **2,774** |

Like for like (without the new G5 phase), After 1 is 2,559 s against 2,909 s before (−12%).

After 2 per phase: spline grid + chosen **252 s** (was 368: −32%; the chosen spline needs 67
fits instead of 79); LightGBM seeds 393, 455, 467, 529, 405 s = 2,249 s (was 2,447). Its wall
time (2,774 s) is *higher* than After 1 although it does strictly less work: during After 2 the
UI dev server took ~1.6 cores the whole time (its CPU time grew by ~9,400 s in ~95 minutes;
the machine sat at 100%), and the LightGBM seeds slowed as the run went on (393 → 529 s). The
user could not stop it (working in that worktree). On a quiet machine the same code should
take roughly 2,000–2,200 s (seed 1 at 357–393 s × 5, spline ~250 s, the rest ~280 s incl. the
G5 variants); that is an estimate, not a measurement.

**Result:** under the 3,600 s budget on every run, with every weeks 7–10 number byte-identical
(checked after each class-A change: the report minus `level_variants` equals the committed
file byte for byte; items 24–25 in the final checklist run).
