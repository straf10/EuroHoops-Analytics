# Weeks 14–16 closeout: M5 roster-aware game predictor

**Status: COMPLETE, with a split checklist record (owner's choice).** J0–J9 are built, tested and committed. The exit
gate passes on EuroLeague validation 2023 and the test seasons were scored once. On 2026-10-05 the owner chose to run
only the fast gate (items 1–19 and 43–47) during this session, because another project's training job held about 40%
of the CPU. The owner runs the full pass (items 1–47) later on a quiet machine. Results are in "Checklist" below.
Model card: `docs/models/m5.md`. Progress log: `reports/week14-16_progress.md` (iterations 1–16).

## Exit gate (PLAN §8, row 14–16, M5 half)
| Part | Status | Evidence |
|---|---|---|
| Roster-aware predictor (projected minutes × player rating + team residual + home + rest option) | built | `models/rotation.py`, `models/rest.py`, `models/m5.py`, `eval/m5_backtest.py` |
| Projected vs oracle rosters | both scored | oracle reported only (`oracle`, `gap` blocks) |
| Blend with M1 and Elo; recalibration | built, declared, not chosen on tuning | `blend` tuning LL 0.602856; Platt 0.600889 vs 0.599344 without |
| Outputs: P(win), spread, total with distributions, over/under for any line | built | Student-t margin, Normal total; `m5.p_over`, `m5.p_cover` |
| **M5 beats M1 on validation (J-h point rule)** | **PASS** | LL 0.583765 vs 0.587714, diff −0.003948, 95% CI [−0.014946, +0.006291] |

M7 (season simulator), the other half of PLAN's row 14–16, was out of scope for this file and gets its own prompt.

## Pre-registration (git order, `scripts/checks/m5_order.py`)
VERDICT 5baa7c1 (tuning only) < VALIDATION cf84a68 < TEST 2401d15.

Before the verdict the grid was widened once (6b428a0), because the first best values were on the lower edges.
The tie-break was also corrected (D10, 8ad605e) on tuning numbers only. No validation number existed before
cf84a68.

## Tuning (EuroLeague 2015–2022, 2,236 games, 414 candidates)
| Model | Tuning log loss |
|---|---|
| **M5 chosen: proj_hc \| core \| half-life 180 d \| ridge 40** | **0.599344** |
| best core (proj_avail@1, within the tie tolerance) | 0.599292 |
| best core_rest | 0.599985 |
| best blend | 0.602856 |
| chosen + Platt | 0.600889 |
| M1 (committed, replayed) | 0.60602 |
| Elo (comparison) | 0.609009 |
| oracle minutes | 0.600601 |

Totals: M1 + rest (CRPS 9.278146) chosen over M1's total (9.300494). Margin distribution: Student-t, scale 10.968172,
df 15. The decayed and availability projections, rest, the blend and Platt all failed to beat the simplest model
by more than the tie tolerance on tuning.

## Validation (2023, 331 games) and test (2024–2025, 732 games), each scored once
| Model | Val LL | Val RMSE | Val totals CRPS | Test LL | Test RMSE | Test totals CRPS |
|---|---|---|---|---|---|---|
| **M5** | **0.5838** | **11.547** | **8.97** | 0.6232 | 11.697 | **9.48** |
| M1 | 0.5877 | 11.677 | 9.05 | 0.6232 | **11.656** | 9.53 |
| Elo | 0.5885 | 11.660 | 10.06 | 0.6246 | 11.680 | 10.39 |
| B0 | 0.6518 | 12.393 | 10.06 | 0.6599 | 12.357 | 10.39 |
| oracle (not a forecast) | 0.5803 | 11.529 | 8.97 | 0.6201 | 11.685 | 9.48 |

- **Test, M5 − M1:** LL −0.000015, 95% CI [−0.009546, +0.009753]. **The validation edge does not carry to test**,
  the same pattern as M3, whose ratings M5 uses.
- **What does carry is totals:** M5 − Elo totals CRPS is −1.093 on validation and −0.914 on test, with CI
  [−1.220, −0.577] on test.
- **Segments (reported):**
  - PAO/OLY test games (163): M5 0.583 vs M1 0.593;
  - short-rest test games (193): 0.597 vs 0.613;
  - Greek clubs right after a game in the other competition (22): 0.623 vs 0.653.

## Value of an injury feed (J8)
- **On average:** projected − oracle log loss is +0.0035 on validation and +0.0031 on test, and both CIs include 0.
- **Where it counts:** in games where a top-3-minutes player was missing from the projection (validation 20, test
  31), the margin RMSE gap is +0.79 [0.07, 1.54] and +0.84 [0.08, 1.65].
- **On tuning, oracle is worse than projected** (0.6006 vs 0.5993; M3 had the same): actual minutes carry blowout
  garbage time.
- `tests/test_m5_gap.py` recomputes the gap from the committed per-game predictions.

## Rest (`reports/m5_rest.json`)
- **EuroLeague:** M5's margin residual regressed on home − away rest differences (OLS, HC0) gives no feature with a
  90% interval excluding 0, in any split.
- **GBL:** short rest is +5.55 [1.16, 9.95] on tuning but −3.04 on validation.
- **Conclusion:** no stable rest effect beyond M5, which matches tuning preferring `core` to `core_rest`.

## GBL (reported, not gated, J-g)
The EuroLeague verdict was scored as a fixed choice (`chosen.fixed`), with GBL box-only player ratings.
- **Validation:** M5 0.4748 vs M1 0.4298 (+0.0450 [+0.0108, +0.0834]).
- **Test:** M5 0.4790 vs M1 0.4835 (−0.0046 [−0.0259, +0.0165]); totals CRPS vs M1 −0.207 [−0.333, −0.077].

GBL box-score ratings are too weak for the player part, as M3's transfer already showed.

## Checklist
The full output is in `reports/week14-16_checklist_run.txt`. This is a fast gate by the owner's decision, not one
uninterrupted full pass:
- **Pass A** (HEAD 2ad21ca) ran items 1–19 and 43–47.
- **Pass B** (same code plus the check-script fix) ran items 45–47.
- Items 20–42 (site screenshots, M2, M3, M4) were skipped. They touch none of M5's code paths, and the owner runs
  them in the full pass.

| # | Check | Result |
|---|---|---|
| 1–12, 14–19 | setup, lint, types, tests + coverage, vulture, web build, Elo, logs, stints, leakage, M1 reports, MLflow, live dry run, build/publish, actionlint | PASS |
| 13 | M1 unit tests + EuroLeague backtest runtime | **FAIL (load)**: 1,204 s vs the 600 s limit. Earlier quiet runs took 111–278 s, and M1 code is byte-identical to `main`. Recheck in the full pass |
| 20–42 | site screenshots, M2, M3, M4 | SKIPPED (owner) |
| 43 | M5 unit tests (75), §3 facts, `minutes.py` unchanged | PASS |
| 44 | M5 leakage suite (44 tests, planted leaks detected) | PASS |
| 45 | M5 reports: two runs byte-identical and equal to the committed ones; GBL fixed choice reproduces; gate block; order | PASS after two record fixes, below |
| 46 | runtime: EuroLeague 840 s (< 1,800), GBL 249 s (< 600), measured under the other job's load | PASS (pass B) |
| 47 | `m5.md` numbers = reports; CONTEXT terms | PASS |

Item 45 needed two fixes:
- **Pass A:** items 45 and 46 failed because the new check scripts passed MLflow the Git Bash `/tmp` scratch path,
  which Windows Python rejects. Fixed with `cygpath -m`, and the same fix was applied to M3's two scripts.
- **Pass B:** the reports reproduced, but the order line failed because the progress file lacked `TEST 2401d15`. The
  line was added and `m5_order.py` rerun alone: `verdict < validation < test: yes`.

## Subagent log
| Wave | Subagent | Delivered | Rounds | Orchestrator fixes |
|---|---|---|---|---|
| 1 | A | J1 rest features (`models/rest.py`) | 1 | — |
| 1 | B | J2 share variants (`models/rotation.py`) | 1 | vulture whitelist merge conflicts (all three wave-1 branches) |
| 1 | C | J3 model core (`models/m5.py`) | 1 | — |
| 2 | D | J4 harness, synthetic league, order check | 1 | CLI + MLflow wiring; inputs cut at the last test season; fixed-choice path for the GBL (J-g); "not gated" label; tie-break D10 |
| 2 | E | J5 leakage suite | 1 | E found a real leak (tuning-only report carried Platt coefficients fitted on validation/test outcomes); fixed in the harness (D8); D's test that encoded the leak and E's now-redundant isolation test updated/removed |

Two worktrees branched from `main` instead of `week-14-16`. Their diffs touched only their own files, and later
prompts told the subagent to merge `week-14-16` first.

## Decisions this file did not cover
D1–D10 are in the progress file:
- D1: `b2b` → `short_rest` (≤ 2.5 d), because no game is ≤ 1.5 d after the previous one.
- D2: shares convention.
- D3: player part = `rapm_margins` unchanged.
- D4: residual and rest in one ridge.
- D5: probit blend.
- D6: distribution fit in-sample on tuning, which sets the leakage scope.
- D7: wave-2 order.
- D8: the Platt leak fix.
- D9: the leakage suite keeps the cutoff-defining game.
- D10: tie-break by simplicity, then loss.

Also:
- M5 inputs stop at the last test season, so the daily log cannot move `data_sha256`.
- `M5_REST_REPORT` and the `m5-rest` research command were added.
- The checklist's unrelated report rewrites (items 10–16 regenerate `possessions.json`, `stints_mart.json`,
  scorecards from today's local data) were restored, not committed.

## Open questions for the owner
1. **Live M5 (J-j):** M5 ties M1 on test, and its only robust gain is totals (over Elo, and over M1 in the GBL).
   Options:
   - keep M5 research-only (the M3 precedent);
   - log M5 totals only as a shadow column;
   - run a full shadow `live_m5.py`.
2. **Availability:** the oracle gap says an injury/lineup feed is worth about 0.8 points of RMSE in the games it
   matters. Is a manual "out" list (owner-maintained CSV, pre-registered) worth building for 2026-27?
3. **Full checklist pass:** run items 1–47 on a quiet machine (`bash scripts/checklist.sh`, Git Bash). Item 13
   (M1 runtime) failed today only under load (1,204 s; earlier quiet runs 111–278 s, and M1 code is unchanged).
4. Done (owner, 2026-10-05): M3's `m3_reports.sh` and `m3_runtime.sh` now build the MLflow URI with
   `cygpath -m` as M5's do. M1's scripts use the default tracking store and were not affected.

## memory.md entry
Added under 2026-10-05 (M5 weeks 14–16, COMPLETE with a split checklist record).
