# Weeks 9–12 closeout: M3 player impact

**Status: INCOMPLETE (stopped by the owner's decision, 2026-09-29, on the weekly usage limit).**
H0–H6 are built and merged, H5's verdict is committed and validation is scored once: **the exit
gate passes**. Not done yet, by the owner's priority order: test scoring (once), `m3_players.json`,
H8 GBL transfer, H9 model card, and the single consecutive run of the full §7 checklist. They
resume after the Oct 1 reset. Progress log: `reports/week9-12_progress.md` (iterations 1–13).

## Exit gate (PLAN §8, row 9–12)
| Part | Status | Evidence |
|---|---|---|
| Stints validated | EuroLeague done earlier (weeks 5–7); GBL built (H3) | `reports/stints_mart.json`; `reports/gbl_stints.json` (GBL: 56.1% of 2018–19 games pass all three checks; H-i 95% rule does not hold) |
| RAPM (+dummy) | built | `models/rapm.py`, `models/rapm_dummy.py`; both tuned |
| Bayesian RAPM | built | `models/rapm_posterior.py`; 90% intervals cover 88.38% over 200 synthetic seasons; not yet applied in a report (`m3_players.json` pending) |
| SPM prior | built | `models/spm.py`; `rapm_spm` is the chosen variant |
| GBL transfer | **not built** (H8 pending) | — |
| **Chosen RAPM variant beats box-only on validation** | **PASS** | `reports/backtest_m3.json` gate block |

## Tuning (2015–2022, 2,236 games) and the verdict
Verdict commit b09d19a (tuning-only report, `validation_scored: false`), before any validation
number existed.

| Model | Tuning RMSE | Chosen parameters |
|---|---|---|
| **rapm_spm** (chosen) | **11.723** | half-life 5,840 d, ridge 4,000, SPM α 1.0, k 250 min |
| rapm | 11.731 | half-life 5,840 d, ridge 2,000 (separate O/D not better) |
| rapm_dummy | 11.736 | 50 minutes (100: 11.747, 200: 11.793, 400: 11.852) |
| m1 (committed, not re-tuned) | 11.853 | — |
| box_only | 11.902 | half-life 182 d, k 250, ridge 300 |
| pir | 12.108 | — |
| b0 | 12.794 | — |

Both grids were widened once, before the verdict, after the first tuning run hit their edges
(iteration 7). RAPM's half-life ends at 5,840 days, the grid top, which is longer than the whole
fit window: effectively no decay.

## Validation (2023, 331 games, scored once; commit e58ae3e)
| Model | RMSE | MAE | Log loss |
|---|---|---|---|
| **rapm_spm** | **11.477** | 8.904 | 0.586 |
| m1 | 11.677 | 8.958 | 0.590 |
| box_only | 11.748 | 9.172 | 0.600 |
| pir | 11.837 | 9.239 | 0.604 |
| b0 | 12.392 | 9.683 | 0.652 |
| rapm_spm, oracle minutes (not a forecast) | 11.470 | 8.971 | 0.584 |

Game-level bootstrap, 1,000 resamples, seed 20261001, RMSE(rapm_spm) − RMSE(other):
- **box_only (gate): −0.271, 95% CI [−0.561, −0.002] → PASS** (narrowly).
- m1 (not gated): −0.200, 95% CI [−0.408, −0.005].
- pir (not gated): −0.360, 95% CI [−0.758, +0.028].

M3 also beats M1, the best model so far, on validation margin RMSE and log loss. Test
(2024–2025) is untouched.

## Checklist
Not run as one consecutive pass (deferred by the owner). Evidence so far:

| # | Check | Result |
|---|---|---|
| 2–6 | ruff, format, mypy, pytest + coverage, vulture | PASS (pre-push run: see progress, iteration 14) |
| 33 | M3 unit tests + §3 facts | parts PASS (unit tests in pytest; `m3_facts.py` held on real data in H0) |
| 34 | M3 leakage | leakage tests in pytest PASS; GBL leakage pending with H8 |
| 35 | report + gate, two runs identical, verdict < validation < test | order PASS (`m3_order.py`); two-run identity not re-run on real data (byte-identical on fixtures) |
| 36 | runtime < 1,800 s | PASS: 787 s (validation run), 784 s (tuning-only) |
| 37 | model card + GBL stints | GBL stints PASS (two builds identical); model card pending (H9) |
| 1, 7–32 | earlier phases | not re-run this phase |

## Subagent log
| Wave | Subagent | Delivered | Rounds | Orchestrator fixes |
|---|---|---|---|---|
| 1 | A | RAPM + harness (H1) | 1 | wired B's baselines; variant registry (tune + fit); M1 replayed from its 2007 warm-up; spell index truncated ids on real data (empty lineups); minutes shares > 1 on real data |
| 1 | B | box-only, PIR, GBL box lines (H2) | 1 | grid widened after edges (tuning only) |
| 1 | C | GBL stints (H3) | 1 | CLI registration |
| 1 | D | posterior (H4) | 1 | — |
| 2 | E | rapm_dummy (H6) | 2 (usage-limit resume) | merge conflict with F; one test widened |
| 2 | F | SPM + rapm_spm (H6) | 2 (usage-limit resume) | — |

## Decisions this file did not cover
D1–D7 in the progress file, plus: shares = 5 × seconds / team's recorded seconds, capped at 1
(real box quirks); grids widened once before the verdict; GBL possession check left at ±2 (not
loosened) with the diagnosis that GBL possession ends look under-detected.

## Open questions for the owner
1. After the reset: score test once, then H8 (GBL SPM transfer), `m3_players.json` and the model card?
2. GBL: back-fill 2020–2025 play-by-play (moves `reports/possessions.json`) and fix the GBL
   possession-end rules, to make GBL RAPM possible later?
3. The gate passed by a narrow margin (CI upper bound −0.002): treat the live 2026-27 season as
   the confirmation before any M3 goes live?

## memory.md entry
Added under 2026-09-29 (M3 weeks 9–12, INCOMPLETE).
