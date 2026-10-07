# Weeks 14–16 (part 2) closeout: M7 season simulator

**Status: COMPLETE (exit gate FAIL, reported honestly).** K0–K9 are built, tested and committed. The full §7 checklist
ran top to bottom in one run with no edits in between (HEAD d4a55fd, 11:47Z–14:42Z on 2026-10-07, quiet machine):
every item PASS except 22 and 25 (the known M2 limitations). The exit gate "sim backtest calibrated" fails the K-h
point rule on validation (the point-strength baseline has a lower Brier), while calibration passes. Model card:
`docs/models/m7.md`. Progress log: `reports/m7_progress.md`. Full checklist output: `reports/m7_checklist_run.txt`.

## Exit gate (PLAN §8, row 14–16, M7 half; rule §0 K-h, EuroLeague validation 2023)
| Part | Status | Evidence |
|---|---|---|
| Strengths sampled from M1's posterior per simulation | built | `team_eff.rating_posteriors`; 90% synthetic coverage in 87–93% |
| Exact formats and tie-breaks | built | `rank` = official final table in all 8 scored EL seasons (`m7_facts.py`) |
| Rank distributions, playoff / play-in / Final Four / title odds | built | `sim/season.py` |
| Mid-season backtest, calibration vs baselines | built | 25/50/75% checkpoints, Brier, RPS, log loss, reliability, Spiegelhalter z |
| Brier below `standings_now` | PASS | 0.091951 vs 0.148148 |
| **Brier below `point_sim` (point rule)** | **FAIL** | 0.091951 vs 0.088277; diff +0.003675, 95% CI [−0.003868, +0.010623] |
| Pooled tuning + validation \|z\| < 1.96 | PASS | z = −0.870358 |
| **Gate** | **FAIL** | `backtest_m7.json:gate.passed = false` |

Why it failed: the baseline that won is `point_sim` (M1's point strengths, full noise), at every validation checkpoint
(25%: 0.127834 vs 0.122470; 50%: 0.101520 vs 0.096895; 75%: 0.046499 vs 0.045464). Reliability is fine (z −1.15 on
validation, −0.62 on test). On test the order flips (sim_full − point_sim −0.000939, CI [−0.007556, +0.004971]).
Strength sampling neither helps nor hurts the EuroLeague direct-cut Brier measurably at 54 + 114 team-checkpoints.
In the GBL it does fix the point-strength baseline's miscalibration (tuning z 2.771 → 0.924).

## Pre-registration (git order, `scripts/checks/m7_order.py`)
VERDICT 8b8fa8c (tuning only) < VALIDATION c65e91c < TEST d885fc8. Nothing was re-tuned after the verdict; the grid was
not widened (the best variant was not on the inflation grid's edge).

## Checklist (full run, HEAD d4a55fd)
| Items | Result |
|---|---|
| 1–21, 23, 24, 26–53 | PASS |
| 22 F2 free throws | FAIL (known M2 limitation: band flags 12 of 78, allowed 7) |
| 25 F6 calibration in the large | FAIL (known M2 limitation: 2018–2021 outside 0.5%) |

New items: 48 M7 units + facts (152 passed, facts PASS); 49 leakage (32 passed); 50 reports (two runs byte-identical,
equal to the committed files, GBL fixed choice reproduces, order yes); 51 runtimes (EL 238 s < 1,800; GBL 58 s < 600;
simulate 7 s < 120; recorded 231 / 53 / 7); 52 simulate (sums hold, nothing written with the failed gate, forced-gate
writes idempotent, existing logs untouched, `daily.yml` has no simulate step, actionlint clean); 53 card (6 passed).
Items 11, 12 and 18 rewrite four live-season reports from the worktree's data copy (`gbl_box_invariants.json`,
`live_scorecard*.json`, `stints_mart.json`); they were restored with `git checkout` after the run (daily-pipeline
outputs, not M7's).

## §3 findings
- **Formats** (table with sources in `docs/models/m7.md` and `sim/formats.py`): bylaws exist only for 2025-26 and
  2026-27 (`ftpserver.euroleague.net`, earlier years 404); earlier cut lines from the official final tables
  (`api-live.euroleague.net/v1/standings`, cached under `data/raw/euroleague/standings/`) and the playoff / play-in
  fields of the games feed. Top 8 straight to the playoffs through 2022-23, top 6 + play-in 7–10 from 2023-24.
- **`rank` vs official tables:** 8/8 exact (overtime games on the regulation score; 2022-23 PAN −2 wins).
- **2020-21:** every game played, no forfeit: kept in tuning.
- **GBL seasons used:** 2020-21, 2021-22, 2022-23 (tuning), 2025-26 (test). Excluded 2023-24 and 2024-25 (the
  quarterfinal field is not the table's top 8; cause not in any source we hold).

## Tuning (EuroLeague 2016, 2017, 2018, 2020, 2022; 252 team-checkpoints)
| model | Brier | RPS | log loss | z |
|---|---|---|---|---|
| **sim_full (chosen)** | 0.10932 | 0.078114 | 0.339104 | −0.42 |
| sim_net | 0.109624 | 0.078228 | 0.339803 | −0.34 |
| sim_inflate 1.5 | 0.110435 | 0.078799 | 0.342678 | −0.87 |
| sim_inflate 2 | 0.111336 | 0.079439 | 0.346574 | −1.28 |
| point_sim | 0.108517 | 0.078143 | 0.33472 | 1.38 |
| elo_sim | 0.103419 | 0.077422 | 0.317801 | 0.62 |
| standings_now | 0.174603 | 0.121102 | 1.608237 | — |

## Validation and test per checkpoint (Brier of the direct cut)
| split | checkpoint | sim_full | point_sim | elo_sim | standings_now |
|---|---|---|---|---|---|
| validation | 25% | 0.127834 | 0.12247 | 0.137887 | 0.222222 |
| validation | 50% | 0.10152 | 0.096895 | 0.118342 | 0.111111 |
| validation | 75% | 0.046499 | 0.045464 | 0.056256 | 0.111111 |
| test | 25% | 0.128772 | 0.131625 | 0.13069 | 0.210526 |
| test | 50% | 0.132624 | 0.135314 | 0.139335 | 0.263158 |
| test | 75% | 0.086374 | 0.083646 | 0.100498 | 0.157895 |

Bootstrap CIs (18 validation / 38 test team clusters): validation sim_full − point_sim +0.003675 [−0.003868, +0.010623],
− standings_now −0.056197 [−0.194373, +0.05435], − elo_sim −0.012211 [−0.033178, +0.005042]; test − point_sim −0.000939
[−0.007556, +0.004971], − standings_now −0.094603 [−0.170126, −0.033202], − elo_sim −0.007585 [−0.024088, +0.007343].
Title-odds Brier (reported): validation sim_full 1.126624 (point_sim 1.1884), test 0.887707 (point_sim 0.8782).
GBL (not gated): tuning Brier sim_full 0.103094 vs point_sim 0.108201 (z 0.924 vs 2.771); test 2025-26 sim_full 0.1751,
every model |z| > 1.96 on that single 13-team season.

## Live
`eurohoops simulate` is built (`live_sim.py`, item 52) and runs in 7 s (10,000 sims), but the gate failed, so per K-j it
writes nothing and is not in `daily.yml`: no simulation log or `sim_latest_*.json` exists. Dry run on the mart copy:
EuroLeague after round 1, GBL after round 0.

## Subagent log
| wave | subagent | deliverable | rounds | notes |
|---|---|---|---|---|
| 1 | A | K1 posterior | 2 | round 1 made the coverage test pass by enlarging the synthetic league after a failure (rejected); root cause σ² without a dof correction → D7; test back to its first design |
| 1 | B | K2 formats | 1 | — |
| 1 | C | K3 engine + `rank_by_wins` | 1 | 10k sims of a 20-team season in ~9 s |
| 2 | D | K4 harness | 1 | could not finish the full suite on the loaded machine; covered by the orchestrator's gates |
| 2b | E | K5 leakage | 1 | its `checkpoint_inputs` refactor re-verified output-identical by the orchestrator |

Fixed by the orchestrator: K0 (facts, played results), the vulture-whitelist merge conflicts, item 9 (merged
origin/main), checklist items 48–53, K6–K9 (real-data runs, live command, the unseen-team prior D11, card). Worktrees
were created by hand (`C:\Python\EH-m7-{a..e}`), not inside `C:\Python\Sports_Project`; all removed with their merged
branches. A's first prompt carried an unexpanded placeholder; the rules were sent before its first commit.

## Decisions this file did not cover, and open questions for the owner
- D1 checkpoint rounds ⌊f·R⌋; D2 duplicate fixture once; D3 deductions in the format; D4 Elo baseline through the
  engine; D5 knockout home order; D6 `rank_by_wins`; D7 σ² dof; D8 GBL pace-dependent noise at the mean pace; D9
  sim_inflate keeps sim_net's noise; D10 GBL splits; D11 unseen live teams at M1's prior (`reports/m7_progress.md`).
- **Open:** (1) Live M7 anyway as a shadow log (like M5), given calibration passed and test favours sim_full? K-j says
  no without the gate; owner's call. (2) A season-simulation page is out of scope until after week 16. (3) The
  promoted GBL clubs' wide prior before round 1 (D11): acceptable, or start them at a promoted-team mean? (4) GBL
  2023-24 / 2024-25 playoff fields: an ESAKE competition notice would let them be scored.

## memory.md entry
`memory.md` lives untracked in the main tree, which this run must not edit; the entry is in
`C:\Python\EH-m7\memory.md` (untracked) for the owner to append.
