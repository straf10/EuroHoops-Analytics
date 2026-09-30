# Weeks 9–12 closeout: M3 player impact

**Status: COMPLETE (2026-09-30).** H0–H9 are built, tested and committed. The exit gate passes
on validation, the test seasons are scored once, and the checklist passes except items 22 and 25,
the two allowed reds. The run was not one uninterrupted pass: see "Checklist". Model card:
`docs/models/m3.md`. Progress log: `reports/week9-12_progress.md` (iterations 1–18).

## Exit gate (PLAN §8, row 9–12)
| Part | Status | Evidence |
|---|---|---|
| Stints validated | EuroLeague (weeks 5–7); GBL built (H3) | `reports/stints_mart.json`; `reports/gbl_stints.json`: 56.1% of GBL 2018–19 games pass all three checks, below the H-i 95% rule |
| RAPM (+dummy) | built, tuned | `models/rapm.py`, `models/rapm_dummy.py` |
| Bayesian RAPM | built, applied | `models/rapm_posterior.py` (90% intervals cover 88.38% on 200 synthetic seasons); `reports/m3_players.json` |
| SPM prior | built, chosen | `models/spm.py`; `rapm_spm` is the chosen variant |
| GBL transfer | built (H8) | `reports/backtest_m3_gbl.json`: EuroLeague models lose to GBL box-only and M1 |
| **Chosen RAPM variant beats box-only on validation** | **PASS** | RMSE diff −0.271, 95% CI [−0.561, −0.002] |

## Pre-registration (git order, `scripts/checks/m3_order.py`)
VERDICT b09d19a (tuning only) < VALIDATION e58ae3e < TEST 33ec255. Every sha cited on this
branch was taken after its commit's attribution trailer was stripped, and later rewrites kept
them (only commits after them changed).

## Tuning (2015–2022, 2,236 games)
| Model | Tuning RMSE | Chosen parameters |
|---|---|---|
| **rapm_spm** (chosen) | **11.723** | half-life 5,840 d, ridge 4,000, SPM α 1.0, k 250 min |
| rapm | 11.731 | half-life 5,840 d, ridge 2,000 |
| rapm_dummy | 11.736 | 50 minutes |
| m1 (committed, not re-tuned) | 11.853 | — |
| box_only | 11.902 | half-life 182 d, k 250, ridge 300 |
| pir | 12.108 | — |
| b0 | 12.794 | — |

## Validation (2023, 331 games) and test (2024–2025, 732 games), each scored once
| Model | Validation RMSE | Validation log loss | Test RMSE | Test MAE | Test log loss |
|---|---|---|---|---|---|
| **rapm_spm** | **11.477** | **0.586** | 11.678 | 9.134 | **0.6231** |
| m1 | 11.677 | 0.590 | **11.656** | **9.093** | 0.6232 |
| box_only | 11.748 | 0.600 | 11.680 | 9.173 | 0.627 |
| pir | 11.837 | 0.604 | 11.757 | 9.257 | 0.634 |
| b0 | 12.392 | 0.652 | 12.359 | 9.674 | 0.660 |

**The validation edge does not carry to test.** On test `rapm_spm` ties box-only, M1 has the
lowest RMSE and MAE, and log loss is a tie. The gate is decided on validation and stays PASS;
the test result is reported as is and is the main caveat on M3.

## GBL transfer (H8; GBL evidence, not the gate)
EuroLeague season SPM models (each used only at GBL cutoffs at or after its own fit time) applied
to GBL box rates, against GBL-native models. Games: tuning 336, validation 163, test 340, none
dropped; the same games per season as `reports/backtest_m1_gbl.json`.

| Model | Validation RMSE | Test RMSE | Test log loss |
|---|---|---|---|
| spm_transfer | 14.454 | 13.304 | 0.566 |
| spm_transfer_scaled (scale 2.43) | 12.714 | 12.155 | 0.495 |
| box_only (GBL) | 12.715 | 11.926 | 0.485 |
| m1 (GBL) | **11.734** | **11.919** | **0.4845** |

Test RMSE, spm_transfer minus box_only: +1.377, 95% CI [0.727, 2.010]. EuroLeague player models do
not beat GBL-native ones in the GBL.

## Player ratings (H7)
`reports/m3_players.json`: 4,577 player-seasons, 1,727 players (2011–2025), `rapm_spm` at each
season's last round cutoff with O, D, O + D, posterior sds and 90% intervals. Median sd of O + D
2.20–2.55 per 100 possessions; Spearman(sd, minutes) −0.753. Posterior mean equals the
walk-forward rating (tests); the EuroLeague backtest re-run with the new hook was byte-identical
to the committed report.

## Checklist
Full output: `reports/week9-12_checklist_run.txt` (pasted below). The owner accepted this record
instead of one uninterrupted pass:
- **Pass A (items 1–23, HEAD 5afcfc0):** item 20 failed on a stale selector: since the site
  redesign the scorecard shows M1 as a line (`p.note.m1`), not a table row. Fixed in
  `scripts/checks/screenshots.py` (used by item 20 only) and item 20 re-run alone: PASS. The
  owner stopped the pass during item 24.
- **Pass B (items 24–37, HEAD df35abd):** item 24's two M2 runs were byte-identical, equal to the
  committed report and within budget (2,078 s, 1,945 s), but its order check failed: the M2
  commits predate the trailer-stripping history rewrite and live under tag `archive/pre-rewrite`,
  not in HEAD's history. `scripts/checks/m2_order.py` now accepts that tag; re-run alone: PASS.
  The M2 runs were not repeated (the fix reads git only).

| # | Check | Result |
|---|---|---|
| 1–19 | setup, lint, types, tests + coverage, vulture, web, Elo, logs, stints, M1, MLflow, live dry run, build/publish, actionlint | PASS |
| 20 | screenshots 1440/390, light/dark | PASS after the check fix (0 px overflow, M1 line visible) |
| 21 | F1 shots | PASS |
| 22 | F2 FT shares | **FAIL (allowed)** |
| 23 | F3/F4 models | PASS |
| 24 | F5 M2 report, order, two runs identical | PASS after the order-check fix |
| 25 | F6 calibration in the large | **FAIL (allowed)** |
| 26–32 | M2 players, charts, MLflow, card, no-dev build, level variants, feature audit | PASS |
| 33 | M3 unit tests, §3 facts | PASS |
| 34 | M3 leakage (RAPM, dummy, SPM, box-only, minutes, GBL) | PASS |
| 35 | `backtest_m3.json` + gate, two runs identical, verdict < validation < test | PASS |
| 36 | M3 runtime < 1,800 s | PASS (875 s) |
| 37 | M3 card numbers = reports; GBL stints | PASS |

## Subagent log
| Wave | Subagent | Delivered | Rounds | Orchestrator fixes |
|---|---|---|---|---|
| 1 | A | RAPM + harness (H1) | 1 | wired B's baselines; variant registry; M1 from its 2007 warm-up; spell ids; minutes shares > 1 |
| 1 | B | box-only, PIR, GBL box lines (H2) | 1 | grid widened after edges (tuning only) |
| 1 | C | GBL stints (H3) | 1 | CLI registration |
| 1 | D | posterior (H4) | 1 | — |
| 2 | E | rapm_dummy (H6) | 2 (usage-limit resume) | merge conflict with F; one test widened |
| 2 | F | SPM + rapm_spm (H6) | 2 (usage-limit resume) | — |
| 3 | S1 | GBL transfer (H8) | 1 | docstring and report key; leak tests now assert a finite prediction first |
| 3 | S2 | player report (H7) | 1 | `team_games` read for both competitions made the data hash mismatch; now EuroLeague only |

## Decisions this file did not cover
D1–D7 in the progress file, plus: shares = 5 × seconds / team seconds, capped at 1; grids widened
once before the verdict; GBL possession check left at ±2; the 2019 player snapshot is the
scheduled (cancelled) round 34, so it holds every 2019-20 game played; no rebuild of the marts
for this phase (new 2026-27 games would change `data_sha256`; M3's frame ignores them, checked:
`m3-players` still matches the backtest hash); the branch was fast-forwarded, not rebased, so the
cited shas stay valid.

## Open questions for the owner
1. M3 did not beat M1 on test and passed the gate narrowly: keep it research-only, or run it live
   as a shadow model in 2026-27 (`live_m3.py`) as the confirmation?
2. GBL: back-fill 2020–2025 play-by-play and fix the GBL possession-end rules, to make GBL RAPM
   possible later?
3. Two commits on `main` from PRs #1 and #2 (f3594c2, 91908a8) still carry `Co-authored-by:
   Cursor`; left as they are by the owner's decision (2026-09-30).

## memory.md entry
Added under 2026-09-30 (M3 weeks 9–12, COMPLETE).

## Checklist run (verbatim)

~~~text
Weeks 9-12 checklist run, 2026-09-30 (scripts/checklist.sh, Git Bash on PATH; times UTC)

Not one uninterrupted pass. The owner accepted this record instead:
- Pass A ran items 1-23; item 20 failed on a stale selector (the redesigned scorecard shows M1
  as a line, not a table row). The owner stopped the pass during item 24.
- Item 20 was re-run alone after the fix to scripts/checks/screenshots.py (the only file it
  changed, used by no other item).
- Pass B ran items 24-37 (SKIP = 1-23). Item 24's two M2 runs were byte-identical, equal to the
  committed report and within budget; its order check failed only because the M2 commits predate
  the trailer-stripping history rewrite and live under tag archive/pre-rewrite. The check was
  fixed to accept that tag and re-run alone (it reads git only; the M2 runs were not repeated).
Result: every item passes except 22 and 25, the two allowed reds.

=== Pass A: items 1-23 (HEAD 5afcfc0, base origin/main, scratch C:\Users\nikos\AppData\Local\Temp\checklist_scratch, start 09:05Z) ===
===== 1 uv sync --frozen =====
Checked 137 packages in 13ms
PASS

===== 2 ruff check =====
[1;32mAll checks passed![0m
PASS

===== 3 ruff format --check =====
178 files already formatted
PASS

===== 4 mypy src =====
Success: no issues found in 65 source files
PASS

===== 5 pytest + coverage =====
36 files skipped due to complete coverage.
Required test coverage of 85% reached. Total coverage: 94.19%
570 passed, 1 warning in 210.79s (0:03:30)
PASS

===== 6 vulture =====
PASS

===== 7 web build from the fixture (copy of web/) =====
npm notice
12:09:55 [build] 2268 page(s) built in 23.60s
12:09:55 [build] Complete!
built 8 entries into site/
PASS

===== 8 Elo backtests reproduce the committed reports =====
5 passed in 0.07s
PASS

===== 9 predictions append-only vs origin/main, Elo model_version unchanged =====
removed/changed lines vs origin/main: 0
reports/backtest_elo.json base=0.2.0+425e6393 now=0.2.0+425e6393
reports/backtest_elo_gbl.json base=0.2.0+df05260c now=0.2.0+df05260c
PASS

===== 10 stint sample reproduces origin/main =====
50 games: all checks 100% (five_on_court 100%, seconds 100%, minutes 100%, points 100%); wrote reports\stint_validation.json
identical
PASS

===== 11 team_games coverage, points, possessions =====
12 games missing, 0 points mismatches; EuroLeague sample within 2: 94.0% of teams; wrote reports\possessions.json
rated games: 6390 with rows: 6378 missing (listed with reasons): 12
points = games scores for every row
EL sample: 50 games, within ±2: 94.0% of team-games, 88.0% of games; mean gap 0.024; FT weight matching PBP 0.4213
PASS

===== 12 stints mart thresholds, two builds identical =====
4246 games, 141625 stints; pass rate 2011-14 95.9%, 2015+ 98.9%; wrote reports\stints_mart.json
stints 4d6c6b5412e607b5ca2b4a66ef235076e700f3eb8c2c1a1d0ff07e4cfd87322c
stint_game_checks 061f7825cb225cba3a9dbe80c2005e8ca9d6558b950cafbb02cf64bc051fec6c
two builds: stored tables identical
two builds: report byte-identical
2011-14 0.9587 (>= 0.95); 2015+ 0.9894 (>= 0.985)
passing games where stint points miss the final: []
PASS

===== 13 M1 unit tests, EuroLeague backtest runtime =====
28 passed in 1.83s
full EuroLeague M1 backtest (--score-test): 278 s (limit 600 s)
PASS

===== 14 leakage tests =====
26 passed in 4.73s
PASS

===== 15 M1 reports complete and reproducible =====
reports/backtest_m1.json: two runs byte-identical
reports/backtest_m1.json: equals the committed report
  gate student_t_const passed -0.000796 [-0.007725, 0.007076]
reports/backtest_m1_gbl.json: two runs byte-identical
reports/backtest_m1_gbl.json: equals the committed report
  gate student_t_pace passed -0.006488 [-0.036953, 0.020842]
PASS

===== 16 MLflow parent + children; no-dev run skips tracking =====
mlruns/ is gitignored
euroleague: parent 5cb30ab5fd1940ae98d996d62e10b257 (euroleague 0.2.0+m1.cbd9aaa3): 30 params, 831 metrics, data_sha256 5491ba42151e…, commit 5afcfc00f1 dirty=true
  child 1bb365235de84b45b73b36c2603606ad normal_const: 35 params, 36 metrics, validation log loss 0.589663
  child 63a9e209754b4a71b90b9df55ca70416 normal_pace: 35 params, 37 metrics, validation log loss 0.589762
  child a7b44fd106cb4f509d99bf722ceed3ac student_t_const: 35 params, 37 metrics, validation log loss 0.587714
  child 35b78c25ae574101ab86cebc0ab69fd4 student_t_pace: 35 params, 38 metrics, validation log loss 0.587792
gbl: parent edac7177bbcd4ff7b95af0a210d828c5 (gbl 0.2.0+m1.51ab8739): 30 params, 818 metrics, data_sha256 ab1e0fe4cfcf…, commit 5afcfc00f1 dirty=true
  child 0924fb3c25a444978994d2fd8d05f706 normal_const: 35 params, 36 metrics, validation log loss 0.431306
  child 8e31f66f6c8f4b14b415bceab889d086 normal_pace: 35 params, 37 metrics, validation log loss 0.430736
  child d39d70dfeafb46bfa75e856a36a304a6 student_t_const: 35 params, 37 metrics, validation log loss 0.430290
  child f8eb790b9e174f7c8bc1299648f2a5ff student_t_pace: 35 params, 38 metrics, validation log loss 0.429759
no-dev env: mlflow not installed
WARNING eurohoops.eval.tracking: MLflow is not installed (dev dependency): backtest not tracked
no-dev backtest: same report, tracking skipped with a warning
PASS

===== 17 live M1 dry run =====
euroleague: clock 2026-09-29T06:00Z, first run 8 M1 rows, second run 0; log unchanged by run 2: True; Elo log byte-identical: True
    game_id,season,round,phase,tipoff_utc,home,away,p_home,exp_margin,exp_total,margin_sigma,margin_df,total_sigma,model,model_version,predicted_at_utc
    E2026_13,2026,2,RS,2026-09-29T16:00:00Z,DUB,BAR,0.5948,2.59,168.47,10.4024,7,16.5309,m1,0.2.0+m1.cbd9aaa3,2026-09-29T06:00:00Z
    E2026_11,2026,2,RS,2026-09-29T17:00:00Z,IST,MAD,0.4163,-2.28,164.42,10.4024,7,16.5309,m1,0.2.0+m1.cbd9aaa3,2026-09-29T06:00:00Z
gbl: clock 2026-09-16T23:15Z, first run 1 M1 rows, second run 0; log unchanged by run 2: True; Elo log byte-identical: True
    game_id,season,round,phase,tipoff_utc,home,away,p_home,exp_margin,exp_total,margin_sigma,margin_df,total_sigma,model,model_version,predicted_at_utc
    GBL2026_3690EC22,2026,26,RS,2026-09-17T09:15:00Z,BB4B460F,2A25C696,0.5650,1.93,170.98,11.6551,20,16.9827,m1,0.2.0+m1.51ab8739,2026-09-16T23:15:00Z
PASS

===== 18 build + score + publish twice leaves the tree unchanged =====
unchanged on the second run
PASS

===== 19 actionlint =====
bash.exe :    Building actionlint-py==1.7.12.25
At C:\Users\nikos\AppData\Local\Temp\ps-script-c6edda10-928e-40cc-917d-44717742b129.ps1:106 char:294
+ ... s=Get-Date; & "C:\Program Files\Git\bin\bash.exe" scripts/checklist.s ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : NotSpecified: (   Building actionlint-py==1.7.12.25:String) [], RemoteException
    + FullyQualifiedErrorId : NativeCommandError

      Built actionlint-py==1.7.12.25
Installed 1 package in 272ms
clean
PASS

===== 20 screenshots 1440/390, light/dark =====
Downloading playwright (36.8MiB)
 Downloaded playwright
Installed 4 packages in 105ms
light 1440: horizontal overflow 0 px; M1 row visible: False
light 390: horizontal overflow 0 px; M1 row visible: False
dark 1440: horizontal overflow 0 px; M1 row visible: False
dark 390: horizontal overflow 0 px; M1 row visible: False
FAIL

===== 21 F1 shot reconciliation, exclusion shares, two builds identical =====
612424 shots, 1795 excluded; feed = box for 99.99% of 2011+ team-games; wrote reports\shots.json
shots f02e93d50418aaee9c934d8d73d0f0dc9d650ea885d5b5e01d20b7cedffe19ee
shots_excluded 1d5a696def51feda1dd7bfeb26ed20947311642d5930546a3f0509aa6937c089
two builds: stored tables identical
two builds: report byte-identical
feed = box for 0.999882 of 2011+ team-games (>= 0.99)
largest validated-season exclusion share 0.005271 in 2011 (<= 0.01)
  2011 0.005271 {'unparseable': 0.000268, 'zero_coordinates': 0.003708, 'label_geometry': 0.001295}
  2012 0.003819 {'unparseable': 0.000199, 'zero_coordinates': 0.002524, 'label_geometry': 0.001096}
  2013 0.004099 {'unparseable': 6.6e-05, 'zero_coordinates': 0.002558, 'label_geometry': 0.001476}
  2014 0.003573 {'unparseable': 0.000193, 'zero_coordinates': 0.002382, 'label_geometry': 0.000998}
  2015 0.003256 {'unparseable': 0.000164, 'zero_coordinates': 0.002039, 'label_geometry': 0.001053}
  2016 0.003978 {'unparseable': 0.000125, 'zero_coordinates': 0.002443, 'label_geometry': 0.001409}
  2017 0.002604 {'unparseable': 6.3e-05, 'zero_coordinates': 0.000973, 'label_geometry': 0.001569}
  2018 0.00242 {'unparseable': 6.3e-05, 'zero_coordinates': 0.001383, 'label_geometry': 0.000974}
  2019 0.001864 {'unparseable': 0.000161, 'zero_coordinates': 0.000482, 'label_geometry': 0.001222}
  2020 0.002027 {'unparseable': 0.000177, 'zero_coordinates': 0.000532, 'label_geometry': 0.001318}
  2021 0.002542 {'unparseable': 0.000166, 'zero_coordinates': 0.000718, 'label_geometry': 0.001658}
  2022 0.002729 {'unparseable': 0.0002, 'zero_coordinates': 0.000726, 'label_geometry': 0.001803}
  2023 0.003218 {'unparseable': 0.00029, 'zero_coordinates': 0.000774, 'label_geometry': 0.002153}
  2024 0.001974 {'unparseable': 0.000217, 'zero_coordinates': 0.000674, 'label_geometry': 0.001083}
  2025 0.001585 {'unparseable': 9.7e-05, 'zero_coordinates': 0.00087, 'label_geometry': 0.000618}
PASS

===== 22 F2 FT reconciliation per season (LOSO rates) =====
2011 development: teams FAIL, bands outside 2 SE: none (team mean |gap| 0.00593 <= 0.00848: True; r 0.136 >= 0.279: False) | season level, not gated: FT points 14.23, expected 13.26, gap +0.971
2012 development: teams FAIL, bands outside 2 SE: none (team mean |gap| 0.00379 <= 0.00833: True; r 0.298 >= 0.364: False) | season level, not gated: FT points 13.22, expected 13.35, gap -0.129
2013 development: teams FAIL, bands outside 2 SE: ['deep3'] (team mean |gap| 0.00573 <= 0.00822: True; r -0.186 >= 0.312: False) | season level, not gated: FT points 12.59, expected 13.54, gap -0.954
2014 development: teams FAIL, bands outside 2 SE: ['long2', 'deep3'] (team mean |gap| 0.00402 <= 0.00901: True; r 0.148 >= 0.301: False) | season level, not gated: FT points 13.69, expected 13.81, gap -0.124
2015 development: teams FAIL, bands outside 2 SE: ['three'] (team mean |gap| 0.00463 <= 0.00836: True; r 0.228 >= 0.317: False) | season level, not gated: FT points 13.71, expected 13.50, gap +0.203
2016 development: teams FAIL, bands outside 2 SE: none (team mean |gap| 0.00599 <= 0.00901: True; r 0.203 >= 0.278: False) | season level, not gated: FT points 13.51, expected 13.74, gap -0.233
2017 development: teams FAIL, bands outside 2 SE: none (team mean |gap| 0.00783 <= 0.00890: True; r -0.421 >= 0.256: False) | season level, not gated: FT points 14.51, expected 13.62, gap +0.889
2018 development: teams FAIL, bands outside 2 SE: none (team mean |gap| 0.00808 <= 0.00913: True; r -0.211 >= 0.288: False) | season level, not gated: FT points 13.70, expected 13.71, gap -0.011
2019 development: teams FAIL, bands outside 2 SE: ['three'] (team mean |gap| 0.00398 <= 0.00816: True; r 0.253 >= 0.341: False) | season level, not gated: FT points 13.69, expected 13.79, gap -0.100
2020 development: teams FAIL, bands outside 2 SE: ['mid', 'three'] (team mean |gap| 0.00506 <= 0.00725: True; r -0.103 >= 0.302: False) | season level, not gated: FT points 13.40, expected 13.42, gap -0.014
2021 development: teams FAIL, bands outside 2 SE: ['rim'] (team mean |gap| 0.00558 <= 0.00855: True; r -0.036 >= 0.341: False) | season level, not gated: FT points 12.30, expected 13.58, gap -1.276
2022 development: teams FAIL, bands outside 2 SE: ['rim', 'mid'] (team mean |gap| 0.00523 <= 0.00695: True; r -0.055 >= 0.272: False) | season level, not gated: FT points 14.44, expected 13.48, gap +0.953
2023 validation: teams FAIL, bands outside 2 SE: ['rim', 'short'] (team mean |gap| 0.00432 <= 0.00706: True; r 0.152 >= 0.315: False) | season level, not gated: FT points 13.34, expected 13.93, gap -0.600
band flags 12 of 78 (allowed 7); team checks pass every season: False; F2 share gate FAILED; wrote reports\free_throws.json
gate: F2 (re-specified 2026-09-26): band and team shares within 2 x their game-level bootstrap SE, every development season and validation; band flags by the count rule (95th percentile of a right model)
band flags 12 of 78 allowed 7 | team checks pass every season: False | passed: False
between-season sd of FT points per team-game (development): 0.6642
season-level gaps within +-0.1/0.2 (not gated): False
FAIL

===== 23 F3/F4 unit tests, Optuna reproducibility, recorded runtimes =====
11 passed in 27.48s
recorded backtest runtime 3130 s (limit 3600 s, from reports\week7-10_progress.md)
recorded study runtime 3508 s (limit 7200 s, from reports\week7-10_progress.md)
PASS

=== Item 20 re-run after the check fix ===
HEAD 5afcfc0, base origin/main, SKIP = every item but 20, start 10:23Z
===== 20 screenshots 1440/390, light/dark =====
light 1440: horizontal overflow 0 px; M1 line visible: True
light 390: horizontal overflow 0 px; M1 line visible: True
dark 1440: horizontal overflow 0 px; M1 line visible: True
dark 390: horizontal overflow 0 px; M1 line visible: True
PASS
FAILS: 0 (end 10:25Z)

=== Pass B: items 24-37 (HEAD df35abd, base origin/main, scratch C:\Users\nikos\AppData\Local\Temp\checklist_scratch24, start 11:06Z) ===
===== 24 F5 backtest_m2.json complete, declaration < verdict < test, two runs identical =====
run 1 (--score-test): 2078 s (limit 3600 s)
run 2 (--score-test): 1945 s (limit 3600 s)
runtime within budget: yes
two runs byte-identical: yes
equals the committed report: yes
declaration afbc08e, verdict 84308f9, test 6b2a5c1
declaration < verdict < test: NO
FAIL

===== 25 F6 calibration in the large per development season =====
calibration in the large: worst 2013 ratio 0.976618; shot-making year-to-year r 0.396322 (90% CI [0.310664, 0.48082]), split-half 0.314567: stable enough to show; wrote reports\m2_teams.json, reports\m2_players.json
team and player reports equal the committed ones
2011 0.996687 ok
2012 1.007724 OUTSIDE 0.5%
2013 0.976618 OUTSIDE 0.5%
2014 1.001616 ok
2015 0.980676 OUTSIDE 0.5%
2016 0.986529 OUTSIDE 0.5%
2017 1.002846 ok
2018 0.990545 OUTSIDE 0.5%
2019 1.019812 OUTSIDE 0.5%
2020 0.990682 OUTSIDE 0.5%
2021 1.005424 OUTSIDE 0.5%
2022 0.996847 ok
FAIL

===== 26 F7 m2_players.json with CIs, stability verdict = F-k rule =====
2 passed, 1 deselected in 0.07s
year-to-year {'n': 235, 'r': 0.396322, 'ci90': [0.310664, 0.48082]} split-half {'player_seasons': 595, 'r': 0.314567} -> stable enough to show
PASS

===== 27 F8 charts exist, geometry test =====
2 passed, 7 deselected in 1.33s
7 charts in docs/models/m2
every chart is referenced from docs/models/m2.md
PASS

===== 28 F9 MLflow parent + children, leakage tests =====
parent d983758dad324237b40290e257eb0621 (m2 704ac09c): data_sha256 89b47bd4b8ce..., commit df35abdb4d dirty=false
  child m2 704ac09c lgbm
  child m2 704ac09c lgbm_iso
  child m2 704ac09c optuna-study
  child m2 704ac09c spline
  child m2 704ac09c spline_iso
  optuna-study: 60 trial values (report 60)
7 passed in 5.18s
PASS

===== 29 F10 model card numbers match the reports =====
3 passed in 0.07s
PASS

===== 30 uv sync --no-dev, then build + predict =====
no-dev env without ['lightgbm', 'optuna', 'matplotlib', 'mlflow']
bash.exe : INFO eurohoops.predict: 0 upcoming games in window, 0 already logged, 0 rows appended
At C:\Users\nikos\AppData\Local\Temp\ps-script-668e8e06-d7a0-4f99-b780-498070950557.ps1:106 char:325
+ ... s=Get-Date; & "C:\Program Files\Git\bin\bash.exe" scripts/checklist.s ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : NotSpecified: (INFO eurohoops....0 rows appended:String) [], RemoteException
    + FullyQualifiedErrorId : NativeCommandError

0 predictions appended to predictions\euroleague_2026-27.csv
INFO eurohoops.live_m1: M1: 0 rows appended to predictions\euroleague_m1_2026-27.csv
0 M1 predictions appended to predictions\euroleague_m1_2026-27.csv
INFO eurohoops.predict: 0 upcoming games in window, 0 already logged, 0 rows appended
0 predictions appended to predictions\gbl_2026-27.csv
INFO eurohoops.live_m1: M1: 0 rows appended to predictions\gbl_m1_2026-27.csv
0 M1 predictions appended to predictions\gbl_m1_2026-27.csv
build + predict ran without the dev group; predictions/ as before
PASS

===== 31 G5 level variants: fields, post-hoc labels, two runs identical, leakage, declaration < run =====
14 passed in 5.82s
lgbm_level: CV log loss 0.629688, validation ECE 0.010049 (meets F-f: False), calibration in the large 0.983489-1.013094
spline_level: CV log loss 0.631511, validation ECE 0.009796 (meets F-f: False), calibration in the large 0.971405-1.010862
missing or unlabelled fields: []
G-g condition: {'meets_f_f_on_validation': False, 'cv_log_loss_below_lgbm': True, 'holds': False}
item 24's two runs: level blocks identical and equal to the report: True
declaration 1de203c, first run 327d0a3
declaration < first run: yes
weeks 7-10 numbers unchanged since the declaration: yes
PASS

===== 32 G1 no outcome-coded M2 feature level (both builders, development shots) =====
spline: 15 flag/level columns, 26 levels, make rates 0.3613-0.5754 on 385604 development shots
lgbm: 5 flag/level columns, 29 levels, make rates 0.0256-0.9378 on 385604 development shots
guard: feed flag fastbreak audited as a feature: 11537 shots, make rate 0.9997 -> caught
guard: feed flag second_chance audited as a feature: 17459 shots, make rate 0.9992 -> caught
guard: feed flag points_off_turnover audited as a feature: 22878 shots, make rate 0.9997 -> caught
no outcome-coded M2 feature level
PASS

===== 33 M3 unit tests, §3 facts =====
93 passed in 29.32s
1. players with stints 2011-2025: 1727 (< 5,000 for a dense posterior)
2. ids with several spellings: 84 of 1727; names with several ids: ['SIMONOVIC, MARKO', 'YURTSEVEN, OMER']
3. player-games of passing games within 60 s: 0.99997 of 89289 (>= 0.999)
4. gbl 2018: 302 team box scores with the 17 H-e columns
4. gbl 2019: 226 team box scores with the 17 H-e columns
4. gbl 2020: 318 team box scores with the 17 H-e columns
4. gbl 2021: 360 team box scores with the 17 H-e columns
4. gbl 2022: 312 team box scores with the 17 H-e columns
4. gbl 2023: 326 team box scores with the 17 H-e columns
4. gbl 2024: 332 team box scores with the 17 H-e columns
4. gbl 2025: 348 team box scores with the 17 H-e columns
4. gbl box headers that differ: 0
m3 facts: hold
PASS

===== 34 M3 leakage (RAPM, dummy, SPM, box-only, minutes, GBL) =====
19 passed, 41 deselected in 8.35s
PASS

===== 35 backtest_m3.json + gate, two runs identical, verdict < validation < test =====
reports/backtest_m3.json: two runs byte-identical
reports/backtest_m3.json: equals the committed report
  gate rapm_spm PASS -0.27147 [-0.560519, -0.001924]
verdict b09d19a, validation e58ae3e, test 33ec255
verdict < validation < test: yes
PASS

===== 36 backtest --model m3 runtime < 1,800 s =====
timed by item 35 in this run
RUNTIME 765 s (limit 1800 s)
recorded RUNTIME 875 s
PASS

===== 37 m3.md numbers = reports; gbl_stints.json pass rates, two builds identical =====
3 passed in 0.08s
reports/gbl_stints.json: two builds byte-identical
reports/gbl_stints.json: equals the committed report
  2018 0.5567 {'five_on_court': 0.9606, 'points': 1.0, 'possessions': 0.5813}
  2019 0.5683 {'five_on_court': 0.9928, 'points': 1.0, 'possessions': 0.5755}
reports/backtest_m3_gbl.json: two runs byte-identical
reports/backtest_m3_gbl.json: equals the committed report
  m3_gbl metrics ok for ['tuning', 'validation', 'test']
PASS

FAILS: 2 (end 12:43Z)
EXIT 2 ELAPSED 5807s

=== Item 24 order check re-run after the fix ===
$ uv run python scripts/checks/m2_order.py   (HEAD 0caf785)
declaration afbc08e, verdict 84308f9, test 6b2a5c1
declaration < verdict < test: yes
~~~
