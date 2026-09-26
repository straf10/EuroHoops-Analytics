# Weeks 7–10 closeout: shot model M2 (xPTS)

**Status: INCOMPLETE.** Every deliverable F1–F10 is built, tested and committed, and the full §6
checklist was run top to bottom once with no edits in between (output below): 26 PASS, 4 FAIL.
The stop condition does not hold: the four red checks are structural or budget failures that no
code fix within the declared design removes, and the exit gate failed on calibration. Nothing
was weakened, skipped or re-run to make it pass. Branch `week-7-10`, not pushed. Loop: 6
iterations of the 35-iteration budget.

## Exit gate (PLAN §8, row 7–10)
| Item | Result |
|---|---|
| 1. Calibrated (F-f on validation) | **FAIL**: chosen `lgbm` ECE 0.0104 (target ≤ 0.010); 2 of 20 bins outside ±0.02 (P 0.401: +0.0338; P 0.694: −0.0236) |
| 2. Beats the baseline | **PASS**: `lgbm` − `spline_iso` validation log loss −0.002019, game-level 95% CI [−0.002910, −0.001255]; 5-seed mean; no single seed flips it |
| 3. Stability decided | **PASS**: F-k verdict "stable enough to show" (year-to-year r 0.396, 90% CI [0.311, 0.481]; split-half 0.315) |
| 4. One clean run | **FAIL**: the single consecutive run has red items 22, 23, 24, 25 (below) |

## Checklist (single consecutive run at HEAD 349acb9)
An earlier full run (13:19Z-16:50Z, HEAD bd58276) was stopped with the user's approval: two
`astro dev` servers from a UI worktree kept a core busy, numpy/BLAS threads spun, and the M1
backtest took 3,298 s instead of 224 s (item 13 red for that reason only). The run below is the
complete rerun on a quiet machine.

| # | Check | Result |
|---|---|---|
| 1 | uv sync --frozen | PASS |
| 2 | ruff check | PASS |
| 3 | ruff format --check | PASS |
| 4 | mypy src | PASS |
| 5 | pytest + coverage | PASS |
| 6 | vulture | PASS |
| 7 | web build from the fixture (copy of web/) | PASS |
| 8 | Elo backtests reproduce the committed reports | PASS |
| 9 | predictions append-only vs origin/main, Elo model_version unchanged | PASS |
| 10 | stint sample reproduces origin/main | PASS |
| 11 | team_games coverage, points, possessions | PASS |
| 12 | stints mart thresholds, two builds identical | PASS |
| 13 | M1 unit tests, EuroLeague backtest runtime | PASS |
| 14 | leakage tests | PASS |
| 15 | M1 reports complete and reproducible | PASS |
| 16 | MLflow parent + children; no-dev run skips tracking | PASS |
| 17 | live M1 dry run | PASS |
| 18 | build + score + publish twice leaves the tree unchanged | PASS |
| 19 | actionlint | PASS |
| 20 | screenshots 1440/390, light/dark | PASS |
| 21 | F1 shot reconciliation, exclusion shares, two builds identical | PASS |
| 22 | F2 FT reconciliation per season (LOSO rates) | **FAIL** |
| 23 | F3/F4 unit tests, Optuna reproducibility, recorded runtimes | **FAIL** |
| 24 | F5 backtest_m2.json complete, declaration < verdict < test, two runs identical | **FAIL** |
| 25 | F6 calibration in the large per development season | **FAIL** |
| 26 | F7 m2_players.json with CIs, stability verdict = F-k rule | PASS |
| 27 | F8 charts exist, geometry test | PASS |
| 28 | F9 MLflow parent + children, leakage tests | PASS |
| 29 | F10 model card numbers match the reports | PASS |
| 30 | uv sync --no-dev, then build + predict | PASS |

FAILS: 4 (22, 23, 24, 25). No item BLOCKED or SKIPPED. Full output of the run (verbatim; HEAD 349acb9, 17:18Z-18:32Z, quiet machine; the web server's access lines of item 20 included):

```text
HEAD 349acb9, base origin/main, scratch /tmp/tmp.ex2SGNyAC0, start 17:18Z

===== 1 uv sync --frozen =====
Checked 137 packages in 12ms
PASS

===== 2 ruff check =====
All checks passed!
PASS

===== 3 ruff format --check =====
122 files already formatted
PASS

===== 4 mypy src =====
Success: no issues found in 45 source files
PASS

===== 5 pytest + coverage =====
31 files skipped due to complete coverage.
Required test coverage of 85% reached. Total coverage: 95.55%
388 passed, 1 warning in 74.73s (0:01:14)
PASS

===== 6 vulture =====
PASS

===== 7 web build from the fixture (copy of web/) =====
npm warn allow-scripts Run `npm approve-scripts --allow-scripts-pending` to review, or `npm approve-scripts <pkg>` to allow.
20:20:39 [build] 1 page(s) built in 2.65s
20:20:39 [build] Complete!
built 2 entries into site/
PASS

===== 8 Elo backtests reproduce the committed reports =====
5 passed in 0.08s
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
28 passed in 1.73s
full EuroLeague M1 backtest (--score-test): 243 s (limit 600 s)
PASS

===== 14 leakage tests =====
26 passed in 5.11s
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
euroleague: parent cc88af9fbaae4990b36fd2401d12fc32 (euroleague 0.2.0+m1.cbd9aaa3): 30 params, 831 metrics, data_sha256 5491ba42151e…, commit 349acb98eb dirty=true
  child 26a2257fae074a608893a1a8a734c6e8 normal_const: 35 params, 36 metrics, validation log loss 0.589663
  child e977834978ce42a08a9aedb293d09a3b normal_pace: 35 params, 37 metrics, validation log loss 0.589762
  child f80b86ef414843f2a55e9eaeb7918e60 student_t_const: 35 params, 37 metrics, validation log loss 0.587714
  child 48ee91d7230043aabbe15af6a3bc26cd student_t_pace: 35 params, 38 metrics, validation log loss 0.587792
gbl: parent 923d266cdddd464d8f988f585f09830a (gbl 0.2.0+m1.51ab8739): 30 params, 818 metrics, data_sha256 ab1e0fe4cfcf…, commit 349acb98eb dirty=true
  child b24da78f200e4446add5a5bae57cff52 normal_const: 35 params, 36 metrics, validation log loss 0.431306
  child fe08930fdcda429da901588c266e74e6 normal_pace: 35 params, 37 metrics, validation log loss 0.430736
  child 31fd509a2705414b86dd89242c1f8902 student_t_const: 35 params, 37 metrics, validation log loss 0.430290
  child 3d0886aa8f314678b5adc526df8162ae student_t_pace: 35 params, 38 metrics, validation log loss 0.429759
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
clean
PASS

===== 20 screenshots 1440/390, light/dark =====
127.0.0.1 - - [26/Sep/2026 20:38:44] "GET /EuroHoops-Analytics/ HTTP/1.1" 200 -
127.0.0.1 - - [26/Sep/2026 20:38:44] "GET /EuroHoops-Analytics/_astro/index.4A5Xiosz.css HTTP/1.1" 200 -
127.0.0.1 - - [26/Sep/2026 20:38:44] "GET /EuroHoops-Analytics/_astro/sofia-sans-latin-wght-normal.B9fg0t1U.woff2 HTTP/1.1" 200 -
127.0.0.1 - - [26/Sep/2026 20:38:44] "GET /EuroHoops-Analytics/_astro/caveat-brush-latin-400-normal.2w-6t9gW.woff2 HTTP/1.1" 200 -
127.0.0.1 - - [26/Sep/2026 20:38:44] "GET /EuroHoops-Analytics/_astro/sofia-sans-extra-condensed-latin-wght-normal.CF4o1LBE.woff2 HTTP/1.1" 200 -
127.0.0.1 - - [26/Sep/2026 20:38:45] "GET /EuroHoops-Analytics/ HTTP/1.1" 200 -
127.0.0.1 - - [26/Sep/2026 20:38:45] "GET /EuroHoops-Analytics/_astro/index.4A5Xiosz.css HTTP/1.1" 200 -
127.0.0.1 - - [26/Sep/2026 20:38:45] "GET /EuroHoops-Analytics/_astro/sofia-sans-latin-wght-normal.B9fg0t1U.woff2 HTTP/1.1" 200 -
127.0.0.1 - - [26/Sep/2026 20:38:45] "GET /EuroHoops-Analytics/_astro/sofia-sans-extra-condensed-latin-wght-normal.CF4o1LBE.woff2 HTTP/1.1" 200 -
127.0.0.1 - - [26/Sep/2026 20:38:45] "GET /EuroHoops-Analytics/_astro/caveat-brush-latin-400-normal.2w-6t9gW.woff2 HTTP/1.1" 200 -
127.0.0.1 - - [26/Sep/2026 20:38:46] "GET /EuroHoops-Analytics/ HTTP/1.1" 200 -
127.0.0.1 - - [26/Sep/2026 20:38:46] "GET /EuroHoops-Analytics/_astro/index.4A5Xiosz.css HTTP/1.1" 200 -
127.0.0.1 - - [26/Sep/2026 20:38:46] "GET /EuroHoops-Analytics/_astro/sofia-sans-latin-wght-normal.B9fg0t1U.woff2 HTTP/1.1" 200 -
127.0.0.1 - - [26/Sep/2026 20:38:46] "GET /EuroHoops-Analytics/_astro/sofia-sans-extra-condensed-latin-wght-normal.CF4o1LBE.woff2 HTTP/1.1" 200 -
127.0.0.1 - - [26/Sep/2026 20:38:46] "GET /EuroHoops-Analytics/_astro/caveat-brush-latin-400-normal.2w-6t9gW.woff2 HTTP/1.1" 200 -
127.0.0.1 - - [26/Sep/2026 20:38:47] "GET /EuroHoops-Analytics/ HTTP/1.1" 200 -
127.0.0.1 - - [26/Sep/2026 20:38:47] "GET /EuroHoops-Analytics/_astro/index.4A5Xiosz.css HTTP/1.1" 200 -
127.0.0.1 - - [26/Sep/2026 20:38:47] "GET /EuroHoops-Analytics/_astro/sofia-sans-latin-wght-normal.B9fg0t1U.woff2 HTTP/1.1" 200 -
127.0.0.1 - - [26/Sep/2026 20:38:47] "GET /EuroHoops-Analytics/_astro/caveat-brush-latin-400-normal.2w-6t9gW.woff2 HTTP/1.1" 200 -
127.0.0.1 - - [26/Sep/2026 20:38:47] "GET /EuroHoops-Analytics/_astro/sofia-sans-extra-condensed-latin-wght-normal.CF4o1LBE.woff2 HTTP/1.1" 200 -
light 1440: horizontal overflow 0 px; M1 row visible: True
light 390: horizontal overflow 0 px; M1 row visible: True
dark 1440: horizontal overflow 0 px; M1 row visible: True
dark 390: horizontal overflow 0 px; M1 row visible: True
PASS

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
2011 development: FT points 14.23, expected 13.26, gap +0.971 (OUTSIDE ±0.1)
2012 development: FT points 13.22, expected 13.35, gap -0.129 (OUTSIDE ±0.1)
2013 development: FT points 12.59, expected 13.54, gap -0.954 (OUTSIDE ±0.1)
2014 development: FT points 13.69, expected 13.81, gap -0.124 (OUTSIDE ±0.1)
2015 development: FT points 13.71, expected 13.50, gap +0.203 (OUTSIDE ±0.1)
2016 development: FT points 13.51, expected 13.74, gap -0.233 (OUTSIDE ±0.1)
2017 development: FT points 14.51, expected 13.62, gap +0.889 (OUTSIDE ±0.1)
2018 development: FT points 13.70, expected 13.71, gap -0.011 (within ±0.1)
2019 development: FT points 13.69, expected 13.79, gap -0.100 (within ±0.1)
2020 development: FT points 13.40, expected 13.42, gap -0.014 (within ±0.1)
2021 development: FT points 12.30, expected 13.58, gap -1.276 (OUTSIDE ±0.1)
2022 development: FT points 14.44, expected 13.48, gap +0.953 (OUTSIDE ±0.1)
2023 validation: FT points 13.34, expected 13.93, gap -0.600 (OUTSIDE ±0.2)
wrote reports\free_throws.json
between-season sd of FT points per team-game (development): 0.6642
FAIL

===== 23 F3/F4 unit tests, Optuna reproducibility, recorded runtimes =====
10 passed in 5.64s
recorded backtest runtime 3130 s (limit 2400 s)
recorded study runtime 3508 s (limit 7200 s)
FAIL

===== 24 F5 backtest_m2.json complete, declaration < verdict < test, two runs identical =====
run 1 (--score-test): 3072 s (limit 2400 s)
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
2 passed, 7 deselected in 0.87s
7 charts in docs/models/m2
every chart is referenced from docs/models/m2.md
PASS

===== 28 F9 MLflow parent + children, leakage tests =====
parent 070f7e4672f8455fb1e95dc8d0a4a2a8 (m2 704ac09c): data_sha256 89b47bd4b8ce..., commit 349acb98eb dirty=true
  child m2 704ac09c lgbm
  child m2 704ac09c lgbm_iso
  child m2 704ac09c optuna-study
  child m2 704ac09c spline
  child m2 704ac09c spline_iso
  optuna-study: 60 trial values (report 60)
5 passed in 3.76s
PASS

===== 29 F10 model card numbers match the reports =====
3 passed in 0.06s
PASS

===== 30 uv sync --no-dev, then build + predict =====
no-dev env without ['lightgbm', 'optuna', 'matplotlib', 'mlflow']
INFO eurohoops.predict: 0 upcoming games in window, 0 already logged, 0 rows appended
0 predictions appended to predictions\euroleague_2026-27.csv
INFO eurohoops.live_m1: M1: 0 rows appended to predictions\euroleague_m1_2026-27.csv
0 M1 predictions appended to predictions\euroleague_m1_2026-27.csv
INFO eurohoops.predict: 0 upcoming games in window, 0 already logged, 0 rows appended
0 predictions appended to predictions\gbl_2026-27.csv
INFO eurohoops.live_m1: M1: 0 rows appended to predictions\gbl_m1_2026-27.csv
0 M1 predictions appended to predictions\gbl_m1_2026-27.csv
removed header-only predictions/euroleague_m1_2026-27.csv (created by this check)
removed header-only predictions/gbl_m1_2026-27.csv (created by this check)
build + predict ran without the dev group; predictions/ as before
PASS

FAILS: 4 (end 18:32Z)
```

## Gate verdict, then test
Validation 2023-24 (41,198 shots), verdict commit 84308f9 (before any test run):

| | Log loss | Brier | ECE |
|---|---|---|---|
| lgbm (challenger, chosen) | 0.633267 | 0.223071 | 0.0104 |
| spline_iso (baseline) | 0.635286 | 0.223827 | 0.0092 |
| difference, game-level 95% CI | −0.002019 [−0.002910, −0.001255] | −0.000756 [−0.001027, −0.000509] | +0.0012 [−0.0026, +0.0056] |

Test 2024-25 + 2025-26 (93,119 shots), scored once after the verdict (commit 6b2a5c1): log loss
−0.002119 [−0.002601, −0.001658], Brier −0.000842 [−0.001034, −0.000654], ECE +0.0006
[−0.0031, +0.0039]; `lgbm` test ECE 0.0126. Same verdict: better than the baseline, not
calibrated. Development and validation numbers of the test run are identical to the verdict run.

## Every variant and the Optuna study
| Variant | CV log loss | Validation log loss | Validation ECE | Test log loss | Post-hoc |
|---|---|---|---|---|---|
| spline (8 knots, L2 1e-4 by CV) | 0.631577 | 0.635097 | 0.0092 | 0.635433 | no |
| spline_iso | 0.631513 | 0.635286 | 0.0092 | 0.635587 | no |
| lgbm (5-seed mean) | 0.629699 | 0.633267 | 0.0104 | 0.633467 | no |
| lgbm_iso | 0.629867 | 0.633333 | 0.0116 | 0.633374 | no |

Optuna: TPE, seed 20261001, 60 trials, pooled LOSO CV log loss on development seasons; best trial
57, CV 0.629734: 15 leaves, learning rate 0.0183, 900 trees, 178 min child samples, feature
fraction 0.955, bagging fraction 0.783, λ₂ 0.140. Runtime 3,508 s (budget 2 h). Seeds: validation
log loss 0.633300 ± 0.000049 (sd over 5 seeds).

**Post-hoc items:** none of the four variants. One change was made after a validation look, and
is labelled: the three outcome-coded feed flags were removed from the features after the first
(invalid) run (see "Decisions" below). The chart redraw (titles, hex edges) came after the first
look at the PNGs; it changes no number.

## F-e free-throw method and its limits
The shot feed logs made free throws only, so FT trips come from the play-by-play (`FTM`/`FTA`),
grouped by team and clock. PBP fouls are untyped, so a trip can be tied to its shot only for
**and-ones** (one-shot trip after the same team's made field goal at the same clock, joined by
`NUMBEROFPLAY` = `NUM_ANOT`); those carry the shot's distance band. Every other trip (fouls on
misses, bonus, technicals) is team-level. Expected FT points = Σ bands FGA × and-one trips per FGA
× points per and-one trip + FGA × other trips per FGA × points per other trip, ratio-of-totals
rates, LOSO on development seasons. **Limit:** league FT points per team-game vary between seasons
(sd 0.66), so out-of-season rates miss a season's level by up to 1.28 points; only 3 of 12
development seasons are within ±0.1 (check 22).

## Stability, stated plainly
Shot-making (points above expected per 100 shots, shrunk) repeats from one season to the next with
r ≈ 0.40 (90% CI 0.31–0.48) and within a season (odd vs even games) at 0.31, just over the 0.30
bar. By the rule fixed beforehand it is **stable enough to show**, but only modestly: about 44% of
the spread between players is signal, the rest is noise. Raw FG% repeats more (0.67) because it
mostly repeats shot selection, which shot-making is designed to remove.

## MLflow run ids (experiment `m2-backtest`, `mlruns/`)
- a6d1ae891d434fe5aa410b2ea58210dd — the verdict run (study + backtest, validation)
- d5ca70bafbbc411791aff2c649bb2e65 — the test run (`--score-test`)
- b94decee1ad343dc97a9e441ae606c5b — the first run, **invalid** (outcome-coded features)
- 070f7e4672f8455fb1e95dc8d0a4a2a8 — checklist item 24's run in the final run (its report is
  byte-identical to the committed one; the check stopped on runtime before a second run)

## Decisions this file did not cover
1. **Outcome-coded flags removed (departs from F-b).** From 2015-16 the feed sets `FASTBREAK`,
   `SECOND_CHANCE` and `POINTS_OFF_TURNOVER` only on made shots (99.8–100% make rate when set).
   The first run used them (CV log loss 0.553 vs 0.630 honest) and was invalid; it was recorded in
   the progress file, never committed, and the flags were dropped by an amended declaration
   (afbc08e) before the second run. The Optuna study was re-run because F4's features changed.
2. **Nested isotonic calibration** (leave-two-seasons-out) so no calibrated out-of-fold value
   depends on its own season; 66 extra fits per configuration and seed.
3. **Challenger choice on the 5-seed mean**; the gate uses the 5-seed mean prediction.
4. **F7 sampling variance by game-level bootstrap**; year-to-year CI by bootstrap over player pairs
   (F-g's game-level bootstrap does not apply across seasons); split-half uncorrected.
5. **Shot table stops at 2025-26**; `eurohoops shots`, `free-throws`, `shot-quality`,
   `shot-charts` are local commands, not part of `build` (the daily workflow is unchanged).
6. **Exclusions:** the (−1, −1) sentinel on 78 field goals is `unparseable`; `label_geometry`
   applies from 2011-12 only.
7. **Context breakdowns** use home / last 24 s / overtime instead of the outcome-coded flags.
8. **Chart subjects:** PAO, OLY, RMB and the three top-FGA players of 2023-24 (Howard, James, Nunn),
   named in the progress file before drawing.
9. **LightGBM on 6 threads** (fixed for determinism).
10. **Checklist item 30** runs `predict --window-hours 0` and removes header-only M1 logs it creates
    (main does not track them yet), so `predictions/` ends as it started.
11. Live-season drift of `live_scorecard.json`, `possessions.json`, `stints_mart.json` during the
    checklist is reverted, never committed from this branch.
12. **Number-preserving speed-up** after the verdict: LightGBM zone codes by a vectorised map
    (verified identical on every development shot). It saved about 1 minute of the M2
    backtest (3,130 → 3,072 s), not the ~9 minutes estimated from a profile: LightGBM fitting
    dominates. Item 24's report is byte-identical to the committed one.
13. **UI work in a separate worktree** ran concurrently by the user; its dev servers were stopped
    for the final run (see above).

## Red checks, last error, and what I would try next
- **22 (F2 FT reconciliation):** mean gaps 0.971 (2011), −0.954 (2013), 0.889 (2017), −1.276
  (2021), 0.953 (2022); validation −0.600. Structural: between-season sd 0.66 > the ±0.1 target.
  Next: decide the target (a tolerance tied to between-season variation, or reconcile within-season
  shares instead of levels); or add a season-level FT environment estimated from the season's own
  early games (a live-usable design, but no longer pure LOSO).
- **24 (F5 report):** `run 1 (--score-test): 3072 s (limit 2400 s)` → the check stops before its
  second run. The run's report equals the committed one byte for byte (tree clean after it), and
  the test run's development/validation numbers equal the verdict run's, so reproducibility is
  shown, but not by the check as written. Same next step as 23.
- **23 (runtime):** backtest 3,130 s with test seasons (2,951 s validation only) > 2,400 s. Next:
  drop the nested isotonic fits for LightGBM seeds 2–5 (calibrate their mean with seed 1's
  leave-two-out fits), or 12 threads; both change the declared computation, so they need a new
  declaration.
- **25 (F6 calibration in the large):** ratio 0.9766–1.0198; 7 of 12 development seasons outside
  0.5%. Same season-level drift. Next: a per-season intercept estimated from the season's first
  N games (or the prior season), evaluated as a declared new variant.
- **Exit-gate item 1 (calibration):** chosen `lgbm` ECE 0.0104, bins at P 0.40 and 0.69 outside
  ±0.02; per-band ECE up to 0.0495 (deep threes). Next: the per-season level variant above, then
  re-declare; also rebuild transition / put-back context from PBP for makes and misses alike.

## Open questions for the user
1. F2's ±0.1 target is below the natural season-to-season variation (sd 0.66): keep it (red),
   or re-specify?
2. Accept the removal of the three outcome-coded flags from F-b? Rebuild "second chance" and
   "after a turnover" from the play-by-play as a future feature?
3. The 40-minute backtest budget: relax it, or trade the nested isotonic fits for speed?
4. Given the calibration miss, should M2's xPTS be used downstream (team pages, M6) with a
   season-level caveat, or wait for a per-season-level variant?
