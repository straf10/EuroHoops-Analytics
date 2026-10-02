# Weeks 12–14 closeout: entity resolution + M4 league translation

**Status: COMPLETE (2026-10-02).** I0–I9 are built, tested and committed. Cross-league player
matching is reported with precision and recall on the owner's labels. The M4 exit gate
("translation beats naive on GBL→EL movers") **FAILS** honestly: six pooled movers cannot show
it, and the test seasons have none. Model card: `docs/models/m4.md`; data doc:
`docs/data/entity.md`; progress log: `reports/week12-14_progress.md` (iterations 1–16).

## Exit gate (PLAN §8, row 12–14)
| Part | Status | Evidence |
|---|---|---|
| Cross-league matching built, P/R reported | **done** | `reports/entity_resolution.json`: 2,997 persons, 191 GBL ids matched |
| Precision/recall on the owner's labels | reported | as frozen: precision 1.0 (95% CI 0.957–1.0), recall 0.9885 (0.938–0.998), 186 labels; after overrides 1.0 / 1.0 (fitted on the labels) |
| Translation factors | built | `reports/m4_translation.json` (δ per stat with 90% intervals, team offset 20.2 [16.9, 23.6] points per 100) |
| **Translation beats same stats on GBL→EL movers** | **FAIL** | pooled 2019–2023, 6 movers: loss difference −0.138, 95% CI [−0.430, 0.018]; test seasons have 0 movers |

## Entity resolution
- Matcher frozen at `d17fc3a` (`w_surname` 0.5, `t_dob` 0.78, `t_near` 0.95, `t_nodob` 0.82,
  `b_club` 0.08, `b_jersey` 0, window ± 2 seasons). Silver set: precision 1.0, recall 0.836
  (18 of the 23 misses are teammates with equal or near birth dates, not matcher errors).
- Owner labels (`2799c49`, after the matcher commit): 186 GBL ids, 87 matched to a EuroLeague id,
  99 none. As frozen: 86 of 86 predicted matches right; one true match missed. **Known miss:**
  Alexander (Sasha) Vezenkov, name score 0.77 under the 0.78 equal-birth-date threshold. Fixed by
  a `label:` override (`96a4d65`); the after-overrides numbers are fitted on the labels.
- Strata (n / predicted / true / correct): greek/matched 50/50/50/50; greek/unmatched 50/0/0/0;
  latin/matched 36/36/36/36; latin/unmatched 50/0/1/0.
- Other silver misses without a label (McKissic, George Papas, Moses Wright, Thomas Walkup) stay
  known misses.

## M4
- GBL→EL movers per split: tuning 5 (2019–2022), validation 1 (2023), test 0 (2024–2025); pooled
  gate 6. Variant `translate` declared by the pre-registered small-sample rule (verdict commit
  `044f670`, before any validation number); validation `04afe68`; test `4c7a89f`.
- Validation (1 mover): translate 0.700 vs same stats 0.586, worse; pooled gate FAIL as above.
- EL→GBL (26, 2019–2025, reported, not gated): translate 0.613 vs same stats 0.616.
- The full tables, δ per stat and the SPM extra are in `docs/models/m4.md` (number-checked).
- **M4 runs on the crosswalk as frozen** (`player_xwalk_frozen`, without `label:` overrides): with
  the Vezenkov override the gate CI upper bound moved 0.018 → 0.020 and EL→GBL movers 26 → 19, so
  the owner chose to keep the pre-registered scores (decision 2026-10-02, commit `7fd1003`).

## Checklist
Full output: `reports/week12-14_checklist_run.txt`. Not one pass over 1–42; the owner chose to
re-run only what the phase touched.

| # | Check | Result |
|---|---|---|
| 1–12 | setup, lint, types, tests + coverage, vulture, web build, Elo, predictions append-only, stints | PASS (HEAD ad2ea65) |
| 13–37 | M1, M2, M3 checks | SKIPPED (no M1/M2/M3 change in this phase) |
| 38 | entity: tests, two builds identical, 106 s < 300 s, matcher < labels < label override | PASS |
| 39 | crosswalk invariants, overrides held | PASS |
| 40 | M4 unit and leakage tests | PASS |
| 41 | `backtest_m4.json` gate, two runs identical, verdict < validation < test, 28 s | PASS |
| 42 | M4 card and entity doc numbers = reports | PASS |

Item 9 failed once because the branch lacked `origin/main`'s newer daily-prediction commits;
after merging main it passed (0 changed lines).

## Subagent log
| Wave | Subagent | Delivered | Rounds | Orchestrator fixes |
|---|---|---|---|---|
| 1 | A | name tables (I1), done by the orchestrator | — | — |
| 1 | B | transliteration + Jaro-Winkler (I2) | 3 | ΓΟΥ/ΑΪ rules sent back; JR/SR, final Σ/Ι/Η rules for the 38/40 fixture gate |
| 1 | C | matcher + crosswalk (I3) | 1 | vectorised blocking, cached keys (240 → 88 s) |
| 2 | D | pair tables (I6) | 1 | — |
| 2 | E | translation model (I7) | 1 | — |

## Decisions this file did not cover
D1–D9 in the progress file, plus:
- **D10 (labels file).** The spreadsheet save stripped leading zeros from 14 GBL ids and turned two
  into scientific notation; ids were rebuilt from the ESAKE link on each row, `None` → `none`.
- **D11 (M4 input).** M4 reads `player_xwalk_frozen`; `player_xwalk` carries every override.
- **D12 (checklist scope).** 13–37 skipped, see above.
- Candidate pairs are 93,344, not the 94,049 first reported (the I2 follow-up changed blocking).

## Open questions for the owner
1. M4 cannot be shown to help or not: keep it research-only, or revisit when more GBL→EL movers
   exist (2026-27 and later)?
2. The four other silver misses (McKissic, Papas, Wright, Walkup): add overrides after a look at
   their pages, or leave them as known misses?

## memory.md entry
Added under 2026-10-02 (weeks 12–14, COMPLETE).

## Checklist run (verbatim)

~~~text
Weeks 12-14 checklist run, 2026-10-02 (scripts/checklist.sh, Git Bash on PATH; times UTC)

Not one uninterrupted pass over 1-42. The owner chose this record (2026-10-02): items that
nothing in this phase touches are not re-run.
- Part A ran items 1-12 on HEAD ad2ea65 (after merging origin/main into the branch: the first
  attempt failed item 9 only because the branch lacked main's two newer daily-prediction commits,
  so 20 prediction lines looked removed; with the merge: 0). Stopped by the owner's decision
  after item 12.
- Items 13-37 (M1, M2, M3 checks) SKIPPED: this phase changed no M1/M2/M3 code, data or report.
- Part B ran items 38-42 on the same HEAD (SKIP = 1-37).
Result: every run item passes (FAILS: 0 in part B; part A items 1-12 PASS).

##### PART A (items 1-12)
HEAD ad2ea65, base origin/main, scratch <scratch>/chk, start 09:38Z

===== 1 uv sync --frozen =====
Checked 137 packages in 14ms
PASS

===== 2 ruff check =====
All checks passed!
PASS

===== 3 ruff format --check =====
208 files already formatted
PASS

===== 4 mypy src =====
Success: no issues found in 76 source files
PASS

===== 5 pytest + coverage =====
38 files skipped due to complete coverage.
Required test coverage of 85% reached. Total coverage: 93.50%
668 passed, 1 warning in 409.95s (0:06:49)
PASS

===== 6 vulture =====
PASS

===== 7 web build from the fixture (copy of web/) =====
npm warn allow-scripts Run `npm approve-scripts --allow-scripts-pending` to review, or `npm approve-scripts <pkg>` to allow.
[2m12:46:48[22m [34m[build][39m 2268 page(s) built in [1m28.37s[22m
[2m12:46:48[22m [34m[build][39m [1mComplete![22m
built 8 entries into site/
PASS

===== 8 Elo backtests reproduce the committed reports =====
5 passed in 0.09s
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
EL sample: 50 games, within �2: 94.0% of team-games, 88.0% of games; mean gap 0.024; FT weight matching PBP 0.4213
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

##### PART B (items 38-42)
HEAD ad2ea65, base origin/main, scratch <scratch>/chkB, start 10:06Z






































===== 38 entity: unit tests, two builds identical, runtime, matcher < labels < label overrides =====
67 passed in 3.58s
reports/entity_resolution.json: two builds byte-identical
marts identical:
player_names 7907 fecc928fe2ee12be4508f11e65d75649ae831bff5ab88c479fef663eb6ab0883
player_xwalk 3189 3d03d405d103fad311a70fa77dde40e17bee980ffd36474fc376d017edbbd838
player_xwalk_frozen 3189 557319d8606895fa0344abae428394c935c50b8c57a9b1f431c099dec982f5bf
reports/entity_resolution.json: equals the committed report
entity RUNTIME 106 s (limit 300 s)
matcher d17fc3a, labels 2799c49
first label override 96a4d65
matcher < labels < label overrides: yes
PASS

===== 39 crosswalk invariants: every box id in one person, overrides held =====
crosswalk rows 3189, persons 2997, duplicate ids 0
GBL player_box ids without a person: 0; EL box ids: 0
override P005353 match P005983: held
override PLRU no_match P012711: held
override 00000069 match P003469: held
crosswalk invariants: hold
PASS

===== 40 M4 unit and leakage tests (pairs, translation model, backtest) =====
27 passed in 50.90s
PASS

===== 41 backtest_m4.json + gate, two runs identical, verdict < validation < test, runtime =====
reports/backtest_m4.json: two runs byte-identical
reports/backtest_m4.json: equals the committed report
reports/m4_translation.json: two runs byte-identical
reports/m4_translation.json: equals the committed report
  gate translate FAIL -0.137504 [-0.429934, 0.018067] pooled 6 movers
verdict 044f670, validation 04afe68, test 4c7a89f
verdict < validation < test: yes
RUNTIME 28 s (limit 600 s); recorded RUNTIME m4 20 s
PASS

===== 42 docs/models/m4.md and docs/data/entity.md numbers = reports =====
4 passed in 0.07s
PASS

FAILS: 0 (end 10:13Z)
~~~
