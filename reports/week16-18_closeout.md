# Weeks 16–18 closeout: M6 player projections, the read-only API, and the new pages

**Status: COMPLETE, with a split checklist record (owner's choice).** L0–L12 are built, tested and committed on `week-16-18`. The M6 gate passes on
EuroLeague validation 2023, and the test seasons were scored once. The product gate holds: every
file the site reads is an API response, and the contract test checks this byte for byte.
On 2026-10-08 the owner stopped the full run after item 24 and chose a split record of the items this phase
touched: 1-11, 17-20, 30-39, 52 and 54-60 (HEADs cb416c6, ff44d72, 14fc9e7). Every item this phase owns (54-60)
passes; the drift the run found in items 9, 10, 35 and 37 is fixed (10 turns green once pushed). The
unchanged-model items (12-16, 21-29, 40-51, 53) run in the next full pass. Full output:
`reports/week16-18_checklist_run.txt`.
Model card: `docs/models/m6.md`. API: `docs/api.md`. Progress log: `reports/week16-18_progress.md`
(iterations 1–12, decisions D1–D25).

## Exit gates (PLAN §8, row 16–18)
| Gate | Status | Evidence |
|---|---|---|
| **M6 (L-h): the chosen next-season projection beats the Marcel baseline on EuroLeague validation 2023 by the point rule, with calibrated intervals** | **PASS** | `proj_shrunk@1` 5.398130 vs `marcel` 5.418873 (diff −0.020743, 95% CI [−0.166008, 0.139934]). Pooled 80% coverage 0.794–0.843; all 14 stats are in the 0.75–0.85 band. |
| **Product (L-j): all pages live from the API** | **PASS** | `publish` and `export-stats` write the API's response bytes (`api/export.py`). `tests/test_api_contract.py` walks `web/src/data` and `web/public/api`. Item 59 on real data: the old and new exports are byte-identical (104 stats files and 966 publish files). |

## Pre-registration (git order, `scripts/checks/m6_order.py`)
VERDICT ab8264b (tuning only) < VALIDATION 00cc78c < TEST 5efeec8; the GBL report followed in 1c158a7.

Before the verdict, three things happened, all on tuning evidence only:
- The method was fixed three times (D17 BRAPM snapshot noise, D18 regressed aging deltas, D19
  interval scale).
- The grid was widened once (D20: half-life {0.5, 1, 2, 3}).
- Three tuning attempts were discarded and recorded in the progress file.

No validation number existed before 00cc78c, and no test number before 5efeec8.

## §3 findings (L0, real data; re-checked by item 54)
- **Possessions:** every player game with playing time has a possession count: EuroLeague 107,692
  player games (2007–2026), GBL 26,832 (2018–2026).
- **Scored sets** (≥ 500 possessions and a senior season before): EuroLeague tuning 2016–2022 has
  1,000 player-seasons, validation 2023 has 165 and test 2024–2025 has 338. GBL targets run
  2019–2025 (71–81 a season from 2020).
  - The scoring runs then dropped players without a target-season truth line: 977, 164 and 327.
  - 2019-20 (the COVID-shortened season) keeps 128 scored players and stays in tuning.
- **Birth dates:** known for 100% of scored player-seasons. Five implausible GBL dates are dropped
  (D16). No age or birth date reaches any committed output: `project` refuses to write a report that
  carries one, and item 57 checks it.
- **Walk-forward availability:** M4 fits cover EuroLeague targets 2019–2025 (D5), and M3 BRAPM
  snapshots 2011–2025 (D4).
- **Route ↔ page table:** progress file §3.6 (every `web/src/data` file is read by a page) and
  `docs/api.md` (route → file).
- **Four factors and schedule strength:** derivable from the box scores and the published Elo, and now
  shown on Teams.

## Tuning (EuroLeague 2016–2022, 977 scored players)
| Variant | Tuning loss |
|---|---|
| **`proj_shrunk`, half-life 1 (chosen)** | **6.090287** |
| `proj_shrunk`, half-life 2 | 6.092360 |
| `proj_shrunk`, half-life 3 | 6.130200 |
| `proj_shrunk`, half-life 0.5 | 6.249258 |
| `proj_full` @1 | 6.166064 |
| `proj_full_spm` @1 | 6.168281 |
| `proj_age` @1 | 6.169617 |
| `marcel` | 6.190196 |
| `same_as_last` | 10.189698 |
| `league_mean` | 13.661100 |

The decay edge: half-life 1 sat on the edge of the first grid, so the grid was widened once (D20).
Half-life 0.5 is clearly worse, so the choice is inside the widened grid (`on_edge` false). Aging
and translation did not help on tuning. Per-stat MAE and coverage are in
`reports/backtest_m6.json` and the card.

## Validation (2023) and test (2024–2025), each scored once
| Split | Players | `proj_shrunk@1` | `marcel` | Diff vs `marcel` [95% CI] | `same_as_last` |
|---|---|---|---|---|---|
| validation | 164 | 5.398130 | 5.418873 | −0.020743 [−0.166008, 0.139934] | 8.512821 |
| test | 327 | 5.901776 | 6.058210 | −0.156434 [−0.289809, −0.032009] | 13.339300 |

- **Validation:** a tie with Marcel in CI terms, and a clear win over `same_as_last` (−3.114691,
  [−4.066925, −2.178728]). Coverage is calibrated (validation only: 0.774–0.854).
- **Test:** a clear lead over Marcel.
- **Rest of season (reported, not gated):** below `marcel` at every checkpoint of every split; for
  example, test at 25% gives 5.574324 vs 5.963480.
- **Movers (GBL ↔ EuroLeague):** too few to read (6 / 2 / 2). On test the loss is 7.258933 vs
  `marcel` 9.229818.
- **GBL (EuroLeague choice fixed, not gated):**
  - Tuning ties `marcel` (−0.092705, [−0.242576, 0.063066]), and so does test (+0.024464,
    [−0.206626, 0.235886]).
  - Validation is worse (6.3149 vs 5.9513, 53 players).
  - Blocks (0.868) and 3P% (0.882) over-cover.

## Product
**Pages** (L11, subagents I and J; every number has a visible interval or sample, checked in the DOM):

| Page | Change |
|---|---|
| `players/[slug]` | 2026-27 projection (80% intervals, exposure, flags in words, gate verdict), impact (projected SPM and BRAPM, M3 ratings per season), M2 shot-making with FGA, "Plays like" top 5 |
| `scouting/` (new) | Over/under board (three dimensions, both leagues), GBL undervalued list with its rule, GBL→EuroLeague translation table with M4's failed gate |
| `standings/` (new) | Ungated M7 season simulation for both leagues, labelled "Not gated" with M7's reason |
| `methodology/` (new) | M1–M7, one line each with gate verdict and card link; full write-up in weeks 18–20 |
| `teams/[code]` | Four factors and schedule strength for the live season |
| Forecasts (`/`) | "Elo beside the team models": Elo, baseline, M1 and the M5 shadow from `/metrics/live` |
| nav | Scouting, Standings, Methodology |

**Screenshots** (not committed; 1440/390, light/dark): the fixture builds are in the scratchpad
`fix/shots` (player, sparse player, scouting, standings, methodology, two teams, forecasts EL/GBL).
The real-data review is in `real/shots` (standings, scouting, Walter Tavares, OLY, forecasts EL/GBL,
methodology).
- Real-data review: 28 of 28 page/viewport runs passed, with 0 px overflow, 0 unlabelled numbers and
  0 console errors.
- 271 player pages show M6 numbers, which equals the 271 exported EuroLeague persons.

**DESIGN.md review findings:**
- Fixed by the subagents after their own screenshots:
  - J: a CSS attribute (`data-comp`) collided with the competition switch and hid the GBL standings
    table; a figure margin indented its caption; a missing space; a repeated heading; an over-wide
    scorecard table.
  - I: a clipped "Sample" cell on phones; a board column too wide on phones; tight spacing above the
    translation table.
- Fixed by me: Scouting was not marked current in the nav; `live.ts` had a hand copy of the display
  codes (it now uses the map that `tests/test_web_m6_codes.py` keeps equal to `publish.py`).
- Logged, not changed:
  - The board's shot-making "Expected" column is always 0.0. This is correct by definition (points
    above xPTS), but redundant.
  - On a fixture build, GBL standings show codes only; real data shows names.
  - Tables scroll inside their own container on 390 px phones, as the existing stats tables do.
  - Sorting the board needs JavaScript; without it the order is z descending.
- Colour stays on data only: pass and fail are told apart by text, and label chips use ink weight.

**API:** `docs/api.md` (16 routes, local run, export mapping, checks). The image builds (`docker
build`), and a `docker compose up` smoke test on real data answered every route probed. The first
person request builds the player-season frame: about 10 s locally and 56 s in Docker on a Windows
bind mount, then under 0.3 s.

**Runtimes** (progress file `RUNTIME` lines; item 58):

| Step | Seconds | Budget |
|---|---|---|
| `backtest --model m6` EuroLeague | 161 | 1,200 |
| `backtest --model m6` GBL | 63 | 300 |
| `project` | 48 | 90 |
| Export through the API | 66 | 180 |
| Astro build | 69 | 240 |

A second real-data export run during a busy session took 139 s, also inside its budget.

## Live
`daily.yml` now runs two new steps after scoring and before the export:
- **`eurohoops project --projections-only`:** the 2026-27 projections and the GBL undervalued list,
  from the committed crosswalk `entity/player_xwalk.csv` (D25). A failure is a warning.
- **`eurohoops sim-ungated`** (EuroLeague and GBL): the Standings page data. It is rewritten only
  after a newly completed round.

The board and "plays like" need the local M2 shot marts, so they are rebuilt locally and stay as
committed. On 2026-10-08 both leagues are at checkpoint 0 (EuroLeague 3 of 38 rounds, GBL 1 of 26),
so the projections are next-season projections. The live report holds 403 players, 103 with
`no_history`, and 3 GBL undervalued players. The gated, append-only `simulate` stays off (M7 failed),
so item 52 still holds.

## Checklist
| Items | Result | Evidence |
|---|---|---|
| 1-6 | PASS | A and C; 1,305 tests, coverage 92.44% |
| 7-8 | PASS | A; web build 2,315 pages; Elo backtests reproduce |
| 9 | PASS | B (ff44d72): 0 removed lines after merging origin/main's 2026-10-08 daily rows (A failed on them) |
| 10 | FAIL until pushed | B: 50 games, all checks 100%; the report differs from origin/main only by the cb416c6 fix (completed seasons only) |
| 11 | PASS | A; 4 games missing, 0 points mismatches |
| 12-16 | SKIPPED | unchanged (stints mart, M1, MLflow); all passed in the stopped full run (HEAD 194ecd7) |
| 17-20 | PASS | A; live M1 dry run 7 then 0 rows; build+score+publish twice unchanged; actionlint; screenshots |
| 21-29 | SKIPPED | M2, unchanged (21 and 23 PASS, 22 known-red in the stopped full run) |
| 30 | PASS | A; no-dev build + predict |
| 31 | FAIL (split artefact) | A; needs item 24's runs, skipped in this split; 14 tests passed |
| 32-34 | PASS | A and C; M3 units 93, leakage 19 |
| 35-37 | PASS | C (14fc9e7): M3 EL and GBL reports byte-identical twice and equal to the committed ones; runtime 788 s of 1,800 s |
| 38-39 | PASS | A; entity 67 tests, 81 s; crosswalk 3,189 rows |
| 40-51, 53 | SKIPPED | M4, M5, M7, unchanged |
| 52 | PASS | A; simulate dry run, workflow per gate |
| 54-55 | PASS | A; M6 units and facts 171, leakage 35 |
| 56 | PASS | A; M6 reports reproduce |
| 57 | PASS | A; project dry run identical twice, no personal fields, RUNTIME project 51 |
| 58 | PASS | A; m6 EL 178 s, GBL 70 s, project 54 s; recorded export 66, astro 69 |
| 59 | PASS | A; API tests 124, 104 stats + 966 publish files byte-identical, contract clean, compose valid |
| 60 | PASS | A; site build, M6 page screenshots 48 runs, 0 unlabelled numbers, 0 console errors, card tests |

## Subagent log
| Wave | Subagent | Deliverable | Rounds | Notes |
|---|---|---|---|---|
| 1 | A | L1 projection | 1, then D14 (round 2) and D17 (round 3) | `impact_noise`; BRAPM noise and drift from lag-1/lag-2 moments |
| 1 | B | L2 aging | 2, then D18 (round 3) | round 1's narrowed synthetic gap was rejected; root cause in D10; the regressed delta method |
| 1 | C | L3 board | 1 | |
| 1 | D | L4 similarity | 1 | |
| 1 | E | L5 API | 1 | |
| 2 | F | L6 harness | 1 | found the impact-noise gap (D14) on synthetic data |
| 2b | G | L7 leakage | 1 | no real leak; 35 tests |
| 3 | H | L10 export through the API | 1 | its one failing test came from my L9 stub; I fixed it |
| 3b | I | L11 player card and Scouting | 2 | round 1 was stopped at the usage limit before writing anything |
| 3b | J | L11 Teams, Standings, Methodology, performance, nav | 1 | I replaced its display-code copy with the tested map |

L8 (tuning, verdict, validation, test, GBL) and L9 (live) were run by me with no subagent running. I
fixed these myself:
- the `el_spm_models` `last` keyword for the live SPM;
- the CLI stub for that keyword;
- the date regex in the personal-field scan;
- three model-card numbers;
- `web_build.sh` always builds on the site.json and API fixtures, so item 60 does not depend on
  whether `publish` ran (`WEB_REAL_DATA=1` keeps the real data for a review).

## Decisions this file did not cover
D1–D25 are in the progress file. In short:
- D1: `sim-ungated` is a separate command.
- D2: M4 failed its gate; the prompt said it passed.
- D3–D7: input frame, impact inputs, translation fallback, drift on tuning only, xPTS for the board.
- D8: pytest-xdist and the fast gate.
- D9: M7 official tables re-fetched (owner).
- D10–D20: method and protocol fixes, all before the verdict.
- D21: the per-player CSVs are kept.
- D22–D25: live checkpoint, board season, the GBL undervalued rule, the committed crosswalk (owner).

Since then, recorded here:
- **D26 (`web_build.sh`):** the build always uses the `site.json` and API fixtures, so item 60 does not
  depend on whether `publish` ran. `WEB_REAL_DATA=1` keeps the real data for a review.
- **D27 (item 10, owner):** the stint validation samples completed seasons only, 2015 up to the season
  before `LIVE_SEASON`. The live season's growing game list reshuffled its pool every round. The sample
  changed once: two 2026 games were replaced, and it is still 50 games at 100%.
- **D28 (items 35/37, owner):** M3's `data_sha256` covers only the seasons up to `test[-1]`. The GBL hash
  is unchanged. The EuroLeague hash was regenerated once, and every number is identical
  (`tests/test_m3_hash.py`).

## Open questions for the owner
- **Next full pass:** run the skipped unchanged-model items (12-16, 21-29, 40-51, 53) on a quiet machine.
  M5 and M7 also hash their whole input tables (`m5_backtest.py`, `m7_backtest.py`), so items 45 and 50
  will likely show the same live-season hash drift as M3 did. The fix would be the same (D28).
- **Push:** `week-16-18` (and the earlier local merge `12a9384` on `main`) are not pushed. Merging this
  phase into `main` and pushing is your call.
- **Board and comparables:** they are rebuilt locally only (they need the shot marts). Should the
  daily run build the M2 shot marts so they move during the season?
- **M4:** it failed its gate, but its factors still feed the GBL undervalued list and the translation
  table. Both say so on the page. Keep, or hide until M4 is redone?
- **Display codes:** they now live in `publish.py` and a JSON copy the site reads, kept equal by a
  test. Should the site read them from the API instead?

## memory.md entry
- 2026-10-08 weeks 16-18 closed (split checklist record, owner's choice): M6 projections (gate PASS),
  read-only API + export through it, live `project` and `sim-ungated` in daily.yml, Scouting/Standings/
  Methodology pages, player card and Teams additions. Fixed live-season drift in item 10 (stint sample) and
  items 35/37 (M3 input hash). Not pushed; next full pass should check M5/M7 hash drift and run the
  skipped unchanged-model items.
