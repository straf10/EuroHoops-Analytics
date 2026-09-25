# Agent Task: EuroHoops Analytics, Weeks 3–5: Evaluation Harness, Odds and Open Spikes

## Role & goal
You are finishing the **weeks 3–5 deliverables** of `PLAN.md` (read §1.3, §5 "Evaluation protocol", §6, §7, §8 rows "2–3" and "3–5", §12) plus the open items from `reports/week3_closeout.md`. You build on the finished weeks 0–3 code (`prompts/week-0-1.md`, `prompts/week-2-3.md`, `docs/spikes/gbl-pbp.md`, `DESIGN.md`).

**Goal, in one sentence:** every forecast the system makes is judged by calibration, a full predictive distribution and a rolling live window. EuroLeague forecasts also get a forward-recorded market benchmark. The two open data spikes (OddsPortal, EL stints) end in a written decision, the GBL play-by-play ingester fills the 2018-20 box-score gaps, and CI catches a broken website before the daily run does.

**Exit gate (PLAN §8, row 3–5):** the harness reproduces the current Elo numbers exactly, leakage tests run in CI, and every check in §6 below passes in **one consecutive run**.

## How you work: the loop
Repeat until the stop condition holds. Do not stop earlier and do not declare success on a partial run.

1. Pick the next unfinished deliverable (§4, in order D1 → D8).
2. Implement it, with tests.
3. Run that deliverable's own checks (§6), then the **whole** checklist.
4. Anything red → find the root cause, fix, go back to 3. Never weaken, skip, `xfail` or delete a check to turn it green. Never raise tolerances you did not justify in writing.
5. Green → commit (small commit, attribution footer from your environment) and append one line to `reports/week3-5_progress.md`: `iteration N | deliverable | checks run | result | commit`.
6. **Stop condition:** all deliverables done **and** the full §6 checklist passes top to bottom in a single run with no edits in between. Paste that run's output into the final report.

Guards:
- **Same failure three times** (same check, same error) → stop patching. Write the diagnosis in the progress file, re-read the relevant code and data, then fix the cause.
- **External blocker** (API down or refusing, ToS or robots.txt forbidding, quota exhausted) → mark the check `BLOCKED`, save the evidence (status code, headers, quoted ToS clause) in the progress file, and continue with the other deliverables. A blocked check is never reported as passed.
- **Resumability:** the progress file plus the commits are your memory. A usage-limit account switch (`cswap auto`, threshold 90%) or a restart can interrupt you at any time. After any interruption, read `reports/week3-5_progress.md` and `git log` first, then continue from the first unfinished step.

## 0. Decisions (defaults in bold; confirm with the user before starting, then record the answers here)
| # | Question | Default |
|---|---|---|
| D-a | Odds source for the forward recorder | **The Odds API, `basketball_euroleague`, markets `h2h,spreads,totals`, region `eu`, decimal odds. GBL gets no market benchmark (not listed)** |
| D-b | What gets committed from odds | **Only a derived, per-game consensus row (de-vigged median implied P(home), median spread, median total, bookmaker count, fetched_at). Raw API responses stay in the gitignored cache. No per-bookmaker prices in git** |
| D-c | Odds in CI | **Daily workflow step, skipped with a warning when the `ODDS_API_KEY` secret is absent. The agent does NOT set the GitHub secret; the user does** |
| D-d | Predictive distribution for CRPS | **Margin ~ Normal(exp_margin, σ), σ fitted on the tuning seasons only (residual SD). Closed-form Gaussian CRPS** |
| D-e | Totals | **A baseline totals predictor (mean total of the competition's previous 2 seasons, frozen per season) scored by totals MAE. No totals model this phase; it only makes the metric exist** |
| D-f | Reliability diagram output | **Bins in the JSON reports and drawn on the website (Astro, inline SVG, DESIGN.md rules). No new Python plotting dependency** |
| D-g | GBL PBP export format | **BasketHotel `export/view/play_by_play` (xlsx) → `openpyxl` as the one new runtime dependency** |
| D-h | Stints | **A validation spike: `parse/stints.py` plus a report. The stints mart itself is M1 (weeks 5–7)** |

**Answered by the user, 2026-09-25: all defaults (D-a to D-h).**

## 1. Hard constraints (all weeks 0–3 constraints still apply)
- **Scope = this file only.** No M1 team model, no xPTS, no RAPM, no MLflow, no FastAPI, no betting presentation. Build nothing "for later" beyond what D-h names.
- **The live logs are sacred.** Existing rows in `predictions/*_2026-27.csv` stay byte-identical. The Elo `model_version` of both competitions must not change. `git diff main -- predictions/` shows only appended lines.
- **Reproduce before extending:** after D1, `eurohoops backtest` for both competitions must reproduce every existing number in `reports/backtest_elo*.json` exactly. New fields are additions; no existing value moves.
- **No dead code** (weeks 0–1 definition). Code ahead of its pipeline step goes in `vulture_whitelist.py` with a one-line reason.
- **Politeness:** ESAKE and BasketHotel ≥ 2 s between requests, identifying User-Agent, gzip raw cache, never re-fetch what is cached. The Odds API: count every call. Read `x-requests-remaining` / `x-requests-used` from every response, log them, and refuse to call when remaining < 20.
- **Secrets:** `ODDS_API_KEY` is read from the environment (`.env` locally). Never print it, log it, put it in a URL you log, or commit it. Tests use a fake transport, never the real key.
- **Data rights (PLAN §9):** no raw HTML/JSON/xlsx in git; only small trimmed fixtures.
- **Website:** new sections follow `DESIGN.md` (three marker inks, flat colour, no eyebrows, reduced motion honoured). The page stays a model benchmark, not betting advice.
- **Do not change any value in `PLAN.md`.** Report factual errors instead.
- **Git:** branch `week-3-5`, small commits. **Do not push.**

## 2. Verified facts (do not re-discover; do verify in tests)
- Metrics live in `src/eurohoops/eval/metrics.py` (`Metrics`, `per_game_log_loss`, `score`, `paired_bootstrap_ci`). The backtest is `eval/backtest.py`, the live scorecard `eval/scorecard.py`. Reports: `reports/backtest_elo.json`, `backtest_elo_gbl.json`, `backtest_elo_history.json`, `live_scorecard*.json`.
- `eurohoops publish` writes `web/src/data/site.json` (`publish.site_data`). The daily workflow then runs `npm ci && npm run build` in `web/` into `site/`.
- EuroLeague PBP (`PlaybyPlay`) is cached locally for 2007-2026 via `ingest --details` and contains `IN`/`OUT` substitution rows.
- GBL PBP (see `docs/spikes/gbl-pbp.md`): `widget-service/show` with `show_export_link=1` gives the internal id, then `export/view/play_by_play` returns xlsx. The export resolved for all 103 checked games, and each export's final score matched. No shot coordinates.
- GBL box-score gaps: 78 games missing, 25 with a points mismatch (`reports/gbl_box_gaps.csv`). The user decided (2026-09-25) to fill 2018-20 from PBP. Box-score models start at 2020-21 until then.
- The Odds API lists `basketball_euroleague` but not the GBL (checked 2026-09-24). Free-tier quota is **unverified**: verify from the response headers of your first call and record it in `docs/data/odds.md`.

## 3. Unverified (verify, then record in `docs/` and a test)
- The Odds API team names versus EuroLeague team codes (build an explicit mapping table; unmatched events are reported, never guessed).
- Whether spreads and totals are offered for every EuroLeague game, and at which lead time.
- The OddsPortal robots.txt and ToS position on automated access.
- Whether every GBL PBP export row carries a period and a clock, and how substitutions and overtime are written.

## 4. Deliverables

### D1: Calibration (ECE + reliability)
- `metrics.py`: `ece(p, y, bins=10)` (equal-width bins, weighted by count) and `reliability(p, y, bins=10)` → per-bin mean predicted, observed rate, n.
- Add both to the backtest report (tuning and test, Elo and B0) and to the live scorecards. Existing fields unchanged.

### D2: Predictive distribution (CRPS) and totals
- σ per competition fitted on tuning seasons only and stored in the backtest report. `crps_normal(mu, sigma, y)` in closed form. Margin CRPS for Elo and B0 (B0 uses the home-margin constant and the same σ).
- The totals baseline per D-e and totals MAE, reported per competition.

### D3: Rolling live monitoring
- Rolling 50-game log loss (Elo and B0) over scored live rows, in the scorecard JSON as a series.
- A `warnings` list in the scorecard: "Elo worse than B0 over the last 50 scored games" (only when n ≥ 50) and "a played game has no result 48 h after tip-off" (data freshness).

### D4: Website: model performance
- Using `site.json` only: a reliability diagram (Elo vs the diagonal, bin sizes visible), the rolling-50 line once n ≥ 50 (an honest empty state before), and ECE and CRPS beside the existing scorecard numbers. Follow DESIGN.md and add any new rule to it.

### D5: Forward odds recorder (EuroLeague)
- `eurohoops odds`: one API call for upcoming EL games, raw response cached (gitignored), consensus rows per D-b appended to `odds/euroleague_2026-27.csv`. The file is append-only; a game can gain rows over time (new snapshots) but old rows never change.
- Scorecard: when a scored game has an odds row fetched before tip-off, add a market column (log loss of de-vigged P(home)), with n shown. Games without odds are excluded from that column only.
- Daily workflow step per D-c; `tests/test_workflow.py` asserts it and its skip-without-secret behaviour. Record quota and cost per call in `docs/data/odds.md`.

### D6: OddsPortal feasibility spike (timebox: one working day)
- Read robots.txt and ToS **first**. If automated access is disallowed, stop there: the report is the decision.
- Otherwise try to extract closing EuroLeague moneyline/spread/total for one past round by hand-level request volume (≤ 30 requests, ≥ 3 s apart).
- Output `docs/spikes/oddsportal.md`: what worked, what blocked, volume/cost estimate, and a go/no-go recommendation. **No production scraper** in this phase.

### D7: EuroLeague stint validation (PLAN §12 item 6)
- `parse/stints.py`: reconstruct the five players on court per team from cached PBP (`IN`/`OUT`), split at every substitution and period boundary.
- Validate a deterministic sample of **50 games** (seeded, spread across 2015-2026). Per game, check: exactly 5 players per team at all times; stint seconds sum to 2400 (+300 per OT) per team; per-player minutes from stints match box-score minutes within ±60 s; points scored during stints sum to the final score.
- `reports/stint_validation.json` with the pass rate per check and a list of failing games with the reason. Write `docs/spikes/stints.md` with the failure patterns and a recommendation for M1. The pass rate is a finding, not a gate: report it honestly, even if it is low.

### D8: GBL PBP ingester + 2018-20 box fill, and CI web build
- `eurohoops ingest --competition gbl --pbp`: the two requests per game (see §2), gzip cache, local backfill only (never in CI). Parse to a typed PBP table with a pandera contract.
- Rebuild box lines from PBP for the 2018-19 and 2019-20 games listed in `reports/gbl_box_gaps.csv`. Mark them `source = "pbp"` and keep them out of any "official box" claim. Refresh the gaps report: every listed game is filled or carries a stated reason.
- `ci.yml`: a `web` job (Node 24, `npm ci`, `npm run build`) using a committed fixture `tests/fixtures/site.json`. A pytest regenerates that fixture from `site_data` on synthetic games and fails if the committed file differs, so the fixture can never drift from the real schema.

## 5. Out of scope (do not build)
M1 team model, stints mart, player tables, entity resolution, MLflow, FastAPI, Docker, GBL odds, betting language or odds display beyond the benchmark column.

## 6. Verification checklist (run top to bottom; all must pass in one run)
Local gate, identical to `.github/workflows/ci.yml`:
1. `uv sync --frozen`
2. `uv run ruff check .`
3. `uv run ruff format --check .`
4. `uv run mypy src`
5. `uv run pytest -q --cov=eurohoops --cov-fail-under=85`
6. `uv run vulture src vulture_whitelist.py --min-confidence 60`
7. `cd web && npm ci && npm run build` (with the fixture copied to `web/src/data/site.json`).

Correctness:
8. `uv run eurohoops backtest` and `--competition gbl` reproduce every pre-existing value in the three backtest reports (a test diffs old vs new JSON on the old keys).
9. Unit tests with hand-computed expectations: ECE on a 4-game toy set; CRPS against a numerical integral within 1e-6; the rolling window at n = 49, 50, 51; de-vig on a 2-way market.
10. Leakage (PLAN §5): σ, the totals baseline and the ECE bins for season *s* are unchanged when any game of season *s* or later is added, deleted or altered. Extend `tests/test_leakage.py`.
11. `git diff main -- predictions/` shows only added lines; both `model_version` values are unchanged.
12. `uv run eurohoops odds` makes **one** live call that returns 200, logs the remaining quota and writes ≥ 1 consensus row (or is `BLOCKED` with evidence). Tests use a fake transport. `grep -r "$ODDS_API_KEY"` over the repo and logs finds nothing.
13. `reports/stint_validation.json` exists, covers exactly 50 games, and is reproducible (same seed, same output twice).
14. `reports/gbl_box_gaps.csv` refreshed: no 2018-20 game left without a fill or a stated reason. PBP-derived team points equal the results-page score for every filled game.
15. `docs/spikes/oddsportal.md`, `docs/spikes/stints.md` and `docs/data/odds.md` exist and each ends in a decision or recommendation.
16. `actionlint` passes on both workflows.
17. The website at 1440 px and 390 px (light and dark) shows the new sections with no horizontal page scroll. Attach the screenshots.
18. `uv run eurohoops build && uv run eurohoops score && uv run eurohoops publish`, run twice, leave the tree unchanged the second time.

## 7. Final report (`reports/week3-5_closeout.md`)
Checklist with pass/BLOCKED per item; the single consecutive §6 run output; new metrics per competition (ECE, CRPS, totals MAE, Elo vs B0 with CIs); odds quota facts; the stint pass rate and main failure pattern; the OddsPortal decision; the GBL fill results; every decision you had to make that this file did not cover; open questions for the user. Then append a session entry to `memory.md`.
