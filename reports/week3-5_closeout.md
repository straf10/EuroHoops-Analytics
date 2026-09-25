# Weeks 3–5 close-out (2026-09-25, branch `week-3-5`, not pushed)

Evaluation harness, odds and open spikes (`prompts/week-3-5.md`). Decisions D-a to D-h: all
defaults (user, 2026-09-25).

## Checklist (§6): one consecutive run, 2026-09-25 13:29–13:31 UTC, at `d058e25`

| # | Check | Result |
|---|---|---|
| 1 | `uv sync --frozen` | pass |
| 2 | `ruff check` | pass |
| 3 | `ruff format --check` | pass |
| 4 | `mypy src` (strict) | pass |
| 5 | `pytest` + coverage | pass: 222 tests, 98.75% |
| 6 | vulture | pass |
| 7 | web build from `tests/fixtures/site.json` | pass |
| 8 | backtests reproduce every pre-existing value | pass: all three reports byte-identical after a rerun; `tests/test_reports.py` diffs old keys against a frozen week-3 copy |
| 9 | hand-computed unit tests (ECE toy set, CRPS vs integral, rolling n = 49/50/51, 2-way de-vig) | pass |
| 10 | leakage (σ, tuning ECE bins, totals baseline) | pass: 18 tests |
| 11 | `predictions/` append-only; model versions | pass: 0 lines removed; `425e6393`, `df05260c`, `db021063` unchanged |
| 12 | `eurohoops odds` live call | **BLOCKED**: HTTP 401 `INVALID_KEY` (evidence below); key found in 0 files |
| 13 | stint validation, 50 games, reproducible | pass: second run byte-identical |
| 14 | GBL gaps filled or explained; filled points = result | pass: 101/101 of 2018-20 filled, 2 with a stated reason |
| 15 | spike and data docs end in a decision | pass |
| 16 | actionlint (both workflows) | pass: 1.7.12 |
| 17 | site at 1440/390 px, light/dark | pass: 0 px horizontal overflow in all 8 views; screenshots in `reports/screenshots/` |
| 18 | build + score + publish twice leaves the tree unchanged | pass |

The run before this one went red on item 17: a 1 px overflow on the real GBL page at 390 px,
caused by the screen-reader table in the new chart. I fixed it (`496b4f0`) and reran every
item from the top.

```text
run started 2026-09-25T13:29:34Z
d058e25 Log iteration 9
===== 1 uv sync --frozen =====
Checked 59 packages in 5ms
===== 2 ruff check =====
All checks passed!
===== 3 ruff format --check =====
67 files already formatted
===== 4 mypy src =====
Success: no issues found in 27 source files
===== 5 pytest + coverage =====
TOTAL                             1662     11    342     14    99%
Required test coverage of 85% reached. Total coverage: 98.75%
222 passed in 20.35s
===== 6 vulture =====
vulture clean
===== 7 web build from the fixture =====
npm warn allow-scripts Run `npm approve-scripts --allow-scripts-pending` to review, or `npm approve-scripts <pkg>` to allow.
16:30:29 [build] ✓ Completed in 740ms.
16:30:30 ✓ Completed in 35ms.
16:30:30 [build] ✓ Completed in 786ms.
16:30:30 [build] Complete!
===== 8 backtests reproduce every pre-existing value =====
reports byte-identical to the committed ones
5 passed in 0.05s
===== 9 hand-computed unit tests =====
============================== 9 passed in 0.71s ==============================
===== 10 leakage =====
18 passed in 1.16s
===== 11 prediction logs append-only; model versions unchanged =====
lines removed or changed vs main: 0
backtest_elo: 0.2.0+425e6393 -> 0.2.0+425e6393
backtest_elo_gbl: 0.2.0+df05260c -> 0.2.0+df05260c
backtest_elo_history: 0.2.0+db021063 -> 0.2.0+db021063
===== 12 eurohoops odds (one live call) =====
INFO eurohoops.odds: odds call: status 401, cost ?, used ?, remaining ?
ERROR eurohoops: odds: HTTP 401 INVALID_KEY
exit 1
BLOCKED: HTTP 401 INVALID_KEY (key in .env rejected)
files containing the key (repo, data, scratch logs; .env excluded): 0
===== 13 stint validation reproducible, 50 games =====
50 games: all checks 100% (five_on_court 100%, seconds 100%, minutes 100%, points 100%); wrote reports\stint_validation.json
second run byte-identical
games 50
matches the committed report
===== 14 GBL gaps: every 2018-20 game filled or explained; filled points = result =====
103 listed; 2018-20: 101/101 filled, points = result; not filled with a reason: 2
reports unchanged by build
===== 15 spike and data docs end in a decision =====
docs/spikes/oddsportal.md: ## Recommendation: **NO-GO**
docs/spikes/stints.md: ## Recommendation for M1 (weeks 5–7)
docs/data/odds.md: ## Decision
===== 16 actionlint =====
actionlint 1.7.12 clean
===== 17 website at 1440/390, light/dark (real data) =====
16:31:00 [build] ✓ Completed in 138ms.
16:31:00 ✓ Completed in 33ms.
16:31:00 [build] ✓ Completed in 381ms.
16:31:00 [build] Complete!
light 1440 euroleague horizontal overflow px: 0
light 1440 gbl horizontal overflow px: 0
light 390 euroleague horizontal overflow px: 0
light 390 gbl horizontal overflow px: 0
dark 1440 euroleague horizontal overflow px: 0
dark 1440 gbl horizontal overflow px: 0
dark 390 euroleague horizontal overflow px: 0
dark 390 gbl horizontal overflow px: 0
===== 18 build + score + publish twice leaves the tree unchanged =====
tree unchanged by the second run
 M odds/api_calls.csv
?? reports/screenshots/
ALL 18 ITEMS DONE 2026-09-25T13:31:31Z (item 12 status above)
```

The two untracked or modified paths at the end come from items 12 and 17: the calls log's row
for the 401 call, and the screenshots. Both are committed with this report.

## New metrics per competition (held-out test seasons; tuning in brackets)

| | EuroLeague live (test 2025-26) | EuroLeague history (test 2024-26) | GBL (test 2024-26) |
|---|---|---|---|
| n | 402 | 732 | 340 |
| σ (margin residual RMS, tuning only) | 11.74 | 11.85 | 12.16 |
| Log loss Elo / B0 | 0.629 / 0.656 | 0.629 / 0.660 | 0.485 / 0.668 |
| **ECE** Elo / B0 | 0.049 / 0.005 (0.051 / 0.013) | 0.046 / 0.001 (0.022 / 0.002) | 0.050 / 0.025 (0.076 / 0.021) |
| Margin MAE Elo / B0 | 9.12 / 9.59 | 9.13 / 9.67 | 9.57 / 12.43 |
| **Margin CRPS** Elo / B0 | 6.52 / 6.82 | 6.55 / 6.89 | 6.81 / 8.88 |
| **Totals MAE** (baseline) | 15.29 (13.74) | 14.59 (14.04) | 14.55 (14.47) |
| Totals baseline for 2026-27 | 170.1 | 170.1 | 165.7 |

Elo minus B0 on the test seasons, paired bootstrap (1,000 resamples), mean and 95% CI:

| | Log loss | Margin abs. error | **Margin CRPS** |
|---|---|---|---|
| EuroLeague live | −0.026 [−0.049, −0.005] | −0.47 [−0.81, −0.13] | −0.30 [−0.50, −0.11] |
| EuroLeague history | −0.031 [−0.052, −0.010] | −0.54 [−0.81, −0.26] | −0.35 [−0.52, −0.18] |
| GBL | −0.183 [−0.226, −0.138] | −2.86 [−3.75, −2.01] | −2.08 [−2.76, −1.45] |

- Elo beats B0 on every scoring rule, and every CI excludes 0.
- **ECE doesn't rank the two models, and it's lower for B0 everywhere.** B0 is almost
  constant, so all its forecasts fall in one bin at its own base rate, which makes it
  calibrated by construction. Elo's ECE of about 0.05 is small. The EuroLeague reliability
  diagram shows slight overconfidence in the tails, with small bins at both ends.
- Totals MAE is 14.5–15.3 points against a naive two-season mean. This is the bar any totals
  model has to clear. The EuroLeague 2025-26 test season scored higher than the baseline
  expected (MAE 15.3).
- Live so far: the EuroLeague has 5 scored games (Elo log loss 0.555 vs 0.675, CRPS 5.95 vs
  7.43, totals MAE 11.0); the GBL has 0. The rolling-50 series is empty for both until game 50.
  There are no warnings.

## Odds (D5, D6)
- **The Odds API is BLOCKED.** The `.env` key is rejected: HTTP 401 `INVALID_KEY` on the free
  `/events` endpoint (about 12:50 UTC) and on both checklist runs of `/odds` (13:28 and 13:30
  UTC, logged in `odds/api_calls.csv`). No response carried quota headers. The stored value
  is 30 hex characters; the provider's keys are normally 32, so it looks truncated.
- **Quota facts from the provider's docs, not yet from headers:**
  - `/odds` costs markets × regions = **3 credits a call**, so the daily run spends about 90 a
    month.
  - The free plan gives **500 credits a month**.
  - `/sports` and `/events` are free.
  - The docs confirm `/odds` also returns in-play events. The recorder drops them.
- **Built and tested on a fake transport:**
  - one call per run, raw response in the gitignored cache
  - consensus rows (median de-vigged P(home), spread, total, bookmaker counts), append-only
  - a calls log with the quota headers, and a refusal to call below 20 remaining
  - the scorecard `market` column: the latest snapshot before tip-off, with Elo and B0 scored
    on the same games
  - a daily step that's skipped with a warning when the secret is missing
- **A leak the tests caught:** httpx logs every request URL at INFO, and this URL carries the
  key. The recorder now silences `httpx`/`httpcore` for the duration of the call.
- **OddsPortal: NO-GO.** The site is geo-blocked in Greece "because of legislative
  restrictions": even `robots.txt` redirects to the notice. Two requests. Getting around a
  legal geo-block is out of bounds (`docs/spikes/oddsportal.md`).

## Stints (D7)
- **Sample** (seed 20260925, 50 games, 2015-16 → 2026-27): **100%** on all four checks: five on
  court, stint seconds, player minutes ±60 s, and points = final.
- **Context over every cached game:**

  | Seasons | Pass all four |
  |---|---|
  | 2015-26 | 99.0% (3,266/3,300) |
  | 2011-15 | 95.9% |
  | 2007-11 | **3.0%** |

- **Main failure pattern:** before 2011-12, 97.8% of substitution clocks read `mm:00` (whole
  minutes only), so player minutes miss by up to a minute per change. From 2011 the failures
  are rare missing or duplicate `IN`/`OUT` rows.
- **Found on the way:** the pre-2015 scoring codes `LAYUPMD` and `DUNK`, and a phantom
  `Extra1: 0` overtime in 13 2015-16 box scores.
- **Recommendation:** build the M1 stints mart from 2011-12 onward, gate each game on the four
  checks, and don't impute (`docs/spikes/stints.md`).

## GBL play-by-play and the 2018-20 fill (D8)
- **Backfill:** 342 games of 2018-19 and 2019-20 in 682 requests at ≥2 s (about 37 min),
  cached once.
- **Validation against ESAKE:**
  - PBP points equal the result in **342/342** games.
  - Shooting lines match in **5,454/5,457**.
  - Every player's seconds match exactly for **486/505** teams.
- **Fill** (`source = "pbp"`, never counted as an official box):
  - 78 missing boxes get PBP team totals (156 teams, all equal to the result).
  - 23 short boxes get the one missing player's line (2–23 points), and all now equal the
    result.
  - The 2 short 2022-23 boxes aren't filled, with a stated reason (outside the 2018-20 scope).
- **Official pass rates are unchanged** (computed on ESAKE rows only). The fill is reported
  separately in `reports/gbl_box_invariants.json` → `pbp_fill` and `reports/gbl_box_gaps.csv`.
- **CI:** a `web` job builds the site (Node 24) from `tests/fixtures/site.json`, and a test
  regenerates that fixture from `site_data` on synthetic games.

## Decisions I made that the task file didn't cover
1. **σ is the residual RMS** (mean zero, ddof 0) around Elo's expected margin on the tuning
   seasons. It's the Normal MLE for a forecast centred on that margin.
2. **The totals baseline is null when either previous season is missing** (no partial
   average). It covers the scored seasons plus the next one, which is the live season.
3. **ECE and reliability use equal-width bins** that are left-closed, with p = 1 in the last
   bin. Empty bins are listed with null values.
4. **The leakage check is split-level:**
   - everything fitted on the tuning seasons (σ, B0, Elo parameters, the tuning split's ECE
     bins, totals MAE) is invariant to deleting, altering or adding games in the test seasons
   - the totals baseline for season *s* is invariant to any change in *s* or later
   - a guard test proves the same edits do move the test numbers
5. **Rolling window:** it scores the provable headline rows in tip-off order. Each point is
   labelled with the last game in its window.
6. **Odds consensus:**
   - Only exact two-way moneylines are used; three-way (draw) markets are skipped.
   - The spread keeps the bookmakers' sign (negative = home favoured).
   - Games are matched to the schedule within 48 h.
   - Rows carry per-market bookmaker counts, which answer the coverage question.
   - Calls get no retries (every retry is billed).
   - `odds/api_calls.csv` is committed. It holds no prices or keys.
7. **`odds/euroleague_teams.csv`:** 30 candidate API names, all `verified=no` until a real call.
   Unmatched names are reported, never fuzzy-matched.
8. **OddsPortal:** I read neither the ToS nor robots.txt from another vantage point, because
   the legal geo-block already decides it.
9. **Stints:**
   - Starters come from the box score, and lineups carry over between periods.
   - The overtime count comes from the box score, counting only scored periods.
   - Empty placeholder games count as failures.
   - I added a `stints` CLI command so the report is regenerated by the pipeline.
10. **GBL fill:**
    - Missing players are matched by an exact counting-stat signature (minutes excluded).
    - A team's PBP line is only added when it closes the gap exactly.
    - PBP player ids are `pbp:<jersey>:<name>`, pending entity resolution.
    - `--pbp` never narrows the staging seasons. This avoids the known `ingest` trap of
      overwriting staging with only the chosen seasons.
    - `build` refreshes the gaps CSV only if it exists.
11. **Site:**
    - The reliability diagram shows the held-out test split; the live season is too small to
      bin.
    - The market column isn't shown on the page (D4 doesn't ask for it).
12. **Dependencies:** `openpyxl` (runtime, D-g) and `types-openpyxl` (dev, needed for mypy
    strict). The screenshots and the actionlint binary are fetched on demand, not added as
    dependencies: Playwright runs via `uv run --with playwright`, and the actionlint download
    was checksum-verified.
13. **Screenshots** are committed as the section crops (8 PNGs, 60–90 KB each). Full-page
    shots stayed local.

## Open questions and actions for the user
1. **Fix the Odds API key.**
   - Replace the 30-character value in `.env` with the full key, then run
     `uv run eurohoops odds` once.
   - That call verifies the quota, the cost per call and the team names. Fix any unmatched
     names in `odds/euroleague_teams.csv` and flip them to `verified=yes`.
   - Then add the `ODDS_API_KEY` repository secret. Until then the daily step prints a warning
     and records nothing.
2. **PLAN §1.3 may be out of date.** It says historical odds are paid, but The Odds API's
   pricing page lists "Historical Odds" on the free plan. Unverified; one `/historical` call
   would settle it. I didn't change PLAN.md.
3. I stopped the stale `astro preview` server (PIDs 8200/10892) with your permission.
   Restart it with `npx astro preview --port 4321` in `web/` if you want it.
4. **Cosmetic:** on the EuroLeague reliability diagram, the "58" and "85" bin labels touch
   neighbouring dots in the crowded middle. The screen-reader table has the exact numbers.
   It's a small nudge if you want it fixed.
5. **Watch on the first CI push:** the new `web` job and the fixture test, which compares
   floating-point output generated on Windows with Linux. I don't expect a difference, since
   values are rounded to 4–6 decimals.
