# Agent Task — EuroHoops Analytics, Weeks 0–1: Bootstrap + Live Elo Prediction Log

## Role & goal
You are implementing the **Weeks 0–1 deliverables** of `PLAN.md` (read §1.1, §3, §5 "Evaluation protocol", §5.1, §7, §8, §12 before starting).

**Goal:** a clean, tested Python project that ingests EuroLeague data, fits a 538-style Elo baseline, backtests it without leakage, and **appends pre-tip-off predictions for upcoming 2026-27 EuroLeague games to a public, append-only log**, run daily by GitHub Actions. Plus a written spike report on GBL play-by-play.

You must **iterate** — implement → run the full verification checklist (§6) → fix → rerun — until **every check passes in one consecutive run**. Do not declare completion otherwise.

---

## 1. Hard constraints
- **Scope = weeks 0–1 only.** No GBL scraper, no website, no MLflow, no FastAPI, no other models. Build nothing "for later".
- **No dead code.** Every function, class, module, CLI command, config key and dependency must be used by the CLI or by production code paths. Tests alone do not count as "used". No commented-out code, no `TODO`/`FIXME`, no `print` debugging, no unused parameters, no speculative abstractions (a base class with one implementation, plugin registries, etc.).
- **Efficiency:**
  - Elo fit/backtest = a single chronological O(N) pass per parameter set; no per-game DataFrame copies; no quadratic loops. The full backtest including the tuning grid must run in **< 10 s** locally.
  - Network: raw responses are cached, gzip-compressed, under `data/raw/`. Completed games and past seasons are **never re-downloaded**. Only the current season's schedule is refreshed. Throttle to ≥ 0.5 s between requests and use exponential-backoff retries (max 5).
  - The daily CI job must be **stateless and cheap**: it rebuilds Elo from schedule/result data only (a handful of requests), with no play-by-play/shot downloads in CI.
- **Do not change any value in `PLAN.md`.** If you find a factual error in it, note it in your final report instead.
- **Git:** make small, logical local commits using the attribution footer from your environment. **Do not push.** The user pushes.
- **Secrets:** none are needed. Never add API keys.

## 2. Verified facts about the data (do not re-discover; do verify in tests)
- Schedule/results: `GET https://api-live.euroleague.net/v2/competitions/E/seasons/E{YYYY}/games` → `{"data": [...]}`. Useful fields: `gameCode`, `identifier`, `phaseType.code` ∈ {`RS`,`PI`,`PO`,`FF`}, `round`, `played`, `gameStatus`, `confirmedDate`, `isNeutralVenue`, `utcDate` (**ISO with `Z` — the only tip-off time to use**), `local`/`road` (club code + score).
  - ⚠️ `date` is **not UTC** (verified: `date` 20:00 vs `utcDate` 18:00Z). Using `date` is a bug.
  - `isNeutralVenue=True` (Final Four) → no home-court advantage.
- Detail endpoints (2007-08 onward; 2000–2006 return empty bodies): `https://live.euroleague.net/api/{Boxscore|PlaybyPlay|Points}?gamecode={n}&seasoncode=E{YYYY}`.
- 2026-27 season: 20 teams, 38 regular-season rounds, 380 RS games, started 2026-09-24.
- Season code `E2025` = the 2025-26 season.

## 3. Deliverables

### D1 — Project scaffold
- `uv`-managed project, Python 3.12, package `eurohoops` under `src/`, console script `eurohoops` (Typer).
- Dev tools: `ruff` (lint + format), `mypy --strict` on `src/`, `pytest` + `pytest-cov`, `vulture`, `pre-commit` (ruff, ruff-format, mypy).
- `.gitignore`: keep the existing `memory.md` entry; add Python/uv/cache entries and `data/`.
- `README.md`: 1-paragraph pitch, how to run in ≤ 3 commands, link to `PLAN.md`. Keep it short.
- CI workflow `ci.yml`: on push/PR → lint, format check, mypy, tests, vulture.

### D2 — EuroLeague ingestion (`eurohoops ingest`)
- Seasons 2023, 2024, 2025, 2026 (option `--seasons`).
- Schedule/results for all seasons. Raw detail JSON (box score, PBP, shots) for **completed** games only, behind the `--details` flag (local backfill for later weeks; not run in CI).
- Immutable, gzip raw cache with a path convention that makes "already have it" a file-existence check (no index file needed). Log cache hits/misses as a count summary.
- Parse schedules into a typed `games` table (Parquet under `data/staging/`) validated by a **pandera** schema: unique `(season, game_code)`, `tipoff_utc` tz-aware UTC, scores non-null iff `played`, home ≠ away, phase in the allowed set.

### D3 — Elo model M0 (`src/eurohoops/models/elo.py`)
- FiveThirtyEight-style: win prob `1 / (1 + 10^(-(diff)/400))` with diff = home − away + HCA (0 if neutral venue); MOV multiplier `((MOV+3)^0.8) / (7.5 + 0.006·EloDiff_winner)`; season-start reversion toward the mean (tunable weight); initial rating 1500.
- Expected margin = `elo_diff / s`, where `s` is fitted by least squares on tuning data (not hard-coded).
- Parameters to tune: `K`, `HCA`, reversion weight. Use a small grid (≤ 60 combos).

### D4 — Backtest (`eurohoops backtest`)
- Walk-forward, chronological by `tipoff_utc`: each prediction uses only games that tipped off before it.
- Splits: 2023-24 = warm-up (no scoring), **2024-25 = tuning**, **2025-26 = test (scored once with the tuned params)**.
- Metrics per split: log loss (primary), Brier, accuracy, margin MAE, n games; for both **Elo** and **B0** (constant home-win rate learned from warm-up + tuning seasons, neutral games = 0.5).
- Paired bootstrap (1,000 resamples, fixed seed) 95% CI of the log-loss difference Elo − B0 on the test split.
- Write `reports/backtest_elo.json` (committed) and print a compact table.

### D5 — Live prediction log (`eurohoops predict`)
- For every 2026-27 game with `played=False`, `confirmedDate=True` and `tipoff_utc` within the next 36 h (window configurable), append one row to `predictions/euroleague_2026-27.csv`:
  `game_id, season, round, phase, tipoff_utc, home, away, p_home, exp_margin, model, model_version, predicted_at_utc`.
- `model_version` = short hash of the tuned parameters plus the code version, so any row is traceable.
- **Append-only & idempotent:** never modify or delete existing rows. Skip a game if a row with the same `(game_id, model_version)` exists. Refuse (and exit non-zero) if `predicted_at_utc >= tipoff_utc`.
- Ratings for live predictions = Elo replayed over all completed games up to now with the tuned params.

### D6 — Live scorecard (`eurohoops score`)
- Joins the log with results and writes `reports/live_scorecard.json`: n, log loss, Brier, accuracy, margin MAE for Elo vs B0. Only rows where `predicted_at_utc < tipoff_utc` are counted. Handles zero completed games gracefully.

### D7 — Daily GitHub Actions workflow `daily.yml`
- Cron `0 8 * * *` + `workflow_dispatch`. Steps: setup uv (with cache) → `eurohoops ingest` (schedules only) → `eurohoops predict` → `eurohoops score` → commit `predictions/` and `reports/` only if they changed, as `github-actions[bot]`. `permissions: contents: write`. Concurrency group so two runs never overlap.

### D8 — GBL play-by-play spike (report only, no production code)
- `docs/spikes/gbl-pbp.md`: how ESAKE's PBP is served (BasketHotel widget: ESAKE hex `idgame` → decimal, `GAME_FULL_VIEW_WIDGET`, `widget-service/show`, `league_id`/`season_id` from `MBT.Integration`), whether it can be fetched with plain HTTP or needs Playwright, and **which fields exist: substitutions? game clock? score? shot coordinates?** Use the finished 2025-26 game `idgame=D6867DA7` (Kolossos 85–87 AEK). Include a trimmed raw sample in `docs/spikes/samples/`. End with a recommendation for weeks 2–3.
- Throwaway exploration scripts must not be committed.

### D9 — Session log
- Append a concise entry to `memory.md` (gitignored): Implemented / Results (key metrics) / Expected vs. went wrong / Next.

## 4. Required tests (no network in tests — use recorded fixtures in `tests/fixtures/`, trimmed to what the tests need)
1. **Parser:** a recorded schedule fixture parses to the expected rows; `utcDate` is used (a test that fails if `date` were used); neutral-venue flag respected.
2. **Contracts:** the pandera schema rejects duplicates, naive datetimes, a played game without scores.
3. **Elo math:** hand-computed expected values for one game (prob, MOV multiplier, rating update); ratings are zero-sum per game; the reversion step is correct.
4. **Leakage:** predictions for games before time T are **bit-identical** when every game after T is deleted, altered (scores flipped) or duplicated.
5. **Log integrity:** a second `predict` run adds zero rows; an existing row is never changed (compare bytes of the pre-existing prefix); the `predicted_at >= tipoff` case exits non-zero.
6. **Cache:** a second ingest with the same inputs performs zero HTTP calls (mock the transport; assert the call count).
7. **Scorecard:** a tiny synthetic log gives hand-computed metrics; late rows are excluded.
- Coverage ≥ 85% on `src/eurohoops` (the network transport and CLI glue may be excluded only via explicit `# pragma: no cover` with a one-line reason).

## 5. Definition of done — results sanity gates
- The Elo test-split (2025-26) log loss is **lower than B0's**. If it is not, stop and investigate (data bugs are more likely than "Elo doesn't work") and report the findings; do not tune on the test split.
- The Elo test-split accuracy is in a plausible range (≈ 0.60–0.72). > 0.75 ⇒ assume leakage and investigate.
- Tuned HCA is positive and plausible (EuroLeague home advantage is sizeable; report the value and the implied home win % at equal ratings).
- 2025-26 game count matches the API (402 games incl. PI/PO/FF, verified 2026-09-24), minus unplayed ones.

## 6. Verification checklist — run ALL, in order, from a clean shell; iterate until all pass in one run
```bash
uv sync --frozen                                   # 1  lockfile consistent
uv run ruff check . && uv run ruff format --check .  # 2  lint + format
uv run mypy src                                    # 3  strict typing, 0 errors
uv run pytest -q --cov=eurohoops --cov-fail-under=85  # 4  all tests, coverage
uv run vulture src --min-confidence 60             # 5  0 findings (no whitelist file)
uv run eurohoops ingest --seasons 2023 2024 2025 2026 --details   # 6  first run: fetches
uv run eurohoops ingest --seasons 2023 2024 2025 2026 --details   # 7  second run: 0 downloads except the 2026 schedule
uv run eurohoops backtest                          # 8  gates in §5 hold; reports/backtest_elo.json written
uv run eurohoops predict && uv run eurohoops predict   # 9  second run appends 0 rows
uv run eurohoops score                             # 10 reports/live_scorecard.json written
git status --porcelain                             # 11 no stray files; data/ and memory.md not tracked
```
Additional manual checks (report the evidence for each):
- 12. `grep -rnE "TODO|FIXME|print\(|breakpoint\(" src` → nothing (logging only).
- 13. Every dependency in `pyproject.toml` is imported somewhere in `src/` or used by a tool config (list each with where it is used).
- 14. Every CLI command is exercised in checks 6–10; every public function is reachable from a CLI command (state the call path for each module).
- 15. Workflows lint cleanly with `actionlint` if available (otherwise validate the YAML and explain why actionlint was skipped).
- 16. Time the backtest (`time uv run eurohoops backtest`) → < 10 s. State the complexity of the Elo pass and the grid search.
- 17. Every row in `predictions/*.csv` satisfies `predicted_at_utc < tipoff_utc` and `0 < p_home < 1`.

## 7. Final report (your last message)
- A table of checks 1–17 with ✅/❌ and one line of evidence each (key output lines, not full logs).
- Backtest table (Elo vs B0, tuning and test) + bootstrap CI + tuned params.
- The number of live predictions logged and the next game's prediction.
- GBL spike conclusion in 3 lines.
- Deviations from this prompt or `PLAN.md`, with reasons. Any known limitations.
- What the user must do to go live: `git push`, and enable Actions write permission if the repo settings restrict it.
