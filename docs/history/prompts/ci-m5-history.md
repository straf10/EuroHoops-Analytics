# Agent Task: EuroHoops: run live M5 in the daily CI (EuroLeague history in the CI marts + stints mart), then monitor it

## Role & goal
You are the **orchestrator** (Opus). You may delegate self-contained subtasks to Sonnet subagents,
but you review their work, run every real command yourself and own the result.

**Goal:** make `eurohoops predict` in `.github/workflows/daily.yml` write the live M5 log
(`predictions/euroleague_m5_2026-27.csv`) every day, before tip-off, from the same inputs the local
run uses. Then watch the next scheduled runs and confirm that it happened.

Read first: `CLAUDE.md`, `memory.md` (entries 2026-09-29 and later), `docs/models/m5.md`,
`src/eurohoops/live_m5.py`, the M5 block of `predict` and `_m5_inputs` / `_m5_player_part_el`
in `src/eurohoops/cli.py`, `.github/workflows/daily.yml`, `.github/workflows/seed-el-history.yml`,
`src/eurohoops/config.py` (`default_seasons`).

## What is already true (verify, do not redo)
- The EuroLeague raw cache for 2007-22 was seeded into the Actions cache on 2026-09-29 (release
  `data-el-raw-2007-2022`, workflow `seed-el-history.yml`, run succeeded). The daily cache
  `euroleague-raw-*` is about 69 MB, so the raw history (box, PBP, points, schedule) is in CI.
- What is missing in CI: (1) the **games/team_games marts only hold the default seasons** (2023 on)
  because `eurohoops ingest` only stages `default_seasons`; (2) **no stints mart**
  (`eurohoops stints --mart` is local-only). So the M5 block raises `typer.Exit` and logs
  "M5 skipped" (the shadow model never blocks Elo/M1).
- Locally the full build + forecast takes ~112 s; it only runs when there are un-logged upcoming games.

## Work
1. **Find exactly what M5 needs in the marts** (seasons from `M5`'s spec warm-up to now, both
   competitions for rest, `team_games`, `stints`, `stint_game_checks`, M1/M3/M5 reports) and
   which of it CI lacks. Check the seeded raw files are enough to build them **without any network
   fetch** (esake.gr and the EuroLeague API: cache-only; never re-fetch what is cached).
2. **Add the steps** to `daily.yml` (between "Build marts" and "Predict upcoming games") that stage
   the history seasons from the raw cache and build the stints mart. Prefer a cache-only flag or
   season list over changing `default_seasons` globally. Keep total daily runtime reasonable and
   record the measured step times.
3. **Do not change any other live log.** Elo and M1 live forecasts must be **identical** with and
   without the extra history in the marts (their warm-up windows are fixed by their reports). Prove
   it: run `predict` for Elo/M1 into scratch log paths against marts built both ways and diff them;
   add a test if a cheap one exists. If they differ, stop and report — do not ship.
4. **M5 parity:** M5's forecasts built from the CI-style marts must equal the local ones (same
   scratch-log diff). Never write to `predictions/` during development (append-only, pre-registered).
5. Failure must stay soft: if the history steps fail, the M5 block skips with a warning and the
   Elo, M1, odds and injuries steps still run and commit.
6. Update `tests/test_workflow.py` for new steps, `scripts/vulture_whitelist.py` if needed, and
   `docs/models/m5.md` (live status: "logged daily in CI from <date>").
7. Run the full checklist from `CLAUDE.md` ("Before commit"). Commit on a branch
   (`ci-m5-history`), no AI attribution lines; open a PR; merge only after the owner OKs it.

## Monitor (after merge)
- Trigger one `workflow_dispatch` run of `daily.yml` if the owner agrees, else wait for the cron.
  Use `gh run watch` / `gh run view --log` to check: the history and stints steps ran, the log
  shows `N M5 predictions appended`, the commit contains `predictions/euroleague_m5_2026-27.csv`,
  and runtime is acceptable.
- Note: scheduled runs have been starting late (cron 08:00 UTC, observed starts 13:47-16:54 UTC).
  Check that games tipping off the same evening are still logged before tip-off (`refuse_late`);
  if runs start too late, report it with the observed start times and propose a fix (an earlier
  cron, or a second run), do not just change it.
- Watch two consecutive scheduled runs, then report.

## Report
Files changed; steps added and their runtimes; Elo/M1 identity proof; M5 parity proof; the run
links and what each logged; any games missed by M5 and why; anything the owner must decide.
Append a session entry to `memory.md`.
