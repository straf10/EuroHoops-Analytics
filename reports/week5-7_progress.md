# Weeks 5–7 progress log

Decisions (user, 2026-09-25): E-a … E-g all defaults (see `prompts/week-5-7.md` §0).
Branch `week-5-7` (from `main` at 55f0791), not pushed.

Format: `iteration N | deliverable | checks run | result | commit`

## Baseline (before any change, 2026-09-25)
Local gate 1-6 green on the branch start (224 tests, coverage 98.75%).

## Facts verified at start (recorded in docs/data/possessions.md with E1)
- EL box JSON: every season 2007-2026, `totr` (team totals) = sum of player lines + `tmr`
  (team row: team rebounds and team turnovers) for FGA2/FGA3/FTA/OREB/DREB/TOV/points, in
  every cached box score (5,008 games; E2018_21 has no players).
- ESAKE box HTML: `ΣΥΝΟΛΟ` = players + `ΟΜΑΔΙΚΑ - ΠΑΓΚΟΣ` row for REBS/D.REBS/O.REBS/TO, every
  cached game 2018-2025; the team row is always 0 in 2018-19..2020-21 and mostly non-zero from
  2021-22 on (ESAKE started recording team rebounds/turnovers during 2021-22).
- GBL PBP (2018-20) has rebound (player + team) and turnover templates.

## Decisions not covered by the task file
- (iteration 1) **Checklist item 7 runs in a clean copy of `web/`** (scratchpad): the user has
  `astro preview --port 4321` and `astro dev` running (started 16:37 and 17:03 local), which
  lock `web/node_modules/**.node` files, so `npm ci` in place fails with EPERM (and its
  partial delete was repaired with `npm install`, lockfile untouched). The copy runs the same
  `cp tests/fixtures/site.json src/data/site.json && npm ci && npm run build`. I did not stop
  the user's servers.
- (iteration 1) **Intermediate iterations:** §6 items 12-17 belong to deliverables not built
  yet; they are reported as "not built yet", every other item must be green before a commit.
  The stop condition still requires all 20 in one run.
- (iteration 1) **One pandera contract for both competitions' team lines**
  (`TEAM_GAMES_SCHEMA`, with a `source` column): both sources feed one table with the same
  columns, so one contract (two rows per game, poss_game = mean, counts ≥ 0) covers each.
- (iteration 1) **EL possession agreement 89.0% < 90%** on the sample: explained in
  `docs/data/possessions.md` (0.44 FT weight vs the EuroLeague-implied 0.42, which gives 94%).
  `team_games` keeps R1's 0.44 as the task specifies.
- (iteration 1) **GBL source order:** ESAKE totals when they reproduce the result, else the PBP
  counts when those do, else listed missing. 2018-19..2020-21 ESAKE totals lack team rebounds
  and team turnovers (documented bias ≈ +0.8 possessions per team).
- (iteration 1) New commands: `build` also writes `team_games` + `team_games_missing`;
  `eurohoops possessions` writes `reports/possessions.json` (kept out of `build` so the daily
  workflow, which has no EuroLeague box cache, never rewrites the report).

## Loop log
