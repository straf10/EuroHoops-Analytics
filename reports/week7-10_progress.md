# Weeks 7–10 progress log (M2 shot model)

Decisions (user, 2026-09-26): F-a … F-l all defaults (see `prompts/week-7-10.md` §0, commit 1e35d0d).
Branch `week-7-10` (from `main` at f9fcbc9), not pushed.

Format: `iteration N | deliverable | checks run | result | commit`

## Facts verified at start (2026-09-26, whole cache; recorded in docs/data/shots.md with F1)
- Shot feed action codes: `2FGM 2FGA 3FGM 3FGA FTM` every season; blocked attempts `2FGAB`/`3FGAB`
  up to 2016-17 only; 2008-09 → 2014-15 made layups/dunks are `LAYUPMD`/`DUNK` and missed layups
  `LAYUPATT` (as in PBP). No missed free throws (`FTA` never appears).
- `POINTS_A`/`POINTS_B` are the **schedule home / away** score **after** the row, in every
  season (all but 2–22 rows a season match the running sum of the feed exactly).
- `CONSOLE` is the time left in the period (`mm:ss`), `MINUTE` counts on through overtime
  (41–45 first OT, 46–50 second, …); they agree on all but 1–3 rows a season.
- `NUM_ANOT` equals the PBP `NUMBEROFPLAY` of the same action (join key shot ↔ PBP).
- Per team-game FGA and made-FG points of the feed equal the box score (`totr`) in every
  2011-12 → 2025-26 team-game but one (E2017_14 KHI: +15 FGA, +18 points) and E2018 one game
  with an empty box score.
- PBP fouls are not typed (`CM` = any personal foul; `RV` = foul drawn): a free-throw trip can
  be tied to a shot only when a made field goal by the same team at the same clock precedes it
  (and-one). Missed-shot shooting fouls leave no shot row.
- League FT points per team-game, 2011-12 → 2022-23: 14.23, 13.22, 12.59, 13.69, 13.71, 13.51,
  14.51, 13.70, 13.69, 13.40, 12.30, 14.44 (season-to-season sd ≈ 0.6 points).

## Decisions not covered by the task file
- (iteration 1) **`eurohoops shots` and `eurohoops free-throws` are local commands** (like
  `possessions`), not part of `build`: the daily workflow has no full shot/PBP cache and M2 has
  no live use. The shot table stops at 2025-26 (`LAST_SEASON`); the live season stays out.
- (iteration 1) **(−1, −1) on a field goal** (78 rows, the free-throw sentinel) is excluded as
  `unparseable`; the `label_geometry` exclusion is applied from 2011-12 only (2007-10 fit no line).
- (iteration 1) **Blocked (`xFGAB`) and layup/dunk codes are not features**: they encode the
  outcome (a blocked shot is always missed, `DUNK`/`LAYUPMD` are made only).
- (iteration 1) **Reports drifted by live games during the checklist** (`live_scorecard.json`,
  `possessions.json`, `stints_mart.json` gained 2026 games E2026_8..10 from the local marts):
  reverted, never committed from this branch; they are the daily workflow's.
- (iteration 2) **F2 / checklist item 22 is red for a structural reason, not a bug.** League FT
  points per team-game vary between development seasons with sd 0.66 (12.30 in 2021-22 to 14.51
  in 2017-18); rates fitted on the other seasons (LOSO, as F2 specifies) cannot know a season's
  level, so the mean gap is ±0.9-1.3 in 2011, 2013, 2017, 2021, 2022 (validation −0.60). No
  shot-profile model fitted out of season can reach ±0.1. I have **not** widened the tolerance
  (that would make the check vacuous) and have not switched to in-season rates (that breaks
  the LOSO rule). Item 22 stays red and is reported as such; open question for the user.

## Loop log
iteration 1 | F1 shot table (marts shots/shots_excluded, reports/shots.json, docs/data/shots.md) | full §6 1-22 (one run, no edits) | 1-21 PASS; 22 FAIL (F2, see decision) — F1 done: feed = box 99.99%, exclusions ≤ 0.53%/season, two builds identical | see F1+F2 commit
iteration 2 | F2 free-throw generation (ft_team_games, reports/free_throws.json) | same run | code + tests green; item 22 FAIL (structural, diagnosed above) | see F1+F2 commit
