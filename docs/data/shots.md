# EuroLeague shots (M2 data): the `Points` feed, the shot table and free throws

Built by `eurohoops shots` (marts `shots`, `shots_excluded`, report `reports/shots.json`) and
`eurohoops free-throws` (mart `ft_team_games`, report `reports/free_throws.json`), from the raw
cache only. Both are local commands (like `possessions`): the daily workflow never runs them.
Code: `src/eurohoops/parse/shot_table.py`, `src/eurohoops/parse/free_throws.py`. Tests:
`tests/test_shot_table.py`, `tests/test_free_throws.py` (real-game fixtures
`tests/fixtures/shots_real_games.json`, `points_E2024_1.json`, `pbp_E2024_5_first_half.json`).

## The shot table
One row per field-goal attempt of every played game from 2007-08 to 2025-26 (the live season
stays out: M2 has no live use). Columns: competition, season, game_id, `event` (the feed's
`NUM_ANOT`), period, seconds left in the period, team, opponent, home (false for both teams at a
neutral venue), shooter id (used only by the player shot-making step, never as a feature),
made, value (2/3), x/y in metres, distance, |angle| (degrees from the line straight out from the
basket: 0 = straight on, 90 = level with the basket, > 90 behind it), distance band, `ZONE`, the
three context flags, the score margin before the shot from the shooter's side, and
`validated_season` (2011-12 on). The pandera contract is `SHOTS_SCHEMA`.

Distance bands (reporting and free throws): 2s `rim` < 1.5 m, `short` 1.5-3 m, `mid` 3-5 m,
`long2` ≥ 5 m; 3s `three` < 8 m, `deep3` ≥ 8 m.

## The §3 questions, answered on the whole cache (2026-09-26)
| Question | Answer | Test |
|---|---|---|
| `POINTS_A`/`POINTS_B`: before or after, which is home? | The schedule **home** (A) and **away** (B) score **after** the row, in every season: every row of every game matches the running sum of the feed's own points except 2-22 rows a season (feed corrections; the score before a shot is therefore taken as the row's after-score minus the shot's own points, not from the running sum). | `test_points_a_b_are_the_home_and_away_score_after_the_row`, `test_margin_before_is_the_score_before_the_shot_from_the_shooters_side` |
| `MINUTE`/`CONSOLE` in overtime and across seasons | `MINUTE` counts on through overtime (41-45 first OT, 46-50 second, … up to 60 in 2023-24); `CONSOLE` is `mm:ss` left in the period in every season. They agree on every shot but 1-3 a season, all `00:-1` at the buzzer (read as 0 s left). | `test_overtime_minutes_and_console_give_period_and_seconds_left`, `test_buzzer_console_minus_one_second_reads_as_zero` |
| Action-code vocabulary | `2FGM 2FGA 3FGM 3FGA FTM` every season; blocked attempts `2FGAB`/`3FGAB` in 2007-08 → 2016-17 only (from 2017-18 a blocked shot is a plain `2FGA`/`3FGA`); `LAYUPMD`/`LAYUPATT`/`DUNK` (made layup, missed layup, made dunk) in 2008-09 → 2014-15, as in the PBP. No missed free throws in this feed. Codes per season are in `reports/shots.json` (`actions`). Blocked and layup/dunk codes carry the outcome, so they are **not** features. | `test_2011_codes_layups_dunks_and_blocked_attempts`, `test_committed_report_meets_the_f1_thresholds` |
| Feed FGA and made-FG points vs the box score | Equal for **8,476 of 8,477** 2011-12 → 2025-26 team-games with a box score (99.99%). The one mismatch, E2017_14 KHI (feed +15 FGA, +18 points), is a **box score missing a player** (P008003): the feed's points add up to the final score 85-77 and the gap is exactly that player's shots. E2018_21 has an empty box score (API placeholder). | `test_box_gap_e2017_14_is_one_player_missing_from_the_box_score`, `test_committed_report_meets_the_f1_thresholds` |
| Does the PBP tie a shooting foul to the shot? | **Only for and-ones.** PBP fouls are untyped (`CM` = any personal foul, `RV` = foul drawn), and a shooting foul on a missed shot leaves no shot row. A one-shot free-throw trip whose last field-goal row is a made field goal by the same team at the same clock is an and-one; its shot is found through `NUMBEROFPLAY` = the feed's `NUM_ANOT`. Two- and three-shot trips cannot be told apart from bonus free throws by type (a three-shot trip is a foul on a three, but without a location). | `test_and_ones_are_tied_to_their_made_shot_in_a_real_game` |
| Share of (0, 0) shots and of 2/3 labels contradicting the geometry | (0, 0): 0.05-0.37% of FGA per validated season (0.13-0.53% in 2007-10). Labels contradicting the FIBA line by more than 0.15 m: 0.06-0.22% per validated season. 78 field goals carry the free-throw sentinel (−1, −1) and are excluded as unparseable. Every validated season excludes ≤ 0.53% of FGA in total (per season and reason in `reports/shots.json`). | `test_unusable_rows_are_excluded_with_a_reason`, `test_committed_report_meets_the_f1_thresholds` |

## The context flags are outcome-coded (not features)
`FASTBREAK`, `SECOND_CHANCE` and `POINTS_OFF_TURNOVER` are scoring tags: from 2015-16 on they
are set only on made shots (99.8-100% of flagged shots are makes in every season 2015-2025;
before 2014-15 they are almost never set). They stay in the table as the feed has them, but M2
never uses them (`OUTCOME_CODED_FLAGS`); per-season shares and make rates are in
`reports/shots.json` (`outcome_coded_flags`). Test:
`test_the_feed_context_flags_are_outcome_coded`. Found in the first M2 run (see
`reports/week7-10_progress.md`).

## Exclusions (`shots_excluded`, never silent)
- `unparseable`: team not in the game, points inconsistent with the code, no score, no clock, no
  coordinates or the free-throw sentinel (−1, −1) on a field goal, unknown action.
- `zero_coordinates`: (0, 0), the feed's missing location.
- `label_geometry` (2011-12 on only): a 3 inside the three-point line moved in by 0.15 m, or a 2
  beyond it moved out by 0.15 m. 2007-10 coordinates fit neither the 6.75 m nor the 6.25 m line
  (`reports/week3_closeout.md` §7a), so the test is not applied there; those seasons are kept,
  flagged `validated_season = False`, and never used for fitting or scoring (F-a).

## Free throws (F-e)
The feed logs made free throws only, so free throws come from the play-by-play (`FTM`/`FTA`).
A trip is a run of free-throw rows of one team at one clock (substitutions and time-outs in
between do not split it). And-ones are tied to their shot's distance band; every other trip
(shooting fouls on misses, bonus free throws, technicals) is team-level. An and-one tied to an
excluded shot has no band and counts as an other trip.

Expected FT points of a team-game = Σ_bands FGA_band × and-one trips per FGA_band × points per
and-one trip + FGA × other trips per FGA × points per other trip; every rate is a ratio of
totals, fitted leave-one-season-out on development seasons (all development seasons for
validation).

**Limit (measured):** league FT points per team-game move from season to season by far more
than the ±0.1 target (development sd ≈ 0.71: 12.30 in 2021-22 to 14.51 in 2017-18), and rates
fitted on other seasons cannot know a season's level. The LOSO gaps are in
`reports/free_throws.json`; only 2018-19, 2019-20 and 2020-21 fall within ±0.1.
