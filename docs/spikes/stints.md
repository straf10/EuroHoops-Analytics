# Spike: EuroLeague stints from play-by-play (PLAN §12 item 6, D-h)

Date: 2026-09-25. Code: `src/eurohoops/parse/stints.py`, `eurohoops stints` →
`reports/stint_validation.json`. The stints **mart** is M1 work (weeks 5–7) and isn't built here.

## Method
- **Starting fives:** box score `IsStarter` (the PBP has no lineup rows).
- **Lineup changes:** `IN`/`OUT` rows in log order. Lineups carry over between periods,
  because period-start changes are logged as ordinary `IN`/`OUT` rows at 10:00 (5:00 in OT).
- **Stint boundaries:** every substitution time and every period end. Several substitutions
  at the same instant make one boundary.
- **Clock:** `MARKERTIME` (time remaining) gives game seconds. `BP`/`EP` rows carry no
  clock. Overtime periods come from `MINUTE` (41–45 = OT1, 46–50 = OT2, …) because
  `ExtraTime` holds every overtime in one list, and an overtime's opening substitutions are
  logged **before** its `BP` row.
- **Points:** all five play types that change the running score. I checked this over every
  cached game from 2007 to 2026 by diffing `POINTS_A`/`POINTS_B`: `2FGM`, `3FGM`, `FTM`,
  plus **`LAYUPMD` and `DUNK` before 2015-16**. Some play types are space-padded (`'2FGM      '`).
- **Overtime count (for the seconds check):** the box score's `ByQuarter`, a source
  independent of the PBP. Only periods where at least one team scored count (see quirks).

## Checks per game
1. **five_on_court:** every stint with positive length has exactly 5 players per team.
2. **seconds:** each team's stint seconds sum to 2400 + 300 per OT, and no stint is negative.
   Stints tile each period by construction, so in practice this checks the overtime count
   against the box score and the clock direction. It is the weakest of the four checks.
3. **minutes:** each player's on-court seconds match the box minutes within ±60 s.
4. **points:** each team's stint points sum to the box score's final.

## Result: the 50-game sample (the D7 finding)
Seed `20260925`. Each season's cached games are shuffled, then drawn round-robin over
2015-16 → 2026-27: 5 games from each of 2015–2018, 4 from each of 2019–2025, and 2 from 2026
(only 2 live-season games have cached PBP). Running it twice gives byte-identical reports.

| Check | Pass rate |
|---|---|
| five_on_court | **100%** |
| seconds | **100%** |
| minutes | **100%** |
| points | **100%** |
| all four | **100% (50/50)** |

A clean 50/50 isn't a sign that the checks can't fail. On the same checks, the full
population fails 1.2% of games from 2015 on, and a sample of 50 comes out clean about half
the time at that rate. The tests show every check failing on a dropped substitution row.

## Context: every cached game (not a gate; same code, run once)
| Seasons | Games | All four pass | Main failure |
|---|---|---|---|
| 2007-08 → 2010-11 | 796 | **3.0%** | minutes: 97.8% of substitution clocks read `mm:00` (whole minutes only) |
| 2011-12 → 2014-15 | 945 | 95.9% | missing/duplicate `IN`/`OUT` rows; minute errors |
| 2015-16 → 2026-27 | 3,300 | **99.0%** | the same, rarer |

2015-26 failures (34 games): 30 × minutes only, 12 × five_on_court (4 or 6 on court, mostly
for 1–60 s), 2 × both, 1 × minutes + points (E2017_14: one team's stint points exceed its
final by 19), and 1 empty game.

## Source quirks found (recorded for M1)
1. **Whole-minute substitution clocks, 2007-10.** Lineup timing in those seasons is off by up
   to a minute on each change. Stints there are unusable for RAPM-style work.
2. **Phantom overtime.** 13 box scores in 2015-16 carry `Extra1: 0` for both teams (and a
   205:00 team-minutes total) in games decided in regulation, with no PBP overtime. My
   first version believed them and failed those 13 on minutes. An overtime now counts only
   if someone scored in it.
3. **Pre-2015 scoring codes:** `LAYUPMD` and `DUNK`. Without them, points fail in almost
   every 2008-14 game (a parser gap, now fixed).
4. **Empty placeholder games:** E2018_21 has `N/D` teams in both PBP and box score, with no
   rows. It's reported as failing, never skipped.
5. **Missing or duplicate substitution rows:** a lineup of 4 or 6 for a stretch, sometimes
   for a whole half (E2017_149: IST at 4 players for 1,430 s).
6. **No starters flagged** (E2015_23: 0 players in the opening stint).

## Recommendation for M1 (weeks 5–7)
- **Build the stints mart from 2011-12 onward** (95–100% of games clean). Keep 2007-10 out
  of any lineup-based model. Their results and box scores stay usable.
- **Gate each game, don't drop it silently.** Store the four check results per game in the
  mart and fit RAPM only on stints from passing games (≈ 99% from 2015). Report the excluded
  share in every model card.
- **Repair only what's unambiguous:**
  - Overtime from scored periods: done.
  - A one-row fix when a lineup has 4 or 6 players and the box minutes identify the player:
    optional, measure first.
  - Don't impute whole missing substitutions.
- **Possessions** (needed for M1 per-100 ratings) aren't validated here. Build and check
  them on the same sample next (possession ends from made shots, defensive rebounds,
  turnovers and final free throws).
