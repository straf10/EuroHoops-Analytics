# Possessions and the `team_games` mart

*Built 2026-09-25 (weeks 5–7, E1). Code: `src/eurohoops/parse/team_box.py`,
`src/eurohoops/parse/possessions.py`, `src/eurohoops/parse/possession_report.py`.
Report: `reports/possessions.json` (`uv run eurohoops possessions`).*

## The table
`eurohoops build` writes `team_games` (one row per team per played, non-forfeit game) and
`team_games_missing` (every such game without two rows, with the reason) into the marts,
from the raw cache only:

| Competition | Source | Rows from |
|---|---|---|
| EuroLeague | `euroleague_box` | box JSON `totr` = player lines + `tmr` (team rebounds, team turnovers) |
| GBL | `esake_box` | ESAKE totals row `ΣΥΝΟΛΟ` (includes the `ΟΜΑΔΙΚΑ - ΠΑΓΚΟΣ` team row) |
| GBL 2018-20 gaps | `gbl_pbp` | BasketHotel play-by-play counts, used only when the ESAKE totals miss the result and the PBP points reproduce it |

Columns: competition, season, game_id, team, opponent, home, points, fga, fta, oreb, dreb, tov,
minutes (40 + 5 per overtime in which someone scored), `poss_raw`, `poss_game`, source.
Contract: `TEAM_GAMES_SCHEMA` (two rows per game, `poss_game` = mean of `poss_raw`).

## Possession definition (PLAN R1)
`poss_raw = FGA − OREB + TOV + 0.44 · FTA` per team; `poss_game` = the mean of both teams.
Team rebounds and team turnovers are included wherever the source records them.

## Coverage (2026-09-25 cache)
Every rated game has two rows except 9 (all listed in `team_games_missing`):

| Game(s) | Reason |
|---|---|
| E2017_14 | box points 66-77 are not the result (the spike's E2017_14 anomaly) |
| E2018_21 | box score has no players (the `N/D` placeholder game) |
| 5 × E2026 | live season: box scores not cached yet (`ingest --details`) |
| 2 × GBL2022 | ESAKE totals (76-90, 78-82) are not the result, no PBP for 2022-23 |

Points in `team_games` equal the `games` scores for every row (0 mismatches; checked in the
report and enforced when the table is built).

## Verified source facts
- **EuroLeague:** in all 5,042 cached box scores (2007-08 → 2026-27; E2018_21 has no players), `totr` equals the sum of
  the player lines plus `tmr` for FGA, FTA, OREB, DREB, TOV and points. Team rebounds and team
  turnovers exist separately (`tmr`) and are in the totals for every season from 2007.
- **ESAKE:** in all 1,262 cached pages with stat tables (of 1,340), the totals row equals players + team row for REBS,
  D.REBS, O.REBS and TO; FGA/FTA equal the player sums. The team row is **always 0 in 2018-19
  → 2020-21** and mostly non-zero from 2021-22 on: ESAKE started recording team rebounds and
  team turnovers during 2021-22. Before that they are simply missing from the totals.
- **GBL play-by-play (2018-20):** has player and team rebounds (`made a offensive rebound`,
  `Offensive rebound`) and turnovers (bad pass, ball handling, travelling, offensive foul, out
  of bounds, 3/5/8/24-second and backcourt violations). On the 264 games with both an ESAKE box
  and a PBP export, per team: points agree exactly 95.6%, FGA 94.7%, FTA 98.1% (the rest are the
  short ESAKE boxes the fill exists for). OREB and TOV are higher in the PBP by 1.53 and 0.69
  per team on average: the team rebounds and team turnovers ESAKE did not record then.
  Consequence: **ESAKE-based possessions for 2018-19 → 2020-21 run about 0.8 per team high**
  (1.53 fewer OREB subtracted, 0.69 fewer TOV added), about 1% of a game. The M1 league mean
  absorbs a level shift like this; it is listed as a limitation in the model card.

## Box formula vs play-by-play count (EuroLeague, the 50-game stint sample)
The PBP counter ends a possession at a made field goal, a defensive rebound after a miss, a
turnover, or the last free throw of a trip (made; a missed last free throw waits for the
rebound). Free throws after an and-one, a technical, unsportsmanlike or disqualifying foul, or
a bench/coach foul end nothing; a miss nobody rebounds before the buzzer ends at the period end.
On all 1,318 team-games of 2022-23 and 2023-24 the PBP event counts (FGA, FTA, OREB, DREB,
TOV) equal the box totals exactly, so the two counts differ only in how free-throw trips
and unrebounded misses are counted.

| Measure (seed 20260925, 50 games, 100 team-games) | Value |
|---|---|
| Teams within ±2 possessions | **89.0%** (target 90%) |
| Games with both teams within ±2 | 80.0% |
| Mean gap, PBP − box | −0.34 (sd 1.07, max 3.36) |
| FT weight that zeroes the mean gap (in-sample) | 0.421 |
| Teams within ±2 at that weight | 94.0% |

**Why the target is missed by one team-game:** the 0.44 free-throw weight is an NBA estimate of
possession-ending trips per FTA. EuroLeague trips end fewer possessions per FTA (more and-ones
and technical free throws per attempt, FIBA's team-foul rules), so the box formula runs ~0.3-0.5
possessions per team above the count. With the weight the data imply (0.42) agreement is 94%.
The remaining spread (sd ≈ 1.1) comes from logging-order cases the counter cannot resolve (a
technical during a live trip, rebounds logged after the next event). The task fixes the R1
formula with 0.44, so `team_games` keeps it; the bias is a constant ~0.5% level shift that the
model's league mean absorbs and it cancels in efficiency *differences* between teams.
