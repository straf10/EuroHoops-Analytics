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

## Possession definition (PLAN R1, EuroLeague free-throw weight since M1 v2)
`poss_raw = FGA − OREB + TOV + 0.42 · FTA` per team; `poss_game` = the mean of both teams.
Team rebounds and team turnovers are included wherever the source records them. Until M1 v1
the weight was R1's 0.44; see "How the free-throw weight is estimated" below.

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

| Measure (seed 20260925, 50 games, 100 team-games) | 0.44 (v1) | 0.42 (v2) |
|---|---|---|
| Teams within ±2 possessions (target 90%) | 89.0% | **94.0%** |
| Games with both teams within ±2 | 80.0% | 88.0% |
| Mean gap, PBP − box | −0.34 | +0.02 |
| FT weight that zeroes the mean gap (in-sample) | 0.421 | 0.421 |

The remaining spread (sd ≈ 1.1) comes from logging-order cases the counter cannot resolve (a
technical during a live trip, rebounds logged after the next event).

## How the free-throw weight is estimated
The weight is the share of free-throw attempts that end a possession. Every other term of the
formula is a count, so the only unknown is `w` in

    PBP possessions ≈ (FGA − OREB + TOV) + w · FTA

The play-by-play counter above gives the actual possession count of each team-game. Choosing `w`
so that the formula is right *on average* (method of moments) gives a closed form:

    w = Σ (PBP − (FGA − OREB + TOV)) / Σ FTA          (sums over team-games)

equivalently `w = w_old + Σ gap / Σ FTA`, where gap = PBP − box formula at the old weight. This
is what `ft_weight_matching_pbp` reports: in `reports/possessions.json` on the 50-game sample,
and in `reports/stints_mart.json` (`pbp_vs_box_possessions`) on every cached game from 2011-12.

| Games (EuroLeague, all cached PBP) | Team-games | Weight matching PBP | Within ±2 at 0.44 | at 0.42 |
|---|---|---|---|---|
| 2011-12 → 2014-15 | 1,890 | 0.437 | 87.3% | 86.0% |
| 2015-16 → today | 6,598 | 0.409 | 89.3% | 91.4% |
| All | 8,488 | 0.416 | 88.9% | **90.2%** |

On the passing stints-mart games (4,172), a game-level bootstrap (2,000 resamples) puts the
all-season weight at 0.415, 95% CI [0.414, 0.417]; by season it drifts from ≈0.44 (2011-15) to
≈0.41 (2016-22) and ≈0.40 (2023-25). **0.42 is the rounded EuroLeague value, a deliberate round
number between the 50-game sample (0.421) and the population (0.416).** It is fixed, not
re-estimated per season, so a possession means the same thing in every season of the backtest.

Why it is below the NBA's 0.44: EuroLeague trips end fewer possessions per FTA (more and-ones
and technical free throws per attempt, FIBA team-foul rules). The weight only moves the level of
possessions (≈0.4 per team per game, ≈0.5%); team efficiency *differences* barely change.

**GBL:** the GBL uses the same weight. It is not estimated on GBL data: the possession counter
exists only for the EuroLeague play-by-play, and BasketHotel play-by-play covers only 2018-20.

## The full population (from the stints mart, `reports/stints_mart.json`)
The possession check of the stints mart uses ±5 per team (a logging gap, not the formula bias)
and passes in 99.7% of games (99.6% at 0.44).
