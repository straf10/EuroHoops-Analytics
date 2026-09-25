# GBL data (ESAKE): structure, quirks and quality

*Built 2026-09-24 from a full crawl: 1,658 pages (results + box scores), 2018-19 → 2026-27.*
Source code: `src/eurohoops/ingest/gbl.py`, `src/eurohoops/parse/esake.py`, `sql/box.sql`.

## Structure

| Season | ESAKE id | Phase A (regular season) | Phase B (playoffs etc.) | Games (played) | Notes |
|---|---|---|---|---|---|
| 2018-19 | `A12E05CD` | 26 rounds, 182 games | 24 games, rounds `01`…`12` | 206 (206) | 3 forfeits (Olympiacos) |
| 2019-20 | `49EEB365` | 26 rounds listed | none | 182 (139) | season stopped (COVID): rounds 20–26 never played |
| 2020-21 | `03FFA3AC` | 22 rounds, 132 games | 27 games | 159 (159) | games without fans |
| 2021-22 | `8C367D67` | 26 rounds, 156 games | 24 games | 180 (180) | |
| 2022-23 | `DC917125` | 22 rounds, 132 games | 24 games | 156 (156) | |
| 2023-24 | `C1AF5EF5` | 22 rounds, 132 games | 31 games (playoffs, play-outs, classification) | 163 (163) | |
| 2024-25 | `4820C134` | 22 rounds, 132 games | 34 games (idem) | 166 (166) | |
| 2025-26 | `44B80BEB` | 26 rounds, 156 games | 18 games | 174 (174) | 13 teams |
| 2026-27 | `184645B9` | 26 rounds, 183 listed | — | 183 (0) | live; see "duplicate fixture" |

- **Rounds.** Each phase's first page (`series=01`) embeds its round list as
  `new Option('<label>', '<code>')` scripts: `01`…`26` in phase A, `201` (QF1) … `405` (F5) in
  new-format playoffs (2023-24 on). Old-format playoff pages (2018-19 → 2022-23) have no list;
  their rounds (`01`, `02`, …) are enumerated until a page is empty.
- **Times** are Athens local time without a year (`Σαβ 4 Οκτ - 16:00`); the year comes from
  the preceding month header. Converted to UTC with `zoneinfo` (EET/EEST). A date without a
  time is unconfirmed (stored as 00:00 Athens, `confirmed_date = False`, never predicted).
- **Teams** are keyed by ESAKE team id (from the logo path), stable across seasons; display
  names follow the current sponsor name (Panathinaikos `00000001`, Olympiacos `00000002`).
- **Scores:** `85 - 87` played; empty `-` unplayed; `20 - ` a forfeit (stored 20-0,
  `forfeit = True`, never used to update ratings or scored).

## Quirks found

| Quirk | Where | Handling |
|---|---|---|
| Forfeits scored `20 - ` | 3 games, 2018-19 (Olympiacos) | `forfeit` flag; excluded from Elo and metrics |
| Season stopped | 2019-20: 43 games never played | past seasons are final; unplayed games ignored |
| No box score published | 52 games in 2018-19, 26 in 2019-20 | page has only the game header; flagged `missing_box` |
| Abandoned / shortened games | e.g. 2022-23 final game 5 `GBL2022_EEF9413F` 35-63, `GBL2023_E4AE7015` 71-35 | kept as results; minutes check flags them in box data |
| Duplicate fixture | 2026-27: Olympiacos–AEK listed twice (second copy in round 18, 13 Feb 2027, date unconfirmed) | none yet; unconfirmed dates are never predicted. **Watch it** when the date is confirmed |
| Greek + Latin names mixed | player and team names | out of scope here (entity resolution, PLAN §4.3) |

## Box-score quality (`reports/gbl_box_invariants.json`)

Checks per team and game, flagged and never dropped: box score present; Σ player points =
totals row = results-page score; minutes = 200 + 25 per overtime (±1 min); no line with made
> attempted or points ≠ 2·2PM + 3·3PM + FTM.

| Season | Pass | Missing box | Points mismatch | Minutes off |
|---|---|---|---|---|
| 2018-19 | 129/203 (63.6%) | 52 | 15 | 74 |
| 2019-20 | 102/139 (73.4%) | 26 | 8 | 37 |
| 2020-21 | 153/159 (96.2%) | 0 | 0 | 6 |
| 2021-22 | 178/180 (98.9%) | 0 | 0 | 2 |
| 2022-23 | 151/156 (96.8%) | 0 | 2 | 5 |
| 2023-24 | 160/163 (98.2%) | 0 | 0 | 3 |
| 2024-25 | 164/166 (98.8%) | 0 | 0 | 2 |
| 2025-26 | 169/174 (97.1%) | 0 | 0 | 5 |

Forfeits are not box-checked. Before 2020-21 the box scores are unreliable (missing pages and
minutes that do not add up); models that need player minutes should start at 2020-21 or
treat 2018-20 with care. No line ever failed the shot-consistency checks.

### The gaps are at the source, and play-by-play covers them (checked 2026-09-25)
Every affected game is listed in `reports/gbl_box_gaps.csv`.

- **78 missing box scores** (52 in 2018-19, 26 in 2019-20):
  - The page carries only the game header, quarter scores and team leaders. There is no stat
    table in `mode=3`, and `mode=2` holds only the PBP widget container.
  - A live refetch of 10 of these pages on 2026-09-25 was identical, so the cache isn't stale.
- **25 "points mismatch" games** (15 in 2018-19, 8 in 2019-20, 2 in 2022-23):
  - In every case one team lists only 10–11 players, and its totals row adds up the listed
    players. The result is 1–23 points higher (median 8) and minutes are below 200.
  - ESAKE simply omits a player from these pages. The parser keeps every row.
- **Parser misses: 0.**
- **Play-by-play coverage:** the BasketHotel PBP export exists for all 103 games, and its
  final score matches the results page in 103/103.
- **Decided (2026-09-25):** fill these games from PBP (team totals and the missing players'
  points and shots) when the GBL PBP ingester lands. Until then, box-score models start at
  2020-21.

### 2026-27 standings
- Olympiacos (`00000002`) and Panathinaikos (`00000001`) start on **−2 points**. This is a
  sanction for the altercation between players in the 2025-26 finals, and it is shown on
  ESAKE's live table. It is recorded in `eurohoops.standings.GBL_2026.points_deducted`.
- The 2026-27 playoff format is still unconfirmed (see `GBL_2026.unverified`).

## Live operation

- The daily workflow fetches only the live season's unfinished rounds (~27 pages at 2 s);
  completed rounds and past seasons come from the raw cache kept in the GitHub Actions cache.
  A cold cache refetches ~350 results pages once (~15 min). Box scores are never fetched in CI.
- GBL games are mostly Saturday/Sunday, 12:15–20:30 Athens time (about 09:00–18:00 UTC). The 08:00 UTC
  run with a 36 h window covers each game twice (the day before and the same day), so one
  delayed or failed run does not lose a prediction.
