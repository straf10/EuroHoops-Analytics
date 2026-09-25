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
  points and shots). **Done in weeks 3–5**, see below.

### Play-by-play ingester and the 2018-20 fill (weeks 3–5)
- `eurohoops ingest --competition gbl --pbp --seasons 2018 2019` makes two requests per game:
  the BasketHotel widget (`show_export_link=1`, for the export id), then the xlsx export.
  - Both requests use the ≥2 s throttle and a gzip cache (`data/raw/gbl/pbp_widget/`,
    `data/raw/gbl/pbp/`), and nothing is ever re-fetched.
  - It's a local backfill and never runs in CI.
  - `--pbp` only chooses which seasons get PBP. The staging tables still cover every default
    season.
- The 2018-19 and 2019-20 backfill covered 342 played games in 682 requests, 2026-09-25.
- **Parse:** `parse/gbl_pbp.py` produces a typed event table (`data/staging/gbl_pbp.parquet`,
  pandera `PBP_SCHEMA`).
  - Every row has an elapsed-game clock `MM:SS`. Periods come from the "Start of …" rows.
  - Substitutions are `(n) Name entered/left the court`.
  - Scoring sentences are `made a free throw`, `performed a 2|3 points …`; misses are
    `missed a …` or `blocked while attempting a …`.
  - Other sentences (rebounds, fouls, the `perfomed a steal` typo, …) are kept as `other`
    with their text.
  - The two team columns are mapped to home/away by the sheet's final score. The first
    column isn't always the home team.
- **Validation against ESAKE, all 342 games:**
  - PBP points equal the results page in **342/342**.
  - On the 505 teams with a complete official box, **5,454/5,457** shooting lines (points,
    2P, 3P and FT made/attempted) match exactly.
  - Every player's seconds match exactly for **486/505** teams.
  - One bug found on the way: minutes have to run to the "End of game" row, not the last
    player event.
- **Fill** (`parse/box.py`, `source = "pbp"`; official rows are `source = "esake"`):
  - **78 games without a box get PBP team totals:** 156 teams, all equal to the result.
  - **23 short boxes get the missing player's line** (2–23 points, median 10), added only
    when exactly one PBP line with stats has no exact match among the ESAKE lines and it
    closes the gap. All 23 now add up to the result.
  - **The 2 short 2022-23 boxes aren't filled:** they're outside the decided 2018-20 scope,
    and the reason is recorded.
  - `reports/gbl_box_gaps.csv` gives each listed game its fill and a reason. The per-season
    `pbp_fill` block in `reports/gbl_box_invariants.json` counts them.
  - The official pass rates above are **computed on ESAKE rows only and unchanged**. A filled
    game is never an "official box".
- PBP player lines are keyed `pbp:<jersey>:<name>`, not by ESAKE player id: entity
  resolution is PLAN §4.3 work. Box-score models that need player identities should still
  start at 2020-21. Team-level work can use 2018-20 through the `pbp` team totals.

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
