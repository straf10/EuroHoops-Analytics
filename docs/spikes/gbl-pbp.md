# Spike: GBL (ESAKE) play-by-play

*Date: 2026-09-24. Test game: 2025-26, `idgame=D6867DA7`, Kolossos H Hotels Collection 85–87 AEK BC.*
*Question (PLAN §4.2, §12.1): how is the PBP served, can we fetch it with plain HTTP, and does it
contain substitutions, game clock, score and shot coordinates?*

## How it is served

1. `esake.gr/el/action/EsakegameView?idgame=D6867DA7&mode=2` only contains an empty
   `<div id="games-pbp">` plus the BasketHotel (MBT / Genius Sports) loader scripts.
2. `widgets.baskethotel.com/static/api/<ApiId>/scripts/pbp.js` builds the widget:
   `GAME_FULL_VIEW_WIDGET` (= widget id **400**), `game_id = MBT.API.convertHexToDex(idgame)`,
   `league_id = MBT.Integration.leagueID`, `season_id = MBT.Integration.seasonID`,
   `show_tabs = ["play_by_play"]`, `use_external_game_ids = 1`, `show_export_link = 1`.
3. `.../static/api/<ApiId>/main.js` sets `MBT.Integration.defaults.leagueID = 30409` and
   `seasonID = 122843`; the ApiId is `55b4cf328e78a7a16e07aefd9518ccb2fb1afa29`.
4. **Hex → decimal is a signed 32-bit conversion**, not plain `int(hex, 16)`:
   `D6867DA7` = 3,599,138,215 unsigned → `game_id = -695829081`. Plain conversion would be wrong.
5. The widget request is a GET to
   `widgets.baskethotel.com/widget-service/show?api=<ApiId>&lang=en&request[0][container]=…&request[0][widget]=400&request[0][param][league_id]=30409&request[0][param][season_id]=…&request[0][param][game_id]=-695829081&request[0][param][use_external_game_ids]=1&request[0][param][show_tabs][0]=play_by_play`.
   It returns JavaScript (`MBT.API.update('<container>', '<escaped HTML>')`). With
   `use_external_game_ids=1` the game resolves even with the default (2026-27) `season_id`;
   the response reveals the real season (`season_id: 131203` for 2025-26) and BasketHotel's
   **internal game id (6084033)**.
   *Update 2026-09-25:* the internal id appears only when the request also sets
   `request[0][param][show_export_link]=1`. It then sits in the export handler as
   `"&game_id=" + 6084033`. Without that parameter, the response carries no internal id.
   With it, the export resolved for all 103 checked 2018-19, 2019-20 and 2022-23 games
   (`reports/gbl_box_gaps.csv`).
6. The PBP rows are not in that HTML (they load through a further partial), but the widget's
   **Export** button is a direct download:
   `widgets.baskethotel.com/widget-service/export/view/play_by_play?api=<ApiId>&game_id=6084033`
   → `.xlsx` (16 KB), no cookies, no JS, no referer needed.

**Plain HTTP is enough. Playwright is not needed.** Path: ESAKE hex id → signed decimal →
`widget-service/show` (to get the internal id) → `export/view/play_by_play` (xlsx).
Parsing needs only `zipfile` + `xml.etree` (or `openpyxl`).

## What the PBP contains

Sheet columns: `Time | Score | <home team> | <away team> | Game Actions` (one row per event;
the column the text sits in tells which team acted). Sample:
[`samples/gbl_pbp_D6867DA7_head_tail.csv`](samples/gbl_pbp_D6867DA7_head_tail.csv)
(first 40 and last 12 events, converted from the xlsx without other changes).

| Field | Present? | Evidence |
|---|---|---|
| **Substitutions** | ✅ Yes | `(9) Andreas PETROPOULOS entered the court` / `… left the court`; 79 "entered" = 10 starters + 69 subs. Replaying them leaves **exactly 5 players per team on court at every clock step** (0 violations in 517 events). |
| **Game clock** | ✅ Yes, elapsed game time `MM:SS` (`00:00` → `40:00`), plus `Start/End of quarter N` markers | |
| **Score** | ✅ Yes, on scoring events only (`0-1`, …, `85-87`) | Points summed from made FT / 2P / 3P events = **85–87**, matching the final score |
| **Shot coordinates** | ❌ No | Shots are text only (`2 points jump shot`, `lay-up`, `dunk`, `3 points jump shot`, `blocked while attempting …`). The widget's `shot_chart` tab is a **server-rendered image** (`part=shot-chart-graph-image`), not data. |
| Players | Jersey number + name (`(24) Frank BARTLEY`); Latin script here, with some double spaces | Entity resolution still needed (PLAN §4.3) |
| Other events | Assists, rebounds (O/D, team), steals, turnovers (typed), blocks, fouls (with fouled player), timeouts, 24-s violations, technicals | |

## Caveats

- Events are English free text, so the parser works on the sentence patterns (e.g. typos like
  `perfomed a steal`). Keep the recorded xlsx as fixtures and fail loudly on unknown patterns.
- Clock resolution is 1 second; several events share a timestamp (a sub and a free throw at
  `39:58`). Stint boundaries must follow row order, not only time.
- `api` id and `leagueID` are hard-coded in ESAKE's `main.js` and can change without notice.
  Cache the raw xlsx; re-read `main.js` if requests start failing.
- Only one game was checked. Older seasons (2018-19 →) are unverified.

## Recommendation for weeks 2–3

1. **GBL RAPM is feasible**: substitutions + clock + score are there and reconcile on the test
   game. Build the GBL PBP ingester on the export endpoint (2 requests per game: `show` for the
   internal id, then the export; throttle ≥ 2 s per PLAN §1.2) after the GBL results scraper and GBL Elo.
2. Before backfilling, run the same two checks (5-on-court, points = final score) on ~20 games
   from 2018-19, 2021-22 and 2025-26 to measure coverage and pass rate.
3. **No GBL shot coordinates**: the GBL gets no xPTS. Use the zone-free shot types
   (lay-up / dunk / jump shot / 3PT) as coarse shot-quality features, and keep the coordinate
   xPTS model EuroLeague-only (PLAN §5.3 already assumes this).
