# Odds: The Odds API forward recorder (EuroLeague)

A **market benchmark**, not a betting product (PLAN §1.3). The page shows at most one
market column in the scorecard and never shows prices.

## Source and call
- `GET https://api.the-odds-api.com/v4/sports/basketball_euroleague/odds`
  with `regions=eu`, `markets=h2h,spreads,totals`, `oddsFormat=decimal`, `dateFormat=iso` (D-a).
- `eurohoops odds` (`src/eurohoops/odds.py`) makes **one call per run**, with no retries, because
  a retry is billed too.
- The GBL isn't listed by The Odds API (checked 2026-09-24), so it has no market benchmark.

## Quota and cost
Per the provider's documentation ([v4 guide](https://the-odds-api.com/liveapi/guides/v4/),
[pricing](https://the-odds-api.com/), read 2026-09-25):

| Fact | Value | Status |
|---|---|---|
| Cost of one `/odds` call | markets × regions = 3 × 1 = **3 credits** | **verified** 2026-09-25: `x-requests-last: 3` |
| `/sports` and `/events` | free ("does not count against the usage quota") | documented |
| Free (Starter) plan | **500 credits / month** | **verified**: `used 3, remaining 497` after the first call |
| Cheapest paid plan | 20K credits for USD 30 / month | documented |
| Daily run cost | 3 credits × ~30 days ≈ **90 credits / month** (18% of the free quota) | derived |

- The pricing page also lists "Historical Odds" under the free plan. PLAN §1.3 says
  historical odds are paid. I haven't checked this; one `/historical` call would settle it
  (the docs say historical calls cost more per market and region).
- Every call's `x-requests-last`, `x-requests-used` and `x-requests-remaining` are logged
  and appended to `odds/api_calls.csv`. That file holds no prices or keys.
- The recorder refuses to call while the last known remaining quota is **below 20**.

### Live verification: BLOCKED, then fixed (2026-09-25)
One call to the free `/events` endpoint with the key in `.env` returned **HTTP 401**
`{"error_code":"INVALID_KEY","message":"API key is not valid. ..."}`, with no quota headers.
The stored value is 30 hex characters. The keys I know of are 32, so it's probably truncated.
**Fixed the same day:** the replaced key returned HTTP 200 at 13:36 UTC (3 events, 15–18
bookmakers each, all offering h2h, spreads and totals; exchanges also send `h2h_lay`, which
the consensus ignores). Its 6 team names were unmatched, as designed, and are now in the map
with `verified=yes`; the other 29 candidates stay `verified=no`.

## What is stored (D-b)
| Where | What | In git |
|---|---|---|
| `data/raw/odds/euroleague/<fetched_at>.json.gz` | raw response (every bookmaker's prices) | no (gitignored) |
| `odds/euroleague_2026-27.csv` | one consensus row per upcoming game per snapshot (append-only) | yes |
| `odds/api_calls.csv` | one row per API call: status, cost, used, remaining, events, rows written | yes |
| `odds/euroleague_teams.csv` | explicit API team name → EuroLeague code map | yes |

Consensus row: `game_id, season, round, tipoff_utc, home, away, commence_utc, fetched_at_utc,
bookmakers, p_home, spread_books, spread_home, total_books, total`.
- `p_home` is the **median de-vigged P(home)** over bookmakers with an exact two-way
  home/away moneyline, de-vigged proportionally: P(home) = (1/o_home) / (1/o_home + 1/o_away).
  Three-way (draw) moneylines are skipped.
- `spread_home` is the median handicap on the home team, in the bookmakers' sign convention
  (negative = home favoured). The market's expected home margin is `-spread_home`.
- `total` is the median over/under line.
- Only events whose `commence_time` is after the fetch time are recorded. The docs confirm
  that `/odds` also returns **in-play** events, and those prices aren't pre-game.
- An event matches a game only when both API team names are in the map and an unplayed
  scheduled game with those home/away codes tips off within 48 h of `commence_time`.
  Anything else is **reported, never guessed** (logged as `unmatched odds event`).

## Secrets
- The key is read from `ODDS_API_KEY` (environment, else `.env`). The API accepts it only as
  the `apiKey` query parameter, so the request URL contains it.
- httpx logs every request URL at INFO. The recorder raises the `httpx`/`httpcore` loggers to
  WARNING for the duration of the call; a test caught this leak.
- Transport and HTTP errors are re-raised without the URL.
- Tests use a fake transport and a fake key, and assert that the key appears in no file or log.
- CI: the daily workflow step reads `secrets.ODDS_API_KEY`. The repository owner sets it.
  Without it, the step prints a warning and is skipped (D-c).

## Scorecard
`market` in `reports/live_scorecard.json`:
- Scored games that have an odds row fetched **before tip-off** use their latest such snapshot.
- The column reports market log loss, plus Elo and B0 log loss on the **same** games, with n.
- Games without odds are left out of this column only. The GBL scorecard has `market: null`.

## Unverified (verify on the first valid call, then update this file and `verified=yes` in the map)
1. The API's EuroLeague team names. All 30 rows in `odds/euroleague_teams.csv` are
   candidates marked `verified=no`. The first call lists every name not in the map.
2. Whether spreads and totals are offered for every game, and at what lead time. The
   `spread_books` and `total_books` counts per row answer this directly.
3. The free quota and cost per call, from the headers.

## Decision
**Keep the forward recorder as built and switch it on once the key works.** Steps:
1. Replace the truncated key in `.env`.
2. Run `uv run eurohoops odds` once and fix any unmatched team names.
3. Add the `ODDS_API_KEY` repository secret.

At 3 credits a day it uses under a fifth of the free quota, which leaves room for a second
daily snapshot closer to tip-off if the lead-time check shows that one morning snapshot is
far from the close.
