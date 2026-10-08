# EuroHoops

Forecasts every EuroLeague and Greek Basket League (GBL) game of the live season, commits each forecast to a public log before tip-off, and scores it against a home-win baseline.

## Forecasts and the log

**Pre-registered log**:
The public, append-only CSV of forecasts (`predictions/*.csv`); a row is never edited or removed, only appended.
_Avoid_: prediction file, history

**Pre-registered row**:
A game's earliest log row stamped before its tip-off; the one row per game that is scored.

**Provable row**:
A row stamped before tip-off that also reached the public log before tip-off.
_Avoid_: valid row (a valid row is only stamped before tip-off)

**Late row**:
A row stamped at or after its game's tip-off; shown on the site as "Logged late", never scored.

**Live season**:
The season being forecast (2026-27).

**B0**:
The home-win baseline: a constant home-win rate and home margin, 0.5 and 0 at a neutral venue.

**M0 / M1 / M2**:
The models: M0 is the Elo rating (the live forecast), M1 the possession-based team efficiency model, M2 the shot-quality model (expected points per shot).

**Projected roster**:
Each player's expected share of a game's minutes, from the team's games before the round's first tip-off; M5's forecast input.
_Avoid_: lineup, starting five

**Oracle roster**:
The minutes players actually played in the game; an upper bound for a roster-aware model, never a forecast and never logged.

**Team residual**:
What a team's results show beyond the sum of its players' ratings (coaching, system, fit), fitted walk-forward in M5.

**Rest difference**:
Home minus away of a rest feature (days since the previous game in either competition, short rest, games in the last 7 days, previous game in the other competition).

## Games and teams

**Rated game**:
A played game that is not a forfeit; only rated games update ratings or are scored.

**Forfeit**:
A game awarded 20-0 by decision; it counts as played but carries no rating information.

**Source code**:
The team code the data source uses (EuroLeague API `ULK`, ESAKE id `00000001`); every table, log and report uses it.
_Avoid_: team id

**Display code**:
The real-life abbreviation the site shows in place of a source code (`FBT`, `PAO`).

## Players across leagues

**Person id**:
One player across the EuroLeague and the GBL (`P:<EuroLeague id>`, else `G:<ESAKE id>`), from the `player_xwalk` mart; source ids stay the keys of every other table.
_Avoid_: player id (that is a source id)

**Dual season**:
A season a player played at least 300 minutes in both leagues (in practice for Panathinaikos or Olympiacos).

**Mover**:
A player with at least 300 minutes in one league in a season, under 100 in the other, and at least 300 in the other the next season.

**Silver pair**:
A GBL id and a EuroLeague id on the same Greek club in the same season, with equal or near birth dates; matcher tuning data, never a label.

**Confirmed time**:
A tip-off time the source has published. A game without one (in the GBL it sits at local midnight of its date) is never forecast; it is the next tip-off only by its date ("time TBC"), when no game ahead has a confirmed time.

## Season simulation (M7)

**Checkpoint**:
A point in a past season where the backtest stops time: after 25%, 50% or 75% of the regular-season rounds, at the next round's first tip-off. Everything a checkpoint forecast uses comes from games before it.

**Strength posterior**:
M1's team ratings read as a Gaussian at a cutoff: the ridge solution as mean, σ²·(G + P)⁻¹ as covariance. Each simulation draws one strength vector from it and keeps it for the whole season.
_Avoid_: rating (the point value M1 forecasts with)

**Cut line**:
The regular-season places that decide a team's path: the direct playoffs (top 8 through 2022-23, top 6 after), the play-in (7–10) and the top 10.

**Simulation log**:
The append-only per-team record of the live season simulation (`predictions/{competition}_sim_2026-27.csv`), one set of rows per completed round; written only when M7's gate passed.

## Player projections and scouting (M6)

**Projection**:
A player's expected per-100-possession rates, shooting percentages and (EuroLeague) impact for the next season or the rest of the current one, each with an 80% interval: his past seasons decayed and shrunk to the league, calibrated on tuning seasons. Minutes are not projected.
_Avoid_: prediction (a game forecast in the log)

**Aging curve**:
The expected year-over-year change of a rate at each age, from players with 500+ possessions in consecutive seasons (delta method on regressed rates, survivor corrected). Ages stay inside the pipeline; no age is published.

**Over/under board**:
Players doing better or worse than expected on shot-making, 3P% or on/off, with a z score and the share of the gap expected to last (from that dimension's year-to-year stability).

**Comparable**:
A past player-season nearest to a player's newest season in standardised box rates (and EuroLeague shot zones); a player is never his own comparable. Shown as "plays like".
_Avoid_: twin (the Shot Profile Twin, shot profile only)

**Read model**:
The one module (`api/readmodel.py`) that turns the marts, logs and committed reports into the JSON the API serves; the static site files are its responses, written in-process.
