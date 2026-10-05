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
