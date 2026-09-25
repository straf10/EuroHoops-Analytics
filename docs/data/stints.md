# EuroLeague stints mart

*Built 2026-09-25 (weeks 5–7, E2). Code: `src/eurohoops/parse/stints_mart.py`.
Command: `uv run eurohoops stints --mart` (after `eurohoops build`); report:
`reports/stints_mart.json`. Method and quirks: `docs/spikes/stints.md`.*

## Tables (in `data/marts/eurohoops.duckdb`)
- **`stints`**: one row per 5-on-5 stint (a substitution by either team closes it) of every
  cached rated game from 2011-12: game_id, season, period, start_s/end_s (game seconds),
  home/away codes, `home_players`/`away_players` (sorted player-id lists), points and
  possessions for each side (possession ends from the play-by-play counter of
  `docs/data/possessions.md`). Failing games keep their stints.
- **`stint_game_checks`**: one row per game with the spike's four checks (five on court,
  seconds, minutes ±60 s, points), a `possessions` check (the PBP count within ±5 of the box
  formula for both teams), `passed` (= the four spike checks) and the reasons.
  A lineup model uses only stints of games with `passed = true`.

2007-08 → 2010-11 are left out: whole-minute substitution clocks (spike quirk 1).

## Result (2026-09-25 cache: 4,246 games, 141,625 stints)
| Span | Games | Pass rate (four checks) | Threshold |
|---|---|---|---|
| 2011-12 → 2014-15 | 945 | **95.9%** | ≥ 95% |
| 2015-16 → 2026-27 | 3,301 | **98.9%** | ≥ 98.5% |

Same as the spike's population run (95.9% and 99.0%), so no regression. In every passing game
the stint points add up to the final score. Two builds give identical tables and a
byte-identical report (the report stores a sha256 of each table). One fix was needed for that:
the spike's minutes check picked the "worst" player of a tie in set order, which changes with
Python's hash seed; it now iterates players in sorted order (the 50-game sample report is
unchanged, it has no failures).
