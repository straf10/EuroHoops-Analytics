# M1 v2: wider grids and a EuroLeague free-throw weight

Declared 2026-09-25, before any v2 run on real data and before the first live M1 row (the live
M1 logs were still empty; the first daily run with M1 is 2026-09-26 08:00 UTC).

## Why a v2, and what it is not
M1 v1 (`0.2.0+m1.0cff42bc` EuroLeague, `0.2.0+m1.3f4861e2` GBL) passed its pre-registered gate.
The user decided on the v1 close-out's open questions 3 and 4 (2026-09-25):

1. **Wider rating and pace grids.** Several v1 tuned values sat on a grid edge (EuroLeague rating
   half-life 480 d = max, rating ridge 250 = min, pace half-life 60 d = min, pace ridge 2 = min;
   GBL rating ridge 250 = min).
2. **Free-throw weight 0.42** instead of R1's 0.44 in `FGA - OREB + TOV + w * FTA`.

Both changes are **post-hoc**: they were chosen after the v1 validation *and* test numbers were
seen. The grid edges were read from the tuning search only, and the weight comes from the
play-by-play vs box comparison, not from any model score. Even so, v2's validation gate and
test numbers are **not a clean hold-out** and are reported as post-hoc everywhere (report,
model card, site). The v1 reports stay in git history (commit 57aa860) for comparison.

## Declared v2 grids (nothing else changes)
| Grid | Axis | v1 | v2 |
|---|---|---|---|
| rating | half_life_days | 60, 120, 240, 480 | 60, 120, 240, 480, 960, 1920 |
| rating | carry | 0.25, 0.5, 0.75, 1.0 | unchanged (1.0 is a natural bound) |
| rating | ridge (possessions) | 250 … 4000 | 62.5, 125, 250, 500, 1000, 2000, 4000 |
| pace | half_life_days | 60, 120, 240, 480 | 30, 60, 120, 240, 480, 960 |
| pace | carry | 0.25, 0.5, 0.75, 1.0 | unchanged |
| pace | ridge (games) | 2, 5, 10, 20 | 0.5, 1, 2, 5, 10, 20 |

Sizes: rating 80 → 168, pace 64 → 144. Same objectives (tuning seasons only), same four margin
variants, same seasons, same comparison-Elo grid, same gate rule (M1 validation log loss <
comparison-Elo validation log loss). If v2 fails the gate, live M1 stops per E-g.

## Free-throw weight
`FT_WEIGHT = 0.42` in `parse/possessions.py`; it changes `poss_raw`/`poss_game` in `team_games`
for both competitions (the GBL has no play-by-play from 2021 on to estimate its own weight, so
it takes the EuroLeague value). Estimation is documented in `docs/data/possessions.md`.
