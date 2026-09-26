# Weeks 7–10b progress log (M2 follow-up: flags, FT shares, runtime, season level)

Decisions (user, 2026-09-26): user decisions 1–4 and sub-decisions G-a … G-g, all defaults
(`prompts/week-7-10b.md` §0, commit c2d850a). Branch `week-7-10`, not pushed.

Format: `iteration N | deliverable | checks run | result | commit`

## Machine state
- 2026-09-26 ~21:50 local: CPU load 3%. Not started by me: `astro dev --port 4321` from
  `C:\Python\eurohoops-ui\web` (node 5520, 11732, started 21:46) and `cswap.exe auto` (python
  14084, 14376). Not killed. Before any timed run I check again and ask the user if the dev
  server is still up (it made BLAS threads spin last time).

- 2026-09-26 (asked before any timed run): **the user chose "run anyway"** with the dev server
  up. Every runtime below is measured with that `astro dev` server running unless stated.

## G2 pre-run finding and the user's decision (before any real-data share number existed)
Simulation on synthetic seasons only (the model right by construction; 200 seasons, ~30 games
per team; `tests/test_free_throws.synthetic_season`): the declared rule "every statistic within
2 SE in every held-out season" fails at least one of the 6 band checks in 27% of seasons (per
band 3.5–10%; the rarest band highest), the team mean-|gap| check in 0%, the r check in 1%;
P(all 13 held-out seasons pass) ≈ 0.015. Cause: ~104 checks at a ~4.6% two-sided false-alarm
rate each. Asked the user; **answer (2026-09-26): count rule.** 2 SE per statistic stays and
each statistic is still flagged; the gate passes when the number of flagged band checks over
all held-out seasons is ≤ the 95th percentile of Binomial(n band checks, 0.0455) (≈ 7 of 78),
and every season's two team checks pass. Fixed in code before the first real-data run.

## Decisions not covered by the task file
- (G2) **Team share = share of per-game rates** (a team's FT points per game over the sum of all
  teams' per-game rates), so playoff games do not inflate a team's share; equal to the plain
  share when every team plays the same number of games.
- (G2) **"2 × SE" for the two team summaries** (G-b does not say what the target is): mean
  |share gap| ≤ 2 × the RMS of the teams' bootstrap gap SEs (the noise scale of one team's gap;
  a right model sits near 0.8 of it); Pearson r ≥ 2 × its bootstrap SE (expected shares track
  actual ones beyond noise). The band gaps have target 0: |gap| ≤ 2 SE.
- (G2) Actual and-one points per band needed new mart columns `and_one_ftm_{band}` in
  `ft_team_games`.
- (G1) **The audit also fails a missed-only level** (make rate ≤ 0.01), not only ≥ 0.99: a
  blocked-shot code is as outcome-coded as a scoring tag. Stricter than asked, not weaker.
- (G1) **A level counts only with ≥ 100 development shots** (`MIN_SET_SHOTS`): a handful of
  shots cannot tell a tag from chance. **Continuous columns** (> 32 distinct values: distance,
  spline basis, clock, margin) are not audited; a flag or one-hot level has ≤ 32 values.
  Fixed before the audit first ran on real data.
- (G1) **The audit runs at the start of every M2 backtest** too (raises on a hit), so it is live
  code, not only a check; it changes no number.
- (sequencing) **Full §6 runs are batched:** items 23/24 run the M2 backtest twice (~100 min
  today) and are red on runtime until G3/G4. I mark G1 and G2 done only after a full run; that
  first full run comes once G3/G4 are in. Each deliverable still gets the fast gate plus every
  item it touched.

## G1 evidence (2026-09-26)
Item 32 on the development shots (385,604): spline 15 flag/level columns (26 levels, make rates
0.3613–0.5754); LightGBM 5 columns (29 levels, 0.0256–0.9378; the extremes are zone J beyond
half court and zone A at the rim, real geometry). Guard: the three feed flags audited as
features are caught (fastbreak 0.9997, second_chance 0.9992, points_off_turnover 0.9997).

Deliberate breaks, each reverted right after (tree clean, check green again):
- `fastbreak` put back in the LightGBM builder under its own name → the name guard raises at
  import (`ValueError: identity or outcome-coded column in the M2 features: (..., 'fastbreak')`).
- the same column renamed `transition` (the name guard cannot see it) → item 32:
  `OUTCOME-CODED transition = 1: 11537 shots, make rate 0.9997…` → FAIL, exit 1.
- `second_chance` added to the spline builder as `putback` → item 32:
  `OUTCOME-CODED putback = 1: 17459 shots, make rate 0.9992…` → FAIL, exit 1.

## Loop log
iteration 1 | G1 flags dropped for good + outcome-coding audit (item 32) | tests/test_feature_audit.py (6), fast gate 1-6, 14, 19, 32 (green after a format fix), items 23/28/29 tests | green; full §6 pending (see sequencing) | 62b9526
