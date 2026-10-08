# Weeks 16–18 progress: M6 player projections and scouting, read-only API, pages from the API

Prompt: `docs/history/prompts/week-16-18.md`. Branch `week-16-18` from `main` 6b2239b (`origin/main` merged:
already up to date on 2026-10-07). Orchestrator Opus 5.5; subagents Sonnet 5.5 in `C:\Python\EH-w16-{a..j}`.

## §0 answers (owner, 2026-10-07)
- Accepted as written: L-a, L-b, L-c, L-d, L-f, L-g, L-h, L-i, L-l, L-m.
- **L-e:** no age bucket in any published output. The GBL undervalued list's "young" filter is "seasons since first
  senior appearance ≤ 3" (`season − debut_season ≤ 3`, `models/player_seasons.py`).
- **L-j:** `fastapi` approved as a runtime dependency (`uvicorn` in the dev group only); added in L0. The API is
  **not** deployed publicly: a `Dockerfile` + `docker compose up` give the one-command local run.
- **L-k (owner reversal):** a **Standings projection page is built**. Its data is a daily *ungated* M7 run:
  a new command writes `reports/sim_ungated_{competition}.json` (rewritten each day, never appended), and the
  page labels it "not gated: M7 failed its validation gate on the point rule (calibrated)". M7's gate, the
  append-only simulation log, `simulate` and every existing M7 output stay unchanged; failed-gate numbers never
  enter the pre-registered record. See D1.
- Stop condition: the full §7 checklist in one run (default), unless the owner later chooses a fast-gate record.

## Decisions §0 did not cover
- **D1 (ungated standings command, owner-approved design 2026-10-07):** the daily run is a separate command,
  `eurohoops sim-ungated`, not a `simulate --ungated` flag. Checklist item 52 (`scripts/checks/sim_dry_run.py`,
  lines 101–104) requires that `daily.yml` contains `eurohoops simulate` iff the M7 gate passed; items 1–53 must
  stay unchanged, and with a separate command item 52 still states the truth: the gated, append-only `simulate`
  is not scheduled. `sim-ungated` reuses `live_sim.run_live_sim` (new caller only, no change to `live_sim`) and
  writes only `SIM_UNGATED[competition]` (`config.py`).
- **D2 (factual error in the prompt, reported, not changed):** §2 says "M4 PASS (pooled small-sample rule)". The
  committed `reports/backtest_m4.json` says `gate.passed = false` (pooled 2019–2023, 6 movers, loss diff −0.137504,
  95% CI [−0.429934, 0.018067]) and `docs/models/m4.md` says FAIL. M6 still uses M4's factors as declared
  (`proj_full`, L-f); whether translation helps is decided on tuning like every variant (L-i).
- **D3 (shared input frame, written in L0):** `models/player_seasons.py` (interface I1 below) is the one
  player-season table L1–L4 read; it is built from the same player games M3/M4/M5 use (`stats.box`,
  `parse.gbl_box_lines`) and the `player_xwalk` mart.
- **D4 (impact inputs):** next-season BRAPM inputs are the committed season-end snapshots in `reports/m3_players.json`
  (seasons 2011–2025, each cut at its own season's last-round tip-off: walk-forward for every later target; the
  target season's own snapshot is the target value). No M3 rebuild is needed for next-season targets. At
  rest-of-season checkpoints no BRAPM snapshot exists at the cutoff: BRAPM is projected from the previous
  season-end snapshots only, flagged `no_impact_input` when there is none; SPM at checkpoints is computed from
  the box rates with the SPM model fitted before the target season (`eval.m3_gbl_backtest.el_spm_models` +
  `choose_model`, new callers only).
- **D5 (translation for targets without an M4 fit):** `fits_by_target_season` covers 2019–2025. Targets 2016–2018
  have no fit, and no GBL→EL mover can exist for them (GBL box scores start in 2018-19), so no fallback is needed;
  the harness asserts it. Live targets (2026) use the 2025 fit (pairs strictly before 2025: walk-forward-safe).
- **D6 (drift noise "on tuning only", L-c):** for a tuning target t the season-to-season drift variance is fitted
  on history before t; for every validation and test target it is frozen at the fit before the first validation
  season (2023). `projection.fit_drift(history, before_season)` is the one function for both.
- **D7 (xPTS for the board):** `shot_xpts` covers 2007–2025 (M2's split labels), not the live season; the live
  board's shot-making needs M2's model applied to 2026 shots (L9, orchestrator). M2's committed shot-making prior
  is estimated on 2011–2022, so the board's backtest helper uses synthetic data only (L3), and the live board
  re-estimates its prior walk-forward.
- **D8 (fast gate speed, owner 2026-10-07):** `pytest-xdist` added (dev group only) and checklist item 5 runs
  `pytest -n 6` (same tests, same coverage floor; 993 passed, 92.90%, 4 min 31 s vs ~20 min). The per-iteration
  fast gate also skips 43, 44, 48, 49 (their test files run inside item 5):
  `SKIP="7 8 10 12 13 15 16 18 20 21 22 23 24 25 26 30 31 35 36 37 38 39 41 43 44 45 46 48 49 50 51"`. The full
  §7 checklist is unchanged otherwise and runs before any deliverable is marked done.
- **D9 (pre-existing, reported to the owner):** item 48's `m7_facts.py` crashes in this tree:
  `data/raw/euroleague/standings/E2016_r30.xml` does not exist (M7's K0 cached the official tables inside the
  removed `C:\Python\EH-m7` worktree). Item 48 still prints PASS because `| tail -1` hides the script's exit
  code. **Resolved (owner, 2026-10-07):** the 8 official final tables (EuroLeague 2016–2018, 2020,
  2022–2025) were re-fetched once from `api-live.euroleague.net/v1/standings` (2 s apart) into the fixture's cache
  paths; each equals `tests/fixtures/m7_official_tables.json`, and `m7_facts.py` exits 0 (m7 facts: PASS).
- **D10 (aging survivor weights, amends I3):** with `survivor_correction=True` every pair is weighted by its
  season-s sampling variance only (1/(var_s + tau_d²)), never by the s+1 exposure, and the s+1 rate of a short
  season is EB-shrunk toward the same-competition, same-age mean of the short s+1 rates; the SE uses the shrunk
  posterior variance. A weight that grows with the s+1 exposure under-represents the decliners (who get the short
  seasons), so the harmonic-mean weights of I3 left part of the survivor bias in (subagent B, round 1). With the
  correction off the uncorrected harmonic weights stay (the planted bias must show).
- **D11 (loss standardisation, walk-forward and fixed):** each stat's between-player SD for the projection
  loss (L-g) is the possession-weighted SD of that stat over player-seasons with ≥ 500 possessions in the seasons
  before the first tuning target (EuroLeague 2008–2015; GBL 2018, the one box season before its first target).
  One fixed SD per stat and competition for every target: comparable across targets and never touching a target
  season (§1 walk-forward). Impact SDs: the same rule on the impact rows (EuroLeague BRAPM snapshots 2011–2015).
- **D12 (rest-of-season targets):** checkpoint f of season t: k = ⌊f·R⌋ completed regular-season rounds (M7's
  rule), cutoff = first tip-off of round k + 1. History = every game before the cutoff (the current season as a
  `partial` row); truth = the player's rates over the regular-season games at or after the cutoff; scored set =
  ≥ 500·(1 − f) possessions after the cutoff and a senior season before t (the L-g floor scaled to the window).
  Reference exposure for the interval = the player's possessions before the cutoff in season t, scaled by
  (1 − f)/f. Impact at checkpoints: SPM truth and inputs from box rates (D4); BRAPM not scored at checkpoints
  (no snapshot exists at the cutoff; reported as not available).
- **D13 (impact truth for next-season targets):** BRAPM truth = the target season's committed season-end snapshot
  (`m3_players.json`, D4), scored with its own sd ignored; SPM truth = the SPM model fitted before the target
  season applied to the target season's box rates. Both enter the loss and CRPS only for EuroLeague targets.
- **D14 (impact measurement noise, amends I2; found by F on synthetic data before any real-data run):** I2 gave
  the impact stats no target-season measurement term, but their truth is itself a noisy measurement (a season-end
  BRAPM snapshot; the SPM of one season's box line). On F's synthetic world 80% intervals covered BRAPM ~4% and
  SPM ~70%. Fix: `project(..., impact_noise={stat: (a, b)})` adds `a + b / exposure` to the impact predictive
  variance; the harness passes SPM `(0, u)` with `u` = F's split-half SPM noise unit fitted before the target, and
  BRAPM `(possession-weighted mean of sd² of the snapshots before the target, 0)`. Decided on synthetic evidence
  only, before the tuning run (pre-registration intact).
- **D15 (group stages only; amended before any number was scored):** history, truth and exposure use the
  group-stage games (`GROUP_PHASES = ("RS", "TS")`: the regular season and the EuroLeague Top 16 of 2007-2015),
  never the knockouts (PO, PI, FF). F's first version kept `RS` only, which dropped the 688 Top 16 games and left
  26 EuroLeague player pairs with 500+ possessions in consecutive seasons before 2016 (the aging curve could not be
  fitted); found by the first tuning attempt, which stopped in `aging_curve` before any projection or score
  existed.
  R = the season's last played regular-season round (equals M7's R except in 2019-20, which was cut short).
- **D16 (implausible birth dates, found by the real-data wiring probe):** 5 GBL players have a source birth
  date that gives an age under 15 in a season they played (placeholders on their ESAKE pages; counted, never
  printed). `player_ages` drops the date of any person whose age falls outside 15–45 in any season (`AGE_RANGE`):
  he takes the no-age path (`no_age` flag). None of the 5 is in a scored set: `m6_facts.py` (now counting
  plausible dates only) still finds 100% coverage, fixture unchanged.
- **D17 (BRAPM snapshot noise; tuning attempt 2 evidence, before any verdict):** M3's snapshot `sd_total` is a
  posterior sd, not the noise of a raw measurement: snapshots spread less (sd 1.74 in 2017) than their posterior sd
  (mean 2.28), so I2's empirical-Bayes prior variance came out negative, was clipped to 0, and every BRAPM weight
  was 0 (the projection was the league mean; tuning MAE 1.60 vs 0.74 for same_as_last; consecutive snapshots
  correlate 0.885). Fix: the snapshot noise N and drift D are estimated from the snapshots before the target by the
  method of moments on lag-1 and lag-2 differences (random-walk truth plus white noise:
  Var(x_{s+1} − x_s) = D + 2N, Var(x_{s+2} − x_s) = 2D + 2N), possession-qualified rows only; BRAPM rows enter
  `project` with sd = √N, `impact_noise["brapm"] = (N, 0)` and the BRAPM drift = D.
- **D18 (aging regression to the mean; tuning attempt 2 evidence, before any verdict):** the delta method on raw
  season-s rates counts the regression of players selected on a lucky season as aging (2018 curve: pts −0.78 per
  year at 29; every projection aged down ~1.0 pts per 100). Fix: each pair's delta is measured from the
  empirical-Bayes-shrunk season-s rate (shrunk toward that season's league mean by its exposure, EB variance from
  that season's qualified rows), the regressed delta method. Amended by subagent B (round 3), accepted: the shrinkage prior is
  estimated on every row of the competition-season (the qualified rows are the lucky ones: their own mean biased
  the no-aging test by −0.2 per year), and its mean includes the age effect of the pre-cutoff rows (young players
  sit below the league mean and old ones above it; shrinking to the plain mean turned that gap into aging and broke
  the unchanged planted-curve test). `aging_curve(..., regress_season_s=False)` keeps the raw method for the test.
- **Tuning attempts (recorded, nothing committed):** attempt 1 stopped in `aging_curve` (D15 Top 16 bug) before any
  projection existed. Attempt 2 (HEAD 3ed1195, 118 s, tuning seasons only, no validation/test number anywhere)
  scored: marcel 6.1902, proj_full_spm@2 6.4206, proj_full_spm@1 6.4448, proj_shrunk@2 6.8489, proj_age@2 7.0827,
  same_as_last 10.1897, league_mean 13.6611; BRAPM MAE variants 1.5956 vs marcel 0.8211; box-stat MAE below marcel
  on 7 of 12 stats. Its outputs were discarded; D17 and D18 are method fixes on that tuning evidence, made before
  the verdict, and the tuning run is repeated after them.

## §3 findings (L0, real data, HEAD 6b2239b; re-checked by `scripts/checks/m6_facts.py`, item 54)
Fixture: `tests/fixtures/m6_facts.json`.
1. **Possessions:** every player game with seconds has a possession count: EuroLeague 107,692 player games
   (2007–2026), GBL 26,832 (2018–2026). Player possessions = team possessions × minutes share (as M3/M4).
2. **Scored-set sizes** (≥ 500 possessions in the season and a senior season before it in either league):

   | season | EL ≥ 500 | EL scored | GBL ≥ 500 | GBL scored |
   |---|---|---|---|---|
   | 2008 | 115 | 82 | | |
   | 2009 | 110 | 85 | | |
   | 2010 | 119 | 94 | | |
   | 2011 | 117 | 93 | | |
   | 2012 | 149 | 120 | | |
   | 2013 | 147 | 122 | | |
   | 2014 | 152 | 119 | | |
   | 2015 | 145 | 126 | | |
   | 2016 | 159 | 141 | | |
   | 2017 | 160 | 128 | | |
   | 2018 | 164 | 128 | 100 | 26 |
   | 2019 | 168 | 128 | 74 | 46 |
   | 2020 | 184 | 167 | 100 | 75 |
   | 2021 | 176 | 150 | 118 | 80 |
   | 2022 | 196 | 158 | 105 | 68 |
   | 2023 | 194 | 165 | 112 | 71 |
   | 2024 | 193 | 157 | 112 | 75 |
   | 2025 | 230 | 181 | 116 | 81 |

   EuroLeague tuning targets 2016–2022: 1,000 player-seasons; validation 2023: 165; test 2024–2025: 338.
   **2019-20** (the COVID-shortened season) has 128 scored players: it passes the floor and stays in tuning.
   GBL targets 2019–2025 (2018-19 is the first box season, so 2019-20 is the first GBL target with a prior GBL
   season).
3. **Birth dates:** known for 100% of scored player-seasons in both leagues (every season). Unknown ages
   (live newcomers) take the no-aging path with the `no_age` flag. The live season's newcomers are not in the
   crosswalk (built on 2007–2025): 136 EuroLeague and 53 GBL 2026 player-game rows; they get `P:<id>` / `G:<id>`
   person ids with `mapped = False`.
4. **M4 walk-forward:** fits exist for EuroLeague targets 2019–2025 (D5).
5. **M3 walk-forward:** BRAPM snapshots 2011–2025 in `m3_players.json`, each cut no later than its own season's
   last game (D4). SPM models per season come from `el_spm_models` (M3 GBL backtest builds them in ~72 s total).
6. **Route ↔ page table** (`web/src/data/**`; every file is read by a page):

   | file | written by | read by |
   |---|---|---|
   | `site.json` | `publish` | `pages/index.astro` |
   | `stats/meta.json` | `export-stats` | `lib/stats.ts meta()` → every stats page, `data/*` endpoints |
   | `stats/players.json` | `export-stats` | `playerIndex()` → `players/[slug]`, `compare`, `data/cards`, `data/player`, `data/names` |
   | `stats/splits.json` | `export-stats` | `players/[slug]` (splits) |
   | `stats/twins.json` | `export-stats` | `twinData()/twinPool()` → `players/[slug]`, `data/twins.json` |
   | `stats/seasons/{s}/players.json` | `export-stats` | `players/index`, `leaders`, `compare`, `data/players`, `data/leaders` |
   | `stats/seasons/{s}/teams.json` | `export-stats` | `teams/index`, `teams/[code]`, `data/team`, `data/season` |
   | `stats/seasons/{s}/games.json` | `export-stats` | `players/[slug]` game log, `data/player` |
   | `stats/seasons/{s}/shots.json` | `export-stats` | `players/[slug]`, `teams/[code]` shot charts |
   | `stats/seasons/{s}/attempts.json` | `export-stats` | `shots/index`, `data/shots` |

   No unread file.
7. **Four factors and schedule strength:** derivable. The player box of both leagues carries FGM/FGA by type,
   FTA, OREB, DREB, TOV and possessions (team sums give eFG%, TOV%, ORB% with the opponent's DREB, FT rate);
   schedule strength = mean opponent rating (Elo, from the published ratings) over played and remaining games.

## Interfaces (fixed in L0; pasted verbatim into every wave-1 prompt)

### I1 Player-season frame — `src/eurohoops/models/player_seasons.py` (orchestrator, done)
`build_player_seasons(player_games: Mapping[str, DataFrame], xwalk, *, partial: Mapping[str, int] | None) ->
DataFrame` validated by `PLAYER_SEASONS_SCHEMA` (strict, unique on person_id, competition, season):

| column | dtype | meaning |
|---|---|---|
| person_id | str | crosswalk person (`P:<EL id>` / `G:<ESAKE id>`) |
| competition | str | `euroleague` or `gbl` |
| season | int64 | start year |
| partial | bool | season cut at a checkpoint (only rows built from games before a cutoff) |
| mapped | bool | the source id is in the crosswalk |
| team | str | source code of the team with the most seconds (ties: smallest code) |
| games | int64 | games with seconds > 0 |
| minutes | float64 | |
| poss | float64 | possessions on court (exposure) |
| pts fg2m fg2a fg3m fg3a ftm fta oreb dreb ast stl blk tov pf | int64 | season counts (`COUNT_COLUMNS`) |
| debut_season | int64 | first season of the person in either league among the rows passed |

Rates: per 100 possessions = `100 * count / poss`. Percentages: `ts = pts / (2 * tsa)` with
`tsa = fg2a + fg3a + 0.44 * fta`; `fg3 = fg3m / fg3a`; `ft = ftm / fta`.
Shared constants (import them; do not redefine): `COUNT_STATS = ("pts", "fg3a", "fta", "ast", "tov", "oreb",
"dreb", "stl", "blk")`, `PCT_STATS = ("ts", "fg3", "ft")`, `IMPACT_STATS = ("spm", "brapm")`,
`PROJECTED_STATS = COUNT_STATS + PCT_STATS + IMPACT_STATS`, `TS_FTA_WEIGHT = 0.44`.
`rate_table(seasons) -> DataFrame`: keys person_id, competition, season, partial; for each stat in COUNT_STATS +
PCT_STATS a column `<stat>` (per-100 rate, or proportion; NaN without an attempt) and `<stat>_n` (its exposure:
possessions, or tsa / fg3a / fta).
`player_ages(bios, xwalk, seasons) -> DataFrame` (`AGES_SCHEMA`: person_id, season, age float64 at 1 October):
**in memory only, never written to any file, report, fixture or page.**

### I2 Projections — `src/eurohoops/models/projection.py` (subagent A)
```python
from eurohoops.models.player_seasons import COUNT_STATS, PCT_STATS, IMPACT_STATS, PROJECTED_STATS, rate_table
AgeAdjust = Callable[[str, FloatArray, FloatArray], FloatArray]  # (stat, age_from, age_to) -> change in the rate;
                                        # the harness passes functools.partial(aging.apply, curve) (I3)
FLAGS = ("no_history", "no_age", "translated", "partial_season", "no_impact_input")
VARIANTS = ("proj_shrunk", "proj_age", "proj_full", "proj_full_spm")

@dataclass(frozen=True)
class ProjectionParams:
    half_life: float                    # seasons; a season k seasons before the target weighs 0.5 ** (k / half_life)
    aging: bool                         # proj_age, proj_full, proj_full_spm
    translation: bool                   # proj_full, proj_full_spm
    impact_prior: Literal["league", "spm"]   # "spm" only in proj_full_spm (BRAPM prior mean = the projected SPM)
    interval: float = 0.8

def variant_params(variant: str, half_life: float) -> ProjectionParams

@dataclass(frozen=True)
class Translation:                      # M4 factors, passed in, never fitted here
    delta: Mapping[str, float]          # stat -> log(EL / GBL) δ of M4's fit for the target season
    c: Mapping[str, float]              # stat -> M4 pseudocount
    target_season: int                  # the M4 fit's target season (pairs strictly before it)
    # to EuroLeague: (rate + c) * exp(δ) - c; to GBL: (rate + c) * exp(-δ) - c; PCT/IMPACT stats are not translated

def fit_drift(history: DataFrame, before_season: int) -> dict[str, float]   # per PROJECTED_STATS stat, season-to-season
    # variance of true rates, from consecutive complete seasons strictly before `before_season` (D6)

def project(
    history: DataFrame,                 # I1 rows; project itself drops what the target may not use (below)
    target: DataFrame,                  # TARGET_SCHEMA: person_id, competition, season, checkpoint, exposure
    params: ProjectionParams,
    *,
    drift: Mapping[str, float],
    impact: DataFrame | None = None,    # IMPACT_SCHEMA: person_id, competition, season, stat (IMPACT_STATS), value, sd
    ages: DataFrame | None = None,      # I1 AGES_SCHEMA
    aging: AgeAdjust | None = None,     # required when params.aging
    translation: Translation | None = None,  # required for a target whose usable rows include the other league
) -> DataFrame                           # PROJECTIONS_SCHEMA
```
Usable rows for a target (season t, checkpoint c): every I1 / impact row with `season < t` (partial or not), plus,
when `c > 0`, the row of season t in the target's competition with `partial = True`. Rows with `season > t`, or
`season == t` and not (`c > 0` and partial), change nothing (the L1 leakage test). `checkpoint` is 0.0 for a
next-season target and the fraction (0.25 / 0.5 / 0.75, or the live fraction) for rest-of-season. The partial
current season has k = 0. `exposure` is the reference possessions for the predictive interval (the harness passes
the player's possessions in his last usable season; minutes are not projected, L-b).

League prior per stat and target competition: the possession-weighted mean of the usable rows of that competition
in seasons t−3…t−1 (for counts and percentages), between-player prior variance by empirical Bayes (method of
moments: weighted variance of observed rates minus mean sampling variance) on the same rows; no grid. Each usable
season's rate is (a) translated to the target league when it is the other league and `params.translation`
(flag `translated`), (b) aged to the target age with `apply(aging, stat, age_row, age_target)` when
`params.aging`, i.e. `aging(stat, age_row, age_target)` (no age → no adjustment, flag `no_age`), then blended with weights decay × exposure (possessions for
counts, attempts for percentages) and shrunk to the prior by the empirical-Bayes weight. Predictive variance =
posterior variance + drift (scaled by the gap in seasons to the newest usable season, at least 1 for next-season
targets) + target-season sampling variance at `exposure`. lo80 / hi80 are the (1 ∓ interval)/2 quantiles of a
Normal predictive, clipped at 0 below.

Impact stats (`spm`, `brapm`): each usable impact row of the target's competition is a measurement with variance
`sd²`; rows are blended with weights decay / sd² (no translation, no aging unless the curve carries the stat) and
shrunk to the prior: `impact_prior="league"` → the possession-weighted mean of the usable impact rows in t−3…t−1
(EB prior variance as above); `"spm"` → for `brapm` only, the target's own projected `spm` mean. No usable impact
row → mean = prior, flag `no_impact_input`. The predictive variance has no sampling term for impact.

`TARGET_SCHEMA`: person_id str, competition str, season int64, checkpoint float64 in [0, 1), exposure float64 > 0.
`IMPACT_SCHEMA`: person_id str, competition str, season int64, stat str in IMPACT_STATS, value float64, sd float64 ≥ 0.
`PROJECTIONS_SCHEMA` (long; unique on person_id, competition, season, checkpoint, stat):

| column | dtype | meaning |
|---|---|---|
| person_id, competition, season, checkpoint | str, str, int64, float64 | the target |
| stat | str | in PROJECTED_STATS, in that order within a target |
| mean, sd, lo80, hi80 | float64 | predictive mean, sd, interval |
| prior_mean | float64 | the league (or SPM) prior mean |
| weight | float64 in [0, 1] | shrinkage weight on the player's own data |
| exposure | float64 ≥ 0 | decayed possessions (counts, impact) or attempts (percentages) behind the mean |
| n_seasons | int64 | usable seasons with exposure > 0 |
| flags | str | `|`-joined sorted subset of FLAGS, `""` for none |

### I3 Aging curve — `src/eurohoops/models/aging.py` (subagent B)
```python
@dataclass(frozen=True)
class AgingCurve:
    cutoff_season: int                  # fitted on pairs (s, s+1) with s + 1 < cutoff_season, complete seasons only
    ages: FloatArray                    # integer ages ascending (e.g. 19 ... 37); ends set by data with >= 30 pairs
    delta: Mapping[str, FloatArray]     # stat -> expected change in the rate from age a to a + 1, at each ages[i]
    se: Mapping[str, FloatArray]        # stat -> its standard error
    n_pairs: Mapping[str, IntArray]

def aging_curve(
    history: DataFrame, ages: DataFrame, cutoff_season: int, *,
    impact: DataFrame | None = None, min_poss: float = 500.0, survivor_correction: bool = True,
) -> AgingCurve      # stats = PROJECTED_STATS (impact ones only when `impact` is given)

def apply(curve: AgingCurve, stat: str, age_from: FloatArray, age_to: FloatArray) -> FloatArray
    # summed change from age_from to age_to (piecewise-linear in fractional ages; beyond the ends the end delta)
```
Pairs are same person, same competition, consecutive complete seasons; the age of a pair is the age in season s.
Rate units as I2 (per 100, proportion, points per 100). Delta method, harmonic-mean-of-exposure weights, a
hierarchical (partial-pooling) estimate per age, smoothed with a quadratic spline. Survivor correction: pairs where
season s qualifies (≥ min_poss) but season s+1 has fewer possessions are kept with their (shrunk) s+1 rate and
harmonic weights instead of being dropped (`survivor_correction=False` drops them: the planted bias must appear).
Imports the stat constants and `rate_table` from `models/player_seasons.py` (I1).
No birth date in any field; ages only.

### I4 Over/under board — `src/eurohoops/models/board.py` (subagent C)
```python
DIMENSIONS = ("shot_making", "fg3_pct", "on_off")
LABELS = ("likely regression", "likely real", "too few attempts")
N_MIN = {
    "shot_making": 200,
    "fg3_pct": 50,
    "on_off": 1000,
}  # FGA, 3PA, on-court possessions (declared)
```
`BOARD_SCHEMA` (unique on person_id, competition, season, dimension):
person_id str, competition str, season int64, dimension str, observed float64, expected float64,
gap float64 (= observed − expected), se float64 > 0, z float64 (= gap / se), stability float64 (year-to-year r
of the dimension, from tuning seasons only), persist float64 in [0, 1], expected_next float64
(= expected + persist · gap), n float64 (attempts or possessions), label str in LABELS.
Rule: `n < N_MIN[dim]` → "too few attempts"; else `persist = n / (n + k_dim)` with
`k_dim = n̄_dim · (1 − r_dim) / r_dim` (n̄: mean n of the stability pairs); "likely real" iff `persist ≥ 0.5` and
`|z| ≥ 1.645`, else "likely regression". Dimensions: shot-making = (points − xPTS) per 100 FGA vs 0 (M2's
`shots` + `shot_xpts`); 3P% vs a prior from FT% (a regression of 3P% on FT% fitted on seasons before the cutoff);
on/off = on-court minus off-court net rating per 100 from the `stints` mart vs the BRAPM total (EuroLeague only).

### I5 "Plays like" similarity — `src/eurohoops/models/similarity.py` (subagent D)
```python
BOX_FEATURES   # per-100 rates of pts fg2a fg3a fta oreb dreb ast stl blk tov pf, plus ts, fg3, ft, and
               # usage = 100 * (fg2a + fg3a + 0.44 * fta + tov) / poss
SHOT_FEATURES  # zone shares of FGA (stats.twins.shot_zones' zones) and shot-making per 100 FGA (EuroLeague only)

@dataclass(frozen=True)
class Embedding:
    keys: DataFrame          # person_id, competition, season (row order of matrix)
    features: tuple[str, ...]
    mean: FloatArray         # from the reference pool only
    sd: FloatArray
    matrix: FloatArray       # standardised features

def embed(seasons: DataFrame, shots: DataFrame | None, *, pool_seasons: Sequence[int],
          min_poss: float = 500.0) -> Embedding
def neighbours(query: DataFrame, emb: Embedding, k: int = 10) -> DataFrame   # SIMILAR_SCHEMA
```
`SIMILAR_SCHEMA` (unique on person_id, competition, season, rank): person_id, competition, season, rank int64 1..k,
match_person_id, match_competition, match_season int64, distance float64 ≥ 0, score float64 (cosine similarity).
A person never matches any of his own seasons. Standardisation uses the pool rows only. When the query or the pool
lacks shot features (GBL), only BOX_FEATURES are used. `stats/twins.py` is untouched.

### I6 Read model and API — `src/eurohoops/api/` (subagent E)
`api/readmodel.py`: pure functions (inputs → JSON-ready dicts), pandera-checked inputs; `api/app.py`: FastAPI,
read-only, `create_app(store: Store, now: Callable[[], datetime]) -> FastAPI`. DuckDB opened `read_only=True`.
Every response body is `json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"` (UTF-8): the exact
bytes `stats.export.write_stats` writes, so the static export can write the response bytes as they are. `site.json`
is the one exception, written today by `logs.write_json` (indent 2): the `/site` route returns those bytes.

| route | read-model function | source |
|---|---|---|
| `GET /site` | `site(store, now)` | `publish.site_data` (today's `site.json`) |
| `GET /stats/{path}` (e.g. `meta.json`, `seasons/2025/players.json`) | `stats_file(store, path, now)` | `stats.export.build_payloads` |
| `GET /games/upcoming?competition=` | `upcoming_games(store, competition, now)` | games mart + prediction logs |
| `GET /predictions/{game_id}` | `prediction(store, game_id)` | `predictions/*.csv` (pre-registered row + later rows) |
| `GET /teams/{competition}/{team}/ratings` | `team_ratings(store, competition, team)` | published Elo / M1 ratings |
| `GET /teams/{competition}/{team}/factors?season=` | `team_factors(store, competition, team, season)` | four factors, schedule strength |
| `GET /players/{person_id}` | `player(store, person_id)` | I1 frame + `m3_players.json` + `m2_players.json` |
| `GET /players/{person_id}/projection` | `player_projection(store, person_id)` | `reports/m6_projections.json` |
| `GET /players/{person_id}/similar` | `player_similar(store, person_id)` | `reports/m6_similarity.json` |
| `GET /scouting/board` | `scouting_board(store)` | `reports/m6_board.json` |
| `GET /scouting/undervalued` | `scouting_undervalued(store)` | `reports/m6_projections.json` (`undervalued`) |
| `GET /scouting/translation` | `scouting_translation(store)` | `reports/m4_translation.json` + `backtest_m4.json` gate |
| `GET /simulations/latest?competition=` | `simulation_latest(store, competition)` | `reports/sim_ungated_{c}.json`, 200 with `"gated": false` and M7's gate verdict; 404 with the verdict when absent |
| `GET /metrics/live` | `metrics_live(store)` | `reports/live_scorecard*.json`, M5 shadow scorecard |
| `GET /openapi.json` | FastAPI | stable across runs |

404 bodies: `{"detail": "<reason>"}`. No route fits a model.

### Report shapes the API reads (written by L8/L9; E builds fixtures with exactly these keys)
- `reports/m6_projections.json`: `{"model": "m6", "variant": str, "half_life": float, "gated": bool,
  "gate_passed": bool, "season": int, "checkpoint": float, "data_sha256": str, "stats": [PROJECTED_STATS],
  "players": [{"person_id", "competition", "team", "name", "debut_season", "stats": {stat: {"mean", "lo80",
  "hi80"}}, "exposure", "n_seasons", "flags": [str]}], "undervalued": [{"person_id", "team", "name",
  "seasons_since_debut", "minutes", "projected_brapm" | null, "projected_spm", "translated_el": {stat: float},
  "reason": str}]}`. No birth date, no age.
- `reports/m6_board.json`: `{"model": "m6", "season": int, "dimensions": {dim: {"stability", "n_min", "k"}},
  "rows": [BOARD_SCHEMA row + "name", "team"]}`.
- `reports/m6_similarity.json`: `{"model": "m6", "season": int, "features": {"euroleague": [...], "gbl": [...]},
  "pool_seasons": [int], "players": {person_id: [{"rank", "person_id", "competition", "season", "name",
  "score"}]}}`.
- `reports/sim_ungated_{competition}.json`: `live_sim.latest_report`'s shape plus `"gated": false,
  "gate": {"passed": false, "reason": str}`.

## Iterations
iteration 1 | L0 branch, progress file, decisions, §3 facts, interfaces, `player_seasons.py`, M6 config, fastapi + pytest-xdist | m6_facts.py PASS (36 checks); fast gate (D8 skip list) FAILS 0 in 441 s; ruff/format/mypy/vulture clean | green | (this commit)
iteration 2 | wave 1 merged: L1 (A), L2 (B, round 2), L3 (C), L4 (D), L5 (E) | each Done-when test file rerun in the main tree after merge (projection/aging/board/similarity/player_seasons 65 passed, API 85 passed), ruff, format, mypy, vulture clean, `docker compose config` valid; fast gate deferred while F runs | green | merges ff1bae2 310c5fb d69573c 3436934 fd9a628
iteration 3 | L6 harness (F) merged; fast gate | F's tests 42 passed in the main tree; the interrupted gate (session end) showed 4 F in item 5, a clean rerun of the whole suite 1180 passed (cause not identified: logged, watched); fast gate (D8 list) FAILS 0 in 649 s, item 5 1180 passed 93.37%; possessions.json restored | green | merge 71a7df4
iteration 4 | D14 (A round 2) merged; SPM wiring hardened (a season without its own SPM model raises, f7d22be); L7 leakage suite (G) merged | projection + harness + leakage files 104 passed in the main tree on the merged code; ruff, format, mypy, vulture clean | green | merges 3db32ab f7d22be fe1dc58
iteration 5 | tuning attempts 1-2 (no verdict; recorded), D15 group stages (3ed1195), D16 implausible birth dates (675d81a), D17 BRAPM snapshot noise (A round 3), D18 regressed aging deltas (B round 3) | harness, leakage, projection, aging tests rerun in the main tree after each merge; ruff, mypy clean | green | 3ed1195 675d81a merges 47cb308 ebe5231

## Subagent log
| wave | subagent | deliverable | rounds | notes |
|---|---|---|---|---|
| 1 | D | L4 similarity | 1 | `tests/test_similarity.py` already existed (name matcher), so its tests are `tests/test_player_similarity.py`; self-retrieval through `self_retrieval_rate` (evaluation only) |
| 1 | C | L3 board | 1 | whitelist merge conflict resolved (both blocks kept) |
| 1 | A | L1 projection | 1 | coverage test on 7 of the 12 box stats with the true target exposure (real-data coverage is the L-h check); `fit_drift` gained an optional `impact` argument |
| 1 | E | L5 read model + API | 1 | `docker build` not run (daemon off); `docker compose config` valid; `/simulations/latest` reason text built from the committed gate numbers (`passed` read from the report); `tests/api_tree.py` shared test helper outside its file list (accepted) |
| 2 | F | L6 harness | 1 | no whitelist block needed; found the impact-noise gap of I2 (D14); R from played rounds; CSV holds the chosen cell + baselines only |
| 1 | A | D14 (round 2) | 1 | `impact_noise` added at output assembly; harness passes SPM (0, u) and BRAPM (a, 0) |
| 2b | G | L7 leakage | 1 | no real leak; scored-set floor lowered to 1 possession in the suite (scoring side only) so whole frames compare exactly; 35 tests, ~40 s at -n 6 |
| 2 | A | D17 (round 3) | 1 | (N, D) by lag-1/lag-2 moments, memoised per freeze season |
| 2 | B | D18 (round 3) | 1 | prior on all rows with an age effect (accepted, recorded under D18) |
| 1 | B | L2 aging | 2 | round 1 narrowed the synthetic possession gap (1200 → 1000) after a failure (rejected); root cause: outcome-dependent weights → D10; original 1200/450 design passes unchanged; impact-stat test added |
