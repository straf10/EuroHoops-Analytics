# Week 3 close-out (2026-09-25, branch `week-3-closeout`)

## Checklist
- [x] 2. GBL Elo tuning grid widened; live parameters frozen for 2026-27
- [x] 1. GBL live logging ready for Sat 3 Oct
- [x] 5. EL round-1 rows flagged as not provable
- [x] 3. GBL box-score gaps classified (fill policy: DECISION NEEDED)
- [x] 7a. EuroLeague shot coordinate system
- [ ] 7b. 2026-27 formats and tiebreak rules

## 2. GBL Elo tuning grid
The old grid's best value (K=40, HCA=130, reversion=0.25) sat on the edge of all three axes.
The GBL backtest now searches K ∈ {10…80} × HCA ∈ {0…260} × reversion ∈ {0…0.75}
(450 combinations; `src/eurohoops/config.py`), on the same splits as before
(warm-up 2018-21, tuning 2022-23, test 2024-25).

Top tuning-set results (log loss):

| K | HCA | reversion | tuning | test |
|---|-----|-----------|--------|------|
| 50 | 170 | 0 | 0.4124 | 0.4851 |
| 50 | 200 | 0 | 0.4127 | 0.4965 |
| 40 | 170 | 0 | 0.4131 | 0.4857 |
| 60 | 170 | 0 | 0.4140 | 0.4851 |
| 50 | 150 | 0 | 0.4144 | 0.4798 |
| **40** | **130** | **0.25** | 0.4313 | **0.4845** |

- The grid best (K=50, HCA=170, reversion=0) lies **inside** the grid on K and HCA. Reversion=0
  (no pull toward the mean) is a natural lower bound, not a grid limit.
- Grid best minus 40/130/0.25, paired bootstrap (1,000 resamples):
  tuning −0.0189 [−0.0323, −0.0050]; **test +0.0005 [−0.0194, +0.0234]**.
- **Decision: keep K=40, HCA=130, reversion=0.25** (the test CI spans 0), and freeze them for
  2026-27 via `Backtest.frozen`. The tuning-set gain doesn't carry over to test, which suggests
  the grid best overfits 2022-23. HCA=170 also implies a 73% home win at equal ratings,
  against 64% observed.
- Check: GBL test log loss 0.4845 (B0 0.6680) reproduces; model versions unchanged for all
  three reports (GBL `0.2.0+df05260c`, EL `0.2.0+425e6393`, EL history `0.2.0+db021063`).
  Each report now records `grid.best`, `grid.best_on_edge` and `grid.best_minus_tuned_log_loss`.
- Note: the EL live grid best also has reversion on the edge (0.25). EL parameters were
  frozen by an earlier decision (D-a), so this is only for information.

## 1. GBL live logging for Sat 3 Oct
- The daily workflow already ingests, predicts and scores both competitions. actionlint
  (1.7.12) passes on `daily.yml` and `ci.yml`. The new `tests/test_workflow.py` fails if a
  GBL step is removed or the daily cron stops being daily.
- Timing: round 1 has 7 games, all with confirmed dates, tipping off from Sat 3 Oct 12:00 to
  Sun 4 Oct 14:30 UTC. The earliest GBL tip-off in the whole 2026-27 schedule is 09:00 UTC.
  The daily run at 08:00 UTC has a 36 h window, so every game is covered by one or two runs
  before tip-off, including Sunday morning games (logged by Saturday's run).
- Dry run on the real mart with a fixed clock and a temporary log (scratch script, not
  committed): the Fri 2 Oct run logs 5 games and the Sat 3 Oct run logs 2. Rerunning both
  adds 0 rows. All 7 round-1 games are logged once, before tip-off, with the frozen model
  `0.2.0+df05260c` (K40/HCA130/rev0.25).
- `tests/test_cli.py` covers the GBL predict path end to end (twice → no duplicates).
- Watch: GitHub can delay scheduled runs. A delay of up to 12 h still leaves every game
  covered by at least one run. Check the Actions tab on Fri 2 Oct.

## 5. EL round-1 rows that aren't provable
- A local commit's timestamp proves nothing, so I used GitHub's server-side push times
  (`gh api repos/straf10/EuroHoops-Analytics/activity`):
  - 16:24:29 UTC 24 Sep: branch creation (`0b60463`), which already contained the 10 v0.1.0
    round-1 rows.
  - 20:29:32 UTC: the push containing the 3 v0.2.0 rows.
- **Only 2 of the 10 round-1 games aren't provable**: E2026_2 (DUB–MAD) and E2026_3 (HTA–MUN),
  both tipping off at 16:00 UTC, 24 minutes before the first push. The other 8 were public
  before tip-off. The earlier claim that all of round 1 was unprovable was too strict.
- `Competition.manual_pushes` records those two push times. A row became public at the first
  manual push at or after its stamp. Later rows are committed by the workflow in the run that
  stamps them.
- The scorecard now splits the counts:
  - Headline `elo`/`b0`: provable rows only.
  - `all_rows`: keeps everything.
  - `rows_not_provable` and `games_not_provable`: list what was dropped.
- The page marks those results with `*` and adds a "Not scored" note.
- The prediction logs are untouched (`git diff -- predictions/` is empty).
- Current EL scorecard (7 finished games): headline n=5, Elo log loss 0.555 vs B0 0.675;
  all_rows n=7.
- Local-only side effect, now reverted: a plain `eurohoops ingest` rewrites EL staging with
  the default seasons 2023-26 and drops the 2007+ history the history backtest needs.
  I rebuilt it from cache (`--seasons 2007 … 2026`, 5,502 games), and all backtest reports
  reproduce byte for byte. Worth knowing before running `ingest` locally.

## 3. GBL box-score gaps
Every affected game is listed with its category in `reports/gbl_box_gaps.csv` (103 rows).

| Group | 2018-19 | 2019-20 | 2022-23 | Category | PBP export exists and final score matches |
|---|---|---|---|---|---|
| Missing box score | 52 | 26 | 0 | (a) page without stats | 78/78 |
| One player missing from the box | 15 | 8 | 2 | (a) page without stats | 25/25 |
| Parser miss | 0 | 0 | 0 | (b) | — |
| Page missing | 0 | 0 | 0 | (c) | — |

- **Missing box score:** all 78 cached pages have the same shape: game header, quarter scores
  and team leaders (5–6 `idplayer` links, all in the `mvp-player-name` widget), and no stat
  table.
  - The `mode=2` view has only quarter scores and the empty PBP container.
  - A live refetch of 10 of them (6 from 2018-19, 4 from 2019-20; 2 s apart) matched the cache,
    so ESAKE hasn't added them since the crawl.
- **Points mismatches:**
  - These weren't in the original list, but they also make the "invariants pass for the games
    that are present" check fail. They are also a source issue.
  - In every one of the 25 games, one team lists 10–11 players, and ESAKE's own totals row
    equals the sum of those players.
  - That totals row is 1–23 points (median 8) below the result, and minutes total 168–219
    against 200/225. Every row on the page has a player link and is parsed.
- **Parser fixes: none needed**, so completeness is unchanged: 2018-19 at 63.6%, 2019-20 at
  73.4% (`reports/gbl_box_invariants.json` unchanged by `eurohoops build`).
- **Done-when check not met as written:** "invariants pass for all games that are present"
  fails for the 25 games above. The cause is ESAKE's data, and the parser can't fix it. The
  remaining `minutes_off` flags (2020-21 → 2025-26: 2–6 per season) are the known
  abandoned/overtime cases in `docs/data/gbl.md`.
- **Play-by-play:**
  - The PBP export path works for all 103 games. It needs `show_export_link=1` on the widget
    request to reveal the internal id (spike doc updated).
  - The last score in the sheet matches the result in 103/103.
  - A full points-from-events reconstruction is part of the weeks 5–7 PBP work.
- **DECISION NEEDED (fill policy):**
  - **Recommended:** fill from PBP when the GBL PBP ingester lands (team totals for the 78
    games, plus the missing player's line for the 25).
  - Until then, box-score models start at 2020-21, and 2018-20 is used for Elo/results only.
  - The alternative is to drop 2018-20 from box-score models permanently.

## 7a. EuroLeague shot coordinate system
Source: the `Points` endpoint (`COORD_X`, `COORD_Y`, cached in `data/raw/euroleague/points`).

- **System:**
  - Units are centimetres, measured from the basket centre.
  - Every shot is mirrored onto one basket.
  - x runs across the court: observed range −740…+746, sidelines at ±750.
  - y runs from the baseline toward half court: minimum −156, where the baseline is at −157.5
    because the basket sits 1.575 m in; maximum about 1,304, i.e. heaves just past half court.
  - Free throws carry the sentinel (−1, −1).
  - About 0.1% of shots sit exactly at (0, 0), which looks like missing coordinates.
  - Which side of the court x > 0 means (the shooter's left or right) is **UNVERIFIED**.
    It doesn't matter for distance-based models.
- **Conversion:** `src/eurohoops/parse/shots.py`. `to_court_coords()` returns metres and turns
  the sentinel into NaN. `beyond_three_line()` tests the FIBA line: the 6.75 m arc plus
  straight corners at |x| = 6.60 m up to y = 1.415 m.
- **Check, 2024-25 (41,533 field-goal attempts, ±0.15 m tolerance):** **99.87%** of labelled
  3s lie beyond the line and **99.90%** of 2s inside it (target ≥ 99%).
  Plot: `reports/el_shots_2024.png`.

  | Seasons | 3s beyond 6.75/6.60 | 2s inside |
  |---|---|---|
  | 2011-12 → 2025-26 | 99.63–99.94% | 99.65–99.96% |
  | 2007-08 → 2010-11 | 91.4–98.0% | 99.7–99.97% |

- **Pre-2011 geometry doesn't fit.**
  - In 2007-08 → 2010-11, 3s fit neither the 6.75 m line nor the old 6.25 m line. Only
    0.5–2.5% of 3s lie at 6.25–6.50 m, where the old line would put many of them.
    In 2007-08, 30% lie at 6.50–6.75 m.
  - The scale or drawing reference probably changed.
  - The module sets `FIRST_VALIDATED_SEASON = 2011`. The xPTS model should start at 2011-12,
    or treat earlier seasons separately.
- **Tests:** `tests/test_shots.py` covers the geometry, the tolerance, the sentinel, and one
  real game as a fixture (`tests/fixtures/points_E2024_1.json`): every labelled shot is on
  the correct side of the line.
- The plot comes from a scratch script run with `uv run --with matplotlib`. It isn't
  committed, and matplotlib isn't a project dependency.
