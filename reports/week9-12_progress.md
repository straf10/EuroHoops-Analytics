# Weeks 9–12 progress: M3 player impact (RAPM → Bayesian RAPM → SPM prior → GBL transfer)

Task file: `docs/history/prompts/week-9-12.md`. Branch `week-9-12` (from `main` 2bbfafc).

## §0 sub-decisions (confirmed by the owner, 2026-09-28)
All defaults accepted as written (H-a … H-j). H-g: closed-form conjugate Gaussian posterior,
not PyMC (PLAN §5.4 says "in PyMC"; the owner chose closed form; recorded as a deviation from
PLAN wording, no PLAN value changed).

| # | Decision |
|---|---|
| H-a | EuroLeague warm-up 2011–2014, tuning 2015–2022, validation 2023, test 2024–2025 (once, after the verdict commit). 2026-27 untouched. |
| H-b | Stint-side rows, O block (+1 offense five) and D block (−1 defense five), target points per 100, weight possessions, home column + intercept; 0-possession sides dropped; λ (shared or O/D) and decay half-life chosen on tuning by future-margin RMSE. |
| H-c | Walk forward by round; margin = P/100 · (h + Σ_home f·r − Σ_away f·r); projected minutes = previous 5 team games (first round: last season with the team, 0 for new players); oracle-minutes variant reported, never gated. |
| H-d | Gate: bootstrap 95% CI (1,000, seed 20261001) of RMSE(chosen RAPM) − RMSE(box-only) on validation has upper bound < 0. MAE and log loss (Normal, σ from tuning) reported; M1 and PIR reported, not gated. |
| H-e | Box-only: shrunk per-100 box rates (PTS, 2PA, 3PA, FTA, OREB, DREB, AST, STL, BLK, TOV, PF), ridge-combined, fitted on tuning with the future-margin target; PIR per minute as the naive baseline. |
| H-f | `rapm_dummy`: players under {0, 50, 100, 200, 400} minutes share a replacement column per team-season; kept only if it wins on tuning. |
| H-g | Closed form: posterior mean = ridge solution, covariance σ²(XᵀWX + Λ)⁻¹, 90% intervals. |
| H-h | SPM = regression of RAPM on per-100 box rates (minutes-weighted); `rapm_spm` shrinks toward it. See decision D2 for the fitting window. |
| H-i | GBL stints from `gbl_pbp`; SPM transfer vs GBL box-only vs PIR on the `config.GBL.m1` splits; GBL RAPM only if ≥ 95% of games pass. |
| H-j | `backtest --model m3` < 1,800 s on a quiet machine (item 36). |

## §3 findings (2026-09-28, real data; re-checked by `scripts/checks/m3_facts.py`, item 33)
1. **Players:** 1,727 distinct EuroLeague players have stints 2011–2025 (4,577 player-seasons;
   242–357 per season). Total on-court minutes: 10% < 12, 25% < 95, median 330, 75% 1,189,
   90% 2,944. Players under 50 / 100 / 200 / 400 total minutes: 310 / 443 / 656 / 928.
   A dense posterior inverse is ≈ 3,456 × 3,456 (95 MB): fine.
2. **Ids stable:** 905 ids span several seasons; 84 ids carry more than one spelling (e.g.
   P000956 ANDUSIC/ANDJUSIC), which is the same id. Only two names map to two ids: Omer
   Yurtseven (P005353 in 2014, P005983 in 2015: one player split in two) and Marko Simonovic
   (RED 2013 `PLRU`, RED 2023 P012711: two different players). Ids are used as they are.
3. **Minutes:** box seconds and stint on-court seconds agree within 60 s for 99.997% of
   89,289 player-games of passing games (max gap 92 s); 99.84% over all games.
4. **GBL box rates:** every cached ESAKE box score 2018–2025 (2,524 team box scores) has the
   same 17 columns: P, 2PM-A, 3PM-A, FTM-A, REBS, D.REBS, O.REBS, AST, BLK, BLK-A, FOULS F,
   FOULS M, STL, TO, TIM.PL., RANK. FOULS M = fouls committed and FOULS F = fouls drawn, and
   RANK = PIR (checked on `tests/fixtures/esake/box_8FC479F6.html`: PIR = PTS + REB + AST + STL
   + BLK + FD − missed FG − missed FT − TO − BLK-A − PF holds for the rows checked). 2018 and
   2019 have 52 and 26 cached pages without box tables (the known gaps). The staged
   `player_box` keeps only points/shots/seconds, so H-e needs a new parser of the full rows.
5. **GBL play-by-play is cached for 2018 and 2019 only** (203 and 139 games). See D4.

## Decisions this file did not cover
- **D1 (shared interface, orchestrator).** `models/minutes.py` (projected and oracle shares,
  expected possessions, round cutoffs) written in H0, so subagents A and B build on one
  interface. Share = seconds on court / game seconds (a team's shares add to 5). P = mean of
  the two teams' season-to-date pace (`poss_game · 40 / minutes`), falling back to the team's
  previous season, then the league mean of all earlier team-games.
- **D2 (SPM window).** H-h says leave-one-season-out on 2011–2022; §1 ("no … SPM coefficient
  used to predict a game may depend on that game or any later one") and item 34 forbid
  coefficients that use later seasons. The SPM used for season s is therefore fitted on RAPM
  of seasons < s only (expanding window). It still never uses season s (H6's test) and is
  stricter than LOSO.
- **D3 (file ownership).** The posterior (H4, subagent D) lives in `models/rapm_posterior.py`
  so that no two subagents edit `models/rapm.py`.
- **D4 (GBL RAPM).** GBL stints can be built only for 2018–2019 (the cached play-by-play).
  GBL validation/test (2023–2025) have none, so GBL RAPM cannot be evaluated there; H8 reports
  SPM transfer vs box-only vs PIR alone (H-i fallback). Back-filling 2020–2025 play-by-play
  would re-stage `gbl_pbp` and move `reports/possessions.json` (§1: existing reports don't
  move): not done; open question for the owner.
- **D5 (M1 comparison).** M1 margins are recomputed from the committed `tuned` block of
  `reports/backtest_m1.json` (no re-tuning, M1 untouched).
- **D6 (dummy threshold).** "Minutes in the fitting window" = the player's time-decayed
  on-court minutes in fitted stints before the cutoff (same decay as the rows).
- **D7 (grids, declared in `config.M3Grid` before any run).** Half-life {182, 365, 730, 1,460}
  days × shared ridge {250 … 8,000} possessions; then separate O/D ridges by one-dimensional
  search at the best half-life (kept only if they lower tuning RMSE); dummy thresholds with the
  chosen half-life and ridge.

## Iterations
iteration | deliverable | checks run | result | commit
1 | H0 branch, progress, §3 facts, minutes module, M3 config | ruff, format, mypy, test_minutes, vulture, m3_facts.py (real data) | green | c059836, 5e7b32a
2 | H4 posterior (subagent D, 1 round) merged | test_rapm_posterior (10 passed, coverage 88.38%), ruff, format, mypy, vulture in main tree; fast gate deferred (items 11/17 touch data/ while A, B, C run) | green | a2ebeab
3 | H3 GBL stints (subagent C, 1 round) merged; gbl-stints registered in cli.py | test_gbl_stints + test_cli (27 passed), ruff, format, mypy, vulture; real-data run deferred until A, B finish | green | 9a4daee
4 | H2 box-only/PIR baselines + GBL box-line parser (subagent B, 1 round) merged | test_box_impact + test_gbl_box_lines (23 passed), ruff, format, mypy, vulture; B flagged team_eff._epoch assumes ns timestamps (M1, out of scope: to check on real data, report only) | green | 6ec3a46

## Owner instructions, 2026-09-28 23:20 (going AFK)
- Work autonomously to the stop condition; then merge `week-9-12` into `main`, push `origin/main` and delete the subagent worktree branches (this overrides §1 "do not push / do not merge").
- If M3 does not beat the current best model, look for legitimate improvements. Interpreted as: extra variants are explored and chosen on **tuning seasons only**, before the verdict commit; anything tried after the verdict is labelled post-hoc and never changes the gate; test is still scored once; leakage tests cover every variant.
5 | H1 RAPM + harness (subagent A, 1 round) merged; B's baselines wired (one BaselineResult, CLI binds BoxGrid) | full pytest 518 passed, cov 95.32%; ruff, format, mypy, vulture | green | edd7361
   note: team_eff._epoch (B's flag) is correct on real data: read_games gives datetime64[ns, UTC] (pandas 3.0.6); M1 unaffected.
6 | real-data probe: fixed shares > 1 (box quirks; share = 5 x sec / team sec, capped) and truncated player ids from empty lineups (spell index); variant registry; M1 from its own warm-up | test_minutes, test_rapm (+ regression test), test_m3_backtest; ruff, mypy | green | ff227ad
7 | first real tuning-only run (EuroLeague, 507 s, not committed as a report): tuning RMSE rapm 11.728 (hl 1460, ridge O 1000 / D 2000), m1 11.853, box_only 11.902, pir 12.108, b0 12.794; oracle minutes 11.777 (worse than projected). Grid edges: rapm half-life 1460 (max); box_only half-life 182 (min) and ridge 300 (max). Grids widened on both sides before any validation number (rapm half-life + 2920, 5840; box-only half-life + 45, 91; ridge + 1000, 3000, 10000) so neither the model nor the baseline is held back by its grid. | tuning only | info | (this commit)
8 | H3 real data: `eurohoops gbl-stints` twice, reports/gbl_stints.json byte-identical (tables sha256 73e0cd2d… / 2cea13b1…). 342 games 2018–19, 12,423 stints; pass all three 56.1% (2018 55.7%, 2019 56.8%); five_on_court 96.1% / 99.3%, points 100% / 100%, possessions ±2 58.1% / 57.6%. H-i (≥ 95%) does not hold → no GBL RAPM (and no GBL PBP for 2023–25 anyway, D4). Diagnosis: GBL counts run mostly 2–3 below poss_raw (EuroLeague PBP vs box within ±2: 90.2% of team-games; GBL ≈ 76%), so GBL possession ends are likely under-detected (team-column turnovers are the first suspect). Not loosened; open item. | gbl-stints ×2, cmp | H3 done; H-i rule fails (allowed outcome) | (this commit)
