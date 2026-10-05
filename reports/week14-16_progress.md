# Weeks 14–16 progress: M5 roster-aware game predictor

Prompt: `docs/history/prompts/week-14-16.md`. Branch `week-14-16` (from `main` db87baf, which equals
`origin/main` on 2026-10-05).

## §0 answers (owner, 2026-10-05)
- J-h gate: **point rule**. M5's EL validation log loss must be lower than M1's (`student_t_const`, committed
  parameters replayed) on the same scored games. The paired bootstrap 95% CI is reported (1,000 resamples, seed 20261020).
- Every other §0 default is accepted as written: J-a, J-b, J-c, J-d, J-e (rest only, no travel distance), J-f, J-g
  (GBL reported, not gated), J-i (choice on tuning), J-j (no live M5), J-k, J-l.

## §3 findings (J0, real data, `games` mart at db87baf)
1. **Tip-off resolution:** `tipoff_utc` is a full timestamp in every season of both competitions. No tip-off sits at
   00:00 UTC (EL 2007–2026, GBL 2018–2026), so rest is measured in fractional days.
2. **Rest-day distribution** (days since the team's previous game in either competition, capped at 7, seasons 2015–2025):
   - EL: 5/25/50/75% quantiles 1.99 / 3.94 / 6.0 / 7.0 days (n 6,589).
   - GBL: 1.96 / 4.91 / 7.0 / 7.0 days (n 2,666).
   - **No game in either competition comes ≤ 1.5 days after the team's previous game**, so the J-e `b2b` flag would
     be constant (see D1).
3. **PAN/OLY games ≤ 3 days after a game in the other competition:**

   | season | EL | GBL |
   |---|---|---|
   | 2018 | 3 | 38 |
   | 2019 | 1 | 17 |
   | 2020 | 5 | 16 |
   | 2021 | 8 | 29 |
   | 2022 | 5 | 37 |
   | 2023 | 7 | 41 |
   | 2024 | 9 | 45 |
   | 2025 | 13 | 44 |

   (EL counts from 2018 on, because GBL data starts in 2018.)
4. **Unrated players:** `rapm.RatingLookup.rating` returns (0, 0) for a player with no column at the cutoff, and
   `rapm_spm` fills a column's prior from SPM. M5 inherits this unchanged: an unrated player counts as 0 (league
   average), not as a replacement-level value.
5. **Per-cutoff ratings without refitting:** `fit_spm(...)` returns a `WalkForward` whose `lookups[i]` is game i's
   pre-round rating lookup. One fit with the frozen M3 `chosen` block serves every M5 variant, so no rating cache is
   needed unless J-k runtime says so.
6. **M1 replay inputs:** `reports/backtest_m1.json` `tuned` = rating (1920 d, carry 0.25, ridge 250), pace (60 d,
   carry 1.0, ridge 2), margin `student_t_const` scale 10.402421 df 7, totals_sigma 16.530916.
   `comparison_elo` = k 20, hca 90, reversion 0.25, margin_scale 23.358237, margin_sigma 11.882097.
   The committed validation numbers to reproduce are M1 log loss 0.587714 and Elo 0.588509, both on n 331.
7. Share coverage and Elo margins on the backtest frame are checked in J4/J6 on real data (recorded there).

## Decisions this file's §0 did not cover
- **D1 (rest flag):** `b2b` (≤ 1.5 days) is replaced by `short_rest` = previous game ≤ 2.5 days before. Reason: the
  schedule data has no game within 1.5 days (finding 2), so the declared flag carries no information. 2.5 days marks
  the EL double-round weeks (Tue→Thu ≈ 2 days) and EL→GBL weekends. Declared before any model fit.
- **D2 (shares convention):** `proj_decay` keeps H-c's convention (cap 1, no renormalisation), so `proj_hc` is its
  limit case. `proj_avail` rescales the remaining players of a team to the team's pre-removal share sum (capped at 1)
  when it zeroes absent players.
- **D3 (player part):** M5's player part is `m3_backtest.rapm_margins` unchanged (it includes the walk-forward RAPM
  home term). The team residual model adds its own home adjustment on top. This makes "core with zero residual and
  rest" equal M3's margin exactly (J3 test).
- **D4 (residual and rest in one fit):** at each round cutoff, a time-decayed ridge on earlier rated games:
  `margin − player_part = h + r_home − r_away + β·rest_diff`. The rest columns are omitted in the `core` form.
  Residual penalty `residual_ridge`, rest penalty `rest_ridge`, h almost free.
- **D5 (blend):** weights for season s are fit on every earlier season's out-of-sample component margins (rated games
  with all components finite). The fit is a probit with non-negative coefficients and no intercept,
  p = Φ(Σ b_k m_k), giving w = b/Σb. Distribution afterwards: `fit_margin_model("student_t_const", …)` on tuning
  (as M1). Platt is kept only if it lowers tuning log loss.

## Interfaces (fixed by the orchestrator, pasted into every subagent prompt)
```python
# models/rest.py (A)
REST_SCHEMA columns: game_id str, side str in {"home","away"}, team str, days_rest float64 in (0, 7],
  short_rest bool, games_last_7d int64 >= 0, other_comp_prev bool
def rest_features(games: pd.DataFrame, other: pd.DataFrame | None, club_map: Mapping[str, str]) -> pd.DataFrame
  # games: the target competition's games-mart rows; other: the other competition's (or None);
  # club_map: other-competition team code -> target-competition code (only these clubs link).
  # One row per (game in `games`, side). Only `played` games with tip-off strictly before g's tip-off count.

# models/rotation.py (B)
def projected_shares_variant(variant: str, games: pd.DataFrame, player_games: pd.DataFrame, *,
    n_games: int = 5, half_life_games: float = 4.0, absent_games: int = 2) -> pd.DataFrame
  # variant in {"proj_hc","proj_decay","proj_avail"}; returns minutes.SHARES_SCHEMA frames.

# models/m5.py (C)
@dataclass(frozen=True) class ResidualFit: adjustment: FloatArray  # (n_games,) h·home_flag + r_home − r_away + β·rest_diff
                                            rest_coef: FloatArray    # (n_games, k) β used for each game (NaN before any fit)
                                            home: FloatArray         # (n_games,) h used for each game
def fit_residual(games: pd.DataFrame, target: FloatArray, rest_diff: FloatArray | None, *,
    half_life_days: float, residual_ridge: float, rest_ridge: float) -> ResidualFit
  # games in tip-off order with home, away, season, tipoff_utc, phase, round, played, forfeit, neutral;
  # target = actual margin − player part (NaN where unusable); walk forward at round cutoffs.
def fit_blend(components: FloatArray, home_won: FloatArray) -> FloatArray   # (k,) weights >= 0, sum 1
def blend_by_season(components: FloatArray, home_won: FloatArray, season: IntArray, rated: BoolArray) -> FloatArray
  # (n_games, k) weights used per game: season s uses fit_blend on all earlier seasons; NaN rows if none.
def fit_platt(p: FloatArray, home_won: FloatArray) -> tuple[float, float]
def apply_platt(p: FloatArray, a: float, b: float) -> FloatArray
def p_over(total: FloatArray, sigma: float, line: float | FloatArray) -> FloatArray
def p_cover(margin: FloatArray, model: MarginModel, line: float | FloatArray) -> FloatArray  # P(home margin > line)

# Per-game prediction frame (D's report writer): game_id, p_home, exp_margin, margin_sigma, margin_df,
#   exp_total, total_sigma, model, variant
```

Facts 1–3 and 6 are re-checked by `scripts/checks/m5_facts.py` (checklist item 43). Facts 4 and 5 are tested in J3/J4.

## Iterations
iteration | deliverable | checks run | result | commit
1 | J0 branch, §0, §3 probes, D1–D5, interfaces | m5_facts.py PASS; ruff | green | (this commit)
