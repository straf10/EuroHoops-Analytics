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
- **D6 (distribution fit, leakage scope):** the margin distribution (Student-t scale and df) and the total σ are fit on
  tuning games in-sample and then frozen, exactly as M1's gate model was. So the leakage guarantee is:
  - exp_margin and exp_total of every game never depend on that game or later games;
  - p_home of validation/test games never depends on any validation/test outcome.

  A tuning game's p_home does depend on later tuning outcomes through the two frozen dispersion numbers. This is the
  declared exception, and it only affects the tuning score.
- **D7 (wave-2 order):** E (leakage suite) starts after D (harness) is merged, because E reuses D's synthetic-league
  builder `tests/m5_synthetic.py`. Running them in parallel would mean two builders or a file shared by two subagents.

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

- **D8 (Platt leak, found by J5):** the tuning-only report carried Platt (a, b) for validation and test seasons. Each
  of those is fit on all earlier outcomes, which include validation. Fixed: the report lists Platt coefficients only for
  warm-up seasons and the scored splits. The choice and the tuning numbers were never affected. D's test that required
  tuning-only `chosen` to equal the full run's (which encoded the leak) now asserts the restricted equality. E's
  "differs only in Platt" test was redundant after the fix and was removed.
- **D9 (leakage-suite delete edit):** deleting the game that defines g's round cutoff moves the cutoff itself, and with
  it the decay clock (shifts ~1e-5). E's delete edit keeps that game and deletes the other round mates and later games.
  That is a change of the cutoff's definition, not information from the future.

- **D10 (tie-break):** J-i says ties go to "the simpler model". The J4 rule continued to grid position after form and
  shares, so a half-life listed earlier won a tie against a lower-loss point of the same model. Grid values are not a
  complexity order. Fixed: form, then shares, then the lowest tuning loss, and grid position only for exact equality.
  Found on tuning run 2 (the widened grid put d120 before d180); no validation number existed.

## Iterations
iteration | deliverable | checks run | result | commit
1 | J0 branch, §0, §3 probes, D1–D5, interfaces | m5_facts.py PASS; ruff | green | 3c603b5
2 | J1 rest (subagent A, round 1) merged | diff reviewed (no leak: searchsorted side=left; scores unread); test_rest 13 passed; ruff, mypy | green | merge of 8b4b01a
3 | J3 model core (subagent C, round 1) merged | diff reviewed (cutoff rows strictly before; blend on seasons < s; tolerances derived from the design, not the output); vulture whitelist conflict resolved by hand; test_m5 + test_rest 25 passed; ruff, mypy, vulture | green | 545112c
4 | J2 shares (subagent B, round 1) merged; wave-1 real-data sanity | diff reviewed; full suite 711 passed, cov 93.61%; ruff, mypy, vulture. Real EL 2011–2025: rest_features 0.2 s, short_rest 15.1% of team-games, PAN/OLY other_comp_prev 31.9%; one days_rest < 1.5 (E2021_300 OLY–UNK, unplayed voided game: never rated, fact 2 holds). Shares: every team-side sums to 5.000 (proj_avail min 4.593: caps); runtimes proj_hc 23 s, proj_decay 89 s, proj_avail 101 s | green | fbe6b8d
5 | J4 harness (subagent D, round 1) merged; CLI `backtest --model m5` + `log_m5_backtest` (orchestrator) | diff reviewed; inputs cut at test[-1] so the daily log cannot move data_sha256; tracking test added | green | cf47d6a, d1a620b
6 | J5 leakage suite (subagent E, round 1) merged; Platt leak D8 fixed by orchestrator; checklist items 43–44 | full suite 782 passed, cov 93.17%; ruff, mypy, vulture | green | 8c3af23
7 | J6 tuning run 1 (EL, tuning only; a different project's training job held ~33% CPU, so this RUNTIME 750 s is not used for item 46) | chosen proj_hc\|core\|d180\|r40, no Platt, total_m1_rest; tuning LL 0.599344 vs M1 0.6060, Elo 0.6090; best overall proj_avail@2\|core\|d180\|r40 0.599296 (within the 0.0005 tie, so the simpler wins). Edges: residual half-life 180 = lower edge, half_life_games 2 = lower edge (best candidate). **Grid widened once** (J6 rule: the best on an edge, before the verdict): half_life_games + 1, residual_half_life_days + 60, 120. Oracle tuning LL 0.6006 > projected: the same as M3 (oracle 0.6031 vs rapm 0.6011), where actual minutes carry blowout garbage time; reported, not a bug | widened, rerun | 6b428a0
8 | J6 tuning run 2 (widened grid, 414 candidates; RUNTIME 968 s, load ~40% from the other job: not used for item 46) | no axis on an edge; best proj_avail@1\|core\|d180\|r40 0.599292; the rule chose proj_hc\|core\|d120\|r40 0.599638 by grid position → D10 tie-break fix + test | fixed, rerun | 8ad605e
9 | **J6 verdict** (tuning run 3, RUNTIME 997 s under the other job's load: not used for item 46) | chosen **proj_hc\|core\|d180\|r40**, no Platt (with 0.600889 vs without 0.599344), total **total_m1_rest** (tuning CRPS 9.278 vs 9.300); margin model student_t_const scale 10.968 df 15; tuning LL 0.599344 vs M1 0.606 (n 2,236); best forms core 0.599292 (proj_avail@1, tied), core_rest 0.599937, blend 0.602529; no axis on an edge; tuning-only report, no validation number anywhere | verdict | 5baa7c1
10 | J7 GBL fixed-choice path (J-g) + test | fixed-choice test passed; ruff, mypy | green | c2ca39f
11 | **J7 EL validation scored once** (RUNTIME 930 s, load from the other job: not for item 46) | same choice as the verdict. Validation n 331:
  - M5 LL **0.583765** vs M1 **0.587714** (the replay equals M1's committed number exactly), diff −0.003948, CI [−0.014946, +0.006291] → **gate PASS** (point rule);
  - Brier diff −0.0010 [−0.0058, +0.0035], margin CRPS −0.044 [−0.147, +0.049], totals CRPS −0.082 [−0.178, +0.006];
  - vs Elo: LL −0.0047 [−0.0191, +0.0083], totals CRPS −1.093 [−1.519, −0.688];
  - oracle LL 0.5803; gap projected − oracle LL +0.0035 [−0.0085, +0.0140], missed_top3 n 20, RMSE gap +0.79 [0.07, 1.54] | validation | cf84a68
12 | GBL labelled not gated; checklist items 45–46 | ruff, mypy, fixed test | green | 997fc3e
13 | **J7 EL test scored once** (2024–2025, n 732; RUNTIME 886 s, load from the other job: not for item 46) | M5 LL **0.6232** vs M1 **0.6232** (diff −0.000015, CI [−0.0095, +0.0098]); Brier +0.0004; margin CRPS +0.026 [−0.055, +0.105]; RMSE 11.697 vs 11.656; totals CRPS −0.055 [−0.120, +0.015]; vs Elo LL −0.0014 [−0.0114, +0.0088], totals CRPS −0.914 [−1.220, −0.577]. **The validation edge does not carry to test** (the same as M3). Oracle 0.6201; gap +0.0031 [−0.0041, +0.0105]; missed_top3 n 31, RMSE gap +0.84 [0.08, 1.65]. Test segments: pan_oly n 163 M5 0.583 vs M1 0.593; short_rest n 193 0.597 vs 0.613 | test | 2401d15
14 | J7 GBL (EL verdict as a fixed choice, reported, not gated; RUNTIME 259 s under the other job's load) + `m5-rest` + J8 gap test | GBL n tuning 336 / validation 163 / test 340, none dropped; M1 GBL replay = committed 0.429759.
  - Validation: M5 0.4748 vs M1 0.4298, diff +0.0450 [+0.0108, +0.0834] (**worse**).
  - Test: M5 0.4790 vs M1 0.4835, diff −0.0046 [−0.0259, +0.0165]; totals CRPS vs M1 −0.207 [−0.333, −0.077], vs Elo −0.685 [−1.127, −0.252].
  - Greek after an EL game: validation n 41 M5 0.273 vs M1 0.219; test n 87 0.166 vs 0.197.
  - m5_rest.json: EL residual on rest differences, every 90% interval spans 0 in every split. GBL tuning short_rest +5.55 [1.16, 9.95] but −3.0 on validation: no stable rest effect beyond M5 (consistent with core over core_rest).
  - test_m5_gap 2 passed | green | (this commit)

VERDICT 5baa7c1
VALIDATION cf84a68
