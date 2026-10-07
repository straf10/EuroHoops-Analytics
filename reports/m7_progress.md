# Weeks 14–16 (part 2) progress: M7 season simulator

Prompt: `docs/history/prompts/week-14-16-m7.md`. Branch `week-14-16-m7` (from `origin/main` 28347b9, worktree
`C:\Python\EH-m7`; another agent works in `C:\Python\EH-ci` on `ci-m5-history`).

## §0 answers (owner, 2026-10-07)
- Every §0 default is accepted as written: K-a, K-b, K-c, K-d, K-e, K-f, K-g, K-h, K-i, K-j, K-k.
- Stop condition: the **full §7 checklist** in one run on a quiet machine (not a fast-gate record).

## §3 findings (K0, real data, `games` mart copied from the main tree at 28347b9)
Re-checked by `scripts/checks/m7_facts.py` (checklist item 48) against `tests/fixtures/m7_seasons.json` and
`tests/fixtures/m7_official_tables.json`.

1. **EuroLeague formats 2016–2025** (season = start year):

   | season | teams | rounds | games/team | direct playoffs | play-in | source of the cut lines |
   |---|---|---|---|---|---|---|
   | 2016, 2017, 2018 | 16 | 30 | 30 | 1–8 | — | official final table + playoff field in the API games feed |
   | 2020, 2022 | 18 | 34 | 34 | 1–8 | — | idem |
   | 2023, 2024 | 18 | 34 | 34 | 1–6 | 7–10 | idem (play-in field = places 7–10) |
   | 2025 | 20 | 38 | 38 | 1–6 | 7–10 | 2025-26 Bylaws Art. 18 + idem |

   - Bylaws: only the 2025-26 and 2026-27 PDFs exist at `ftpserver.euroleague.net/general/{yyyy_yy}_EuroLeague_Bylaws.pdf`
     (2016-17 … 2024-25 return 404, checked 2026-10-07). Cached under `data/raw/regulations/` with `SOURCES.txt`.
     2025-26 Art. 18: 20 teams double round robin; top 6 to the playoffs; 7–10 play-in (A 7v8, B 9v10,
     C loser A v winner B, at the better-placed team; winner A = 7th, winner C = 8th); playoffs best of 5, 1vC, 4v5,
     3v6, 2vA, games 1, 2, 5 at the higher-placed team; Final Four.
   - For the earlier seasons the cut lines come from the EuroLeague's own data: the official final tables
     (`api-live.euroleague.net/v1/standings?seasonCode=E{s}&roundNumber={last}`, cached under
     `data/raw/euroleague/standings/`, accessed 2026-10-07) and the playoff/play-in participants in the official games
     feed. Playoff field = table top 8 in 2016–2022; play-in field = places 7–10 and places 1–6 in the playoffs in
     2023–2025. Games 1 and 2 of every playoff series were at the better-placed team in every scored season.
   - Every scored season: every regular-season game played, no forfeit, each team plays every other twice.
2. **2020-21:** all 306 regular-season games were played (postponed games were made up), no forfeits, every team
   34 games in the official table. It can be ranked exactly; it stays in the tuning set.
3. **`rank` on full seasons:** reproduces the official final table of **all 8 scored EuroLeague seasons** exactly, with
   overtime games ranked on their regulation score (`EndOfQuarter` Q4 in the cached box scores; 7–19 overtime games
   per season) and 2022-23's Panathinaikos two-win deduction. Code: `src/eurohoops/sim/played.py`.
4. **GBL formats** (ESAKE results pages in the mart; reported only):
   - 2020, 2022: 12 teams, 22 rounds. 2021, 2025: 13 teams, 26 rounds (a bye per round, 24 games per team).
     2023, 2024: 12 teams, 22 rounds. 2018: 14 teams with 2 forfeits and 52 games without a box score (M1 warm-up).
     2019: stopped (COVID).
   - **Scored (reported): 2020, 2021, 2022, 2025.** In each, the quarterfinal field equals the top 8 of `rank`'s table.
   - **Excluded: 2023-24 and 2024-25.** The quarterfinal field is not the table's top 8 (2023-24: 9th in, 8th out,
     tied on 8 wins; 2024-25: 9th in with 8 wins, 8th out with 9). The cause (sanction, withdrawal or a tie-break)
     is not in any source we hold, so the cut-line target is ambiguous. Small-sample cost accepted (reported only).
   - Not verified for any GBL season: the full order below the cut (only the top-8 set is checked), series lengths
     before 2023-24 and the tie-breaks. Carried as `unverified` notes.
5. **M1 posterior:** `DecayedRidge` keeps `gram`, `rhs`, `penalty`, `seen`; at a cutoff that is enough for
   (G + P)⁻¹ over the seen columns. σ² is defined in the interfaces below; synthetic coverage is K1's test.
6. **2026-27 fixtures:** EuroLeague 380 rows (38 per team, every ordered pair once; 10 games played on 2026-10-07).
   GBL 183 rows: Olympiacos–AEK is listed twice (the copy in round 18 has an unconfirmed date); 182 ordered pairs.
   The live simulator plays the remaining fixtures in tip-off order and keeps the first row of a duplicated ordered
   pair (decision D2).

## Decisions this file's §0 did not cover
- **D1 (checkpoints):** after k = ⌊f · R⌋ completed rounds, f ∈ {0.25, 0.5, 0.75}, R = regular-season rounds
  (EL 30 → 7/15/22, 34 → 8/17/25, 38 → 9/19/28; GBL 22 → 5/11/16, 26 → 6/13/19). Cutoff = first tip-off of round
  k + 1. Checkpoint index (for the seed) = position in the list of (season, checkpoint) sorted by season then f.
- **D2 (duplicate fixture):** a remaining ordered pair listed twice is simulated once (the earlier row).
- **D3 (deductions):** a season's deductions are part of its format (`Format.points_deducted`, two points per win as
  in `standings.Format`): EL 2022-23 Panathinaikos = 4 points (2 wins). They are applied to simulated final tables
  and to the targets. The sanction date is not recorded; it only moves places 17–18.
- **D4 (Elo baseline):** `elo_sim` runs in the same engine with fixed strengths: off = T/2 + c·R/2, def = c·R/2,
  home = c·hca/2, pace 100 (μ_p = 100, pace effects 0), where R are the comparison-Elo ratings at the cutoff, T is
  M1's expected total, and c matches the Elo win-probability slope at an even game under M1's noise:
  c = scale · ln 10 / (1600 · f_t(0)). So `elo_sim` uses Elo's win probabilities (exact at an even game) with the
  same score model.
- **D5 (knockouts):** the higher seed hosts games 1, 2, 5 of a best of 5 and games 1, 3 of a best of 3; single games
  are at the better-placed team except the Final Four (neutral). Knockout winners use the unrounded margin. GBL
  semifinals: winner(1/8) v winner(4/5), winner(2/7) v winner(3/6) (unverified, as GBL_2026).
- **D6 (tie-break speed):** `standings` gains one function, `rank_by_wins` (new, `rank` untouched), so the engine
  passes only the games of tied teams; a test proves it equals `rank` on 1,000 random tables.

- **D7 (σ² degrees of freedom, amends the K1 interface):** σ² = Σ d_i·poss_i·r_i² / (Σ d_i − p_eff) with
  p_eff = tr((G + P)⁻¹ G), the standard ridge residual-variance estimator. The first definition (denominator Σ d_i)
  is biased low by about (n − p)/n; K1's synthetic coverage test exposed it (subagent A, round 1). Fixed in the
  estimator; the test's league size, seed and band stay as first designed.

- **D8 (GBL noise):** M1-GBL's margin model is `student_t_pace`; inside a simulation the scale is constant per
  checkpoint, scale · sqrt(mean pace of the remaining fixtures / ref_pace).
- **D9 (sim_inflate noise):** `sim_inflate(c)` keeps `sim_net`'s noise scale computed from the uninflated covariance and
  multiplies only the strength covariance by c (K-f: "sim_net with the strength covariance × c").
- **D10 (GBL splits):** the GBL never chooses (fixed EuroLeague verdict). Its scored seasons are labelled tuning 2020,
  2021, 2022 and test 2025 (validation empty; 2023-24 excluded by finding 4), test scored only with `--score-test`.

## Interfaces (fixed before wave 1; pasted verbatim into every subagent prompt)

```python
# --- models/team_eff.py (K1, subagent A) ---------------------------------------------------
@dataclass(frozen=True)
class RatingPosterior:
    """M1's rating posterior at a cutoff (K-b): Gaussian over the seen columns of the design
    [μ, h, off · n, def · n] (``rating_design`` order)."""

    teams: tuple[str, ...]  # History.teams (all n, design order)
    columns: IntArray  # (k,) seen design columns, ascending
    labels: tuple[str, ...]  # (k,) "mu", "h", "off:<team>", "def:<team>"
    mean: FloatArray  # (k,) DecayedRidge.solve() at the cutoff, seen columns
    cov: FloatArray  # (k, k) sigma2 · inv(G + P) over the seen columns
    sigma2: float  # Σ d_i·poss_i·r_i² / Σ d_i over the rows used (d_i = decay weight at the cutoff)


def rating_posteriors(
    history: History, params: DecayParams, cutoffs: Sequence[tuple[float, int]]
) -> list[RatingPosterior]:
    """Posterior at each (cutoff epoch seconds, season), in one walk-forward pass: rows with
    row_time < cutoff, the model advanced to (cutoff, season) as rating_fits does."""


def pace_points(
    history: History, params: DecayParams, cutoffs: Sequence[tuple[float, int]]
) -> list[FloatArray]:
    """Pace solution [μ_p, pace · n] (1 + n,) at each cutoff, as pace_fits uses it."""


# --- sim/formats.py (K2, subagent B) --------------------------------------------------------
def season_format(competition: str, season: int) -> Format:
    """``season`` = start year. 2026 returns standings.EUROLEAGUE_2026 / GBL_2026 unchanged;
    scored past seasons come from the K0 table; anything else raises ValueError."""


def cut_lines(fmt: Format) -> dict[str, tuple[int, ...]]:
    """{"direct": playoffs_direct} plus, when the format has a play-in, "play_in": play_in and
    "top10": playoffs_direct + play_in."""


# --- sim/season.py (K3, subagent C) ---------------------------------------------------------
StrengthSampler = Callable[[int, np.random.Generator], tuple[FloatArray, FloatArray, FloatArray]]
# (n_sims, rng) -> (off (sims, teams), def_ (sims, teams), home (sims,)), teams in `teams` order.
# μ is folded into off: ORtg_H = off[H] + home·hf - def[A], ORtg_A = off[A] - home·hf - def[H]
# (hf = 0 at a neutral venue). One draw per simulation, kept for the whole season.


@dataclass(frozen=True)
class NoiseModel:
    scale: float
    df: float | None  # Student-t degrees of freedom; None = Normal


@dataclass(frozen=True)
class PaceModel:
    mu: float
    team: FloatArray  # (teams,), `teams` order; a game's pace = mu + team[H] + team[A]


# Remaining fixtures: pd.DataFrame with columns
#   home: str, away: str, neutral: bool, tipoff_utc: datetime64[ns, UTC]
# sorted by tipoff_utc (ties by home); source team codes; no duplicated ordered pair.


@dataclass(frozen=True)
class SimOutput:
    teams: tuple[str, ...]
    n_sims: int
    seed: int
    rank_counts: IntArray  # (teams, len(teams)): [i, j] = sims where team i finished j + 1
    wins: FloatArray  # (teams,) mean final regular-season wins (played + simulated)
    p_direct: FloatArray  # (teams,) P(final place in fmt.playoffs_direct)
    p_play_in: FloatArray  # (teams,) P(place in fmt.play_in); zeros without a play-in
    p_top10: FloatArray  # (teams,) P(place <= 10)
    p_playoffs: FloatArray  # (teams,) P(in the playoff bracket: EL playoffs / GBL quarterfinals)
    p_semis: FloatArray  # (teams,) EL: P(Final Four); GBL: P(semifinals)
    p_final: FloatArray  # (teams,) P(reached the final)
    p_title: FloatArray  # (teams,) P(won the title)


def simulate(
    *,
    teams: Sequence[str],
    played: Sequence[Result],
    remaining: pd.DataFrame,
    sampler: StrengthSampler,
    pace: PaceModel,
    noise: NoiseModel,
    fmt: Format,
    n_sims: int,
    seed: int,
) -> SimOutput:
    """Regular season: margin = round(pace·(ORtg_H - ORtg_A)/100 + scale·e), e ~ t(df) or N(0,1);
    a 0 becomes ±1 with probability ½; home points = round((total + margin)/2) with
    total = pace·(ORtg_H + ORtg_A)/100, away points = home points - margin. Final table per
    sim: wins (played + simulated) minus fmt.deducted_wins(); if all unique, ordered by wins,
    else standings.rank_by_wins. Knockouts per fmt (D5) with the same draw."""


# --- standings.py (new function only, owned by C; rank() untouched) --------------------------
def rank_by_wins(
    wins: Mapping[str, int],
    results: Sequence[Result],
    deducted_wins: Mapping[str, int] | None = None,
) -> list[str]:
    """rank() given each team's wins (before deductions): equal to rank(all results) when
    ``results`` holds at least every game of each team that is tied after deductions."""
```

## Iterations
iteration | deliverable | checks run | result | commit
1 | K0: progress file, §3 facts (`m7_facts.py`, fixtures), `sim/played.py` | ruff, format, mypy, vulture, test_sim_played, m7_facts | PASS | 291b312
2 | wave 1: K2 formats (B, 1 round), K3 engine + `rank_by_wins` (C, 1 round), K1 posterior (A, 2 rounds: D7) merged | per-deliverable tests rerun after merge; fast gate | see iteration 3 | e273bdc, 84915fe
3 | fast gate after wave 1 (HEAD 84915fe/d5bd175) | items 1-6, 9, 11, 14, 17, 19, 27-29, 32-34, 40, 42-44 | 2 FAIL: item 9 (origin/main had moved: 20 newer prediction rows missing on the branch; fixed by merging origin/main, be4a070, item 9 PASS, 0 changed lines); item 28 (its MLflow half reads the M2 runs item 24 writes in the same SCRATCH, and item 24 is in the fast-gate SKIP list; its leakage half passes, 7 passed). Item 5: 895 passed. All other run items PASS | be4a070
4 | K4 harness (D, 1 round) merged; checklist items 48-51 added | ruff, format, mypy, vulture; 168 tests over the M7 + neighbouring files | PASS | 0b13d23 merge, 5cc64c2
5 | K5 leakage suite (E, 1 round) merged; E's `checkpoint_inputs` refactor checked output-identical by the orchestrator (pre/post code on the synthetic league: EL default/tuning-only/test, GBL fixed; JSON + CSV byte-identical) | items 48, 49 | PASS (49: 32 passed) | merge of m7-e
6 | K6 tuning run and verdict | tuning-only report checked free of validation/test numbers | sim_full chosen | 8b8fa8c

VERDICT 8b8fa8c
7 | K7 validation scored once | gate block | FAIL (point_sim lower Brier) | c65e91c

VALIDATION c65e91c
8 | K7 test scored once | m7_order | sim_full level with point_sim on test | d885fc8

TEST d885fc8
9 | K8 live simulate (gated; dry run on real marts), item 52 | test_live_sim (9), sim_dry_run.py, ruff/mypy/vulture | PASS | (this commit)
10 | K9 card + CONTEXT + item 53; full §7 checklist in one run (HEAD d4a55fd, 11:47Z-14:42Z, quiet machine) | items 1-53 | all PASS except 22, 25 (known M2) | d4a55fd; closeout (this commit)

## K6: tuning and verdict (EuroLeague, tuning seasons 2016, 2017, 2018, 2020, 2022; 252 team-checkpoints)
`uv run eurohoops backtest --model m7 --tuning-only` (quiet machine: CPU 12%, no other python jobs), 129 s.

| model | Brier (direct cut) | RPS (final rank) | log loss | Spiegelhalter z |
|---|---|---|---|---|
| sim_full | 0.10932 | 0.078114 | 0.339104 | -0.42 |
| sim_net | 0.109624 | 0.078228 | 0.339803 | -0.34 |
| sim_inflate_1.5 | 0.110435 | 0.078799 | 0.342678 | -0.87 |
| sim_inflate_2 | 0.111336 | 0.079439 | 0.346574 | -1.28 |
| point_sim (baseline) | 0.108517 | 0.078143 | 0.33472 | 1.38 |
| elo_sim (baseline) | 0.103419 | 0.077422 | 0.317801 | 0.62 |
| standings_now (baseline) | 0.174603 | 0.121102 | 1.608237 | — (0/1 forecasts) |

- **Choice (K-i): `sim_full`**, lowest tuning RPS among the variants (0.078114; sim_net +0.000114 is within the 0.002
  tie tolerance, and the tie goes to the simpler sim_full anyway).
- **Inflation grid edge:** RPS rises with c (1 < 1.5 < 2); the best variant is not the grid's largest c, so the grid
  is not widened (`best_on_edge` false).
- Observed on tuning, recorded before validation: the point-strength baseline has a lower Brier than sim_full
  (difference sim_full − point_sim +0.0008, 95% CI [−0.0034, 0.0046]) and elo_sim is lower still (+0.0059,
  CI [−0.0013, 0.0130]); sim_full beats standings_now (−0.0653, CI [−0.1110, −0.0232]). No re-tuning follows from this.
- The tuning-only report and CSV hold no validation or test number (checked: seasons 2016–2022 only; the only "2023"
  strings are 2022-23 cutoff dates).

## K7: validation 2023 (scored once; 18 teams × 3 checkpoints = 54 team-checkpoints), 166 s
| model | Brier | RPS | log loss | z |
|---|---|---|---|---|
| sim_full (chosen) | 0.091951 | 0.084036 | 0.294073 | -1.15 |
| point_sim | 0.088277 | 0.0841 | 0.2868 | -0.81 |
| elo_sim | 0.1042 | 0.0837 | 0.3271 | -0.09 |
| standings_now | 0.148148 | 0.1285 | 1.3646 | — |

**Gate: FAIL.** sim_full's validation Brier is below standings_now's (0.091951 < 0.148148) but not below point_sim's
(0.088277); pooled tuning + validation Spiegelhalter z = −0.87 (|z| < 1.96, not miscalibrated). Bootstrap
(18 team clusters): sim_full − point_sim +0.0037, 95% CI [−0.0039, +0.0106] (spans 0); − standings_now −0.0562
[−0.1944, +0.0544]; − elo_sim −0.0122 [−0.0332, +0.0050]. point_sim is ahead at every checkpoint (25%: 0.1278 vs 0.1225;
50%: 0.1015 vs 0.0969; 75%: 0.0465 vs 0.0455). Reliability (10 bins) shows no systematic over- or under-confidence
(z −1.15 on validation). Consequence (K-j): `eurohoops simulate` is built but not added to the daily workflow.

## K7: test 2024, 2025 (scored once, after the validation commit; 114 team-checkpoints), 231 s for the full run
| model | Brier | RPS | log loss | z |
|---|---|---|---|---|
| sim_full (chosen) | 0.115923 | 0.078104 | 0.349359 | -0.62 |
| point_sim | 0.1169 | 0.0780 | 0.3484 | 0.63 |
| elo_sim | 0.1235 | 0.0777 | 0.3674 | 0.81 |
| standings_now | 0.2105 | 0.1381 | 1.9391 | — |

On test sim_full is level with point_sim (−0.0009, CI [−0.0076, +0.0050]), ahead of elo_sim (−0.0076,
[−0.0241, +0.0073]) and well ahead of standings_now (−0.0946, [−0.1701, −0.0332]). With validation (point_sim ahead,
CI spanning 0) the honest reading: sampling M1's strength posterior neither helps nor hurts the direct-cut Brier
measurably at these sample sizes; it is calibrated (|z| < 1.96 in every split) and clearly beats the current table.
The gate stays FAIL (decided on validation, the point rule).

RUNTIME m7 euroleague 231

## K7: GBL (reported, not gated; the EuroLeague verdict `sim_full` as a fixed choice; tuning 2020-2022, test 2025)
| split | sim_full | point_sim | elo_sim | standings_now | sim_full z |
|---|---|---|---|---|---|
| tuning (111) | 0.1031 | 0.1082 | 0.1135 | 0.1622 | 0.92 (point_sim 2.77, elo_sim 4.14) |
| test 2025 (39) | 0.1751 | 0.1886 | 0.1616 | 0.2564 | 3.02 (every model > 1.96) |

In the GBL the point-strength baseline is miscalibrated on tuning (z 2.77) and sampling the posterior fixes it; the one
test season (13 teams) is miscalibrated for every model. GBL rules stay unverified (format notes in the report).

RUNTIME m7 gbl 53

## K8: live simulate (gate FAIL → built, not scheduled)
`uv run eurohoops simulate [--competition gbl] [--dry-run]` (`src/eurohoops/live_sim.py`): state = completed rounds k
(rounds 1..k all played; EL 2026-27 k = 1 on the mart copy of 28347b9, GBL k = 0), cutoff = round k + 1's first tip-off,
M1 posterior + pace at the cutoff (committed `tuned`), the verdict `sim_full`, 10,000 simulations, seed 20261101 + k.
It appends to `predictions/{competition}_sim_2026-27.csv` and writes `reports/sim_latest_{competition}.json` only when
the EuroLeague gate passed; one set of rows per completed round. **The gate failed, so no log file or latest report
exists, and `daily.yml` is unchanged (K-j).** Item 52 (`scripts/checks/sim_dry_run.py`) checks this on the real marts.
- **D11 (unseen live teams):** a team with no game before the cutoff (the two promoted GBL clubs before round 1) keeps
  M1's ridge prior, mean 0 (where M1 itself forecasts it) and variance σ²/ridge, independent. With the GBL's ridge
  (62.5 possessions) that prior is wide: before round 1 each promoted club gets ~12% title odds in a dry run, which is
  the honest consequence of no data and of M1's league-mean prior, and fades once it plays. Recorded as a limitation.

RUNTIME simulate 7

## Subagent log
- Worktrees: created by the orchestrator (`git worktree add C:\Python\EH-m7-{a,b,c} -b m7-{a,b,c} week-14-16-m7`), not by
  the Agent tool's `isolation: "worktree"`, which would have created them inside `C:\Python\Sports_Project` (off limits
  in this run). Same isolation; no merge-from-main step needed.
- Wave 1 A (K1 posterior): round 1 passed the coverage test only after moving the synthetic league from 10 to 12
  teams; rejected (test tuned to the observed result). Root cause: σ² without a degrees-of-freedom correction (D7).
  Round 2: estimator fixed, test back to 10 teams / same seed / same band; green. Prompt glitch: A's first prompt
  carried an unexpanded `{COMMON}` placeholder; the project rules were sent at once by message, before A's first commit.
- Wave 1 B (K2 formats): 1 round, green, no changes requested.
- Wave 1 C (K3 engine): 1 round, green. Measured: EL 20 teams / 369 games / 10k sims 9.1 s; 18-20 teams 4k sims 2-3 s.
  Merge conflicts in `scripts/vulture_whitelist.py` (three appended blocks) resolved by keeping all blocks.
- Wave 2 D (K4 harness): 1 round, no changes requested; D could not finish the full suite on the loaded machine (ran the
  touched test files); the orchestrator's fast gate covers it.
- Wave 2b E (K5 leakage): 1 round. Reading of "a change to a game at or after the cutoff": result edits leave every
  forecast and input bit-identical; schedule edits (deleting/adding a remaining fixture) leave the posterior, pace, Elo and
  standings state identical while the simulated schedule (an allowed input, §1) changes. Planted leaks (sampler reading
  the final table; posterior through the season end) are both detected.
