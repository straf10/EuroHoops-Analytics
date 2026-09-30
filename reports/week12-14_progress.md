# Weeks 12–14 progress: entity resolution + M4 league translation

Task file: `docs/history/prompts/week-12-14.md`. Branch `week-12-14` (from `main` c222eba).

## §0 sub-decisions (confirmed by the owner, 2026-09-30)
All defaults accepted as written (I-a … I-l). The owner's answers to the open points:
- I-c: fetch birth dates only if the spike shows both sources carry them.
- I-g: the owner labels the 200-id sample.
- I-h: the gate is against `same_stats` (PLAN wording); `team_offset` and `shrunk_same_stats`
  are reported, not gated. The small-sample rule (< 20 validation movers → 2019–2023 pooled,
  `translate` declared) stands.
- No new dependency: Jaro-Winkler written in-house.

| # | Decision |
|---|---|
| I-a | EuroLeague names 2007–2025, GBL 2018–2025 (including the `pbp:` keys); 2026-27 untouched; no EuroCup. |
| I-b | Person id `P:<EL id>` else `G:<ESAKE id>`; mart `player_xwalk`; source ids stay the keys everywhere else. |
| I-c | Bio spike (5 + 5 pages); birth dates as a matching feature and label evidence, kept in `data/` only. |
| I-d | Look-alike fold, accent strip, ELOT 743 + reverse-phonetic variants (≤ 64), sorted tokens; in-house Jaro-Winkler. |
| I-e | Blocking by career overlap (±2 seasons) + same Greek club-season; weighted score; one-to-one assignment; threshold on silver only; `entity/overrides.csv` wins. |
| I-f | Silver: same Greek club, same season, same jersey, both leagues; built without names; tuning only. |
| I-g | 200 GBL ids (≥ 100 min), 4 strata × 50; owner labels `entity/labels.csv` after the matcher commit; P/R with Wilson CIs as frozen and after overrides. |
| I-h | Movers GBL→EL (≥ 300 / < 100 / ≥ 300 min); standardised per-stat squared error, EL-minutes weighted; tuning 2019–2022, validation 2023, test 2024–2025 once; gate: player bootstrap (1,000, seed 20261015) CI upper < 0 vs `same_stats`. |
| I-i | The 11 H-e per-100 rates; M3 SPM as a reported extra. |
| I-j | Per-stat log-ratio model with dual indicator and player random effect (marginal likelihood); `translate_team` adds prior-season team net ratings. |
| I-k | Team offset from PAO/OLY club-seasons in both leagues; baseline `team_offset`; net-rating offset with bootstrap CI reported. |
| I-l | `eurohoops entity` < 300 s; `backtest --model m4` < 600 s. |

## §3 findings (2026-09-30, real data)
1. **GBL name markup:** all 993 ESAKE ids in `player_box` use the same markup (the prompt's
   "682" came from a too-narrow selector). Box-table cell:
   `<a …idplayer=ID…>#<jersey><div…photo…></div><span>SURNAME</span> FIRST</a>`; `##` = no
   jersey. Each id has exactly one (surname, first name) spelling across all its games. The 8
   other `player_box` ids are the `pbp:<jersey>:<First Last>` keys (Latin, first name first).
2. **ESAKE ids:** no (surname, first name) spelling belongs to two ESAKE ids. Reuse of one id by
   two people is not visible from names; birth dates (item 3) will show it.
3. **Bios (I-c spike, 5 + 5):** both sources carry birth dates, so the owner's rule applies and
   they are fetched. ESAKE `EsakeplayerView?mode=1` has a two-cell table `ΗΜ. ΓΕΝΝΗΣΗΣ`
   `dd-mm-yyyy`, `ΧΩΡΑ`, `ΥΨΟΣ`, `ΘΕΣΗ`. EuroLeague
   `v2/competitions/E/seasons/E{YYYY}/people?limit=2000` returns every person of a season in
   one call (883 in 2024; without `limit` it pages at 500); player rows have `type` `J`,
   `person.code` = box id without the `P`, `person.birthDate`. 2,349 of the 2,376 EL box ids
   2007–2025 have a birth date; the other 27 are placeholder ids (`000000`, `1`, `1234`, …).
   ESAKE player pages are slow (≈ 30 s each with timeouts and retries on 2026-09-30), so
   `eurohoops ingest-bios` fetches the PAO/OLY players first, then the rest by GBL minutes.
4. **Jerseys:** the GBL box gives a jersey on 22–85% of lines depending on the season (2018
   0.30 … 2025 0.85); 33 of 796 id-seasons show two jerseys. **The jersey-based silver set of
   I-f is noisy:** of its 44 unique pairs, 10 are different people by eye (e.g. `CALATHES
   NICK` ↔ `THOMAS, DESHAUN`, `ΣΛΟΥΚAΣ` ↔ `FALL, MOUSTAPHA`): players often wear another
   number in the other league. See D2.
5. **Mover counts (I4, after the tuning and before any M4 fit; `models/translation_pairs` on the
   crosswalk, I-h thresholds):** GBL→EL movers by EuroLeague target season: 2019: 1, 2020: 1,
   2021: 3, 2023: 1 — **6 in all, 1 in validation (2023), 0 in test (2024–2025)**. EL→GBL movers
   by GBL season: 2018: 10, 2021: 7, 2022: 8, 2023: 3, 2024: 2, 2025: 6 (36). Duals (same
   season, ≥ 300 minutes in both): 107 (6–17 per season). So the I-h small-sample rule applies
   (1 < 20 validation movers): the gate pools 2019–2023 (6 movers) with `translate` declared, no
   choice on data. It has almost no power, and test has no GBL→EL mover to score; this is
   reported as it is, not re-specified after seeing the counts.

## I4: matcher (frozen at the matcher commit)
- `ENTITY_PARAMS`: `w_surname` 0.5, `t_dob` 0.78, `t_near` 0.95, `t_nodob` 0.82, `b_club` 0.08,
  `b_jersey` 0, window ± 2 seasons, near = typo forms (D9). Tuning re-run after the I2
  transliteration fix (word-initial ΓΟΥ → W, ΑΪ): same choice (`reports/entity_tuning.json`).
- Real run (`eurohoops entity`, 71 s): 94,049 candidate pairs, 196 accepted, **190 GBL ids
  matched**, 2,998 persons; overrides: the two EuroLeague duplicates.
- Silver (dates visible): precision 1.0, recall 0.836 (117/140). By eye, 18 of the 23 misses
  are silver noise, not matcher errors: teammates with equal or ≤ 7-day birth dates (e.g.
  OSMAN ↔ KOUZELOGLOU, the Kalaitzakis twins), each GBL id matched to its own EuroLeague id.
  The 5 true misses: Vezenkov (first names ALEXANTER / ALEKSANDAR, 0.77), McKissic (0.699),
  George Papas (ESAKE `ΠΑΠΑΘΑΝΑΣΙΟΥ ΤΖΟΡΤΖ ΠΙΤΕΡΣ`, 0.741), Moses Wright (near date, 0.862),
  Thomas Walkup (near date, 0.873). No override added for them before the labels (that would
  be labelling by eye); they stay known misses unless the owner's labels or review add them.
- The 8 `pbp:` keys link to no ESAKE id: their team-seasons have no official line with that
  jersey (ESAKE's `#0` placeholder, D6) and a matching name, so they stay their own persons.

## Silver set and match rule, fixed before any tuning (2026-09-30)
- **D2 (silver set re-keyed, deviation from I-f).** Positives: a GBL id and an EL id on the same
  Greek club in the same season (`00000001` ↔ `PAN`, `00000002` ↔ `OLY`) with the **same birth
  date** (both known). Negatives: every other pair of that club-season with both birth dates
  known and different. Still built without names, so tuning the name score on it is not
  circular; jersey stays a matcher feature. Reason: item 4 (≈ 23% of the jersey silver pairs
  are wrong by eye), decided before any score was computed on it.
- **D3 (match rule, refines I-e).** name score = max(w·surname sim + (1 − w)·first-name sim,
  full sorted-name sim), each sim the best Jaro-Winkler over transliteration variants. Birth
  dates equal → match iff name score ≥ `t_dob`; different → never (only an override can link
  them); unknown on either side → match iff name score + `b_club`·same club-season +
  `b_jersey`·same jersey ≥ `t_nodob`. One-to-one assignment on the score. Tuning rule, fixed
  now: `t_dob` = the largest value keeping silver recall = 1 (all equal-date silver positives
  accepted); `w`, `b_club`, `b_jersey` and `t_nodob` on the silver set with birth dates
  **hidden**, maximising F1 subject to silver precision ≥ 0.99.
- **D4 (club map).** The GBL ↔ EL club map lives in `config.GREEK_EL_CLUBS` (used by the bio
  fetch order and the matcher) instead of a separate `entity/clubs.csv`: one source of truth.
- **D6 (GBL jersey `0`).** ESAKE prints `#0` for 313 GBL name rows 2018–2025, up to 12 players
  of one team-season: a placeholder, so it counts as no jersey. EuroLeague `Dorsal` `0` is kept.
- **D7 (wave 2 early).** D (pairs) and E (translation model) started before I4, on fixed column
  contracts and synthetic data only; they read no data and no matcher output.
- **D8 (label strata).** I-g's strata "best candidate score ≥ threshold / below" become
  {Greek-script, Latin-only} × {matched, not matched by the frozen matcher}: the same split,
  measured on the decision itself. The sheet with birth dates is written to
  `data/entity/labels_todo.csv` (never committed); only `gbl_id, stratum, stratum_size,
  el_id_true` go into `entity/labels.csv`.
- **D9 (birth dates disagree between sources; before the matcher commit, no label read).** With
  every GBL bio cached (993/993; EL 2,591), the first tuning run showed silver precision capped
  at 0.955 at every grid point. The capped pairs are silver "negatives" that are the same person
  with dates a few days or a month apart (Gist −2 d, Mitrou-Long −5 d, Abosi −1 d, Petrusev
  −4 d, Balcerowski one month), and equal dates can be teammates' coincidences (GBL Papapetrou
  and EL Lekavičius share one). Changes: a `near` status for typo-like disagreements (≤ 7 days,
  the same day one month apart, day/month swapped, or the year off by one; a first version with
  ≤ 31 days made 166 silver positives out of teammates born within a month of each other, so it
  was narrowed to the forms observed: 140 positives); silver positives = equal **or near**
  dates. Thresholds from the name scores of all 316 equal-date and 894 (±31-day) near-date
  candidate pairs (no label; bimodal: coincidences up to 0.778 for equal dates and 0.918 for
  near ones, true pairs from 0.771 and 0.871): `t_dob` 0.78, `t_near` 0.95 (loses e.g. Moses
  Wright at 0.871 to keep precision). The silver search now tunes only the missing-date branch
  (582 pairs, 220 EL ids without a date); its `t_nodob` and `b_club` grid was widened once (the
  first run chose the 0.86 / 0.08 edges). Result (dates hidden): `w_surname` 0.5 (the grid's
  lower edge, not widened again), `t_nodob` 0.82, `b_club` 0.08, `b_jersey` 0; silver precision
  1.0, recall 0.864.
- **D5 (translit fixture).** I0's pairs file starts with 17 pairs checked by eye (same
  club-season); it is extended to 40 with equal-birth-date PAO/OLY pairs once their bios are
  cached. Written from source ids by script, so the Latin look-alike letters are exact.

## Decisions this file did not cover
- **D1 (prompt correction before start).** The prompt first listed the two EuroLeague names with
  two ids as undecided; `reports/week9-12_progress.md` §3 had already settled them (Yurtseven:
  one person, two ids; Marko Simonovic: two people). Corrected in the task file before its
  first commit.

## Iterations
iteration | deliverable | checks run | result | commit
---|---|---|---|---
1 | branch, task file, progress file with §0 | none (docs only) | ok | cf1dc86
2 | §3 spikes; `ingest/bios.py` + `eurohoops ingest-bios`; seed translit fixture; D2–D5 | ruff, format, mypy, pytest (573; `test_workflow.py` needs Git Bash first on PATH: 12/12 then), vulture | ok | 3cb9e93
3 | I1 name tables (orchestrator) | test_player_names, real-data coverage (1,001 GBL / 2,188 EL ids) | ok | 8f5c5b7
4 | wave 1 B (I2) merged; real-name check; ΓΟΥ/ΑΪ rules sent back | I2 tests re-run (38) | ok | 0ce44e9
5 | wave 1 C (I3) merged; matcher speed-up (240 → 88 s) | I3 tests re-run (16) | ok | 04acc91, fb435f0
6 | entity pipeline + `eurohoops entity`; D8 | entity tests (31) | ok | d5a5aeb
7 | bios complete (993/993 GBL); D9 near dates; tuning | entity tests (32), tuning run | ok | 44f198c, b405c44
8 | B's fix, D (I6), E (I7) merged; re-tuning; real run; mover counts | 90 merged tests, mypy | ok | 614a249
9 | I4 matcher commit | real run, silver metrics | ok | d17fc3a
10 | I5 label sheet drafted (186 ids; `latin/matched` has 36 in all); I8 code | test_m4_backtest (6), mypy | ok | 6e2643e
11 | I8 verdict: tuning only (5 movers); variant `translate` declared by the small-sample rule | backtest --tuning-only (18 s) | ok | 044f670
12 | I8 validation scored once; gate (pooled 2019–2023, 6 movers) FAIL: translate − same_stats −0.138, 95% CI [−0.430, 0.018] | backtest (19 s) | ok (gate outcome FAIL, allowed) | (this commit)

## Pre-registration order
MATCHER d17fc3a
VERDICT 044f670
