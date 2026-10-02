# Players across leagues: the entity crosswalk

Weeks 12–14 I1–I5 (`docs/history/prompts/week-12-14.md`, decisions D2–D9 in
`reports/week12-14_progress.md`). Command: `eurohoops entity` (local, reads the marts, the raw
cache and the cached bios). Reports: `reports/entity_resolution.json`,
`reports/entity_tuning.json`. Every decimal below is in the Numbers table, checked by
`tests/test_model_card_m4.py`.

## Sources
- **EuroLeague:** box-score lines (`Player_ID` `P` + 6 digits, space-padded; `Player`
  `SURNAME, NAME`; `Dorsal`), every rated game 2007–2025. Birth dates from the v2 API
  (`/v2/competitions/E/seasons/E{YYYY}/people?limit=2000`, one call per season).
- **GBL (ESAKE):** the box-table cell of each cached game page,
  `#<jersey><photo><span>SURNAME</span> FIRST`, mostly in Greek capitals with Latin look-alike
  letters inside Greek words (`ΚΩΝΣΤAΝΤΙΝΟΣ` has a Latin A). `##` and `#0` mean no jersey
  (ESAKE's placeholder, D6). Birth dates from each player's `EsakeplayerView` page
  (`ΗΜ. ΓΕΝΝΗΣΗΣ`), fetched once at ≥ 2 s per page (`eurohoops ingest-bios`, local only).
- Birth dates are a matching feature only: they stay in `data/`, never in a committed file.

## Transliteration and similarity (`entity/translit.py`, `entity/similarity.py`)
Per token, the majority script wins and look-alikes are folded into it; accents and diaeresis
are removed after the diaeresis has been read (`ΑΪ` is not `ΑΙ`). Greek tokens get ELOT 743 plus
reverse-phonetic variants for foreign names (ΜΠ→B, ΝΤ→D, ΓΚ→G, ΤΖ→J, word-initial ΓΟΥ→W,
word-final Σ→CE/SE, word-final Ι/Η→EE, …), at most 64, tokens sorted so name order does not
matter. `latin_key` drops generational suffixes (JR/SR/II/III/IV). Similarity is Jaro-Winkler.
The 40-pair fixture gate is ≥ 38 surnames at Jaro-Winkler 0.95; Papathanasiou→Papas and
Ogkast→Auguste stay below.

## Matcher (`entity/match.py`, D3 and D9)
- Blocking: career windows within 2 seasons, and the same Greek club-season, equal or near
  birth dates, or a matching first letter.
- Name score: the best of the weighted surname and first-name similarity and the full-name one.
- Birth dates: **equal** → match if the name score ≥ 0.78; **near** (the two sources disagree
  like a typo: ≤ 7 days, the same day one month apart, day and month swapped, the year off by
  one) → ≥ 0.95; **different** → never (only an override); **unknown** → a stricter name rule
  with a same-club bonus. One-to-one assignment on the score; `entity/overrides.csv` wins.
- The two sources' birth dates do disagree: Gist by 2 days, Balcerowski by a month; and
  teammates share birthdays (the Kalaitzakis twins), which is why a date alone is not enough.

## Tuning: the silver set
Built without names: a GBL id and a EuroLeague id on PAO (`00000001` ↔ `PAN`) or OLY
(`00000002` ↔ `OLY`) in the same season with equal or near birth dates are positives, other
pairs of that club-season negatives. The missing-date rule was tuned on it with dates hidden
(precision 1.0, recall 0.864286). With dates visible the matcher scores precision 1.0 and recall
0.835714; by eye 18 of the 23 misses are silver noise (teammates born within days of each
other, each matched to his own EuroLeague id), and 5 are true misses (Vezenkov, McKissic, George
Papas, Moses Wright, Thomas Walkup).

## Result
The crosswalk holds 2,997 persons: 191 GBL ids matched to a EuroLeague id, 4 ids linked by
override (Omer Yurtseven's two EuroLeague ids, and Vezenkov's GBL and EuroLeague ids), and every
other id its own person. The 8 `pbp:` keys of the 2018–20 fill found no ESAKE id with their
jersey and name.

**Precision and recall on the owner-labelled sample (I-g).** The owner labelled 186 GBL ids in
4 strata (`entity/labels.csv`: 87 matched to a EuroLeague id, 99 with none), after the matcher
commit. Matcher as frozen: precision 1.0 (95% CI 0.957242 to 1.0), recall 0.988506 (95% CI
0.937728 to 0.997968), recall weighted by stratum size 0.993377. Every match the matcher made on
the sample is right (86 of 86), in the Greek-script and the Latin-only strata alike; the strata
it left unmatched hold 100 ids, one of them a true match. That one is a known miss:

- **Alexander (Sasha) Vezenkov** (GBL `00000069`, EuroLeague `P003469`): same birth date, same
  Olympiacos #14 career; the name score is 0.77, under the 0.78 threshold for equal birth dates.

An override from the labels (reason starting `label:`, so the frozen run excludes it) fixes
him; after it, precision and recall on the sample are 1.0 (recall 95% CI 0.957712 to 1.0). That
second set is fitted on the labels and is not an estimate. The M4 evaluation reads the crosswalk
as frozen (`player_xwalk_frozen`, no `label:` overrides), so its pre-registered scores do not
move; `player_xwalk` carries every override.

## Numbers
| Value | Shown | Source |
|---|---|---|
| persons | 2997 | `entity_resolution.json:persons` |
| matched GBL ids | 191 | `entity_resolution.json:matched_gbl_ids` |
| candidate pairs | 93344 | `entity_resolution.json:candidate_pairs` |
| accepted pairs | 196 | `entity_resolution.json:accepted_pairs` |
| override ids | 4 | `entity_resolution.json:by_method.override` |
| silver positives | 140 | `entity_resolution.json:silver.positives` |
| silver negatives | 3055 | `entity_resolution.json:silver.negatives` |
| silver precision | 1.0 | `entity_resolution.json:silver.precision` |
| silver recall | 0.835714 | `entity_resolution.json:silver.recall` |
| silver misses | 23 | `entity_resolution.json:silver.fn` |
| label precision, as frozen | 1.0 | `entity_resolution.json:labels.as_frozen.precision` |
| label precision CI low | 0.957242 | `entity_resolution.json:labels.as_frozen.precision_ci95.0` |
| label recall, as frozen | 0.988506 | `entity_resolution.json:labels.as_frozen.recall` |
| label recall CI low | 0.937728 | `entity_resolution.json:labels.as_frozen.recall_ci95.0` |
| label recall CI high | 0.997968 | `entity_resolution.json:labels.as_frozen.recall_ci95.1` |
| label recall, weighted | 0.993377 | `entity_resolution.json:labels.as_frozen.recall_weighted` |
| label recall, after overrides | 1.0 | `entity_resolution.json:labels.after_overrides.recall` |
| label recall CI low, after overrides | 0.957712 | `entity_resolution.json:labels.after_overrides.recall_ci95.0` |
| GBL birth dates | 993 | `entity_resolution.json:bios.gbl` |
| EuroLeague birth dates | 2591 | `entity_resolution.json:bios.euroleague` |
| tuning recall, dates hidden | 0.864286 | `entity_tuning.json:chosen_silver_hidden_dates.recall` |
| tuning precision, dates hidden | 1.0 | `entity_tuning.json:chosen_silver_hidden_dates.precision` |
| t_dob | 0.78 | `entity_tuning.json:chosen.t_dob` |
| t_near | 0.95 | `entity_tuning.json:chosen.t_near` |
| t_nodob | 0.82 | `entity_tuning.json:chosen.t_nodob` |
