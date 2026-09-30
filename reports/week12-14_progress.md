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

## §3 findings
Pending (iteration 2).

## Decisions this file did not cover
- **D1 (prompt correction before start).** The prompt first listed the two EuroLeague names with
  two ids as undecided; `reports/week9-12_progress.md` §3 had already settled them (Yurtseven:
  one person, two ids; Marko Simonovic: two people). Corrected in the task file before its
  first commit.

## Iterations
iteration | deliverable | checks run | result | commit
---|---|---|---|---
1 | branch, task file, progress file with §0 | none (docs only) | ok | (this commit)
