---
version: 1
slug: "web-src-components-shottwin-astro"
primary_target: "web/src/components/ShotTwin.astro"
related_targets: ["web/src/pages/players/[slug].astro"]
---

# Shot Profile Twin (player page section)

Mode: Operate. Audience: fans and recruiters on a player page. Job: see which EuroLeague player-seasons since 2007-08 shot most like this player over his last 5, 10 or 20 games or his latest season, and exactly where the match holds or breaks. Constraints: data from `twins.json` (stats exporter, `stats/twins.py`), precomputed per window; pool is player-seasons with 150+ located attempts; a window never matches the seasons its own games come from; pre-2011-12 twins carry the approximate-coordinates flag.

User decisions (2026-09-27): windows 5/10/20 + season; profile = where (14 zones) + how well (FG% by band against the league) + 3PA and FT rate; pool = all seasons, 150+ FGA, his other seasons count and are marked; top 5 twins as a section on the player page.

## Direction contract

THESIS: two shot diets mirrored zone by zone, so every difference is a lopsided row. Refuses the similar-players default: a list of names with a similarity score and nothing to show why.

OWN-WORLD: The Quiet Reference unchanged: paper and ink, Geist tabular, hairline rules, him in Model Blue, the twin in Baseline Gray, ink controls, no cards.

STORY: the visitor picks a window, sees five twins ranked by match, picks one, and reads the mirror: zones where both bars reach equally are the shared habit, a one-sided row is where they part; FG% carets say whether each converts better than the league.

FIRST VIEWPORT: title, one-line explainer, window segmented control right. The five twins as one row of choices (name, season · club, match score), the first chosen. Then the staff: 14 zone names down a centre column from rim to deep three; his share bars grow left in blue, the twin's grow right in gray, share figures at the outer ends; per-band FG%-against-league caret beside each side's band. Two style rows (3PA rate, FT rate) and a method note close it. Symmetry is the layout law; on phones the columns thin but never merge.

FORM: Mirror staff (butterfly on a centre staff), position 6 of 7 on the ranked list, seed key ea3f8f82. Signature interaction: choosing another twin or window grows the right-hand bars (and, for a window, the left) from their old lengths to the new ones in one 200 ms ease-out; nothing animates on load.

FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance
