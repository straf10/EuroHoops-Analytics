---
version: 1
slug: "web-src-pages-compare-index-astro"
primary_target: "web/src/pages/compare/index.astro"
related_targets: []
---

# Compare

Mode: Operate. Audience: fans settling an argument, recruiters checking the depth of the numbers. Job: put up to five players side by side, each as a chosen season or his whole career, and see who leads each measure. Players only; no teams. Constraints: club colours on the shirt only; career numbers are sums of season totals; a percentage below its attempt floor is shown muted and never leads its row.

## Direction contract

THESIS: a comparison is a set of head-to-head rows. Each measure is one row across the five, the leader marked. Refuses the category default of overlaid radar charts and a sea of coloured series.

OWN-WORLD: The Quiet Reference unchanged: paper and ink, Geist tabular figures, hairline rules. Jersey cards head the columns (the bounded club-colour exception). One blue dot marks each row's leader; values stay ink; ranks in muted text. No per-player series colours.

STORY: a visitor adds players by name search (or arrives from a player page or Leaders), picks a season or career for each, reads the rows-led tally, scans the grid by section (Scoring, Shooting, Playmaking, Rebounding, Defence), and opens any player's page.

FIRST VIEWPORT: filter row (player search with up to five chips, Scope segment Season/Career, Per game/Per 36/Totals). Then five columns: jersey card heads (shirt, name, club, season select, remove), the rows-led tally line, and the grid: measure name at left, five value cells each with its place among the compared, the row leader with a blue dot and semibold ink.

FORM: Head-to-head grid, position 5 of 7 on the ranked list, seed key c70c626f. Signature interaction: on adding a player, changing a season or the rate, the blue leader dots move to their new cells (160ms) and the tally recounts; nothing animates on load.

FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance
