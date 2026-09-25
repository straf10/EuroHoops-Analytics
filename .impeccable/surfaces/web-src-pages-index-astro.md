---
version: 1
slug: "web-src-pages-index-astro"
primary_target: "web/src/pages/index.astro"
related_targets: []
---

# Surface brief: EuroHoops home page (web/src/pages/index.astro)

Scope: the single public page on GitHub Pages. Mode: persuade (recruiter-first, fan-usable).
Audience/job: a hiring lead understands in one minute that forecasts are pre-registered and honestly scored; a fan sees tonight's games and win probabilities.
Proof/content: predictions/*.csv (first logged row per game), reports/live_scorecard*.json, team Elo from the marts. v1 = predictions, results, scorecard, power ratings.
Constraints: Astro static build fed by JSON from `eurohoops publish`; no logos/photos; no betting framing; 390 px; light + dark.

## Direction contract

THESIS: The coach's clipboard. Every forecast is drawn up on the board before tip-off and marked up after the whistle. Refuses the dark KPI-tile stats dashboard.

OWN-WORLD: White melamine board (cool, not cream) inside a brushed-aluminium frame; three marker inks only: black (structure, numbers), blue (the model's call), red (the result mark-up). Printed court lines in faint grey form the grid. Teams are round magnetic pucks with a three-letter code. Marker hand only for short annotations; all numbers in a tabular grotesk.

STORY: Visitor sees tonight's games placed on the board by P(home), learns each was logged before tip-off, sees past calls circled hit or crossed miss in red, checks the scorecard against the baseline, browses the Elo ladder, follows the link to the public log.

FIRST VIEWPORT: Framed half-court board, left two-thirds: next-round games as puck pairs on a 0-100% axis drawn on the half-court line, blue marker percentages. Right third: headline "Called before tip-off.", one line of subtext, primary action "Read the log" (GitHub). Scorecard strip in the frame's top rail: Elo vs baseline log loss.

FORM: Coach's tactics clipboard, position 3 of my ordered list, seed key 0fd69dc5. Signature interaction: pucks slide from centre to their probability on load (damped, once); hovering/focusing a puck draws the marker line to its game row.

FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance
