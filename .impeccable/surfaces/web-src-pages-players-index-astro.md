---
version: 1
slug: "web-src-pages-players-index-astro"
primary_target: "web/src/pages/players/index.astro"
related_targets: ["web/src/pages/players/[slug].astro"]
---

# Players (dashboard and player pages)

Mode: Operate. Audience: recruiters skimming for rigor, EuroLeague fans looking up a player. Job: find any player in any of 20 seasons, rank the league on any measure, open a player's history, game log and shot profile. Constraints: no photos or logos; all numbers derive from the exported raw totals; honest coverage (pre-2011-12 shot locations flagged).

Player page structure was specified by the approved plan (season history, game log, shot chart with zone profile against the league); it was shaped directly, not rolled. Shot Profile Twin arrives in Phase 6.

## Direction contract

THESIS: pick a measure and the league lines up by it. Refuses the category default of a bare leaderboard table where the ranking is only a column of numbers.

OWN-WORLD: The Quiet Reference unchanged: paper and ink, Geist with tabular figures, hairline rules, blue for the ranked measure, gray for the league-average reference, ink controls.

STORY: a visitor chooses season, window (season, last 5/10/20), rate (per game, per 36, per 100) and a measure; sees the top 25 as dots on one shared scale against the league average; scans or searches the full table; opens a player.

FIRST VIEWPORT: heading and one-line explainer; one filter row (season, window, rate, team, minimum minutes, search); left five columns the ranked dot chart for the sorted measure; right seven columns the sortable table. Every column header re-ranks both.

FORM: Metric ranking, position 3 of 6 on the ranked list, seed key 0fc47d09. Signature interaction: changing measure, window or rate slides every dot to its new position in one 200 ms ease-out transition while names and values update in place; nothing animates on load.

FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance

## Adaptations recorded at the finish review (2026-09-27)

- Owner decision (2026-09-27, recommendations accepted): the explainer is one line, "Every EuroLeague player since 2007-08, ranked on any measure. Click a column to re-rank."
- Owner decision (2026-09-27, recommendations accepted, on "Players: chart and table rows don't line up. Leave it."): the chart keeps its 27px slots and the table its 34px rows.
- Owner decision (2026-09-26/27): desktop table in its own scroll box with a foot fade and palette scrollbar; phones list the first 50 rows, then "Show all N players". Phones label every other axis tick.
