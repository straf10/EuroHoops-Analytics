---
version: 1
slug: "web-src-pages-teams-code-astro"
primary_target: "web/src/pages/teams/[code].astro"
related_targets: ["web/src/pages/teams/index.astro"]
---

# Team pages

Mode: Operate. Audience: recruiters skimming for rigor, fans checking their club. Job: see how a club played in any of its EuroLeague seasons (record, ratings, four factors against every other club), where it shot from and what it allowed, and who played. Constraints: no logos or club colours; ratings derive from exported team and opponent totals; pre-2011-12 shot locations flagged, never hidden.

## Direction contract

THESIS: a team is read by where it ranks. Every rating and four factor is a strip of every club's dot that season, this club's in blue. Refuses the category default of a stat-box grid of isolated numbers.

OWN-WORLD: The Quiet Reference unchanged: paper and ink, Geist tabular figures, hairline rules; blue is this club, Baseline Gray the other clubs and the dashed league average; red only on the courts' below-average cells. Ink controls.

STORY: a visitor lands on a club, sees record, net rating and pace, reads five offense and five defense strips with a rank each, compares shots taken and allowed against the league, opens a player from the roster, or jumps to another club by its dot.

FIRST VIEWPORT: club name and one fact line (record, net rating, pace) with the season select on the right; below, two columns, Offense and Defense, each five dot strips (rating, eFG%, TOV%, OREB% or DREB%, FT rate) with value and rank.

FORM: Where they rank, position 3 of 7 on the ranked list, seed key 30ddfd3b. Signature interaction: on a season switch the club's blue dot slides to its new place on every strip (200 ms ease-out) while the gray field redraws; every other club's dot is a link to its page.

FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance
