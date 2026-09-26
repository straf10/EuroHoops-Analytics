---
version: 1
slug: "web-src-pages-shots-index-astro"
primary_target: "web/src/pages/shots/index.astro"
related_targets: []
---

# Shots explorer

Mode: Operate. Audience: fans and analysts, recruiters checking depth. Job: pick the league, a team (shots taken or allowed) or a player in any season, then cut the attempts by quarter, distance band and play type and read where and how well they were taken against the league under the same cut. Constraints: every attempt comes from the exported per-season attempts file; pre-2011-12 locations flagged.

## Direction contract

THESIS: the filters are charts. Each facet (quarter, distance, play type) is a small bar chart of the current selection's attempts that is also the control. Refuses the category default of a dropdown sidebar beside a static chart.

OWN-WORLD: The Quiet Reference unchanged: court in Axis hairlines on Chart Surface; blue for included bars and made shots; Baseline Gray for excluded bars and the league reference; red only for below-league cells and misses (hollow, so shape also carries it).

STORY: choose subject and season, choose view (frequency, FG% against the league, attempts), click bars to cut; every facet and the court update together; the totals line states attempts, FG% and points per shot against the league.

FIRST VIEWPORT: one row of subject, season and view controls; court on the left seven columns; on the right the totals line then stacked facets Quarter, Distance, Play type, and Clear filters.

FORM: Filters that are charts, position 5 of 7 on the ranked list, seed key 425eeb26. Signature interaction: a click on a facet bar toggles it; other facets' bars resize in place (200 ms) and the court redraws.

FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance
