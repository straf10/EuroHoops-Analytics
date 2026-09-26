---
version: 1
slug: "web-src-pages-leaders-index-astro"
primary_target: "web/src/pages/leaders/index.astro"
related_targets: []
---

# Leaders

Mode: Operate. Audience: fans and recruiters who want the best players on one measure. Job: see who leads the EuroLeague on a stat for one season, over whole careers, or across the best single seasons, and the same for one club (its roster, its all-time leaders, its best seasons). Constraints: season floor is 50% of the most games anyone played and 15 min a game; career floor is 60 games; counting totals need no floor; percentages keep their attempt floors. Club colours appear on the shirt only (the bounded exception in DESIGN.md). No photos, likenesses, crests or kit copies.

## Direction contract

THESIS: the leaders are a five on the floor. The top five stand on a half court as jersey backs, the next fifteen sit on the bench beside it. Refuses the category default of a plain top-N table.

OWN-WORLD: The Quiet Reference unchanged: paper and ink, Geist tabular figures, hairlines, the shared court in Axis hairlines on Chart Surface. The one bounded exception: SVG jersey backs in two club colours (body, trim) with surname and number in light or dark ink. No data colour on the floor or bench (the ranking is the order); ink controls.

STORY: a visitor picks a scope, a stat and optionally a club, reads the five on court and the bench order, sees each shirt's stat line, and opens a player page or compares the five.

FIRST VIEWPORT: filter row (Scope segment Season/Career/Best seasons, Season select, Stat select, Per game/Totals segment, Club select). Below: left, the half court (about two thirds) with five jersey slots, each shirt, name, club and PTS REB AST TS% PIR with the sorted stat in ink semibold; right, the bench, ranks 6 to 20 with a mini shirt, name, club, season, games and value. Under 620px of floor the shirts keep only the value; the five full lines list below the court. Compare these five link under the court.

FORM: Court lineup, position 4 of 7 on the ranked list, seed key 17ed15d4. Signature interaction: on a change of stat, scope or club, the shirts on court exchange with a 200ms fade-and-rise while the bench rows reorder in place; nothing animates on load.

FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance
