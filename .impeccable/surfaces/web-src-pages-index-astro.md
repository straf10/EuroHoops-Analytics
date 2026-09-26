---
version: 1
slug: "web-src-pages-index-astro"
primary_target: "web/src/pages/index.astro"
related_targets: []
---

# Surface brief: EuroHoops site (web/src/pages/index.astro and the new stats pages)

Scope: the whole public site. Home and Forecasts in Phase 1; Players, player pages, Teams, Leaders, Compare, Shots and Shot Profile Twin follow. Mode: operate (stats tool), home leans persuade.
Audience/job: recruiters see rigor and craft in a minute; EuroLeague fans look up players, teams, shots and tonight's forecasts.
Proof/content: pre-registered predictions log, live scorecard, backtests, Elo ratings; later EuroLeague box scores, play-by-play and shot coordinates 2007-08 to today.
Constraints: Astro static, JSON exported by the Python pipeline; no logos or photos; no betting framing; 390 px; light and dark follow the system with a toggle.
Peers named by the user (craft bar): boxscorelab.com, databallr.com. The user rejected every themed metaphor (clipboard, station board, tickets, terminals): the product is a premium, minimal stats site, played straight.

## Direction contract

THESIS: A premium, quiet EuroLeague reference: the data is the only loud thing. Refuses both the themed-metaphor site and the dark neon KPI-tile dashboard.

OWN-WORLD: Near-neutral paper (#f9f9f7 page, #fcfcfb chart surface) and near-black (#0d0d0d page, #1a1a19 surface). One UI family (Geist) with tabular figures in every column. Hairline rules, no cards, no shadows except popovers. Controls are ink, not colour: the selected segment is filled ink. Colour lives only in data: validated categorical order starting blue (the model), blue sequential ramp, blue-to-red diverging with a gray midpoint, stepped bands.

STORY: The visitor lands on the league (tonight's forecasts first, then the stats sections as they ship), scans a table, opens a player or team, compares, and can trace every forecast to the public log.

FIRST VIEWPORT: Slim single-line top bar (wordmark, sections, competition switch, theme toggle, GitHub). Left: headline "Called before tip-off." with one line of subtext and the log link. Right, wider: tonight's games as rows with a split probability bar per game (home share in blue, away in gray), tip-off time, and expected margin.

FORM: Category standard played straight (the user's exit from the direction round, after two re-rolls), seed key fa463703. Signature interaction: every forecast bar grows out of even odds (50%) to its call when its panel appears, on load and on each competition switch (320 ms ease-out, 40 ms stagger; panels fade in over 200 ms); on the stats pages, switching per-mode or season retargets every bar and number in one 200 ms ease-out transition. Hover or focus on any mark shows a precise tooltip.

FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance

## Adaptations recorded at the finish review (2026-09-27)

- Hero subtext runs to three lines on desktop, not one. Kept as written: it is factual product copy, and copy is replaced only with the owner's word. Open for the owner to shorten.
- Split bar: the away share is always the same Baseline Gray (no dimming for the underdog); the favourite is carried by its ink semibold percentage.
- Owner decision (2026-09-27): on phones the competition switch takes a full-width second row of the bar, so the site links keep the first row and "Forecasts" shows in full.
