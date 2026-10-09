# Site feedback backlog (2026-10-09)

Notes from the owner's review of the live site. Not yet planned or scheduled; `docs/PLAN.md` decides order.

Guiding rule for everything below: the average user wants stats and a quick answer. Strip AI-ish disclosures, internal status labels and methodology prose from every page except the pages that exist to hold it (About, Methodology, Glossary).

## Scope decisions (made)
- **Remove GBL entirely** for now (site, API, pipeline output shown to users).
- **Remove the Place column** from Standings. Teams are already sorted by expWins; the placement buckets (Top 6, Play-in, etc.) carry the information.
- **Hide the Scouting page** for now. It shows little because most samples are tiny.
- **Remove "Methodology" from the top category links.** It moves to the footer.
- **Fold the Shots page into team and player pages.** It does not earn a page of its own.

## Global chrome
- **Footer disclosure**: replace the current block ("A model benchmark, not betting advice…", "Updated Fri 9 Oct, 20:26 Athens", "Code, data and logs on GitHub") with:
  - A line such as "Stats from the EuroLeague API. Not affiliated with the EuroLeague, Greek Basket League or any team."
  - A freshness line such as "Data through Oct 9, 2026 · build 20261009T1133" (build = the current build id/hash).
  - No GitHub link in the footer.
- **Footer link row** (above the disclosure): clickable links to secondary pages: About, Methodology, Glossary, Support, Downloads, and possibly more.
- **Name and logo**: the site needs a new name (EuroHoops is a well-known European media brand, so the name must change). The new logo should read as basketball: half court, 3-point line, ball, or similar. Open: pick the name and logo direction.
- **Colour**: EuroLeague's identity is black and orange. Add orange accents on data and chrome. Keep the "Quiet Reference" rule in `docs/DESIGN.md` that colour appears only on data, so reconcile the two before applying it.

## Standings
- Remove the Place column (see above).
- Shorten the explainer. Current text (too long): "How each season could finish: the rest of the fixtures played many times over, then the tables counted. The EuroLeague projection passed its validation gate (v2, over seven past seasons)…" Target: one short sentence, or nothing.
- Remove the internal status label "Gated" and the "M7 passed its validation gate (gate v2, reports/backtest_m7_v2.json)" wording from the UI entirely. Users do not need it.
- Remove the probability-table explainer ("Place is the median place in the simulated tables, with the 80% range…", "Chances are shares of the 10,000 simulations."). If a probability needs a hint, use a single short tooltip.

## Methodology page (new structure)
- Use the same structure as the pasted example: a short lead, then Source, Refresh, What each build contains (generated table), Derived metrics and what they assume, Citing, Corrections.
- **Adapt it, do not copy it.** The pasted copy is written for BoxScore Lab and NBA Stats. Rewrite every sentence for EuroHoops and the EuroLeague API. Do not carry over the other site's name, its "not affiliated with the NBA" line, or its contact profile (`github.com/adrielconde9`). Use the owner's own contact.
- Remove internal labels ("Gate passed", "Runs beside Elo as a shadow log", and similar). Describe the models in plain words only.
- The coverage table should be generated from the build, not typed by hand.

## About page
- Use the pasted structure ("Why it exists", "Who builds it", "How to read it", "Keeping it free", "Independence") as a template, rewritten for EuroHoops.
- Cut it to a few short paragraphs. Drop the "six daily games" line and the "877 players / 30 teams" counts unless they are generated.
- Keep the "not affiliated" statement and the "not betting advice" statement, but say them once, here.

## Glossary page (new)
- Plain-language definitions of the terms the site uses. Start from the pasted list (eFG%, xeFG%, TS%, ORtg, DRtg, Net rating, Pace, Poss, SOS, USG%, AST%, TOV%, Net (stab.), Weight, Sample, etc.). Keep only the terms that appear on EuroHoops pages, and write each in one or two plain sentences.
- Drop the "Everything else" and "Value / Final / Fit / Call" entries if those scouting-model fields are not shown on the site (they belong to the hidden Scouting page).

## Downloads page (new)
- Let anyone download the data as CSV.
- Open question: whether free-tier hosting can carry it. Check size per dataset before promising it. Options: publish only the marts users need, or serve files from a release asset.

## Data scope
- Keep only the most recent 3 to 5 seasons, and add older seasons gradually.
- Add validation on each load: missing data, format or column mismatches, and special seasons (for example COVID-affected 2019-20 / 2020-21). Flag these in the build, not silently.

## Models
- The owner does not understand why the site runs several models. Goal: one strong model, either standalone or as an ensemble of smaller ones.
- Action: audit which models are live, which are shadow or experimental, and which are used on the site. Propose a single model with a clear reason, then retire the rest. Do this with the pre-registered prediction rules in mind: `predictions/*.csv` stays append-only.

## Leaders page
- Good overall. Remove the clutter in a few places (extra explainers, badges, and labels the average user does not read).

## Teams page
- Keep the idea of showing all core stats and team ranks.
- The per-stat diagram/graph is not clear. Replace the chart type. Open: choose between ranked bars with the league median marked, or a simple value with rank and a small range bar.

## Open questions for the owner
1. New site name and logo direction.
2. Whether Downloads ships on the free tier (depends on dataset size).
3. Which single model to keep, or whether to ensemble.
4. Teams page chart type.

## Round 2 notes (2026-10-09)

### Teams page
- **Colour the comparison.** Where a stat is shown against the league average, colour it green if better and red if worse, so the eye goes to it.
- **Team header block.** Under the team name, show a small section with:
  - Record and season, for example "25–19 · 2025-26".
  - Net rating, coloured green or red, with the offensive/defensive split in a shadowed comment line, for example "108.6 off · 113.9 def".
  - Pace.
  - Seasons in EuroLeague.
  - League rank, for example "23rd in the league".
  - Playoff qualification, in its own sub-box.
  - Each item sits in its own sub-box. Drop the current single-line header text.
- **Four Factors layout.** Use the Four Factors section style as the model for the per-stat graphics. Reference layout: the Dallas Mavericks team page on boxscorelab.com. It is less cluttered. Copy the layout idea only, with our own wording and data.

### Players page (list)
- **Simplify the filters**, following the decluttered filter bar on boxscorelab.com/players/. Keep only:
  - Metric (the sort/stat you want).
  - Season.
  - Minutes: a number the user types, or a choice from preset thresholds.
  - Show top: 5, 10, 25, 50, 100, all.
  - Position: Guard, Wing, Big. The owner is most interested in this one.
- Remove the other filters.

### Player page (individual)
- **Header.** Keep the mini description (name, number, team, "N EuroLeague seasons, first to last season", "Compare with others", and the one-line season summary). Use it as the model for the header.
- **Season-by-season stats move lower.** The top of the page should show the current/summary numbers, not the full season-by-season table.
- **New sections, in this order** (reference: boxscorelab.com/players/luka-doncic/):
  1. Against-the-league graph for key metrics, with a leaderboard beside it.
  2. Form section (recent games).
  3. Shot chart: frequency and attempts heatmap.
  4. Shot quality.
  5. On and off the floor.
  6. Clutch.
  7. Projection: how well the player is expected to score by season end (ours).
  8. Shot profile: where the player is better or worse than league average (ours). Place this beside the other graphs.
  9. Shot twin: our analysis of where the player shoots, how well, and the style. Data is thin, so see the caveat below.
- **Data caveat for shot twin.** Current samples are too small to be meaningful. Options to decide: filter by season, or wait for more games before showing it. Until then, hide it or label it as preliminary.
- **Shot chart style.** The owner prefers the regular style (circles for makes, x's for misses) over the beehive style. Use the regular style.

### Forecasts page
- Redesign the layout. The current page is hard to read.
- Do not show only the last 12 games and how they scored. Show:
  - A clear view of our predictions.
  - Overall score (accuracy or similar), with a graph of it over time.
- Remove the "Read the log" button from the headline. Move it lower on the page, smaller, for people who want the raw log.
- Keep the wording plain (no internal status labels, see Round 1 rules).

### Reminder: copy and references
- The reference sites (boxscorelab.com) are for layout and section order only. Write every label and sentence for EuroHoops.

## Owner answers (2026-10-09)
- Order: quick wins first, redesigns after.
- GBL: removed for users only (site, API, published output). Ingestion, GBL logs and the daily GBL steps keep running; existing rows stay (append-only).
- Seasons: models keep training on full history; the site shows the most recent 5 seasons.
- Scouting and Shots: unlinked, code kept, old URLs redirect (Shots → team/player pages, Scouting → home).
- Rename: brand only (site title, header, meta, logo). Repo, package and CLI stay `eurohoops`. Owner wants name ideas.
- Logo: simple SVG mark drawn in code, monochrome with an orange accent.
- Colour: option (a), orange is the data accent and chrome stays neutral. Green/red only on metrics compared with the league average, kept minimal; consult the design skills (taste, impeccable) when applying it.
- No Support page yet. Contact = owner's GitHub (`github.com/straf10`), no email yet.
- Footer freshness line uses the build date.
- Models: audit first, recommend, owner decides. Retired models stop getting rows; past rows stay.
- Teams: the Four Factors layout settles the chart question. Header follows the season switcher. League rank = current standings place. Playoff box: actual result for past seasons; current season shows only the standing.
- Players list: Position Guard / Wing (= Forward) / Big (= Center). Minutes = per game, typed or presets.
- Player page: on/off and clutch only for seasons whose stints are already processed. Projection = projected season-end points per game. Shot twin hidden below a shot threshold. Leaderboard beside the against-the-league graph = season top players in that metric, this player highlighted.
- Forecasts: compare against bookmaker odds; show the best model.
- Downloads and hosting: check sizes, free hosting, and services (Cloudflare, Sentry, others).
- Leaders: remove every gated / not-gated / validation label.

## Plan (draft, 2026-10-09)

### Facts checked
- Hosting today: GitHub Pages, built and deployed by `daily.yml` (no custom domain).
- Mart sizes as CSV (all seasons, uncompressed): shots 87 MB, shot_xpts 28 MB, stints 19 MB, player_box 1.8 MB, team_games 1.2 MB, games 0.8 MB, the rest < 1 MB. Over 5 seasons per-season files are a few MB each (shots largest), well under Pages' and Cloudflare's per-file limits. Downloads fit the free tier.
- The live projection (M6) is per 100 possessions. Season-end points per game must be derived: projected per-100 rate × projected possessions per game (minutes share × team pace). New derivation, not a new model.
- Site-wide, 66 occurrences of gate / shadow / benchmark / Place / GitHub wording across 18 files in `web/src`.
- Pre-registered rule (PLAN 8.1 R1): Elo is the published forecast until M5 (EL) beats it at the round-19 look or at season end. "Show the best model" has to work inside that rule: see the open question below.

### Phase A: quick wins (one branch, small PRs)
1. GBL out of the site: nav, pages, competition switches, `publish` output, API JSON. Pipeline untouched.
2. Standings: drop the Place column, the explainers and every gated label; at most one short tooltip on the chance columns.
3. Remove internal labels site-wide (gated, gate passed, shadow log, gate v2, report paths). Methodology describes models in plain words only.
4. Hide Scouting and Shots: remove from nav, add redirects; keep code and vulture whitelist entries as needed.
5. Nav: Methodology leaves the top links.
6. Footer: a link row (About, Methodology, Glossary, Downloads); the "Stats from the EuroLeague API. Not affiliated with the EuroLeague or any team." line; "Data through <build date> · build <UTC stamp>-<short SHA>"; no GitHub link.
7. Seasons shown: one constant (last 5 seasons) applied in `publish` / `export-stats`; models unaffected.
8. Leaders: strip extra explainers, badges and labels.

### Phase B: identity and secondary pages
1. Name: owner picks from the shortlist; brand strings changed in one place (layout, meta, README headline).
2. Logo: SVG mark (half court with the arc), favicon set.
3. `DESIGN.md` amendment: orange as the data accent (replaces the current primary data hue where it fits), green/red only for "vs league" deltas, never as the only cue (add arrow or sign). Run the design skills before applying it.
4. About, Methodology (lead, Source, Refresh, generated coverage table, Derived metrics, Citing, Corrections) and Glossary (only terms the site shows), all written fresh.
5. Downloads: gzipped CSV per season for games, team games, player box, shots (5 seasons), generated at publish time with a manifest (rows, size, date). Full history later via a GitHub Release asset if wanted.
6. Load validation: per-season coverage checks (missing games, column or format mismatch) and declared special seasons (2019-20 cancelled, 2020-21 no crowds); flags go into the build report and the Methodology coverage table.

### Phase C: redesigns
1. Teams: header sub-boxes (record and season, net rating with off/def line, pace, EuroLeague seasons, standings place, playoff result for past seasons), season switcher, Four Factors-style per-stat graphics with league marks, green/red deltas.
2. Players list: filters reduced to Metric, Season, Min/game (typed or presets), Show top, Position.
3. Player page: header kept; summary first; sections in the owner's order; shot chart in makes ○ / misses × style; on/off and clutch only for processed seasons; projected season-end PPG; shot twin hidden below the threshold; season-by-season table at the bottom.
4. Forecasts: one model's predictions up front, accuracy over time chart against the bookmaker (de-vigged odds), raw-log link small at the bottom.

### Phase D: models audit
Report: what each model (Elo, M1, M5, M6, M7) does, what is live, logged or shown, and its backtest and live numbers. Recommendation, owner decides. Retired models stop logging; past rows stay.

### Phase E: hosting and services (all free unless noted)
- Hosting: move to Cloudflare Pages (or Workers static assets) via direct upload from the existing GitHub Actions build: unlimited bandwidth, custom headers and redirects (`_redirects`, needed for Phase A4), preview deploys. GitHub Pages works too but has no redirects or headers and a soft 100 GB/month bandwidth cap.
- Domain: the only real cost (~€10/year), DNS on Cloudflare (free TLS, CDN, caching).
- Analytics: Cloudflare Web Analytics (free, cookieless, no banner needed).
- Errors: Sentry free plan in the browser bundle (front-end errors) and Sentry Cron monitoring or healthchecks.io for the daily pipeline (alert when the 08:00 run fails or does not report).
- Uptime: UptimeRobot or Better Stack free monitor on the home page.
- Search: Google Search Console plus sitemap (`@astrojs/sitemap`).
- Verify current free-tier limits at setup time; they change.

### Decided (2026-10-09, second round)
1. Name: **6.75 Analytics**, short brand **6.75** (the EuroLeague/FIBA 3-point distance in metres).
2. Models: the site stays Elo-based for now (R1 kept). No new model is planned in weeks 18-20, so the plan is to pick the best existing model and fine-tune it; the Phase D audit names it.

### Open for the owner
1. Buy a domain (yes or no), which decides whether Phase E moves hosting now.
