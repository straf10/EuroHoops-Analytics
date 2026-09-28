# Task: shrink the EuroHoops site and fix three phone UX problems

Repo: `C:\Python\Sports_Project` (Windows 10; PowerShell and Git Bash). The site is Astro (`web/`). It's built to `site/` from `web/src/data/site.json` and `web/src/data/stats/`, and deployed to GitHub Pages at the base path `/EuroHoops-Analytics` by `.github/workflows/daily.yml`.

Read `CLAUDE.md`, `DESIGN.md`, `PRODUCT.md`, `CONTEXT.md` and the last entries of `memory.md` first. Their rules override this prompt.
- `DESIGN.md` is the design authority: "The Quiet Reference".
- `memory.md` includes "Site perf pass" (2026-09-27) and its follow-up.

## Branches
- `perf/site` is ahead of `main` by 6 commits and is not merged.
- **G1–G2 (no visible change):** `git switch perf/site && git switch -c perf/site-2`.
- **G3–G5 (visible phone changes):** when G1–G2 are done, `git switch -c ux/phone` from `perf/site-2`.
- Never push or merge.

## Step 0 (first commit on `perf/site-2`)
- In `CLAUDE.md`, replace the stale "Coach's Clipboard: three marker inks, flat colour" line so it points to `DESIGN.md` ("The Quiet Reference").
- The `code-review` skill asks for `docs/agents/issue-tracker.md`, which doesn't exist. Don't create it: run `code-review` on the Standards axis only and skip the Spec axis.

## Baseline (build from `perf/site` HEAD `fb71d94`, frozen data)
- Site 167.8 MB raw; deploy tar.gz 33.0 MB; 2,268 HTML pages (2,196 player pages).
- HTML 120.8 MB, of which player pages 111.9 MB (median 56 KB, largest 97.7 KB; `mike-james` 97.4 KB raw, 17.0 KB gz).
  - Shot Twin section: ~30 KB raw on `mike-james`, roughly 40 MB across all player pages.
  - Player pages: tooltip attributes 7.5 MB, inline SVG 18.0 MB.
- JSON 46.8 MB: `data/player` 21.8 MB (the `fields`/`logFields`/`gameFields` headers repeat per season), `data/shots` 12.8 MB, `data/team` 5.4 MB, `data/players` 2.9 MB, `data/cards` 1.9 MB, `data/leaders` 1.9 MB, `names.json` 0.1 MB.
- `data/shots/2025.json`: 51,700 shots, already columnar and bit-packed (`x`, `y`, `flags`, `player`, `team`, `opp`), 1.08 MB raw, ~280 KB gz.
- DOM: Players dashboard ~8,400 nodes (all 252 rows rendered without JS); teams index ~4,400 nodes (20 seasons' tables, 19 hidden).
- Re-measure all of this with the harness before starting; if a number differs, trust the harness and say so.

## Goals
Work through them in this order. Each has a pass condition; it's done only when the condition passes.

### G1. Deploy and transfer size
The gates are what the deploy and visitors pay for: tar.gz and gzipped bytes. Raw size is reported but is not a gate.

Pass condition:
- Deploy tar.gz ≥ 10% smaller than 33.0 MB (≤ 29.7 MB). Stretch: ≥ 15%.
- Gzipped HTML per page type does not grow; the player page type gets smaller.
- No meaningful loss: every page still shows the same data. No page, season or player may be dropped.
- Anything a visitor saw without JavaScript on first load is still there without JavaScript.

Shot Twin table (decided):
- The **default twin** (the one shown on load) stays in the HTML, readable without JS, but in compact markup (for example the butterfly bars as one SVG or fewer elements instead of one element per mark).
- The **other four twins** load with JavaScript when chosen, from compact data (a shared file if that is smaller overall than per-player data).
- Reserve the section's height so choosing a twin causes no layout shift.

Ideas to evaluate, don't assume:
- Deduplicate the per-player JSON headers.
- Precision and packing of the shot and card data.
- Emitting data files once instead of per page.
- Removing JSON endpoints nobody fetches.

### G2. DOM size of the data-heavy pages
Measure with Puppeteer `page.metrics().Nodes` after `networkidle0`, at **both 390 and 1280 px**.

Pass condition:
- Teams index ≤ 1,500 nodes at load. The hidden seasons may come from data on a season switch, because switching already needs JS; the default season stays complete without JS.
- Players dashboard: all 252 rows stay in the HTML (no-JS and crawlers see the whole table). Node count must not grow; reduce it where it costs no content (for example chart and cell markup). Report the result; it is not a gate.
- Switching season or filter gives the same content as today.

### G3. Shots: first tap on a phone (branch `ux/phone`)
Data policy (decided):
- After the `load` event, when the browser is idle (`requestIdleCallback`, with a timeout fallback), prefetch the current season's shots.
- Skip the prefetch when `navigator.connection.saveData` is true or `effectiveType` is `2g`/`slow-2g`; those visitors keep today's load-on-first-touch.
- A link with filters still loads at once.
- Update the pressed state on tap immediately, even while data is loading.
- A smaller payload per season is still welcome if the totals stay identical.

Pass condition, on an emulated mid-range phone (Puppeteer `KnownDevices["Moto G Power"]`, slow 4G, 4× CPU):
- Bytes up to the `load` event ≤ 70 KB. The idle prefetch after it is expected and reported.
- A tap on a facet bar changes the visible pressed state within 100 ms.
- After the prefetch has finished, the court and totals reflect the tap within 1 s of the tap.
- With `saveData`, the pressed state still changes within 100 ms; report tap-to-data.
- No layout shift.
- The same taps give the same totals, URL and filters as the current build.
- The prefetch must not raise TBT beyond the Lighthouse thresholds below.

### G4. Wide tables on phones (branch `ux/phone`)
Several tables scroll sideways at 390 px: the player seasons table and game log, team pages, the teams index, the Players dashboard, Compare.

Approach (decided): on phones, tables shed columns into **column groups** chosen with the existing segmented control (for example Scoring / Shooting / Rebounding), following `DESIGN.md`. The name column stays sticky.

Pass condition, at 360, 390 and 430 px wide:
- The page never scrolls sideways (`document.documentElement.scrollWidth <= innerWidth`).
- On every table, the name/identity column plus the column the page is about are visible without scrolling: the sorted column; PTS/PIR on logs; the ranked value. On **Compare**: the measure column plus the first player; the other players may scroll inside the box with an edge fade.
- Choosing a group that holds the sorted column, or sorting on a column, keeps that column in view.
- Any table that still scrolls shows an edge fade.
- Nothing hides data outright: every shed column is reachable through a group.
- New segmented labels follow the pinned-width rule (see Traps): measure them and check wrapping under the fallback font.
- Update the Sortable table section of `DESIGN.md` to describe the result.

### G5. Tooltips on touch screens (branch `ux/phone`)
`Base.astro` shows `data-tip` on hover and focus; touch has no good path. Some marks with tooltips are also links, for example the team-page rank-strip dots.

Pass condition, under touch emulation:
- Every `data-tip` mark that is not a link opens its tooltip with one tap.
- A second tap, a tap outside, or a scroll closes it.
- Links navigate on the first tap. Every link with a tooltip gets an `aria-label` carrying the tip's text. Where the tip holds information not visible elsewhere on the page, show it as visible text as well, following `DESIGN.md`, and record the change there.
- No double-tap zoom and no 300 ms delay.
- The desktop hover and focus behaviour stays pixel-identical.
- Tooltip bodies stay built through `web/src/lib/tip.ts` (`tip` tag and `dataTip`), and `tests/test_web_tips.py` keeps passing.

## Budgets across goals
- **Gz growth:** G4 and G5 may grow gzipped HTML by at most 1% or 300 B per page type, whichever is larger. G1–G2 may not grow it at all.
- **Lighthouse regressions** (median of 3, in each of the `simulate` and `devtools` modes) are: LCP +100 ms, TBT +30 ms, CLS +0.01, or score −1. Smaller differences count as noise.
- G3's idle prefetch is the one allowed byte increase in Lighthouse totals, on `/shots/` only.

## Rules
- **Visible changes:** allowed only on `ux/phone` for G3–G5, and they must follow `DESIGN.md`:
  - ink controls, Colour-Is-Data rule, Model Blue for the model only
  - hover gated behind `(hover: hover) and (pointer: fine)`
  - reduced motion respected
  - record what changed in `DESIGN.md`

  G1 and G2 must keep pixel and content parity.
- **Data sources:** build only from the local cache. No network scraping: esake.gr and the EuroLeague API are cache-only.
- **Protected files and codes:**
  - Never touch `predictions/*.csv` or `odds/*.csv`.
  - Team codes stay as source codes. Display renames live only in `DISPLAY_CODES` in `publish.py`.
- **Commits:**
  - Small themed commits, each with its measured gain in the message.
  - Before each commit run the full suite from `CLAUDE.md` ("Before commit") plus `npm run build` in `web/`.
  - End each message with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **Measurements:** don't run `astro dev` alongside builds or measurements.
- **Judgment calls:** work autonomously. Take the cautious option and list it in the report.

## Traps already found (don't repeat them)
- **Lighthouse modes:** simulated throttling misled once. A font preload "fixed" CLS in simulation but not under DevTools throttling, and it added 150 ms of LCP. Confirm every Lighthouse claim in both modes.
- **Already tried and reverted:**
  - Moving Base's inline script to an external file: simulated LCP got 150–300 ms worse on home/teams/leaders.
  - Relative hex path commands: anti-aliased pixel diffs.
  - A metric-matched fallback font: Windows-only and uneven.
  - Building tooltips on hover instead of in the HTML: kept as is by the owner (loses no-JS tooltips for ~1–6 KB gz per page).
- **Pinned widths:** the filter rows use widths pinned to Geist (`--seg-w`, `--seg-w-short`, `--select-w` in rem) so the font swap doesn't rewrap them. If you add or change a segmented label or a select's options, re-measure and re-check wrapping under the fallback font.
- **Timestamps:** `eurohoops publish` rewrites timestamps. Copy `web/src/data` to the scratchpad first, and build every variant from that frozen copy so parity diffs are meaningful.
- **Line endings:** Python on Windows writes CRLF in text mode. Use `newline=""` or write bytes when editing files.
- **Git Bash path conversion:** it rewrites arguments starting with `/`. Set `MSYS_NO_PATHCONV=1` for scripts that take URL paths.
- **Browser:** Chrome isn't installed. Use Edge at `C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe`. Install `lighthouse`, `puppeteer-core`, `pngjs` and `pixelmatch` into the scratchpad, not `web/`.

## Verification harness
Build it first in the scratchpad. It's the feedback loop for everything.
1. **Static server:** `node serve.mjs <dir> <port>` serving at `/EuroHoops-Analytics`, gzipped like GitHub Pages. Serve the frozen baseline build on one port and the working build on another.
2. **Sizes:** total and per-type bytes, tar.gz size, and per page type the raw/gz/br HTML, JS and CSS. Page types:
   - home `/`
   - `/players/`
   - `/players/mike-james/`
   - `/teams/`
   - `/teams/oly/`
   - `/compare/?p=mike-james,kendrick-nunn,sasha-vezenkov,vasilije-micic,nikola-mirotic`
   - `/leaders/`
   - `/shots/`
3. **Parity checker:** a Python script that parses every HTML page of baseline and new builds into a canonical form. It keeps tags, sorted attributes and text; parses `data-tip` bodies as HTML; drops `astro-*` classes and hashed `/_astro/` names; normalises path `d` to absolute points. JSON files must parse to equal values. Prove it goes red on a one-character tooltip change before trusting it.
   - For changes that move content behind JavaScript (the four non-default twins, the teams index's hidden seasons), compare the DOM after scripts run and after the relevant interaction, not the raw HTML.
   - For the compact default-twin markup, compare the rendered result (screenshots) plus the text content and values, since the markup itself changes.
4. **Screenshots:** full-page captures of the 8 page types plus empty Compare, at 390 and 1280 px, light and dark, reduced motion, after `document.fonts.ready`. Pixel-diff them with threshold 0. Take two baseline runs to confirm they're deterministic.
5. **Lighthouse mobile:** median of 3 runs, simulated and DevTools: LCP, CLS, TBT, bytes, requests, score.
6. **Heap and DOM nodes:** GC, then `JSHeapUsedSize` and `Nodes` after load, at 390 and 1280 px. Exercise every select and radio 3 times, then 10 more, and check for leaks.
7. **Font-swap check:** block `*.woff2` against loaded fonts at widths 320–1300 in steps of 5, and confirm filter rows (and any new segmented control) wrap identically.
8. **Touch checks:** Puppeteer device emulation, driving each G3–G5 condition with taps and asserting it: tooltip open/close, link navigation, `scrollWidth`, tap-to-feedback timing via `performance.now()` in the page, with and without `saveData`.

## Loop
For each goal, repeat until its pass condition holds:
1. Measure.
2. Form a hypothesis.
3. Change one thing.
4. Rebuild.
5. Re-run the harness.
6. Commit only if the goal's numbers improved and every other goal's numbers, the budgets and parity checks still hold. Otherwise revert and note why.

- If three consecutive attempts at a goal make no progress, stop that goal and report what blocks it.
- After G1–G2 on `perf/site-2`, and again after G3–G5 on `ux/phone`: run `code-review` (Standards axis) on the branch's range (`perf/site...perf/site-2`, then `perf/site-2...ux/phone`), then `simplify`, then the whole harness again. Repeat the goal loop if anything regressed.
- Report how many passes it took.

Skills to load when their step comes up: `diagnosing-bugs` (the loop), `impeccable` (optimize, adapt, harden), `mobile-native` (G3–G5), `codebase-design` (where data flows into pages, for G1/G2), `code-review`, `simplify`.

## Deliverable
- A before/after table:
  - tar.gz and raw site size, bytes per type, largest and median pages
  - DOM nodes for Players and the teams index at 390 and 1280 px
  - gz HTML/JS per page type
  - Lighthouse (both modes) per page type
  - heap
  - Shots bytes to `load`, prefetch bytes, tap-to-feedback and tap-to-data times (with and without `saveData`)
  - sideways-scroll results at 360/390/430
  - tooltip touch results
- Every commit with its measured gain, grouped by branch.
- Changes tried and reverted, with the reason.
- "Needs owner decision": anything that trades no-JS content, or any visible change beyond G3–G5.
- An entry appended to `memory.md`. Leave both branches unmerged.
