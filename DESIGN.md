---
name: EuroHoops Analytics
description: A quiet EuroLeague reference where the data is the only loud thing.
colors:
  paper: "#f9f9f7"
  chart-surface: "#fcfcfb"
  sunk: "#f1f1ee"
  ink: "#111110"
  ink-soft: "#52514e"
  muted: "#6d6b66"
  axis: "#898781"
  rule: "#e1e0d9"
  rule-strong: "#c3c2b7"
  model-blue: "#2a78d6"
  baseline-gray: "#8f8d86"
  against-red: "#e34948"
  night: "#0e0e0d"
  night-surface: "#1a1a19"
  night-sunk: "#1c1c1b"
  night-ink: "#f2f1ec"
  night-ink-soft: "#c3c2b7"
  night-muted: "#9c9a92"
  night-rule: "#2c2c2a"
  night-rule-strong: "#45453f"
  night-model-blue: "#3987e5"
  night-baseline-gray: "#6a6862"
  night-against-red: "#e66767"
  freq-1: "#86b6ef"
  freq-2: "#5598e7"
  freq-3: "#2a78d6"
  freq-4: "#1c5cab"
  freq-5: "#104281"
  night-freq-1: "#184f95"
  night-freq-2: "#256abf"
  night-freq-3: "#3987e5"
  night-freq-4: "#6da7ec"
  night-freq-5: "#9ec5f4"
typography:
  display:
    fontFamily: "Geist Variable, Segoe UI, system-ui, sans-serif"
    fontSize: "2.75rem"
    fontWeight: 620
    lineHeight: 1.04
    letterSpacing: "-0.035em"
  headline:
    fontFamily: "Geist Variable, Segoe UI, system-ui, sans-serif"
    fontSize: "1.5rem"
    fontWeight: 600
    lineHeight: 1.15
    letterSpacing: "-0.02em"
  title:
    fontFamily: "Geist Variable, Segoe UI, system-ui, sans-serif"
    fontSize: "1rem"
    fontWeight: 600
    lineHeight: 1.15
    letterSpacing: "-0.01em"
  body:
    fontFamily: "Geist Variable, Segoe UI, system-ui, sans-serif"
    fontSize: "1rem"
    fontWeight: 400
    lineHeight: 1.5
  data:
    fontFamily: "Geist Variable, Segoe UI, system-ui, sans-serif"
    fontSize: "0.875rem"
    fontWeight: 400
    lineHeight: 1.5
    fontFeature: "tnum"
  label:
    fontFamily: "Geist Variable, Segoe UI, system-ui, sans-serif"
    fontSize: "0.75rem"
    fontWeight: 500
    lineHeight: 1.4
rounded:
  bar-end: "4px"
  control-inner: "6px"
  control: "8px"
  pill: "999px"
spacing:
  gutter: "clamp(16px, 4vw, 40px)"
  section: "88px"
  section-phone: "64px"
  section-head: "28px"
  column-gap: "56px"
  container: "1240px"
components:
  button-primary:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.chart-surface}"
    rounded: "{rounded.control}"
    padding: "10px 16px"
  segment:
    backgroundColor: "{colors.sunk}"
    textColor: "{colors.ink-soft}"
    rounded: "{rounded.control}"
    padding: "2px"
  segment-selected:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.chart-surface}"
    rounded: "{rounded.control-inner}"
    padding: "4px 11px"
  icon-button:
    backgroundColor: "transparent"
    textColor: "{colors.ink-soft}"
    rounded: "{rounded.control}"
    size: "34px"
  icon-button-hover:
    backgroundColor: "{colors.sunk}"
    textColor: "{colors.ink}"
  outcome-hit:
    backgroundColor: "rgb(42 120 214 / 0.1)"
    textColor: "{colors.ink}"
    rounded: "{rounded.pill}"
    padding: "2px 9px 2px 7px"
  outcome-miss:
    backgroundColor: "rgb(227 73 72 / 0.1)"
    textColor: "{colors.ink}"
    rounded: "{rounded.pill}"
    padding: "2px 9px 2px 7px"
  tooltip:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.chart-surface}"
    rounded: "6px"
    padding: "7px 10px"
---

# Design System: EuroHoops Analytics

## Overview

**Creative North Star: "The Quiet Reference"**

EuroHoops reads like a well-kept reference work for EuroLeague numbers: neutral paper, one typeface, hairline rules, and colour that appears only where there is data. It sits alongside boxscorelab and databallr rather than trying to look like a basketball object; the owner rejected every themed metaphor (clipboard, station board, match tickets, terminals), so nothing on the page imitates a physical thing.

Density is that of a stats product, not a landing page: tables with many rows, tabular figures in every column, charts with precise hover tooltips. Controls are drawn in ink, not colour, so the one blue on screen always means "the model". Motion is scarce and only ever reports a change of data.

Light and dark are equal citizens. The page follows the visitor's system setting, and a remembered toggle overrides it both ways.

**Key Characteristics:**
- Neutral paper and near-black night, never cream, never pure white or black.
- One family, Geist, carries everything from the headline to axis ticks.
- Hairlines separate; nothing is a card.
- Colour lives in data marks only; text always wears ink tokens.
- Tabular figures wherever numbers stack.

## Colors

A near-neutral ground with three data inks: the model's blue, a neutral gray for baselines, and a red for the opposite pole.

### Primary
- **Model Blue** (light and dark values in the frontmatter): the model and everything it claims. Forecast shares, the Elo line, calibration dots, above-average ratings, the "Hit" outcome icon, focus rings, text selection.

### Secondary
- **Against Red**: the other pole of a diverging reading. Below-average ratings, falling ratings, the "Miss" outcome icon. Never used for text.

### Neutral
- **Paper / Night**: the page ground.
- **Chart Surface**: the plot area of every chart, one step off the page.
- **Sunk**: control tracks and hovered rows.
- **Ink, Ink Soft, Muted**: primary text, secondary text, and the smallest text that still clears 4.5:1 (captions, tick labels, table heads).
- **Axis**: axis and reference strokes only, never text.
- **Rule / Rule Strong**: row hairlines, and the heavier rule under a table head or section title.
- **Baseline Gray**: the neutral data series (the home-win baseline, the away share of a forecast).

### Named Rules
**The Colour-Is-Data Rule.** Blue, gray and red appear only on data marks (bars, lines, dots, stems, outcome icons and their washes). Controls, links and headings are ink.

**The Ink Text Rule.** Text never wears a data colour. A value beside a blue dot is set in ink; identity comes from the mark beside it.

**The Club-Colour Exception.** The one deliberate exception to Colour-Is-Data, approved by the owner for Leaders and Compare: jersey backs wear their club's two colours (body and trim). The colours live in `lib/jersey.ts` (`CLUB_COLOURS`, one curated pair per display code, a club's usual colours across 2007-2026), apart from the chart palette and never used as a token. They paint the shirt only: never a bar, dot, text, rule or background. The surname and number flip between the two page inks (#111110 or #fcfcfb), whichever contrasts more with the body; every pair clears 4.5:1 for that ink. A Rule Strong hairline outlines every shirt so a white or black shirt holds its edge on paper and night alike. No crests, sponsor marks, stripes or copies of real kits: one generic sleeveless cut for every club. Shirt red never sits beside a below-average red mark: Leaders and Compare carry no red data marks.

**The Validated Order Rule.** Chart colours come from the validated order (blue, then gray, then red) and are re-checked with the dataviz validator against both surfaces before any new series joins them. A fourth series folds into "Other" or small multiples.

## Typography

**Body Font:** Geist Variable (with Segoe UI, system-ui)

**Character:** one neutral grotesk for every role, ranked by weight and size steps of about 1.2, never by a second family.

### Hierarchy
- **Display** (620, 2.75rem, 1.04, 2.25rem under 980px): the single page headline.
- **Headline** (600, 1.5rem, 1.15): section titles.
- **Title** (600, 1rem, 1.15): panel titles inside a section, sitting on a strong rule.
- **Body** (400, 1rem, 1.5): prose, capped at 38 to 64ch.
- **Data** (400, 0.875rem, tabular figures): every table cell and chart label.
- **Label** (500, 0.75rem, sentence case): table heads, legends, captions.

### Named Rules
**The Tabular Rule.** Every column of numbers, axis tick and tooltip uses tabular figures. Standalone large numbers may stay proportional.

**The No-Eyebrow Rule.** A heading carries its own weight. No small uppercase label above it, ever.

## Layout

A single centred column (max 1240px) with a fluid gutter. Sections stack with 88px of space above each title (64px on phones), a title plus a one-sentence explainer directly below it, then 28px to the content. Paired panels (scorecard, calibration, ratings) sit in two columns with a 56px gap and collapse to one column below 820 to 900px. The first viewport is a split: headline, explainer and log link on the left, tonight's forecasts on the wider right; it stacks copy-first under 980px.

Tables scroll inside their own box when they cannot fit, and on phones they shed columns (full names, date) before they scroll, so the verdict column is always in view. Tonight's rows use container queries: full names only when the panel is wider than 820px, a two-row layout plus a margin row under 520px.

## Elevation & Depth

Flat. Depth comes from tonal steps (paper, chart surface, sunk) and hairline rules, not shadows. The only shadows in the system belong to things that float above the page: the tooltip, and the translucent sticky top bar with its backdrop blur.

### Named Rules
**The Flat Page Rule.** Nothing that sits in the page flow casts a shadow or is boxed as a card.

## Shapes

Gentle, small radii. Controls and buttons are 8px, their inner selected segment 6px, data bars have 4px rounded ends and stay square where two segments meet, outcome chips are full pills, dots are circles. Tables and sections are square and open, bounded only by rules.

## Components

### Buttons
- **Shape:** gently rounded (8px).
- **Primary:** ink fill with paper text, 10px by 16px, 0.9375rem medium. One per page ("Read the log").
- **Hover / Active:** the trailing arrow icon nudges up and right; press scales to 0.97 over 140ms.
- **Icon buttons:** 34px square, transparent, soft ink; hover fills with Sunk and darkens to ink; press scales to 0.94.

### Chips
- **Outcome chips:** full pills with ink text, a Phosphor check or cross icon, and a 10% wash of the icon's colour: blue for a hit, red for a miss. "Not scored" is a muted chip on Sunk with a hairline ring and links to the footnote.

### Navigation
- **Top bar:** 56px, sticky, paper at 88% with a saturating blur and a bottom hairline. Wordmark (a blue ring and "EuroHoops"), site links (Players, Leaders, Compare, Teams, Shots, Forecasts) in soft ink with the current page in ink at 550 weight, then the competition switch (forecast pages only), theme toggle and GitHub icon on the right. Under 560px the wordmark keeps only its ring and the link row scrolls sideways inside the bar (no scrollbar, a 24px fade at the end that still has links, 14px at the start once scrolled); on load it lands on a word boundary with the current page clear of the fade, or at the row's end when the bar is too narrow for both fades. The page itself never scrolls sideways.
- **Segmented control:** a Sunk track with a hairline ring; the selected option is filled ink with paper text. Built on radio inputs so it works without JavaScript; labels shorten ("GBL") under 480px.

### Tooltip
Ink fill, paper text, 6px radius, soft drop shadow, max 260px. Opens after 120ms, then instantly for neighbouring marks while warm; enters from 2px below at 0.98 scale over 125ms. Every chart mark and forecast bar carries one, reachable by keyboard focus too.

### Forecast split bar (signature)
One row per upcoming game: tip-off time, away code, a 10px split bar, home code, expected margin. The bar is the away share in Baseline Gray meeting the home share in Model Blue at P(home), with a 2px gap, 4px outer ends and a thin even-odds tick standing proud at 50%. Both shares keep their full colour in every row, as the legend shows; the favourite's percentage is ink and semibold, the other muted. On load and on each competition switch every bar grows out of even odds to its call (320ms ease-out, 40ms stagger).

### Filters
One row above the content, separated from it by a hairline: small muted labels over the controls. Selects and the search field are 32px, 8px radius, Chart Surface fill with a Rule Strong inner ring that darkens to soft ink on hover, a Phosphor caret or magnifier in soft ink. Choices with two to four options are segmented controls. The row wraps; nothing hides.

### Sortable table
Dense rows (0.8125rem, tabular) with hairline rules and a sticky header on the page colour. Every numeric head is a button with a tooltip naming the measure; the sorted column's head carries a 2px ink underline and its cells go ink and semibold. Values that fail a qualifying floor stay visible but muted, with the floor in a tooltip. The name column sticks on horizontal scroll (8.5rem on phones, the team code on a second line). On the Players dashboard the table scrolls inside a box beside the chart, with a thin Rule Strong scrollbar and a 48px fade at its foot that lifts at the last row; under 1080px it runs in the page, first 50 rows and then a full-width "Show all N players" button (Sunk fill, Rule Strong ring).

### Ranked dot chart (Players signature)
The top 25 on the sorted measure as a Cleveland dot plot: rank, name (a 10.5rem track, 10.25rem on phones, so full names fit), a dotted guide per row, a blue 10px dot with a paper ring, the value in ink. Vertical gridlines at nice ticks and a dashed Baseline Gray line for the average of the players ranked. Rows are fixed slots: a change of measure, window or rate slides each dot to its new position (200ms ease-out) while names and values swap; nothing animates on load.

### Shot chart
Half court drawn in Axis hairlines on Chart Surface, basket at the top. Pointy-top hexagons of 0.5 m: size by how often the player shot from the cell, colour by his FG% there against the league's from the same cell, shrunk toward the league on small samples, in five diverging steps (red, light red, Baseline Gray, light blue, blue; the light steps mix the hue half into the surface). A key for colour and size sits under the chart, beside a distance-band table with share and FG% against the league and a caret for differences of 2.5 points or more.

Marks are clipped to the floor inside the sideline and baseline hairlines, and the backboard and rim are drawn above them so the basket stays legible under the busiest cells. On phones a distance table's range ("under 1.5 m") drops under the band name so Diff stays in view.

Shared by the player, team and Shots pages (`lib/court.ts`, `styles/stats.css`), in three layers:
- **FG% against the league** (above). On a team's **Allowed** chart the scale flips, so blue always means good for that club: blue where opponents shoot worse than the league. Its key is drawn in the flipped order.
- **Frequency** (Shots explorer): the same hexes, coloured by how often the cell was used in five steps of one blue ramp (tokens `freq-1` to `freq-5`, validated as an ordinal ramp in both themes; on night the ramp brightens toward "most").
- **Attempts**: one mark per shot, a filled Model Blue dot for a make and a hollow Against Red ring for a miss, so shape carries the outcome too.

Seasons before 2011-12 carry a one-line caveat under the chart; they are never hidden.

### Rank strips (Team signature)
One row per rating or four factor, in two columns (Offense, Defense) under a Title on a strong rule: the measure's name (tooltip gives the formula), the club's value in ink semibold, its rank ("3rd of 18", counted from the best, so lower wins for turnover rate and the defensive measures), then a bare rail with every other club as an 8px Baseline Gray dot with a page ring, a dashed Baseline Gray league-average tick, and this club as a 12px Model Blue dot with a 2px page ring. Each gray dot links to that club's page for the same season. On a season switch the blue dot slides to its new place (200ms ease-out) while the gray field redraws. Under 560px the rail drops below the name, value and rank.

### Facet bars (Shots signature)
The explorer's filters are bar charts. Each facet (Quarter, Distance, Play) is a head on a strong rule and one button row per option: label, a 10px bar with a 4px rounded end whose width is the option's attempts, the attempts, FG% and the difference from the comparison. A kept option's bar is Model Blue, a filtered-out one Baseline Gray; a chosen option's label goes ink semibold. Each facet counts the shots the other facets keep, so it shows what a click would give; bars resize in place (200ms). The comparison is the league with the same filters, or every league shot when the subject is the league. The play-by-play marks fastbreaks and second chances on made shots only (from 2015-16), so the Play facet counts makes and its share; it filters only in the Frequency and Attempts views and is disabled with a one-line reason otherwise. Above the facets, a three-figure totals line (attempts, FG%, points per shot, each over its comparison in muted text; for the unfiltered league, which is its own comparison, only "every located shot" under attempts); on phones the view switch reads Freq. / FG% / Att.; the method note sits under the court, beside the taller facet column; below them, an ink Clear filters button that goes quiet (Sunk, muted) when nothing is filtered.

### Rating ladder
Ranked rows: rank, code, name, a bare rail with a dashed league-average line, a stem from average to the team's dot (blue above, red below, Baseline Gray exactly on 1500), the rating, and the season change with a blue or red caret beside muted figures. No filled track behind the rail.

### Jersey back (Leaders and Compare)
A fantasy-style shirt seen from behind, drawn in SVG (`jerseySvg` in `lib/jersey.ts`): one sleeveless cut in a 100 × 112 box, the club's body colour, trim bands at the neck, armholes and hem, the surname across the shoulders (squeezed to fit when longer than nine letters) and the number large below it, both Geist in the flipped ink. A season shows the number and club he ended that season with; a career shows the club and number with the most games. Sizes: 72px on the court, 64px heading a Compare column, 30px on a bench row; under 48px the surname is dropped (it would render near 5px) and the shirt carries the number only. Decorative (`aria-hidden`) wherever the name sits beside it in text. Never a photo or a likeness.

### Court lineup (Leaders signature)
The shared court (Axis hairlines on Chart Surface), closed here at the halfway line with the centre-circle half (`court(…, closed)`), with five shirts placed over it so no line touches text (wings beyond the arc on the 45° lines): the top five by rank, first at the point, second and third on the wings, fourth and fifth on the blocks. For a club's season it is the roster's five who started most, ordered by their share of assists against rebounds (the data has no positions), and the caption says so. Each slot is shirt, rank and name, club and season, the ranked value large in ink semibold with its head, then the card line (PTS REB AST TS% PIR, the sorted one in ink semibold, the rest soft ink). The text block wears the Chart Surface colour so court lines never run through it. Beside the court (below it under 980px) the bench, under a head that matches the court's (title, one-line note, strong rule): ranks 6 to 20, or the rest of the roster, as rows of rank, 30px shirt, name over club · season · games · minutes, and the value; values under a floor are muted with the reason in a tooltip. Under 620px of floor the shirts keep only the value and the five full lines list below the court. On a change of scope, measure, rate or club the five shirts fade and rise 6px into place (200ms, 30ms steps from the point outward) and the bench rows slide to their new order (200ms); a page opened from a link paints without motion.

### Head-to-head grid (Compare signature)
Up to five players, each a season (a select in the column head lists every season with its club) or a career. A table: the measure column sticks on the left; each player owns an equal column headed by a 64px shirt, name (links to the player page), club, the season select, and a quiet remove button. Under the heads a Rows led tally ("of 22"), then groups (Playing time, Scoring, Playmaking, Rebounding, Defence, Overall) on strong rules. Each cell is two fixed slots, so figures line up down a column: the value in ink, right-aligned, and its place among the compared in muted small text; the row's leader gets an 8px Model Blue dot hugging its value and semibold ink, and so does the Rows led leader. The remove button sits beside its shirt. Fewer turnovers lead their row; a percentage under its attempt floor is muted and never leads. The only colour on the grid is that dot. On a change the dots pop into their new cells (160ms). Below 5 × 9rem plus the measure column the table scrolls inside its box. Empty state: the search, plus two ready-made fives (career PIR leaders, career scorers).

### Mirror staff (Shot twins, player page)
The player's last 5, 10 or 20 games or his latest full season (a segmented control) against the five closest EuroLeague player-seasons since 2007-08 (`components/ShotTwin.astro`, maths in `lib/twin.ts`, data from the exporter's `twins.json`). The five twins sit in one row of choices under a hairline: rank, name, season · club; twins are always other players, never his own seasons, the match score large in ink; the chosen one takes a 2px ink rule on top and semibold ink, and under 820px the row becomes a list with the score on the right. Below, a table read as a butterfly: a centre staff of labels, him on the left in Model Blue, the twin on the right in Baseline Gray, each column headed by the name and a 10px swatch. Three groups on strong rules: Where they shoot (14 zones, rim to deep three, bars on one shared scale per page), How well (FG% per band with the points above the league and a blue or red caret from 2.5 points and 10 attempts; "few shots" under that), Style (3PA and FT rates as bars). Bars are 10px with a 4px outer end, square at the staff, no track; axis hairlines mark the staff edges and the share figure rides at each bar's end ("<1%" for a trace). Symmetry is the law: on phones the staff narrows to short labels ("Mid L") and the columns thin but never merge. Choosing a twin slides the right-hand marks to their new lengths, a window both sides (transform, 200ms ease-out); nothing animates on load. A method note under it gives the weights, the pool size and floor, and what 100 and 37 mean. Pre-2011-12 twins carry the approximate-locations line.

### Charts
Plot areas sit on Chart Surface with 1px solid gridlines in Rule; reference lines (the calibration diagonal) are dashed soft ink. Lines are 2px with round joins, dots carry a 2px paper ring, a legend appears for two or more series and end labels name the last values. Every chart has a text caption and a screen-reader table.

## Do's and Don'ts

### Do:
- **Do** keep every forecast beside the home-win baseline wherever a metric appears.
- **Do** carry outcome with an icon and a word as well as colour ("Hit", "Miss", "Not scored").
- **Do** animate only a change of data: bars growing from even odds, panels fading in after a competition switch (200ms), ranked dots sliding when the measure changes. Everything collapses to instant under reduced motion.
- **Do** gate hover effects behind a fine-pointer media query.

### Don't:
- **Don't** add team logos, crests or player photos; none are licensed. Club colours appear only on jersey backs (the Club-Colour Exception), never on data or text.
- **Don't** present odds, stakes or "picks"; this is a model benchmark, not betting advice.
- **Don't** reintroduce a themed metaphor (clipboard, board, ticket, terminal) or a handwriting face.
- **Don't** put a filled background track behind a bar or rating rail.
- **Don't** colour text with a data colour, or use green for success; blue and red are the only outcome inks.
