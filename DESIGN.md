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
- **Top bar:** 56px, sticky, paper at 88% with a saturating blur and a bottom hairline. Wordmark (a blue ring and "EuroHoops"), section links in soft ink that darken on hover, then the competition switch, theme toggle and GitHub icon on the right. Section links drop below 900px.
- **Segmented control:** a Sunk track with a hairline ring; the selected option is filled ink with paper text. Built on radio inputs so it works without JavaScript; labels shorten ("GBL") under 480px.

### Tooltip
Ink fill, paper text, 6px radius, soft drop shadow, max 260px. Opens after 120ms, then instantly for neighbouring marks while warm; enters from 2px below at 0.98 scale over 125ms. Every chart mark and forecast bar carries one, reachable by keyboard focus too.

### Forecast split bar (signature)
One row per upcoming game: tip-off time, away code, a 10px split bar, home code, expected margin. The bar is the away share in Baseline Gray meeting the home share in Model Blue at P(home), with a 2px gap, 4px outer ends and a thin even-odds tick standing proud at 50%. The favourite's percentage is ink and semibold; the other is muted. On load and on each competition switch every bar grows out of even odds to its call (320ms ease-out, 40ms stagger).

### Rating ladder
Ranked rows: rank, code, name, a bare rail with a dashed league-average line, a stem from average to the team's dot (blue above, red below), the rating, and the season change with a blue or red caret beside muted figures. No filled track behind the rail.

### Charts
Plot areas sit on Chart Surface with 1px solid gridlines in Rule; reference lines (the calibration diagonal) are dashed soft ink. Lines are 2px with round joins, dots carry a 2px paper ring, a legend appears for two or more series and end labels name the last values. Every chart has a text caption and a screen-reader table.

## Do's and Don'ts

### Do:
- **Do** keep every forecast beside the home-win baseline wherever a metric appears.
- **Do** carry outcome with an icon and a word as well as colour ("Hit", "Miss", "Not scored").
- **Do** animate only a change of data: bars growing from even odds, panels fading in after a competition switch (200ms). Everything collapses to instant under reduced motion.
- **Do** gate hover effects behind a fine-pointer media query.

### Don't:
- **Don't** add team logos, player photos or club colours; none are licensed.
- **Don't** present odds, stakes or "picks"; this is a model benchmark, not betting advice.
- **Don't** reintroduce a themed metaphor (clipboard, board, ticket, terminal) or a handwriting face.
- **Don't** put a filled background track behind a bar or rating rail.
- **Don't** colour text with a data colour, or use green for success; blue and red are the only outcome inks.
