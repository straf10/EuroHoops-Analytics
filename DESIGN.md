---
name: EuroHoops Analytics
description: Pre-registered basketball forecasts, drawn up on a coach's clipboard before tip-off.
colors:
  aluminium-frame: "#b7bec3"
  melamine-board: "#f4f6f5"
  board-sunk: "#eaeeed"
  court-line: "#d3dad9"
  marker-black: "#1c2126"
  marker-black-soft: "#4a535b"
  marker-blue: "#1f4fb8"
  marker-red: "#c4362a"
  glass-board: "#121518"
  glass-frame: "#33393e"
  glass-marker-blue: "#7aa2ff"
  glass-marker-red: "#ff6f61"
typography:
  display:
    fontFamily: "Sofia Sans Extra Condensed Variable"
    fontSize: "clamp(3.2rem, 6.4vw, 5.6rem)"
    fontWeight: 800
    lineHeight: 0.9
    letterSpacing: "-0.015em"
  headline:
    fontFamily: "Sofia Sans Extra Condensed Variable"
    fontSize: "clamp(2.2rem, 4.4vw, 3.4rem)"
    fontWeight: 800
    lineHeight: 0.95
  title:
    fontFamily: "Sofia Sans Extra Condensed Variable"
    fontSize: "clamp(1.5rem, 2.4vw, 1.9rem)"
    fontWeight: 800
  body:
    fontFamily: "Sofia Sans Variable"
    fontSize: "1rem"
    fontWeight: 400
    lineHeight: 1.5
    fontFeature: "tnum"
  label:
    fontFamily: "Sofia Sans Variable"
    fontSize: "0.78rem"
    fontWeight: 600
    letterSpacing: "0.06em"
  annotation:
    fontFamily: "Caveat Brush"
    fontSize: "1.15rem"
rounded:
  board: "14px"
  pill: "999px"
  puck: "50%"
spacing:
  gutter: "clamp(16px, 3.2vw, 44px)"
  band: "clamp(44px, 6vw, 84px)"
components:
  button-primary:
    backgroundColor: "{colors.marker-black}"
    textColor: "{colors.melamine-board}"
    rounded: "{rounded.pill}"
    padding: "13px 22px"
  button-primary-hover:
    backgroundColor: "{colors.marker-blue}"
    textColor: "{colors.melamine-board}"
  switch-active:
    backgroundColor: "{colors.marker-blue}"
    textColor: "{colors.melamine-board}"
    rounded: "{rounded.pill}"
  puck:
    backgroundColor: "{colors.marker-blue}"
    textColor: "{colors.melamine-board}"
    rounded: "{rounded.puck}"
    size: "40px"
---

# Design System: EuroHoops Analytics

## Overview

**Creative North Star: "The Coach's Clipboard"**

Every forecast is drawn up on the board before tip-off and marked up after the whistle. The page is a white melamine tactics board in an aluminium frame (a black glass board in dark mode), carrying printed court lines and three marker inks. It refuses the dark KPI-tile stats dashboard: numbers sit on the board as a coach would write them, not in tiles.

**Key Characteristics:**
- One board, one frame, three inks.
- Court geometry is the chart: the half-court line is 50%, the baskets are certainty.
- Handwriting annotates; it never carries a number that has to be compared.

## Colors

### Primary
- **Marker Blue** (`#1f4fb8`, dark `#7aa2ff`): the model's call. Pucks, call annotations, the better number in a pair, the active switch, focus rings.

### Secondary
- **Marker Red** (`#c4362a`, dark `#ff6f61`): the result mark-up only. Rings around hits, strikes through misses, falling ratings, the not-scored asterisk.

### Neutral
- **Melamine Board** (`#f4f6f5`) / **Glass Board** (`#121518`): the page field, cool and never cream.
- **Aluminium Frame** (`#b7bec3`) / **Glass Frame** (`#33393e`): body ground and the marker tray.
- **Marker Black** (`#1c2126`) and its soft tone (`#4a535b`): structure, numbers, secondary text.
- **Court Line** (`#d3dad9`): printed court markings and row rules.

### Named Rules
**The Three Inks Rule.** Black, blue and red are the only inks. Blue is the model, red is what happened. No fourth accent, ever.

**The Flat Ink Rule.** Inks and frame are flat colour. No gradients imitating brushed metal, glossy magnets or marker sheen.

## Typography

**Display Font:** Sofia Sans Extra Condensed (sports-sheet compression; covers Greek)
**Body Font:** Sofia Sans with tabular numerals (covers Greek team names)
**Annotation Font:** Caveat Brush (marker hand, Latin only: codes, percentages, short notes)

### Hierarchy
- **Display** (800, clamp 3.2 to 5.6rem, 0.9, uppercase): the one hero headline.
- **Headline** (800, clamp 2.2 to 3.4rem, uppercase): section titles.
- **Title** (800, clamp 1.5 to 1.9rem): sub-blocks such as scorecard columns.
- **Body** (400, 1rem, 1.5): prose, capped near 62ch.
- **Label** (600, 0.78rem, 0.06em tracking, uppercase): table headers and axis labels only, never above a heading.

### Named Rules
**The Marker Hand Rule.** Caveat Brush writes annotations: the round title, puck percentages, the call being marked. Every number a reader compares is set in Sofia Sans.

## Layout

A single board, max 1360px, inside the frame. Bands stack with `clamp(44px, 6vw, 84px)` vertical padding and a 1px court-line rule between them; the scorecard band sits on the sunk board tone. The hero is 2:1 court to copy and stacks copy-first under 980px. Two-column blocks (scorecard, ratings) collapse to one under 760px. Tables scroll inside their own box on phones; the page never scrolls sideways.

## Elevation & Depth

Flat board, physical objects. Only magnets carry shadow (`0 2px 3px` plus a soft `0 6px 14px -6px`, tinted from the shadow token). The board itself sits in the frame with an inset shadow.

## Shapes

The board has a 14px radius. Controls are pills. Pucks and rating magnets are circles. Tables and bands are square.

## Components

### Buttons
Primary is a marker-black pill with board-coloured text; hover turns it marker blue and nudges the arrow icon up and right; press scales to 0.97. One primary action per page ("Read the log").

### Navigation
Header rail: wordmark with a blue magnet dot, the competition switch (a pill segmented control built on radio inputs, so it works without JavaScript), section links, GitHub icon. Links collapse to the icon under 980px.

### Court board (signature)
Each upcoming game is a lane across a printed court. The puck sits at P(home), labelled with the favourite's code and its chance in marker blue. On load each puck slides from centre court to its call (900ms, `cubic-bezier(0.23, 1, 0.32, 1)`, 60ms stagger). Hover or focus draws the call as a blue marker stroke from centre court to the puck. Empty state: a marker note naming the next tip-off.

### Result mark-up
A red hand-drawn ring around a correct call, a red strike through a wrong one, always with the word "hit" or "miss" beside it. Marks draw on as they scroll into view.

### Rating ladder
Ranked rows with code, name, a rail with a black magnet at the team's rating, the rating, and the season change (blue up, red down, with a caret icon).

## Do's and Don'ts

### Do:
- **Do** show Elo beside the home-win baseline every time a metric appears.
- **Do** keep scored and not-scored rows visibly distinct.
- **Do** honour reduced motion: pucks and marks render in place.

### Don't:
- **Don't** add team logos, photos or club colours. None are licensed, and the three inks carry identity.
- **Don't** present odds, stakes or "picks": this is a model benchmark, not betting advice.
- **Don't** put a label or eyebrow above a heading.
- **Don't** use a filled track behind a bar or rating rail.
