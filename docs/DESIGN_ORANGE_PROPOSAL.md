# Proposal: orange as the data accent (Phase B3)

Status: **proposal, awaiting the owner's approval.** Nothing below is applied yet.

The Quiet Reference stays: neutral paper and night, one typeface (Geist), hairlines, colour only on data, text always in ink. What changes is *which* hue the data wears. Every number below comes from the dataviz skill's validator (`validate_palette.js`, OKLab ΔE ×100, Machado 2009 CVD simulation) and WCAG 2 contrast.

## 1. The orange

| | Light | Dark |
|---|---|---|
| Orange (`--data-1`, and `--brand-orange`) | **#d9590b** | **#e2691f** |
| Contrast on page (paper / night) | 3.70 : 1 | 5.77 : 1 |
| Contrast on chart surface | 3.80 : 1 | 5.20 : 1 |
| OKLCH lightness band (light 0.43–0.77, dark 0.48–0.67) | pass | pass |

- One orange: the logo arc and the data accent are the same token. The provisional dark value `#f0782a` fails the dark lightness band (L 0.70 > 0.67), so dark moves to `#e2691f`.
- Marks need 3 : 1 and pass in both themes. Body text needs 4.5 : 1 and light orange is 3.7 : 1, so **orange never colours text** (the Ink Text Rule already says so). Values beside an orange mark stay ink.

## 2. The opposite pole moves from red to blue

Today: blue is the model and "above average", red is "below average" and "Miss". With orange as the accent:

| Pair | Normal vision ΔE | Worst CVD ΔE | Verdict |
|---|---|---|---|
| orange #d9590b ↔ red #e34948 | 6.2 | 3.7 (deutan) | **fails**, even for full colour vision |
| orange #d9590b ↔ blue #2a78d6 | 33.0 | 26.5 (protan) | passes by a wide margin |
| orange #e2691f ↔ blue #3987e5 (dark) | 32.3 | 26.9 | passes |

So red cannot stay as the other pole. **Proposal: the current blue becomes the opposite pole** (`--data-neg` → `#2a78d6` light / `#3987e5` dark). Orange/blue is also the usual hot/cold pairing for shooting, so it reads naturally on the shot charts.

## 3. Which marks switch

| Mark | Today | Proposed |
|---|---|---|
| Forecast split bar, home share | blue | orange (away share stays Baseline Gray) |
| Elo / calibration lines and dots, scorecard | blue | orange |
| Players ranked dot chart, Compare leader dot, team rank-strip club dot | blue | orange |
| Shot twin: him / twin | blue / gray | orange / gray |
| Shots facet bars (kept) | blue | orange |
| Attempts chart: make / miss | blue dot / red ring | orange dot / blue ring (shape still carries it) |
| Outcome chips Hit / Miss (icon + word) | blue / red wash | orange / blue wash |
| Rating ladder above / below average, carets | blue / red | orange / blue |
| Shot chart FG% vs league (diverging, 5 steps) | red – gray – blue | **blue (cold) – gray – orange (hot)**; a team's Allowed chart keeps its flip, so orange always means good for that club |
| Frequency ramp `freq-1…5` | blue ramp | orange ramp (below) |
| Focus ring, text selection | blue | orange (3.7 : 1 clears the 3 : 1 non-text rule) |
| Jersey backs (Club-Colour Exception) | unchanged | unchanged |

Baseline Gray stays the neutral series. It "fails" the validator's chroma floor by design: it is meant to read as gray.

## 4. Frequency ramp (ordinal, one hue)

| Step | Light | Dark (brightens toward "most") |
|---|---|---|
| freq-1 | #eea06a | #8c3a0c |
| freq-2 | #e47a34 | #b44c10 |
| freq-3 | #d9590b | #e2691f |
| freq-4 | #a94307 | #f29a5e |
| freq-5 | #73300a | #f8c9a2 |

Validator: monotone lightness, every step ≥ 0.06 ΔL apart, hue spread 11° (light) / 17° (dark), palest step 2.07 : 1 (light) and 2.26 : 1 (dark) against the chart surface (floor 2 : 1). All pass.

## 5. Green and red: only "vs league" deltas

- Used only where a metric is compared with the league average (Phase C: team header, rank strips, player against-the-league graph). Never on a chart series, never on the shot maps.
- Green `#1f8a4c` / red `#d23c3c` (light), `#3fae6a` / `#e66767` (dark). Contrast on page: 4.15 / 4.47 (light), 6.87 / 5.98 (dark).
- Green vs red fails colour-blind separation (worst ΔE 5.3 light, 1.9 dark, deutan), as every green/red pair does. So **never colour alone**: every delta carries a sign and an arrow, as the owner already asked. The arrow follows the sign (▲ above the league, ▼ below) and the colour follows better or worse. A turnover rate above the league average is therefore a red ▲.
- Text rule: the arrow glyph wears green/red; the number stays ink. Light green and red sit just under 4.5 : 1, so colouring the number itself would fail small-text contrast. This is the one place colour touches text, and only as a glyph.

## 6. DESIGN.md changes if approved

- Colors: rename `model-blue` → `accent-orange`, `against-red` → `counter-blue`; add `delta-up` / `delta-down` and the orange `freq` ramp; light and dark values as above.
- Named rules: "Colour-Is-Data" names orange, blue and gray; new **Delta Rule** (green/red only for vs-league deltas, always sign + arrow, glyph only); the Validated Order Rule becomes orange, then gray, then blue.
- Don'ts: "blue and red are the only outcome inks" becomes "orange and blue"; "no green for success" stays, apart from the Delta Rule.
- Navigation: describe the 6.75 wordmark (baseline, orange arc, basket). Also fix the stale entries (Shots link, GitHub icon, GBL switch).
- Code: swap the values of `--data-1`, `--data-neg` and `--freq-*` in `global.css` and set `--brand-orange` equal to `--data-1`. Then sweep the 74 token uses in `web/src` for hard-coded blues or reds and anything named "blue" in labels or comments (for example "Model Blue" in captions).

## Open choices for the owner

1. Approve orange `#d9590b` / `#e2691f`, or ask for a warmer or deeper step.
2. Opposite pole = blue (recommended), or gray only (no second hue; simpler but loses the hot/cold shot map).
3. Deltas: arrow glyph coloured with the number in ink (recommended), or the whole figure coloured (needs darker greens/reds for 4.5 : 1).
