// Column groups: on a phone a wide table sheds columns into groups, chosen with a segmented
// control. The control is radio inputs, so the CSS in stats.css (and on the Players page) shows
// the checked group without JavaScript; wider screens never see it and keep every column.

/** Each table's groups, in order (the first shows on load). The control's width is pinned in CSS
 *  to its width in Geist (--seg-w in stats.css and the Players page), as the filter rows' controls
 *  are, so the fallback font never rewraps it: re-measure there when a label changes. */
export const GROUPS = {
  line: ["Scoring", "All-round", "Shooting"], // season lines, rosters, game log: 14.27rem
  bands: ["FG%", "Share"], // 6.946rem
  clubs: ["Record", "Ratings", "Pace"], // 11.859rem
  players: ["Scoring", "All-round", "Shots", "Efficiency"], // 18.125rem
};

/** The control as HTML: one radio per group, the first checked. */
export const colsHtml = (name: string, labels: string[]): string =>
  `<fieldset class="seg cols" aria-label="Columns">${labels
    .map((l, i) => `<label><input type="radio" name="${name}"${i ? "" : " checked"}>${l}</label>`)
    .join("")}</fieldset>`;
