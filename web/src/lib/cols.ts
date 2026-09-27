// Column groups: on a phone a wide table sheds columns into groups, chosen with a segmented
// control. The control is radio inputs, so the CSS in stats.css (and on the Players page) shows
// the checked group without JavaScript; wider screens never see it and keep every column.

/** Each table's groups, in order (the first shows on load), and the control's width in Geist
 *  (pinned, as the filter rows' controls are, so the fallback font never rewraps it). */
export const GROUPS = {
  line: { labels: ["Scoring", "All-round", "Shooting"], w: "14.27rem" }, // season lines, rosters, game log
  bands: { labels: ["FG%", "Share"], w: "6.946rem" },
  clubs: { labels: ["Record", "Ratings", "Pace"], w: "11.859rem" },
  players: { labels: ["Scoring", "All-round", "Shots", "Efficiency"], w: "18.125rem" },
};

/** The control as HTML: one radio per group, the first checked. */
export const colsHtml = (name: string, { labels, w }: { labels: string[]; w: string }): string =>
  `<fieldset class="seg cols" aria-label="Columns" style="--seg-w:${w}">${labels
    .map((l, i) => `<label><input type="radio" name="${name}"${i ? "" : " checked"}>${l}</label>`)
    .join("")}</fieldset>`;
