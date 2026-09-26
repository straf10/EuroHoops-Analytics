// Half-court drawing and the three shot-chart layers (FG% against the league, frequency, single
// attempts). Pure strings, shared by the build and the in-page updates of the player, team and
// Shots pages. Court metres, basket at the origin; drawn with the baseline on top.

export const esc = (s: string) =>
  s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c] ?? c);

/** [q, r, attempts, makes, league FG% in the cell x10 (-1: no league shots there)] */
export type HexCell = [q: number, r: number, att: number, made: number, league: number];

const SQRT3 = Math.sqrt(3);
const BASELINE = 1.575;
export const DEPTH = 9.3; // metres from the basket shown; deeper heaves are left off the chart
const toY = (y: number) => y + BASELINE;

const VIEWBOX = `-7.7 -0.2 15.4 ${(DEPTH + BASELINE + 0.4).toFixed(2)}`;

function courtLines(): string {
  const arcY = Math.sqrt(6.75 ** 2 - 6.6 ** 2); // where the corner lines meet the arc
  const ft = 5.8 - BASELINE;
  return [
    `<path class="line" d="M-7.5 ${toY(DEPTH)}V0H7.5V${toY(DEPTH)}"/>`,
    `<path class="line" d="M-2.45 0V${toY(ft)}H2.45V0"/>`,
    `<path class="line" d="M-1.8 ${toY(ft)}A1.8 1.8 0 0 0 1.8 ${toY(ft)}"/>`,
    `<path class="line" d="M-6.6 0V${toY(arcY)}A6.75 6.75 0 0 0 6.6 ${toY(arcY)}V0"/>`,
    `<path class="line" d="M-1.25 ${toY(0)}A1.25 1.25 0 0 0 1.25 ${toY(0)}"/>`,
    `<path class="line" d="M-0.9 ${toY(-0.375)}H0.9"/>`,
    `<circle class="rim" cx="0" cy="${toY(0)}" r="0.225"/>`,
  ].join("");
}

export const court = (layer: string, label: string) =>
  `<svg class="court" viewBox="${VIEWBOX}" role="img" aria-label="${esc(label)}"><rect class="floor" x="-7.5" y="0" width="15" height="${toY(DEPTH)}"/>${courtLines()}${layer}</svg>`;

function hexPath(cx: number, cy: number, r: number): string {
  const pts: string[] = [];
  for (let i = 0; i < 6; i++) {
    const a = ((60 * i - 30) * Math.PI) / 180;
    pts.push(`${(cx + r * Math.cos(a)).toFixed(3)} ${(cy + r * Math.sin(a)).toFixed(3)}`);
  }
  return `M${pts.join("L")}Z`;
}

/** Axial (q, r) of the pointy-top hexagon holding a point (cube rounding, as the exporter). */
export function hexOf(x: number, y: number, radius: number): [number, number] {
  const fq = ((SQRT3 / 3) * x - y / 3) / radius;
  const fr = ((2 / 3) * y) / radius;
  const fs = -fq - fr;
  let q = Math.round(fq);
  let r = Math.round(fr);
  const s = Math.round(fs);
  const dq = Math.abs(q - fq);
  const dr = Math.abs(r - fr);
  const ds = Math.abs(s - fs);
  if (dq > dr && dq > ds) q = -r - s;
  else if (dr > ds) r = -q - s;
  return [q, r];
}

/** Five steps of "FG% here against the league here", shrunk toward the league on small samples. */
export function tone(att: number, made: number, leaguePermille: number): number {
  if (leaguePermille < 0) return 0;
  const league = leaguePermille / 1000;
  const k = 8; // prior weight in attempts
  const diff = 100 * ((made + k * league) / (att + k) - league);
  if (diff <= -7) return -2;
  if (diff <= -2.5) return -1;
  if (diff < 2.5) return 0;
  return diff < 7 ? 1 : 2;
}

export interface HexOptions {
  label: string;
  /** Colour by the step `color(cell)` returns (a class name) instead of FG% against the league. */
  color?: (cell: HexCell, total: number) => string;
  /** Flip the diverging scale: blue where the shooters do worse than the league (defence). */
  invert?: boolean;
  /** Tooltip body for one cell. */
  tip?: (cell: HexCell, total: number) => string;
}

const defaultTip = ([, , att, made, lg]: HexCell) => {
  const league = lg < 0 ? "no league shots here" : `league ${(lg / 10).toFixed(1)}%`;
  return `<b>${made} of ${att}</b> (${((100 * made) / att).toFixed(0)}%)<br>${league}`;
};

/** Hexes sized by how often the cell was used; coloured by FG% against the league unless told. */
export function hexChart(hex: HexCell[], radius: number, o: HexOptions): string {
  const shown = hex.filter(([, r]) => radius * 1.5 * r <= DEPTH);
  const total = hex.reduce((a, c) => a + c[2], 0);
  const most = Math.max(1, ...shown.map((c) => c[2]));
  const full = Math.max(1, most * 0.6); // the busiest cells all draw at full size
  const tip = o.tip ?? defaultTip;
  const cells = shown
    .map((cell) => {
      const [q, r, att, made, lg] = cell;
      const cx = radius * SQRT3 * (q + r / 2);
      const cy = toY(radius * 1.5 * r);
      const size = radius * 0.94 * (0.32 + 0.68 * Math.sqrt(Math.min(1, att / full)));
      const cls = o.color ? o.color(cell, total) : `t${(o.invert ? -1 : 1) * tone(att, made, lg) || 0}`;
      return `<path class="${cls}" d="${hexPath(cx, cy, size)}" data-tip="${esc(tip(cell, total))}"/>`;
    })
    .join("");
  return court(`<g class="cells">${cells}</g>`, o.label);
}

/** One mark per attempt: filled dot for a make, hollow ring for a miss. */
export function dotChart(points: { x: number; y: number; made: boolean }[], label: string): string {
  const made: string[] = [];
  const miss: string[] = [];
  const r = 0.13;
  for (const p of points) {
    if (p.y > DEPTH || Math.abs(p.x) > 7.5) continue;
    const d = `M${(p.x - r).toFixed(2)} ${toY(p.y).toFixed(2)}a${r} ${r} 0 1 0 ${2 * r} 0a${r} ${r} 0 1 0 ${-2 * r} 0`;
    (p.made ? made : miss).push(d);
  }
  return court(
    `<path class="miss" d="${miss.join("")}"/><path class="made" d="${made.join("")}"/>`,
    label,
  );
}
