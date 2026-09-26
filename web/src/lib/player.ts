// Player page renderers: shot chart, distance bands against the league, game log, season lines.
// Pure (types only from ./stats), so the build and the in-page season switch share them.

import type { Bands, PlayerSeason } from "./stats";

const esc = (s: string) =>
  s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c] ?? c);

export const BAND_LABELS: Record<string, [string, string]> = {
  rim: ["At the rim", "under 1.5 m"],
  short: ["Short", "1.5 to 3 m"],
  mid: ["Mid-range", "3 to 5 m"],
  long2: ["Long two", "5 m and out"],
  three: ["Three", "under 8 m"],
  deep3: ["Deep three", "8 m and out"],
};

export type Totals = Record<string, number>;
export const totalsOf = (fields: string[], values: number[]): Totals =>
  Object.fromEntries(fields.map((f, i) => [f, values[i]]));

const one = (x: number) => x.toFixed(1);
export const pct = (num: number, den: number) => (den > 0 ? `${((100 * num) / den).toFixed(1)}` : "–");
export const perGame = (t: Totals, x: number) => (t.gp ? one(x / t.gp) : "–");
export const ts = (t: Totals) => pct(t.pts, 2 * (t.fg2a + t.fg3a + 0.44 * t.fta));

/** Season-by-season rows: per game, with the career line computed from summed totals. */
export const SEASON_COLS: { head: string; about: string; cell: (t: Totals) => string }[] = [
  { head: "GP", about: "Games played", cell: (t) => String(t.gp) },
  { head: "GS", about: "Games started", cell: (t) => String(t.gs) },
  { head: "MIN", about: "Minutes per game", cell: (t) => perGame(t, t.sec / 60) },
  { head: "PTS", about: "Points per game", cell: (t) => perGame(t, t.pts) },
  { head: "REB", about: "Rebounds per game", cell: (t) => perGame(t, t.oreb + t.dreb) },
  { head: "AST", about: "Assists per game", cell: (t) => perGame(t, t.ast) },
  { head: "STL", about: "Steals per game", cell: (t) => perGame(t, t.stl) },
  { head: "BLK", about: "Blocks per game", cell: (t) => perGame(t, t.blk) },
  { head: "TOV", about: "Turnovers per game", cell: (t) => perGame(t, t.tov) },
  { head: "PIR", about: "Performance Index Rating per game", cell: (t) => perGame(t, t.pir) },
  { head: "FG%", about: "Field goals made per attempt", cell: (t) => pct(t.fg2m + t.fg3m, t.fg2a + t.fg3a) },
  { head: "3P%", about: "Three-pointers made per attempt", cell: (t) => pct(t.fg3m, t.fg3a) },
  { head: "FT%", about: "Free throws made per attempt", cell: (t) => pct(t.ftm, t.fta) },
  { head: "TS%", about: "True shooting: PTS / (2 × (FGA + 0.44 × FTA))", cell: ts },
];

// ---- Shot chart ----------------------------------------------------------------------------
// Court metres, basket at the origin; drawn with the baseline on top (y grows down the page).

const SQRT3 = Math.sqrt(3);
const BASELINE = 1.575;
const DEPTH = 9.3; // metres from the basket shown; deeper heaves are left off the chart
const toY = (y: number) => y + BASELINE;

export const COURT_VIEWBOX = `-7.7 -0.2 15.4 ${(DEPTH + BASELINE + 0.4).toFixed(2)}`;

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

function hexPath(cx: number, cy: number, r: number): string {
  const pts: string[] = [];
  for (let i = 0; i < 6; i++) {
    const a = ((60 * i - 30) * Math.PI) / 180;
    pts.push(`${(cx + r * Math.cos(a)).toFixed(3)} ${(cy + r * Math.sin(a)).toFixed(3)}`);
  }
  return `M${pts.join("L")}Z`;
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

export function shotChart(s: PlayerSeason, radius: number): string {
  const shown = s.hex.filter(([q, r]) => radius * 1.5 * r <= DEPTH);
  const most = Math.max(1, ...shown.map((c) => c[2]));
  const full = Math.max(1, most * 0.6); // the busiest cells all draw at full size
  const cells = shown
    .map(([q, r, att, made, lg]) => {
      const cx = radius * SQRT3 * (q + r / 2);
      const cy = toY(radius * 1.5 * r);
      const size = radius * 0.94 * (0.32 + 0.68 * Math.sqrt(Math.min(1, att / full)));
      const league = lg < 0 ? "no league shots here" : `league ${(lg / 10).toFixed(1)}%`;
      const tip = `<b>${made} of ${att}</b> (${((100 * made) / att).toFixed(0)}%)<br>${league}`;
      return `<path class="t${tone(att, made, lg)}" d="${hexPath(cx, cy, size)}" data-tip="${esc(tip)}"/>`;
    })
    .join("");
  const attempts = s.hex.reduce((a, c) => a + c[2], 0);
  const title = `${s.label} shot chart: ${attempts} located field-goal attempts`;
  return `<svg class="court" viewBox="${COURT_VIEWBOX}" role="img" aria-label="${esc(title)}"><rect class="floor" x="-7.5" y="0" width="15" height="${toY(DEPTH)}"/>${courtLines()}<g class="cells">${cells}</g></svg>`;
}

// ---- Distance bands ------------------------------------------------------------------------

function share(b: Bands, key: string) {
  const total = Object.values(b).reduce((a, [att]) => a + att, 0);
  return total ? (100 * b[key][0]) / total : 0;
}

export function bandsHtml(s: PlayerSeason, up: string, down: string): string {
  const rows = Object.keys(BAND_LABELS)
    .map((key) => {
      const [att, made] = s.bands[key] ?? [0, 0];
      const [la, lm] = s.leagueBands[key] ?? [0, 0];
      const [label, range] = BAND_LABELS[key];
      const mine = att ? (100 * made) / att : null;
      const theirs = la ? (100 * lm) / la : null;
      const diff = mine !== null && theirs !== null && att >= 10 ? mine - theirs : null;
      const mark =
        diff === null || Math.abs(diff) < 2.5
          ? ""
          : `<span class="car ${diff > 0 ? "up" : "down"}" aria-hidden="true">${diff > 0 ? up : down}</span>`;
      const diffText = diff === null ? "–" : `${diff > 0 ? "+" : diff < 0 ? "−" : ""}${Math.abs(diff).toFixed(1)}`;
      return `<tr><th scope="row">${label}<small>${range}</small></th><td class="num">${att}</td><td class="num">${share(s.bands, key).toFixed(0)}%</td><td class="num muted">${share(s.leagueBands, key).toFixed(0)}%</td><td class="num">${mine === null ? "–" : mine.toFixed(1)}</td><td class="num muted">${theirs === null ? "–" : theirs.toFixed(1)}</td><td class="num diff">${mark}${diffText}</td></tr>`;
    })
    .join("");
  return rows;
}

// ---- Game log ------------------------------------------------------------------------------

export function logHtml(s: PlayerSeason): string {
  const at = (name: string) => s.logFields.indexOf(name);
  const g = (name: string) => s.gameFields.indexOf(name);
  const lines = [...s.log].reverse();
  if (!lines.length) return `<tr class="empty"><td colspan="15">No box-score lines this season.</td></tr>`;
  return lines
    .map((line) => {
      const game = s.games[String(line[at("game_id")])];
      const team = String(line[at("team")]);
      const n = (name: string) => Number(line[at(name)]);
      const [date, round, phase, home, away, hs, as] = [
        game?.[g("date")],
        game?.[g("round")],
        game?.[g("phase")],
        game?.[g("home")],
        game?.[g("away")],
        Number(game?.[g("home_score")]),
        Number(game?.[g("away_score")]),
      ];
      const isHome = home === team;
      const opp = isHome ? away : home;
      const own = isHome ? hs : as;
      const their = isHome ? as : hs;
      const won = own > their;
      const d = new Date(`${date}T12:00:00Z`).toLocaleDateString("en-GB", { day: "numeric", month: "short" });
      const pm = n("pm");
      const phaseText = phase === "RS" ? `Round ${round}` : String(phase);
      return `<tr><td class="date" data-tip="${esc(`${date} · ${phaseText}`)}">${d}</td><td class="opp"><span class="at">${isHome ? "vs" : "@"}</span>${esc(String(opp))}</td><td class="res"><span class="wl ${won ? "w" : "l"}">${won ? "W" : "L"}</span>${own}–${their}</td><td class="num">${Math.round(n("sec") / 60)}</td><td class="num strong">${n("pts")}</td><td class="num">${n("oreb") + n("dreb")}</td><td class="num">${n("ast")}</td><td class="num">${n("stl")}</td><td class="num">${n("blk")}</td><td class="num">${n("tov")}</td><td class="num">${n("fg2m") + n("fg3m")}-${n("fg2a") + n("fg3a")}</td><td class="num">${n("fg3m")}-${n("fg3a")}</td><td class="num">${n("ftm")}-${n("fta")}</td><td class="num">${n("pir")}</td><td class="num">${pm > 0 ? "+" : pm < 0 ? "−" : ""}${Math.abs(pm)}</td></tr>`;
    })
    .join("");
}
