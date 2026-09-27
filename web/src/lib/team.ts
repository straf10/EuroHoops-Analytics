// Team page renderers: ratings and four factors from team and opponent totals, the rank strips
// (every club's dot, this club's in blue), both shot charts, the bands and the roster.
// Pure (types only from ./stats), so the build and the in-page season switch share them.

import { hexChart, type HexCell } from "./court";
import { esc } from "./format";
import { tip, tipAttr } from "./tip";
import { SEASON_COLS, bandRows, totalsOf, type Totals } from "./player";
import { teamHref } from "./players";
import type { TeamSeason } from "./stats";

/** Ratings and four factors of one side, from its totals and its opponents'. */
export function factors(fields: string[], own: number[], against: number[]) {
  const t = totalsOf(fields, own);
  const o = totalsOf(fields, against);
  const fga = (x: Totals) => x.fg2a + x.fg3a;
  const efg = (x: Totals) => (100 * (x.fg2m + x.fg3m + 0.5 * x.fg3m)) / fga(x);
  const tov = (x: Totals) => (100 * x.tov) / (fga(x) + 0.44 * x.fta + x.tov);
  const ortg = (100 * t.pts) / t.poss;
  const drtg = (100 * o.pts) / o.poss;
  return {
    pace: ((t.poss + o.poss) / 2 / t.game_sec) * 2400,
    ortg,
    drtg,
    net: ortg - drtg,
    efg: efg(t),
    tov: tov(t),
    oreb: (100 * t.oreb) / (t.oreb + o.dreb),
    ftr: (100 * t.ftm) / fga(t),
    oefg: efg(o),
    otov: tov(o),
    dreb: (100 * t.dreb) / (t.dreb + o.oreb),
    oftr: (100 * o.ftm) / fga(o),
  };
}
type Factors = ReturnType<typeof factors>;
type Key = keyof Factors;

interface Metric {
  key: Key;
  name: string;
  about: string;
  high: boolean; // is a higher value better for this club?
}

export const OFFENSE: Metric[] = [
  { key: "ortg", name: "Offensive rating", about: "Points scored per 100 possessions", high: true },
  { key: "efg", name: "Effective FG%", about: "(FGM + 0.5 × 3PM) / FGA", high: true },
  { key: "tov", name: "Turnover rate", about: "Turnovers per 100 plays: TOV / (FGA + 0.44 × FTA + TOV)", high: false },
  { key: "oreb", name: "Offensive rebound %", about: "Share of its own misses the club rebounded", high: true },
  { key: "ftr", name: "Free-throw rate", about: "Free throws made per 100 field-goal attempts", high: true },
];
export const DEFENSE: Metric[] = [
  { key: "drtg", name: "Defensive rating", about: "Points allowed per 100 opponent possessions", high: false },
  { key: "oefg", name: "Opponent eFG%", about: "Opponents' effective FG%", high: false },
  { key: "otov", name: "Turnovers forced", about: "Opponent turnovers per 100 opponent plays", high: true },
  { key: "dreb", name: "Defensive rebound %", about: "Share of opponents' misses the club rebounded", high: true },
  { key: "oftr", name: "Opponent free-throw rate", about: "Opponent free throws made per 100 of their field-goal attempts", high: false },
];

export const ordinal = (n: number) => {
  const tail = n % 100 >= 11 && n % 100 <= 13 ? "th" : ({ 1: "st", 2: "nd", 3: "rd" } as Record<number, string>)[n % 10] ?? "th";
  return `${n}${tail}`;
};
export const signed = (x: number, digits = 1) => `${x > 0 ? "+" : x < 0 ? "−" : ""}${Math.abs(x).toFixed(digits)}`;


export interface Strip {
  key: Key;
  name: string;
  about: string;
  value: string;
  rank: string;
  me: number; // position on the track, 0-100
  tip: string;
  field: string; // the other clubs' dots and the league line
}

/** Where the club sits among every club that season, one strip per measure. */
export function strips(t: TeamSeason, metrics: Metric[], base: string): Strip[] {
  const all = t.clubs.map((c) => ({ c, f: factors(t.fields, c.totals, c.opp) }));
  const league = factors(t.fields, t.league, t.league);
  const me = factors(t.fields, t.totals, t.opp);
  return metrics.map(({ key, name, about, high }) => {
    const values = all.map((x) => x.f[key]);
    const lo = Math.min(...values, league[key]);
    const hi = Math.max(...values, league[key]);
    const pad = (hi - lo) * 0.06 || 1;
    const at = (v: number) => (100 * (v - lo + pad)) / (hi - lo + 2 * pad);
    const better = values.filter((v) => (high ? v > me[key] : v < me[key])).length;
    const rank = `${ordinal(better + 1)} of ${values.length}`;
    const text = (v: number) => v.toFixed(1);
    const dots = all
      .filter((x) => x.c.code !== t.code)
      .map(({ c, f }) => {
        const dotTip = tip`<b>${c.code}</b> ${c.name}<br>${name}: ${text(f[key])}`;
        return `<a class="s-dot" style="left:${at(f[key]).toFixed(2)}%" href="${teamHref(base, c.code, t.season)}" data-tip="${tipAttr(dotTip)}" aria-label="${esc(`${c.name}: ${text(f[key])}`)}"></a>`;
      })
      .join("");
    const avgTip = tip`League average: ${text(league[key])}`;
    return {
      key,
      name,
      about,
      value: text(me[key]),
      rank,
      me: at(me[key]),
      tip: tip`<b>${t.code}</b> ${t.name}<br>${name}: ${text(me[key])}, ${rank}`,
      field: `<span class="s-avg" style="left:${at(league[key]).toFixed(2)}%" data-tip="${tipAttr(avgTip)}"></span>${dots}`,
    };
  });
}

export const stripHtml = (s: Strip) =>
  `<div class="strip" data-key="${s.key}"><span class="s-name"><span data-tip="${tipAttr(tip`<b>${s.name}</b><br>${s.about}`)}" tabindex="0">${s.name}</span></span><span class="s-val">${s.value}</span><span class="s-rank">${s.rank}</span><span class="s-track"><span class="s-field">${s.field}</span><span class="s-me" style="left:${s.me.toFixed(2)}%" data-tip="${tipAttr(s.tip)}"></span></span></div>`;

// ---- Head ----------------------------------------------------------------------------------

export function facts(t: TeamSeason) {
  const f = factors(t.fields, t.totals, t.opp);
  return {
    record: `${t.w}–${t.l}`,
    net: signed(f.net),
    pace: f.pace.toFixed(1),
    lede: `${t.w} wins and ${t.l} losses in ${t.label}, scoring ${f.ortg.toFixed(1)} points per 100 possessions and allowing ${f.drtg.toFixed(1)}.`,
  };
}

// ---- Shots ---------------------------------------------------------------------------------

const cellTip = (who: string) => ([, , att, made, lg]: HexCell) => {
  const league = lg < 0 ? "no league shots here" : `league ${(lg / 10).toFixed(1)}%`;
  return tip`${who}<b>${made} of ${att}</b> (${((100 * made) / att).toFixed(0)}%)<br>${league}`;
};

export function teamCharts(t: TeamSeason, radius: number) {
  const n = (cells: HexCell[]) => cells.reduce((a, c) => a + c[2], 0);
  return {
    taken: hexChart(t.taken, radius, {
      label: `${t.code} ${t.label}: ${n(t.taken)} located field-goal attempts taken`,
      tip: cellTip(""),
    }),
    allowed: hexChart(t.allowed, radius, {
      label: `${t.code} ${t.label}: ${n(t.allowed)} located opponent attempts allowed`,
      invert: true,
      tip: cellTip("Opponents "),
    }),
  };
}

export const teamBands = (t: TeamSeason, up: string, down: string) => ({
  taken: bandRows(t.bands, t.leagueBands, up, down),
  allowed: bandRows(t.bandsAllowed, t.leagueBands, up, down, true),
});

// ---- Roster --------------------------------------------------------------------------------

export function rosterHtml(t: TeamSeason, base: string): string {
  if (!t.roster.length) return `<tr class="empty"><td colspan="16">No box-score lines this season.</td></tr>`;
  return t.roster
    .map((p) => {
      const tt = totalsOf(t.rosterFields, p.totals);
      const name = p.slug ? `<a href="${base}/players/${p.slug}/">${esc(p.name)}</a>` : esc(p.name);
      return `<tr><td class="num muted">${esc(p.dorsal)}</td><th scope="row">${name}</th>${SEASON_COLS.map((c) => `<td class="num">${c.cell(tt)}</td>`).join("")}</tr>`;
    })
    .join("");
}
