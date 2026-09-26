// Shots explorer: the selection (league, a team's shots taken or allowed, a player), the facet
// filters (quarter, distance, play type) and what they show. One pass over the season's attempts
// gives the court cells, every facet's bars (each counted under the other facets' filters, so a
// facet shows what choosing in it would give) and the league under the same filters.
// Pure, so the build and the page share it.

import { dotChart, esc, hexChart, hexOf, type HexCell } from "./court";
import { BAND_LABELS } from "./player";

export interface Attempts {
  season: number;
  label: string;
  validated: boolean;
  radius: number;
  bits: Record<"made" | "three" | "fastbreak" | "second_chance" | "band" | "period", number>;
  bands: string[];
  teams: [code: string, name: string][];
  /** [id, name, slug, team code] */
  players: [string, string, string, string][];
  x: number[]; // court centimetres
  y: number[];
  flags: number[];
  player: number[];
  team: number[];
  opp: number[];
}

export type Who = "league" | "team" | "player";
export type Side = "taken" | "allowed";
export type ViewKind = "fg" | "freq" | "dots";
export type Facet = "q" | "band" | "play";

export interface State {
  season: number;
  who: Who;
  team: string;
  side: Side;
  player: string;
  view: ViewKind;
  q: string[]; // "1".."4", "ot"; empty: every quarter
  band: string[]; // band names; empty: every distance
  play: string[]; // "fb", "sc", "other"; empty: every play
}

export const FACETS: { key: Facet; title: string; options: [value: string, label: string, about: string][] }[] = [
  {
    key: "q",
    title: "Quarter",
    options: [
      ["1", "1st", "First quarter"],
      ["2", "2nd", "Second quarter"],
      ["3", "3rd", "Third quarter"],
      ["4", "4th", "Fourth quarter"],
      ["ot", "OT", "Overtime periods"],
    ],
  },
  {
    key: "band",
    title: "Distance",
    options: Object.entries(BAND_LABELS).map(([k, [label, range]]) => [k, label, range]),
  },
  {
    key: "play",
    title: "Play",
    options: [
      ["fb", "Fastbreak", "Made on a fastbreak"],
      ["sc", "Second chance", "Made after an offensive rebound"],
      ["other", "Neither", "Made shots that were neither"],
    ],
  },
];

export const VIEWS: [ViewKind, string, string][] = [
  ["freq", "Frequency", "Freq."],
  ["fg", "FG% vs league", "FG%"],
  ["dots", "Attempts", "Att."],
];

export const defaultState = (season: number): State => ({
  season,
  who: "league",
  team: "",
  side: "taken",
  player: "",
  view: "freq",
  q: [],
  band: [],
  play: [],
});

// ---- URL state -----------------------------------------------------------------------------

const list = (v: string | null, allowed: string[]) =>
  (v ?? "").split(",").filter((x) => allowed.includes(x));

export function readState(params: URLSearchParams, fallback: number, seasons: number[]): State {
  const s = defaultState(fallback);
  const season = Number(params.get("season"));
  if (seasons.includes(season)) s.season = season;
  s.team = params.get("team")?.toUpperCase() ?? "";
  s.player = params.get("player")?.toUpperCase() ?? "";
  s.who = s.player ? "player" : s.team ? "team" : "league";
  s.side = params.get("side") === "allowed" ? "allowed" : "taken";
  const view = params.get("view");
  s.view = view === "fg" || view === "dots" ? view : "freq";
  for (const f of FACETS) s[f.key] = list(params.get(f.key), f.options.map((o) => o[0]));
  return s;
}

export function writeState(s: State, fallback: number): string {
  const p = new URLSearchParams();
  if (s.season !== fallback) p.set("season", String(s.season));
  if (s.who === "team" && s.team) {
    p.set("team", s.team);
    if (s.side === "allowed") p.set("side", "allowed");
  }
  if (s.who === "player" && s.player) p.set("player", s.player);
  if (s.view !== "freq") p.set("view", s.view);
  for (const f of FACETS) if (s[f.key].length) p.set(f.key, s[f.key].join(","));
  const q = p.toString();
  return q ? `?${q}` : "";
}

// ---- One pass ------------------------------------------------------------------------------

export interface Bar {
  value: string;
  label: string;
  about: string;
  att: number;
  made: number;
  leagueAtt: number;
  leagueMade: number;
  on: boolean; // included by the current filter
}

export interface Result {
  att: number;
  made: number;
  pts: number;
  /** the comparison: the league with the same filters, or every league shot for the league itself */
  leagueAtt: number;
  leagueMade: number;
  leaguePts: number;
  /** a play filter keeps made shots only, so FG% no longer means anything */
  makesOnly: boolean;
  /** why the play facet cannot filter, or "" when it can */
  playOff: string;
  /** made shots under the quarter and distance filters: [selection, league], the play shares' base */
  playMakes: [number, number];
  isLeague: boolean;
  facets: Record<Facet, Bar[]>;
  cells: HexCell[];
  points: { x: number; y: number; made: boolean }[];
}

/** The first season whose play-by-play marks fastbreaks and second chances. */
export const FIRST_PLAY_SEASON = 2015;

export function playOff(a: Attempts, s: State): string {
  if (a.season < FIRST_PLAY_SEASON) return "The play-by-play marks fastbreaks and second chances from 2015-16 on.";
  if (s.view === "fg") return "Only made shots are marked, so this filter works in the Frequency and Attempts views.";
  return "";
}

export function run(a: Attempts, s: State): Result {
  const bit = (f: number, name: keyof Attempts["bits"], width = 1) => (f >> a.bits[name]) & ((1 << width) - 1);
  const teamAt = a.teams.findIndex(([c]) => c === s.team);
  const playerAt = a.players.findIndex(([id]) => id === s.player);
  const isLeague = !(s.who === "team" && teamAt >= 0) && !(s.who === "player" && playerAt >= 0);
  const subject = isLeague
    ? () => true
    : s.who === "team"
      ? (i: number) => (s.side === "taken" ? a.team[i] : a.opp[i]) === teamAt
      : (i: number) => a.player[i] === playerAt;
  const off = playOff(a, s);
  const on = { q: new Set(s.q), band: new Set(s.band), play: new Set(off ? [] : s.play) };
  const makesOnly = on.play.size > 0;
  // [mine att, mine made, league att, league made] per facet value; the play facet counts makes.
  const counts = Object.fromEntries(
    FACETS.map((f) => [f.key, Object.fromEntries(f.options.map(([v]) => [v, [0, 0, 0, 0]]))]),
  ) as Record<Facet, Record<string, number[]>>;
  const mine = new Map<string, [number, number, number, number]>();
  const league = new Map<string, [number, number]>();
  const everyCell = new Map<string, [number, number]>();
  const points: Result["points"] = [];
  let att = 0,
    made = 0,
    pts = 0,
    leagueAtt = 0,
    leagueMade = 0,
    leaguePts = 0,
    allAtt = 0,
    allMade = 0,
    allPts = 0;
  const playMakes: [number, number] = [0, 0];
  const bump = (m: Map<string, [number, number]>, key: string, isMade: number) => {
    const c = m.get(key) ?? [0, 0];
    c[0]++;
    c[1] += isMade;
    m.set(key, c);
  };

  for (let i = 0; i < a.flags.length; i++) {
    const f = a.flags[i];
    const period = f >> a.bits.period;
    const q = period >= 5 ? "ot" : String(period);
    const band = a.bands[bit(f, "band", 3)];
    const isMade = bit(f, "made");
    const fb = bit(f, "fastbreak") === 1;
    const sc = bit(f, "second_chance") === 1;
    const plays = !isMade ? [] : fb || sc ? [...(fb ? ["fb"] : []), ...(sc ? ["sc"] : [])] : ["other"];
    const qOk = !on.q.size || on.q.has(q);
    const bOk = !on.band.size || on.band.has(band);
    const pOk = !makesOnly || plays.some((p) => on.play.has(p));
    const value = isMade ? 2 + bit(f, "three") : 0;
    const mineToo = subject(i);
    const x = a.x[i] / 100;
    const y = a.y[i] / 100;
    const [hq, hr] = hexOf(x, y, a.radius);
    const key = `${hq},${hr}`;
    if (isLeague) {
      bump(everyCell, key, isMade);
      allAtt++;
      allMade += isMade;
      allPts += value;
    }
    // Facet bars: each under the other two facets' filters.
    const tally = (facet: Facet, k: string) => {
      const c = counts[facet][k];
      c[2]++;
      c[3] += isMade;
      if (mineToo) {
        c[0]++;
        c[1] += isMade;
      }
    };
    if (bOk && pOk) tally("q", q);
    if (qOk && pOk) tally("band", band);
    if (qOk && bOk && isMade) {
      for (const p of plays) tally("play", p);
      playMakes[1]++;
      if (mineToo) playMakes[0]++;
    }
    if (!(qOk && bOk && pOk)) continue;

    bump(league, key, isMade);
    leagueAtt++;
    leagueMade += isMade;
    leaguePts += value;
    if (!mineToo) continue;
    const mc = mine.get(key) ?? [hq, hr, 0, 0];
    mc[2]++;
    mc[3] += isMade;
    mine.set(key, mc);
    att++;
    made += isMade;
    pts += value;
    points.push({ x, y, made: isMade === 1 });
  }

  // The league against itself says nothing: compare it with every league shot instead.
  const base = isLeague ? everyCell : league;
  const facets = Object.fromEntries(
    FACETS.map((f) => [
      f.key,
      f.options.map(([value, label, about]) => {
        const [ma, mm, la, lm] = counts[f.key][value];
        return {
          value,
          label,
          about,
          att: ma,
          made: mm,
          leagueAtt: isLeague && f.key !== "play" ? allAtt : la,
          leagueMade: isLeague && f.key !== "play" ? allMade : lm,
          on: !on[f.key].size || on[f.key].has(value),
        };
      }),
    ]),
  ) as Record<Facet, Bar[]>;
  const cells: HexCell[] = [...mine.entries()]
    .map(([key, [q, r, n, m]]): HexCell => {
      const [la, lm] = base.get(key) ?? [0, 0];
      return [q, r, n, m, la ? Math.round((1000 * lm) / la) : -1];
    })
    .sort((p, q) => p[0] - q[0] || p[1] - q[1]);
  return {
    att,
    made,
    pts,
    leagueAtt: isLeague ? allAtt : leagueAtt,
    leagueMade: isLeague ? allMade : leagueMade,
    leaguePts: isLeague ? allPts : leaguePts,
    makesOnly,
    playOff: off,
    playMakes,
    isLeague,
    facets,
    cells,
    points,
  };
}

// ---- Rendering -----------------------------------------------------------------------------

const pct = (m: number, n: number) => (n ? ((100 * m) / n).toFixed(1) : "–");
const signed = (x: number) => `${x > 0 ? "+" : x < 0 ? "−" : ""}${Math.abs(x).toFixed(1)}`;
const count = (n: number) => n.toLocaleString("en-GB");

export function subjectLabel(a: Attempts, s: State): string {
  if (s.who === "team") {
    const name = a.teams.find(([c]) => c === s.team)?.[1] ?? s.team;
    return s.side === "taken" ? `${name}, shots taken` : `${name}, shots allowed`;
  }
  if (s.who === "player") return a.players.find(([id]) => id === s.player)?.[1] ?? "Player";
  return "Every EuroLeague shot";
}

const freqStep = ([, , att]: HexCell, most: number) => {
  const r = att / most;
  return r >= 0.6 ? "f5" : r >= 0.3 ? "f4" : r >= 0.15 ? "f3" : r >= 0.06 ? "f2" : "f1";
};

export function courtHtml(a: Attempts, s: State, r: Result): string {
  const label = `${subjectLabel(a, s)}, ${a.label}: ${r.att} located attempts`;
  if (s.view === "dots") return dotChart(r.points, label);
  const invert = s.who === "team" && s.side === "allowed";
  const who = invert ? "Opponents " : "";
  const tip = ([, , att, made, lg]: HexCell, total: number) =>
    (r.makesOnly
      ? `${who}<b>${att} made shots</b>, ${((100 * att) / total).toFixed(1)}% of these<br>`
      : `${who}<b>${made} of ${att}</b> (${((100 * made) / att).toFixed(0)}%), ${((100 * att) / total).toFixed(1)}% of these shots<br>`) +
    (lg < 0
      ? "no league shots here"
      : r.isLeague
        ? `every league shot ${(lg / 10).toFixed(1)}%`
        : `league ${(lg / 10).toFixed(1)}% with the same filters`);
  if (s.view === "freq") {
    const most = Math.max(1, ...r.cells.map((c) => c[2]));
    return hexChart(r.cells, a.radius, { label, tip, color: (c) => freqStep(c, most) });
  }
  return hexChart(r.cells, a.radius, { label, tip, invert });
}

export function totalsHtml(r: Result): string {
  const ppsMine = r.att ? (r.pts / r.att).toFixed(2) : "–";
  const ppsLeague = r.leagueAtt ? (r.leaguePts / r.leagueAtt).toFixed(2) : "–";
  const of = r.isLeague ? "every shot" : "league";
  // The whole league with nothing filtered is its own comparison: say so once, not three times.
  if (r.isLeague && !r.makesOnly && r.att === r.leagueAtt)
    return `<div><dt>Attempts</dt><dd>${count(r.att)}</dd><dd class="lg">every located shot</dd></div><div><dt>FG%</dt><dd>${pct(r.made, r.att)}</dd><dd class="lg">&nbsp;</dd></div><div><dt>Points per shot</dt><dd>${ppsMine}</dd><dd class="lg">&nbsp;</dd></div>`;
  if (r.makesOnly)
    return `<div><dt>Made shots</dt><dd>${count(r.att)}</dd><dd class="lg">of ${count(r.leagueAtt)}</dd></div><div><dt>FG%</dt><dd>–</dd><dd class="lg">made shots only</dd></div><div><dt>Points per make</dt><dd>${ppsMine}</dd><dd class="lg">${of} ${ppsLeague}</dd></div>`;
  return `<div><dt>Attempts</dt><dd>${count(r.att)}</dd><dd class="lg">of ${count(r.leagueAtt)}</dd></div><div><dt>FG%</dt><dd>${pct(r.made, r.att)}</dd><dd class="lg">${of} ${pct(r.leagueMade, r.leagueAtt)}</dd></div><div><dt>Points per shot</dt><dd>${ppsMine}</dd><dd class="lg">${of} ${ppsLeague}</dd></div>`;
}

export interface FacetContext {
  facet: Facet;
  chosen: string[];
  r: Result;
}

/** Text of one bar: FG% against the comparison, or for the play facet the share of made shots. */
function barText(b: Bar, c: FacetContext) {
  if (c.facet === "play") {
    const [mine, lg] = c.r.playMakes;
    const share = mine ? (100 * b.att) / mine : null;
    const theirs = lg ? (100 * b.leagueAtt) / lg : null;
    return {
      fg: share === null ? "–" : share.toFixed(1),
      diff:
        share !== null && theirs !== null && !c.r.isLeague && !c.r.playOff && mine >= 10
          ? signed(share - theirs)
          : "–",
      tip: `<b>${esc(b.label)}</b> ${esc(b.about)}<br>${b.att} made shots, ${share === null ? "–" : share.toFixed(1)}% of the makes; league ${theirs === null ? "–" : theirs.toFixed(1)}%`,
    };
  }
  const fg = b.att ? (100 * b.made) / b.att : null;
  const lg = b.leagueAtt ? (100 * b.leagueMade) / b.leagueAtt : null;
  const fgText = fg === null || c.r.makesOnly ? "–" : fg.toFixed(1);
  const of = c.r.isLeague ? "every shot" : "league";
  return {
    fg: fgText,
    diff: fg !== null && lg !== null && b.att >= 10 && !c.r.makesOnly ? signed(fg - lg) : "–",
    tip: c.r.makesOnly
      ? `<b>${esc(b.label)}</b> ${esc(b.about)}<br>${b.att} made shots`
      : `<b>${esc(b.label)}</b> ${esc(b.about)}<br>${b.made} of ${b.att} (${fgText}%), ${of} ${lg === null ? "–" : lg.toFixed(1)}%`,
  };
}

const disabled = (c: FacetContext) => c.facet === "play" && c.r.playOff !== "";

/** One facet's bar rows: width by attempts, blue when kept, gray when filtered out. */
export function facetRows(bars: Bar[], c: FacetContext): string {
  const most = Math.max(1, ...bars.map((b) => b.att));
  const off = disabled(c);
  return bars
    .map((b) => {
      const t = barText(b, c);
      return `<button type="button" class="bar${b.on && !off ? " on" : ""}" data-facet="${c.facet}" data-value="${b.value}" aria-pressed="${c.chosen.includes(b.value)}"${off ? " disabled" : ""} data-tip="${esc(t.tip)}"><span class="b-label">${esc(b.label)}</span><span class="b-track"><span class="b-bar" style="width:${((100 * b.att) / most).toFixed(1)}%"></span></span><span class="b-att">${count(b.att)}</span><span class="b-fg">${t.fg}</span><span class="b-diff">${t.diff}</span></button>`;
    })
    .join("");
}

/** Patch facet rows in place so the bars resize instead of redrawing (the signature). */
export function patchFacet(root: HTMLElement, bars: Bar[], c: FacetContext) {
  const most = Math.max(1, ...bars.map((b) => b.att));
  const off = disabled(c);
  for (const b of bars) {
    const row = root.querySelector<HTMLButtonElement>(`.bar[data-value="${b.value}"]`);
    if (!row) continue;
    const t = barText(b, c);
    row.classList.toggle("on", b.on && !off);
    row.disabled = off;
    row.setAttribute("aria-pressed", String(c.chosen.includes(b.value)));
    row.querySelector<HTMLElement>(".b-bar")!.style.width = `${((100 * b.att) / most).toFixed(1)}%`;
    row.querySelector(".b-att")!.textContent = count(b.att);
    row.querySelector(".b-fg")!.textContent = t.fg;
    row.querySelector(".b-diff")!.textContent = t.diff;
    row.dataset.tip = t.tip;
  }
}
