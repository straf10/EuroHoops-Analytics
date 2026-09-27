// Leaders: pools of player lines (a season, whole careers, the best single seasons, or one club),
// qualifying floors, ranking, and the HTML for the court lineup and the bench. Pure functions,
// run at build time for the first paint and in the browser on every filter change.

import { esc } from "./format";
import { dataTip, tip } from "./tip";
import { jerseySvg } from "./jersey";
import { BY_KEY, playerHref } from "./players";

export type Scope = "season" | "career" | "best";
export type Mode = "game" | "total";
export type Totals = Record<string, number>;

/** One line as the JSON endpoints ship it. ``season`` is 0 on a career line. */
export type Wire = [slug: string, name: string, team: string, dorsal: string, season: number, first: number, last: number, totals: number[]];

export interface Pool {
  fields: string[];
  /** season -> the most games any player played in it (the season floor's reference) */
  most: Record<string, number>;
  rows: Wire[];
}

export interface Entry {
  key: string;
  slug: string;
  name: string;
  team: string;
  dorsal: string;
  season: number;
  first: number;
  last: number;
  t: Totals;
}

export interface View {
  scope: Scope;
  season: number;
  stat: string;
  mode: Mode;
  club: string; // "" = the whole league
}

// Only the fields the leaders' measures read; the endpoints ship these, in this order.
export const LEADER_FIELDS = ["gp", "gs", "sec", "pts", "fg2m", "fg2a", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb", "ast", "stl", "tov", "blk", "pir", "pm"];

export const STATS = ["pts", "reb", "ast", "stl", "blk", "pir", "ts", "efg", "fg3_pct", "ft_pct", "pm"];
export const CARD_STATS = ["pts", "reb", "ast", "ts", "pir"];
export const SCOPES: { key: Scope; label: string }[] = [
  { key: "season", label: "Season" },
  { key: "career", label: "Career" },
  { key: "best", label: "Best seasons" },
];
export const MODES: { key: Mode; label: string }[] = [
  { key: "game", label: "Per game" },
  { key: "total", label: "Totals" },
];
export const CAREER_GAMES = 60;
export const SEASON_MINUTES = 15;
export const BENCH = 15;

export const seasonLabel = (s: number) => `${s}-${String((s + 1) % 100).padStart(2, "0")}`;

export function entries(pool: Pool): Entry[] {
  return pool.rows.map(([slug, name, team, dorsal, season, first, last, values]) => {
    const t: Totals = {};
    pool.fields.forEach((f, i) => (t[f] = values[i]));
    return { key: `${slug}:${season}:${team}`, slug, name, team, dorsal, season, first, last, t };
  });
}

/** Per-season club lines -> one line per player over all of them: club and number worn most. */
export function careers(lines: Entry[]): Entry[] {
  const by = new Map<string, Entry[]>();
  for (const e of lines) by.set(e.slug, [...(by.get(e.slug) ?? []), e]);
  return [...by.values()].map((own) => {
    const t: Totals = {};
    for (const e of own) for (const [k, v] of Object.entries(e.t)) t[k] = (t[k] ?? 0) + v;
    const most = (pick: (e: Entry) => string) => {
      const gp = new Map<string, number>();
      for (const e of own) gp.set(pick(e), (gp.get(pick(e)) ?? 0) + e.t.gp);
      return [...gp].reduce((a, b) => (b[1] > a[1] ? b : a))[0];
    };
    const last = own[own.length - 1];
    return {
      key: `${last.slug}:0`,
      slug: last.slug,
      name: last.name,
      team: most((e) => e.team),
      dorsal: most((e) => e.dorsal),
      season: 0,
      first: Math.min(...own.map((e) => e.first)),
      last: Math.max(...own.map((e) => e.last)),
      t,
    };
  });
}

const measure = (key: string) => BY_KEY[key] ?? BY_KEY.pts;

export function valueOf(key: string, t: Totals, mode: Mode): number | null {
  const m = measure(key);
  const raw = m.value(t);
  if (raw === null || m.kind !== "count" || mode === "total") return raw;
  return t.gp ? raw / t.gp : null;
}

export function text(key: string, v: number | null, mode: Mode): string {
  const m = measure(key);
  if (v === null || !Number.isFinite(v)) return "–";
  const digits = m.kind === "count" && mode === "total" ? 0 : m.digits;
  const body = Math.abs(v).toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
  if (v < 0 && Number(Math.abs(v).toFixed(digits)) !== 0) return `−${body}`;
  return m.signed && v > 0 ? `+${body}` : body;
}

const seasonFloor = (most: Record<string, number>, season: number) => Math.ceil((most[season] ?? 0) / 2);

/** Why a line is not ranked on the view's stat, or "" when it is. */
export function failsFloor(e: Entry, view: View, most: Record<string, number>): string {
  const m = measure(view.stat);
  if (m.qualifies && !m.qualifies(e.t)) return m.floor ?? "Too few attempts";
  if (view.mode === "total" && m.kind === "count") return "";
  if (e.season === 0) return e.t.gp >= CAREER_GAMES ? "" : `Fewer than ${CAREER_GAMES} games`;
  const games = seasonFloor(most, e.season);
  if (e.t.gp < games) return `Fewer than ${games} games, half the most anyone played`;
  if (e.t.sec / e.t.gp < SEASON_MINUTES * 60) return `Under ${SEASON_MINUTES} minutes a game`;
  return "";
}

export interface Ranked {
  e: Entry;
  value: number | null;
  text: string;
  rank: number; // 0 when not ranked
  floor: string;
}

export function rank(list: Entry[], view: View, most: Record<string, number>): Ranked[] {
  const rows = list.map((e) => {
    const value = valueOf(view.stat, e.t, view.mode);
    return { e, value, text: text(view.stat, value, view.mode), rank: 0, floor: value === null ? "No value" : failsFloor(e, view, most) };
  });
  rows.sort((a, b) => {
    const qa = !a.floor, qb = !b.floor;
    if (qa !== qb) return qa ? -1 : 1;
    const d = (b.value ?? -Infinity) - (a.value ?? -Infinity);
    return d || b.e.t.sec - a.e.t.sec || a.e.name.localeCompare(b.e.name);
  });
  let n = 0;
  for (const r of rows) if (!r.floor) r.rank = ++n;
  return rows;
}

export interface Board {
  lineup: boolean; // a club's season: the most-played five, not the top five
  five: Ranked[]; // in court order: point, left wing, right wing, left block, right block
  bench: Ranked[];
}

/** The five on court and the bench. A club's season is its roster: the five who started most. */
export function board(list: Entry[], view: View, most: Record<string, number>): Board {
  const ranked = rank(list, view, most);
  const lineup = view.scope === "season" && !!view.club;
  let five: Ranked[];
  let bench: Ranked[];
  if (lineup) {
    const starters = [...ranked].sort((a, b) => b.e.t.gs - a.e.t.gs || b.e.t.sec - a.e.t.sec).slice(0, 5);
    const share = (r: Ranked) => r.e.t.ast / Math.max(1, r.e.t.ast + r.e.t.oreb + r.e.t.dreb);
    five = starters.sort((a, b) => share(b) - share(a));
    bench = ranked.filter((r) => !five.includes(r));
  } else {
    const top = ranked.filter((r) => r.rank);
    five = top.slice(0, 5);
    bench = top.slice(5, 5 + BENCH);
  }
  return { lineup, five, bench };
}

export function title(view: View, clubName = ""): string {
  const m = measure(view.stat);
  const rate = m.kind === "count" ? (view.mode === "game" ? " per game" : ", totals") : "";
  const where = view.club ? ` · ${clubName || view.club}` : "";
  const when = view.scope === "season" ? seasonLabel(view.season) : view.scope === "career" ? "careers since 2007-08" : "best single seasons since 2007-08";
  return `${m.name}${rate}, ${when}${where}`;
}

export function floorNote(view: View, most: Record<string, number>): string {
  const m = measure(view.stat);
  const counts = view.mode === "total" && m.kind === "count";
  const lines: string[] = [];
  if (!counts) {
    lines.push(
      view.scope === "career"
        ? `Ranked: at least ${CAREER_GAMES} games.`
        : view.scope === "season"
          ? `Ranked: at least ${seasonFloor(most, view.season)} games (half the most anyone played) and ${SEASON_MINUTES} minutes a game.`
          : `Ranked: in its season, at least half the most games anyone played and ${SEASON_MINUTES} minutes a game.`,
    );
  } else lines.push("Totals are ranked without a floor.");
  if (m.floor) lines.push(`${m.floor}: not ranked.`);
  return lines.join(" ");
}

const when = (e: Entry) =>
  e.season ? seasonLabel(e.season) : e.first === e.last ? seasonLabel(e.first) : `${seasonLabel(e.first)} to ${seasonLabel(e.last)}`;

function statLine(e: Entry, view: View): string {
  return CARD_STATS.map((k) => {
    const on = k === view.stat ? " on" : "";
    return `<span class="st${on}"><small>${measure(k).head}</small>${text(k, valueOf(k, e.t, view.mode), view.mode)}</span>`;
  }).join("");
}

function games(e: Entry): string {
  const min = e.t.gp ? (e.t.sec / e.t.gp / 60).toFixed(1) : "–";
  return `${e.t.gp} games · ${min} min`;
}

/** Slot anchors on the closed court drawing, in % of its width and height (point, wings, blocks).
 *  Wings sit beyond the arc on the 45° lines and the point well above the halfway line, so no
 *  court line reaches a player's text. */
export const SLOTS: [number, number][] = [
  [50, 56],
  [13.6, 51],
  [86.4, 51],
  [35.7, 14.6],
  [64.3, 14.6],
];

export function fiveHtml(b: Board, view: View, base: string): string {
  const head = measure(view.stat).head;
  return b.five
    .map((r, i) => {
      const { e } = r;
      const [x, y] = SLOTS[i];
      const tag = b.lineup ? "" : `<span class="rk">${r.rank}</span>`;
      const muted = r.floor ? " muted" : "";
      const floorAttr = r.floor ? dataTip(tip`${r.floor}`) : "";
      return (
        `<li class="slot" style="--x:${x}%;--y:${y}%" data-key="${esc(e.key)}">` +
        `<a class="shirt-link" href="${playerHref(base, e.slug)}" tabindex="-1" aria-hidden="true">${jerseySvg({ code: e.team, name: e.name, number: e.dorsal, size: 72 })}</a>` +
        `<span class="tag">` +
        `<span class="who">${tag}<a href="${playerHref(base, e.slug)}">${esc(e.name)}</a></span>` +
        `<span class="meta">${esc(e.team)} · ${when(e)}</span>` +
        `<span class="big${muted}"${floorAttr}>${r.text}<small>${head}</small></span>` +
        `<span class="line">${statLine(e, view)}</span>` +
        `</span>` +
        `</li>`
      );
    })
    .join("");
}

export function benchHtml(b: Board, view: View, base: string): string {
  return b.bench
    .map((r) => {
      const { e } = r;
      const floorAttr = r.floor ? dataTip(tip`${r.floor}`) : "";
      return (
        `<li data-key="${esc(e.key)}">` +
        `<span class="rk">${r.rank || ""}</span>` +
        jerseySvg({ code: e.team, name: e.name, number: e.dorsal, size: 30 }) +
        `<span class="nm"><a href="${playerHref(base, e.slug)}">${esc(e.name)}</a><small>${esc(e.team)} · ${when(e)} · ${games(e)}</small></span>` +
        `<span class="val${r.floor ? " muted" : ""}"${floorAttr}>${r.text}</span>` +
        `</li>`
      );
    })
    .join("");
}

/** The phone list under the court: the five with their full stat lines. */
export function fiveLinesHtml(b: Board, view: View, base: string): string {
  return b.five
    .map(
      (r) =>
        `<li>${b.lineup ? "" : `<span class="rk">${r.rank}</span>`}<span class="nm"><a href="${playerHref(base, r.e.slug)}">${esc(r.e.name)}</a><small>${esc(r.e.team)} · ${when(r.e)} · ${games(r.e)}</small></span><span class="line">${statLine(r.e, view)}</span></li>`,
    )
    .join("");
}

/** The bench's one-line note, so its head matches the court's (title, note, rule). */
export function benchNote(b: Board): string {
  if (!b.bench.length) return "Nobody else qualifies.";
  return b.lineup ? "The rest of the roster, ranked." : `Ranks 6 to ${5 + b.bench.length}.`;
}

export function caption(b: Board): string {
  if (!b.five.length) return "";
  return b.lineup
    ? "On court: the five who started most, placed by their share of assists against rebounds (the data has no positions). The rest of the roster is on the bench, ranked."
    : `On court: the top five, first at the point, second and third on the wings, fourth and fifth on the blocks. Bench: ranks 6 to ${5 + b.bench.length}.`;
}

/** Link to Compare with the five on court. */
export function compareHref(b: Board, view: View, base: string): string {
  const root = base.replace(/\/$/, "");
  const career = view.scope === "career";
  const p = b.five.map((r) => (career ? r.e.slug : `${r.e.slug}:${r.e.season}`)).join(",");
  return `${root}/compare/?scope=${career ? "career" : "season"}&p=${p}`;
}

export const DEFAULT_VIEW: Omit<View, "season"> = { scope: "season", stat: "pts", mode: "game", club: "" };

export function toQuery(view: View, defaultSeason: number): string {
  const q = new URLSearchParams();
  if (view.scope !== DEFAULT_VIEW.scope) q.set("scope", view.scope);
  if (view.scope === "season" && view.season !== defaultSeason) q.set("season", String(view.season));
  if (view.stat !== DEFAULT_VIEW.stat) q.set("stat", view.stat);
  if (view.mode !== DEFAULT_VIEW.mode) q.set("mode", view.mode);
  if (view.club) q.set("club", view.club);
  const s = q.toString();
  return s ? `?${s}` : "";
}

export function fromQuery(search: string, seasons: number[], defaultSeason: number, clubs: string[]): View {
  const q = new URLSearchParams(search);
  const scope = SCOPES.some((s) => s.key === q.get("scope")) ? (q.get("scope") as Scope) : DEFAULT_VIEW.scope;
  const season = Number(q.get("season"));
  const stat = q.get("stat") ?? "";
  const club = (q.get("club") ?? "").toUpperCase();
  return {
    scope,
    season: seasons.includes(season) ? season : defaultSeason,
    stat: STATS.includes(stat) ? stat : DEFAULT_VIEW.stat,
    mode: q.get("mode") === "total" ? "total" : "game",
    club: clubs.includes(club) ? club : "",
  };
}

