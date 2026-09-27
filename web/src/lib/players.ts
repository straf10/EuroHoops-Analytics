// Players dashboard: measures, filtering, ranking, and the HTML for the ranked chart and table.
// Pure functions of the season data and the view state, run both at build time (the first
// paint) and in the browser (every filter change), so the two can never disagree.

import { esc } from "./format";
import { tip, tipAttr } from "./tip";
import type { PlayerRow, SeasonPlayers } from "./stats";

export type Window = "season" | "last5" | "last10" | "last20";
export type Rate = "game" | "36" | "100";
export type ColumnSet = "box" | "shooting";

export interface View {
  season: number;
  window: Window;
  rate: Rate;
  sort: string;
  dir: "desc" | "asc";
  team: string; // "" = every team
  minMin: number; // minutes per game
  q: string;
  set: ColumnSet;
}

type Totals = Record<string, number>;

export interface Measure {
  key: string;
  head: string; // column head
  name: string; // chart title and tooltip, e.g. "Points"
  about: string;
  kind: "count" | "pct" | "fixed"; // counts follow the rate; percentages and fixed values do not
  digits: number;
  value: (t: Totals) => number | null;
  /** Enough attempts to rank on this percentage (per game). */
  qualifies?: (t: Totals) => boolean;
  floor?: string; // why a player is not ranked on it
  signed?: boolean;
}

const fga = (t: Totals) => t.fg2a + t.fg3a;
const fgm = (t: Totals) => t.fg2m + t.fg3m;
const ratio = (num: number, den: number) => (den > 0 ? (100 * num) / den : null);

const count = (key: string, head: string, name: string, about: string, extra: Partial<Measure> = {}): Measure => ({
  key,
  head,
  name,
  about,
  kind: "count",
  digits: 1,
  value: (t) => t[key],
  ...extra,
});

export const MEASURES: Measure[] = [
  { key: "gp", head: "GP", name: "Games played", about: "Games played", kind: "fixed", digits: 0, value: (t) => t.gp },
  {
    key: "min",
    head: "MIN",
    name: "Minutes",
    about: "Minutes per game, whatever the rate",
    kind: "fixed",
    digits: 1,
    value: (t) => (t.gp ? t.sec / t.gp / 60 : null),
  },
  count("pts", "PTS", "Points", "Points"),
  count("reb", "REB", "Rebounds", "Rebounds, offensive and defensive", { value: (t) => t.oreb + t.dreb }),
  count("ast", "AST", "Assists", "Assists"),
  count("stl", "STL", "Steals", "Steals"),
  count("blk", "BLK", "Blocks", "Shots blocked"),
  count("tov", "TOV", "Turnovers", "Turnovers"),
  count("pir", "PIR", "PIR", "Performance Index Rating, the EuroLeague's official efficiency score"),
  count("pm", "+/-", "Plus-minus", "Points scored minus points allowed while on court", { signed: true }),
  count("fga", "FGA", "Field-goal attempts", "Field-goal attempts, twos and threes", { value: fga }),
  {
    key: "fg_pct",
    head: "FG%",
    name: "Field-goal %",
    about: "Field goals made per attempt",
    kind: "pct",
    digits: 1,
    value: (t) => ratio(fgm(t), fga(t)),
    qualifies: (t) => fga(t) >= 3 * t.gp,
    floor: "Fewer than 3 field-goal attempts a game",
  },
  {
    key: "fg2_pct",
    head: "2P%",
    name: "Two-point %",
    about: "Two-pointers made per attempt",
    kind: "pct",
    digits: 1,
    value: (t) => ratio(t.fg2m, t.fg2a),
    qualifies: (t) => t.fg2a >= 2 * t.gp,
    floor: "Fewer than 2 two-point attempts a game",
  },
  count("fg3a", "3PA", "Three-point attempts", "Three-point attempts"),
  {
    key: "fg3_pct",
    head: "3P%",
    name: "Three-point %",
    about: "Three-pointers made per attempt",
    kind: "pct",
    digits: 1,
    value: (t) => ratio(t.fg3m, t.fg3a),
    qualifies: (t) => t.fg3a >= t.gp,
    floor: "Fewer than 1 three-point attempt a game",
  },
  count("fta", "FTA", "Free-throw attempts", "Free-throw attempts"),
  {
    key: "ft_pct",
    head: "FT%",
    name: "Free-throw %",
    about: "Free throws made per attempt",
    kind: "pct",
    digits: 1,
    value: (t) => ratio(t.ftm, t.fta),
    qualifies: (t) => t.fta >= t.gp,
    floor: "Fewer than 1 free-throw attempt a game",
  },
  {
    key: "efg",
    head: "eFG%",
    name: "Effective FG%",
    about: "Field-goal % with a made three worth one and a half makes",
    kind: "pct",
    digits: 1,
    value: (t) => ratio(fgm(t) + 0.5 * t.fg3m, fga(t)),
    qualifies: (t) => fga(t) >= 3 * t.gp,
    floor: "Fewer than 3 field-goal attempts a game",
  },
  {
    key: "ts",
    head: "TS%",
    name: "True shooting %",
    about: "Points per shooting chance, free throws included: PTS / (2 × (FGA + 0.44 × FTA))",
    kind: "pct",
    digits: 1,
    value: (t) => ratio(t.pts, 2 * (fga(t) + 0.44 * t.fta)),
    qualifies: (t) => fga(t) >= 3 * t.gp,
    floor: "Fewer than 3 field-goal attempts a game",
  },
  {
    key: "usg",
    head: "USG%",
    name: "Usage %",
    about: "Share of his team's shots, free throws and turnovers used while he was on court",
    kind: "pct",
    digits: 1,
    value: (t) => {
      const team = t.tm_fga + 0.44 * t.tm_fta + t.tm_tov;
      return t.sec > 0 && team > 0 ? (100 * (fga(t) + 0.44 * t.fta + t.tov) * t.game_sec) / (t.sec * team) : null;
    },
  },
];

export const BY_KEY = Object.fromEntries(MEASURES.map((m) => [m.key, m]));

export const SETS: Record<ColumnSet, string[]> = {
  box: ["gp", "min", "pts", "reb", "ast", "stl", "blk", "tov", "pir", "pm"],
  shooting: ["fga", "fg_pct", "fg2_pct", "fg3a", "fg3_pct", "fta", "ft_pct", "efg", "ts", "usg"],
};

export const WINDOWS: { key: Window; label: string }[] = [
  { key: "season", label: "Season" },
  { key: "last5", label: "Last 5" },
  { key: "last10", label: "Last 10" },
  { key: "last20", label: "Last 20" },
];
export const RATES: { key: Rate; label: string; long: string }[] = [
  { key: "game", label: "Per game", long: "per game" },
  { key: "36", label: "Per 36", long: "per 36 minutes" },
  { key: "100", label: "Per 100", long: "per 100 possessions" },
];
export const MIN_MINUTES = [0, 10, 15, 20];

export const DEFAULT_VIEW: Omit<View, "season"> = {
  window: "season",
  rate: "game",
  sort: "pts",
  dir: "desc",
  team: "",
  minMin: 10,
  q: "",
  set: "box",
};

export interface Line {
  row: PlayerRow;
  totals: Totals;
  values: Record<string, number | null>;
  ranked: boolean; // qualifies for the sorted measure
}

function totalsOf(fields: string[], values: number[]): Totals {
  const t: Totals = {};
  fields.forEach((f, i) => (t[f] = values[i]));
  return t;
}

export function rated(m: Measure, t: Totals, rate: Rate): number | null {
  const raw = m.value(t);
  if (raw === null || m.kind !== "count") return raw;
  if (rate === "game") return t.gp ? raw / t.gp : null;
  if (rate === "36") return t.sec ? (raw * 2160) / t.sec : null;
  return t.poss ? (raw * 100) / t.poss : null;
}

const fold = (s: string) =>
  s
    .normalize("NFKD")
    .replace(/[̀-ͯ]/g, "")
    .toLowerCase();

/** The table's rows in order: players passing the filters, sorted on the measure (unranked last). */
export function lines(data: SeasonPlayers, view: View): Line[] {
  const measure = BY_KEY[view.sort] ?? BY_KEY.pts;
  const all = data.players.map((row) => {
    const totals = totalsOf(data.fields, row.totals[view.window] ?? row.totals.season);
    const values: Record<string, number | null> = {};
    for (const m of MEASURES) values[m.key] = rated(m, totals, view.rate);
    return { row, totals, values, ranked: true };
  });
  // The games floor keeps cameo seasons out: a quarter of the most games anyone played.
  const most = Math.max(0, ...all.map((l) => l.totals.gp));
  const floorGames = view.window === "season" && view.minMin > 0 ? Math.ceil(most / 4) : 0;
  const q = fold(view.q.trim());
  const kept = all.filter(
    (l) =>
      l.totals.gp >= floorGames &&
      (l.totals.gp ? l.totals.sec / l.totals.gp / 60 : 0) >= view.minMin &&
      (!view.team || l.row.teams.includes(view.team)) &&
      (!q || fold(l.row.name).includes(q)),
  );
  for (const l of kept) l.ranked = l.values[measure.key] !== null && (measure.qualifies?.(l.totals) ?? true);
  const sign = view.dir === "desc" ? -1 : 1;
  return kept.sort((a, b) => {
    if (a.ranked !== b.ranked) return a.ranked ? -1 : 1;
    const av = a.values[measure.key] ?? -Infinity;
    const bv = b.values[measure.key] ?? -Infinity;
    return av === bv ? a.row.name.localeCompare(b.row.name) : sign * (av - bv);
  });
}

export function format(m: Measure, v: number | null): string {
  if (v === null || !Number.isFinite(v)) return "–";
  const text = Math.abs(v).toFixed(m.digits);
  if (m.signed && v > 0) return `+${text}`;
  if (v < 0 && Number(text) !== 0) return `−${text}`;
  return text;
}

export const teamHref = (base: string, code: string, season?: number) =>
  `${base.replace(/\/$/, "")}/teams/${code.toLowerCase()}/${season ? `?season=${season}` : ""}`;
export const playerHref = (base: string, slug: string) => `${base.replace(/\/$/, "")}/players/${slug}/`;

export function measureTitle(view: View): string {
  const m = BY_KEY[view.sort] ?? BY_KEY.pts;
  const rate = RATES.find((r) => r.key === view.rate)!.long;
  return m.kind === "count" ? `${m.name} ${rate}` : m.key === "min" ? "Minutes per game" : m.name;
}

function niceTicks(lo: number, hi: number, count = 6): number[] {
  const span = hi - lo || 1;
  const raw = span / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((s) => s * mag).find((s) => span / s <= count) ?? 10 * mag;
  const ticks: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) ticks.push(Number(v.toFixed(6)));
  return ticks;
}

export const CHART_ROWS = 25;

export interface Chart {
  title: string;
  rows: { line: Line; x: number; text: string }[];
  ticks: { x: number; text: string }[];
  avg: { x: number; text: string; n: number } | null;
}

/** The ranked dot chart: the first 25 ranked rows on one scale, with the listed players' average. */
export function chart(ls: Line[], view: View): Chart {
  const m = BY_KEY[view.sort] ?? BY_KEY.pts;
  const ranked = ls.filter((l) => l.ranked);
  const shown = ranked.slice(0, CHART_ROWS);
  const values = ranked.map((l) => l.values[m.key] as number);
  const mean = values.length ? values.reduce((a, b) => a + b, 0) / values.length : null;
  const inView = [...shown.map((l) => l.values[m.key] as number), ...(mean === null ? [] : [mean])];
  let lo = Math.min(...inView);
  let hi = Math.max(...inView);
  if (!inView.length) [lo, hi] = [0, 1];
  const pad = (hi - lo || Math.abs(hi) || 1) * 0.08;
  lo -= pad;
  hi += pad;
  if (m.kind !== "pct" && !m.signed && lo < 0) lo = 0;
  const x = (v: number) => (100 * (v - lo)) / (hi - lo);
  const top = view.dir === "desc" ? "Highest" : "Lowest";
  return {
    title: `${top} ${Math.min(CHART_ROWS, shown.length)}: ${measureTitle(view)}`,
    rows: shown.map((line) => ({ line, x: x(line.values[m.key] as number), text: format(m, line.values[m.key]) })),
    ticks: niceTicks(lo, hi).map((v) => ({ x: x(v), text: format({ ...m, digits: Number.isInteger(v) ? 0 : 1 }, v) })),
    avg: mean === null ? null : { x: x(mean), text: format(m, mean), n: values.length },
  };
}

/** The tooltip of one chart row: the player, his clubs and the value charted. */
export const slotTip = (row: PlayerRow, c: Chart, text: string) =>
  tip`<b>${row.name}</b> ${row.teams.join(", ")}<br>${c.title.split(": ")[1]}: ${text}`;

/** Chart rows as HTML. Always CHART_ROWS slots, so a change retargets dots instead of rebuilding. */
export function chartHtml(c: Chart, base: string): string {
  const slots = Array.from({ length: CHART_ROWS }, (_, i) => c.rows[i]);
  return slots
    .map((s, i) => {
      if (!s) return `<li class="slot" hidden></li>`;
      const { row } = s.line;
      const body = slotTip(row, c, s.text);
      return `<li class="slot" style="--x:${s.x.toFixed(2)}"><span class="rk">${i + 1}</span><a class="nm" href="${playerHref(base, row.slug)}">${esc(row.name)}</a><span class="track" data-tip="${tipAttr(body)}" tabindex="-1"><span class="pos"><i class="dot"></i></span></span><span class="val">${s.text}</span></li>`;
    })
    .join("");
}

export function ticksHtml(c: Chart): string {
  return c.ticks.map((t) => `<span style="--x:${t.x.toFixed(2)}">${t.text}</span>`).join("");
}

export function gridHtml(c: Chart): string {
  return c.ticks.map((t) => `<i style="--x:${t.x.toFixed(2)}"></i>`).join("");
}

export function headHtml(view: View): string {
  const cols = SETS[view.set].map((key) => {
    const m = BY_KEY[key];
    const on = key === view.sort;
    const sort = on ? (view.dir === "desc" ? "descending" : "ascending") : "none";
    return `<th scope="col" class="num${on ? " on" : ""}" aria-sort="${sort}"><button type="button" data-sort="${key}" data-tip="${tipAttr(tip`<b>${m.name}</b><br>${m.about}`)}">${m.head}</button></th>`;
  });
  return `<tr><th scope="col" class="c-rk"><span class="sr-only">Rank</span></th><th scope="col" class="c-nm">Player</th>${cols.join("")}</tr>`;
}

export function bodyHtml(ls: Line[], view: View, base: string): string {
  if (!ls.length) {
    return `<tr class="empty"><td colspan="${SETS[view.set].length + 2}">No players match. Lower the minutes floor, pick another team, or clear the search.</td></tr>`;
  }
  let rank = 0;
  return ls
    .map((l) => {
      if (l.ranked) rank += 1;
      const cells = SETS[view.set]
        .map((key) => {
          const m = BY_KEY[key];
          const v = l.values[key];
          const short = m.qualifies && !m.qualifies(l.totals);
          const cls = ["num", key === view.sort ? "on" : "", short ? "short" : ""].filter(Boolean).join(" ");
          const floorAttr = short ? ` data-tip="${tipAttr(tip`${m.floor ?? ""}`)}"` : "";
          return `<td class="${cls}"${floorAttr}>${format(m, v)}</td>`;
        })
        .join("");
      return `<tr><td class="c-rk">${l.ranked ? rank : ""}</td><th scope="row" class="c-nm"><a href="${playerHref(base, l.row.slug)}">${esc(l.row.name)}</a><span class="tm">${l.row.teams.map((t) => `<a href="${teamHref(base, t, view.season)}">${esc(t)}</a>`).join(" · ")}</span></th>${cells}</tr>`;
    })
    .join("");
}

/** View <-> query string: only values that differ from the defaults are written. */
export function toQuery(view: View, defaultSeason: number): string {
  const p = new URLSearchParams();
  if (view.season !== defaultSeason) p.set("season", String(view.season));
  for (const [k, v] of Object.entries(DEFAULT_VIEW) as [keyof typeof DEFAULT_VIEW, unknown][]) {
    if (view[k] !== v) p.set(k, String(view[k]));
  }
  const s = p.toString();
  return s ? `?${s}` : "";
}

export function fromQuery(search: string, seasons: number[], defaultSeason: number): View {
  const p = new URLSearchParams(search);
  const pick = <T extends string>(key: string, allowed: readonly T[], fallback: T): T => {
    const v = p.get(key) as T | null;
    return v && allowed.includes(v) ? v : fallback;
  };
  const season = Number(p.get("season"));
  const minMin = Number(p.get("minMin"));
  return {
    season: seasons.includes(season) ? season : defaultSeason,
    window: pick("window", WINDOWS.map((w) => w.key), DEFAULT_VIEW.window),
    rate: pick("rate", RATES.map((r) => r.key), DEFAULT_VIEW.rate),
    sort: pick("sort", MEASURES.map((m) => m.key), DEFAULT_VIEW.sort),
    dir: pick("dir", ["desc", "asc"] as const, DEFAULT_VIEW.dir),
    team: p.get("team") ?? "",
    minMin: MIN_MINUTES.includes(minMin) && p.has("minMin") ? minMin : DEFAULT_VIEW.minMin,
    q: p.get("q") ?? "",
    set: pick("set", ["box", "shooting"] as const, DEFAULT_VIEW.set),
  };
}
