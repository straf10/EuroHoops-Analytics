// M6 on the site (weeks 16-18): the player card's projection, impact, shot-making and "plays like"
// blocks and the Scouting page. The data is what `eurohoops publish` writes below src/data/api
// (the read-only API's responses, one file per route); every file may be absent.

import codes from "./m6_codes.json";
import { signed } from "./format";
import { playerIndex } from "./stats";

// ---- Data -------------------------------------------------------------------------------------

export interface Band {
  mean: number;
  lo80: number;
  hi80: number;
}

export interface ProjectionRow {
  person_id: string;
  competition: string;
  team: string;
  name: string;
  debut_season: number;
  stats: Record<string, Band>;
  /** Decayed possessions behind the projection. */
  exposure: number;
  n_seasons: number;
  flags: string[];
}

export interface ProjectionReport {
  model: string;
  variant: string;
  half_life: number;
  gated: boolean;
  gate_passed: boolean;
  /** The projected season's first year: 2026 is 2026-27. */
  season: number;
  checkpoint: number;
  stats: string[];
  projections: ProjectionRow[];
}

export interface ImpactSeason {
  season: number;
  o: number;
  d: number;
  total: number;
  sd_total: number;
  ci90_total: [number, number];
  minutes: number;
  games: number;
  seen: boolean;
}

export interface ShotMakingSeason {
  season: number;
  split: string;
  fga: number;
  shot_making: number;
  ci90: [number, number];
}

export interface PlayerIndex {
  person_id: string;
  name: string;
  impact: { model: string; units: string; seasons: ImpactSeason[] } | null;
  shot_making: { model: string; units: string; seasons: ShotMakingSeason[] } | null;
}

export interface SimilarRow {
  rank: number;
  person_id: string;
  competition: string;
  season: number;
  name: string;
  score: number;
  query_season?: number;
}

export interface SimilarReport {
  model: string;
  season: number;
  person_id: string;
  features: Record<string, string[]>;
  pool_seasons: number[];
  similar: SimilarRow[];
}

export interface Dimension {
  stability: number;
  pairs?: number;
  n_min: number;
  labelled?: boolean; // false: the gap does not repeat year to year, so rows carry no label
  k: number;
}

export interface BoardRow {
  person_id: string;
  competition: string;
  season: number;
  dimension: string;
  observed: number;
  expected: number;
  gap: number;
  se: number;
  z: number;
  persist: number;
  expected_next: number;
  n: number;
  label: string | null;
  name: string;
  team: string;
}

export interface Board {
  model: string;
  season: number;
  dimensions: Record<string, Dimension>;
  rows: BoardRow[];
}

export interface UndervaluedRow {
  person_id: string;
  team: string;
  name: string;
  seasons_since_debut: number;
  /** Per game. */
  minutes: number;
  projected_spm: number;
  translated_el: Record<string, number>;
  reason: string;
}

export interface Undervalued {
  model: string;
  variant: string;
  gated: boolean;
  gate_passed: boolean;
  season: number;
  undervalued: UndervaluedRow[];
}

export interface Fit {
  c: number;
  delta: number;
  delta_lo90: number;
  delta_hi90: number;
  n_pairs: number;
  n_persons: number;
}

export interface Translation {
  model: string;
  gate: {
    rule: string;
    variant: string;
    reference: string;
    n_movers: number;
    seasons: number[];
    loss_diff: { mean: number; ci95: [number, number] };
    passed: boolean;
  };
  gate_passed: boolean;
  fits_by_target_season: Record<string, { translate: Record<string, Fit> }>;
  team_offset: { label: string; mean: number; ci95: [number, number] };
}

// Lazy, like lib/stats.ts: a page only parses the files it reads, and no export means no files.
const files = {
  ...import.meta.glob<unknown>("../data/api/players/*/*.json", { import: "default" }),
  ...import.meta.glob<unknown>("../data/api/scouting/*.json", { import: "default" }),
};
const cache = new Map<string, Promise<unknown>>();

async function load<T>(relative: string): Promise<T | null> {
  const key = `../data/api/${relative}`;
  const loader = files[key] as (() => Promise<unknown>) | undefined;
  if (!loader) return null;
  if (!cache.has(key)) cache.set(key, loader());
  return (await cache.get(key)) as T;
}

/** A EuroLeague stats player's file key: source id X is person `P:X`, file `P_X`. */
export const personKey = (id: string): string => `P_${id}`;

export interface PlayerM6 {
  projection: ProjectionReport | null;
  index: PlayerIndex | null;
  similar: SimilarReport | null;
}

export async function playerM6(id: string): Promise<PlayerM6> {
  const dir = `players/${personKey(id)}`;
  const [projection, index, similar] = await Promise.all([
    load<ProjectionReport>(`${dir}/projection.json`),
    load<PlayerIndex>(`${dir}/index.json`),
    load<SimilarReport>(`${dir}/similar.json`),
  ]);
  return { projection, index, similar };
}

let slugs: Promise<Record<string, string>> | null = null;

/** EuroLeague person id (`P:` + source id) -> player page slug, for the players the site has pages for. */
export function playerSlugs(): Promise<Record<string, string>> {
  slugs ??= playerIndex().then((ps) => Object.fromEntries(ps.map((p) => [`P:${p.id}`, p.slug])));
  return slugs;
}

export async function scoutingData() {
  const [board, undervalued, translation] = await Promise.all([
    load<Board>("scouting/board.json"),
    load<Undervalued>("scouting/undervalued.json"),
    load<Translation>("scouting/translation.json"),
  ]);
  return { board, undervalued, translation };
}

// ---- Words and numbers ------------------------------------------------------------------------

/** 2026 -> "2026-27". */
export const seasonLabel = (s: number): string => `${s}-${String((s + 1) % 100).padStart(2, "0")}`;

export const COMPETITIONS: Record<string, string> = { euroleague: "EuroLeague", gbl: "Greek League" };

/** A source team code as the site shows it (publish.py's DISPLAY_CODES; tests keep the copy in step). */
export const displayCode = (competition: string, code: string): string =>
  (codes as Record<string, Record<string, string>>)[competition]?.[code] ?? code;

export const thousands = (x: number): string => Math.round(x).toLocaleString("en-GB");

/** Counts per 100 possessions and shooting percentages: a stat's label, unit and kind. */
export interface StatInfo {
  key: string;
  label: string;
  unit: string;
  pct?: boolean;
}

export const PROJECTED: StatInfo[] = [
  { key: "pts", label: "Points", unit: "per 100 possessions" },
  { key: "oreb", label: "Offensive rebounds", unit: "per 100 possessions" },
  { key: "dreb", label: "Defensive rebounds", unit: "per 100 possessions" },
  { key: "ast", label: "Assists", unit: "per 100 possessions" },
  { key: "tov", label: "Turnovers", unit: "per 100 possessions" },
  { key: "stl", label: "Steals", unit: "per 100 possessions" },
  { key: "blk", label: "Blocks", unit: "per 100 possessions" },
  { key: "fg3a", label: "3-point attempts", unit: "per 100 possessions" },
  { key: "fta", label: "Free-throw attempts", unit: "per 100 possessions" },
  { key: "ts", label: "True shooting", unit: "% of shot value", pct: true },
  { key: "fg3", label: "3-point", unit: "% made", pct: true },
  { key: "ft", label: "Free-throw", unit: "% made", pct: true },
];

export const fixed = (x: number, digits = 1): string => (Math.abs(x) < 0.5 * 10 ** -digits ? 0 : x).toFixed(digits).replace(/^-/, "−");
export const percent = (x: number): string => `${fixed(100 * x)}%`;
export const range = (lo: number, hi: number, f: (x: number) => string = fixed): string => `${f(lo)} to ${f(hi)}`;

/** The projection flags in words (lib/projection.py FLAGS). */
export const FLAGS: Record<string, string> = {
  no_history: "No usable past season: the projection is the league average.",
  no_age: "No aging adjustment was applied.",
  translated: "Part of the history is from the other league and is translated.",
  partial_season: "Includes the part played of the season in progress.",
  no_impact_input: "No impact input: BRAPM leans on its league prior.",
};

/** Where the model stands on its validation gate, in words. */
export function gateWords(r: { model: string; gated: boolean; gate_passed: boolean }): string {
  const name = r.model.toUpperCase();
  if (!r.gated) return `${name} is not gated: no validation gate has been run on it.`;
  return r.gate_passed
    ? `${name} passed its validation gate on past seasons.`
    : `${name} did not pass its validation gate on past seasons: read these as the model's guess, not a checked forecast.`;
}

// ---- The board --------------------------------------------------------------------------------

export interface DimensionInfo {
  key: string;
  title: string;
  about: string;
  /** What `n` counts. */
  nUnit: string;
  /** Observed, expected and gap are fractions shown as percent. */
  pct: boolean;
  unit: string;
}

export const DIMENSIONS: DimensionInfo[] = [
  {
    key: "shot_making",
    title: "Shot-making",
    about: "Points per 100 field-goal attempts above what the shots' difficulty predicts (M2).",
    nUnit: "field-goal attempts",
    pct: false,
    unit: "points per 100 FGA",
  },
  {
    key: "fg3_pct",
    title: "3-point shooting",
    about: "3P% against the 3P% expected from the player's free-throw percentage.",
    nUnit: "3-point attempts",
    pct: true,
    unit: "% (gap in percentage points)",
  },
  {
    key: "on_off",
    title: "On/off",
    about: "Team net rating with him on court minus off court, against his BRAPM rating.",
    nUnit: "possessions on court",
    pct: false,
    unit: "net points per 100 possessions",
  },
];

export const LABEL_WORDS: Record<string, string> = {
  "likely real": "Likely real",
  "likely regression": "Likely regression",
  "too few attempts": "Too few attempts",
  "within noise": "Within noise",
};

export const boardNumber = (d: DimensionInfo, x: number, gap = false): string =>
  d.pct ? (gap ? signed(100 * x) : percent(x)) : gap ? signed(x) : fixed(x);

// ---- Translation ------------------------------------------------------------------------------

export const TRANSLATED: Record<string, string> = Object.fromEntries(PROJECTED.map((s) => [s.key, s.label]));

/** The newest target season's fits, or null. */
export function newestFits(t: Translation): { season: number; fits: Record<string, Fit> } | null {
  const seasons = Object.keys(t.fits_by_target_season).map(Number).sort((a, b) => b - a);
  return seasons.length ? { season: seasons[0], fits: t.fits_by_target_season[String(seasons[0])].translate } : null;
}
