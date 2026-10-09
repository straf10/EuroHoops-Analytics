// The API files `eurohoops publish` writes below src/data/api (build time only): the season
// simulation and a team's four factors. Every file may be absent (a fresh
// checkout, a route that answered 404): each loader then gives null and the page says so.

import codes from "./m6_codes.json";

export interface SimTeam {
  team: string;
  expected_wins: string | number;
  p_direct_playoffs: string | number;
  p_play_in: string | number;
  p_top10: string | number;
  p_final_four: string | number;
  p_title: string | number;
}

export interface Simulation {
  competition: string;
  season: number;
  after_round: number;
  cutoff_utc: string;
  n_sims: number;
  teams: SimTeam[];
  p_playoffs: Record<string, number>;
  p_final: Record<string, number>;
}

export interface Factor {
  value: number | null;
  reason: string | null;
}

export interface Factors {
  season: number;
  games: number;
  factors: Record<"efg_pct" | "tov_pct" | "orb_pct" | "ft_rate", Factor>;
  schedule_strength: Factor & { games?: number; rating?: string };
}

type Loader = () => Promise<unknown>;
const sims = import.meta.glob<unknown>("../data/api/simulations/*.json", { import: "default" });
const teams = import.meta.glob<unknown>("../data/api/teams/*/*/factors.json", { import: "default" });

const read = async <T>(files: Record<string, Loader>, key: string): Promise<T | null> =>
  files[key] ? ((await files[key]()) as T) : null;

export const simulation = (competition: string) =>
  read<Simulation>(sims, `../data/api/simulations/${competition}.json`);

export const teamFactors = (competition: string, source: string) =>
  read<Factors>(teams, `../data/api/teams/${competition}/${source}/factors.json`);

// Display codes (what the site shows) by source code: publish.py's DISPLAY_CODES, the copy
// tests/test_web_m6_codes.py keeps in step. API files use source codes, stats and site files display codes.
const DISPLAY: Record<string, Record<string, string>> = codes;

export const displayCode = (competition: string, source: string) => DISPLAY[competition]?.[source] ?? source;

export const sourceCode = (competition: string, shown: string) =>
  Object.entries(DISPLAY[competition] ?? {}).find(([, d]) => d === shown)?.[0] ?? shown;

// ---- Formatting: one place, so every page rounds alike ----

export const count = (n: number) => n.toLocaleString("en-GB");
export const percent = (x: number, digits = 1) => `${(x * 100).toFixed(digits)}%`;
/** A simulated share: whole percents, "<1%" below one, ">99%" above (it is a count of simulations). */
export const share = (x: number) => (x > 0 && x < 0.005 ? "<1%" : x > 0.995 && x < 1 ? ">99%" : `${Math.round(x * 100)}%`);
export const plural = (n: number, word: string) => `${count(n)} ${word}${n === 1 ? "" : "s"}`;
