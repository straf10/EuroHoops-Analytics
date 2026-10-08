// The API files `eurohoops publish` writes below src/data/api (build time only): the ungated season
// simulation, the live scorecards and a team's four factors. Every file may be absent (a fresh
// checkout, a route that answered 404): each loader then gives null and the page says so.

import codes from "./m6_codes.json";

export interface SimTeam {
  team: string;
  expected_wins: string | number;
  rank_p10: number;
  rank_p50: number;
  rank_p90: number;
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
  gated: boolean;
  gate: { passed: boolean; reason: string };
  teams: SimTeam[];
  p_playoffs: Record<string, number>;
  p_final: Record<string, number>;
}

/** A model's scorecard block: n scored games and the metrics (null while there are none). */
export interface ScoreBlock {
  n: number;
  log_loss: number | null;
  brier: number | null;
  accuracy: number | null;
  margin_mae: number | null;
  ece: number | null;
}

export interface ShadowBlock {
  /** The model's own scores, absent until it has a row. */
  own: ScoreBlock | null;
  /** Its log loss and Elo's on the games both logged. */
  same: { n: number; own: number | null; elo: number | null } | null;
  /** Rows in its log, scored or not. */
  rows: number | null;
}

export interface LiveScorecard {
  competition: string;
  elo: ScoreBlock;
  b0: ScoreBlock;
  m1: ShadowBlock;
  m5: ShadowBlock;
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
const metrics = import.meta.glob<unknown>("../data/api/metrics/live.json", { import: "default" });
const teams = import.meta.glob<unknown>("../data/api/teams/*/*/factors.json", { import: "default" });

const read = async <T>(files: Record<string, Loader>, key: string): Promise<T | null> =>
  files[key] ? ((await files[key]()) as T) : null;

export const simulation = (competition: string) =>
  read<Simulation>(sims, `../data/api/simulations/${competition}.json`);

export const teamFactors = (competition: string, source: string) =>
  read<Factors>(teams, `../data/api/teams/${competition}/${source}/factors.json`);

interface RawShadow {
  rows_in_log?: number;
  same_games_as_elo?: Record<string, number | null>;
  [model: string]: unknown;
}

const shadow = (raw: RawShadow | undefined, model: "m1" | "m5"): ShadowBlock => {
  const own = (raw?.[model] as ScoreBlock | undefined) ?? null;
  const same = raw?.same_games_as_elo;
  return {
    own,
    same: same ? { n: Number(same.n), own: same[`${model}_log_loss`] ?? null, elo: same.elo_log_loss ?? null } : null,
    rows: raw?.rows_in_log ?? null,
  };
};

export async function liveScorecard(competition: string): Promise<LiveScorecard | null> {
  const file = await read<{ competitions: { competition: string; scorecard: Record<string, unknown> }[] }>(
    metrics,
    "../data/api/metrics/live.json",
  );
  const card = file?.competitions.find((c) => c.competition === competition)?.scorecard;
  if (!card) return null;
  return {
    competition,
    elo: card.elo as ScoreBlock,
    b0: card.b0 as ScoreBlock,
    m1: shadow(card.m1 as RawShadow | undefined, "m1"),
    m5: shadow(card.m5 as RawShadow | undefined, "m5"),
  };
}

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
