// Types and formatters of the season forecast record (site.json `forecasts`, built by
// eurohoops.eval.forecasts). Numbers arrive final; this only words them.

import type { Team } from "./format";

export interface FcMetrics {
  n: number;
  picks: number;
  right: number;
  right_pct: number | null;
  log_loss: number | null;
  brier: number | null;
  margin_mae: number | null;
  within: { "5": number | null; "10": number | null };
  totals_mae: number | null;
}

export interface FcGame {
  game_id: string;
  round: number;
  phase: string;
  tipoff_utc: string;
  home: Team;
  away: Team;
  home_score: number;
  away_score: number;
  /** Only the models that logged this game before tip-off. `right` is null where p_home is 0.5. */
  models: Record<string, { p_home: number; exp_margin: number; right: boolean | null }>;
}

export interface Forecasts {
  models: { key: string; label: string; first_round: number | null }[];
  season: ({ model: string } & FcMetrics)[];
  shared: { n: number; rows: ({ model: string } & FcMetrics)[] };
  rounds: { phase: string; round: number | null; games: number; models: Record<string, FcMetrics> }[];
  games: FcGame[];
}

const PHASES: Record<string, string> = {
  RS: "Regular season",
  PI: "Play-in",
  PO: "Playoffs",
  FF: "Final Four",
};

export const phaseName = (phase: string) => PHASES[phase] ?? phase;

/** "Round 3" for the regular season, the phase's name for a postseason group. */
export const groupLabel = (phase: string, round: number | null) =>
  phase === "RS" && round !== null ? `Round ${round}` : phaseName(phase);

/** The value of the games filter for a round or phase group. */
export const groupKey = (phase: string, round: number | null) => (round === null ? phase : `${phase}-${round}`);

export const dash = "–";
export const num = (x: number | null, digits: number) => (x === null ? dash : x.toFixed(digits));
export const share = (x: number | null) => (x === null ? dash : `${Math.round(x * 100)}%`);
export const rightOf = (m: FcMetrics) =>
  m.picks === 0 ? dash : `${m.right} of ${m.picks} (${share(m.right_pct)})`;
