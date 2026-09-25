export interface Team {
  code: string;
  name: string;
}

export interface Game {
  game_id: string;
  round: number;
  tipoff_utc: string;
  predicted_at_utc: string;
  home: Team;
  away: Team;
  p_home: number;
  exp_margin: number;
}

export interface Result extends Game {
  home_score: number;
  away_score: number;
  hit: boolean;
  provable: boolean;
}

export interface Metrics {
  n: number;
  log_loss: number | null;
  brier: number | null;
  accuracy: number | null;
  margin_mae: number | null;
}

export interface Rating extends Team {
  rating: number;
  change: number;
}

export interface Competition {
  key: string;
  title: string;
  season: string;
  logged: number;
  next_tipoff_utc: string | null;
  upcoming: Game[];
  results: Result[];
  scorecard: { elo: Metrics; b0: Metrics; not_provable: number };
  backtest: {
    tuning_seasons: number[];
    test_seasons: number[];
    params: { k: number; hca: number; reversion: number };
    test: { elo: Metrics; b0: Metrics };
    log_loss_diff: { mean: number; ci95: [number, number] } | null;
  };
  ratings: Rating[];
}

export interface SiteData {
  generated_at_utc: string;
  competitions: Competition[];
}

const ATHENS = "Europe/Athens";
const day = new Intl.DateTimeFormat("en-GB", {
  timeZone: ATHENS,
  weekday: "short",
  day: "numeric",
  month: "short",
});
const time = new Intl.DateTimeFormat("en-GB", {
  timeZone: ATHENS,
  hour: "2-digit",
  minute: "2-digit",
  hour12: false,
});

export const athensDay = (iso: string) => day.format(new Date(iso));
export const athensTime = (iso: string) => time.format(new Date(iso));

export const pct = (p: number) => `${Math.round(p * 100)}%`;

/** The favourite and its win chance, as the model called it. */
export function call(g: Game): { team: Team; p: number } {
  return g.p_home >= 0.5 ? { team: g.home, p: g.p_home } : { team: g.away, p: 1 - g.p_home };
}

export const signed = (x: number, digits = 1) =>
  `${x > 0 ? "+" : x < 0 ? "−" : ""}${Math.abs(x).toFixed(digits)}`;

export const seasonLabel = (s: number) => `${s}-${String((s + 1) % 100).padStart(2, "0")}`;

export function seasonSpan(seasons: number[]): string {
  const first = seasons[0];
  const last = seasons[seasons.length - 1];
  return first === last ? seasonLabel(first) : `${seasonLabel(first)} to ${seasonLabel(last)}`;
}
