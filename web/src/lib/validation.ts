// Per-season coverage checks written by the pipeline (web/src/data/stats/validation.json).

export interface ValidationFlag {
  code: string;
  severity: "info" | "warn" | "error";
  message: string;
}

export interface ValidationSeason {
  season: number;
  label: string;
  published: boolean;
  games_scheduled: number;
  games_played: number;
  box_missing: number;
  shots_games: number;
  flags: ValidationFlag[];
  special: { kind: string; note: string } | null;
}

export interface Validation {
  /** ISO-8601 UTC time the checks ran. */
  built: string;
  seasons: ValidationSeason[];
}

/** The seasons the site shows, newest first. */
export function publishedSeasons(v: Validation): ValidationSeason[] {
  return v.seasons.filter((s) => s.published).sort((a, b) => b.season - a.season);
}
