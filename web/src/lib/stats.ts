// Stats-site data written by `eurohoops export-stats` into src/data/stats (build time only).
// Every number is a raw total; lib/players.ts derives the rates.

import type { Attempts } from "./shots";

export type Cell = [q: number, r: number, att: number, made: number, pts: number];
export type Bands = Record<string, [att: number, made: number]>;

export interface SeasonMeta {
  season: number;
  label: string;
  live: boolean;
  games: number;
  games_without_box: number;
  games_with_shots: number;
  shots: number;
  shots_unplaced: number;
  coords_validated: boolean;
  players: number;
}

export interface Meta {
  generated_at: string;
  hex_radius_m: number;
  bands: string[];
  windows: string[];
  player_fields: string[];
  team_fields: string[];
  teams: Record<string, string>;
  seasons: SeasonMeta[];
}

export interface PlayerRow {
  id: string;
  name: string;
  slug: string;
  teams: string[];
  dorsal: string;
  totals: Record<string, number[]>; // "season", "last5", "last10", "last20"
  bands: Bands;
}

export interface SeasonPlayers {
  season: number;
  fields: string[];
  bands: string[];
  players: PlayerRow[];
}

export interface TeamRow {
  code: string;
  name: string;
  w: number;
  l: number;
  totals: number[];
  opp: number[];
  bands: Bands;
  bands_allowed: Bands;
}

export interface SeasonTeams {
  season: number;
  fields: string[];
  bands: string[];
  league: { totals: number[]; bands: Bands };
  teams: TeamRow[];
}

export interface SeasonGames {
  season: number;
  game_fields: string[];
  log_fields: string[];
  games: Record<string, [string, number, string, string, string, number, number]>;
  logs: Record<string, (string | number)[][]>;
}

export interface SeasonShots {
  season: number;
  hex_radius_m: number;
  league: Cell[];
  teams: Record<string, { taken: Cell[]; allowed: Cell[] }>;
  players: Record<string, Cell[]>;
}

export interface PlayerIndexRow {
  id: string;
  name: string;
  slug: string;
  seasons: number[];
}

type Kind = "players" | "teams" | "games" | "shots";
type Loader = () => Promise<unknown>;

// Lazy: a page only parses the files it reads. Missing data (no export yet) means no files.
const files = import.meta.glob<unknown>("../data/stats/**/*.json", { import: "default" });
const cache = new Map<string, Promise<unknown>>();

function load<T>(relative: string): Promise<T> | null {
  const key = `../data/stats/${relative}`;
  const loader = files[key] as Loader | undefined;
  if (!loader) return null;
  if (!cache.has(key)) cache.set(key, loader());
  return cache.get(key) as Promise<T>;
}

export const hasStats = () => `../data/stats/meta.json` in files;

export async function meta(): Promise<Meta | null> {
  return (await load<Meta>("meta.json")) ?? null;
}

export async function playerIndex(): Promise<PlayerIndexRow[]> {
  return (await load<{ players: PlayerIndexRow[] }>("players.json"))?.players ?? [];
}

export function season<K extends Kind>(
  year: number,
  kind: K,
): Promise<
  K extends "players"
    ? SeasonPlayers
    : K extends "teams"
      ? SeasonTeams
      : K extends "games"
        ? SeasonGames
        : SeasonShots
> {
  const found = load(`seasons/${year}/${kind}.json`);
  if (!found) throw new Error(`no ${kind} data for ${year}; run eurohoops export-stats`);
  return found as never;
}

/** The season the pages open on: the live one once it has a real sample, else the last full one. */
export function defaultSeason(m: Meta): SeasonMeta {
  const done = [...m.seasons].reverse();
  return done.find((s) => !s.live || s.games >= 30) ?? done[0];
}

// ---- Player pages -------------------------------------------------------------------------

export interface PlayerSeason {
  season: number;
  label: string;
  validated: boolean;
  teams: string[];
  dorsal: string;
  fields: string[];
  totals: number[];
  bands: Bands;
  leagueBands: Bands;
  /** [q, r, att, made, league FG% in the cell x10] per occupied cell */
  hex: [number, number, number, number, number][];
  logFields: string[];
  gameFields: string[];
  log: (string | number)[][];
  games: Record<string, SeasonGames["games"][string]>;
}

export interface PlayerData {
  id: string;
  name: string;
  slug: string;
  seasons: PlayerSeason[];
}

const byId = new Map<number, Map<string, PlayerRow>>();
const leagueCells = new Map<number, Map<string, [number, number]>>();

async function playerRow(year: number, id: string) {
  if (!byId.has(year)) {
    const data = await season(year, "players");
    byId.set(year, new Map(data.players.map((p) => [p.id, p])));
  }
  return { fields: (await season(year, "players")).fields, row: byId.get(year)!.get(id) };
}

async function league(year: number) {
  if (!leagueCells.has(year)) {
    const shots = await season(year, "shots");
    leagueCells.set(year, new Map(shots.league.map(([q, r, att, made]) => [`${q},${r}`, [att, made]])));
  }
  return leagueCells.get(year)!;
}

/** Everything one player page shows, season by season (oldest first). */
export async function playerData(p: PlayerIndexRow): Promise<PlayerData> {
  const m = (await meta())!;
  const labels = new Map(m.seasons.map((s) => [s.season, s]));
  const seasons: PlayerSeason[] = [];
  for (const year of p.seasons) {
    const { fields, row } = await playerRow(year, p.id);
    if (!row) continue;
    const [games, shots, teams, cells] = await Promise.all([
      season(year, "games"),
      season(year, "shots"),
      season(year, "teams"),
      league(year),
    ]);
    const log = games.logs[p.id] ?? [];
    const ids = new Set(log.map((line) => String(line[0])));
    seasons.push({
      season: year,
      label: labels.get(year)?.label ?? String(year),
      validated: labels.get(year)?.coords_validated ?? true,
      teams: row.teams,
      dorsal: row.dorsal,
      fields,
      totals: row.totals.season,
      bands: row.bands,
      leagueBands: teams.league.bands,
      hex: (shots.players[p.id] ?? []).map(([q, r, att, made]) => {
        const [la, lm] = cells.get(`${q},${r}`) ?? [0, 0];
        return [q, r, att, made, la ? Math.round((1000 * lm) / la) : -1];
      }),
      logFields: games.log_fields,
      gameFields: games.game_fields,
      log,
      games: Object.fromEntries(Object.entries(games.games).filter(([id]) => ids.has(id))),
    });
  }
  return { id: p.id, name: p.name, slug: p.slug, seasons };
}

// ---- Team pages ---------------------------------------------------------------------------

export interface TeamIndexRow {
  code: string;
  name: string; // the latest name
  seasons: number[];
}

/** Every club that played a EuroLeague season, by display code (oldest season first). */
export async function teamIndex(): Promise<TeamIndexRow[]> {
  const m = await meta();
  if (!m) return [];
  const clubs = new Map<string, TeamIndexRow>();
  for (const s of m.seasons) {
    for (const t of (await season(s.season, "teams")).teams) {
      const row = clubs.get(t.code) ?? { code: t.code, name: t.name, seasons: [] };
      row.name = t.name;
      row.seasons.push(s.season);
      clubs.set(t.code, row);
    }
  }
  return [...clubs.values()].sort((a, b) => a.code.localeCompare(b.code));
}

export interface ClubLine {
  code: string;
  name: string;
  w: number;
  l: number;
  totals: number[];
  opp: number[];
}

export interface RosterLine {
  id: string;
  name: string;
  slug: string;
  dorsal: string;
  /** summed game-log lines for this club: gp, gs, then the log's number fields */
  totals: number[];
}

export interface TeamSeason {
  code: string;
  season: number;
  label: string;
  validated: boolean;
  name: string;
  w: number;
  l: number;
  fields: string[];
  totals: number[];
  opp: number[];
  league: number[];
  /** every club that season, for the rank strips */
  clubs: ClubLine[];
  bands: Bands;
  bandsAllowed: Bands;
  leagueBands: Bands;
  taken: [number, number, number, number, number][];
  allowed: [number, number, number, number, number][];
  rosterFields: string[];
  roster: RosterLine[];
}

const withLeague = (cells: Cell[], lg: Map<string, [number, number]>) =>
  cells.map(([q, r, att, made]): [number, number, number, number, number] => {
    const [la, lm] = lg.get(`${q},${r}`) ?? [0, 0];
    return [q, r, att, made, la ? Math.round((1000 * lm) / la) : -1];
  });

/** Everything one team page shows for one season. */
export async function teamSeason(code: string, year: number): Promise<TeamSeason | null> {
  const m = (await meta())!;
  const [teams, shots, games, players, cells] = await Promise.all([
    season(year, "teams"),
    season(year, "shots"),
    season(year, "games"),
    season(year, "players"),
    league(year),
  ]);
  const row = teams.teams.find((t) => t.code === code);
  if (!row) return null;
  const info = m.seasons.find((s) => s.season === year);
  const numbers = games.log_fields.filter((f) => f !== "game_id" && f !== "team" && f !== "starter");
  const at = (f: string) => games.log_fields.indexOf(f);
  const people = new Map(players.players.map((p) => [p.id, p]));
  const roster: RosterLine[] = [];
  for (const [id, log] of Object.entries(games.logs)) {
    const lines = log.filter((line) => line[at("team")] === code);
    if (!lines.length) continue;
    const sums = numbers.map((f) => lines.reduce((a, line) => a + Number(line[at(f)]), 0));
    const gs = lines.reduce((a, line) => a + Number(line[at("starter")]), 0);
    const p = people.get(id);
    roster.push({
      id,
      name: p?.name ?? id,
      slug: p?.slug ?? "",
      dorsal: p?.dorsal ?? "",
      totals: [lines.length, gs, ...sums],
    });
  }
  const sec = 2 + numbers.indexOf("sec");
  roster.sort((a, b) => b.totals[sec] - a.totals[sec]);
  const hex = shots.teams[code] ?? { taken: [], allowed: [] };
  return {
    code,
    season: year,
    label: info?.label ?? String(year),
    validated: info?.coords_validated ?? true,
    name: row.name,
    w: row.w,
    l: row.l,
    fields: teams.fields,
    totals: row.totals,
    opp: row.opp,
    league: teams.league.totals,
    clubs: teams.teams.map(({ code, name, w, l, totals, opp }) => ({ code, name, w, l, totals, opp })),
    bands: row.bands,
    bandsAllowed: row.bands_allowed,
    leagueBands: teams.league.bands,
    taken: withLeague(hex.taken, cells),
    allowed: withLeague(hex.allowed, cells),
    rosterFields: ["gp", "gs", ...numbers],
    roster,
  };
}

// ---- Shots explorer -----------------------------------------------------------------------

interface RawAttempts {
  season: number;
  bits: Attempts["bits"];
  players: string[];
  teams: string[];
  x: number[];
  y: number[];
  flags: number[];
  player: number[];
  team: number[];
  opp: number[];
}

/** One season's attempts with the names the explorer's pickers need. */
export async function explorerData(year: number): Promise<Attempts | null> {
  const raw = await load<RawAttempts>(`seasons/${year}/attempts.json`);
  if (!raw) return null;
  const m = (await meta())!;
  const [players, teams] = await Promise.all([season(year, "players"), season(year, "teams")]);
  const info = m.seasons.find((s) => s.season === year);
  const people = new Map(players.players.map((p) => [p.id, p]));
  const clubs = new Map(teams.teams.map((t) => [t.code, t.name]));
  return {
    season: year,
    label: info?.label ?? String(year),
    validated: info?.coords_validated ?? true,
    radius: m.hex_radius_m,
    bits: raw.bits,
    bands: m.bands,
    teams: raw.teams.map((c) => [c, clubs.get(c) ?? m.teams[c] ?? c]),
    players: raw.players.map((id) => {
      const p = people.get(id);
      return [id, p?.name ?? id, p?.slug ?? "", p?.teams[p.teams.length - 1] ?? ""];
    }),
    x: raw.x,
    y: raw.y,
    flags: raw.flags,
    player: raw.player,
    team: raw.team,
    opp: raw.opp,
  };
}
