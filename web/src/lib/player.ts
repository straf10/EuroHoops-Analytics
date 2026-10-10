// Player page renderers: shot chart, distance bands against the league, game log, season lines.
// Pure (types only from ./stats), so the build and the in-page season switch share them.

import { hexChart } from "./court";
import { esc } from "./format";
import { dataTip, tip } from "./tip";
import type { Bands, PlayerData, PlayerSeason } from "./stats";

/** data/player/{slug}.json: each log row carries its game's details (gameFields) after its own
 * fields, and the league's bands are left to data/season/{season}.json, which a season's pages share. */
export type PackedPlayer = Omit<PlayerData, "seasons"> & { seasons: Omit<PlayerSeason, "games" | "leagueBands">[] };

export const packPlayer = (d: PlayerData): PackedPlayer => ({
  ...d,
  seasons: d.seasons.map(({ leagueBands: _, games, log, ...s }) => ({ ...s, log: log.map((row) => [...row, ...games[String(row[0])]]) })),
});

/** The seasons as the renderers read them; each season's leagueBands still to be joined. */
export const unpackPlayer = (p: PackedPlayer): PlayerData => ({
  ...p,
  seasons: p.seasons.map((s) => {
    const n = s.logFields.length;
    return {
      ...s,
      leagueBands: {},
      log: s.log.map((row) => row.slice(0, n)),
      games: Object.fromEntries(s.log.map((row) => [row[0], row.slice(n)])) as PlayerSeason["games"],
    };
  }),
});

export const BAND_LABELS: Record<string, [string, string]> = {
  rim: ["At the rim", "under 1.5 m"],
  short: ["Short", "1.5 to 3 m"],
  mid: ["Mid-range", "3 to 5 m"],
  long2: ["Long two", "5 m and out"],
  three: ["Three", "under 8 m"],
  deep3: ["Deep three", "8 m and out"],
};

export type Totals = Record<string, number>;
export const totalsOf = (fields: string[], values: number[]): Totals =>
  Object.fromEntries(fields.map((f, i) => [f, values[i]]));

const one = (x: number) => x.toFixed(1);
export const pct = (num: number, den: number) => (den > 0 ? `${((100 * num) / den).toFixed(1)}` : "–");
export const perGame = (t: Totals, x: number) => (t.gp ? one(x / t.gp) : "–");
export const ts = (t: Totals) => pct(t.pts, 2 * (t.fg2a + t.fg3a + 0.44 * t.fta));

/** Season-by-season rows: per game, with the career line computed from summed totals. */
export const SEASON_COLS: { head: string; about: string; cell: (t: Totals) => string }[] = [
  { head: "GP", about: "Games played", cell: (t) => String(t.gp) },
  { head: "GS", about: "Games started", cell: (t) => String(t.gs) },
  { head: "MIN", about: "Minutes per game", cell: (t) => perGame(t, t.sec / 60) },
  { head: "PTS", about: "Points per game", cell: (t) => perGame(t, t.pts) },
  { head: "REB", about: "Rebounds per game", cell: (t) => perGame(t, t.oreb + t.dreb) },
  { head: "AST", about: "Assists per game", cell: (t) => perGame(t, t.ast) },
  { head: "STL", about: "Steals per game", cell: (t) => perGame(t, t.stl) },
  { head: "BLK", about: "Blocks per game", cell: (t) => perGame(t, t.blk) },
  { head: "TOV", about: "Turnovers per game", cell: (t) => perGame(t, t.tov) },
  { head: "PIR", about: "Performance Index Rating per game", cell: (t) => perGame(t, t.pir) },
  { head: "FG%", about: "Field goals made per attempt", cell: (t) => pct(t.fg2m + t.fg3m, t.fg2a + t.fg3a) },
  { head: "3P%", about: "Three-pointers made per attempt", cell: (t) => pct(t.fg3m, t.fg3a) },
  { head: "FT%", about: "Free throws made per attempt", cell: (t) => pct(t.ftm, t.fta) },
  { head: "TS%", about: "True shooting: PTS / (2 × (FGA + 0.44 × FTA))", cell: ts },
];

// ---- Shot chart ----------------------------------------------------------------------------

export function shotChart(s: PlayerSeason, radius: number): string {
  const attempts = s.hex.reduce((a, c) => a + c[2], 0);
  return hexChart(s.hex, radius, { label: `${s.label} shot chart: ${attempts} located field-goal attempts` });
}

// ---- Distance bands ------------------------------------------------------------------------

function share(b: Bands, key: string) {
  const total = Object.values(b).reduce((a, [att]) => a + att, 0);
  return total ? (100 * b[key][0]) / total : 0;
}

const signed = (x: number, digits = 1) =>
  `${x > 0 ? "+" : x < 0 ? "−" : ""}${Math.abs(x).toFixed(digits)}`;

/** Share and FG% per distance band against the league. `invert` for shots allowed: the caret
 *  turns orange where opponents shoot worse than the league. */
export function bandRows(b: Bands, league: Bands, up: string, down: string, invert = false): string {
  return Object.keys(BAND_LABELS)
    .map((key) => {
      const [att, made] = b[key] ?? [0, 0];
      const [la, lm] = league[key] ?? [0, 0];
      const [label, range] = BAND_LABELS[key];
      const mine = att ? (100 * made) / att : null;
      const theirs = la ? (100 * lm) / la : null;
      const diff = mine !== null && theirs !== null && att >= 10 ? mine - theirs : null;
      const good = diff !== null && (invert ? diff < 0 : diff > 0);
      const mark =
        diff === null || Math.abs(diff) < 2.5
          ? ""
          : `<span class="car ${good ? "up" : "down"}" aria-hidden="true">${diff > 0 ? up : down}</span>`;
      return `<tr><th scope="row">${label}<small>${range}</small></th><td class="num">${att}</td><td class="num">${share(b, key).toFixed(0)}%</td><td class="num muted">${share(league, key).toFixed(0)}%</td><td class="num">${mine === null ? "–" : mine.toFixed(1)}</td><td class="num muted">${theirs === null ? "–" : theirs.toFixed(1)}</td><td class="num diff">${mark}${diff === null ? "–" : signed(diff)}</td></tr>`;
    })
    .join("");
}

export const bandsHtml = (s: PlayerSeason, up: string, down: string) =>
  bandRows(s.bands, s.leagueBands, up, down);

// ---- Game log ------------------------------------------------------------------------------

export function logHtml(s: PlayerSeason): string {
  const at = (name: string) => s.logFields.indexOf(name);
  const g = (name: string) => s.gameFields.indexOf(name);
  const lines = [...s.log].reverse();
  if (!lines.length) return `<tr class="empty"><td colspan="15">No box-score lines this season.</td></tr>`;
  return lines
    .map((line) => {
      const game = s.games[String(line[at("game_id")])];
      const team = String(line[at("team")]);
      const n = (name: string) => Number(line[at(name)]);
      const [date, round, phase, home, away, hs, as] = [
        game?.[g("date")],
        game?.[g("round")],
        game?.[g("phase")],
        game?.[g("home")],
        game?.[g("away")],
        Number(game?.[g("home_score")]),
        Number(game?.[g("away_score")]),
      ];
      const isHome = home === team;
      const opp = isHome ? away : home;
      const own = isHome ? hs : as;
      const their = isHome ? as : hs;
      const won = own > their;
      const d = new Date(`${date}T12:00:00Z`).toLocaleDateString("en-GB", { day: "numeric", month: "short" });
      const pm = n("pm");
      const phaseText = phase === "RS" ? `Round ${round}` : String(phase);
      return `<tr><td class="date"${dataTip(tip`${date} · ${phaseText}`)}>${d}</td><td class="opp"><span class="at">${isHome ? "vs" : "@"}</span>${esc(String(opp))}</td><td class="res"><span class="wl ${won ? "w" : "l"}">${won ? "W" : "L"}</span>${own}–${their}</td><td class="num">${Math.round(n("sec") / 60)}</td><td class="num strong">${n("pts")}</td><td class="num">${n("oreb") + n("dreb")}</td><td class="num">${n("ast")}</td><td class="num">${n("stl")}</td><td class="num">${n("blk")}</td><td class="num">${n("tov")}</td><td class="num">${n("fg2m") + n("fg3m")}-${n("fg2a") + n("fg3a")}</td><td class="num">${n("fg3m")}-${n("fg3a")}</td><td class="num">${n("ftm")}-${n("fta")}</td><td class="num">${n("pir")}</td><td class="num">${pm > 0 ? "+" : pm < 0 ? "−" : ""}${Math.abs(pm)}</td></tr>`;
    })
    .join("");
}
