// Shot Profile Twin (player page): counts from twins.json -> the mirror staff's bars and figures.
// Pure (no data imports), so the build and the in-page twin and window switch share it.

/** Zone -> [label, phone label, where on the court]. Left and right are as drawn on the shot chart. */
export const ZONE_LABELS: Record<string, [string, string, string]> = {
  rim: ["At the rim", "Rim", "under 1.5 m"],
  short_l: ["Short, left", "Short L", "1.5 to 3 m, wider than 45° to the left"],
  short_c: ["Short, middle", "Short M", "1.5 to 3 m, within 45° of the middle"],
  short_r: ["Short, right", "Short R", "1.5 to 3 m, wider than 45° to the right"],
  mid_l: ["Mid-range, left", "Mid L", "3 to 5 m, wider than 45° to the left"],
  mid_c: ["Mid-range, middle", "Mid M", "3 to 5 m, within 45° of the middle"],
  mid_r: ["Mid-range, right", "Mid R", "3 to 5 m, wider than 45° to the right"],
  long2_l: ["Long two, left", "Long 2 L", "5 m and out, wider than 45° to the left"],
  long2_c: ["Long two, middle", "Long 2 M", "5 m and out, within 45° of the middle"],
  long2_r: ["Long two, right", "Long 2 R", "5 m and out, wider than 45° to the right"],
  corner3_l: ["Left corner three", "Corner 3 L", "a three from the left corner"],
  three_c: ["Above the break", "Above break", "a three from above the break, under 8 m"],
  corner3_r: ["Right corner three", "Corner 3 R", "a three from the right corner"],
  deep3: ["Deep three", "Deep 3", "a three from 8 m and out"],
};

export const RATE_LABELS: [string, string][] = [
  ["3PA rate", "Three-point attempts per field-goal attempt"],
  ["FT rate", "Free-throw attempts per field-goal attempt"],
];

export interface Profile {
  att: number;
  zones: number[]; // share of located attempts, 0-1
  bands: { att: number; fg: number | null; diff: number | null }[]; // FG% and points above the league
  rates: (number | null)[]; // 3PA rate, FT rate (0-1)
}

export function profile(fields: string[], zones: string[], bands: string[], counts: number[]): Profile {
  const at = (name: string) => counts[fields.indexOf(name)] ?? 0;
  const zoneAtt = zones.map((z) => at(`att_${z}`));
  const att = zoneAtt.reduce((a, b) => a + b, 0);
  const inBand = (b: string) =>
    zones.reduce((sum, z, i) => sum + ((z.startsWith("corner3") ? "three" : z.split("_")[0]) === b ? zoneAtt[i] : 0), 0);
  const fga = at("fga");
  return {
    att,
    zones: zoneAtt.map((n) => (att ? n / att : 0)),
    bands: bands.map((b) => {
      const n = inBand(b);
      const made = at(`made_${b}`);
      const expected = at(`exp_${b}`);
      return {
        att: n,
        fg: n ? (100 * made) / n : null,
        diff: n >= 10 ? (100 * (made - expected)) / n : null, // the band table's floor
      };
    }),
    rates: [fga ? at("fg3a") / fga : null, fga ? at("fta") / fga : null],
  };
}

/** A shared axis top: the largest value rounded up to the next 10%. */
export const axisTop = (values: number[]) => Math.max(0.1, Math.ceil(Math.max(0, ...values) * 10 - 1e-9) / 10);

const signed = (x: number) => `${x > 0 ? "+" : x < 0 ? "−" : ""}${Math.abs(x).toFixed(1)}`;

/** FG% with a caret when he beats (or trails) the league by 2.5 points or more, from 10 attempts. */
export function fgHtml(b: Profile["bands"][number], up: string, down: string): string {
  if (b.fg === null) return `<span class="none">no shots</span>`;
  const fg = `<span class="fig">${b.fg.toFixed(1)}%</span>`;
  if (b.diff === null) return `${fg}<span class="d">few shots</span>`;
  const mark =
    Math.abs(b.diff) < 2.5
      ? ""
      : `<span class="car ${b.diff > 0 ? "up" : "down"}" aria-hidden="true">${b.diff > 0 ? up : down}</span>`;
  return `${fg}<span class="d">${mark}${signed(b.diff)}</span>`;
}

const percent = (x: number) => (x > 0 && x < 0.005 ? "<1%" : `${Math.round(100 * x)}%`);
export const shareText = percent;
export const rateText = (x: number | null) => (x === null ? "–" : percent(x));
