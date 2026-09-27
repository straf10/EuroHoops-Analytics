// The teams index rows: shared by the build (the open season) and the season switch (the others).
// No imports, so the switch's script stays small enough to inline in the page.

/** One club's line in the teams index: [code, name, W–L, ORtg, DRtg, Net, Pace], as shown. */
export type ClubCells = [string, string, string, string, string, string, string];

const esc = (s: string) => s.replace(/[&<>"]/g, (c) => `&#${c.charCodeAt(0)};`);

/** A season's rows in rank order; links carry the season unless it is the one team pages open on. */
export const clubRowsHtml = (base: string, season: number | undefined, rows: ClubCells[]): string =>
  rows
    .map(
      ([code, name, wl, ortg, drtg, net, pace], i) =>
        `<tr><td class="num muted">${i + 1}</td><th scope="row"><a href="${base}/teams/${code.toLowerCase()}/${season ? `?season=${season}` : ""}"><span class="code">${esc(code)}</span>${esc(name)}</a></th>` +
        `<td class="num">${wl}</td><td class="num wide">${ortg}</td><td class="num wide">${drtg}</td><td class="num strong">${net}</td><td class="num wide">${pace}</td></tr>`,
    )
    .join("");
