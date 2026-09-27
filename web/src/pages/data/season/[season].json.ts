// What every club and player page of one season shares, for their season switches: the season's
// clubs (the rank strips) and the league's bands. One file per season, not a copy in each club's
// and player's data file.
import type { APIRoute, GetStaticPaths } from "astro";
import { meta, seasonShared } from "../../../lib/stats";

export const getStaticPaths: GetStaticPaths = async () =>
  ((await meta())?.seasons ?? []).map((s) => ({ params: { season: String(s.season) } }));

export const GET: APIRoute = async ({ params }) =>
  new Response(JSON.stringify(await seasonShared(Number(params.season))), {
    headers: { "Content-Type": "application/json" },
  });
