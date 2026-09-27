// Every club of one season, for the team page's rank strips after a season switch (one file per
// season rather than a copy in each club's data/team file).
import type { APIRoute, GetStaticPaths } from "astro";
import { meta, seasonClubs } from "../../../lib/stats";

export const getStaticPaths: GetStaticPaths = async () =>
  ((await meta())?.seasons ?? []).map((s) => ({ params: { season: String(s.season) } }));

export const GET: APIRoute = async ({ params }) =>
  new Response(JSON.stringify(await seasonClubs(Number(params.season))), {
    headers: { "Content-Type": "application/json" },
  });
