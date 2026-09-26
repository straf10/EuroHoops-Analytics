// Season player rows for the Players page's filters, fetched when a visitor changes season.
import type { APIRoute, GetStaticPaths } from "astro";
import { meta, season } from "../../../lib/stats";

export const getStaticPaths: GetStaticPaths = async () => {
  const m = await meta();
  return (m?.seasons ?? []).map((s) => ({ params: { season: String(s.season) } }));
};

export const GET: APIRoute = async ({ params }) =>
  new Response(JSON.stringify(await season(Number(params.season), "players")), {
    headers: { "Content-Type": "application/json" },
  });
