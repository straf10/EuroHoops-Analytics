// One season's located attempts for the Shots explorer (fetched on load and on a season change).
import type { APIRoute, GetStaticPaths } from "astro";
import { explorerData, meta } from "../../../lib/stats";

export const getStaticPaths: GetStaticPaths = async () => {
  const m = await meta();
  return (m?.seasons ?? []).map((s) => ({ params: { season: String(s.season) } }));
};

export const GET: APIRoute = async ({ params }) =>
  new Response(JSON.stringify(await explorerData(Number(params.season))), {
    headers: { "Content-Type": "application/json" },
  });
