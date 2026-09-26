// Leaders' pools: every season, whole careers, and the best single seasons (fetched on a switch).
import type { APIRoute, GetStaticPaths } from "astro";
import { leaderPool, meta } from "../../../lib/stats";

export const getStaticPaths: GetStaticPaths = async () => {
  const m = await meta();
  return ["career", "best", ...(m?.seasons ?? []).map((s) => String(s.season))].map((pool) => ({ params: { pool } }));
};

export const GET: APIRoute = async ({ params }) => {
  const key = params.pool === "career" || params.pool === "best" ? params.pool : Number(params.pool);
  return new Response(JSON.stringify(await leaderPool(key)), { headers: { "Content-Type": "application/json" } });
};
