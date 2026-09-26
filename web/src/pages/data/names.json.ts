// Every player's slug, name and first and last season: Compare's search.
import type { APIRoute } from "astro";
import { nameIndex } from "../../lib/stats";

export const GET: APIRoute = async () =>
  new Response(JSON.stringify(await nameIndex()), { headers: { "Content-Type": "application/json" } });
