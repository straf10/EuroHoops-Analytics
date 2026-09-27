// Every Shot twins pool player-season (fetched by a player page on its first twin or window switch).
import type { APIRoute } from "astro";
import { twinPool } from "../../lib/stats";

export const GET: APIRoute = async () =>
  new Response(JSON.stringify(await twinPool()), { headers: { "Content-Type": "application/json" } });
