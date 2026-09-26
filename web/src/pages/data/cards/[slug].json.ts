// One player's seasons and career in the compact shape Compare fetches.
import type { APIRoute, GetStaticPaths } from "astro";
import { cardData, playerIndex } from "../../../lib/stats";

export const getStaticPaths: GetStaticPaths = async () => (await playerIndex()).map((p) => ({ params: { slug: p.slug } }));

export const GET: APIRoute = async ({ params }) =>
  new Response(JSON.stringify(await cardData(params.slug!)), { headers: { "Content-Type": "application/json" } });
