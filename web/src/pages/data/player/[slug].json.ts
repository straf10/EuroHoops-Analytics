// One player's seasons for the player page's season switches (fetched on the first switch).
import type { APIRoute, GetStaticPaths } from "astro";
import { playerData, playerIndex, type PlayerIndexRow } from "../../../lib/stats";

export const getStaticPaths: GetStaticPaths = async () =>
  (await playerIndex()).map((p) => ({ params: { slug: p.slug }, props: { p } }));

export const GET: APIRoute = async ({ props }) =>
  new Response(JSON.stringify(await playerData((props as { p: PlayerIndexRow }).p)), {
    headers: { "Content-Type": "application/json" },
  });
