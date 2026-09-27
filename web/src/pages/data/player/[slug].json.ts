// One player's seasons for the player page's season switches (fetched on the first switch), packed
// (packPlayer in lib/player.ts).
import type { APIRoute, GetStaticPaths } from "astro";
import { packPlayer } from "../../../lib/player";
import { playerData, playerIndex, type PlayerIndexRow } from "../../../lib/stats";

export const getStaticPaths: GetStaticPaths = async () =>
  (await playerIndex()).map((p) => ({ params: { slug: p.slug }, props: { p } }));

export const GET: APIRoute = async ({ props }) => {
  const data = await playerData((props as { p: PlayerIndexRow }).p);
  return new Response(JSON.stringify(packPlayer(data)), { headers: { "Content-Type": "application/json" } });
};
