// One player's seasons for the player page's season switches (fetched on the first switch), packed:
// each log row carries its game's details (gameFields) after its own, and the league's bands come
// from data/season/{season}.json, which every page of that season shares.
import type { APIRoute, GetStaticPaths } from "astro";
import { playerData, playerIndex, type PlayerIndexRow } from "../../../lib/stats";

export const getStaticPaths: GetStaticPaths = async () =>
  (await playerIndex()).map((p) => ({ params: { slug: p.slug }, props: { p } }));

export const GET: APIRoute = async ({ props }) => {
  const data = await playerData((props as { p: PlayerIndexRow }).p);
  const seasons = data.seasons.map(({ leagueBands: _, games, log, ...s }) => ({
    ...s,
    log: log.map((row) => [...row, ...games[String(row[0])]]),
  }));
  return new Response(JSON.stringify({ ...data, seasons }), { headers: { "Content-Type": "application/json" } });
};
