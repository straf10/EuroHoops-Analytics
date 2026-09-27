// One club's season for the team page's season switch (fetched when the visitor changes season),
// without the season's clubs: those are the same for every club, in data/clubs/{season}.json.
import type { APIRoute, GetStaticPaths } from "astro";
import { teamIndex, teamSeason } from "../../../../lib/stats";

export const getStaticPaths: GetStaticPaths = async () =>
  (await teamIndex()).flatMap((t) =>
    t.seasons.map((s) => ({ params: { code: t.code.toLowerCase(), season: String(s) }, props: { code: t.code, s } })),
  );

export const GET: APIRoute = async ({ props }) => {
  const { code, s } = props as { code: string; s: number };
  const { clubs: _, ...own } = (await teamSeason(code, s))!;
  return new Response(JSON.stringify(own), {
    headers: { "Content-Type": "application/json" },
  });
};
