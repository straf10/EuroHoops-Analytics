// One club's season for the team page's season switch (fetched when the visitor changes season),
// without what every club shares that season (its clubs, the league's bands: data/season/{season}.json).
import type { APIRoute, GetStaticPaths } from "astro";
import { teamIndex, teamOwn } from "../../../../lib/stats";

export const getStaticPaths: GetStaticPaths = async () =>
  (await teamIndex()).flatMap((t) =>
    t.seasons.map((s) => ({ params: { code: t.code.toLowerCase(), season: String(s) }, props: { code: t.code, s } })),
  );

export const GET: APIRoute = async ({ props }) => {
  const { code, s } = props as { code: string; s: number };
  return new Response(JSON.stringify(await teamOwn(code, s)), {
    headers: { "Content-Type": "application/json" },
  });
};
