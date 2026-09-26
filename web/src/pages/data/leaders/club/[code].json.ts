// One club's player lines, season by season, for the Leaders page's club views.
import type { APIRoute, GetStaticPaths } from "astro";
import { clubCodes, clubPool } from "../../../../lib/stats";

export const getStaticPaths: GetStaticPaths = async () =>
  (await clubCodes()).map((code) => ({ params: { code: code.toLowerCase() }, props: { code } }));

export const GET: APIRoute = async ({ props }) =>
  new Response(JSON.stringify(await clubPool((props as { code: string }).code)), {
    headers: { "Content-Type": "application/json" },
  });
