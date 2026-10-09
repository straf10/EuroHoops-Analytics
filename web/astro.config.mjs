import { defineConfig } from "astro/config";

const base = "/EuroHoops-Analytics";

// Built into ../site, which the daily workflow uploads to GitHub Pages.
export default defineConfig({
  site: "https://straf10.github.io",
  base,
  outDir: "../site",
  trailingSlash: "ignore",
  // Scouting and Shots are hidden: their code stays in src/pages/_scouting and _shots.
  redirects: {
    "/scouting": `${base}/`,
    "/shots": `${base}/teams/`,
  },
  build: { inlineStylesheets: "auto" },
  // Scope component styles with an astro-* class rather than a data-astro-cid-* attribute: the
  // same specificity (one class = one attribute), but ~9 bytes shorter on every scoped element.
  scopedStyleStrategy: "class",
});
