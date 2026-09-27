import { defineConfig } from "astro/config";

// Built into ../site, which the daily workflow uploads to GitHub Pages.
export default defineConfig({
  site: "https://straf10.github.io",
  base: "/EuroHoops-Analytics",
  outDir: "../site",
  trailingSlash: "ignore",
  build: { inlineStylesheets: "auto" },
  // Scope component styles with an astro-* class rather than a data-astro-cid-* attribute: the
  // same specificity (one class = one attribute), but ~9 bytes shorter on every scoped element.
  scopedStyleStrategy: "class",
});
