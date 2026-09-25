import { defineConfig } from "astro/config";

// Built into ../site, which the daily workflow uploads to GitHub Pages.
export default defineConfig({
  site: "https://straf10.github.io",
  base: "/EuroHoops-Analytics",
  outDir: "../site",
  trailingSlash: "ignore",
  build: { inlineStylesheets: "auto" },
});
