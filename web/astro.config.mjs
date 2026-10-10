import { defineConfig } from "astro/config";

// Built into ../site, which the daily workflow uploads to Cloudflare.
export default defineConfig({
  site: "https://675hoops.com",
  base: "/",
  outDir: "../site",
  trailingSlash: "ignore",
  // Scouting and Shots are hidden: their code stays in src/pages/_scouting and _shots. Their
  // redirects are real 301s in public/_redirects (Cloudflare static assets).
  build: { inlineStylesheets: "auto" },
  // Scope component styles with an astro-* class rather than a data-astro-cid-* attribute: the
  // same specificity (one class = one attribute), but ~9 bytes shorter on every scoped element.
  scopedStyleStrategy: "class",
});
