# EuroHoops Analytics

An end-to-end ML system for EuroLeague and Greek Basket League (GBL) games. It ingests
official data automatically, fits leak-free models and publishes **public, append-only logs
of pre-tip-off predictions** for the 2026-27 season ([`predictions/`](predictions/)),
committed daily by GitHub Actions before the games start and shown on a static page. The
current model is a FiveThirtyEight-style Elo baseline; its walk-forward backtests against a
home-win-rate baseline are in [`reports/`](reports/) (`backtest_elo*.json`), next to the live
scorecards and the GBL box-score quality report. Roadmap: [`PLAN.md`](PLAN.md).

## Run it

```bash
uv sync
uv run eurohoops ingest && uv run eurohoops ingest --competition gbl && uv run eurohoops build
uv run eurohoops predict && uv run eurohoops predict --competition gbl
```

`backtest` re-tunes Elo and rewrites the reports, `score` rebuilds a scorecard and `publish`
writes `web/src/data/site.json` (each takes `--competition gbl` where it applies). `ingest --details`
also caches game details: EuroLeague box scores, play-by-play and shots; GBL box scores. The
first GBL ingest reads ~350 ESAKE pages at 2 s each; later runs only refresh unfinished rounds.

## Website

The public page is an [Astro](https://astro.build) static site in [`web/`](web/) that renders
the data `eurohoops publish` exports. The daily workflow builds it into `site/` and deploys it to
GitHub Pages. Locally, after `uv run eurohoops publish`:

```bash
cd web && npm ci
npm run dev      # http://localhost:4321/EuroHoops-Analytics/
npm run build    # writes ../site
```

Design direction and tokens: [`DESIGN.md`](DESIGN.md); product brief: [`PRODUCT.md`](PRODUCT.md).
