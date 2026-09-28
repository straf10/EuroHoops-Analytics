# EuroHoops Analytics

[![ci](https://github.com/straf10/EuroHoops-Analytics/actions/workflows/ci.yml/badge.svg)](https://github.com/straf10/EuroHoops-Analytics/actions/workflows/ci.yml)
[![daily predictions](https://github.com/straf10/EuroHoops-Analytics/actions/workflows/daily.yml/badge.svg)](https://github.com/straf10/EuroHoops-Analytics/actions/workflows/daily.yml)

An end-to-end ML system for EuroLeague and Greek Basket League (GBL) games. It ingests official
data automatically, fits leak-free models and publishes **public, append-only logs of
pre-tip-off predictions** for the 2026-27 season ([`predictions/`](predictions/)), committed
daily by GitHub Actions before the games start and shown on a static site.

**Live site:** https://straf10.github.io/EuroHoops-Analytics/

Two models are live, each with its own log: a FiveThirtyEight-style Elo baseline and M1, a
possession-based team efficiency model ([model card](docs/models/m1.md)). A third, M2 (expected
points per shot), feeds the site's shot charts but no live prediction ([model card](docs/models/m2.md)).
Every model's walk-forward backtest and pre-registration record lives in
[`reports/`](reports/) (`backtest_elo*.json`, `backtest_m1*.json`, …), next to the live
scorecards and the GBL box-score quality report.

## Pipeline

```
ingest → build → backtest → predict → score → publish
```

| Stage | Code | Output |
|---|---|---|
| ingest | `src/eurohoops/ingest/` (EuroLeague API, ESAKE.gr HTML) | raw cache (local only) |
| parse | `src/eurohoops/parse/` (games, box scores, shots, possessions, stints, …) | staging Parquet |
| build | `src/eurohoops/marts.py` | `data/marts/eurohoops.duckdb` |
| model | `src/eurohoops/models/` (Elo, M1 team efficiency), `standings.py` | — |
| eval | `src/eurohoops/eval/` (backtests, scorecards) | `reports/*.json` |
| predict | `predict.py` (Elo), `live_m1.py` (M1) | `predictions/*_2026-27.csv` |
| odds | `odds.py` (The Odds API, EuroLeague benchmark) | `odds/*.csv` (append-only) |
| publish | `publish.py` → [`web/`](web/) (Astro) | static site |

`.github/workflows/daily.yml` runs the full pipeline at 08:00 UTC and deploys the site;
`ci.yml` lints, type-checks, tests and builds on every push.

## Run it

```bash
uv sync
uv run eurohoops ingest && uv run eurohoops ingest --competition gbl && uv run eurohoops build
uv run eurohoops predict && uv run eurohoops predict --competition gbl
```

`backtest` re-tunes Elo and rewrites the reports, `score` rebuilds a scorecard and `publish`
writes `web/src/data/site.json` (each takes `--competition gbl` where it applies). `ingest
--details` also caches game details: EuroLeague box scores, play-by-play and shots; GBL box
scores. The first GBL ingest reads ~350 ESAKE pages at 2 s each; later runs only refresh
unfinished rounds.

### Website

The public page is an [Astro](https://astro.build) static site in [`web/`](web/) that renders
the data `eurohoops publish` exports. Locally, after `uv run eurohoops publish`:

```bash
cd web && npm ci
npm run dev      # http://localhost:4321/EuroHoops-Analytics/
npm run build    # writes ../site
```

### Checks

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy src
uv run pytest -q --cov=eurohoops --cov-fail-under=85
uv run vulture src vulture_whitelist.py --min-confidence 60
```

## Documentation

| | |
|---|---|
| [`docs/PLAN.md`](docs/PLAN.md) | Roadmap and current priorities |
| [`docs/PRODUCT.md`](docs/PRODUCT.md) | Product brief: audience, evidence, principles |
| [`docs/DESIGN.md`](docs/DESIGN.md) | Front-end design system ("The Quiet Reference") |
| [`docs/CONTEXT.md`](docs/CONTEXT.md) | Domain glossary |
| [`docs/models/`](docs/models/) | Model cards (Elo, M1, M2) |
| [`docs/data/`](docs/data/) | Data documentation (shots, possessions, stints, odds, GBL) |
| [`docs/history/`](docs/history/) | Build-phase record: prompts, progress and closeout notes cited as the pre-registration evidence in the code and model cards |

## Repository layout

```
src/eurohoops/     pipeline code (ingest, parse, models, eval, predict, stats)
web/               Astro static site
tests/             pytest suite (214 files, ≥85% coverage gate)
docs/              product/design/model docs, data docs, build history
predictions/       append-only pre-registered forecast logs
odds/              append-only market benchmark logs
reports/           backtests, scorecards, QA checklists (pre-registration evidence)
```

`data/`, `site/`, and Astro's build output are generated and gitignored; see
[`.gitignore`](.gitignore).

## License

All rights reserved — see [`LICENSE`](LICENSE). This repository is public for portfolio and
reference purposes; it is not licensed for reuse.
