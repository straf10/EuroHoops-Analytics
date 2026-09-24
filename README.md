# EuroHoops Analytics

An end-to-end ML system for EuroLeague and Greek Basket League games. It ingests official
data automatically, fits leak-free models and publishes a **public, append-only log of
pre-tip-off predictions** for the 2026-27 season
([`predictions/`](predictions/)), committed daily by GitHub Actions before each game starts.
The current model is a FiveThirtyEight-style Elo baseline; its backtest is in
[`reports/backtest_elo.json`](reports/backtest_elo.json). The roadmap is in [`PLAN.md`](PLAN.md).

## Run it

```bash
uv sync
uv run eurohoops ingest && uv run eurohoops backtest
uv run eurohoops predict && uv run eurohoops score
```
