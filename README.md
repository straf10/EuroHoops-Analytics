# EuroHoops Analytics

An end-to-end ML system for EuroLeague and Greek Basket League games. It ingests official
data automatically, fits leak-free models and publishes a **public, append-only log of
pre-tip-off predictions** for the 2026-27 season ([`predictions/`](predictions/)), committed
daily by GitHub Actions before the games start. The current model is a FiveThirtyEight-style
Elo baseline; its walk-forward backtest against a home-win-rate baseline is in
[`reports/backtest_elo.json`](reports/backtest_elo.json) and the live scorecard in
[`reports/live_scorecard.json`](reports/live_scorecard.json). Roadmap: [`PLAN.md`](PLAN.md).

## Run it

```bash
uv sync
uv run eurohoops ingest    # schedules + results -> data/staging/euroleague_games.parquet
uv run eurohoops predict   # append upcoming games (next 36 h) to the prediction log
```

`eurohoops backtest` re-tunes Elo and rewrites the report; `eurohoops score` rebuilds the
scorecard; `eurohoops ingest --details` also caches box scores, play-by-play and shots.
