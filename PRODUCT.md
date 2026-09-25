# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Astro static site (user choice, 2026-09-25). Python pipeline stays the source of truth: `eurohoops publish` exports JSON; Astro renders it at build time; GitHub Pages deploys the output daily from `.github/workflows/daily.yml`.

## Users

Recruiter-first, fan-usable. Primary: hiring leads for ML-engineer and sports-analyst roles skimming a portfolio project; they need to see rigor and craft within a minute. Secondary: EuroLeague / Greek Basket League fans checking today's games and their team's win probability.

## Product Purpose

EuroHoops forecasts every EuroLeague and GBL game of 2026-27 and commits each forecast to a public, append-only log before tip-off, then scores it against a home-win baseline. Success: a visitor can see the upcoming forecasts, how past forecasts fared, and why the numbers can be trusted.

## Positioning

Pre-registered, out-of-sample predictions for a whole season: forecasts cannot be tuned after the fact. Failures are reported as openly as wins. It is a model benchmark, never betting advice.

## Operating Context

Pipeline ingest → build → backtest → predict → score → publish runs daily at 08:00 UTC. The page updates once a day. Tip-off times are shown in Athens time. Code and logs live at github.com/straf10/EuroHoops-Analytics.

## Capabilities and Constraints

- v1 scope: live predictions (upcoming games: P(home), expected margin), recent results with hit/miss, scorecard (log loss, Brier, accuracy, margin MAE; Elo vs baseline), team power ratings (current Elo per team, trend).
- Out of scope for v1: shot charts, season simulation, player models, odds.
- Games whose prediction reached the public log after tip-off are shown but flagged and not scored.
- Model today: MOV-adjusted Elo (FiveThirtyEight-style). Player models (xPTS, RAPM) are planned, not built.

## Brand Commitments

Name: EuroHoops Analytics. No gambling framing (no odds-style presentation, no "picks" hype).

## Evidence on Hand

- `predictions/*_2026-27.csv` (pre-registered log), `reports/live_scorecard*.json`, `reports/backtest_elo*.json` (19 EuroLeague seasons 2007-2026, GBL backtest), DuckDB marts with games and teams.
- No team logos or player photos are licensed; do not use them. No testimonials or press.

## Product Principles

1. Show the evidence, not the claim: every number traces to a committed log or report.
2. Honest by default: small samples, misses and unscorable games are visible.
3. Readable in one minute by a non-specialist, deep enough for a specialist.
4. Benchmark, not bet.

## Accessibility & Inclusion

Works at 390 px phone width; light and dark themes; color is never the only carrier of hit/miss.
