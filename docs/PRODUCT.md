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
- Stats site (user decision 2026-09-26, ahead of PLAN §9's week-16 UI slot): EuroLeague-only stats pages modelled on boxscorelab.com and databallr.com. Stats-first home; Players dashboard (any metric, season or last 5/10/20 games, per-game/per-36/per-100); player pages with season history, game log, shot chart and Shot Profile Twin (last X games matched against every EuroLeague player-season since 2007-08); team pages with shots taken and allowed; Leaders; Compare; Shots explorer. All 20 seasons (2007-08 to today). Forecasts move to their own page and keep GBL.
- Season simulation is shown on a Standings page as "not gated": M7 failed its validation gate, so its numbers never enter the pre-registered record (owner decision 2026-10-07). Still out of scope: odds on the site, tracking data (none exists).
- Games whose prediction reached the public log after tip-off are shown but flagged and not scored.
- Models today: MOV-adjusted Elo (FiveThirtyEight-style) publishes the forecasts; M1 and M5 run as shadow logs. Player models are built: xPTS shot-making (M2), RAPM/SPM impact (M3), league translation (M4), and player projections with an over/under board and comparables (M6, gate passed). Every model number on the site shows its interval and sample.

## Brand Commitments

Name: EuroHoops Analytics. No gambling framing (no odds-style presentation, no "picks" hype).

## Evidence on Hand

- `predictions/*_2026-27.csv` (pre-registered log), `reports/live_scorecard*.json`, `reports/backtest_elo*.json` (19 EuroLeague seasons 2007-2026, GBL backtest), DuckDB marts with games and teams.
- No team logos or player photos are licensed; do not use them. No testimonials or press.
- Club colours: one bounded exception (owner decision, 2026-09-26). Leaders and Compare draw players as generic SVG jersey backs in their club's two colours with surname and number; no crests, sponsors, kit copies or likenesses. Everywhere else colour stays data-only (docs/DESIGN.md, Club-Colour Exception).

## Product Principles

1. Show the evidence, not the claim: every number traces to a committed log or report.
2. Honest by default: small samples, misses and unscorable games are visible.
3. Readable in one minute by a non-specialist, deep enough for a specialist.
4. Benchmark, not bet.

## Accessibility & Inclusion

Works at 390 px phone width; light and dark themes; color is never the only carrier of hit/miss.
