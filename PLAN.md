# EuroHoops Analytics — Project Plan
*Greek Basket League (GBL) + EuroLeague analytics & prediction platform*
*Plan version 1.0 — 2026-09-24 · Target roles: ML Engineer + Sports Analyst*

---

## 0. One-paragraph pitch (what the finished project says about you)

An end-to-end, production-style ML system that ingests official EuroLeague and Greek Basket League data automatically, builds **player-level models** (shot quality, regularized adjusted plus-minus, cross-league translation, projections), composes them into **calibrated game and season forecasts**, and proves them with a **public, timestamped, out-of-sample prediction log for the full 2026-27 season** — served through an API and a clean website, with a written methodology that reports failures as honestly as wins.

The differentiators vs. the typical portfolio project, in priority order:
1. **Live, pre-registered predictions** for a whole season (cannot be overfit after the fact).
2. **Player modelling depth** (xPTS + RAPM + league translation), not just a team classifier.
3. **Leak-free evaluation** with proper scoring rules, calibration, baselines and significance tests.
4. **Real engineering**: scheduled pipelines, data contracts, tests, experiment tracking, deployed API.

---

## 1. Verified data inventory (checked 2026-09-24)

### 1.1 EuroLeague — official public API (no key)  ⭐ primary modelling dataset
| Endpoint (verified live) | Content | Coverage |
|---|---|---|
| `live.euroleague.net/api/PlaybyPlay?gamecode=N&seasoncode=E{YYYY}` | Full PBP incl. **substitutions (`IN`/`OUT`)**, shots, fouls, TOs, rebounds, assists, timeouts, game clock, running score | **2007-08 → today** (2000–2006 return empty; spot-checked game 1 of 6 seasons) |
| `live.euroleague.net/api/Points?...` | Every shot with **`COORD_X`/`COORD_Y`**, `ZONE`, `FASTBREAK`, `SECOND_CHANCE`, `POINTS_OFF_TURNOVER`, clock, score, UTC timestamp | 2007-08 → today |
| `live.euroleague.net/api/Boxscore?...` | Player/team box scores | 2007-08 → today |
| `api-live.euroleague.net/v2/competitions/E/seasons/E{YYYY}/games` (+ `/games/{n}/stats`) | Schedule, results, rosters, player bios (height, DOB, nationality) | Swagger at `api-live.euroleague.net/swagger` |

- 2026-27: **20 teams, 38 regular-season rounds, 380 games, 24 Sep 2026 → 16 Apr 2027**, then play-in/playoffs/Final Four.
- Python wrapper: `euroleague-api` (PyPI, maintained). Use it for exploration; own ingestion code in production so you control retries, caching and schema.
- Competition code `U` = EuroCup (same API) — optional extra data for league translation.
- ⚠️ The API is public but **undocumented/unofficial for third parties** → cache every raw response; it can change without notice.

### 1.2 Greek Basket League — ESAKE (esake.gr), official site, HTML
| Page (verified) | Content |
|---|---|
| `/el/action/EsakeResults?idchampionship={SEASON_ID}&idseason={PHASE}&series={ROUND}` | Fixtures/results per round → game IDs (hex, e.g. `D6867DA7`) |
| `/el/action/EsakegameView?idgame={ID}&mode=3` | **Full box score**: PTS, 2PM-A, 3PM-A, FTM-A, OREB/DREB, AST, BLK, BLK-against, fouls committed/drawn, STL, TO, minutes, PIR |
| `/el/action/EsakegameView?idgame={ID}&mode=2` | **Play-by-play**, rendered client-side by a BasketHotel (Genius Sports) widget: game ID hex→decimal, `GAME_FULL_VIEW_WIDGET`, `show_tabs=["play_by_play"]`, served from `widgets.baskethotel.com/widget-service/show?…`, has an export option |
| `EsakeplayerView`, `EsaketeamView`, `EsakeRanking`, `EsakeStats` | Player/team pages, standings, leaders |

- Season dropdown goes back to **1992-93**; season IDs are hex (2025-26 = `44B80BEB`, 2026-27 = `184645B9`). Box-score depth for old seasons **not yet verified**.
- 2025-26 had a **two-phase format** (Α Φάση regular season, Β Φάση) → the simulator must encode the format per season.
- GBL 2026-27 starts after the Super Cup (26–27 Sep).
- No robots.txt (404); still: throttle (≥2 s between requests), cache raw HTML, identify your scraper, never re-scrape what you already have.
- ⚠️ **Mixed scripts in names inside the same box score**: e.g. `ΓΚAΟΥΝΤΛΟΚ AΝΤΡΙΟΥ` (Andrew Goudelock, Greek transliteration) next to `PETROPOULOS ANDREAS` (Latin). EuroLeague uses `SURNAME, NAME`. → Entity resolution is a real sub-project (§4.3).
- ❓ Unverified: whether GBL PBP includes substitutions (needed for GBL RAPM) and shot coordinates (probably not). **Week-1 spike.**

### 1.3 Betting odds (benchmark only)
- **No verified free, clean historical EuroLeague/GBL odds source.** Options: OddsPortal archives (JS-heavy scraping, ToS grey), The Odds API historical (paid). 
- Plan: (a) spike 1 day on OddsPortal for EuroLeague closing moneyline/spread/total; (b) if not feasible, **benchmark against Elo + naive baselines**, and start **recording current odds yourself** each round from now on (a forward-only dataset). Source: The Odds API lists **EuroLeague (`basketball_euroleague`) but not the GBL** (checked 2026-09-24); free-tier quota to confirm at signup. GBL gets no market benchmark unless a manual/scraped source is found.
- Positioning: odds are a *benchmark of market efficiency*, not a betting product. Keep it that way on the site (recruiters at clubs care; gambling framing hurts).

### 1.4 Not usable / out of scope
- Tracking data (SportVU-style) — not public for Europe → no EPV (Cervone et al. 2016) or defensive-tracking models.
- Pre-game injury/availability feeds — none public → affects the game predictor design (§5.6).

---

## 2. Research foundation — what the literature says, and what we take from it

| # | Paper | What it establishes | How we use it |
|---|---|---|---|
| R1 | Kubatko, Oliver, Pelton & Rosenbaum (2007), *A Starting Point for Analyzing Basketball Statistics*, JQAS 3(3) | Possession framework; equal possessions per game; standard formulas | Possession/pace/ORtg/DRtg definitions; stint possession counting |
| R2 | Oliver (2004), *Basketball on Paper* | Four Factors (eFG%, TOV%, ORB%, FTr) | Team-analysis page + interpretable features |
| R3 | Glickman & Stern (1998), *A State-Space Model for NFL Scores*, JASA 93 | Team strengths as latent states evolving over time (Kalman) | Team rating model v2 (dynamic strengths) |
| R4 | Manner (2016), *Modeling and Forecasting the Outcomes of NBA Basketball Games*, JQAS 12(1) | Margin-based model + dynamic state-space strengths; weak evidence for heteroscedasticity; time variation shows mainly across seasons | Margin → win-prob mapping; test game-specific variance |
| R5 | FiveThirtyEight NBA Elo methodology | MOV-adjusted Elo, K-factor, home advantage, season-start mean reversion | **Baseline #1** |
| R6 | Sill (2010), *Improved NBA Adjusted +/- Using Regularization and Out-of-Sample Testing*, MIT Sloan | Ridge-regularized APM ~doubles out-of-sample accuracy; evaluate player metrics by predicting future games | RAPM + **the evaluation protocol for player metrics** |
| R7 | Deshpande & Jensen (2016), *Estimating an NBA Player's Impact on His Team's Chances of Winning*, JQAS 12(2) | Bayesian regression on win-probability changes, posterior uncertainty for player rankings | Bayesian RAPM variant + uncertainty intervals on player cards |
| R8 | Watts, Pipping-Gamón & Wyner (2026), *Dummy RAPM*, arXiv:2608.19454 | Dummy indicators for excluded low-minute players reduce held-out margin RMSE (small but consistent, 13/16 seasons) | Low-minute player handling in RAPM (very relevant for small GBL samples) |
| R9 | Chang et al. (2014), *Quantifying Shot Quality in the NBA*, MIT Sloan | eFG% conflates shot quality and shot-making → ESQ (expected) and EFG+ (above expected) | **xPTS model** and shot-making-over-expected metric |
| R10 | Franks, D'Amour, Cervone & Bornn (2016), *Meta-Analytics*, JQAS 12(4) | Judge metrics by **stability, discrimination, independence** | Decide which "over/underperformance" signals are skill vs noise |
| R11 | Efron & Morris (1975), *Data Analysis Using Stein's Estimator*, JASA 70 | Shrinkage beats raw averages for small samples | Empirical-Bayes shrinkage of shooting %, xPTS residuals |
| R12 | Vaci et al. (2019), *Large data and Bayesian modeling — aging curves of NBA players*, Behavior Research Methods 51 | Hierarchical Bayesian aging curves | Player projections (age effect) |
| R13 | Terner & Franks (2021), *Modeling Player and Team Performance in Basketball*, Annu. Rev. Stat. Appl. 8 | Survey: team strategy, player value, shot models, production curves | Reading map; cite in write-up |
| R14 | Walsh & Joshi (2024), *ML for Sports Betting: Accuracy or Calibration?*, MLWA 16 (+ 2025 corrigendum) | Selecting models by **calibration** beats selecting by accuracy; corrigendum fixed feature-engineering bugs | Model selection by log-loss/calibration; **lesson: unit-test the feature pipeline** |
| R15 | Hubáček, Šourek & Železný (2019), *Exploiting Sports-Betting Market Using ML*, IJF 35(2) | Beating accuracy ≠ beating the market; decorrelate from bookmaker | Framing of market benchmark |
| R16 | Giasemidis (2020), *Descriptive and Predictive Analysis of EuroLeague Games and the Wisdom of Basketball Crowds*, arXiv:2002.08465 | ML on EuroLeague 2016–19: **≤67% test accuracy**; crowd predictions beat the models | Realistic accuracy ceiling for EuroLeague; crowd as benchmark |
| R17 | Lampis, Ntzoufras, Vassalos & Dimitriou (2023), *Predictions of European Basketball Match Results with ML*, J. Sports Analytics | 5,214 games 2013–18 incl. **GBL**, EuroLeague, EuroCup, ACB | Closest prior work for GBL; compare against it |
| R18 | Ballı & Özdemir (2021), EuroLeague prediction w/ hybrid features, Chaos Solitons & Fractals | Reports **98.9% accuracy** | **Cautionary example**: pre-game prediction cannot plausibly reach this (ceiling ≈ 67–70%); such numbers indicate features from the game being predicted. Your write-up should explicitly explain why yours is lower and honest |
| R19 | German Basketball Analytics, *Ranking all Major European Basketball Leagues* (Substack) | Team-rating gaps between competitions (e.g. ABA teams ≈21 pts worse in EuroLeague) | Prior for GBL↔EuroLeague translation |

**Take-aways that shape the plan:**
- Accuracy ceiling for pre-game EuroLeague prediction is ~67–70% (R16). Anything far above that is leakage. Primary metric = **log loss**, plus Brier, calibration (R14).
- Player metrics must be judged by **out-of-sample game prediction** (R6) and by **stability/discrimination** (R10), not by "looks right".
- Small samples dominate GBL → **shrinkage/regularization everywhere** (R6, R8, R11).

---

## 3. Architecture

```
            ┌──────────── scheduled (GitHub Actions cron) ────────────┐
Sources     │  ingest  →  raw (immutable)  →  staging  →  marts        │
EL API ─────┤  httpx+retry   JSON/HTML cache  Parquet     DuckDB       │
ESAKE HTML ─┤                               pandera contracts          │
BasketHotel ┤                                                          │
Odds (fwd) ─┘                                                          │
            │  features (as-of, point-in-time)  →  models  →  predict  │
            │  MLflow tracking/registry           backtest   publish   │
            └──────────────────────────────────────────────────────────┘
                        │ append-only prediction log (git, public)
                        ▼
              FastAPI (read-only)  →  Website (static, charts)
```

**Stack (each tool must earn its place):**
| Concern | Choice | Why |
|---|---|---|
| Env/packaging | Python 3.12, `uv`, `ruff`, `mypy`, `pytest`, pre-commit | Standard, fast |
| Ingestion | `httpx` + `tenacity`, `selectolax` for HTML; Playwright only if BasketHotel needs it | Reliable, testable |
| Storage | Raw JSON/HTML (gzip) → Parquet → **DuckDB** | Zero-cost, fast, SQL, portable; Postgres only if the API needs it |
| Data contracts | `pandera` schemas + invariant checks (5 players on court, score reconciles with box score) | Catch silent data bugs |
| Modelling | scikit-learn, LightGBM, statsmodels, **PyMC** | Classical + Bayesian |
| Experiments | **MLflow** (tracking + model registry) | MLE signal; reproducibility |
| Orchestration | GitHub Actions cron → Typer CLI (`eurohoops ingest / build / train / predict / publish`) | Free, visible, enough. Dagster is an optional later upgrade |
| Serving | **FastAPI** (Docker) on a free/low-cost host | Documented API = MLE signal |
| Frontend | Static site (Astro or Next.js static export) + Observable Plot / Vega-Lite charts | Timeboxed; polish comes last |
| Monitoring | Rolling log-loss/calibration vs baselines, data-freshness checks, CI failure alerts | "Model in production" story |

**Repo layout**
```
EuroHoops-Analytics/
  src/eurohoops/{ingest,parse,entity,features,models,eval,sim,serve,cli}/
  sql/            # staging → marts transforms
  tests/          # unit (parsers, features), property tests (no leakage), data-contract tests
  notebooks/      # exploration only; nothing production lives here
  predictions/    # append-only public log (CSV/JSON), committed by CI before tip-off
  site/           # frontend
  docs/           # methodology, model cards, ADRs (architecture decision records)
  .github/workflows/
```

---

## 4. Data engineering workstream

### 4.1 EuroLeague ingestion
- Backfill 2007-08 → 2025-26: schedule, box score, PBP, shots (19 seasons × roughly 190–380 games; 2019-20 is partial). Throttle, cache, idempotent (skip existing).
- Daily job during the season: new results + upcoming fixtures.

### 4.2 GBL ingestion
- Scrape results per season/phase/round → game IDs → box scores (mode=3).
- **Spike (week 1):** BasketHotel `widget-service/show` for PBP. Decide: direct HTTP vs Playwright network capture. Record what PBP contains (subs? clock? coordinates?).
- Backfill target: last **8 seasons** first (2018-19 →), extend only if useful.

### 4.3 Entity resolution (players & teams across EL, GBL, seasons)
- Greek→Latin transliteration (ELOT 743 + custom rules for foreign names written in Greek, e.g. `ΓΚ`→`G`, `ΝΤ`→`D`, `ΜΠ`→`B`), normalization (accents, order `SURNAME, NAME`).
- Candidate matching: fuzzy (Jaro-Winkler / rapidfuzz) + blocking by season, team, jersey number, minutes-played consistency.
- Manual override table (CSV, version-controlled) for the hard cases. Report precision/recall on a hand-labelled sample (~200 pairs). *This is an interview story.*
- Team name history (sponsor names change every year: "Panathinaikos AKTOR", "ΣΚΑΪ ΠΑΝΑΘΗΝΑΪΚΟΣ AKTOR") → canonical club IDs.

### 4.4 Derived tables
- **Possessions** (R1 rules) and **stints** (lineup segments from `IN`/`OUT`) with points for/against and possessions per stint.
- Validation: exactly 5 on court per team at all times; stint points sum to final score; possessions per team within ±2 per game. Games failing checks are flagged, not silently dropped.
- Point-in-time snapshots: every feature is computed "as of" a timestamp.

### 4.5 Data quirks to model or flag
- 2019-20 season cancelled mid-way; 2020-21 largely **without fans** (home advantage ≈ smaller) → home-court parameter by season.
- 2021-22: Russian clubs removed mid-season (their games voided/forfeited); format changes (16→18→20 teams, play-in added in recent seasons).
- GBL phase formats differ by season.

---

## 5. Modelling workstream

### Evaluation protocol (applies to everything — build it FIRST)
- **Walk-forward by round**: to predict round *r* of season *s*, train only on data with timestamps before round *r*'s first tip-off.
- **Splits**: tuning seasons (e.g. 2015-16 → 2022-23), validation 2023-24, **held-out test 2024-25 + 2025-26 (touched once)**, then **live 2026-27**.
- **Metrics**: log loss (primary), Brier, ECE + reliability diagrams; margin RMSE/MAE; totals MAE; **CRPS** for predictive distributions; accuracy reported only as secondary.
- **Report every metric per competition.** GBL has many lopsided games (Olympiacos/Panathinaikos vs the rest) where the winner is obvious → win-prob log loss says little there; **spread and total error are the more informative GBL metrics**.
- Training data = everything strictly before the prediction point (incl. 2007-14 seasons); the season ranges above only define where tuning/validation/test *scores* are computed.
- **Baselines** (every model must beat them to ship): (B0) home team wins with league base rate; (B1) Elo (R5); (B2) market, where available.
- **Significance**: paired bootstrap on per-game loss differences, CIs reported.
- **Leakage tests** (automated, in CI): adding, deleting or altering any game after the prediction timestamp must not change a single feature value; features for game *g* must be identical whether or not *g*'s own events exist in the DB.

### 5.1 M0 — Elo baseline (weeks 0–1 for EL, week 2 for GBL)
- 538-style: MOV multiplier, home advantage, season-start reversion (R5); tuned K on tuning seasons. Separate ratings per competition, optional shared rating for Greek clubs in both.
- Output win prob + (via fitted linear map) expected margin. **Goes live immediately.**

### 5.2 M1 — Team efficiency model (possession-based)
- Adjusted ORtg/DRtg per team via ridge / Bayesian hierarchical regression on game (or stint) efficiency with home effect and opponent adjustment; exponential time decay.
- Predict: pace (team pace effects) × efficiencies → **expected points per team → margin & total**.
- Predictive distribution: margin ~ Normal/Student-t with variance tested per R4; win prob = P(margin>0).
- v2: dynamic state-space strengths (R3/R4) fit with a Kalman filter or PyMC.

### 5.3 M2 — Shot-quality model (xPTS), EuroLeague
- Target: make/miss per shot (FGA); xPTS = p(make) × value.
- Features: distance & angle from coords (verify court coordinate system & units first), zone, 2/3, `FASTBREAK`, `SECOND_CHANCE`, `POINTS_OFF_TURNOVER`, time remaining in period/game, score margin, home/away, season (league-wide shooting trends).
- **Do NOT include shooter identity** (quality ≠ skill, R9).
- Known gap: shooting fouls on missed shots are not FGAs → xPTS undervalues rim pressure. Model free-throw generation separately (FTA per possession/shot zone) and report "total shot value" = xPTS + expected FT points.
- Models: logistic GAM/splines baseline → LightGBM → calibrated (isotonic on held-out). Leave-one-season-out CV.
- Outputs: team shot quality (offense & defense allowed), **player shot-making = Σ(actual − xPTS)** shrunk with empirical Bayes (R11), per 100 shots with intervals.
- Stability check (R10): year-to-year correlation of shot-making over expected vs raw eFG%.

### 5.4 M3 — Player impact: RAPM → Bayesian RAPM → box-score prior
1. **RAPM** (R6): design matrix of stints (+1 offense players, −1 defense players → separate O-RAPM and D-RAPM), target points per 100 possessions, possession weights, ridge λ by CV that **predicts future game margins** (R6 protocol). Multi-season with time decay. Low-minute handling via dummy indicators (R8).
2. **Bayesian RAPM** in PyMC (R7): posterior intervals → uncertainty shown on player cards.
3. **Box-score prior ("SPM")**: regress multi-season RAPM on box-score rates (like BPM) on EuroLeague data. Use it as the prior mean in RAPM (lower variance) **and** apply it to GBL box scores → GBL player impact even if GBL PBP lacks subs. *This transfer step is a showcase piece.*
4. Evaluation: out-of-sample margin prediction of next-N games using minutes-weighted player ratings vs team-only ratings vs PIR (EuroLeague's official index as a naive baseline).

### 5.5 M4 — League translation (GBL ↔ EuroLeague)
- Data: players/teams appearing in both competitions (Olympiacos, Panathinaikos every season; players moving between GBL clubs and EL clubs across seasons).
- Team level: estimate competition offset using Greek teams' ratings in both (R19 as prior).
- Player level: mixed model of per-100-possession rates (and SPM) with competition fixed effects + player random effects → **translation factors per stat**, with uncertainty.
- Validation: predict EuroLeague production of players who moved GBL→EL the following season; compare to naive "same stats" and "team-offset only".

### 5.6 M5 — Game predictor v2 (roster-aware)
- Team strength = Σ(projected minutes × player rating) + team residual (coaching/system) + home + rest/travel (days since last game, back-to-backs with EL midweek + GBL weekend — **very relevant for Olympiacos/Panathinaikos**).
- Availability: no injury feed → **two variants reported honestly**: (a) *projected roster* (from recent rotations; legit for live), (b) *oracle roster* (who actually played; upper bound, clearly labelled, never used for the live log).
- Stack/blend with M1 and Elo; pick by log loss on validation; recalibrate.
- Outputs: P(win), expected margin (spread), expected total with distribution → over/under probabilities for any line.

### 5.7 M6 — Player projections & scouting
- Rest-of-season and next-season projection: shrunk current rates + aging curve (R12, hierarchical) + translation (M4) if changing league.
- **Over/under-performance board**: for each player, observed vs expected on dimensions that R10 says are stable (e.g. shot-making over xPTS, 3P% vs FT%-informed prior, on/off vs RAPM). Flag "likely regression" vs "likely real".
- **Similarity search**: player embedding from standardized per-100 rates + xPTS profile + shot map → nearest neighbours ("plays like…").
- "Undervalued" list for GBL: high impact/translation, low minutes or age ≤ 24.

### 5.8 M7 — Season simulator
- Monte Carlo (10k+) of remaining schedule. **Sample team strengths from their posterior per simulation** (not just game noise) — otherwise probabilities are overconfident.
- Encode exact formats & tiebreakers: EuroLeague 2026-27 (top-6 playoffs, 7–10 play-in — verify rules), GBL phases/playoffs per season.
- Outputs per team: final-position distribution, playoff/play-in/title odds; track them over the season (chart).
- Evaluate: backtest mid-season simulations on past seasons (Brier on "made playoffs", RPS on final rank).

---

## 6. Product (website + API)

Pages (minimal first, polish last):
1. **Predictions** — next round: win prob, spread, total, confidence; after games: result + running scorecard vs baselines. *(Live from week 2, even if ugly.)*
2. **Model performance** — live log loss/Brier vs Elo & baselines, reliability diagram, cumulative chart. Updated automatically.
3. **Standings projection** — simulation odds, trajectory over the season.
4. **Teams** — four factors, ratings, shot profile, schedule strength.
5. **Players / Scouting** — player card (RAPM ± interval, xPTS shot-making, shot chart, projection, comparables), over/under-performer board, GBL→EL translation.
6. **Methodology** — model cards, data sources, limitations, what failed.

API (FastAPI, read-only): `/games/upcoming`, `/predictions/{game_id}`, `/teams/{id}/ratings`, `/players/{id}`, `/simulations/latest`, `/metrics/live`. OpenAPI docs public.

---

## 7. MLOps & quality

- **Pre-registration**: CI job writes predictions to `predictions/` and commits **before tip-off**; commit hash + timestamp = proof. Never edit past rows; corrections are new rows. EuroLeague rounds span Tue–Fri and GBL plays weekends → run the predict job **daily** (e.g. 08:00 UTC) for that day's games, plus a round-level preview when the round opens. Store `predicted_at` and `tipoff_utc` in every row; the scorecard only counts rows where `predicted_at < tipoff_utc`.
- CI on every push: ruff, mypy, pytest (parsers with saved fixtures, feature leakage tests, contract tests).
- MLflow: every training run logs data snapshot hash, params, metrics, artifacts; registry stage "production" is what the predictor loads.
- Retraining cadence: ratings update after every round; heavier models (xPTS, RAPM prior) monthly.
- Monitoring: rolling 50-game log loss vs Elo; calibration drift; data freshness (fail loudly if a round is missing); scraper breakage alerts.
- Docs: model cards per model, ADRs for major decisions, README with architecture diagram and "how to reproduce in 3 commands".
- Docker image for API + pipeline; one-command local run.

---

## 8. Timeline (≈20–25 h/week; weeks start Mon 28 Sep 2026; at 30+ h/week compress weeks 3–18 by ~25%, but never skip exit gates)

EuroLeague started **24 Sep 2026**; GBL starts late Sep/early Oct. The live log must start ASAP — missing the early rounds is acceptable, missing a month is not.

| Weeks | Phase | Deliverables | Exit gate |
|---|---|---|---|
| **0–1** (now → Sun 11 Oct) | Bootstrap | Repo, CI, EL ingestion (current + 3 seasons), Elo M0, **first live EL predictions committed by round 3–4 at the latest**, GBL PBP spike report | Predictions commit automatically before tip-off |
| 2–3 | Data foundation | GBL **results** scraper first (8 seasons) → GBL Elo **live by week 3**; then GBL box scores, full EL backfill 2007→, DuckDB marts, pandera contracts, minimal predictions page (static) | Contracts pass; page live; both competitions logged |
| 3–5 | Evaluation harness | Walk-forward backtester, metrics, baselines, bootstrap CIs, leakage tests; Elo tuned | Harness reproduces Elo numbers; leakage tests in CI |
| 5–7 | Team model M1 | Possession/stint tables, adjusted efficiencies, margin/total distributions; MLflow | M1 beats Elo on validation log loss (or documented why not) |
| 7–10 | Shot model M2 | xPTS model, calibration, player shot-making, shot charts | ECE small, beats spline baseline; stability analysis done |
| 9–12 | Player impact M3 | Stints validated, RAPM (+dummy), Bayesian RAPM, SPM prior, GBL transfer | RAPM beats box-only on future-margin prediction |
| 12–14 | Entity resolution + M4 | Cross-league player matching (P/R reported), translation factors | Translation beats naive on GBL→EL movers |
| 14–16 | M5 + M7 | Roster-aware predictor (projected vs oracle), season simulator | M5 beats M1 on validation; sim backtest calibrated |
| 16–18 | M6 + product | Projections, scouting board, similarity; API; site polish | All pages live from API |
| 18–20 | Write-up | Methodology page, 2–3 blog posts, README, demo video (3 min) | A stranger can reproduce and understand it |
| 20+ (→ Jun 2027) | Run the season | Weekly checks, end-of-season report: live results vs baselines, incl. EL playoffs/F4 and GBL finals | Honest final report published |

---

## 9. Risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| ESAKE/BasketHotel change markup or block | Medium | Raw cache, polite throttling, parser fixture tests, alert on failure; box-score path independent of PBP |
| GBL PBP lacks substitutions | Medium | SPM transfer (M3.3) gives GBL player impact from box scores |
| EuroLeague API changes | Low–Med | Raw cache; wrapper library as fallback |
| No odds benchmark | High (historical) | Elo/crowd baselines; forward-record odds from now |
| Small GBL sample → noisy models | Certain | Shrinkage, pooling with EL, uncertainty intervals shown everywhere |
| Scope creep / polishing UI too early | High | Exit gates; UI work only after Week 16 except the predictions + performance pages |
| Model doesn't beat Elo | Possible | Report it honestly + analysis why; still a strong project if the engineering and evaluation are rigorous |
| Data rights | Low–Med | Personal, non-commercial, attributed use; publish derived metrics and visuals, **never re-host bulk raw data**; honor takedown requests |
| Hosting costs | Low | Everything above fits free tiers; budget ceiling €10/month (domain + small VM if a free API host disappears) |
| You can't explain the models | Depends on you | You write the core model code yourself; I review/pair. Keep a `docs/learning-log.md` |

---

## 10. What makes it hireable — packaging

**Resume bullets (fill with real numbers at the end):**
- Built an automated ML platform forecasting EuroLeague & Greek Basket League games; live, pre-registered season-long predictions achieved log loss X vs Y for Elo (Z% improvement, 95% CI [..]).
- Developed a shot-quality (xPTS) model on 19 seasons / N shots of EuroLeague data (calibrated, ECE = …) and a Bayesian RAPM player-impact model; transferred impact estimates to the GBL via a box-score prior.
- Engineered ingestion + entity resolution across bilingual (Greek/Latin) sources (precision P / recall R), with data contracts, CI, MLflow registry, and a FastAPI service.

**Interview-readiness checklist (you must be able to answer):**
- How do you guarantee no leakage? Show the test.
- Why log loss over accuracy? Why did you pick λ this way?
- Why shrinkage, and how much? What does the posterior interval mean on this player card?
- What didn't work and why?
- How would this change with tracking data / injury feeds / more money?

**For sports-analyst roles specifically:** 3–4 written scouting reports (e.g., "3 GBL players ready for EuroLeague minutes") built from your models, with visuals. Clubs hire people who can communicate, not only model.

---

## 11. Explicitly out of scope (v1)
Deep learning models, tracking-data methods, in-game live win probability (stretch goal after season), injury prediction, betting recommendations/staking, GBL seasons before 2018-19, Greek A2 league, women's leagues.

---

## 12. Open verification items (week 1)
1. BasketHotel PBP: endpoint, fields, substitutions present? Coordinates?
2. EuroLeague shot coordinate system (origin, units, orientation) — plot a season of shots.
3. GBL old-season box-score availability (spot-check 2018-19, 2014-15).
4. 2026-27 formats & tiebreak rules (EuroLeague play-in, GBL phases).
5. OddsPortal feasibility (1-day timebox).
6. Stint validation pass rate on a sample of 50 EL games.

---

## Sources
- EuroLeague API: `live.euroleague.net/api/*`, `api-live.euroleague.net` (verified 2026-09-24) · [euroleague_api (GitHub)](https://github.com/giasemidis/euroleague_api) · [Shot charts in Python](https://g-giasemidis.medium.com/create-euroleague-shot-charts-in-python-7ba4aa574644)
- [ESAKE / GBL](https://www.esake.gr/) (verified 2026-09-24)
- R1 [Kubatko et al. 2007](https://www.researchgate.net/publication/4985986_A_Starting_Point_for_Analyzing_Basketball_Statistics)
- R4 [Manner 2016](https://static.uni-graz.at/fileadmin/_Persoenliche_Webseite/manner_hans/Publikationen/NBA_revision2.pdf)
- R5 [FiveThirtyEight NBA Elo](https://fivethirtyeight.com/features/how-we-calculate-nba-elo-ratings)
- R6 [Sill 2010](https://supermariogiacomazzo.github.io/STOR538_WEBSITE/Articles/Basketball/Basketball_Sill.pdf)
- R7 [Deshpande & Jensen 2016](https://arxiv.org/abs/1604.03186)
- R8 [Dummy RAPM 2026](https://arxiv.org/pdf/2608.19454)
- R9 [Chang et al. 2014](https://www.semanticscholar.org/paper/Quantifying-Shot-Quality-in-the-NBA-Chang-Maheswaran/2ef041e6d5b83979d37cb0526b7bcc9f63c5b59e)
- R10 [Franks et al. 2016](https://arxiv.org/abs/1609.09830)
- R13 [Terner & Franks 2021](https://arxiv.org/pdf/2007.10550)
- R14 [Walsh & Joshi 2024](https://arxiv.org/abs/2303.06021) · [corrigendum](https://www.sciencedirect.com/science/article/pii/S2666827025000106)
- R15 [Hubáček et al. 2019](http://ida.felk.cvut.cz/zelezny/pubs/ijf.2019.pdf)
- R16 [Giasemidis 2020](https://arxiv.org/abs/2002.08465)
- R17 [Lampis et al. 2023](https://journals.sagepub.com/doi/10.3233/JSA-220639)
- R18 [Ballı & Özdemir 2021](https://www.sciencedirect.com/science/article/abs/pii/S0960077921004732)
- R19 [Ranking all Major European Basketball Leagues](https://germanbasketballanalytics.substack.com/p/ranking-all-major-european-basketball)
- R3 Glickman & Stern (1998) JASA 93:25–35; R11 Efron & Morris (1975) JASA 70:311–319; R12 Vaci et al. (2019) Behav. Res. Methods 51 — cited from knowledge, not re-fetched.
