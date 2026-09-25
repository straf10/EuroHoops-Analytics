"""``eurohoops`` command line: ingest -> build -> backtest -> predict -> score -> publish."""

import json
import logging
import os
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
import typer

from eurohoops.config import (
    BOX_INVARIANTS_REPORT,
    COMPETITIONS,
    EUROLEAGUE,
    GBL,
    GBL_BOX_FILL,
    GBL_BOX_GAPS,
    GBL_PBP,
    GBL_PLAYER_BOX,
    GBL_TEAM_BOX,
    LIVE_SEASON,
    MART_PATH,
    ODDS_CALLS,
    ODDS_RAW_DIR,
    ODDS_TEAMS,
    SITE_DATA,
    SQL_DIR,
    STINT_REPORT,
)
from eurohoops.eval.backtest import format_table, load_tuned_model, run_backtest
from eurohoops.eval.scorecard import build_scorecard
from eurohoops.ingest import euroleague, gbl
from eurohoops.ingest.http import Fetcher, make_client
from eurohoops.marts import (
    box_invariants,
    build_marts,
    read_games,
    read_teams,
    refresh_box_gaps,
)
from eurohoops.odds import OddsApiError, OddsPaths, api_key, record_odds
from eurohoops.parse.box import build_box_tables
from eurohoops.parse.games import (
    build_games_table,
    build_gbl_tables,
    build_teams_table,
    write_table,
)
from eurohoops.parse.gbl_pbp import build_pbp_table
from eurohoops.parse.stints import validate_sample
from eurohoops.predict import LatePredictionError, predict_upcoming
from eurohoops.publish import Section, site_data

app = typer.Typer(no_args_is_help=True, add_completion=False)
log = logging.getLogger("eurohoops")


class CompetitionName(StrEnum):
    euroleague = "euroleague"
    gbl = "gbl"


CompetitionOption = Annotated[
    CompetitionName, typer.Option("--competition", help="Which competition")
]


def utc_now() -> datetime:
    return datetime.now(UTC)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")


@app.callback()
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def ingest(
    ctx: typer.Context,
    competition: CompetitionOption = CompetitionName.euroleague,
    seasons: Annotated[
        list[int] | None,
        typer.Option(help="Season start years, space-separated: --seasons 2023 2024 2025 2026"),
    ] = None,
    details: Annotated[
        bool,
        typer.Option(help="Also cache game details (EuroLeague: box/PBP/shots; GBL: box scores)"),
    ] = False,
    pbp: Annotated[
        bool,
        typer.Option(help="GBL: also cache the chosen seasons' play-by-play (local backfill)"),
    ] = False,
) -> None:
    """Fetch results into the raw cache and rebuild the competition's staging tables.

    With ``--pbp`` the chosen seasons only select which games get play-by-play; the staging
    tables still cover every default season.
    """
    try:
        extra = [int(arg) for arg in ctx.args]
    except ValueError as exc:
        raise typer.BadParameter(f"unexpected arguments {ctx.args}") from exc
    comp = COMPETITIONS[competition]
    chosen = sorted({*(seasons or []), *extra}) or list(comp.default_seasons)
    if pbp and comp is not GBL:
        raise typer.BadParameter("--pbp is for --competition gbl (EuroLeague PBP: --details)")
    with make_client() as client:
        if comp is GBL:
            fetcher = Fetcher(client, gbl.MIN_INTERVAL_S)
            staged = sorted({*chosen, *comp.default_seasons}) if pbp else chosen
            rounds = gbl.ingest_gbl(
                fetcher,
                comp.raw_dir,
                staged,
                details,
                LIVE_SEASON,
                pbp_seasons=chosen if pbp else (),
            )
            games, teams = build_gbl_tables(rounds)
            if details or pbp:
                tables = build_box_tables(comp.raw_dir, games)
                write_table(tables.player_box, GBL_PLAYER_BOX)
                write_table(tables.team_box, GBL_TEAM_BOX)
                write_table(tables.fill, GBL_BOX_FILL)
            if pbp:
                write_table(build_pbp_table(comp.raw_dir, games), GBL_PBP)
        else:
            fetcher = Fetcher(client, euroleague.MIN_INTERVAL_S)
            schedules = euroleague.ingest_seasons(
                fetcher, comp.raw_dir, chosen, details, LIVE_SEASON
            )
            games, teams = build_games_table(schedules), build_teams_table(schedules)
    write_table(games, comp.staging_games)
    write_table(teams, comp.staging_teams)
    log.info(
        "wrote %d games (%d played) to %s", len(games), games["played"].sum(), comp.staging_games
    )


@app.command()
def build() -> None:
    """Build the DuckDB marts from staging; with GBL box scores, write the invariant report."""
    if build_marts(MART_PATH, SQL_DIR):
        report = box_invariants(MART_PATH)
        _write_json(BOX_INVARIANTS_REPORT, report)
        refresh_box_gaps(MART_PATH, GBL_BOX_GAPS)
        for season, stats in report["seasons"].items():
            typer.echo(f"gbl box scores {season}: {stats['passed']}/{stats['games']} pass")
    typer.echo(f"marts written to {MART_PATH}")


@app.command()
def stints() -> None:
    """Validate EuroLeague stints on the seeded 50-game sample (reads the local raw cache)."""
    report = validate_sample(EUROLEAGUE.raw_dir)
    _write_json(STINT_REPORT, report)
    rates = ", ".join(f"{check} {rate:.0%}" for check, rate in report["pass_rate"].items())
    typer.echo(
        f"{report['games']} games: all checks {report['all_checks_pass_rate']:.0%} ({rates}); "
        f"wrote {STINT_REPORT}"
    )


@app.command()
def backtest(competition: CompetitionOption = CompetitionName.euroleague) -> None:
    """Tune Elo and score it vs B0; the live backtest report holds the live parameters."""
    comp = COMPETITIONS[competition]
    games = read_games(MART_PATH, comp.name)
    for spec in (comp.live_backtest, *comp.history_backtests):
        report = run_backtest(games, spec)
        _write_json(spec.report, report)
        typer.echo(f"{spec.report}\n{format_table(report)}")


@app.command()
def predict(
    competition: CompetitionOption = CompetitionName.euroleague,
    window_hours: Annotated[float, typer.Option(help="Predict games tipping off within")] = 36.0,
) -> None:
    """Append pre-tip-off predictions for upcoming live-season games to the public log."""
    comp = COMPETITIONS[competition]
    try:
        added = predict_upcoming(
            read_games(MART_PATH, comp.name),
            load_tuned_model(comp.live_backtest.report),
            log_path=comp.prediction_log,
            season=LIVE_SEASON,
            replay_from=comp.live_backtest.warmup[0],
            window=timedelta(hours=window_hours),
            clock=utc_now,
        )
    except LatePredictionError as exc:
        log.error("refusing to log: %s", exc)
        raise typer.Exit(code=1) from exc
    typer.echo(f"{added} predictions appended to {comp.prediction_log}")


@app.command()
def odds() -> None:
    """Record EuroLeague market odds: one The Odds API call, consensus rows appended."""
    key = api_key(os.environ, Path(".env"))
    if key is None:
        log.error("ODDS_API_KEY is not set (environment or .env); no call made")
        raise typer.Exit(code=1)
    assert EUROLEAGUE.odds_log is not None
    paths = OddsPaths(EUROLEAGUE.odds_log, ODDS_CALLS, ODDS_TEAMS, ODDS_RAW_DIR)
    try:
        with make_client() as client:
            summary = record_odds(
                client, key, read_games(MART_PATH, EUROLEAGUE.name), paths, utc_now
            )
    except OddsApiError as exc:
        log.error("odds: %s", exc)
        raise typer.Exit(code=1) from None
    typer.echo(
        f"{summary['rows_written']} consensus rows from {summary['events']} events appended to "
        f"{paths.log}; {len(summary['unmatched'])} unmatched; "
        f"requests remaining {summary['requests_remaining'] or '?'}"
    )


@app.command()
def score(competition: CompetitionOption = CompetitionName.euroleague) -> None:
    """Score the prediction log against results and write the competition's scorecard."""
    comp = COMPETITIONS[competition]
    card = build_scorecard(
        comp.prediction_log,
        read_games(MART_PATH, comp.name),
        load_tuned_model(comp.live_backtest.report),
        utc_now(),
        manual_pushes=comp.manual_pushes,
        odds_path=comp.odds_log,
    )
    _write_json(comp.scorecard, card)
    typer.echo(json.dumps(card, indent=2))


@app.command()
def publish() -> None:
    """Write the site data (web/src/data/site.json) the Astro front-end renders."""
    sections = [
        Section(
            key=comp.name,
            title=title,
            log=(
                pd.read_csv(comp.prediction_log, dtype={"game_id": str})
                if comp.prediction_log.exists()
                else pd.DataFrame()
            ),
            scorecard=json.loads(comp.scorecard.read_text(encoding="utf-8")),
            backtest=json.loads(comp.live_backtest.report.read_text(encoding="utf-8")),
            games=read_games(MART_PATH, comp.name),
            names=dict(read_teams(MART_PATH, comp.name).itertuples(index=False)),
            model=load_tuned_model(comp.live_backtest.report),
            season=LIVE_SEASON,
            replay_from=comp.live_backtest.warmup[0],
        )
        for title, comp in (("EuroLeague", EUROLEAGUE), ("Greek Basket League", GBL))
    ]
    _write_json(SITE_DATA, site_data(sections, utc_now()))
    typer.echo(f"wrote {SITE_DATA}")
