"""``eurohoops`` command line: ingest -> build -> backtest -> predict -> score -> publish."""

import json
import logging
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
    GBL_PLAYER_BOX,
    GBL_TEAM_BOX,
    LIVE_SEASON,
    MART_PATH,
    SITE_DATA,
    SQL_DIR,
)
from eurohoops.eval.backtest import format_table, load_tuned_model, run_backtest
from eurohoops.eval.scorecard import build_scorecard
from eurohoops.ingest import euroleague, gbl
from eurohoops.ingest.http import Fetcher, make_client
from eurohoops.marts import box_invariants, build_marts, read_games, read_teams
from eurohoops.parse.box import build_box_tables
from eurohoops.parse.games import (
    build_games_table,
    build_gbl_tables,
    build_teams_table,
    write_table,
)
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
) -> None:
    """Fetch results into the raw cache and rebuild the competition's staging tables."""
    try:
        extra = [int(arg) for arg in ctx.args]
    except ValueError as exc:
        raise typer.BadParameter(f"unexpected arguments {ctx.args}") from exc
    comp = COMPETITIONS[competition]
    chosen = sorted({*(seasons or []), *extra}) or list(comp.default_seasons)
    with make_client() as client:
        if comp is GBL:
            fetcher = Fetcher(client, gbl.MIN_INTERVAL_S)
            rounds = gbl.ingest_gbl(fetcher, comp.raw_dir, chosen, details, LIVE_SEASON)
            games, teams = build_gbl_tables(rounds)
            if details:
                player_box, team_box = build_box_tables(comp.raw_dir, games)
                write_table(player_box, GBL_PLAYER_BOX)
                write_table(team_box, GBL_TEAM_BOX)
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
        for season, stats in report["seasons"].items():
            typer.echo(f"gbl box scores {season}: {stats['passed']}/{stats['games']} pass")
    typer.echo(f"marts written to {MART_PATH}")


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
def score(competition: CompetitionOption = CompetitionName.euroleague) -> None:
    """Score the prediction log against results and write the competition's scorecard."""
    comp = COMPETITIONS[competition]
    card = build_scorecard(
        comp.prediction_log,
        read_games(MART_PATH, comp.name),
        load_tuned_model(comp.live_backtest.report),
        comp.manual_pushes,
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
