"""``eurohoops`` command line: ingest -> backtest -> predict -> score."""

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, Any

import typer

from eurohoops.config import (
    BACKTEST_REPORT,
    DEFAULT_SEASONS,
    GAMES_PATH,
    LIVE_SEASON,
    PREDICTION_LOG,
    RAW_DIR,
    SCORECARD_REPORT,
)
from eurohoops.eval.backtest import format_table, load_tuned_model, run_backtest
from eurohoops.eval.scorecard import build_scorecard
from eurohoops.ingest.euroleague import ingest_seasons
from eurohoops.ingest.http import Fetcher, make_client
from eurohoops.parse.games import build_games_table, read_games, write_games
from eurohoops.predict import LatePredictionError, predict_upcoming

app = typer.Typer(no_args_is_help=True, add_completion=False)
log = logging.getLogger("eurohoops")


def utc_now() -> datetime:
    return datetime.now(UTC)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


@app.callback()
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def ingest(
    ctx: typer.Context,
    seasons: Annotated[
        list[int] | None,
        typer.Option(help="Season start years, space-separated: --seasons 2023 2024 2025 2026"),
    ] = None,
    details: Annotated[
        bool, typer.Option(help="Also cache box score, PBP and shots of completed games")
    ] = False,
) -> None:
    """Fetch schedules (and optionally game details) into the raw cache; rebuild the games table."""
    try:
        extra = [int(arg) for arg in ctx.args]
    except ValueError as exc:
        raise typer.BadParameter(f"unexpected arguments {ctx.args}") from exc
    chosen = sorted({*(seasons or []), *extra}) or list(DEFAULT_SEASONS)
    with make_client() as client:
        schedules = ingest_seasons(Fetcher(client), RAW_DIR, chosen, details)
    games = build_games_table(schedules)
    write_games(games, GAMES_PATH)
    log.info("wrote %d games (%d played) to %s", len(games), games["played"].sum(), GAMES_PATH)


@app.command()
def backtest() -> None:
    """Tune Elo on the tuning season, score tuning + test, write reports/backtest_elo.json."""
    report = run_backtest(read_games(GAMES_PATH))
    _write_json(BACKTEST_REPORT, report)
    typer.echo(format_table(report))


@app.command()
def predict(
    window_hours: Annotated[float, typer.Option(help="Predict games tipping off within")] = 36.0,
) -> None:
    """Append pre-tip-off predictions for upcoming live-season games to the public log."""
    try:
        added = predict_upcoming(
            read_games(GAMES_PATH),
            load_tuned_model(BACKTEST_REPORT),
            log_path=PREDICTION_LOG,
            season=LIVE_SEASON,
            window=timedelta(hours=window_hours),
            clock=utc_now,
        )
    except LatePredictionError as exc:
        log.error("refusing to log: %s", exc)
        raise typer.Exit(code=1) from exc
    typer.echo(f"{added} predictions appended to {PREDICTION_LOG}")


@app.command()
def score() -> None:
    """Score the prediction log against results, write reports/live_scorecard.json."""
    card = build_scorecard(
        PREDICTION_LOG, read_games(GAMES_PATH), load_tuned_model(BACKTEST_REPORT)
    )
    _write_json(SCORECARD_REPORT, card)
    typer.echo(json.dumps(card, indent=2))
