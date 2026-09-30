"""``eurohoops`` command line: ingest -> build -> backtest -> predict -> score -> publish."""

import json
import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from functools import partial
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from eurohoops import research
from eurohoops.config import (
    BIO_EL_SEASONS,
    BOX_INVARIANTS_REPORT,
    COMPETITIONS,
    EUROLEAGUE,
    GBL,
    GBL_BOX_FILL,
    GBL_BOX_GAPS,
    GBL_PBP,
    GBL_PLAYER_BOX,
    GBL_TEAM_BOX,
    GREEK_EL_CLUBS,
    LIVE_SEASON,
    M3,
    M3_GBL,
    MART_PATH,
    ODDS_CALLS,
    ODDS_RAW_DIR,
    ODDS_TEAMS,
    PLAYER_BIOS,
    POSSESSION_REPORT,
    SITE_DATA,
    SQL_DIR,
    STINT_REPORT,
    STINTS_MART_REPORT,
    TEAM_CONTINUITY_REPORT,
    Competition,
)
from eurohoops.eval.backtest import TunedModel, format_table, load_tuned_model, run_backtest
from eurohoops.eval.m1_backtest import format_m1_table, run_m1_backtest
from eurohoops.eval.m3_backtest import format_m3_table, run_m3_backtest
from eurohoops.eval.m3_gbl_backtest import el_spm_models, format_m3_gbl_table, run_m3_gbl_backtest
from eurohoops.eval.scorecard import build_scorecard
from eurohoops.eval.tracking import default_tracking_uri, log_backtest, log_m3_backtest
from eurohoops.ingest import euroleague, gbl
from eurohoops.ingest.bios import build_bios, ingest_bios
from eurohoops.ingest.http import Fetcher, make_client
from eurohoops.live_m1 import load_m1, predict_upcoming_m1
from eurohoops.logs import write_json
from eurohoops.marts import (
    box_invariants,
    build_marts,
    read_games,
    read_table,
    read_teams,
    refresh_box_gaps,
    write_tables,
)
from eurohoops.models.box_impact import BoxGrid, box_only_margins, pir_margins
from eurohoops.odds import OddsApiError, OddsPaths, api_key, record_odds
from eurohoops.parse.box import build_box_tables
from eurohoops.parse.continuity import continuity_report
from eurohoops.parse.games import (
    build_games_table,
    build_gbl_tables,
    build_gbl_team_seasons,
    build_team_seasons,
    build_teams_table,
    write_table,
)
from eurohoops.parse.gbl_box_lines import build_gbl_player_games
from eurohoops.parse.gbl_pbp import build_pbp_table
from eurohoops.parse.possession_report import possession_report
from eurohoops.parse.stints import validate_sample
from eurohoops.parse.stints_mart import build_stints_mart, mart_report
from eurohoops.parse.team_box import TEAM_GAMES_SCHEMA, build_team_games
from eurohoops.predict import LatePredictionError, predict_upcoming
from eurohoops.publish import DISPLAY_CODES, Section, site_data
from eurohoops.stats.box import build_box_games
from eurohoops.stats.export import STATS_DIR, Inputs, build_payloads, load_cached_games, write_stats
from eurohoops.stats.shots import build_shots

app = typer.Typer(no_args_is_help=True, add_completion=False)
log = logging.getLogger("eurohoops")


class CompetitionName(StrEnum):
    euroleague = "euroleague"
    gbl = "gbl"


class ModelName(StrEnum):
    elo = "elo"
    m1 = "m1"
    m2 = "m2"
    m3 = "m3"


CompetitionOption = Annotated[
    CompetitionName, typer.Option("--competition", help="Which competition")
]


def utc_now() -> datetime:
    return datetime.now(UTC)


def _gbl_pbp() -> pd.DataFrame | None:
    return pd.read_parquet(GBL_PBP) if GBL_PBP.exists() else None


def _both_games() -> pd.DataFrame:
    return pd.concat(
        [read_games(MART_PATH, c.name).assign(competition=c.name) for c in (EUROLEAGUE, GBL)],
        ignore_index=True,
    )


@dataclass(frozen=True)
class Live:
    """What predict, score and publish read for a competition's live season."""

    games: pd.DataFrame  # the competition's games mart
    model: TunedModel  # the live Elo parameters (its live backtest report)
    season: int
    replay_from: int  # Elo replays from the live backtest's first warm-up season


def _live(comp: Competition) -> Live:
    return Live(
        read_games(MART_PATH, comp.name),
        load_tuned_model(comp.live_backtest.report),
        LIVE_SEASON,
        comp.live_backtest.warmup[0],
    )


def _team_games(competition: str) -> pd.DataFrame:
    """The competition's ``team_games`` rows (none before the first ``build``)."""
    table = read_table(MART_PATH, "team_games", competition)
    return pd.DataFrame(columns=list(TEAM_GAMES_SCHEMA.columns)) if table is None else table


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
            team_seasons = build_gbl_team_seasons(rounds)
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
            team_seasons = build_team_seasons(schedules)
    write_table(games, comp.staging_games)
    write_table(teams, comp.staging_teams)
    write_table(team_seasons, comp.staging_team_seasons)
    log.info(
        "wrote %d games (%d played) to %s", len(games), games["played"].sum(), comp.staging_games
    )


@app.command("ingest-bios")
def ingest_bios_command() -> None:
    """Cache player bios for entity resolution (ESAKE player pages for every staged GBL player,
    EuroLeague season people 2007-2025) and stage their birth dates and countries.

    Local backfill, like ``ingest --pbp``: never run in CI. Needs the GBL box scores staged.
    """
    if not GBL_PLAYER_BOX.exists():
        log.error("no staged GBL box scores; run: eurohoops ingest --competition gbl --details")
        raise typer.Exit(code=1)
    box = pd.read_parquet(GBL_PLAYER_BOX, columns=["player_id", "team", "seconds"])
    per_player = box.assign(big_two=box["team"].isin(GREEK_EL_CLUBS)).groupby("player_id")
    order = per_player.agg(big_two=("big_two", "any"), seconds=("seconds", "sum"))
    # the Greek EuroLeague clubs' players first (the silver set), then by GBL minutes
    gbl_ids = order.sort_values(["big_two", "seconds"], ascending=False, kind="stable").index
    with make_client() as client:
        fetched = ingest_bios(
            Fetcher(client, gbl.MIN_INTERVAL_S),
            Fetcher(client, euroleague.MIN_INTERVAL_S),
            gbl_raw=GBL.raw_dir,
            el_raw=EUROLEAGUE.raw_dir,
            gbl_ids=gbl_ids,
            el_seasons=BIO_EL_SEASONS,
        )
    bios = build_bios(GBL.raw_dir, EUROLEAGUE.raw_dir)
    write_table(bios, PLAYER_BIOS)
    dated = bios.groupby("competition")["birth_date"].apply(lambda s: int(s.notna().sum()))
    typer.echo(
        f"fetched {fetched[0]} ESAKE pages, {fetched[1]} EuroLeague seasons; "
        f"{len(bios)} bios, with a birth date: {dated.to_dict()}; wrote {PLAYER_BIOS}"
    )


@app.command()
def build() -> None:
    """Build the DuckDB marts from staging (and ``team_games`` from the raw box-score cache).

    With GBL box scores staged, also write the invariant report.
    """
    has_box = build_marts(MART_PATH, SQL_DIR)
    team_games = build_team_games(
        read_games(MART_PATH, EUROLEAGUE.name),
        read_games(MART_PATH, GBL.name),
        (EUROLEAGUE.raw_dir, GBL.raw_dir),
        _gbl_pbp(),
    )
    write_tables(
        MART_PATH, {"team_games": team_games.table, "team_games_missing": team_games.missing}
    )
    typer.echo(
        f"team_games: {team_games.table['game_id'].nunique()} games, "
        f"{len(team_games.missing)} played games without box lines"
    )
    if has_box:
        report = box_invariants(MART_PATH)
        write_json(BOX_INVARIANTS_REPORT, report)
        refresh_box_gaps(MART_PATH, GBL_BOX_GAPS)
        for season, stats in report["seasons"].items():
            typer.echo(f"gbl box scores {season}: {stats['passed']}/{stats['games']} pass")
    _team_continuity()
    typer.echo(f"marts written to {MART_PATH}")


def _team_continuity() -> None:
    """Stage-to-marts ``team_seasons`` and print the live season's team changes (skipped before
    the first ingest that stages team seasons). The report file is ``eurohoops continuity``:
    the daily workflow stages fewer seasons, so ``build`` never rewrites it."""
    staged = [c for c in (EUROLEAGUE, GBL) if c.staging_team_seasons.exists()]
    if not staged:
        return
    team_seasons = pd.concat(
        [pd.read_parquet(c.staging_team_seasons).assign(competition=c.name) for c in staged],
        ignore_index=True,
    )
    write_tables(MART_PATH, {"team_seasons": team_seasons})
    report = continuity_report(team_seasons)
    for competition, block in report.items():
        live = block["seasons"].get(str(LIVE_SEASON))
        if live is None or live.get("first_season_in_data"):
            continue
        shown = DISPLAY_CODES.get(competition, {})
        changes = ", ".join(
            f"{kind} {' '.join(shown.get(e['team'], e['team']) for e in live[kind])}"
            for kind in ("new", "returning", "left")
            if live[kind]
        )
        typer.echo(f"{competition} {LIVE_SEASON} teams: {changes or 'no changes'}")
    # ESAKE ids mean nothing to a reader: every live GBL team needs a display code on the site.
    live_gbl = team_seasons[
        (team_seasons["competition"] == GBL.name) & (team_seasons["season"] == LIVE_SEASON)
    ]
    unnamed = sorted(set(live_gbl["team"]) - set(DISPLAY_CODES["gbl"]))
    if unnamed:
        typer.echo(f"gbl {LIVE_SEASON}: no display code for {' '.join(unnamed)} (publish.py)")
    flagged = sum(len(block["review_name_changes"]) for block in report.values())
    if flagged:
        typer.echo(f"team continuity: {flagged} name changes to review (eurohoops continuity)")


@app.command()
def continuity() -> None:
    """Write reports/team_continuity.json from the marts: new, returning and departed codes per
    season, and name changes to review (a code another club took over would show up there)."""
    team_seasons = read_table(MART_PATH, "team_seasons")
    if team_seasons is None:
        log.error("no team_seasons in the marts; run: eurohoops ingest, then eurohoops build")
        raise typer.Exit(1)
    report = continuity_report(team_seasons)
    write_json(TEAM_CONTINUITY_REPORT, report)
    for competition, block in report.items():
        for item in block["review_name_changes"]:  # names (Greek for the GBL) are in the file
            typer.echo(f"review: {competition} {item['season']} {item['team']} changed name")
    typer.echo(f"wrote {TEAM_CONTINUITY_REPORT}")


@app.command()
def stints(
    mart: Annotated[
        bool,
        typer.Option(help="Build the 2011+ stints mart and write reports/stints_mart.json instead"),
    ] = False,
) -> None:
    """Validate EuroLeague stints on the seeded 50-game sample (reads the local raw cache).

    With ``--mart``: build the ``stints`` and ``stint_game_checks`` tables for every cached
    game from 2011-12 (needs ``eurohoops build`` first for ``team_games``).
    """
    if mart:
        games = read_games(MART_PATH, EUROLEAGUE.name)
        team_games = read_table(MART_PATH, "team_games", EUROLEAGUE.name)
        if team_games is None:
            log.error("no team_games in the marts; run: eurohoops build")
            raise typer.Exit(code=1)
        built = build_stints_mart(EUROLEAGUE.raw_dir, games)
        write_tables(MART_PATH, {"stints": built.stints, "stint_game_checks": built.checks})
        summary = mart_report(built, games, team_games)
        write_json(STINTS_MART_REPORT, summary)
        spans = {
            span: "no games" if rate is None else f"{rate:.1%}"
            for span, rate in (
                ("2011-14", summary["pass_rate_2011_2014"]),
                ("2015+", summary["pass_rate_2015_on"]),
            )
        }
        typer.echo(
            f"{summary['games']} games, {summary['stints']} stints; pass rate 2011-14 "
            f"{spans['2011-14']}, 2015+ {spans['2015+']}; wrote {STINTS_MART_REPORT}"
        )
        return
    report = validate_sample(EUROLEAGUE.raw_dir)
    write_json(STINT_REPORT, report)
    rates = ", ".join(f"{check} {rate:.0%}" for check, rate in report["pass_rate"].items())
    typer.echo(
        f"{report['games']} games: all checks {report['all_checks_pass_rate']:.0%} ({rates}); "
        f"wrote {STINT_REPORT}"
    )


app.command()(research.shots)


app.command(name="free-throws")(research.free_throws)


app.command(name="gbl-stints")(research.gbl_stints)


@app.command()
def possessions() -> None:
    """Validate ``team_games``: coverage, points, box vs play-by-play possessions."""
    table = read_table(MART_PATH, "team_games")
    missing = read_table(MART_PATH, "team_games_missing")
    if table is None or missing is None:
        log.error("no team_games in the marts; run: eurohoops build")
        raise typer.Exit(code=1)
    report = possession_report(
        table,
        missing=missing,
        games=_both_games(),
        euroleague_raw=EUROLEAGUE.raw_dir,
        gbl_raw=GBL.raw_dir,
        gbl_pbp=_gbl_pbp(),
    )
    write_json(POSSESSION_REPORT, report)
    sample = report["euroleague_pbp_sample"]
    agreement = (
        "no EuroLeague play-by-play cached"
        if sample is None
        else f"EuroLeague sample within 2: {sample['within_tolerance_share_of_teams']:.1%} of teams"
    )
    typer.echo(
        f"{len(report['missing_games'])} games missing, "
        f"{len(report['points_mismatches'])} points mismatches; {agreement}; "
        f"wrote {POSSESSION_REPORT}"
    )


def _backtest_m3_gbl(score_test: bool, tuning_only: bool) -> None:
    """``backtest --model m3 --competition gbl`` (week 9-12 H8): EL SPM transfer to GBL.

    Requires the committed EuroLeague M3 verdict (``reports/backtest_m3.json``) and GBL M1
    parameters. No MLflow logging for GBL."""
    if tuning_only:
        log.error("m3 GBL backtest has no variant choice; --tuning-only is not supported")
        raise typer.Exit(code=1)
    if not M3.report.exists():
        log.error("no committed M3 report; run: eurohoops backtest --model m3")
        raise typer.Exit(code=1)
    m3_report = json.loads(M3.report.read_text(encoding="utf-8"))
    chosen = m3_report.get("chosen")
    if not chosen or chosen.get("variant") != "rapm_spm":
        log.error("M3 chosen variant must be rapm_spm (see reports/backtest_m3.json)")
        raise typer.Exit(code=1)
    if GBL.m1 is None or not GBL.m1.report.exists():
        log.error(
            "no committed GBL M1 report; run: eurohoops backtest --model m1 --competition gbl"
        )
        raise typer.Exit(code=1)
    tuned_m1 = json.loads(GBL.m1.report.read_text(encoding="utf-8"))["tuned"]
    el_games = read_games(MART_PATH, EUROLEAGUE.name)
    stints = read_table(MART_PATH, "stints")
    checks = read_table(MART_PATH, "stint_game_checks")
    if stints is None or checks is None:
        log.error("stints/stint_game_checks missing; run: eurohoops stints --mart")
        raise typer.Exit(code=1)
    el_player_games = build_box_games(EUROLEAGUE.raw_dir, el_games).players
    gbl_games = read_games(MART_PATH, GBL.name)
    gbl_team_games = read_table(MART_PATH, "team_games", GBL.name)
    if gbl_team_games is None:
        log.error("no GBL team_games in the marts; run: eurohoops build")
        raise typer.Exit(code=1)
    gbl_box = build_gbl_player_games(GBL.raw_dir, gbl_games, gbl_team_games)
    el = el_spm_models(el_games, el_player_games, stints, checks, chosen)
    report = run_m3_gbl_backtest(
        gbl_games,
        gbl_team_games,
        gbl_box.table,
        spec=M3_GBL,
        el=el,
        tuned_m1=tuned_m1,
        box_only_fn=partial(box_only_margins, grid=BoxGrid()),
        pir_fn=pir_margins,
        box_pages_skipped=gbl_box.skipped,
        score_test=score_test,
    )
    write_json(M3_GBL.report, report)
    typer.echo(f"{M3_GBL.report}\n{format_m3_gbl_table(report)}")


def _backtest_m3(
    competition: CompetitionName, score_test: bool, tuning_only: bool, tracking_uri: str
) -> None:
    """``backtest --model m3`` (week 9-12 H1/H8): EuroLeague or GBL SPM transfer."""
    if competition is CompetitionName.gbl:
        _backtest_m3_gbl(score_test, tuning_only)
        return
    if competition is not CompetitionName.euroleague:
        log.error("m3 backtest supports EuroLeague and GBL only")
        raise typer.Exit(code=1)
    if EUROLEAGUE.m1 is None or not EUROLEAGUE.m1.report.exists():
        log.error("no committed M1 report; run: eurohoops backtest --model m1")
        raise typer.Exit(code=1)
    tuned_m1 = json.loads(EUROLEAGUE.m1.report.read_text(encoding="utf-8"))["tuned"]
    games = read_games(MART_PATH, EUROLEAGUE.name)
    team_games = read_table(MART_PATH, "team_games", EUROLEAGUE.name)
    stints = read_table(MART_PATH, "stints")
    checks = read_table(MART_PATH, "stint_game_checks")
    if team_games is None or stints is None or checks is None:
        log.error(
            "stints/stint_game_checks/team_games missing; run: eurohoops build / stints --mart"
        )
        raise typer.Exit(code=1)
    player_games = build_box_games(EUROLEAGUE.raw_dir, games).players
    report = run_m3_backtest(
        games,
        team_games,
        player_games,
        stints,
        checks,
        spec=M3,
        tuned_m1=tuned_m1,
        box_only_fn=partial(box_only_margins, grid=BoxGrid()),
        pir_fn=pir_margins,
        tuning_only=tuning_only,
        score_test=score_test,
    )
    write_json(M3.report, report)
    typer.echo(f"{M3.report}\n{format_m3_table(report)}")
    run_id = log_m3_backtest(report, tracking_uri)
    if run_id is not None:
        typer.echo(f"MLflow parent run {run_id}")


@app.command()
def backtest(
    *,
    competition: CompetitionOption = CompetitionName.euroleague,
    model: Annotated[
        ModelName, typer.Option(help="elo (live model), m1, m2 (shots) or m3 (player impact)")
    ] = ModelName.elo,
    score_test: Annotated[
        bool,
        typer.Option(help="M1/M2/M3: also score the test seasons (only after the gate verdict)"),
    ] = False,
    tuning_only: Annotated[
        bool,
        typer.Option(help="M3: only the tuning grid and chosen variant (the verdict commit)"),
    ] = False,
    search: Annotated[
        bool,
        typer.Option(help="M2: run the declared Optuna study first (only when F4 changes)"),
    ] = False,
    tracking_uri: Annotated[
        str | None,
        typer.Option(help="M1/M3: MLflow tracking URI (default: SQLite store under ./mlruns)"),
    ] = None,
) -> None:
    """Tune and score a model vs its baselines; the Elo live report holds the live parameters."""
    if model is ModelName.m2:
        research.backtest_m2(score_test, search, tracking_uri or default_tracking_uri())
        return
    if model is ModelName.m3:
        _backtest_m3(competition, score_test, tuning_only, tracking_uri or default_tracking_uri())
        return
    comp = COMPETITIONS[competition]
    games = read_games(MART_PATH, comp.name)
    if model is ModelName.m1:
        team_games = read_table(MART_PATH, "team_games", comp.name)
        if team_games is None or comp.m1 is None:
            log.error("no team_games in the marts; run: eurohoops build")
            raise typer.Exit(code=1)
        report = run_m1_backtest(
            games, team_games, comp.m1, score_test=score_test, progress=log.info
        )
        write_json(comp.m1.report, report)
        typer.echo(f"{comp.m1.report}\n{format_m1_table(report)}")
        run_id = log_backtest(
            report, comp.name, report["data_sha256"], tracking_uri or default_tracking_uri()
        )
        if run_id is not None:
            typer.echo(f"MLflow parent run {run_id}")
        return
    for spec in (comp.live_backtest, *comp.history_backtests):
        report = run_backtest(games, spec)
        write_json(spec.report, report)
        typer.echo(f"{spec.report}\n{format_table(report)}")


app.command(name="shot-quality")(research.shot_quality)
app.command(name="m3-players")(research.m3_players)


app.command(name="shot-charts")(research.shot_charts)


@app.command()
def predict(
    competition: CompetitionOption = CompetitionName.euroleague,
    window_hours: Annotated[float, typer.Option(help="Predict games tipping off within")] = 36.0,
) -> None:
    """Append pre-tip-off predictions for upcoming live-season games to the public log."""
    comp = COMPETITIONS[competition]
    live = _live(comp)
    try:
        added = predict_upcoming(
            live.games,
            live.model,
            log_path=comp.prediction_log,
            season=live.season,
            replay_from=live.replay_from,
            window=timedelta(hours=window_hours),
            clock=utc_now,
        )
        typer.echo(f"{added} predictions appended to {comp.prediction_log}")
        m1 = None if comp.m1 is None else load_m1(comp.m1.report)
        if m1 is not None and comp.m1_prediction_log is not None:
            added = predict_upcoming_m1(
                live.games,
                _team_games(comp.name),
                m1,
                log_path=comp.m1_prediction_log,
                season=live.season,
                replay_from=live.replay_from,
                window=timedelta(hours=window_hours),
                clock=utc_now,
            )
            typer.echo(f"{added} M1 predictions appended to {comp.m1_prediction_log}")
    except LatePredictionError as exc:
        log.error("refusing to log: %s", exc)
        raise typer.Exit(code=1) from exc


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
    live = _live(comp)
    card = build_scorecard(
        comp.prediction_log,
        live.games,
        live.model,
        utc_now(),
        manual_pushes=comp.manual_pushes,
        odds_path=comp.odds_log,
        m1_log_path=comp.m1_prediction_log,
    )
    write_json(comp.scorecard, card)
    typer.echo(json.dumps(card, indent=2))


@app.command()
def publish() -> None:
    """Write the site data (web/src/data/site.json) the Astro front-end renders."""
    sections = []
    for title, comp in (("EuroLeague", EUROLEAGUE), ("Greek Basket League", GBL)):
        live = _live(comp)
        sections.append(
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
                games=live.games,
                names=dict(read_teams(MART_PATH, comp.name).itertuples(index=False)),
                model=live.model,
                season=live.season,
                replay_from=live.replay_from,
            )
        )
    write_json(SITE_DATA, site_data(sections, utc_now()))
    typer.echo(f"wrote {SITE_DATA}")


@app.command("export-stats")
def export_stats(
    from_cache: Annotated[
        bool,
        typer.Option(help="Read games from the cached schedules instead of the marts (no DuckDB)"),
    ] = False,
    raw_dir: Annotated[Path, typer.Option(help="EuroLeague raw cache")] = EUROLEAGUE.raw_dir,
    out: Annotated[Path, typer.Option(help="Where the stats JSON goes")] = STATS_DIR,
) -> None:
    """Write the EuroLeague stats-site data (players, teams, game logs, shot hex bins)."""
    if from_cache:
        games, teams = load_cached_games(raw_dir)
    else:
        games, teams = (
            read_games(MART_PATH, EUROLEAGUE.name),
            read_teams(MART_PATH, EUROLEAGUE.name),
        )
    inputs = Inputs(
        games=games,
        names=dict(teams.itertuples(index=False)),
        box=build_box_games(raw_dir, games),
        shots=build_shots(raw_dir, games),
        codes=DISPLAY_CODES[EUROLEAGUE.name],
        live_season=LIVE_SEASON,
    )
    files = build_payloads(inputs, utc_now())
    write_stats(out, files)
    seasons = files["meta.json"]["seasons"]
    typer.echo(
        f"wrote {len(files)} files to {out}: {len(seasons)} seasons, "
        f"{len(files['players.json']['players'])} players, "
        f"{len(inputs.box.missing)} played games without a box score"
    )
