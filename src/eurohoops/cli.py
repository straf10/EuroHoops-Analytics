"""``eurohoops`` command line: ingest -> build -> backtest -> predict -> score -> publish."""

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from functools import partial
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import pandas as pd
import typer
from fastapi.testclient import TestClient

from eurohoops import research
from eurohoops.api.app import create_app
from eurohoops.api.export import (
    WEB_DIR,
    RawStore,
    publish_files,
    stats_files,
    write_publish,
    write_stats_files,
)
from eurohoops.api.readmodel import Store, stats_inputs
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
    INJURY_LOG,
    INJURY_RAW_DIR,
    LIVE_SEASON,
    M2_REPORT,
    M3,
    M3_GBL,
    M3_PLAYERS_REPORT,
    M4,
    M5,
    M5_GBL,
    M6,
    M6_BOARD,
    M6_GBL,
    M6_PROJECTIONS,
    M6_SIMILARITY,
    M7,
    M7_GBL,
    M7_V2,
    MART_PATH,
    ODDS_CALLS,
    ODDS_RAW_DIR,
    ODDS_TEAMS,
    PLAYER_BIOS,
    PLAYER_XWALK_FILE,
    POSSESSION_REPORT,
    SIM_LATEST,
    SIM_LOGS,
    SIM_UNGATED,
    SITE_DATA,
    SQL_DIR,
    STINT_REPORT,
    STINTS_MART_REPORT,
    TEAM_CONTINUITY_REPORT,
    Competition,
    M5Backtest,
    M6Backtest,
    M7Backtest,
)
from eurohoops.eval.backtest import TunedModel, format_table, load_tuned_model, run_backtest
from eurohoops.eval.m1_backtest import format_m1_table, run_m1_backtest
from eurohoops.eval.m3_backtest import (
    TunedRapm,
    build_rapm_inputs,
    format_m3_table,
    rapm_margins,
    run_m3_backtest,
)
from eurohoops.eval.m3_gbl_backtest import (
    ElSpm,
    choose_model,
    el_spm_models,
    format_m3_gbl_table,
    run_m3_gbl_backtest,
)
from eurohoops.eval.m5_backtest import (
    Choice,
    M5Inputs,
    PlayerPartFn,
    format_m5_table,
    run_m5_backtest,
)
from eurohoops.eval.m6_backtest import (
    M6Inputs,
    SpmFn,
    TargetInputs,
    World,
    format_m6_table,
    prepare,
    run_m6_backtest,
)
from eurohoops.eval.m7_backtest import M7Inputs, format_m7_table, run_m7_backtest, scored_splits
from eurohoops.eval.m7_gate_v2 import format_gate_v2, run_gate_v2
from eurohoops.eval.scorecard import build_scorecard
from eurohoops.eval.tracking import (
    default_tracking_uri,
    log_backtest,
    log_m3_backtest,
    log_m5_backtest,
    log_m6_backtest,
    log_m7_backtest,
)
from eurohoops.ingest import euroleague, gbl
from eurohoops.ingest.bios import build_bios, ingest_bios
from eurohoops.ingest.http import Fetcher, make_client
from eurohoops.injuries import InjuryPaths, record_injuries, team_name_map
from eurohoops.live_m1 import load_m1, predict_upcoming_m1
from eurohoops.live_m5 import live_spec, load_m5, predict_upcoming_m5
from eurohoops.live_m6 import (
    board_report,
    inputs_digest,
    live_checkpoint,
    live_people,
    live_projections,
    person_names,
    personal_fields,
    player_rows,
    projections_report,
    query_rows,
    similarity_report,
    undervalued,
)
from eurohoops.live_sim import (
    LIVE_SIMS,
    latest_report,
    load_sim,
    log_rows,
    run_live_sim,
    write_live_sim,
)
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
from eurohoops.models.board import (
    DIMENSIONS,
    Stability,
    board,
    derive_n_min,
    fg3_observations,
    fg3_rotation_n,
    on_off_observations,
    rotation_n,
    rotation_player_seasons,
    shot_making_observations,
    stability,
)
from eurohoops.models.box_impact import STAT_COLUMNS, BoxGrid, box_only_margins, pir_margins
from eurohoops.models.elo import FloatArray
from eurohoops.models.minutes import expected_possessions
from eurohoops.models.player_seasons import (
    AGES_SCHEMA,
    COUNT_STATS,
    build_player_seasons,
    person_ids,
    player_ages,
    read_xwalk_file,
)
from eurohoops.models.projection import Translation
from eurohoops.models.similarity import BOX_FEATURES, embed, neighbours, raw_features
from eurohoops.models.spm import fit_spm
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
from eurohoops.parse.player_names import gbl_names
from eurohoops.parse.possession_report import possession_report
from eurohoops.parse.schemas import validated
from eurohoops.parse.stints import validate_sample
from eurohoops.parse.stints_mart import build_stints_mart, mart_report
from eurohoops.parse.team_box import TEAM_GAMES_SCHEMA, build_team_games
from eurohoops.predict import LatePredictionError, predict_upcoming
from eurohoops.publish import DISPLAY_CODES
from eurohoops.sim.formats import season_format
from eurohoops.sim.played import regulation_scores
from eurohoops.stats.box import build_box_games
from eurohoops.stats.downloads import DOWNLOADS_DIR, build_downloads, write_downloads
from eurohoops.stats.export import STATS_DIR

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
    m4 = "m4"
    m5 = "m5"
    m6 = "m6"
    m7 = "m7"


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
    *,
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
    from_season: Annotated[
        int | None,
        typer.Option(help="Stage every season from this start year to the live one"),
    ] = None,
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
    history = range(from_season, LIVE_SEASON + 1) if from_season is not None else ()
    chosen = sorted({*(seasons or []), *extra, *history}) or list(comp.default_seasons)
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


app.command(name="entity")(research.entity)


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


def _m5_inputs(
    comp: Competition, other: Competition, spec: M5Backtest, club_map: dict[str, str]
) -> tuple[M5Inputs, pd.DataFrame]:
    """M5's inputs for ``comp``: every game up to the last test season (the live season never
    enters, so the daily log cannot move the report), the other competition's games for rest,
    M1's committed parameters and comparison Elo. Also returns the backtest frame (M3's order)."""
    if comp.m1 is None or not comp.m1.report.exists():
        log.error("no committed M1 report for %s; run: eurohoops backtest --model m1", comp.name)
        raise typer.Exit(code=1)
    m1_report = json.loads(comp.m1.report.read_text(encoding="utf-8"))
    last = spec.test[-1]
    games = read_games(MART_PATH, comp.name)
    games = games[games["season"] <= last].reset_index(drop=True)
    other_games = read_games(MART_PATH, other.name)
    other_games = other_games[other_games["season"] <= last].reset_index(drop=True)
    team_games = read_table(MART_PATH, "team_games", comp.name)
    if team_games is None:
        log.error("no team_games in the marts; run: eurohoops build")
        raise typer.Exit(code=1)
    team_games = team_games[team_games["game_id"].isin(set(games["game_id"]))]
    if comp is EUROLEAGUE:
        player_games = build_box_games(comp.raw_dir, games).players
    else:
        player_games = build_gbl_player_games(comp.raw_dir, games, team_games).table
    inputs = M5Inputs(
        games=games,
        team_games=team_games.reset_index(drop=True),
        player_games=player_games,
        other_games=other_games,
        club_map=club_map,
        tuned_m1=m1_report["tuned"],
        elo=m1_report["comparison_elo"],
    )
    frame = (
        games[games["season"].between(spec.warmup[0], last)]
        .sort_values("tipoff_utc")
        .reset_index(drop=True)
    )
    return inputs, frame


def _same_frame(expected: pd.DataFrame, frame: pd.DataFrame) -> None:
    if not frame["game_id"].astype(str).equals(expected["game_id"].astype(str)):
        raise ValueError("the player part was fitted on a different frame than the harness's")


def _m5_player_part_el(inputs: M5Inputs, frame: pd.DataFrame) -> PlayerPartFn:
    """M3's committed ``rapm_spm`` (``reports/backtest_m3.json`` ``chosen``, never re-tuned),
    fitted walk-forward once over the frame; the player part is H-c's ``rapm_margins``."""
    chosen = json.loads(M3.report.read_text(encoding="utf-8"))["chosen"]
    if chosen["variant"] != "rapm_spm":
        log.error("M3 chosen variant must be rapm_spm (see reports/backtest_m3.json)")
        raise typer.Exit(code=1)
    stints = read_table(MART_PATH, "stints")
    checks = read_table(MART_PATH, "stint_game_checks")
    if stints is None or checks is None:
        log.error("stints/stint_game_checks missing; run: eurohoops stints --mart")
        raise typer.Exit(code=1)
    extra = {k: chosen[k] for k in ("target_half_life_days", "target_ridge_o", "target_ridge_d")}
    tuned = TunedRapm(
        chosen["half_life_days"],
        chosen["ridge_o"],
        chosen["ridge_d"],
        chosen["tuning_rmse"],
        grid={},
        extra={**extra, "k": chosen["k"], "alpha": chosen["alpha"]},
    )
    rapm_inputs = build_rapm_inputs(stints, checks, frame, inputs.player_games)
    wf = fit_spm(frame, rapm_inputs, tuned)
    possessions = expected_possessions(frame, inputs.team_games)

    def player_part(games: pd.DataFrame, shares: pd.DataFrame) -> FloatArray:
        _same_frame(frame, games)
        return rapm_margins(games, wf, shares, possessions)

    return player_part


def _m5_player_part_gbl(inputs: M5Inputs, frame: pd.DataFrame, spec: M5Backtest) -> PlayerPartFn:
    """GBL box-only player ratings (J-g: no GBL RAPM), tuned on the tuning seasons per call."""
    possessions = expected_possessions(frame, inputs.team_games)
    rated = (frame["played"] & ~frame["forfeit"]).to_numpy()
    tuning = rated & frame["season"].isin(spec.tuning).to_numpy()

    def player_part(games: pd.DataFrame, shares: pd.DataFrame) -> FloatArray:
        _same_frame(frame, games)
        result = box_only_margins(
            games, inputs.player_games, shares, possessions, tuning, grid=BoxGrid()
        )
        return result.margin

    return player_part


def _backtest_m5(
    competition: CompetitionName, score_test: bool, tuning_only: bool, tracking_uri: str
) -> None:
    """``backtest --model m5`` (week 14-16 J4): the roster-aware predictor vs M1, Elo and B0."""
    if competition is CompetitionName.euroleague:
        spec, comp, other, club_map = M5, EUROLEAGUE, GBL, dict(GREEK_EL_CLUBS)
    else:
        spec, comp, other = M5_GBL, GBL, EUROLEAGUE
        club_map = {el: gbl_id for gbl_id, el in GREEK_EL_CLUBS.items()}
    fixed = None
    if comp is GBL:  # J-g: the EuroLeague verdict's choice, no GBL-specific choice
        if tuning_only or not M5.report.exists():
            log.error("GBL M5 scores the committed EuroLeague verdict; run the EuroLeague first")
            raise typer.Exit(code=1)
        fixed = Choice(**json.loads(M5.report.read_text(encoding="utf-8"))["chosen"]["choice"])
    started = time.perf_counter()
    inputs, frame = _m5_inputs(comp, other, spec, club_map)
    if comp is EUROLEAGUE:
        player_part = _m5_player_part_el(inputs, frame)
    else:
        player_part = _m5_player_part_gbl(inputs, frame, spec)
    report, predictions = run_m5_backtest(
        inputs,
        spec=spec,
        player_part=player_part,
        tuning_only=tuning_only,
        score_test=score_test,
        fixed=fixed,
    )
    write_json(spec.report, report)
    predictions.to_csv(spec.games_report, index=False, lineterminator="\n")
    typer.echo(f"{spec.report}\n{format_m5_table(report)}")
    typer.echo(f"RUNTIME backtest m5 {comp.name}: {time.perf_counter() - started:.0f} s")
    run_id = log_m5_backtest(report, comp.name, tracking_uri)
    if run_id is not None:
        typer.echo(f"MLflow run {run_id}")


def _m7_inputs(comp: Competition, spec: M7Backtest, seasons: tuple[int, ...]) -> M7Inputs:
    """M7's inputs for ``comp``: every game up to the last test season (the live season never
    enters, so the daily log cannot move the report), M1's committed parameters and comparison
    Elo, and the EuroLeague's regulation scores of the scored regular seasons."""
    if comp.m1 is None or not comp.m1.report.exists():
        log.error("no committed M1 report for %s; run: eurohoops backtest --model m1", comp.name)
        raise typer.Exit(code=1)
    m1_report = json.loads(comp.m1.report.read_text(encoding="utf-8"))
    games = read_games(MART_PATH, comp.name)
    games = games[games["season"] <= spec.test[-1]].reset_index(drop=True)
    team_games = read_table(MART_PATH, "team_games", comp.name)
    if team_games is None:
        log.error("no team_games in the marts; run: eurohoops build")
        raise typer.Exit(code=1)
    team_games = team_games[team_games["game_id"].isin(set(games["game_id"]))]
    scored = games[(games["phase"] == "RS") & games["season"].isin(seasons)]
    return M7Inputs(
        competition=comp.name,
        games=games,
        team_games=team_games.reset_index(drop=True),
        regulation=regulation_scores(comp.raw_dir, scored) if comp is EUROLEAGUE else {},
        tuned_m1=m1_report["tuned"],
        elo=m1_report["comparison_elo"],
    )


def _backtest_m7(
    competition: CompetitionName, score_test: bool, tuning_only: bool, tracking_uri: str
) -> None:
    """``backtest --model m7`` (week 14-16 K4): season simulation vs the real final tables."""
    spec, comp = (M7, EUROLEAGUE) if competition is CompetitionName.euroleague else (M7_GBL, GBL)
    fixed = None
    if comp is GBL:  # K-a: the EuroLeague verdict, no GBL-specific choice
        if tuning_only or not M7.report.exists():
            log.error("GBL M7 scores the committed EuroLeague verdict; run the EuroLeague first")
            raise typer.Exit(code=1)
        fixed = json.loads(M7.report.read_text(encoding="utf-8"))["chosen"]["key"]
    started = time.perf_counter()
    inputs = _m7_inputs(comp, spec, tuple(scored_splits(spec, tuning_only, score_test)))
    report, rows = run_m7_backtest(
        inputs, spec=spec, tuning_only=tuning_only, score_test=score_test, fixed=fixed
    )
    write_json(spec.report, report)
    rows.to_csv(spec.teams_report, index=False, lineterminator="\n")
    typer.echo(f"{spec.report}\n{format_m7_table(report)}")
    typer.echo(f"RUNTIME backtest m7 {comp.name}: {time.perf_counter() - started:.0f} s")
    run_id = log_m7_backtest(report, comp.name, tracking_uri)
    if run_id is not None:
        typer.echo(f"MLflow run {run_id}")


def _backtest_m7_v2(competition: CompetitionName) -> None:
    """``backtest --model m7 --gate-v2``: the rolling-origin gate v2 (docs/models/m7.md), run once;
    EuroLeague only (the GBL is not gated)."""
    if competition is not CompetitionName.euroleague:
        log.error("M7 gate v2 is the EuroLeague's; the GBL is not gated")
        raise typer.Exit(code=1)
    started = time.perf_counter()
    # every season, the selection one included: its overtime games need their regulation scores
    inputs = _m7_inputs(EUROLEAGUE, M7, M7_V2.tuning)
    report, rows = run_gate_v2(inputs, spec=M7_V2)
    write_json(M7_V2.report, report)
    rows.to_csv(M7_V2.teams_report, index=False, lineterminator="\n")
    typer.echo(f"{M7_V2.report}\n{format_gate_v2(report)}")
    typer.echo(f"RUNTIME backtest m7 gate v2: {time.perf_counter() - started:.0f} s")


def _m6_brapm(xwalk: pd.DataFrame) -> pd.DataFrame:
    """The season-end BRAPM snapshots of ``reports/m3_players.json`` as ``IMPACT_SCHEMA`` rows
    (``total`` and ``sd_total`` of the players the snapshot saw, mapped to persons)."""
    report = json.loads(M3_PLAYERS_REPORT.read_text(encoding="utf-8"))
    frames = []
    for season, block in report["seasons"].items():
        players = pd.DataFrame(block["players"])
        players = players[players["seen"]]
        ids = person_ids(players["player_id"], EUROLEAGUE.name, xwalk)
        frames.append(
            pd.DataFrame(
                {
                    "person_id": ids["person_id"].to_numpy(),
                    "competition": EUROLEAGUE.name,
                    "season": int(season),
                    "stat": "brapm",
                    "value": players["total"].to_numpy(dtype="float64"),
                    "sd": players["sd_total"].to_numpy(dtype="float64"),
                }
            )
        )
    rows = pd.concat(frames, ignore_index=True)
    rows = rows.drop_duplicates(["person_id", "competition", "season", "stat"])
    return rows.sort_values(["season", "person_id"]).reset_index(drop=True)


def _m6_translations(stats: tuple[str, ...] = COUNT_STATS) -> dict[int, Translation]:
    """M4's fit for each target season (``reports/m4_translation.json``) for ``stats``: the count
    stats, or SPM's eleven for the undervalued list."""
    fits = json.loads(M4.translation_report.read_text(encoding="utf-8"))["fits_by_target_season"]
    return {
        int(season): Translation(
            delta={s: float(fit["translate"][s]["delta"]) for s in stats},
            c={s: float(fit["translate"][s]["c"]) for s in stats},
            target_season=int(season),
        )
        for season, fit in fits.items()
    }


def _league_rates(player_games: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per competition and season the league's per-100 rate of each SPM stat (all lines with
    seconds played, possession-weighted), indexed by (competition, season)."""
    frames = []
    for competition, all_lines in player_games.items():
        lines = all_lines[all_lines["sec"] > 0]
        sums = lines.groupby("season")[[*STAT_COLUMNS, "poss"]].sum()
        rates = 100.0 * sums[list(STAT_COLUMNS)].div(sums["poss"], axis=0)
        frames.append(rates.assign(competition=competition).set_index("competition", append=True))
    return pd.concat(frames).reorder_levels(["competition", "season"]).sort_index()


def _m6_spm(el: ElSpm, league: pd.DataFrame) -> SpmFn:
    """The injected SPM: ``before_season``'s EuroLeague SPM model (fitted on data before that
    season; the latest one when there is none for it) applied to each row's per-100 box rates
    minus the league rate of the row's previous season (never the row's own season: at a
    checkpoint that would read games after the cutoff; the first season has none and uses
    its own, which no checkpoint reads); SPM = O + D. Unshrunk rates. A season without its own
    model raises: falling back to a later model would read data after the season's start."""

    def spm(frame: pd.DataFrame, before_season: int) -> "pd.Series[float]":
        if before_season not in el.fit_time:
            raise ValueError(f"no EuroLeague SPM model for season {before_season}")
        model = choose_model(el, el.fit_time[before_season])
        if model is None:
            raise ValueError(f"no EuroLeague SPM model for season {before_season}")
        base = []
        for competition, season in zip(frame["competition"], frame["season"], strict=True):
            rates = league.loc[competition]
            earlier = rates.loc[: int(season) - 1]
            base.append((earlier if len(earlier) else rates).iloc[-1].to_numpy(dtype="float64"))
        features = frame[list(model.stats)].to_numpy(dtype="float64") - np.array(base)
        o, d = model.predict(features)
        return pd.Series(o + d, index=frame.index)

    return spm


def _m6_bios() -> pd.DataFrame:
    if not PLAYER_BIOS.exists():
        log.error("no player bios; run: eurohoops bios")
        raise typer.Exit(code=1)
    return pd.read_parquet(PLAYER_BIOS)


def _m6_inputs(spec: M6Backtest, *, live: bool = False) -> M6Inputs:
    """M6's inputs: both competitions' games and player games up to the last test season (the live
    season never enters, so the daily log cannot move the report), the crosswalk, in-memory ages,
    the BRAPM snapshots, M4's translations and the SPM wired from M3's EuroLeague models.
    ``live`` (``eurohoops project``): up to the live season, the committed crosswalk file (D25)
    and no ages (the chosen variant does not age, and the daily workflow has no bios)."""
    last = LIVE_SEASON if live else spec.test[-1]
    games = {c.name: read_games(MART_PATH, c.name) for c in (EUROLEAGUE, GBL)}
    games = {name: g[g["season"] <= last].reset_index(drop=True) for name, g in games.items()}
    team_games = read_table(MART_PATH, "team_games", GBL.name)
    xwalk = read_xwalk_file(PLAYER_XWALK_FILE) if live else read_table(MART_PATH, "player_xwalk")
    stints = read_table(MART_PATH, "stints")
    checks = read_table(MART_PATH, "stint_game_checks")
    if team_games is None or xwalk is None or stints is None or checks is None:
        log.error("team_games/player_xwalk/stints missing; run: eurohoops build / entity / stints")
        raise typer.Exit(code=1)
    team_games = team_games[team_games["game_id"].isin(set(games[GBL.name]["game_id"]))]
    player_games = {
        EUROLEAGUE.name: build_box_games(EUROLEAGUE.raw_dir, games[EUROLEAGUE.name]).players,
        GBL.name: build_gbl_player_games(GBL.raw_dir, games[GBL.name], team_games).table,
    }
    ages = (
        validated(pd.DataFrame({"person_id": [], "season": [], "age": []}), AGES_SCHEMA)
        if live
        else player_ages(_m6_bios(), xwalk, build_player_seasons(player_games, xwalk))
    )
    chosen = json.loads(M3.report.read_text(encoding="utf-8"))["chosen"]
    if chosen["variant"] != "rapm_spm":
        log.error("M3 chosen variant must be rapm_spm (see reports/backtest_m3.json)")
        raise typer.Exit(code=1)
    el_games = read_games(MART_PATH, EUROLEAGUE.name)
    el = el_spm_models(
        el_games[el_games["played"]] if live else el_games,
        player_games[EUROLEAGUE.name],
        stints,
        checks,
        chosen,
        last=LIVE_SEASON if live else M3.test[-1],
    )
    return M6Inputs(
        games=games,
        player_games=player_games,
        xwalk=xwalk,
        ages=ages,
        brapm=_m6_brapm(xwalk),
        spm=_m6_spm(el, _league_rates(player_games)),
        translations=_m6_translations(),
    )


def _backtest_m6(
    competition: CompetitionName, score_test: bool, tuning_only: bool, tracking_uri: str
) -> None:
    """``backtest --model m6`` (weeks 16-18 L6): player projections vs Marcel and the naive
    baselines, next-season and rest-of-season."""
    spec, comp = (M6, EUROLEAGUE) if competition is CompetitionName.euroleague else (M6_GBL, GBL)
    fixed = None
    if comp is GBL:  # L-a: the EuroLeague verdict, no GBL-specific choice
        if tuning_only or not M6.report.exists():
            log.error("GBL M6 scores the committed EuroLeague verdict; run the EuroLeague first")
            raise typer.Exit(code=1)
        chosen = json.loads(M6.report.read_text(encoding="utf-8"))["chosen"]
        fixed = {"variant": chosen["variant"], "half_life": chosen["half_life"]}
    started = time.perf_counter()
    inputs = _m6_inputs(spec)
    report, players = run_m6_backtest(
        inputs,
        spec=spec,
        competition=comp.name,
        tuning_only=tuning_only,
        score_test=score_test,
        fixed=fixed,
    )
    write_json(spec.report, report)
    spec.players_report.parent.mkdir(parents=True, exist_ok=True)
    players.to_csv(spec.players_report, index=False, lineterminator="\n")
    typer.echo(f"{spec.report}\n{format_m6_table(report)}")
    typer.echo(f"RUNTIME backtest m6 {comp.name}: {time.perf_counter() - started:.0f} s")
    run_id = log_m6_backtest(report, comp.name, tracking_uri)
    if run_id is not None:
        typer.echo(f"MLflow run {run_id}")


def _live_names(inputs: M6Inputs) -> dict[str, str]:
    """Display names: EuroLeague box spellings, the GBL live season's box pages and, locally, the
    ``player_names`` mart for the older GBL seasons."""
    live = inputs.player_games[GBL.name]
    live = live[live["season"] == LIVE_SEASON]
    box = live[["game_id", "team", "player_id", "sec"]].rename(columns={"sec": "seconds"})
    frames = [gbl_names(GBL.raw_dir, box)]
    mart = read_table(MART_PATH, "player_names", GBL.name) if MART_PATH.exists() else None
    if mart is not None:
        frames.insert(0, mart[mart["season"] < LIVE_SEASON])
    gbl = pd.concat(frames, ignore_index=True)
    return person_names(inputs.player_games[EUROLEAGUE.name], gbl, inputs.xwalk)


def _m2_shots(season: int) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """The ``shots`` and ``shot_xpts`` marts up to ``season`` and M2's chosen xPTS variant."""
    shots, xpts = read_table(MART_PATH, "shots"), read_table(MART_PATH, "shot_xpts")
    if shots is None or xpts is None:
        log.error("shots/shot_xpts missing; run: eurohoops shots, backtest --model m2")
        raise typer.Exit(code=1)
    variant = str(json.loads(M2_REPORT.read_text(encoding="utf-8"))["gate"]["chosen"])
    return shots[shots["season"] <= season], xpts, variant


def _m6_board(
    world: World, season: int, shots: pd.DataFrame, xpts: pd.DataFrame, variant: str
) -> tuple[pd.DataFrame, dict[str, Stability]]:
    """The over/under board of ``season`` (D23), stabilities from the M6 tuning seasons."""
    stints, checks = read_table(MART_PATH, "stints"), read_table(MART_PATH, "stint_game_checks")
    if stints is None or checks is None:
        log.error("no stints mart; run: eurohoops stints --mart")
        raise typer.Exit(code=1)
    xwalk = world.inputs.xwalk
    seasons = json.loads(M3_PLAYERS_REPORT.read_text(encoding="utf-8"))["seasons"]
    brapm = pd.concat(
        [
            pd.DataFrame(block["players"])[["player_id", "total", "sd_total"]].assign(season=int(s))
            for s, block in seasons.items()
        ],
        ignore_index=True,
    )
    complete = world.complete[~world.complete["partial"]]
    # N_MIN per dimension (rule v2): median n of EuroLeague rotation player-seasons, tuning only.
    rotation = rotation_player_seasons(complete, M6.tuning)
    tuned = {
        "shot_making": pd.concat(
            [shot_making_observations(shots, xpts, xwalk, s, variant) for s in M6.tuning]
        ),
        "on_off": pd.concat(
            [on_off_observations(stints, checks, brapm, xwalk, s) for s in M6.tuning]
        ),
    }
    n_min = {
        "shot_making": derive_n_min(rotation_n(tuned["shot_making"], rotation)),
        "fg3_pct": derive_n_min(fg3_rotation_n(complete, rotation)),
        "on_off": derive_n_min(rotation_n(tuned["on_off"], rotation)),
    }
    found: dict[str, dict[int, pd.DataFrame]] = {d: {} for d in DIMENSIONS}
    for s in (*M6.tuning, season):
        found["shot_making"][s] = shot_making_observations(shots, xpts, xwalk, s, variant)
        found["fg3_pct"][s] = fg3_observations(complete, s, n_min["fg3_pct"])
        found["on_off"][s] = on_off_observations(stints, checks, brapm, xwalk, s)
    stabilities = {
        d: stability(pd.concat([found[d][s] for s in M6.tuning]), d, M6.tuning, n_min[d])
        for d in DIMENSIONS
    }
    return board({d: found[d][season] for d in DIMENSIONS}, stabilities), stabilities


def _m6_similar(  # noqa: PLR0917 -- the live inputs, passed through
    world: World,
    people: list[str],
    season: int,
    shots: pd.DataFrame,
    xpts: pd.DataFrame,
    variant: str,
) -> tuple[pd.DataFrame, list[int], bool]:
    """Each live person's "plays like" list: his newest complete season with ``min_poss``
    possessions against every complete season up to ``season``."""
    joined = shots.merge(
        xpts.loc[xpts["variant"] == variant, ["game_id", "event", "p_make"]],
        on=["game_id", "event"],
    )
    mapped = person_ids(joined["shooter"], EUROLEAGUE.name, world.inputs.xwalk)["person_id"]
    frame = joined.assign(person_id=mapped.to_numpy(), xpts=joined["p_make"] * joined["value"])
    frame = frame[["person_id", "season", "made", "value", "x", "y", "band", "xpts"]]
    done = world.complete[~world.complete["partial"] & (world.complete["season"] <= season)]
    pool = sorted(int(s) for s in done["season"].unique())
    emb = embed(done, frame, pool_seasons=pool, min_poss=M6.min_poss)
    query = query_rows(world.complete, people, LIVE_SEASON, M6.min_poss)
    features = raw_features(query, frame[frame["person_id"].isin(set(query["person_id"]))])
    return neighbours(features, emb), pool, len(emb.features) > len(BOX_FEATURES)


def _payload_sha(payload: dict[str, Any]) -> str:
    return hashlib.sha256((json.dumps(payload, indent=2) + "\n").encode("utf-8")).hexdigest()


def _live_block(  # noqa: PLR0917 -- one competition's live inputs
    inputs: M6Inputs,
    world: World,
    comp: Competition,
    spec: M6Backtest,
    report: dict[str, Any],
    chosen: dict[str, Any],
    translation: Translation,
) -> tuple[float, pd.DataFrame, TargetInputs]:
    """One competition's live checkpoint, projections and inputs."""
    games = inputs.games[comp.name]
    regular = games[(games["season"] == LIVE_SEASON) & (games["phase"] == "RS")]
    rounds = season_format(comp.name, LIVE_SEASON).regular_season_rounds
    checkpoint = live_checkpoint(regular, rounds, spec.checkpoints)
    rows, ti = live_projections(
        inputs,
        comp.name,
        LIVE_SEASON,
        checkpoint,
        live_people(world.complete, comp.name, LIVE_SEASON),
        chosen=chosen,
        scale=report["calibration"]["scale"][f"{checkpoint:g}"],
        spec=spec,
        translation=translation,
        rounds=rounds,
        world=world,
    )
    return checkpoint, rows, ti


@app.command()
def project(
    *,
    dry_run: Annotated[
        bool, typer.Option(help="Compute and print each report's sha256; write nothing")
    ] = False,
    projections_only: Annotated[
        bool,
        typer.Option(
            help="Only reports/m6_projections.json: the board and the comparables need the local "
            "M2 shot marts (the daily workflow)"
        ),
    ] = False,
) -> None:
    """Live M6 (weeks 16-18 L9): the 2026-27 projections and the GBL undervalued list
    (reports/m6_projections.json), the over/under board (m6_board.json) and the "plays like"
    lists (m6_similarity.json). Runs only when the EuroLeague M6 gate passed (L-l)."""
    started = time.perf_counter()
    specs = ((EUROLEAGUE, M6), (GBL, M6_GBL))
    if not all(spec.report.exists() for _, spec in specs):
        log.error("no committed M6 reports; run: eurohoops backtest --model m6")
        raise typer.Exit(code=1)
    reports = {c.name: json.loads(s.report.read_text(encoding="utf-8")) for c, s in specs}
    if not reports[EUROLEAGUE.name]["gate"]["passed"]:
        typer.echo("M6 did not pass its validation gate: the committed reports stay (L-l)")
        return
    chosen = reports[EUROLEAGUE.name]["chosen"]
    inputs = _m6_inputs(M6, live=True)
    world = prepare(inputs)
    newest = max(inputs.translations)  # D5: M4's newest fit, pairs strictly before it
    names = _live_names(inputs)
    checkpoints, blocks, players = {}, [], []
    projected = {}
    for comp, spec in specs:
        checkpoint, rows, ti = _live_block(
            inputs, world, comp, spec, reports[comp.name], chosen, inputs.translations[newest]
        )
        checkpoints[comp.name] = checkpoint
        projected[comp.name] = rows
        blocks.append(ti)
        live = world.complete[
            (world.complete["season"] == LIVE_SEASON) & (world.complete["competition"] == comp.name)
        ]
        players += player_rows(rows, live, names)
    gbl = world.complete[
        (world.complete["season"] == LIVE_SEASON) & (world.complete["competition"] == GBL.name)
    ]
    cheap = undervalued(
        projected[GBL.name],
        world.complete,
        LIVE_SEASON,
        translation=_m6_translations(STAT_COLUMNS)[newest],
        spm=inputs.spm,
        names=names,
        teams=dict(zip(gbl["person_id"], gbl["team"], strict=True)),
        min_poss=M6.min_poss,
    )
    payloads = {
        M6_PROJECTIONS: projections_report(
            chosen=chosen,
            gate_passed=True,
            season=LIVE_SEASON,
            checkpoints=checkpoints,
            digest=inputs_digest(blocks),
            players=players,
            undervalued_rows=cheap,
        )
    }
    if not projections_only:
        m3_seasons = json.loads(M3_PLAYERS_REPORT.read_text(encoding="utf-8"))["seasons"]
        season = max(int(s) for s in m3_seasons)  # D23: the newest complete season
        shots, xpts, variant = _m2_shots(season)
        rows, stabilities = _m6_board(world, season, shots, xpts, variant)
        done = world.complete[world.complete["season"] == season]
        teams = {
            (str(p), str(c)): str(t)
            for p, c, t in zip(done["person_id"], done["competition"], done["team"], strict=True)
        }
        payloads[M6_BOARD] = board_report(rows, stabilities, season, names, teams)
        people = sorted({p["person_id"] for p in players})
        similar, pool, shot_features = _m6_similar(world, people, season, shots, xpts, variant)
        payloads[M6_SIMILARITY] = similarity_report(
            similar, LIVE_SEASON, pool, shot_features, names
        )
    for path, payload in payloads.items():
        if found := personal_fields(payload):
            log.error("%s would carry an age or a birth date: %s", path, found[:5])
            raise typer.Exit(code=1)
        typer.echo(f"{path}: sha256 {_payload_sha(payload)}")
        if not dry_run:
            write_json(path, payload)
    typer.echo(
        f"{len(players)} projections (checkpoints {checkpoints}), {len(cheap)} undervalued"
        + ("; dry run, nothing written" if dry_run else "")
    )
    typer.echo(f"RUNTIME project: {time.perf_counter() - started:.0f} s")


@app.command("sim-ungated")
def sim_ungated(competition: CompetitionOption = CompetitionName.euroleague) -> None:
    """The live season simulation for the Standings page (owner, D1): ``simulate``'s run written
    to reports/sim_ungated_{competition}.json whatever M7's gate says; labelled gated only for the
    EuroLeague once a gate passed (gate v2 did, 2026-10-09), never for the GBL. Never appends to
    the simulation log; rewritten only when the number of completed rounds changed (its time
    stamp is the cutoff), so a day without a completed round changes nothing."""
    comp = COMPETITIONS[competition]
    model = load_sim(M7.report, M7_V2.report)
    if comp.m1 is None or model is None or not comp.m1.report.exists():
        log.error("no committed M7 or M1 report; run: eurohoops backtest --model m7")
        raise typer.Exit(code=1)
    started = time.perf_counter()
    games = read_games(MART_PATH, comp.name)
    games = games[games["season"] <= LIVE_SEASON].reset_index(drop=True)
    team_games = read_table(MART_PATH, "team_games", comp.name)
    if team_games is None:
        log.error("no team_games in the marts; run: eurohoops build")
        raise typer.Exit(code=1)
    fmt = season_format(comp.name, LIVE_SEASON)
    tuned = json.loads(comp.m1.report.read_text(encoding="utf-8"))["tuned"]
    spec = M7 if comp is EUROLEAGUE else M7_GBL
    run = run_live_sim(
        games, team_games, tuned, model, fmt=fmt, season=LIVE_SEASON, spec=spec, n_sims=LIVE_SIMS
    )
    path = SIM_UNGATED[comp.name]
    if run is None:
        typer.echo(f"{comp.name}: the {LIVE_SEASON} regular season is over; {path} stays")
        return
    gated = model.gate_passed and comp is EUROLEAGUE
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    if old is not None and (old["season"], old["after_round"], old.get("gated")) == (
        LIVE_SEASON,
        run.after_round,
        gated,
    ):
        typer.echo(f"{comp.name}: after round {run.after_round} already in {path}")
        return
    report = latest_report(run, model, fmt, LIVE_SEASON, run.cutoff.to_pydatetime())
    if comp is not EUROLEAGUE:
        reason = "the GBL is not gated (M7 gate v2): EuroLeague verdict, GBL rules unverified"
    elif model.gate_passed:
        reason = f"M7 passed its validation gate (gate v2, {M7_V2.report.as_posix()})"
    else:
        reason = f"M7 failed its validation gate ({M7.report.as_posix()}): this run is not gated"
    write_json(path, {**report, "gated": gated, "gate": {"passed": gated, "reason": reason}})
    label = "gated" if gated else "not gated"
    typer.echo(f"wrote {path}: after round {run.after_round}, seed {run.seed} ({label})")
    typer.echo(f"RUNTIME sim-ungated {comp.name}: {time.perf_counter() - started:.0f} s")


@app.command()
def backtest(  # noqa: PLR0911 -- one dispatch return per model
    *,
    competition: CompetitionOption = CompetitionName.euroleague,
    model: Annotated[
        ModelName,
        typer.Option(
            help="elo (live), m1, m2, m3 (players), m4, m5 (roster), m6 (projections), m7 (season)"
        ),
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
    gate_v2: Annotated[
        bool,
        typer.Option(
            help="M7: the rolling-origin gate v2 (docs/models/m7.md), EuroLeague, run once"
        ),
    ] = False,
) -> None:
    """Tune and score a model vs its baselines; the Elo live report holds the live parameters."""
    if gate_v2:
        if model is not ModelName.m7:
            log.error("--gate-v2 belongs to --model m7")
            raise typer.Exit(code=1)
        _backtest_m7_v2(competition)
        return
    if model is ModelName.m2:
        research.backtest_m2(score_test, search, tracking_uri or default_tracking_uri())
        return
    if model is ModelName.m3:
        _backtest_m3(competition, score_test, tuning_only, tracking_uri or default_tracking_uri())
        return
    if model is ModelName.m4:
        research.backtest_m4(score_test=score_test, tuning_only=tuning_only)
        return
    if model is ModelName.m5:
        _backtest_m5(competition, score_test, tuning_only, tracking_uri or default_tracking_uri())
        return
    if model is ModelName.m6:
        _backtest_m6(competition, score_test, tuning_only, tracking_uri or default_tracking_uri())
        return
    if model is ModelName.m7:
        _backtest_m7(competition, score_test, tuning_only, tracking_uri or default_tracking_uri())
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
app.command(name="m5-rest")(research.m5_rest)


app.command(name="shot-charts")(research.shot_charts)


def _m5_live_build(comp: Competition) -> tuple[M5Inputs, PlayerPartFn]:
    """The live M5 harness inputs and EuroLeague player part: the backtest's own builders over
    every game up to the live season (``live_spec``)."""
    spec = live_spec(M5, LIVE_SEASON)
    inputs, frame = _m5_inputs(comp, GBL, spec, dict(GREEK_EL_CLUBS))
    return inputs, _m5_player_part_el(inputs, frame)


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
        m5 = load_m5(M5.report) if comp.m5_prediction_log is not None else None
        if m5 is not None and comp.m5_prediction_log is not None:
            try:
                added = predict_upcoming_m5(
                    live.games,
                    m5,
                    partial(_m5_live_build, comp),
                    log_path=comp.m5_prediction_log,
                    season=live.season,
                    window=timedelta(hours=window_hours),
                    clock=utc_now,
                )
                typer.echo(f"{added} M5 predictions appended to {comp.m5_prediction_log}")
            except typer.Exit:  # a shadow model never blocks the Elo and M1 logs
                log.warning("M5 skipped: its marts, stints or reports are missing here")
            except LatePredictionError:
                raise
            except Exception:  # e.g. marts without the tuning seasons: still never blocks
                log.exception("M5 skipped: its forecast failed; nothing appended to its log")
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
def injuries() -> None:
    """Record the BasketNews EuroLeague injury report: one request, a snapshot appended.

    A failed fetch, an unparseable page or a torn log is a warning, never a non-zero exit.
    """
    names = team_name_map(
        read_teams(MART_PATH, EUROLEAGUE.name), read_games(MART_PATH, EUROLEAGUE.name), LIVE_SEASON
    )
    paths = InjuryPaths(INJURY_LOG, INJURY_RAW_DIR)
    try:
        with make_client() as client:
            summary = record_injuries(client, names, paths, utc_now)
    except ValueError as exc:
        log.warning("injuries: %s; nothing recorded", exc)
        return
    typer.echo(
        f"injuries: {summary['status']}; {summary['rows']} rows, {summary['teams']} teams"
        f" (page updated {summary.get('page_updated_utc') or '?'}); log {paths.log}; "
        f"unmapped teams: {', '.join(summary['unmapped']) or 'none'}"
    )


@app.command()
def simulate(
    *,
    competition: CompetitionOption = CompetitionName.euroleague,
    dry_run: Annotated[
        bool, typer.Option(help="Compute and print the simulation; write nothing")
    ] = False,
) -> None:
    """Simulate the rest of the live regular season (M7, K8) after the completed rounds; the log
    is appended only when the EuroLeague M7 gate (v1 or v2) passed and the round is not logged
    yet. The GBL is not gated (gate v2, docs/models/m7.md): its log is never appended."""
    comp = COMPETITIONS[competition]
    model = load_sim(M7.report, M7_V2.report)
    if model is not None and comp is not EUROLEAGUE:
        model = replace(model, gate_passed=False)
    if comp.m1 is None or model is None or not comp.m1.report.exists():
        log.error("no committed M7 or M1 report; run: eurohoops backtest --model m7")
        raise typer.Exit(code=1)
    started = time.perf_counter()
    games = read_games(MART_PATH, comp.name)
    games = games[games["season"] <= LIVE_SEASON].reset_index(drop=True)
    team_games = read_table(MART_PATH, "team_games", comp.name)
    if team_games is None:
        log.error("no team_games in the marts; run: eurohoops build")
        raise typer.Exit(code=1)
    fmt = season_format(comp.name, LIVE_SEASON)
    tuned = json.loads(comp.m1.report.read_text(encoding="utf-8"))["tuned"]
    spec = M7 if comp is EUROLEAGUE else M7_GBL
    run = run_live_sim(
        games, team_games, tuned, model, fmt=fmt, season=LIVE_SEASON, spec=spec, n_sims=LIVE_SIMS
    )
    if run is None:
        typer.echo(f"{comp.name}: the {LIVE_SEASON} regular season is over; nothing to simulate")
        return
    now = datetime.now(UTC)
    for row in log_rows(run, model, LIVE_SEASON, now):
        typer.echo(
            f"{row['team']:<10}{row['p_direct_playoffs']:>8}{row['p_play_in']:>8}"
            f"{row['p_final_four']:>8}{row['p_title']:>8}{row['expected_wins']:>7}"
            f"  {row['rank_p10']}-{row['rank_p50']}-{row['rank_p90']}"
        )
    typer.echo(f"after round {run.after_round}, seed {run.seed}, gate passed {model.gate_passed}")
    if not dry_run:
        n = write_live_sim(
            run,
            model,
            fmt=fmt,
            season=LIVE_SEASON,
            at=now,
            log_path=SIM_LOGS[comp.name],
            report_path=SIM_LATEST[comp.name],
        )
        typer.echo(f"{n} rows appended to {SIM_LOGS[comp.name]}")
    typer.echo(f"RUNTIME simulate {comp.name}: {time.perf_counter() - started:.0f} s")


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
        m5_log_path=comp.m5_prediction_log,
    )
    write_json(comp.scorecard, card)
    typer.echo(json.dumps(card, indent=2))


@app.command()
def publish() -> None:
    """Write the site data (site.json and api/) the Astro front-end renders, from the API."""
    now = utc_now()
    store = Store()
    client = TestClient(create_app(store, lambda: now))
    files = publish_files(client, store, typer.echo)
    write_publish(WEB_DIR, files)
    typer.echo(f"wrote {SITE_DATA} and {len(files) - 1} API files")


@app.command("export-stats")
def export_stats(
    from_cache: Annotated[
        bool,
        typer.Option(help="Read games from the cached schedules instead of the marts (no DuckDB)"),
    ] = False,
    raw_dir: Annotated[Path, typer.Option(help="EuroLeague raw cache")] = EUROLEAGUE.raw_dir,
    out: Annotated[Path, typer.Option(help="Where the stats JSON goes")] = STATS_DIR,
) -> None:
    """Write the EuroLeague stats-site data (players, teams, game logs, shots), from the API."""
    now = utc_now()
    store = RawStore(from_cache=from_cache, raw_dir=raw_dir)
    files = stats_files(TestClient(create_app(store, lambda: now)))
    write_stats_files(out, files)
    write_downloads(DOWNLOADS_DIR, build_downloads(stats_inputs(store), now))
    seasons = json.loads(files["meta.json"])["seasons"]
    typer.echo(
        f"wrote {len(files)} files to {out}: {len(seasons)} seasons, "
        f"{len(json.loads(files['players.json'])['players'])} players, "
        f"{sum(s['games_without_box'] for s in seasons)} played games without a box score"
    )
