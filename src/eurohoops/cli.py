"""``eurohoops`` command line: ingest -> build -> backtest -> predict -> score -> publish."""

import json
import logging
import os
import time
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
    FREE_THROWS_REPORT,
    GBL,
    GBL_BOX_FILL,
    GBL_BOX_GAPS,
    GBL_PBP,
    GBL_PLAYER_BOX,
    GBL_TEAM_BOX,
    LIVE_SEASON,
    M2_CHART_PLAYERS,
    M2_CHART_TEAMS,
    M2_CHARTS_DIR,
    M2_PLAYERS_REPORT,
    M2_REPORT,
    M2_SEASONS,
    M2_TEAMS_REPORT,
    MART_PATH,
    ODDS_CALLS,
    ODDS_RAW_DIR,
    ODDS_TEAMS,
    POSSESSION_REPORT,
    SHOTS_REPORT,
    SITE_DATA,
    SQL_DIR,
    STINT_REPORT,
    STINTS_MART_REPORT,
    TEAM_CONTINUITY_REPORT,
)
from eurohoops.eval.backtest import format_table, load_tuned_model, run_backtest
from eurohoops.eval.m1_backtest import format_m1_table, run_m1_backtest
from eurohoops.eval.m2_backtest import run_m2_backtest
from eurohoops.eval.m2_backtest import search as m2_search
from eurohoops.eval.scorecard import build_scorecard
from eurohoops.eval.shot_making import player_report
from eurohoops.eval.team_shot_quality import shots_with_xpts, team_report
from eurohoops.eval.team_shot_quality import team_games as team_shot_games
from eurohoops.eval.tracking import default_tracking_uri, log_backtest, log_m2_backtest
from eurohoops.ingest import euroleague, gbl
from eurohoops.ingest.http import Fetcher, make_client
from eurohoops.live_m1 import load_m1, predict_upcoming_m1
from eurohoops.marts import (
    box_invariants,
    build_marts,
    read_games,
    read_table,
    read_teams,
    refresh_box_gaps,
    write_tables,
)
from eurohoops.odds import OddsApiError, OddsPaths, api_key, record_odds
from eurohoops.parse.box import build_box_tables
from eurohoops.parse.continuity import continuity_report
from eurohoops.parse.free_throws import build_ft_team_games, ft_report
from eurohoops.parse.games import (
    build_games_table,
    build_gbl_tables,
    build_gbl_team_seasons,
    build_team_seasons,
    build_teams_table,
    write_table,
)
from eurohoops.parse.gbl_pbp import build_pbp_table
from eurohoops.parse.possession_report import possession_report
from eurohoops.parse.shot_table import build_shot_table, reconcile, shot_report
from eurohoops.parse.stints import validate_sample
from eurohoops.parse.stints_mart import build_stints_mart, mart_report
from eurohoops.parse.team_box import TEAM_GAMES_SCHEMA, build_team_games
from eurohoops.predict import LatePredictionError, predict_upcoming
from eurohoops.publish import DISPLAY_CODES, Section, site_data

app = typer.Typer(no_args_is_help=True, add_completion=False)
log = logging.getLogger("eurohoops")


class CompetitionName(StrEnum):
    euroleague = "euroleague"
    gbl = "gbl"


class ModelName(StrEnum):
    elo = "elo"
    m1 = "m1"
    m2 = "m2"


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


def _team_games(competition: str) -> pd.DataFrame:
    """The competition's ``team_games`` rows (none before the first ``build``)."""
    table = read_table(MART_PATH, "team_games", competition)
    return pd.DataFrame(columns=list(TEAM_GAMES_SCHEMA.columns)) if table is None else table


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
        _write_json(BOX_INVARIANTS_REPORT, report)
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
    _write_json(TEAM_CONTINUITY_REPORT, report)
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
        _write_json(STINTS_MART_REPORT, summary)
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
    _write_json(STINT_REPORT, report)
    rates = ", ".join(f"{check} {rate:.0%}" for check, rate in report["pass_rate"].items())
    typer.echo(
        f"{report['games']} games: all checks {report['all_checks_pass_rate']:.0%} ({rates}); "
        f"wrote {STINT_REPORT}"
    )


@app.command()
def shots() -> None:
    """Build the ``shots`` and ``shots_excluded`` marts (EuroLeague, from the raw shot cache) and
    write reports/shots.json: exclusions per season and the feed-vs-box reconciliation.

    Local only, like ``possessions``: the daily workflow never runs it (M2 has no live use)."""
    games = read_games(MART_PATH, EUROLEAGUE.name)
    table = build_shot_table(EUROLEAGUE.raw_dir, games)
    write_tables(
        MART_PATH,
        {"shots": table.shots, "shots_excluded": table.excluded, "shooters": table.shooters},
    )
    report = shot_report(table, reconcile(table, EUROLEAGUE.raw_dir, games))
    _write_json(SHOTS_REPORT, report)
    typer.echo(
        f"{report['shots']} shots, {report['excluded']} excluded; feed = box for "
        f"{report['reconciliation_match_rate_validated']:.2%} of 2011+ team-games; "
        f"wrote {SHOTS_REPORT}"
    )


@app.command(name="free-throws")
def free_throws() -> None:
    """Build ``ft_team_games`` (FT trips and points per team-game from EuroLeague play-by-play,
    and-ones tied to their shot) and write reports/free_throws.json (needs ``eurohoops shots``)."""
    shots_table = read_table(MART_PATH, "shots")
    excluded = read_table(MART_PATH, "shots_excluded")
    if shots_table is None or excluded is None:
        log.error("no shots in the marts; run: eurohoops shots")
        raise typer.Exit(code=1)
    table = build_ft_team_games(EUROLEAGUE.raw_dir, shots_table, excluded)
    write_tables(MART_PATH, {"ft_team_games": table})
    report = ft_report(table, M2_SEASONS.development, M2_SEASONS.validation)
    _write_json(FREE_THROWS_REPORT, report)
    for season, block in report["seasons"].items():
        typer.echo(
            f"{season} {block['split']}: FT points {block['ft_points_per_team_game']:.2f}, "
            f"expected {block['expected_per_team_game']:.2f}, gap {block['mean_gap']:+.3f} "
            f"({'within' if block['within_tolerance'] else 'OUTSIDE'} ±{block['tolerance']})"
        )
    typer.echo(f"wrote {FREE_THROWS_REPORT}")


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
    _write_json(POSSESSION_REPORT, report)
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


@app.command()
def backtest(
    competition: CompetitionOption = CompetitionName.euroleague,
    model: Annotated[
        ModelName, typer.Option(help="elo (live model), m1, or m2 (EuroLeague shot model)")
    ] = ModelName.elo,
    score_test: Annotated[
        bool,
        typer.Option(help="M1/M2: also score the test seasons (only after the gate verdict)"),
    ] = False,
    search: Annotated[
        bool,
        typer.Option(help="M2: run the declared Optuna study first (only when F4 changes)"),
    ] = False,
    tracking_uri: Annotated[
        str | None,
        typer.Option(help="M1: MLflow tracking URI (default: SQLite store under ./mlruns)"),
    ] = None,
) -> None:
    """Tune and score a model vs its baselines; the Elo live report holds the live parameters."""
    if model is ModelName.m2:
        _backtest_m2(score_test, search, tracking_uri or default_tracking_uri())
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
        _write_json(comp.m1.report, report)
        typer.echo(f"{comp.m1.report}\n{format_m1_table(report)}")
        run_id = log_backtest(
            report, comp.name, report["data_sha256"], tracking_uri or default_tracking_uri()
        )
        if run_id is not None:
            typer.echo(f"MLflow parent run {run_id}")
        return
    for spec in (comp.live_backtest, *comp.history_backtests):
        report = run_backtest(games, spec)
        _write_json(spec.report, report)
        typer.echo(f"{spec.report}\n{format_table(report)}")


def _backtest_m2(score_test: bool, search_first: bool, tracking_uri: str) -> None:
    """M2 (EuroLeague only): the declared variants, the gate, and ``shot_xpts`` in the marts."""
    shots_table = read_table(MART_PATH, "shots")
    if shots_table is None:
        log.error("no shots in the marts; run: eurohoops shots")
        raise typer.Exit(code=1)
    if search_first:
        start = time.monotonic()
        study = m2_search(
            shots_table[shots_table["validated_season"]], M2_SEASONS.development, progress=log.info
        )
        log.info("optuna study: %.0f s", time.monotonic() - start)
    elif M2_REPORT.exists():
        study = json.loads(M2_REPORT.read_text(encoding="utf-8"))["optuna_study"]
    else:
        log.error("no stored Optuna result in %s; run: backtest --model m2 --search", M2_REPORT)
        raise typer.Exit(code=1)
    start = time.monotonic()
    report, xpts = run_m2_backtest(shots_table, M2_SEASONS, study, score_test, log.info)
    log.info("backtest (without the study): %.0f s", time.monotonic() - start)
    _write_json(M2_REPORT, report)
    write_tables(MART_PATH, {"shot_xpts": xpts})
    g = report["gate"]
    diff = g["log_loss_challenger_minus_baseline"]
    typer.echo(
        f"{M2_REPORT}: {g['challenger']} vs {g['baseline']} validation log loss "
        f"{diff['mean']:+.5f} {diff['ci95']}; calibrated {g['calibrated']}; "
        f"gate {'PASSED' if g['passed'] else 'FAILED'} (chosen M2: {g['chosen']})"
    )
    run_id = log_m2_backtest(report, tracking_uri)
    if run_id is not None:
        typer.echo(f"MLflow parent run {run_id}")


@app.command(name="shot-quality")
def shot_quality() -> None:
    """Team shot quality (F6: mart ``team_shot_quality``, reports/m2_teams.json) and player
    shot-making with its stability verdict (F7: reports/m2_players.json), from the out-of-fold
    xPTS of the last ``backtest --model m2`` (``shot_xpts``)."""
    tables = {
        n: read_table(MART_PATH, n) for n in ("shots", "shot_xpts", "ft_team_games", "shooters")
    }
    missing = [n for n, t in tables.items() if t is None]
    if missing:
        log.error(
            "missing marts %s; run: eurohoops shots, free-throws, backtest --model m2", missing
        )
        raise typer.Exit(code=1)
    shots_table, xpts, ft, shooters = (
        tables[n] for n in ("shots", "shot_xpts", "ft_team_games", "shooters")
    )
    assert shots_table is not None and xpts is not None and ft is not None and shooters is not None
    scored = shots_with_xpts(shots_table, xpts)
    later = sorted(set(scored["season"]) - set(M2_SEASONS.development))
    per_game = team_shot_games(scored, ft, M2_SEASONS.development, later)
    write_tables(MART_PATH, {"team_shot_quality": per_game})
    variant = str(xpts["variant"].iloc[0])
    teams = team_report(scored, per_game, M2_SEASONS.development, variant)
    _write_json(M2_TEAMS_REPORT, teams)
    games = read_games(MART_PATH, EUROLEAGUE.name)
    split_of = {
        s: name
        for name, seasons in (
            ("development", M2_SEASONS.development),
            ("validation", M2_SEASONS.validation),
            ("test", M2_SEASONS.test),
        )
        for s in seasons
    }
    players = player_report(
        scored,
        games.set_index("game_id")["tipoff_utc"],
        dict(zip(shooters["shooter"], shooters["name"], strict=True)),
        M2_SEASONS.development,
        split_of,
    )
    _write_json(M2_PLAYERS_REPORT, players)
    large = teams["calibration_in_the_large"]
    worst = max(large.items(), key=lambda kv: abs(kv[1]["ratio"] - 1.0))
    stability = players["stability"]
    typer.echo(
        f"calibration in the large: worst {worst[0]} ratio {worst[1]['ratio']}; "
        f"shot-making year-to-year r {stability['year_to_year']['shrunk_shot_making']['r']} "
        f"(90% CI {stability['year_to_year']['shrunk_shot_making']['ci90']}), split-half "
        f"{stability['split_half']['r']}: {stability['verdict']}; wrote {M2_TEAMS_REPORT}, "
        f"{M2_PLAYERS_REPORT}"
    )


@app.command(name="shot-charts")
def shot_charts() -> None:
    """Static M2 shot charts (F8) in docs/models/m2/ for the validation season: the league xPTS
    surface, and actual minus expected for the declared teams and the top-FGA players."""
    from eurohoops.eval.shot_charts import residual_chart, xpts_surface  # noqa: PLC0415

    tables = {n: read_table(MART_PATH, n) for n in ("shots", "shot_xpts", "shooters")}
    if any(t is None for t in tables.values()):
        log.error("missing marts; run: eurohoops shots, backtest --model m2")
        raise typer.Exit(code=1)
    shots_table, xpts, shooters = (tables[n] for n in ("shots", "shot_xpts", "shooters"))
    assert shots_table is not None and xpts is not None and shooters is not None
    season = M2_SEASONS.validation[0]
    scored = shots_with_xpts(shots_table, xpts)
    scored = scored[scored["season"] == season]
    label = f"{season}-{(season + 1) % 100:02d}"
    xpts_surface(
        scored,
        f"EuroLeague {label}: expected points per shot (M2, out of sample)",
        M2_CHARTS_DIR / f"xpts_surface_{season}.png",
    )
    names = dict(zip(shooters["shooter"], shooters["name"], strict=True))
    for team in M2_CHART_TEAMS:
        shown = DISPLAY_CODES[EUROLEAGUE.name].get(team, team)
        residual_chart(
            scored[scored["team"] == team],
            f"{shown} {label}: actual minus expected points per shot",
            M2_CHARTS_DIR / f"team_{shown}_{season}.png",
        )
    top = scored["shooter"].value_counts().sort_index().sort_values(ascending=False, kind="stable")
    for shooter in top.index[:M2_CHART_PLAYERS]:
        residual_chart(
            scored[scored["shooter"] == shooter],
            f"{names.get(shooter, shooter)} {label}: actual minus expected points per shot",
            M2_CHARTS_DIR / f"player_{shooter}_{season}.png",
        )
        typer.echo(f"player {shooter} {names.get(shooter, '')}: {top[shooter]} FGA")
    typer.echo(f"wrote charts to {M2_CHARTS_DIR}")


@app.command()
def predict(
    competition: CompetitionOption = CompetitionName.euroleague,
    window_hours: Annotated[float, typer.Option(help="Predict games tipping off within")] = 36.0,
) -> None:
    """Append pre-tip-off predictions for upcoming live-season games to the public log."""
    comp = COMPETITIONS[competition]
    games = read_games(MART_PATH, comp.name)
    try:
        added = predict_upcoming(
            games,
            load_tuned_model(comp.live_backtest.report),
            log_path=comp.prediction_log,
            season=LIVE_SEASON,
            replay_from=comp.live_backtest.warmup[0],
            window=timedelta(hours=window_hours),
            clock=utc_now,
        )
        typer.echo(f"{added} predictions appended to {comp.prediction_log}")
        m1 = None if comp.m1 is None else load_m1(comp.m1.report)
        if m1 is not None and comp.m1_prediction_log is not None:
            added = predict_upcoming_m1(
                games,
                _team_games(comp.name),
                m1,
                log_path=comp.m1_prediction_log,
                season=LIVE_SEASON,
                replay_from=comp.live_backtest.warmup[0],
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
    card = build_scorecard(
        comp.prediction_log,
        read_games(MART_PATH, comp.name),
        load_tuned_model(comp.live_backtest.report),
        utc_now(),
        manual_pushes=comp.manual_pushes,
        odds_path=comp.odds_log,
        m1_log_path=comp.m1_prediction_log,
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
