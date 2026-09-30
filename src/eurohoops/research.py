"""The local M2 and M3 research commands (``eurohoops shots``, ``free-throws``, ``shot-quality``,
``shot-charts``, ``backtest --model m2`` and ``gbl-stints``): the daily workflow never runs them.
cli.py registers them under their names; nothing here is fetched, everything reads the marts and
the raw cache."""

import json
import logging
import time
from typing import Annotated

import pandas as pd
import typer

from eurohoops.config import (
    EL_PLAYER_NAMES,
    ENTITY_LABELS,
    ENTITY_LABELS_TODO,
    ENTITY_OVERRIDES,
    ENTITY_REPORT,
    ENTITY_SEASONS,
    ENTITY_TUNING_REPORT,
    EUROLEAGUE,
    FREE_THROWS_REPORT,
    GBL,
    GBL_PBP,
    GBL_PLAYER_BOX,
    GBL_PLAYER_NAMES,
    GBL_STINTS_REPORT,
    GREEK_EL_CLUBS,
    M2_CHART_PLAYERS,
    M2_CHART_TEAMS,
    M2_CHARTS_DIR,
    M2_PLAYERS_REPORT,
    M2_REPORT,
    M2_SEASONS,
    M2_TEAMS_REPORT,
    M3,
    M3_PLAYERS_REPORT,
    MART_PATH,
    SHOTS_REPORT,
)
from eurohoops.entity.pipeline import (
    draft_labels,
    label_metrics,
    read_overrides,
    run_entity,
    silver_metrics,
    silver_set,
    tune,
)
from eurohoops.eval.m2_backtest import run_m2_backtest
from eurohoops.eval.m2_backtest import search as m2_search
from eurohoops.eval.m3_backtest import TunedRapm, build_rapm_inputs, prepare_data
from eurohoops.eval.m3_players import season_end_players
from eurohoops.eval.shot_making import player_report
from eurohoops.eval.team_shot_quality import shots_with_xpts, team_report
from eurohoops.eval.team_shot_quality import team_games as team_shot_games
from eurohoops.eval.tracking import log_m2_backtest
from eurohoops.ingest.bios import build_bios
from eurohoops.logs import write_json
from eurohoops.marts import read_games, read_table, write_tables
from eurohoops.parse.free_throws import LEVEL_CHECK, build_ft_team_games, ft_report
from eurohoops.parse.games import write_table
from eurohoops.parse.gbl_stints import H_I_THRESHOLD, build_gbl_stints_mart
from eurohoops.parse.gbl_stints import mart_report as gbl_stints_report
from eurohoops.parse.player_names import euroleague_names, gbl_names
from eurohoops.parse.shot_table import build_shot_table, reconcile, shot_report
from eurohoops.publish import DISPLAY_CODES
from eurohoops.stats.box import build_box_games

log = logging.getLogger("eurohoops")


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
    write_json(SHOTS_REPORT, report)
    typer.echo(
        f"{report['shots']} shots, {report['excluded']} excluded; feed = box for "
        f"{report['reconciliation_match_rate_validated']:.2%} of 2011+ team-games; "
        f"wrote {SHOTS_REPORT}"
    )


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
    write_json(FREE_THROWS_REPORT, report)
    for season, block in report["seasons"].items():
        shares = block["shares"]
        bands_out = [b for b, v in shares["bands"].items() if not v["within_tolerance"]]
        teams = shares["teams"]
        typer.echo(
            f"{season} {block['split']}: teams {'PASS' if shares['teams_pass'] else 'FAIL'}, "
            f"bands outside 2 SE: {bands_out or 'none'} (team mean |gap| "
            f"{teams['mean_abs_share_gap']:.5f} <= {teams['mean_abs_tolerance']:.5f}: "
            f"{teams['mean_abs_within_tolerance']}; r {teams['pearson_r']:.3f} >= "
            f"{teams['r_tolerance']:.3f}: {teams['r_within_tolerance']}) | {LEVEL_CHECK}: FT "
            f"points {block['ft_points_per_team_game']:.2f}, expected "
            f"{block['expected_per_team_game']:.2f}, gap {block['mean_gap']:+.3f}"
        )
    gate = report["share_check"]
    typer.echo(
        f"band flags {gate['band_flags']} of {gate['band_checks']} (allowed "
        f"{gate['band_flags_allowed']}); team checks pass every season: "
        f"{gate['team_checks_pass_every_season']}; F2 share gate "
        f"{'PASSED' if gate['passed'] else 'FAILED'}; wrote {FREE_THROWS_REPORT}"
    )


def backtest_m2(score_test: bool, search_first: bool, tracking_uri: str) -> None:
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
    tipoff = read_games(MART_PATH, EUROLEAGUE.name).set_index("game_id")["tipoff_utc"]
    report, xpts = run_m2_backtest(
        shots_table, M2_SEASONS, study, score_test, log.info, tipoff=tipoff
    )
    log.info("backtest (without the study): %.0f s", time.monotonic() - start)
    start = time.monotonic()
    write_json(M2_REPORT, report)
    write_tables(MART_PATH, {"shot_xpts": xpts})
    log.info("TIMING report + shot_xpts mart written: %.1f s", time.monotonic() - start)
    g = report["gate"]
    diff = g["log_loss_challenger_minus_baseline"]
    typer.echo(
        f"{M2_REPORT}: {g['challenger']} vs {g['baseline']} validation log loss "
        f"{diff['mean']:+.5f} {diff['ci95']}; calibrated {g['calibrated']}; "
        f"gate {'PASSED' if g['passed'] else 'FAILED'} (chosen M2: {g['chosen']})"
    )
    start = time.monotonic()
    run_id = log_m2_backtest(report, tracking_uri)
    log.info("TIMING MLflow: %.1f s", time.monotonic() - start)
    if run_id is not None:
        typer.echo(f"MLflow parent run {run_id}")


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
    write_json(M2_TEAMS_REPORT, teams)
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
    write_json(M2_PLAYERS_REPORT, players)
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


def m3_players() -> None:
    """Per-player-season RAPM ratings and uncertainty at each season's last-round cutoff (H7)."""
    if not M3.report.is_file():
        log.error("missing %s; run: eurohoops backtest --model m3", M3.report)
        raise typer.Exit(code=1)
    bt_report = json.loads(M3.report.read_text(encoding="utf-8"))
    chosen = bt_report["chosen"]
    variant = str(chosen["variant"])
    skip = ("variant", "half_life_days", "ridge_o", "ridge_d", "tuning_rmse", "spm")
    extra = {k: v for k, v in chosen.items() if k not in skip}
    tuned = TunedRapm(
        half_life_days=float(chosen["half_life_days"]),
        ridge_o=float(chosen["ridge_o"]),
        ridge_d=float(chosen["ridge_d"]),
        tuning_rmse=float(chosen["tuning_rmse"]),
        grid={},
        extra=extra,
    )
    games = read_games(MART_PATH, EUROLEAGUE.name)
    team_games = read_table(MART_PATH, "team_games", EUROLEAGUE.name)
    stints = read_table(MART_PATH, "stints")
    checks = read_table(MART_PATH, "stint_game_checks")
    missing = [
        n
        for n, t in (
            ("team_games", team_games),
            ("stints", stints),
            ("stint_game_checks", checks),
        )
        if t is None
    ]
    if missing:
        log.error("missing marts %s; run: eurohoops build", missing)
        raise typer.Exit(code=1)
    assert team_games is not None and stints is not None and checks is not None
    player_games = build_box_games(EUROLEAGUE.raw_dir, games).players
    data = prepare_data(games, team_games, player_games, stints, checks, spec=M3)
    if data.snapshot != bt_report["data_sha256"]:
        log.error(
            "marts changed since backtest (snapshot %s != report %s); re-run backtest --model m3",
            data.snapshot,
            bt_report["data_sha256"],
        )
        raise typer.Exit(code=1)
    inputs = build_rapm_inputs(stints, checks, data.games, player_games)
    out = season_end_players(data.games, inputs, player_games, tuned, variant, M3, data.snapshot)
    write_json(M3_PLAYERS_REPORT, out)
    summary = out["summary"]
    typer.echo(
        f"{M3_PLAYERS_REPORT}: {summary['player_seasons']} player-seasons, "
        f"Spearman(sd_total, minutes)={summary['sd_total_minutes_spearman']}"
    )


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
        f"EuroLeague {label}\nexpected points per shot (M2, out of sample)",
        M2_CHARTS_DIR / f"xpts_surface_{season}.png",
    )
    names = dict(zip(shooters["shooter"], shooters["name"], strict=True))
    for team in M2_CHART_TEAMS:
        shown = DISPLAY_CODES[EUROLEAGUE.name].get(team, team)
        residual_chart(
            scored[scored["team"] == team],
            f"{shown} {label}\nactual minus expected points per shot",
            M2_CHARTS_DIR / f"team_{shown}_{season}.png",
        )
    top = scored["shooter"].value_counts().sort_index().sort_values(ascending=False, kind="stable")
    for shooter in top.index[:M2_CHART_PLAYERS]:
        residual_chart(
            scored[scored["shooter"] == shooter],
            f"{names.get(shooter, shooter)} {label}\nactual minus expected points per shot",
            M2_CHARTS_DIR / f"player_{shooter}_{season}.png",
        )
        typer.echo(f"player {shooter} {names.get(shooter, '')}: {top[shooter]} FGA")
    typer.echo(f"wrote charts to {M2_CHARTS_DIR}")


def entity(
    tune_params: Annotated[
        bool, typer.Option("--tune", help="D3 search on the silver set (entity_tuning.json)")
    ] = False,
    draft: Annotated[
        bool, typer.Option("--draft-labels", help="Write the owner's label sheet (I5) to data/")
    ] = False,
) -> None:
    """Weeks 12-14 I3-I5: player names of both competitions (``player_names`` mart and staging),
    the cross-league matcher and the ``player_xwalk`` mart, and reports/entity_resolution.json
    (counts, silver-set metrics, and precision/recall on entity/labels.csv once it exists).

    ``--tune`` runs the D3 search on the silver set and writes reports/entity_tuning.json
    instead; ``--draft-labels`` writes the label sheet. Local only: reads the marts, the raw
    cache and the cached bios (``ingest-bios``).
    """
    start = time.monotonic()
    player_box = pd.read_parquet(GBL_PLAYER_BOX)
    player_box = player_box[player_box["game_id"].str.slice(3, 7).astype(int).isin(ENTITY_SEASONS)]
    games = read_games(MART_PATH, EUROLEAGUE.name)
    games = games[games["season"].isin(ENTITY_SEASONS)]
    gbl = gbl_names(GBL.raw_dir, player_box)
    el = euroleague_names(build_box_games(EUROLEAGUE.raw_dir, games).players)
    write_table(gbl, GBL_PLAYER_NAMES)
    write_table(el, EL_PLAYER_NAMES)
    names = pd.concat([el, gbl], ignore_index=True)
    bios = build_bios(GBL.raw_dir, EUROLEAGUE.raw_dir)
    if tune_params:
        result = tune(names, bios, GREEK_EL_CLUBS)
        write_json(ENTITY_TUNING_REPORT, result)
        typer.echo(
            f"silver {result['silver']}; chosen {result['chosen']} "
            f"(dates hidden: {result['chosen_silver_hidden_dates']}); wrote {ENTITY_TUNING_REPORT}"
        )
        return
    overrides = read_overrides(ENTITY_OVERRIDES)
    if draft:
        sheet = draft_labels(names, bios, overrides, GREEK_EL_CLUBS)
        ENTITY_LABELS_TODO.parent.mkdir(parents=True, exist_ok=True)
        sheet.to_csv(ENTITY_LABELS_TODO, index=False, encoding="utf-8-sig")
        typer.echo(f"{len(sheet)} ids {sheet['stratum'].value_counts().to_dict()}")
        typer.echo(f"wrote {ENTITY_LABELS_TODO}")
        return
    run = run_entity(names, bios, overrides, GREEK_EL_CLUBS)
    write_tables(MART_PATH, {"player_names": names, "player_xwalk": run.xwalk})
    report = run.report
    pos, neg = silver_set(names, bios, GREEK_EL_CLUBS)
    report["silver"] = silver_metrics(run.matches, pos, neg)
    report["bios"] = {
        c: int(group["birth_date"].notna().sum()) for c, group in bios.groupby("competition")
    }
    if ENTITY_LABELS.exists():
        report["labels"] = label_metrics(run, overrides, pd.read_csv(ENTITY_LABELS, dtype=str))
    write_json(ENTITY_REPORT, report)
    typer.echo(
        f"{len(run.matches)} cross-league matches, {run.xwalk['person_id'].nunique()} persons; "
        f"silver precision {report['silver']['precision']} recall {report['silver']['recall']}; "
        f"wrote {ENTITY_REPORT}\nRUNTIME {time.monotonic() - start:.0f} s"
    )


def gbl_stints() -> None:
    """Build the GBL ``gbl_stints``/``gbl_stint_game_checks`` marts (H3, weeks 9-12) from the
    staged GBL play-by-play (``config.GBL_PBP``) and write reports/gbl_stints.json: pass rates
    per check and season, and whether the H-i rule (>= 95% of games pass) holds.

    Local only, like ``possessions`` and ``stints --mart``: the daily workflow never runs it.
    Play-by-play is cached for 2018 and 2019 only (D4, reports/week9-12_progress.md), so only
    those seasons get stints; GBL RAPM (H-i) is attempted only if the H-i rule holds.
    """
    if not GBL_PBP.exists():
        log.error("no staged GBL play-by-play; run: eurohoops ingest --competition gbl --pbp")
        raise typer.Exit(code=1)
    team_games = read_table(MART_PATH, "team_games", GBL.name)
    if team_games is None:
        log.error("no team_games in the marts; run: eurohoops build")
        raise typer.Exit(code=1)
    pbp = pd.read_parquet(GBL_PBP)
    games = read_games(MART_PATH, GBL.name)
    mart = build_gbl_stints_mart(pbp, games, team_games)
    write_tables(MART_PATH, {"gbl_stints": mart.stints, "gbl_stint_game_checks": mart.checks})
    report = gbl_stints_report(mart)
    write_json(GBL_STINTS_REPORT, report)
    for season, block in sorted(report["seasons"].items()):
        typer.echo(
            f"{season}: {block['passed']}/{block['games']} games passed "
            f"({block['pass_rate']:.2%}); by check {block['pass_rate_by_check']}"
        )
    typer.echo(
        f"overall pass rate {report['overall_pass_rate']}; H-i (>= {H_I_THRESHOLD:.0%}) holds: "
        f"{report['h_i_rule_holds']}; wrote {GBL_STINTS_REPORT}"
    )
