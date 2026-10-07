"""Live M7 (K8): the 2026-27 season simulation after each completed regular-season round.

The state is the backtest's checkpoint rule (D1) applied to today: the completed rounds are
the longest run of rounds 1..k whose regular-season games are all played, and the cutoff is the
first tip-off of round k + 1. M1's committed parameters give the strength posterior and pace
point on every game before the cutoff; the variant is the EuroLeague backtest's verdict
(``reports/backtest_m7.json`` ``chosen``), and 10,000 simulations play the remaining fixtures
(``remaining_fixtures``: tip-off order, a duplicated ordered pair once) under the season's
format with ``seed`` + k.

The log is append-only and written only when the EuroLeague M7 gate passed (K-j); a dry run
computes the same numbers and writes nothing. A competition that already has rows for the
current number of completed rounds gets none (one set of rows per completed round).
"""

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eurohoops.config import M7Backtest
from eurohoops.eval.m7_backtest import (
    Strengths,
    gaussian_sampler,
    noise_models,
    remaining_fixtures,
    restrict_posterior,
)
from eurohoops.logs import TIME_FORMAT, append_rows
from eurohoops.models.elo import FloatArray
from eurohoops.models.team_eff import (
    DecayParams,
    MarginModel,
    RatingPosterior,
    pace_points,
    prepare_history,
    rating_posteriors,
)
from eurohoops.sim.played import season_results
from eurohoops.sim.season import PaceModel, SimOutput, simulate
from eurohoops.standings import Format

SIM_LOG_COLUMNS = (
    "simulated_at_utc",
    "season",
    "after_round",
    "team",
    "p_direct_playoffs",
    "p_play_in",
    "p_top10",
    "p_final_four",
    "p_title",
    "expected_wins",
    "rank_p10",
    "rank_p50",
    "rank_p90",
    "model",
    "model_version",
)
LIVE_SIMS = 10_000

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class LiveSim:
    """The verdict of the EuroLeague M7 backtest: the chosen variant and whether its gate passed."""

    key: str
    spread: float
    net: bool
    version: str
    gate_passed: bool


def load_sim(report_path: Path) -> LiveSim | None:
    """The chosen variant of an M7 backtest report; None when there is no report."""
    if not report_path.exists():
        return None
    report = json.loads(report_path.read_text(encoding="utf-8"))
    chosen = report["chosen"]
    return LiveSim(
        key=str(chosen["key"]),
        spread=float(chosen["spread"]),
        net=bool(chosen["net"]),
        version=str(report["model_version"]),
        gate_passed=bool(report.get("gate", {}).get("passed")),
    )


def completed_rounds(regular: pd.DataFrame) -> int:
    """k: every regular-season game of rounds 1..k is played, and round k + 1 is not complete."""
    done = 0
    for _, games in regular.groupby("round", sort=True):
        if not games["played"].all():
            break
        done += 1
    return done


def logged_rounds(log_path: Path, season: int) -> set[int]:
    """The ``after_round`` values already logged for ``season``."""
    if not log_path.exists() or not log_path.stat().st_size:
        return set()
    logged = pd.read_csv(log_path, usecols=["season", "after_round"])
    return set(logged.loc[logged["season"] == season, "after_round"].astype(int))


@dataclass(frozen=True)
class LiveRun:
    after_round: int
    cutoff: pd.Timestamp
    seed: int
    out: SimOutput


def run_live_sim(
    games: pd.DataFrame,
    team_games: pd.DataFrame,
    tuned_m1: dict[str, Any],
    model: LiveSim,
    *,
    fmt: Format,
    season: int,
    spec: M7Backtest,
    n_sims: int = LIVE_SIMS,
) -> LiveRun | None:
    """The simulation of ``season`` after its completed rounds; None when the regular season
    is over (no round left to cut at). ``games`` holds every season of the competition."""
    games = games.sort_values("tipoff_utc", kind="stable").reset_index(drop=True)
    regular = games[(games["season"] == season) & (games["phase"] == "RS")]
    done = completed_rounds(regular)
    following = regular[regular["round"] == done + 1]
    if following.empty:
        return None
    cutoff: pd.Timestamp = following["tipoff_utc"].min()
    teams = tuple(sorted({*regular["home"], *regular["away"]}))
    upto = games[games["season"] <= season].reset_index(drop=True)
    # Fits read only the rows that tipped off before the cutoff; the whole schedule only fixes
    # the team list, so a team without a game yet still has (unseen) columns.
    history = prepare_history(upto, team_games[team_games["game_id"].isin(upto["game_id"])])
    when = [(float(np.float64(cutoff.as_unit("ns").value) / 1e9), season)]
    posterior = rating_posteriors(history, DecayParams(**tuned_m1["rating"]), when)[0]
    point = pace_points(history, DecayParams(**tuned_m1["pace"]), when)[0]
    column = {team: i for i, team in enumerate(history.teams)}
    pace = PaceModel(float(point[0]), np.array([point[1 + column[team]] for team in teams]))
    strengths = with_prior(posterior, teams, float(tuned_m1["rating"]["ridge"]))
    remaining = remaining_fixtures(regular, cutoff)
    noise, net = noise_models(
        spec,
        teams,
        remaining,
        strengths=strengths,
        pace=pace,
        margin=MarginModel(**tuned_m1["margin"]),
    )
    seed = spec.seed + done
    out = simulate(
        teams=teams,
        played=season_results(regular[regular["tipoff_utc"] < cutoff]),
        remaining=remaining,
        sampler=gaussian_sampler(strengths, model.spread),
        pace=pace,
        noise=net if model.net else noise,
        fmt=fmt,
        n_sims=n_sims,
        seed=seed,
    )
    return LiveRun(done, cutoff, seed, out)


def with_prior(posterior: RatingPosterior, teams: Sequence[str], ridge: float) -> Strengths:
    """``restrict_posterior`` for the live season, where a newly promoted team may have no
    game before the cutoff: its off and def columns are unseen, so they keep M1's ridge prior,
    mean 0 (M1 forecasts such a team at the league mean too) and variance sigma2 / ridge,
    independent of everything else."""
    seen = set(posterior.labels)
    new = [t for t in teams if f"off:{t}" not in seen and f"def:{t}" not in seen]
    if not new:
        return restrict_posterior(posterior, teams)
    prior_var = posterior.sigma2 / ridge
    k = len(posterior.labels)
    labels = (*posterior.labels, *(f"off:{t}" for t in new), *(f"def:{t}" for t in new))
    mean = np.concatenate([posterior.mean, np.zeros(2 * len(new))])
    cov = np.zeros((len(labels), len(labels)))
    cov[:k, :k] = posterior.cov
    cov[k:, k:] = np.eye(2 * len(new)) * prior_var
    columns = np.arange(len(labels), dtype=np.int64)
    extended = replace(posterior, labels=labels, mean=mean, cov=cov, columns=columns)
    return restrict_posterior(extended, teams)


def _rank_quantile(counts: FloatArray, q: float) -> int:
    """The smallest place whose cumulative share reaches ``q``."""
    return int(np.searchsorted(np.cumsum(counts) / counts.sum(), q - 1e-12)) + 1


def log_rows(run: LiveRun, model: LiveSim, season: int, at: datetime) -> list[dict[str, Any]]:
    out = run.out
    return [
        {
            "simulated_at_utc": at.strftime(TIME_FORMAT),
            "season": season,
            "after_round": run.after_round,
            "team": team,
            "p_direct_playoffs": f"{out.p_direct[i]:.4f}",
            "p_play_in": f"{out.p_play_in[i]:.4f}",
            "p_top10": f"{out.p_top10[i]:.4f}",
            "p_final_four": f"{out.p_semis[i]:.4f}",
            "p_title": f"{out.p_title[i]:.4f}",
            "expected_wins": f"{out.wins[i]:.2f}",
            "rank_p10": _rank_quantile(out.rank_counts[i], 0.1),
            "rank_p50": _rank_quantile(out.rank_counts[i], 0.5),
            "rank_p90": _rank_quantile(out.rank_counts[i], 0.9),
            "model": f"m7-{model.key}",
            "model_version": model.version,
        }
        for i, team in enumerate(out.teams)
    ]


def latest_report(
    run: LiveRun, model: LiveSim, fmt: Format, season: int, at: datetime
) -> dict[str, Any]:
    """``reports/sim_latest_{competition}.json``: the log rows plus the run's settings and the
    format's sources and unverified rules (carried into every GBL output)."""
    out = run.out
    return {
        "competition": fmt.competition,
        "season": season,
        "after_round": run.after_round,
        "cutoff_utc": run.cutoff.strftime(TIME_FORMAT),
        "simulated_at_utc": at.strftime(TIME_FORMAT),
        "model": f"m7-{model.key}",
        "model_version": model.version,
        "n_sims": out.n_sims,
        "seed": run.seed,
        "format": {"sources": list(fmt.sources), "unverified": list(fmt.unverified)},
        "teams": log_rows(run, model, season, at),
        "p_playoffs": dict(zip(out.teams, np.round(out.p_playoffs, 4).tolist(), strict=True)),
        "p_final": dict(zip(out.teams, np.round(out.p_final, 4).tolist(), strict=True)),
    }


def write_live_sim(
    run: LiveRun,
    model: LiveSim,
    *,
    fmt: Format,
    season: int,
    at: datetime,
    log_path: Path,
    report_path: Path,
) -> int:
    """Append the run's rows and write the latest report, unless the gate failed or this
    number of completed rounds is already logged; the number of rows appended."""
    if not model.gate_passed:
        log.info("M7 did not pass its validation gate: no simulation logged")
        return 0
    if run.after_round in logged_rounds(log_path, season):
        log.info("M7 %s: after round %d already logged", fmt.competition, run.after_round)
        return 0
    rows = log_rows(run, model, season, at)
    append_rows(log_path, SIM_LOG_COLUMNS, rows)
    report_path.write_text(
        json.dumps(latest_report(run, model, fmt, season, at), indent=2) + "\n", encoding="utf-8"
    )
    return len(rows)
