"""Live M7 (K8): the state after the completed rounds, the gated append-only log, idempotence."""

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eurohoops.live_sim import (
    SIM_LOG_COLUMNS,
    LiveRun,
    LiveSim,
    completed_rounds,
    load_sim,
    logged_rounds,
    run_live_sim,
    with_prior,
    write_live_sim,
)
from eurohoops.models.team_eff import RatingPosterior
from tests.m7_synthetic import build_inputs, small_spec, synthetic_formats

LIVE = 2021  # the synthetic league's last season plays the live one
DONE = 9  # rounds 1-9 complete; round 10 half played
AT = datetime(2021, 12, 10, 12, 0, tzinfo=UTC)
MODEL = LiveSim(key="sim_full", spread=1.0, net=False, version="m7-sim_full-v1", gate_passed=True)


def _live_games(competition: str = "euroleague") -> tuple[pd.DataFrame, pd.DataFrame]:
    """The synthetic league with ``LIVE``'s rounds after ``DONE`` unplayed (round DONE + 1 half
    played), and its PO/FF games removed."""
    inputs = build_inputs(competition=competition)
    games = inputs.games.copy()
    games = games[~((games["season"] == LIVE) & (games["phase"] != "RS"))]
    live = (games["season"] == LIVE) & (games["round"] > DONE)
    half = live & (games["round"] == DONE + 1) & (games["game_code"] % 2 == 0)
    unplayed = live & ~half
    games.loc[unplayed, ["home_score", "away_score"]] = pd.NA
    games.loc[unplayed, "played"] = False
    team_games = inputs.team_games[
        inputs.team_games["game_id"].isin(games.loc[~unplayed, "game_id"])
    ]
    return games.reset_index(drop=True), team_games.reset_index(drop=True)


def _run(competition: str = "euroleague", n_sims: int = 400) -> LiveRun:
    games, team_games = _live_games(competition)
    inputs = build_inputs(competition=competition)
    run = run_live_sim(
        games,
        team_games,
        inputs.tuned_m1,
        MODEL,
        fmt=synthetic_formats(competition, LIVE),
        season=LIVE,
        spec=small_spec(),
        n_sims=n_sims,
    )
    assert run is not None
    return run


def test_completed_rounds_stop_at_the_first_incomplete_round() -> None:
    games, _ = _live_games()
    regular = games[(games["season"] == LIVE) & (games["phase"] == "RS")]
    assert completed_rounds(regular) == DONE
    assert completed_rounds(games[(games["season"] == LIVE - 1) & (games["phase"] == "RS")]) == 22


def test_live_run_state_and_probability_sums() -> None:
    run = _run()
    fmt = synthetic_formats("euroleague", LIVE)
    out = run.out
    assert run.after_round == DONE
    assert run.seed == small_spec().seed + DONE
    # the cutoff is round DONE + 1's first tip-off, so its played half is not in the state
    first_tipoff = pd.Timestamp("2021-10-01T18:00:00Z") + pd.Timedelta(days=7 * DONE)
    assert run.cutoff == first_tipoff
    assert np.isclose(out.p_direct.sum(), len(fmt.playoffs_direct))
    assert np.isclose(out.p_play_in.sum(), len(fmt.play_in))
    assert np.isclose(out.p_top10.sum(), 10)
    assert np.isclose(out.p_semis.sum(), 4)
    assert np.isclose(out.p_title.sum(), 1)
    # wins: DONE games played by every team, 22 in all
    assert np.isclose(out.wins.sum(), 12 * 22 / 2)  # one win per game of the season


def test_the_season_over_gives_no_run() -> None:
    inputs = build_inputs()
    games = inputs.games[~((inputs.games["season"] == LIVE) & (inputs.games["phase"] != "RS"))]
    run = run_live_sim(
        games.reset_index(drop=True),
        inputs.team_games,
        inputs.tuned_m1,
        MODEL,
        fmt=synthetic_formats("euroleague", LIVE),
        season=LIVE,
        spec=small_spec(),
        n_sims=50,
    )
    assert run is None


def test_log_is_gated_append_only_and_idempotent(tmp_path: Path) -> None:
    run = _run(n_sims=100)
    log_path, report = tmp_path / "sim.csv", tmp_path / "sim_latest.json"
    fmt = synthetic_formats("euroleague", LIVE)
    failed = replace(MODEL, gate_passed=False)
    kwargs = {"fmt": fmt, "season": LIVE, "at": AT, "log_path": log_path, "report_path": report}
    assert write_live_sim(run, failed, **kwargs) == 0
    assert not log_path.exists() and not report.exists()
    assert write_live_sim(run, MODEL, **kwargs) == 12
    first = log_path.read_bytes()
    assert write_live_sim(run, MODEL, **kwargs) == 0
    assert log_path.read_bytes() == first
    logged = pd.read_csv(log_path)
    assert tuple(logged.columns) == SIM_LOG_COLUMNS
    assert logged_rounds(log_path, LIVE) == {DONE}
    assert np.isclose(logged["p_title"].sum(), 1.0, atol=12 * 5e-5)  # 4-decimal rounding
    assert (logged["rank_p10"] <= logged["rank_p50"]).all()
    assert (logged["rank_p50"] <= logged["rank_p90"]).all()
    latest = json.loads(report.read_text(encoding="utf-8"))
    assert latest["after_round"] == DONE and latest["format"]["sources"] == ["synthetic"]


def test_gbl_report_carries_the_unverified_notes(tmp_path: Path) -> None:
    run = _run("gbl", n_sims=100)
    fmt = replace(synthetic_formats("gbl", LIVE), unverified=("playoff positions",))
    report = tmp_path / "sim_latest_gbl.json"
    write_live_sim(
        run,
        MODEL,
        fmt=fmt,
        season=LIVE,
        at=AT,
        log_path=tmp_path / "gbl_sim.csv",
        report_path=report,
    )
    assert json.loads(report.read_text(encoding="utf-8"))["format"]["unverified"] == [
        "playoff positions"
    ]


def _posterior() -> RatingPosterior:
    return RatingPosterior(
        teams=("A", "B", "C", "D"),
        columns=np.array([0, 1, 2, 3, 4, 6, 7, 8], dtype=np.int64),
        labels=("mu", "h", "off:A", "off:B", "off:D", "def:A", "def:B", "def:D"),
        mean=np.array([100.0, 2.0, 1.0, -1.0, -3.0, 0.5, -0.5, -2.0]),
        cov=np.diag([0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.6, 0.65]),
        sigma2=250.0,
    )


def test_a_promoted_team_starts_at_the_replaced_teams_mean() -> None:
    # C is new (no column); B and D left the league, so C starts at their average rating
    strengths = with_prior(_posterior(), ("A", "C"), ridge=125.0, replaced=("B", "D"))
    # [mu, h, off A, off C, def A, def C]
    assert np.allclose(strengths.mean, [100.0, 2.0, 1.0, -2.0, 0.5, -1.25])
    # the variance is M1's ridge prior sigma2 / ridge, independent of the rest
    assert np.allclose(np.diag(strengths.cov), [0.1, 0.2, 0.3, 2.0, 0.5, 2.0])
    assert strengths.cov[3, 2] == 0.0 and strengths.cov[5, 4] == 0.0


def test_without_a_rated_replaced_team_the_start_is_the_league_mean() -> None:
    strengths = with_prior(_posterior(), ("A", "C"), ridge=125.0, replaced=("Z",))
    assert np.allclose(strengths.mean, [100.0, 2.0, 1.0, 0.0, 0.5, 0.0])


def test_with_every_team_seen_it_is_restrict_posterior() -> None:
    seen = with_prior(_posterior(), ("A", "B", "D"), ridge=125.0, replaced=("C",))
    assert np.array_equal(seen.mean, _posterior().mean)


def test_load_sim_reads_the_verdict_and_gate(tmp_path: Path) -> None:
    path = tmp_path / "backtest_m7.json"
    assert load_sim(path) is None
    path.write_text(
        json.dumps(
            {
                "model_version": "m7-sim_net-v1",
                "chosen": {"key": "sim_net", "spread": 1.0, "net": True},
                "gate": {"passed": False},
            }
        ),
        encoding="utf-8",
    )
    model = load_sim(path)
    assert model == LiveSim("sim_net", 1.0, True, "m7-sim_net-v1", False)
    path.write_text(
        json.dumps({"model_version": "v", "chosen": {"key": "k", "spread": 1.0, "net": False}}),
        encoding="utf-8",
    )
    loaded = load_sim(path)
    assert loaded is not None and not loaded.gate_passed  # a tuning-only report has no gate


@pytest.mark.parametrize("competition", ["euroleague", "gbl"])
def test_same_inputs_same_numbers(competition: str) -> None:
    a, b = _run(competition, n_sims=100), _run(competition, n_sims=100)
    assert np.array_equal(a.out.rank_counts, b.out.rank_counts)
