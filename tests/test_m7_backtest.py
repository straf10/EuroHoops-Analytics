"""The M7 backtest harness (week 14-16 K4) on the synthetic league of ``tests/m7_synthetic.py``:
end to end (tuning only, validation, test, a fixed choice), the report shape, the gate, the
tuning-only discipline, reproducibility, the baselines, the scoring rules by hand, the checkpoint
rule, the sampler and the verdict-order check script."""

import json
import math
import re
import subprocess
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pandera.errors
import pytest
from scipy import stats
from typer.testing import CliRunner

from eurohoops import cli
from eurohoops.eval.m7_backtest import (
    BASELINES,
    ROW_COLUMNS,
    TEAMS_SCHEMA,
    M7Inputs,
    Strengths,
    _title_brier,
    champion_of,
    checkpoint_cutoff,
    checkpoint_round,
    choose_variant,
    cluster_bootstrap_ci,
    elo_strength_scale,
    format_m7_table,
    gate_block,
    gaussian_sampler,
    margin_noise,
    model_keys,
    net_noise,
    noise_variance,
    remaining_fixtures,
    rps_of,
    run_m7_backtest,
    scored_splits,
    spiegelhalter_z,
    standings_order,
    strength_variance,
    variants,
)
from eurohoops.eval.tracking import default_tracking_uri, log_m7_backtest
from eurohoops.logs import write_json
from eurohoops.models.elo import win_probability
from eurohoops.models.team_eff import MarginModel
from eurohoops.parse.schemas import validated
from eurohoops.sim.played import season_results
from eurohoops.sim.season import NoiseModel, PaceModel, simulate
from eurohoops.standings import Result, rank
from tests.m7_synthetic import (
    DEDUCTION,
    ROUNDS,
    TEAMS,
    build_inputs,
    small_spec,
    synthetic_formats,
)

SPEC = small_spec()
INPUTS = build_inputs()
REPO = Path(__file__).parent.parent
FIXTURES = Path(__file__).parent / "fixtures"
ORDER_SCRIPT = REPO / "scripts" / "checks" / "m7_order.py"
MODELS = model_keys(SPEC)
REPORT_KEYS = {
    "model",
    "model_version",
    "competition",
    "seasons",
    "checkpoints",
    "n_sims",
    "seeds",
    "formats",
    "data_sha256",
    "tuning_only",
    "validation_scored",
    "test_scored",
    "grid",
    "chosen",
    "metrics",
    "per_checkpoint",
    "reliability",
    "title_brier",
    "differences",
    "gate",
}
METRIC_KEYS = {"n", "brier", "rps", "log_loss", "brier_top10", "spiegelhalter_z"}
GATE_KEYS = {
    "rule",
    "variant",
    "brier_chosen",
    "brier_point_sim",
    "brier_standings_now",
    "spiegelhalter_z_pooled",
    "passed",
    "brier_vs_point_sim",
    "brier_vs_standings_now",
    "brier_vs_elo_sim",
    "gated",
}
Run = tuple[dict[str, Any], pd.DataFrame]


def run(inputs: M7Inputs = INPUTS, **kwargs: Any) -> Run:
    return run_m7_backtest(inputs, spec=SPEC, formats=synthetic_formats, **kwargs)


@pytest.fixture(scope="module")
def tuning_run() -> Run:
    return run(tuning_only=True)


@pytest.fixture(scope="module")
def full_run() -> Run:
    return run()


@pytest.fixture(scope="module")
def test_run() -> Run:
    return run(score_test=True)


@pytest.fixture(scope="module")
def gbl_run() -> Run:
    return run(build_inputs("gbl"), fixed="sim_net")


def csv_text(rows: pd.DataFrame, path: Path) -> bytes:
    rows.to_csv(path, index=False, lineterminator="\n")
    return path.read_bytes()


# --- End to end -------------------------------------------------------------------------------


def test_end_to_end_tuning_then_validation_then_test(
    tuning_run: Run, full_run: Run, test_run: Run
) -> None:
    for (report, rows), seasons, scored in (
        (tuning_run, (2017, 2018), (False, False)),
        (full_run, (2017, 2018, 2019), (True, False)),
        (test_run, (2017, 2018, 2019, 2020, 2021), (True, True)),
    ):
        assert (report["validation_scored"], report["test_scored"]) == scored
        assert sorted(set(rows["season"])) == list(seasons)
        assert len(rows) == len(seasons) * 3 * len(TEAMS) * len(MODELS)
        assert list(rows.columns) == list(ROW_COLUMNS)
        assert report["model"] == "m7"
        assert report["model_version"] == f"m7-{report['chosen']['key']}-v1"
    assert "gate" not in tuning_run[0]
    assert full_run[0]["gate"]["variant"] == full_run[0]["chosen"]["key"]


def test_the_tuning_numbers_do_not_depend_on_the_other_splits(
    tuning_run: Run, full_run: Run, test_run: Run
) -> None:
    """Seeds follow the (season, fraction) order, so adding later seasons changes no tuning row."""
    base = tuning_run[1]
    for _, rows in (full_run, test_run):
        pd.testing.assert_frame_equal(base, rows[rows["split"] == "tuning"].reset_index(drop=True))
    for report, _ in (full_run, test_run):
        assert report["metrics"]["tuning"] == tuning_run[0]["metrics"]["tuning"]
        assert report["chosen"] == tuning_run[0]["chosen"]


def test_a_fixed_choice_scores_that_variant_and_is_not_gated(gbl_run: Run) -> None:
    report, rows = gbl_run
    assert report["competition"] == "gbl"
    assert report["chosen"]["key"] == "sim_net"
    assert "fixed" in report["chosen"]
    assert report["model_version"] == "m7-sim_net-v1"
    assert report["gate"]["gated"] is False
    assert rows["p_top10"].isna().all()  # the GBL has no play-in
    assert report["metrics"]["validation"]["sim_net"]["brier_top10"] is None
    with pytest.raises(ValueError, match="fixed variant"):
        run(fixed="point_sim")


def test_every_model_forecasts_a_proper_distribution(test_run: Run) -> None:
    rows = test_run[1]
    lines = {season: synthetic_formats("euroleague", season) for season in set(rows["season"])}
    for (season, _, model), group in rows.groupby(["season", "checkpoint", "model"]):
        assert group["p_title"].sum() == pytest.approx(1.0, abs=1e-5), model
        assert group["exp_rank"].sum() == pytest.approx(len(TEAMS) * (len(TEAMS) + 1) / 2, abs=1e-4)
        assert group["p_direct"].sum() == pytest.approx(
            len(lines[season].playoffs_direct), abs=1e-4
        )
        assert group["rps"].between(0.0, 1.0).all()


def test_targets_are_the_real_final_table_with_deductions(full_run: Run) -> None:
    rows = full_run[1]
    for season, group in rows[rows["model"] == "sim_full"].groupby("season"):
        fmt = synthetic_formats("euroleague", int(season))
        games = INPUTS.games
        regular = games[(games["season"] == season) & (games["phase"] == "RS")]
        table = rank(season_results(regular), fmt.deducted_wins())
        place = {team: i for i, team in enumerate(table, start=1)}
        assert (group["team"].map(place) == group["final_place"]).all()
        assert (group["made_direct"] == group["final_place"].isin(fmt.playoffs_direct)).all()
        if fmt.play_in:
            assert (group["made_top10"] == (group["final_place"] <= 10)).all()
        else:
            assert group["made_top10"].isna().all()
        last_final_four = games[(games["season"] == season) & (games["phase"] == "FF")]
        champion = _winner(last_final_four.sort_values("tipoff_utc").iloc[-1])
        assert group["champion"].sum() == 3  # one champion, in each of the 3 checkpoints
        assert set(group.loc[group["champion"] == 1, "team"]) == {champion}
    # the deduction changes the table, so the target above is not vacuous for 2018
    games = INPUTS.games
    regular = games[(games["season"] == DEDUCTION[0]) & (games["phase"] == "RS")]
    fmt = synthetic_formats("euroleague", DEDUCTION[0])
    assert rank(season_results(regular), fmt.deducted_wins()) != rank(season_results(regular))


def _winner(game: pd.Series) -> str:
    return str(game["home"] if game["home_score"] > game["away_score"] else game["away"])


def test_the_champion_rule_follows_the_competition() -> None:
    games = INPUTS.games[INPUTS.games["season"] == 2018]
    final_four = games[games["phase"] == "FF"].sort_values("tipoff_utc").iloc[-1]
    playoffs = games[games["phase"] == "PO"].sort_values("tipoff_utc").iloc[-1]
    assert champion_of(games, "FF") == _winner(final_four)
    assert champion_of(games, "PO") == _winner(playoffs)
    assert champion_of(games[games["phase"] == "RS"], "FF") is None


# --- The report ---------------------------------------------------------------------------------


def test_report_keys_and_types(test_run: Run) -> None:
    report, _ = test_run
    assert set(report) == REPORT_KEYS
    assert isinstance(report["seasons"], dict)
    assert report["seasons"] == {
        "tuning": [2017, 2018],
        "validation": [2019],
        "test": [2020, 2021],
    }
    assert report["checkpoints"] == [0.25, 0.5, 0.75]
    assert report["n_sims"] == 200
    assert len(report["data_sha256"]) == 64
    assert set(report["formats"]) == {"2017", "2018", "2019", "2020", "2021"}
    assert set(report["formats"]["2018"]) == {"label", "teams", "rounds", "sources", "unverified"}
    assert [set(s) for s in report["seeds"]] == [
        {"season", "fraction", "round", "cutoff", "seed"}
    ] * 15
    grid = report["grid"]
    assert set(grid) == {"objective", "tie_tolerance", "candidates", "baselines", "best_on_edge"}
    assert list(grid["candidates"]) == [v.key for v in variants(SPEC)]
    assert list(grid["baselines"]) == list(BASELINES)
    assert isinstance(grid["best_on_edge"], bool)
    assert set(report["chosen"]) >= {"key", "spread", "net", "tuning_rps", "tuning_brier"}
    for split in ("tuning", "validation", "test"):
        assert list(report["metrics"][split]) == list(MODELS)
        assert set(report["metrics"][split]["sim_full"]) == METRIC_KEYS
        assert list(report["per_checkpoint"][split]) == ["0.25", "0.5", "0.75"]
        assert list(report["per_checkpoint"][split]["0.5"]) == list(MODELS)
        assert set(report["per_checkpoint"][split]["0.5"]["sim_net"]) == METRIC_KEYS
        assert list(report["reliability"][split]) == list(MODELS)
        assert len(report["reliability"][split]["sim_full"]) == 10
        assert set(report["reliability"][split]["sim_full"][0]) == {
            "low",
            "high",
            "n",
            "mean_p",
            "observed",
        }
        assert list(report["title_brier"][split]) == list(MODELS)
        assert set(report["differences"][split]) == {
            "brier_vs_point_sim",
            "brier_vs_standings_now",
            "brier_vs_elo_sim",
        }
        ci = report["differences"][split]["brier_vs_point_sim"]
        assert set(ci) == {"mean", "ci95", "clusters", "resamples", "seed"}
        assert ci["ci95"][0] <= ci["ci95"][1]
        assert (ci["clusters"], ci["resamples"], ci["seed"]) == (
            len(TEAMS) * len(report["seasons"][split]),
            200,
            20261102,
        )
    assert set(report["gate"]) == GATE_KEYS
    json.dumps(report)  # JSON-ready: no numpy scalars, no NaN


def test_the_report_numbers_match_the_csv_rows(test_run: Run) -> None:
    report, rows = test_run
    for split, by_model in report["metrics"].items():
        for model, metrics in by_model.items():
            part = rows[(rows["split"] == split) & (rows["model"] == model)]
            p, y = part["p_direct"], part["made_direct"]
            assert metrics["n"] == len(part)
            assert metrics["brier"] == pytest.approx(((p - y) ** 2).mean(), abs=1e-6)
            assert metrics["rps"] == pytest.approx(part["rps"].mean(), abs=1e-6)
            clipped = p.clip(1e-4, 1 - 1e-4)
            loss = -(y * np.log(clipped) + (1 - y) * np.log(1 - clipped)).mean()
            assert metrics["log_loss"] == pytest.approx(loss, abs=1e-6)
    for split, by_fraction in report["per_checkpoint"].items():
        for fraction, by_model in by_fraction.items():
            part = rows[
                (rows["split"] == split)
                & (rows["model"] == "sim_net")
                & (rows["checkpoint"] == float(fraction))
            ]
            assert by_model["sim_net"]["n"] == len(part)
            assert by_model["sim_net"]["rps"] == pytest.approx(part["rps"].mean(), abs=1e-6)
    assert report["metrics"]["tuning"]["sim_full"]["brier_top10"] is None  # no play-in there
    play_in = rows[(rows["split"] == "validation") & (rows["model"] == "sim_full")]
    top = (play_in["p_top10"] - play_in["made_top10"]) ** 2
    assert report["metrics"]["validation"]["sim_full"]["brier_top10"] == pytest.approx(
        top.mean(), abs=1e-6
    )


def test_the_choice_is_the_lowest_tuning_rps_with_the_simplicity_tie_break(
    full_run: Run,
) -> None:
    report, _ = full_run
    table = [(k, m["rps"]) for k, m in report["grid"]["candidates"].items()]
    assert report["chosen"]["key"] == table[choose_variant(table, SPEC.tie_tolerance)][0]
    best = min(rps for _, rps in table)
    assert report["chosen"]["tuning_rps"] - best < SPEC.tie_tolerance
    assert "point_sim" not in table[choose_variant(table, SPEC.tie_tolerance)][0]


def test_choose_variant_ties_go_to_the_simpler_variant() -> None:
    order = ["sim_full", "sim_net", "sim_inflate_1.5", "sim_inflate_2"]

    def pick(rps: list[float]) -> str:
        return order[choose_variant(list(zip(order, rps, strict=True)), 0.002)]

    assert pick([0.1, 0.0985, 0.099, 0.1001]) == "sim_full"  # all within 0.002 of the best
    assert pick([0.101, 0.0985, 0.099, 0.1001]) == "sim_net"  # full is 0.0025 behind
    assert pick([0.12, 0.11, 0.1095, 0.1]) == "sim_inflate_2"  # a clear winner
    assert pick([0.002, 0.0, 0.5, 0.5]) == "sim_net"  # a difference of exactly the tolerance
    with pytest.raises(ValueError, match="finite"):
        choose_variant([("sim_full", math.nan)], 0.002)


def test_the_format_table_prints_the_gate(full_run: Run, tuning_run: Run) -> None:
    text = format_m7_table(full_run[0])
    assert full_run[0]["chosen"]["key"] in text
    assert "gate" in text and ("PASS" in text or "FAIL" in text)
    assert "gate" not in format_m7_table(tuning_run[0]) and "validation" not in format_m7_table(
        tuning_run[0]
    )


# --- The gate ---------------------------------------------------------------------------------


def test_gate_block_passes_and_fails() -> None:
    kwargs: dict[str, Any] = {"differences": {"brier_vs_point_sim": None}}
    passing = gate_block(
        "sim_net",
        brier_chosen=0.10,
        brier_point_sim=0.11,
        brier_standings_now=0.20,
        z_pooled=-1.5,
        **kwargs,
    )
    assert passing["passed"] is True
    assert passing["variant"] == "sim_net" and "brier_vs_point_sim" in passing
    for brier, point, standings, z in (
        (0.12, 0.11, 0.20, 0.0),  # not below point_sim
        (0.10, 0.11, 0.09, 0.0),  # not below standings_now
        (0.10, 0.11, 0.20, 1.96),  # miscalibrated: |z| not below 1.96
        (0.10, 0.11, 0.20, -2.5),
        (0.11, 0.11, 0.20, 0.0),  # a tie is not a pass (strict)
    ):
        failing = gate_block(
            "sim_net",
            brier_chosen=brier,
            brier_point_sim=point,
            brier_standings_now=standings,
            z_pooled=z,
            **kwargs,
        )
        assert failing["passed"] is False, (brier, point, standings, z)
    undecided = gate_block(
        "sim_net",
        brier_chosen=None,
        brier_point_sim=None,
        brier_standings_now=None,
        z_pooled=None,
        **kwargs,
    )
    assert undecided["passed"] is None


def test_the_report_gate_follows_the_rule(full_run: Run) -> None:
    report, rows = full_run
    gate, chosen = report["gate"], report["chosen"]["key"]
    validation = rows[rows["split"] == "validation"]

    def brier(model: str) -> float:
        part = validation[validation["model"] == model]
        return float(((part["p_direct"] - part["made_direct"]) ** 2).mean())

    pooled = rows[(rows["model"] == chosen) & rows["split"].isin(["tuning", "validation"])]
    z = spiegelhalter_z(
        pooled["p_direct"].to_numpy(dtype=float), pooled["made_direct"].to_numpy(dtype=float)
    )
    assert z is not None
    assert gate["spiegelhalter_z_pooled"] == pytest.approx(z, abs=1e-6)
    assert gate["brier_chosen"] == pytest.approx(brier(chosen), abs=1e-6)
    assert gate["brier_point_sim"] == pytest.approx(brier("point_sim"), abs=1e-6)
    assert gate["brier_standings_now"] == pytest.approx(brier("standings_now"), abs=1e-6)
    expected = (
        brier(chosen) < brier("point_sim")
        and brier(chosen) < brier("standings_now")
        and abs(z) < 1.96
    )
    assert gate["passed"] is expected
    assert gate["gated"] is True
    assert gate["brier_vs_point_sim"] == report["differences"]["validation"]["brier_vs_point_sim"]


# --- The tuning-only discipline -----------------------------------------------------------------


def _tokens(value: Any) -> list[Any]:
    """Every dict key and string value of a JSON tree."""
    if isinstance(value, dict):
        return [t for k, v in value.items() for t in [k, *_tokens(v)]]
    if isinstance(value, list):
        return [t for v in value for t in _tokens(v)]
    return [value] if isinstance(value, str) else []


# a season number on its own (not part of a number, a decimal or a calendar date: the cutoffs of
# the 2018 season fall in 2019)
LATER_SEASON = re.compile(r"(?<![\d.-])(2019|2020|2021)(?![\d-])")


def _text(report: dict[str, Any]) -> str:
    return json.dumps({k: v for k, v in report.items() if k != "data_sha256"}, indent=2)


def test_tuning_only_holds_no_validation_or_test_number(
    tuning_run: Run, test_run: Run, tmp_path: Path
) -> None:
    report, rows = tuning_run
    assert "validation" not in _tokens(report) and "test" not in _tokens(report)
    assert not LATER_SEASON.search(_text(report))
    assert set(rows["split"]) == {"tuning"} and set(rows["season"]) == {2017, 2018}
    assert not LATER_SEASON.search(csv_text(rows, tmp_path / "t.csv").decode())
    assert {s["season"] for s in report["seeds"]} == {2017, 2018}
    assert set(report["formats"]) == {"2017", "2018"}
    # the search is not vacuous: the full report does contain those tokens
    assert "validation" in _tokens(test_run[0]) and LATER_SEASON.search(_text(test_run[0]))
    # --score-test cannot reintroduce them when tuning only
    again, _ = run(tuning_only=True, score_test=True)
    assert again == report


def test_scored_splits() -> None:
    assert scored_splits(SPEC, True, True) == {2017: "tuning", 2018: "tuning"}
    assert scored_splits(SPEC, False, False) == {2017: "tuning", 2018: "tuning", 2019: "validation"}
    assert list(scored_splits(SPEC, False, True)) == [2017, 2018, 2019, 2020, 2021]


# --- Reproducibility ----------------------------------------------------------------------------


def test_two_runs_are_byte_identical(full_run: Run, tmp_path: Path) -> None:
    second = run()
    for name, (report, rows) in {"a": full_run, "b": second}.items():
        write_json(tmp_path / f"{name}.json", report)
        csv_text(rows, tmp_path / f"{name}.csv")
    assert (tmp_path / "a.json").read_bytes() == (tmp_path / "b.json").read_bytes()
    assert (tmp_path / "a.csv").read_bytes() == (tmp_path / "b.csv").read_bytes()
    assert b"\r" not in (tmp_path / "a.csv").read_bytes()
    assert b"\r" not in (tmp_path / "a.json").read_bytes()


def test_a_different_seed_changes_the_simulations(full_run: Run) -> None:
    other, _ = run_m7_backtest(
        INPUTS, spec=replace(SPEC, seed=SPEC.seed + 1000), formats=synthetic_formats
    )
    assert other["metrics"] != full_run[0]["metrics"]
    assert (
        other["metrics"]["tuning"]["standings_now"]
        == full_run[0]["metrics"]["tuning"]["standings_now"]
    )  # the one model without a draw


# --- Baselines and the sampler ------------------------------------------------------------------


def _engine_inputs() -> tuple[Strengths, dict[str, Any]]:
    """A posterior-shaped Strengths for the 12 teams and a small engine problem."""
    rng = np.random.default_rng(3)
    k = 2 + 2 * len(TEAMS)
    mean = np.concatenate([[100.0, 3.0], rng.normal(0.0, 2.0, 2 * len(TEAMS))])
    root = rng.normal(0.0, 0.3, (k, k))
    games = INPUTS.games
    regular = games[(games["season"] == 2019) & (games["phase"] == "RS")]
    cutoff = checkpoint_cutoff(regular, 11)
    problem = {
        "teams": TEAMS,
        "played": season_results(regular[regular["tipoff_utc"] < cutoff]),
        "remaining": remaining_fixtures(regular, cutoff),
        "pace": PaceModel(70.0, rng.normal(0.0, 1.0, len(TEAMS))),
        "noise": NoiseModel(10.0, 7.0),
        "fmt": synthetic_formats("euroleague", 2019),
        "n_sims": 300,
        "seed": 5,
    }
    return Strengths(mean, root @ root.T + np.eye(k)), problem


def test_point_sim_equals_sim_full_with_zero_covariance_exactly() -> None:
    strengths, problem = _engine_inputs()
    zero = replace(strengths, cov=np.zeros_like(strengths.cov))
    full_zero = gaussian_sampler(zero, 1.0)  # sim_full with no covariance
    point = gaussian_sampler(strengths, 0.0)  # point_sim: c = 0
    a, b = np.random.default_rng(1), np.random.default_rng(1)
    for x, y in zip(full_zero(50, a), point(50, b), strict=True):
        assert np.array_equal(x, y)
    assert a.random() == b.random()  # the same normals were drawn
    out_zero = simulate(sampler=full_zero, **problem)
    out_point = simulate(sampler=point, **problem)
    assert np.array_equal(out_zero.rank_counts, out_point.rank_counts)
    assert np.array_equal(out_zero.p_title, out_point.p_title)
    # every simulation gets exactly M1's point strengths
    off, def_, home = point(10, np.random.default_rng(2))
    n = len(TEAMS)
    assert np.array_equal(
        off, np.broadcast_to(strengths.mean[2 : 2 + n] + strengths.mean[0], off.shape)
    )
    assert np.array_equal(def_, np.broadcast_to(strengths.mean[2 + n :], def_.shape))
    assert np.array_equal(home, np.full(10, strengths.mean[1]))


def test_the_sampler_draws_the_posterior() -> None:
    """Sample mean and covariance of many draws match m and c * S (sampling error ~ 1/sqrt(N))."""
    rng = np.random.default_rng(8)
    n = 3
    k = 2 + 2 * n  # [mu, h, off * 3, def * 3]
    mean = rng.normal(0.0, 1.0, k)
    root = rng.normal(0.0, 1.0, (k, k))
    cov = root @ root.T / k + 0.5 * np.eye(k)
    draws = 200_000
    off, def_, home = gaussian_sampler(Strengths(mean, cov), 2.0)(draws, np.random.default_rng(4))
    assert (off.shape, def_.shape, home.shape) == ((draws, n), (draws, n), (draws,))
    # The margin's strength part for team 0 at home to team 1, 2h + off0 - off1 + def0 - def1
    # (mu folded into off cancels), is a'theta with this a, so it is Normal with mean a'm and
    # variance a'(2S)a. The sample mean has sd sqrt(var / N); the sample variance has relative
    # sd sqrt(2 / N) = 0.3%, so 5 sd is a 1.6% band.
    a = np.zeros(k)
    a[[1, 2, 3, 2 + n, 3 + n]] = [2.0, 1.0, -1.0, 1.0, -1.0]
    value = 2.0 * home + off[:, 0] - off[:, 1] + def_[:, 0] - def_[:, 1]
    variance = a @ (2.0 * cov) @ a
    assert value.mean() == pytest.approx(a @ mean, abs=5 * math.sqrt(variance / draws))
    assert value.var() == pytest.approx(variance, rel=0.016)


def test_standings_now_equals_rank_on_the_checkpoint_table(test_run: Run) -> None:
    report, rows = test_run
    tied = 0
    for cp in report["seeds"]:
        cutoff = pd.Timestamp(cp["cutoff"])
        games = INPUTS.games
        regular = games[(games["season"] == cp["season"]) & (games["phase"] == "RS")]
        played = season_results(regular[regular["tipoff_utc"] < cutoff])
        deducted = synthetic_formats("euroleague", cp["season"]).deducted_wins()
        part = rows[
            (rows["model"] == "standings_now")
            & (rows["season"] == cp["season"])
            & (rows["checkpoint"] == cp["fraction"])
        ].sort_values("exp_rank")
        wins = Counter(r.winner for r in played)
        tied += len({wins[t] - deducted.get(t, 0) for t in TEAMS}) < len(TEAMS)
        assert list(part["team"]) == rank(played, deducted)
        assert list(part["exp_rank"]) == list(range(1, len(TEAMS) + 1))  # a point mass
        assert part["p_title"].tolist() == [1.0] + [0.0] * (len(TEAMS) - 1)
        assert set(part["p_direct"]) <= {0.0, 1.0}
    assert tied  # the checkpoints include tied tables, so the tie-breaks are exercised


def test_standings_order_ranks_by_win_percentage_then_rank() -> None:
    results = [
        Result("A", "G", 80, 70),
        Result("F", "B", 80, 70),
        Result("F", "H", 80, 70),
        Result("G", "F", 75, 70),
        Result("B", "F", 75, 70),
    ]
    teams = ["A", "B", "F", "G", "H", "Z"]
    # hand-worked: A is 1-0 (100%), F 2-2, G 1-1, B 1-1 (50%), H 0-1, Z has not played.
    # rank() is [F, A, B, G, H]: F leads on 2 wins, A tops the 1-win group on point difference
    # (+10 against -5), and B/G tie on every criterion (alphabetical). Percentages put A first,
    # then the 50% teams in rank's order F, B, G, then H (0-1), then Z (no games).
    assert rank(results) == ["F", "A", "B", "G", "H"]
    assert standings_order(teams, results, {}) == ["A", "F", "B", "G", "H", "Z"]
    # a deducted win comes off the numerator: A is (1 - 1) / 1 = 0%, in the 0% group behind H
    # because rank() puts a team with deducted wins last among the teams it is tied with
    assert rank(results, {"A": 1}) == ["F", "B", "G", "H", "A"]
    assert standings_order(teams, results, {"A": 1}) == ["F", "B", "G", "H", "A", "Z"]


def test_elo_sim_uses_elos_win_probability_at_an_even_game() -> None:
    """With margin = c * d the engine's P(home win) has Elo's slope at d = 0; the second-order
    terms of the logistic and of the t cdf are relative (ln10 * d / 400)^2 / 12 ~ 3e-6 at d = 1."""
    d = 1.0
    for noise in (NoiseModel(10.402421, 7.0), NoiseModel(11.4, 20.0), NoiseModel(9.0, None)):
        z = elo_strength_scale(noise) * d / noise.scale
        engine = stats.norm.cdf(z) if noise.df is None else stats.t.cdf(z, noise.df)
        assert engine - 0.5 == pytest.approx(win_probability(d) - 0.5, rel=1e-4)
    normal = NoiseModel(10.0, None)
    assert elo_strength_scale(normal) == pytest.approx(
        10.0 * math.log(10.0) * math.sqrt(2.0 * math.pi) / 1600.0, rel=1e-12
    )


def test_net_noise_by_hand() -> None:
    # t(4): V = 10^2 * 4/2 = 200; strength variance 50 leaves 150 = scale^2 * 4/2 -> scale^2 = 75
    t = net_noise(NoiseModel(10.0, 4.0), 50.0, 0.5)
    assert (t.scale, t.df) == (pytest.approx(math.sqrt(75.0)), 4.0)
    assert noise_variance(t) + 50.0 == pytest.approx(noise_variance(NoiseModel(10.0, 4.0)))
    # Normal: V = 100; strength variance 36 leaves 64 -> scale 8
    assert net_noise(NoiseModel(10.0, None), 36.0, 0.5).scale == pytest.approx(8.0)
    # the floor: strength variance above V leaves nothing; the scale is 50% of M1's
    assert net_noise(NoiseModel(10.0, 4.0), 500.0, 0.5).scale == pytest.approx(5.0)
    # a spare variance below the floor is lifted to it: V - 90 = 10 -> 3.16 < 5
    assert net_noise(NoiseModel(10.0, None), 90.0, 0.5).scale == pytest.approx(5.0)
    with pytest.raises(ValueError, match="df > 2"):
        noise_variance(NoiseModel(10.0, 2.0))


def test_strength_variance_by_hand() -> None:
    """Two teams, [mu, h, off0, off1, def0, def1] with covariance 2 * I. Home team 0, away team 1:
    a = [0, 2, +1, -1, +1, -1] (h: 2*hf, off_H, off_A, def_H, def_A), so a'Sa = 2 * (4 + 4) = 16."""
    cov = 2.0 * np.eye(6)
    home, away = np.array([0]), np.array([1])
    pace = np.array([100.0])
    assert strength_variance(cov, home, away, np.array([False]), pace) == pytest.approx(16.0)
    assert strength_variance(cov, home, away, np.array([True]), pace) == pytest.approx(8.0)
    assert strength_variance(cov, home, away, np.array([False]), pace / 2) == pytest.approx(4.0)
    # two games average: (16 + 16) / 2 with the second game the reverse fixture
    both = strength_variance(
        cov, np.array([0, 1]), np.array([1, 0]), np.array([False, False]), np.array([100.0, 100.0])
    )
    assert both == pytest.approx(16.0)


def test_a_pace_variant_scales_the_noise_with_the_mean_pace() -> None:
    constant = margin_noise(MarginModel("student_t_const", 10.0, 7.0), 77.0)
    assert (constant.scale, constant.df) == (10.0, 7.0)
    paced = margin_noise(MarginModel("student_t_pace", 11.4, 20.0, 70.0), 77.0)
    assert paced.scale == pytest.approx(11.4 * math.sqrt(77.0 / 70.0))
    assert margin_noise(MarginModel("normal_const", 9.0), 70.0).df is None


# --- Scoring rules by hand ------------------------------------------------------------------------


def test_rps_by_hand() -> None:
    # P = (0.5, 0.3, 0.2), final place 2: P(<=1) = 0.5 vs 0, P(<=2) = 0.8 vs 1
    assert rps_of(np.array([0.5, 0.3, 0.2]), 2) == pytest.approx((0.25 + 0.04) / 2)
    assert rps_of(np.array([0.0, 1.0, 0.0]), 2) == 0.0  # certain and right
    assert rps_of(np.array([1.0, 0.0, 0.0]), 3) == 1.0  # certain and as wrong as possible
    # the distance matters: a near miss scores better than a far one
    near = rps_of(np.array([0.0, 1.0, 0.0, 0.0]), 1)
    far = rps_of(np.array([0.0, 0.0, 0.0, 1.0]), 1)
    assert near < far


def test_spiegelhalter_z_by_hand() -> None:
    # p = (0.2, 0.8), y = (0, 1): numerator -0.12 - 0.12, variance 2 * 0.36 * 0.16
    z = spiegelhalter_z(np.array([0.2, 0.8]), np.array([0.0, 1.0]))
    assert z == pytest.approx(-1.0 / math.sqrt(2.0), rel=1e-12)
    assert spiegelhalter_z(np.array([0.5, 0.5]), np.array([0.0, 1.0])) is None  # zero variance
    assert spiegelhalter_z(np.array([0.0, 1.0]), np.array([0.0, 1.0])) is None


def test_title_brier_by_hand() -> None:
    rows = pd.DataFrame(
        {
            "season": [2000] * 4 + [2001] * 2,
            "checkpoint": [0.25, 0.25, 0.5, 0.5, 0.25, 0.25],
            "p_title": [0.6, 0.4, 0.9, 0.1, 0.5, 0.5],
            "champion": pd.array([1, 0, 1, 0, None, None], dtype="Int64"),
        }
    )
    # 2000: (0.4^2 + 0.4^2) = 0.32 and (0.1^2 + 0.1^2) = 0.02, mean 0.17; 2001 is unknown
    assert _title_brier(rows) == pytest.approx(0.17)
    assert _title_brier(rows[rows["season"] == 2001]) is None


def test_cluster_bootstrap_resamples_whole_clusters() -> None:
    diff = np.array([1.0, 1.0, 3.0, 3.0])
    cluster = np.array([0, 0, 1, 1])
    mean, low, high = cluster_bootstrap_ci(diff, cluster, 1000, 20261102)
    # cluster means are 1 and 3, so a resample's mean is 1, 2 or 3 with P = 1/4, 1/2, 1/4: the
    # 2.5% and 97.5% points are the extremes
    assert (mean, low, high) == (2.0, 1.0, 3.0)
    assert cluster_bootstrap_ci(diff, cluster, 1000, 20261102) == (mean, low, high)
    assert all(math.isnan(v) for v in cluster_bootstrap_ci(np.array([]), np.array([], int), 10, 1))
    # unequal clusters weight rows by cluster size
    same = cluster_bootstrap_ci(np.array([2.0, 2.0, 2.0]), np.array([0, 0, 1]), 100, 3)
    assert same == (2.0, 2.0, 2.0)


# --- Checkpoints --------------------------------------------------------------------------------


def test_checkpoint_rounds_are_floor_fraction_of_the_rounds() -> None:
    fmt = synthetic_formats("euroleague", 2019)
    for rounds, expected in ((30, (7, 15, 22)), (34, (8, 17, 25)), (38, (9, 19, 28))):
        sized = replace(fmt, regular_season_rounds=rounds)
        assert tuple(checkpoint_round(sized, f) for f in (0.25, 0.5, 0.75)) == expected
    for rounds, expected in ((22, (5, 11, 16)), (26, (6, 13, 19))):
        sized = replace(fmt, regular_season_rounds=rounds)
        assert tuple(checkpoint_round(sized, f) for f in (0.25, 0.5, 0.75)) == expected


def test_the_cutoff_is_the_first_tipoff_of_the_next_round(test_run: Run) -> None:
    games = INPUTS.games
    for cp in test_run[0]["seeds"]:
        regular = games[(games["season"] == cp["season"]) & (games["phase"] == "RS")]
        assert cp["round"] == math.floor(cp["fraction"] * ROUNDS)
        following = regular[regular["round"] == cp["round"] + 1]
        assert pd.Timestamp(cp["cutoff"]) == following["tipoff_utc"].min()
        # the state is exactly the rounds before it
        before = regular[regular["tipoff_utc"] < pd.Timestamp(cp["cutoff"])]
        assert before["round"].max() == cp["round"]
    with pytest.raises(ValueError, match="no regular-season round"):
        checkpoint_cutoff(games[games["season"] == 2019], ROUNDS + 5)


def test_seeds_follow_the_season_then_fraction_order(test_run: Run) -> None:
    seeds = test_run[0]["seeds"]
    keys = [(s["season"], s["fraction"]) for s in seeds]
    assert keys == sorted(keys)
    assert [s["seed"] for s in seeds] == [SPEC.seed + i for i in range(len(seeds))]
    assert len(keys) == 5 * 3


def _frame(rows: list[tuple[str, str, int, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "home": [r[0] for r in rows],
            "away": [r[1] for r in rows],
            "round": [r[2] for r in rows],
            "tipoff_utc": pd.to_datetime([r[3] for r in rows], utc=True),
            "neutral": False,
        }
    )


def test_a_postponed_game_is_remaining_and_a_repeated_pair_keeps_its_first_row() -> None:
    season = _frame(
        [
            ("A", "B", 1, "2019-10-01T18:00:00Z"),
            ("C", "D", 1, "2019-10-01T18:15:00Z"),
            ("A", "C", 2, "2019-10-15T18:00:00Z"),
            ("B", "D", 2, "2019-10-22T18:00:00Z"),  # round 2, postponed past the cutoff
            ("D", "A", 3, "2019-10-29T18:00:00Z"),
            ("C", "B", 3, "2019-10-29T18:00:00Z"),
            ("C", "B", 9, "2019-12-01T18:00:00Z"),  # D2: the same ordered pair listed again
        ]
    )
    cutoff = checkpoint_cutoff(season, 2)  # first tip-off of round 3
    assert cutoff == pd.Timestamp("2019-10-29T18:00:00Z")
    rest = remaining_fixtures(season, cutoff)
    assert list(rest.columns) == ["home", "away", "neutral", "tipoff_utc"]
    # sorted by tip-off then home; ("C", "B") appears once, at its first time
    assert list(zip(rest["home"], rest["away"], strict=True)) == [("C", "B"), ("D", "A")]
    assert rest["tipoff_utc"].is_monotonic_increasing
    earlier = remaining_fixtures(season, pd.Timestamp("2019-10-10T00:00:00Z"))
    assert ("B", "D") in set(zip(earlier["home"], earlier["away"], strict=True))
    assert len(earlier) == 4  # A-C, B-D, D-A, C-B (the second C-B is dropped)


# --- The sample, the schema and the tracking ------------------------------------------------


def test_the_schema_accepts_the_sample_and_the_run(full_run: Run, tmp_path: Path) -> None:
    sample = pd.read_csv(FIXTURES / "m7_teams_sample.csv")
    assert list(sample.columns) == list(ROW_COLUMNS)
    validated(sample, TEAMS_SCHEMA)
    path = tmp_path / "rows.csv"
    full_run[1].to_csv(path, index=False, lineterminator="\n")
    back = validated(pd.read_csv(path), TEAMS_SCHEMA)
    assert len(back) == len(full_run[1])
    bad = sample.assign(p_direct=[1.2, *sample["p_direct"][1:]])
    with pytest.raises(pandera.errors.SchemaError):
        validated(bad, TEAMS_SCHEMA)
    with pytest.raises(pandera.errors.SchemaError):
        validated(sample.assign(split="holdout"), TEAMS_SCHEMA)
    with pytest.raises(pandera.errors.SchemaError):
        validated(pd.concat([sample, sample], ignore_index=True), TEAMS_SCHEMA)  # duplicated keys


def test_m7_backtest_is_one_run_with_the_choice_and_the_report(
    full_run: Run, tmp_path: Path
) -> None:
    import mlflow  # noqa: PLC0415 - the dev dependency under test

    report, _ = full_run
    uri = default_tracking_uri(tmp_path / "mlruns")
    run_id = log_m7_backtest(report, "euroleague", uri)
    assert run_id is not None
    mlflow.set_tracking_uri(uri)
    runs = mlflow.search_runs(experiment_names=["m7-backtest"])
    assert isinstance(runs, pd.DataFrame)
    logged = runs[runs["run_id"] == run_id].iloc[0]
    assert logged["params.data_sha256"] == report["data_sha256"]
    assert logged["params.chosen.key"] == report["chosen"]["key"]
    assert logged["metrics.chosen.tuning_rps"] == report["chosen"]["tuning_rps"]
    assert [a.path for a in mlflow.MlflowClient().list_artifacts(run_id)] == [
        "backtest_m7_euroleague.json"
    ]


def test_the_cli_refuses_a_tuning_only_gbl_run() -> None:
    """The GBL scores the committed EuroLeague verdict, so it has nothing to do when tuning only."""
    result = CliRunner().invoke(
        cli.app, ["backtest", "--model", "m7", "--competition", "gbl", "--tuning-only"]
    )
    assert result.exit_code == 1


# --- scripts/checks/m7_order.py -----------------------------------------------------------------


def _git(repo: Path, *args: str) -> str:
    done = subprocess.run(
        [
            "git",
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.com",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    return done.stdout.strip()


def _commit(repo: Path, message: str, *, report: dict[str, bool], progress: list[str]) -> str:
    (repo / "reports").mkdir(exist_ok=True)
    (repo / "reports" / "backtest_m7.json").write_text(json.dumps(report), encoding="utf-8")
    (repo / "reports" / "m7_progress.md").write_text("\n".join(progress), encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _check(repo: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ORDER_SCRIPT)], cwd=repo, capture_output=True, text=True, check=False
    )


def _history(repo: Path, *, leak_validation_at_verdict: bool = False) -> dict[str, str]:
    """verdict (tuning only) -> validation scored -> test scored."""
    _git(repo, "init", "-q")
    verdict = _commit(
        repo,
        "verdict",
        report={"validation_scored": leak_validation_at_verdict, "test_scored": False},
        progress=["# progress"],
    )
    validation = _commit(
        repo,
        "validation",
        report={"validation_scored": True, "test_scored": False},
        progress=["# progress", f"VERDICT {verdict}"],
    )
    test = _commit(
        repo,
        "test",
        report={"validation_scored": True, "test_scored": True},
        progress=["# progress", f"VERDICT {verdict}", f"VALIDATION {validation}"],
    )
    return {"verdict": verdict, "validation": validation, "test": test}


def test_m7_order_accepts_verdict_then_validation_then_test(tmp_path: Path) -> None:
    shas = _history(tmp_path)
    _commit(
        tmp_path,
        "name the commits",
        report={"validation_scored": True, "test_scored": True},
        progress=[
            f"VERDICT {shas['verdict']}",
            f"VALIDATION {shas['validation']}",
            f"TEST {shas['test']}",
        ],
    )
    done = _check(tmp_path)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "verdict < validation < test: yes" in done.stdout


def test_m7_order_needs_the_test_commit_once_test_is_scored(tmp_path: Path) -> None:
    shas = _history(tmp_path)
    _commit(
        tmp_path,
        "forget the test commit",
        report={"validation_scored": True, "test_scored": True},
        progress=[f"VERDICT {shas['verdict']}", f"VALIDATION {shas['validation']}"],
    )
    done = _check(tmp_path)
    assert done.returncode == 1
    assert "NO" in done.stdout


def test_m7_order_rejects_a_validation_number_at_the_verdict_commit(tmp_path: Path) -> None:
    shas = _history(tmp_path, leak_validation_at_verdict=True)
    _commit(
        tmp_path,
        "name the commits",
        report={"validation_scored": True, "test_scored": True},
        progress=[
            f"VERDICT {shas['verdict']}",
            f"VALIDATION {shas['validation']}",
            f"TEST {shas['test']}",
        ],
    )
    assert _check(tmp_path).returncode == 1


def test_m7_order_does_not_need_a_test_commit_before_test_is_scored(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")
    verdict = _commit(
        tmp_path,
        "verdict",
        report={"validation_scored": False, "test_scored": False},
        progress=["# progress"],
    )
    _commit(
        tmp_path,
        "validation",
        report={"validation_scored": True, "test_scored": False},
        progress=[f"VERDICT {verdict}"],
    )
    head = _git(tmp_path, "rev-parse", "HEAD")
    _commit(
        tmp_path,
        "name validation",
        report={"validation_scored": True, "test_scored": False},
        progress=[f"VERDICT {verdict}", f"VALIDATION {head}"],
    )
    assert _check(tmp_path).returncode == 0
