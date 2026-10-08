"""The M6 backtest harness (weeks 16-18 L6) on the synthetic league of ``tests/m6_synthetic.py``:
the report shape, the choice and the gate, the tuning-only discipline, byte-stable output, the
Marcel baseline against a hand computation, the loss definition, the D11 SDs, the D12 checkpoint
and scored set, the CLI wiring on stubbed loaders and the MLflow payload."""

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import typer

from eurohoops import cli
from eurohoops.config import M3, M4
from eurohoops.eval.m3_gbl_backtest import ElSpm
from eurohoops.eval.m6_backtest import (
    BASELINES,
    GATE_BASELINES,
    PLAYER_COLUMNS,
    PLAYERS_SCHEMA,
    M6Inputs,
    TargetInputs,
    _exposure,
    baseline_predictions,
    cell_key,
    cells,
    choose_cell,
    cluster_bootstrap_ci,
    format_m6_table,
    gate_block,
    loss_sds,
    marcel_age_factor,
    model_scores,
    player_losses,
    prepare,
    run_m6_backtest,
    score_block,
    scored_splits,
    spm_noise_unit,
    target_inputs,
)
from eurohoops.eval.tracking import log_m6_backtest
from eurohoops.logs import write_json
from eurohoops.models.box_impact import STAT_COLUMNS
from eurohoops.models.player_seasons import COUNT_COLUMNS
from eurohoops.models.projection import VARIANTS
from eurohoops.models.spm import SpmModel
from tests.m6_synthetic import build_inputs, small_spec, synthetic_rounds

SPEC = small_spec()
FIXTURES = Path(__file__).parent / "fixtures" / "m6"
COMPETITION = "euroleague"
BAND = (0.75, 0.85)
SPLIT_WORDS = {"validation", "test"}
LATER_SEASONS = {*SPEC.validation, *SPEC.test}


@dataclass(frozen=True)
class Run:
    report: dict[str, Any]
    players: pd.DataFrame


def _run(inputs: M6Inputs, competition: str = COMPETITION, **kwargs: Any) -> Run:
    report, players = run_m6_backtest(inputs, spec=SPEC, competition=competition, **kwargs)
    return Run(report, players)


@pytest.fixture(scope="module")
def inputs() -> M6Inputs:
    return build_inputs()


@pytest.fixture(scope="module")
def tuning_run(inputs: M6Inputs) -> Run:
    return _run(inputs, tuning_only=True)


@pytest.fixture(scope="module")
def validation_run(inputs: M6Inputs) -> Run:
    return _run(inputs)


@pytest.fixture(scope="module")
def full_run(inputs: M6Inputs) -> Run:
    return _run(inputs, score_test=True)


@pytest.fixture(scope="module")
def gbl_run(inputs: M6Inputs) -> Run:
    # the edge of the grid and the variant that is not the EuroLeague synthetic's choice
    return _run(inputs, "gbl", score_test=True, fixed={"variant": "proj_age", "half_life": 3.0})


@pytest.fixture(scope="module")
def world(inputs: M6Inputs) -> Any:
    return prepare(inputs)


# --- End to end and the report shape --------------------------------------------------------------


def test_end_to_end_tuning_then_validation_then_test(
    tuning_run: Run, validation_run: Run, full_run: Run
) -> None:
    tuning = tuning_run.report
    assert tuning["tuning_only"] and not tuning["validation_scored"] and not tuning["test_scored"]
    assert "gate" not in tuning and "validation" not in tuning and "test" not in tuning
    middle = validation_run.report
    assert middle["validation_scored"] and not middle["test_scored"]
    assert "validation" in middle and "test" not in middle
    assert middle["gate"]["passed"] in (True, False)
    assert middle["gate"]["gated"] is True
    last = full_run.report
    assert last["test_scored"] and {"validation", "test", "gate"} <= set(last)
    # what an earlier split reports does not depend on the later ones being scored
    assert tuning["tuning"] == middle["tuning"] == last["tuning"]
    assert tuning["chosen"] == middle["chosen"] == last["chosen"]
    assert middle["validation"] == last["validation"]
    assert middle["gate"] == last["gate"]
    assert tuning["data_sha256"] == last["data_sha256"]
    assert full_run.players["split"].nunique() == 3


def test_the_gbl_scores_the_fixed_cell_and_has_no_impact_stats(gbl_run: Run) -> None:
    report = gbl_run.report
    assert report["competition"] == "gbl"
    assert report["chosen"]["variant"] == "proj_age"
    assert report["chosen"]["half_life"] == 3.0
    assert report["chosen"]["on_edge"] is True
    assert "fixed" in report["chosen"]
    assert report["model_version"] == "m6-proj_age-h3-v1"
    assert report["gate"]["gated"] is False
    assert report["stats"] == ["pts", "fg3a", "fta", "ast", "tov", "oreb", "dreb", "stl", "blk",
                               "ts", "fg3", "ft"]  # fmt: skip
    assert "spm" not in report["sd"] and "brapm" not in report["sd"]
    # the grid is still reported, and the fixed cell is the one scored after tuning
    assert len(report["tuning"]["next_season"]) == len(cells(SPEC)) + len(BASELINES)
    assert list(report["validation"]["next_season"]) == ["proj_age@3", *BASELINES]
    assert set(gbl_run.players["model"]) == {"proj_age@3", *BASELINES}


def test_a_fixed_cell_off_the_grid_is_an_error(inputs: M6Inputs) -> None:
    with pytest.raises(ValueError, match="not on the grid"):
        _run(inputs, fixed={"variant": "proj_full", "half_life": 7.0})


def test_report_keys_and_types(tuning_run: Run, full_run: Run) -> None:
    top = {
        "model", "model_version", "competition", "splits", "stats", "checkpoints", "min_poss",
        "interval", "sd", "grid", "tuning", "chosen", "data_sha256", "tuning_only",
        "validation_scored", "test_scored", "movers", "differences",
    }  # fmt: skip
    assert set(tuning_run.report) == top
    assert set(full_run.report) == top | {"validation", "test", "gate"}
    report = full_run.report
    assert report["model"] == "m6" and report["competition"] == COMPETITION
    assert report["splits"] == {"tuning": [2015, 2016], "validation": [2017], "test": [2018, 2019]}
    assert report["checkpoints"] == [0.5]
    assert set(report["grid"]) == {"objective", "variants", "half_lives", "tie_tolerance"}
    assert report["grid"]["variants"] == list(VARIANTS)
    assert report["grid"]["half_lives"] == [1.0, 2.0, 3.0]
    stats = report["stats"]
    assert len(stats) == 14 and stats[-2:] == ["spm", "brapm"]
    assert set(report["sd"]) == set(stats) and all(v > 0 for v in report["sd"].values())
    assert set(report["chosen"]) == {"variant", "half_life", "on_edge", "tuning_loss"}
    assert report["chosen"]["variant"] in VARIANTS
    assert isinstance(report["chosen"]["on_edge"], bool)
    assert len(report["data_sha256"]) == 64

    tuning = report["tuning"]
    assert set(tuning) == {"seasons", "next_season", "rest_of_season"}
    models = [cell_key(v, h) for v, h in cells(SPEC)] + list(BASELINES)
    assert list(tuning["next_season"]) == models
    for name, metrics in tuning["next_season"].items():
        assert set(metrics) == {"n", "loss", "mae", "coverage", "crps"}
        assert set(metrics["mae"]) == set(stats)
        assert set(metrics["crps"]) == {"spm", "brapm"}
        if name in BASELINES:
            assert metrics["coverage"] is None
        else:
            assert set(metrics["coverage"]) == set(stats)
    chosen = cell_key(report["chosen"]["variant"], report["chosen"]["half_life"])
    for split in ("validation", "test"):
        block = report[split]
        assert set(block) == {"seasons", "next_season", "rest_of_season"}
        assert list(block["next_season"]) == [chosen, *BASELINES]
        assert set(block["rest_of_season"]) == {"0.5"}
        for by_model in block["rest_of_season"].values():
            assert list(by_model) == [chosen, *BASELINES]
            assert by_model[chosen]["n"] > 0
    assert report["test"]["seasons"] == [2018, 2019]
    assert set(report["movers"]) == {"tuning", "validation", "test"}
    assert set(report["movers"]["tuning"]) == {chosen, *BASELINES}
    assert set(report["movers"]["tuning"][chosen]) == {"n", "loss"}
    assert set(report["differences"]["validation"]) == {f"vs_{b}" for b in BASELINES}
    diff = report["differences"]["validation"]["vs_marcel"]
    assert set(diff) == {"mean", "ci95", "clusters", "resamples", "seed"}
    assert (diff["resamples"], diff["seed"]) == (200, SPEC.bootstrap_seed)
    gate = report["gate"]
    assert set(gate) == {
        "rule", "gated", "variant", "half_life", "loss_chosen", "loss_marcel", "loss_same_as_last",
        "loss_ok", "coverage_pooled", "coverage_band", "coverage_ok", "passed", "loss_vs_marcel",
        "loss_vs_same_as_last",
    }  # fmt: skip
    assert set(gate["coverage_pooled"]) == set(stats)


def test_the_players_csv_matches_its_schema_and_holds_no_personal_column(
    full_run: Run,
) -> None:
    players = full_run.players
    assert list(players.columns) == list(PLAYER_COLUMNS)
    PLAYERS_SCHEMA.validate(players)
    assert not {"name", "birth_date", "age", "dob"} & set(players.columns)
    baselines = players[players["model"].isin(BASELINES)]
    assert baselines[["lo80", "hi80"]].isna().all().all()
    variants = players[~players["model"].isin(BASELINES)]
    assert variants[["lo80", "hi80"]].notna().all().all()
    assert (variants["lo80"] <= variants["mean"]).all()
    assert (variants["mean"] <= variants["hi80"]).all()
    assert set(players["checkpoint"]) == {0.0, 0.5}
    assert set(players.loc[players["split"] == "tuning", "season"]) == {2015, 2016}


def test_the_players_fixture_matches_the_schema() -> None:
    fixture = pd.read_csv(FIXTURES / "backtest_players.csv")
    PLAYERS_SCHEMA.validate(fixture)
    assert list(fixture.columns) == list(PLAYER_COLUMNS)


def test_the_report_numbers_match_the_csv_rows(full_run: Run) -> None:
    """The loss rebuilt from the rounded CSV rows: K / n * the present squared errors per player,
    weighted by the window's possessions (the tolerance is the CSV's 6-decimal rounding)."""
    report, players = full_run.report, full_run.players
    k = len(report["stats"])
    chosen = cell_key(report["chosen"]["variant"], report["chosen"]["half_life"])
    for split in ("tuning", "validation", "test"):
        for model in (chosen, *BASELINES):
            rows = players[
                (players["split"] == split)
                & (players["model"] == model)
                & (players["checkpoint"] == 0.0)
            ]
            grouped = rows.groupby(["season", "person_id"])
            loss = grouped["sq_err_std"].sum() * k / grouped.size()
            weight = grouped["weight"].first()
            rebuilt = float((loss * weight).sum() / weight.sum())
            block = report["tuning" if split == "tuning" else split]["next_season"][model]
            assert rebuilt == pytest.approx(block["loss"], abs=1e-4)
            assert len(loss) == block["n"]


def test_the_gate_follows_the_rule_from_the_csv(validation_run: Run) -> None:
    report, players = validation_run.report, validation_run.players
    gate = report["gate"]
    chosen = gate["variant"], gate["half_life"]
    name = cell_key(*chosen)
    assert chosen == (report["chosen"]["variant"], report["chosen"]["half_life"])
    k = len(report["stats"])

    def validation_loss(model: str) -> float:
        rows = players[
            (players["split"] == "validation")
            & (players["model"] == model)
            & (players["checkpoint"] == 0.0)
        ]
        grouped = rows.groupby("person_id")
        loss = grouped["sq_err_std"].sum() * k / grouped.size()
        weight = grouped["weight"].first()
        return float((loss * weight).sum() / weight.sum())

    losses = {m: validation_loss(m) for m in (name, *GATE_BASELINES)}
    assert gate["loss_ok"] == all(losses[name] < losses[b] for b in GATE_BASELINES)
    pooled = players[
        players["split"].isin(["tuning", "validation"])
        & (players["model"] == name)
        & (players["checkpoint"] == 0.0)
    ]
    for stat, value in gate["coverage_pooled"].items():
        rows = pooled[pooled["stat"] == stat]
        inside = ((rows["truth"] >= rows["lo80"]) & (rows["truth"] <= rows["hi80"])).mean()
        assert value == pytest.approx(inside, abs=0.01)
    in_band = all(BAND[0] <= v <= BAND[1] for v in gate["coverage_pooled"].values())
    assert gate["coverage_ok"] == in_band
    assert gate["passed"] == (gate["loss_ok"] and in_band)
    for baseline in GATE_BASELINES:
        diff = gate[f"loss_vs_{baseline}"]
        assert diff["mean"] == pytest.approx(losses[name] - losses[baseline], abs=1e-4)
        low, high = diff["ci95"]
        assert low <= diff["mean"] <= high


def test_the_format_table_prints_the_gate(validation_run: Run, tuning_run: Run) -> None:
    table = format_m6_table(validation_run.report)
    assert "gate (validation loss" in table and ("PASS" in table or "FAIL" in table)
    assert "marcel" in table and "chosen:" in table
    assert "gate" not in format_m6_table(tuning_run.report)


# --- The gate block -------------------------------------------------------------------------------


def _gate(**kwargs: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "loss_chosen": 1.0,
        "loss_marcel": 1.5,
        "loss_same_as_last": 2.0,
        "coverage": {"pts": 0.8, "ts": 0.76, "spm": 0.85},
        "band": BAND,
        "differences": {},
        "gated": True,
    }
    return gate_block("proj_full", 2.0, **{**values, **kwargs})


def test_gate_block_passes_and_fails() -> None:
    assert _gate()["passed"] is True
    # the loss must beat both baselines, strictly
    for worse in ({"loss_marcel": 0.9}, {"loss_same_as_last": 0.9}, {"loss_marcel": 1.0}):
        failed = _gate(**worse)
        assert failed["passed"] is False and failed["loss_ok"] is False
    # a pooled coverage outside the band fails the gate whatever the loss
    for stat, value in (("pts", 0.9), ("ts", 0.7), ("spm", 0.8500001), ("spm", 0.7499999)):
        failed = _gate(coverage={"pts": 0.8, "ts": 0.8, "spm": 0.8, stat: value})
        assert failed["passed"] is False and failed["coverage_ok"] is False
        assert failed["loss_ok"] is True
    # undecided without a validation loss or a coverage
    assert _gate(loss_chosen=None)["passed"] is None
    assert _gate(coverage={"pts": None})["passed"] is None
    assert _gate(gated=False)["gated"] is False
    assert _gate(loss_chosen=1.00000049)["loss_chosen"] == 1.0


# --- Choice ---------------------------------------------------------------------------------------


def _table(losses: dict[str, float]) -> list[tuple[str, float, float]]:
    return [(v, h, losses.get(cell_key(v, h), 9.0)) for v, h in cells(SPEC)]


def test_cells_are_in_simplicity_order() -> None:
    order = [cell_key(v, h) for v, h in cells(SPEC)]
    assert order[:3] == ["proj_shrunk@1", "proj_shrunk@2", "proj_shrunk@3"]
    assert order[-1] == "proj_full_spm@3" and len(order) == 12


def test_choose_cell_takes_the_lowest_loss() -> None:
    table = _table({"proj_age@3": 7.0, "proj_full@1": 7.2, "proj_shrunk@1": 8.0})
    assert table[choose_cell(table, 0.005)][:2] == ("proj_age", 3.0)


def test_choose_cell_ties_go_to_the_simpler_variant() -> None:
    # 7.97 is 0.376% below 8.0: inside the 0.5% tolerance, so proj_full (simpler) wins
    table = _table({"proj_full@2": 8.0, "proj_full_spm@2": 7.97})
    assert table[choose_cell(table, 0.005)][:2] == ("proj_full", 2.0)
    # 7.9 is 1.27% below: outside it
    table = _table({"proj_full@2": 8.0, "proj_full_spm@2": 7.9})
    assert table[choose_cell(table, 0.005)][:2] == ("proj_full_spm", 2.0)
    # the simplicity order runs shrunk < age < full across variants and half-lives
    table = _table({"proj_age@3": 7.99, "proj_shrunk@2": 8.0, "proj_full@1": 7.98})
    assert table[choose_cell(table, 0.005)][:2] == ("proj_shrunk", 2.0)


def test_choose_cell_ties_go_to_the_smaller_half_life_within_a_variant() -> None:
    table = _table({"proj_full@1": 8.0, "proj_full@2": 7.99, "proj_full@3": 7.98})
    assert table[choose_cell(table, 0.005)][:2] == ("proj_full", 1.0)
    # an exact tie too, and a zero tolerance picks the strict minimum
    table = _table({"proj_full@2": 7.0, "proj_full@3": 7.0})
    assert table[choose_cell(table, 0.0)][:2] == ("proj_full", 2.0)
    table = _table({"proj_full@1": 7.001, "proj_full@3": 7.0})
    assert table[choose_cell(table, 0.0)][:2] == ("proj_full", 3.0)


def test_choose_cell_skips_cells_without_a_loss() -> None:
    table = _table({"proj_shrunk@1": math.nan, "proj_age@1": 7.0})
    assert table[choose_cell(table, 0.005)][:2] == ("proj_age", 1.0)
    with pytest.raises(ValueError, match="finite"):
        choose_cell([("proj_age", 1.0, math.nan)], 0.005)


def test_on_edge_flags_the_grid_edges(tuning_run: Run, gbl_run: Run) -> None:
    chosen = tuning_run.report["chosen"]
    assert chosen["on_edge"] == (chosen["half_life"] in (1.0, 3.0))
    assert gbl_run.report["chosen"]["on_edge"] is True  # fixed at the grid's largest half-life


# --- Discipline -----------------------------------------------------------------------------------


def _strings_and_keys(value: Any, skip: tuple[str, ...] = ("data_sha256",)) -> list[str]:
    out: list[str] = []
    if isinstance(value, dict):
        for k, v in value.items():
            out.append(str(k))
            if k not in skip:
                out += _strings_and_keys(v)
    elif isinstance(value, list):
        for v in value:
            out += _strings_and_keys(v)
    elif isinstance(value, str):
        out.append(value)
    elif isinstance(value, int) and not isinstance(value, bool) and value in LATER_SEASONS:
        out.append(str(value))
    return out


def test_tuning_only_holds_no_validation_or_test_number(tuning_run: Run) -> None:
    report, players = tuning_run.report, tuning_run.players
    later = SPLIT_WORDS | {str(s) for s in LATER_SEASONS}
    flags = {"validation_scored", "test_scored"}  # present, and false
    found = [
        t
        for t in _strings_and_keys(report)
        if t not in flags and any(w in t.split("@")[0] for w in later)
    ]
    assert not found, found
    assert set(report["splits"]) == {"tuning"}
    assert set(report["movers"]) == set(report["differences"]) == {"tuning"}
    assert set(players["split"]) == {"tuning"}
    assert set(players["season"]) == set(SPEC.tuning)
    assert report["validation_scored"] is False and report["test_scored"] is False


def test_scored_splits() -> None:
    assert scored_splits(SPEC, True, True) == {2015: "tuning", 2016: "tuning"}
    assert scored_splits(SPEC, False, False) == {2015: "tuning", 2016: "tuning", 2017: "validation"}
    assert set(scored_splits(SPEC, False, True).values()) == {"tuning", "validation", "test"}


def test_two_runs_are_byte_identical(validation_run: Run, inputs: M6Inputs, tmp_path: Path) -> None:
    again = _run(inputs)
    for name, run in (("a", validation_run), ("b", again)):
        write_json(tmp_path / f"{name}.json", run.report)
        run.players.to_csv(tmp_path / f"{name}.csv", index=False, lineterminator="\n")
    assert (tmp_path / "a.json").read_bytes() == (tmp_path / "b.json").read_bytes()
    assert (tmp_path / "a.csv").read_bytes() == (tmp_path / "b.csv").read_bytes()


def test_playoff_games_never_enter_a_number(inputs: M6Inputs, world: Any) -> None:
    """The synthetic playoff game carries absurd lines (40 of every count in 25 minutes). Every
    number is a function of the world's regular-season lines and the games' regular-season
    rounds, so the world without the game is the same table: the same numbers."""
    bare = build_inputs(playoffs=False)
    other = prepare(bare)
    pd.testing.assert_frame_equal(other.complete, world.complete)
    games = inputs.games["euroleague"]
    playoff_ids = set(games.loc[games["phase"] == "PO", "game_id"])
    assert playoff_ids and inputs.player_games["euroleague"]["game_id"].isin(playoff_ids).any()
    assert not world.lines["euroleague"]["game_id"].isin(playoff_ids).any()
    pd.testing.assert_frame_equal(bare.brapm, inputs.brapm)  # its exposure is regular-season too
    assert not (bare.games["euroleague"]["phase"] == "PO").any()


def test_planted_aging_makes_the_aged_variants_better(tuning_run: Run) -> None:
    """Ignoring a planted decline of -0.006 (age - 26)^2 in the log rate is costly on the tuning
    seasons; projecting with the fitted curve is not."""
    grid = tuning_run.report["tuning"]["next_season"]
    assert grid["proj_age@2"]["loss"] < grid["proj_shrunk@2"]["loss"]
    assert grid["proj_full@2"]["loss"] < grid["marcel"]["loss"] < grid["league_mean"]["loss"]


def test_cluster_bootstrap_resamples_whole_clusters() -> None:
    # two clusters with weighted means 1 and 3: a resample's mean is 1, 2 or 3
    numerator = np.array([1.0, 1.0, 3.0, 3.0])
    denominator = np.ones(4)
    cluster = np.array([0, 0, 1, 1])
    mean, low, high = cluster_bootstrap_ci(numerator, denominator, cluster, 1000, 7)
    assert mean == 2.0 and low == 1.0 and high == 3.0
    again = cluster_bootstrap_ci(numerator, denominator, cluster, 1000, 7)
    assert again == (mean, low, high)
    empty = cluster_bootstrap_ci(np.zeros(0), np.zeros(0), np.zeros(0, dtype=np.int64), 10, 1)
    assert all(math.isnan(v) for v in empty)


# --- The loss, by hand ----------------------------------------------------------------------------


def _truth_frame() -> TargetInputs:
    """Two players; X has no free-throw attempt (ft truth NaN). Only ``score_block`` reads it."""
    truth = pd.DataFrame(
        {
            "person_id": ["A", "X"],
            "poss": [1000.0, 500.0],
            "mover": [False, True],
            "pts": [24.0, 20.0],
            "ft": [0.8, math.nan],
        }
    )
    return TargetInputs(
        competition=COMPETITION, season=2020, checkpoint=0.0, cutoff=None, stats=("pts", "ft"),
        history=pd.DataFrame(), ages=pd.DataFrame(), impact=None, curve=None,  # type: ignore[arg-type]
        drift={}, translation=None, targets=pd.DataFrame(), truth=truth,
    )  # fmt: skip


def _prediction(means: dict[tuple[str, str], float], width: float | None) -> pd.DataFrame:
    rows = [
        {
            "person_id": p,
            "stat": s,
            "mean": m,
            "lo80": np.nan if width is None else m - width,
            "hi80": np.nan if width is None else m + width,
            "sd": np.nan,
        }
        for (p, s), m in means.items()
    ]
    return pd.DataFrame(rows)


def test_the_loss_renormalises_over_the_stats_present() -> None:
    sds = {"pts": 2.0, "ft": 0.1}
    model = _prediction(
        {("A", "pts"): 22.0, ("A", "ft"): 0.7, ("X", "pts"): 23.0, ("X", "ft"): 0.5}, 1.5
    )
    point = _prediction(
        {("A", "pts"): 20.0, ("A", "ft"): 0.8, ("X", "pts"): 20.0, ("X", "ft"): 0.8}, None
    )
    scored = score_block(_truth_frame(), {"model": model, "point": point}, sds)
    block = {"season": 2020, "checkpoint": 0.0}  # what the run attaches to a block's rows
    rows = scored["model"].assign(**block)
    # X's ft truth is NaN: its row is dropped for every model
    assert list(zip(rows["person_id"], rows["stat"], strict=True)) == [
        ("A", "pts"), ("A", "ft"), ("X", "pts"),
    ]  # fmt: skip
    a = ((24 - 22) / 2) ** 2 + ((0.8 - 0.7) / 0.1) ** 2  # 1 + 1 = 2, both stats present: K / n = 1
    x = ((20 - 23) / 2) ** 2 * 2 / 1  # one of K = 2 stats present: the weight doubles
    losses = player_losses(rows)
    assert losses["loss"].to_numpy() == pytest.approx([a, x])
    expected = (1000 * a + 500 * x) / 1500
    metrics = model_scores(rows, ["pts", "ft"])
    assert metrics["loss"] == pytest.approx(expected)
    assert metrics["n"] == 2
    # per-stat MAE, weighted by the possessions of the players present
    assert metrics["mae"]["pts"] == pytest.approx((1000 * 2 + 500 * 3) / 1500)
    assert metrics["mae"]["ft"] == pytest.approx(0.1)
    # coverage: A's pts miss their interval, his ft sit inside theirs, X's pts miss too
    assert metrics["coverage"] == {"pts": 0.0, "ft": 1.0}
    # a point-only model has no coverage and the same stat rows
    baseline = model_scores(scored["point"].assign(**block), ["pts", "ft"])
    assert baseline["coverage"] is None
    # A: pts (24 - 20) / 2 squared = 4 and ft 0, K / n = 1; X: pts 0
    assert baseline["loss"] == pytest.approx((1000 * 4.0 + 500 * 0.0) / 1500)


# --- Marcel, by hand ------------------------------------------------------------------------------


def _history_row(
    person: str, season: int, poss: float, *, pts: int, fta: int, ftm: int
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "person_id": person,
        "competition": COMPETITION,
        "season": season,
        "partial": False,
        "mapped": True,
        "team": "E01",
        "games": 20,
        "minutes": poss / 2.0,
        "poss": poss,
        "debut_season": 2010,
    }
    row.update(dict.fromkeys(COUNT_COLUMNS, 0))
    row.update({"pts": pts, "fta": fta, "ftm": ftm})
    return row


def _marcel_inputs() -> TargetInputs:
    """Target season 2020. A: three seasons (a age in 2020: 25); B: four rows with 2016 and 2018
    missing (age 31); C: one short season, no age. The league is the 2019 season (A, B, C)."""
    rows = [
        _history_row("A", 2017, 1000, pts=200, fta=100, ftm=70),
        _history_row("A", 2018, 2000, pts=500, fta=200, ftm=160),
        _history_row("A", 2019, 1000, pts=300, fta=50, ftm=40),
        _history_row("B", 2014, 1000, pts=100, fta=30, ftm=15),
        _history_row("B", 2015, 800, pts=120, fta=40, ftm=20),
        _history_row("B", 2017, 500, pts=100, fta=40, ftm=20),
        _history_row("B", 2019, 1500, pts=450, fta=100, ftm=75),
        _history_row("C", 2019, 100, pts=40, fta=10, ftm=5),
    ]
    ages = pd.DataFrame({"person_id": ["A", "B"], "season": [2019, 2017], "age": [24.0, 28.0]})
    targets = pd.DataFrame(
        {
            "person_id": ["A", "B", "C"],
            "competition": COMPETITION,
            "season": 2020,
            "checkpoint": 0.0,
            "exposure": 1000.0,
        }
    )
    return TargetInputs(
        competition=COMPETITION, season=2020, checkpoint=0.0, cutoff=None, stats=("pts", "ft"),
        history=pd.DataFrame(rows), ages=ages, impact=None, curve=None,  # type: ignore[arg-type]
        drift={}, translation=None, targets=targets, truth=pd.DataFrame(),
    )  # fmt: skip


def test_marcel_matches_a_hand_computation() -> None:
    predictions = baseline_predictions(_marcel_inputs())
    marcel = predictions["marcel"].set_index(["person_id", "stat"])["mean"]
    # league 2019: 790 points in 2600 possessions, 120 made of 160 free throws
    league_pts = 100 * 790 / 2600
    league_ft, ft_per_poss = 120 / 160, 160 / 2600
    young, old = 1 + 0.006 * (29 - 25), 1 + 0.003 * (29 - 31)
    # counts: weights 5/4/3 x possessions, 1,200 possessions of the league mean
    a = (5 * 1000 * 30 + 4 * 2000 * 25 + 3 * 1000 * 20 + 1200 * league_pts) / (16000 + 1200)
    assert marcel[("A", "pts")] == pytest.approx(a * young)
    # B: 2019, 2017 and 2015 (the three most recent usable seasons: the gaps shift the weights)
    b = (5 * 1500 * 30 + 4 * 500 * 20 + 3 * 800 * 15 + 1200 * league_pts) / (11900 + 1200)
    assert marcel[("B", "pts")] == pytest.approx(b * old)
    # C: one season of 100 possessions, no age: no adjustment
    c = (5 * 100 * 40 + 1200 * league_pts) / (500 + 1200)
    assert marcel[("C", "pts")] == pytest.approx(c)
    # a percentage: attempts x weight, regressed by 1,200 x the league's attempts per possession
    pad = 1200 * ft_per_poss
    a_ft = (5 * 50 * 0.8 + 4 * 200 * 0.8 + 3 * 100 * 0.7 + pad * league_ft) / (1350 + pad)
    assert marcel[("A", "ft")] == pytest.approx(a_ft * young)
    b_ft = (5 * 100 * 0.75 + 4 * 40 * 0.5 + 3 * 40 * 0.5 + pad * league_ft) / (
        5 * 100 + 4 * 40 + 3 * 40 + pad
    )
    assert marcel[("B", "ft")] == pytest.approx(b_ft * old)
    c_ft = (5 * 10 * 0.5 + pad * league_ft) / (50 + pad)
    assert marcel[("C", "ft")] == pytest.approx(c_ft)


def test_the_naive_baselines_by_hand() -> None:
    predictions = baseline_predictions(_marcel_inputs())
    last = predictions["same_as_last"].set_index(["person_id", "stat"])["mean"]
    assert (last[("A", "pts")], last[("B", "pts")], last[("C", "pts")]) == (30.0, 30.0, 40.0)
    assert last[("A", "ft")] == 0.8 and last[("C", "ft")] == 0.5
    league = predictions["league_mean"].set_index(["person_id", "stat"])["mean"]
    assert league[("A", "pts")] == league[("C", "pts")] == pytest.approx(100 * 790 / 2600)
    assert league[("B", "ft")] == pytest.approx(0.75)
    for frame in predictions.values():
        assert frame[["lo80", "hi80", "sd"]].isna().all().all()


def test_marcel_age_factor() -> None:
    factor = marcel_age_factor(np.array([25.0, 29.0, 33.0, np.nan]))
    assert factor == pytest.approx([1.024, 1.0, 0.988, 1.0])


def test_marcel_without_a_usable_season_is_the_league_mean() -> None:
    ti = _marcel_inputs()
    ti = replace(ti, targets=pd.concat([ti.targets, ti.targets.iloc[:1].assign(person_id="Z")]))
    predictions = baseline_predictions(ti)
    marcel = predictions["marcel"].set_index(["person_id", "stat"])["mean"]
    last = predictions["same_as_last"].set_index(["person_id", "stat"])["mean"]
    assert marcel[("Z", "pts")] == pytest.approx(100 * 790 / 2600)
    assert last[("Z", "ft")] == pytest.approx(0.75)


# --- D11, D12, D5 ---------------------------------------------------------------------------------


def test_the_loss_sds_read_only_the_seasons_before_tuning(inputs: M6Inputs) -> None:
    base = loss_sds(prepare(inputs), SPEC, COMPETITION)
    first = min(SPEC.tuning)
    # a hand computation of one stat from the raw lines of the seasons before tuning
    lines = inputs.player_games[COMPETITION].merge(
        inputs.games[COMPETITION].query("phase == 'RS'")[["game_id", "tipoff_utc"]], on="game_id"
    )
    lines = lines[(lines["sec"] > 0) & (lines["season"] < first)]
    seasons = lines.groupby(["player_id", "season"])[["pts", "poss"]].sum()
    seasons = seasons[seasons["poss"] >= SPEC.min_poss]
    rate = 100 * seasons["pts"] / seasons["poss"]
    mean = (rate * seasons["poss"]).sum() / seasons["poss"].sum()
    sd = math.sqrt(((rate - mean) ** 2 * seasons["poss"]).sum() / seasons["poss"].sum())
    assert base["pts"] == pytest.approx(sd)

    # plant a wild value in every season from the first tuning one on: nothing changes
    wild = {
        comp: frame.assign(
            pts=np.where(frame["season"] >= first, frame["pts"] * 100 + 500, frame["pts"]),
            ast=np.where(frame["season"] >= first, 0, frame["ast"]),
        )
        for comp, frame in inputs.player_games.items()
    }
    brapm = inputs.brapm.assign(
        value=np.where(inputs.brapm["season"] >= first, 1e6, inputs.brapm["value"])
    )
    planted = replace(inputs, player_games=wild, brapm=brapm)
    assert loss_sds(prepare(planted), SPEC, COMPETITION) == base
    # and the seasons before it do count
    early = {
        comp: frame.assign(pts=np.where(frame["season"] < first, frame["pts"] * 3, frame["pts"]))
        for comp, frame in inputs.player_games.items()
    }
    assert (
        loss_sds(prepare(replace(inputs, player_games=early)), SPEC, COMPETITION)["pts"]
        > base["pts"]
    )


def test_the_exposure_rules_by_hand() -> None:
    def row(person: str, season: int, poss: float, partial: bool = False) -> dict[str, Any]:
        return {"person_id": person, "season": season, "poss": poss, "partial": partial}

    history = pd.DataFrame(
        [
            row("P1", 2018, 1000),
            row("P1", 2019, 800),
            row("P1", 2020, 300, True),
            row("P2", 2019, 600),
        ]
    )
    people = pd.Series(["P1", "P2"])
    # next-season: the newest season before 2020 (the partial 2020 row is not usable there)
    assert _exposure(history, people, 2020, 0.0) == pytest.approx([800.0, 600.0])
    # a checkpoint at f = 0.25: 300 played before the cutoff stand for 300 * 0.75 / 0.25 after it;
    # P2 has not played yet: his newest season scaled by the share left
    assert _exposure(history, people, 2020, 0.25) == pytest.approx([900.0, 450.0])
    assert _exposure(history, people, 2020, 0.75) == pytest.approx([100.0, 150.0])


def _independent_lines(inputs: M6Inputs, competition: str, season: int) -> pd.DataFrame:
    lines = inputs.player_games[competition]
    games = inputs.games[competition]
    keep = games[games["phase"] == "RS"][["game_id", "round", "tipoff_utc"]]
    lines = lines.merge(keep, on="game_id")
    person = inputs.xwalk[inputs.xwalk["competition"] == competition].set_index("source_id")
    lines["person_id"] = lines["player_id"].map(person["person_id"])
    return lines[(lines["season"] == season) & (lines["sec"] > 0)]


def test_a_checkpoint_cuts_the_season_at_the_first_tipoff_of_the_next_round(
    inputs: M6Inputs, world: Any
) -> None:
    season, fraction = 2016, 0.5
    ti = target_inputs(inputs, COMPETITION, season, fraction, spec=SPEC, world=world)
    done = math.floor(fraction * synthetic_rounds(COMPETITION, season))  # k = 14 of R = 28
    games = inputs.games[COMPETITION]
    rs = games[(games["season"] == season) & (games["phase"] == "RS")]
    cutoff = rs.loc[rs["round"] == done + 1, "tipoff_utc"].min()
    assert done == 14 and ti.cutoff == cutoff

    lines = _independent_lines(inputs, COMPETITION, season)
    before = lines[lines["tipoff_utc"] < cutoff].groupby("person_id")["poss"].sum()
    after = lines[lines["tipoff_utc"] >= cutoff].groupby("person_id")["poss"].sum()
    # the history: every earlier season, and this one only as the partial row of the games before
    assert ti.history["season"].max() == season
    current = ti.history[ti.history["season"] == season]
    assert (
        current["partial"].all()
        and (ti.history["partial"] == (ti.history["season"] == season)).all()
    )
    assert set(current["competition"]) == {COMPETITION}
    partial = current.set_index("person_id")["poss"]
    assert partial.to_dict() == pytest.approx(before.to_dict())

    # the scored set: at least 500 * (1 - f) possessions after the cutoff and a season before
    earlier = [
        _independent_lines(inputs, c, s) for c in ("euroleague", "gbl") for s in range(2010, season)
    ]
    prior = set(pd.concat(earlier)["person_id"])
    expected = {p for p, poss in after.items() if poss >= 500 * (1 - fraction) and p in prior}
    assert set(ti.targets["person_id"]) == expected and expected
    assert ti.truth.set_index("person_id")["poss"].to_dict() == pytest.approx(
        {p: after[p] for p in expected}
    )
    # the reference exposure: his possessions before the cutoff scaled by (1 - f) / f
    exposure = ti.targets.set_index("person_id")["exposure"]
    for person in list(expected)[:20]:
        if person in before:
            assert exposure[person] == pytest.approx(before[person] * (1 - fraction) / fraction)
    # no BRAPM at a cutoff (no snapshot exists) and no impact row of this season
    assert "brapm" not in ti.stats and "brapm" not in ti.truth
    assert ti.impact is not None
    assert ((ti.impact["season"] < season) | (ti.impact["stat"] == "spm")).all()
    assert ti.impact[ti.impact["stat"] == "brapm"]["season"].max() < season


def test_r_is_the_last_played_round_and_a_short_season_moves_the_cutoff(
    inputs: M6Inputs, world: Any
) -> None:
    """Without ``rounds``, R is read from the games; an interrupted season has fewer rounds
    (EuroLeague 2019-20 stopped at round 28 of 34)."""
    season = 2016
    full = target_inputs(inputs, COMPETITION, season, 0.75, spec=SPEC, world=world)
    assert (
        full.cutoff
        == target_inputs(
            inputs, COMPETITION, season, 0.75, spec=SPEC, world=world, rounds=synthetic_rounds
        ).cutoff
    )
    games = inputs.games[COMPETITION]
    cut = games.assign(
        played=games["played"] & ~((games["season"] == season) & (games["round"] > 20))
    )
    short = target_inputs(
        replace(inputs, games={**inputs.games, COMPETITION: cut}),
        COMPETITION, season, 0.75, spec=SPEC, world=world,
    )  # fmt: skip
    rounds = games[(games["season"] == season) & (games["phase"] == "RS")]
    assert (
        short.cutoff == rounds.loc[rounds["round"] == math.floor(0.75 * 20) + 1, "tipoff_utc"].min()
    )
    assert short.cutoff < full.cutoff


def test_a_next_season_target_reads_nothing_of_its_own_season(inputs: M6Inputs, world: Any) -> None:
    season = 2016
    ti = target_inputs(inputs, COMPETITION, season, 0.0, spec=SPEC, world=world)
    assert ti.cutoff is None
    assert ti.history["season"].max() == season - 1 and not ti.history["partial"].any()
    assert ti.impact is not None and ti.impact["season"].max() == season - 1
    assert set(ti.stats) == set(ti.truth.columns) - {"person_id", "poss", "mover"}
    assert "brapm" in ti.stats
    lines = _independent_lines(inputs, COMPETITION, season)
    total = lines.groupby("person_id")["poss"].sum()
    truth = ti.truth.set_index("person_id")["poss"]
    assert (truth >= SPEC.min_poss).all()
    assert truth.to_dict() == pytest.approx({p: total[p] for p in truth.index})
    assert set(ti.targets["person_id"]) <= set(total[total >= SPEC.min_poss].index)
    assert ti.truth["mover"].any()  # the synthetic league plants movers
    assert ti.translation is not None and ti.translation.target_season == season


def test_a_mover_without_a_translation_is_an_error(inputs: M6Inputs, world: Any) -> None:
    bare = replace(inputs, translations={})
    with pytest.raises(ValueError, match="no M4 translation"):
        target_inputs(bare, COMPETITION, 2016, 0.0, spec=SPEC, world=world)


# --- The CLI wiring on stubbed loaders and MLflow -------------------------------------------------


def _el_spm() -> ElSpm:
    rng = np.random.default_rng(3)
    models = {
        season: SpmModel(
            season=season,
            stats=STAT_COLUMNS,
            mean=np.zeros(len(STAT_COLUMNS)),
            std=np.ones(len(STAT_COLUMNS)),
            o_intercept=0.0,
            o_coef=rng.normal(0.0, 0.05, len(STAT_COLUMNS)),
            d_intercept=0.0,
            d_coef=rng.normal(0.0, 0.05, len(STAT_COLUMNS)),
            alpha=1.0,
            n_players=100,
        )
        for season in range(2010, 2021)
    }
    fit_time = {season: float(season) for season in models}
    return ElSpm(models=models, fit_time=fit_time, k=250.0, half_life_days=365.0)


def _stub_loaders(monkeypatch: pytest.MonkeyPatch, inputs: M6Inputs, tmp_path: Path) -> None:
    """Point every loader of ``cli._m6_inputs`` at the synthetic world."""
    monkeypatch.chdir(tmp_path)
    games = inputs.games
    monkeypatch.setattr(cli, "read_games", lambda _path, name: games[name])
    tables = {
        "team_games": pd.DataFrame({"game_id": games["gbl"]["game_id"]}),
        "player_xwalk": inputs.xwalk,
        "stints": pd.DataFrame(),
        "stint_game_checks": pd.DataFrame(),
    }
    monkeypatch.setattr(cli, "read_table", lambda _path, name, _comp=None: tables[name])

    class Box:
        def __init__(self, table: pd.DataFrame) -> None:
            self.players = self.table = table

    monkeypatch.setattr(
        cli, "build_box_games", lambda _raw, _games: Box(inputs.player_games["euroleague"])
    )
    monkeypatch.setattr(
        cli, "build_gbl_player_games", lambda _raw, _games, _tg: Box(inputs.player_games["gbl"])
    )
    # birth dates reproducing the synthetic ages
    ages = inputs.ages.drop_duplicates("person_id")
    born = {
        r.person_id: pd.Timestamp(f"{r.season}-10-01") - pd.Timedelta(days=r.age * 365.2425)
        for r in ages.itertuples()
    }
    bios = inputs.xwalk.assign(birth_date=inputs.xwalk["person_id"].map(born))
    monkeypatch.setattr(cli, "_m6_bios", lambda: bios[["competition", "source_id", "birth_date"]])
    monkeypatch.setattr(cli, "el_spm_models", lambda *_args: _el_spm())
    # the committed reports the wiring reads
    snaps = inputs.brapm.assign(
        player_id=lambda d: d["person_id"].str.removeprefix("P:"), total=lambda d: d["value"]
    )
    players_json = {
        "seasons": {
            str(season): {
                "players": [
                    {"player_id": r.player_id, "total": r.total, "sd_total": r.sd, "seen": True}
                    for r in rows.itertuples()
                ]
            }
            for season, rows in snaps.groupby("season")
        }
    }
    (tmp_path / "m3.json").write_text(
        json.dumps({"chosen": {"variant": "rapm_spm"}}), encoding="utf-8"
    )
    (tmp_path / "m3_players.json").write_text(json.dumps(players_json), encoding="utf-8")
    fits = {
        str(season): {"translate": {s: {"delta": t.delta[s], "c": t.c[s]} for s in t.delta}}
        for season, t in inputs.translations.items()
    }
    (tmp_path / "m4.json").write_text(json.dumps({"fits_by_target_season": fits}), encoding="utf-8")
    monkeypatch.setattr(cli, "M3", replace(M3, report=tmp_path / "m3.json"))
    monkeypatch.setattr(cli, "M3_PLAYERS_REPORT", tmp_path / "m3_players.json")
    monkeypatch.setattr(cli, "M4", replace(M4, translation_report=tmp_path / "m4.json"))
    spec = small_spec()
    monkeypatch.setattr(
        cli, "M6", replace(spec, report=tmp_path / "r.json", players_report=tmp_path / "p.csv")
    )
    monkeypatch.setattr(
        cli,
        "M6_GBL",
        replace(spec, report=tmp_path / "rg.json", players_report=tmp_path / "pg.csv"),
    )
    monkeypatch.setattr(cli, "log_m6_backtest", lambda *_args: None)


def test_the_cli_wires_the_marts_into_a_run_and_gbl_scores_the_euroleague_verdict(
    inputs: M6Inputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_loaders(monkeypatch, inputs, tmp_path)
    # the GBL refuses a tuning-only run and a run without the EuroLeague verdict
    with pytest.raises(typer.Exit):
        cli._backtest_m6(cli.CompetitionName.gbl, False, False, "unused")
    cli._backtest_m6(cli.CompetitionName.euroleague, False, True, "unused")
    report = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert report["tuning_only"] and report["competition"] == "euroleague"
    assert report["stats"][-2:] == ["spm", "brapm"]
    assert list(pd.read_csv(tmp_path / "p.csv").columns) == list(PLAYER_COLUMNS)
    # the GBL run scores the committed EuroLeague verdict as its fixed cell
    seen: dict[str, Any] = {}

    class Stop(Exception):
        pass

    def capture(_inputs: M6Inputs, **kwargs: Any) -> None:
        seen.update(kwargs)
        raise Stop

    monkeypatch.setattr(cli, "run_m6_backtest", capture)
    with pytest.raises(Stop):
        cli._backtest_m6(cli.CompetitionName.gbl, False, False, "unused")
    assert seen["competition"] == "gbl" and seen["tuning_only"] is False
    assert seen["fixed"] == {
        "variant": report["chosen"]["variant"],
        "half_life": report["chosen"]["half_life"],
    }


def test_the_wired_spm_is_the_models_o_plus_d_on_lagged_league_rates() -> None:
    el = _el_spm()
    rates = pd.DataFrame(
        {s: [10.0, 20.0] for s in STAT_COLUMNS}
        | {"season": [2012, 2013], "competition": "euroleague"}
    )
    lines = {
        "euroleague": pd.DataFrame(
            {
                "season": [2011, 2012, 2012],
                **{s: [4, 6, 6] for s in STAT_COLUMNS},
                "poss": [100.0, 100.0, 100.0],
                "sec": [60, 60, 60],
            }
        )
    }
    league = cli._league_rates(lines)
    assert league.loc[("euroleague", 2011), "pts"] == pytest.approx(4.0)
    assert league.loc[("euroleague", 2012), "pts"] == pytest.approx(6.0)  # 100 * 12 / 200
    frame = cli._m6_spm(el, league)(rates, 2013)
    model = el.models[2013]
    # 2012 is lagged to 2011's league rate (4), 2013 to 2012's (6)
    features = np.array([[10.0 - 4.0] * len(STAT_COLUMNS), [20.0 - 6.0] * len(STAT_COLUMNS)])
    o, d = model.predict(features)
    assert frame.to_numpy() == pytest.approx(o + d)


def test_log_m6_backtest_logs_no_later_split_when_tuning_only(
    tuning_run: Run, tmp_path: Path
) -> None:
    mlflow = pytest.importorskip("mlflow")
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    run_id = log_m6_backtest(tuning_run.report, COMPETITION, uri)
    assert run_id is not None
    run = mlflow.get_run(run_id)
    names = {*run.data.params, *run.data.metrics, *run.data.tags}
    assert run.data.params["chosen.variant"] == tuning_run.report["chosen"]["variant"]
    assert run.data.params["seasons.tuning"] == "2015,2016"
    assert run.data.params["test_scored"] == "False"
    assert not any("validation" in n or "seasons.test" in n for n in names), names
    assert any(n.startswith("tuning.next_season.") for n in names)
    assert not any(n.startswith("splits") for n in names)


# --- D14: the impact truths are noisy measurements --------------------------------------------


def test_the_chosen_cells_impact_intervals_cover_near_the_nominal_rate(
    validation_run: Run,
) -> None:
    """Tuning + validation rows of the chosen cell: coverage of the SPM and BRAPM truths within
    3 binomial SEs of 0.8 for the number of scored players n (SE = sqrt(0.8 * 0.2 / n); a few
    hundred player-targets per stat here, so +-0.06 to +-0.09). Without the D14 noise
    term the interval is that of the true impact and misses the noisy truth far more often."""
    rows = validation_run.players
    chosen = rows[rows["model"].str.startswith("proj_")]
    for stat in ("spm", "brapm"):
        mine = chosen[chosen["stat"] == stat]
        n = len(mine)
        assert n > 100
        covered = float(((mine["truth"] >= mine["lo80"]) & (mine["truth"] <= mine["hi80"])).mean())
        assert abs(covered - 0.8) <= 3 * math.sqrt(0.8 * 0.2 / n), (stat, n, covered)


def test_the_impact_noise_is_walk_forward_and_by_hand(inputs: M6Inputs, world: Any) -> None:
    season = 2016
    ti = target_inputs(inputs, COMPETITION, season, 0.0, spec=SPEC, world=world)
    snaps = inputs.brapm[
        (inputs.brapm["stat"] == "brapm") & (inputs.brapm["season"] < season)
    ].merge(
        ti.history[["person_id", "competition", "season", "poss"]],
        on=["person_id", "competition", "season"],
    )
    by_hand = float((snaps["poss"] * snaps["sd"] ** 2).sum() / snaps["poss"].sum())
    assert ti.impact_noise is not None
    assert ti.impact_noise["brapm"] == (pytest.approx(by_hand, rel=1e-12), 0.0)
    assert ti.impact_noise["spm"] == (0.0, spm_noise_unit(world, season))
    # a wild sd in the target season (and after) changes nothing
    wild = inputs.brapm.copy()
    wild.loc[wild["season"] >= season, "sd"] = 1e6
    planted = replace(inputs, brapm=wild)
    again = target_inputs(planted, COMPETITION, season, 0.0, spec=SPEC, world=prepare(planted))
    assert again.impact_noise == ti.impact_noise
    # no snapshot before the target: no BRAPM entry, SPM keeps its own
    bare = replace(inputs, brapm=inputs.brapm[inputs.brapm["season"] >= season])
    none = target_inputs(bare, COMPETITION, season, 0.0, spec=SPEC, world=prepare(bare))
    assert none.impact_noise == {"spm": ti.impact_noise["spm"]}
    # the GBL has no impact stats, so no noise
    gbl = target_inputs(inputs, "gbl", SPEC.validation[0], 0.0, spec=SPEC, world=world)
    assert gbl.impact_noise is None
