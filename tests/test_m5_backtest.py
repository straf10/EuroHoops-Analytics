"""The M5 backtest harness (week 14-16 J4) on the synthetic league of ``tests/m5_synthetic.py``:
the report shape, one scored game set for every model, the gate, the tuning-only discipline,
``predict_games`` against the returned frame, the M1 replay, the tie-break, reproducibility and
the verdict-order check script."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from eurohoops.eval import m5_backtest
from eurohoops.eval.m3_backtest import m1_margins_for
from eurohoops.eval.m5_backtest import (
    Choice,
    M5Inputs,
    _blend_weights,
    _chosen_details,
    _missed_top3,
    _Option,
    _prepare,
    _projected,
    _segments,
    candidate_key,
    choose_candidate,
    format_m5_table,
    gate_block,
    predict_games,
    run_m5_backtest,
)
from tests.m5_synthetic import build_inputs, fake_player_part, small_spec

# Heavy (whole synthetic leagues per test): CI runs it on PRs, nightly and model changes.
pytestmark = pytest.mark.slow

SPEC = small_spec()
REPO = Path(__file__).parent.parent
ORDER_SCRIPT = REPO / "scripts" / "checks" / "m5_order.py"
MODELS = ("m5", "m1", "elo", "b0")
METRIC_KEYS = {
    "n",
    "log_loss",
    "brier",
    "accuracy",
    "margin_mae",
    "ece",
    "reliability",
    "margin_crps",
    "margin_rmse",
    "totals_mae",
    "totals_crps",
}
FRAME_COLUMNS = [
    "game_id",
    "season",
    "split",
    "model",
    "variant",
    "p_home",
    "exp_margin",
    "margin_sigma",
    "margin_df",
    "exp_total",
    "total_sigma",
    "actual_margin",
    "actual_total",
]


@pytest.fixture(scope="module")
def inputs() -> M5Inputs:
    return build_inputs()


@pytest.fixture(scope="module")
def validation_run(inputs: M5Inputs) -> tuple[dict[str, Any], pd.DataFrame]:
    return run_m5_backtest(inputs, spec=SPEC, player_part=fake_player_part)


@pytest.fixture(scope="module")
def full_run(inputs: M5Inputs) -> tuple[dict[str, Any], pd.DataFrame]:
    return run_m5_backtest(inputs, spec=SPEC, player_part=fake_player_part, score_test=True)


@pytest.fixture(scope="module")
def tuning_run(inputs: M5Inputs) -> tuple[dict[str, Any], pd.DataFrame]:
    return run_m5_backtest(inputs, spec=SPEC, player_part=fake_player_part, tuning_only=True)


def _choice(report: dict[str, Any]) -> Choice:
    return Choice(**report["chosen"]["choice"])


# --- Structure -----------------------------------------------------------------------------


def test_the_report_has_every_required_key(full_run: tuple[dict[str, Any], pd.DataFrame]) -> None:
    report, frame = full_run
    assert set(report) == {
        "model",
        "model_version",
        "seasons",
        "data_sha256",
        "tuning_only",
        "validation_scored",
        "test_scored",
        "grid",
        "chosen",
        "games_per_split",
        "metrics",
        "other_variants",
        "oracle",
        "gate",
        "comparisons",
        "segments",
        "gap",
    }
    assert report["model"] == "m5"
    assert report["model_version"].split("+")[1].startswith("m5.")
    assert len(report["data_sha256"]) == 64
    assert report["tuning_only"] is False
    assert report["validation_scored"] and report["test_scored"]
    assert report["seasons"] == {
        "warmup": [2016, 2017],
        "tuning": [2018, 2019],
        "validation": [2020],
        "test": [2021, 2022],
    }

    grid = report["grid"]
    assert grid["size"] == len(grid["candidates"]) == 15  # 3 shares options x (2 + 2 + 1)
    assert set(grid["best_on_edge"]) == {
        "half_life_games",
        "residual_half_life_days",
        "residual_ridge",
        "rest_ridge",
    }
    chosen = report["chosen"]
    assert set(chosen) >= {"choice", "key", "tuning_log_loss", "platt", "total", "margin_model"}
    assert set(chosen["platt"]) == {"kept", "log_loss_with", "log_loss_without", "coefficients"}
    assert set(chosen["total"]) == {"variant", "tuning_crps", "tuning_sigma"}
    assert set(chosen["margin_model"]) == {"variant", "scale", "df"}
    assert chosen["key"] in grid["candidates"] or chosen["key"] == candidate_key(_choice(report))

    for split in ("tuning", "validation", "test"):
        assert set(report["games_per_split"][split]) == {"rated", "scored", "dropped"}
        assert set(report["metrics"][split]) == set(MODELS)
        for metrics in report["metrics"][split].values():
            assert set(metrics) == METRIC_KEYS
            assert all(metrics[k] is not None for k in METRIC_KEYS - {"reliability"})
        assert set(report["segments"][split]) == {
            "pan_oly",
            "short_rest_any",
            "other_comp_prev_any",
            "early_rounds",
            "greek_after_other",
        }
        for segment in report["segments"][split].values():
            assert set(segment) == {
                "n",
                "m5_log_loss",
                "m1_log_loss",
                "m5_margin_crps",
                "m1_margin_crps",
            }
        assert set(report["oracle"][split]) == METRIC_KEYS
        assert set(report["other_variants"]["core"][split]) == METRIC_KEYS
    assert report["oracle"]["label"] == "oracle, not a forecast"
    assert set(report["other_variants"]) == {"core", "core_rest", "blend"}

    assert set(report["gap"]) == {"validation", "test"}
    for gap in report["gap"].values():
        assert set(gap) == {
            "n",
            "log_loss_projected_minus_oracle",
            "rmse_projected_minus_oracle",
            "missed_top3",
        }
        assert set(gap["missed_top3"]) == {
            "n",
            "log_loss_projected_minus_oracle",
            "rmse_projected_minus_oracle",
        }
    gate = report["gate"]
    assert gate["rule"].startswith("M5 validation log loss < M1 validation log loss")
    assert gate["variant"] == chosen["key"]
    for suffix in ("log_loss", "brier", "margin_crps", "totals_crps"):
        ci = gate[f"{suffix}_m5_minus_m1"]
        assert ci["ci95"][0] <= ci["mean"] <= ci["ci95"][1]
        assert ci["resamples"] == SPEC.bootstrap_resamples and ci["seed"] == SPEC.bootstrap_seed
        elo_ci = report["comparisons"]["validation"]["m5_minus_elo"][f"{suffix}_m5_minus_elo"]
        assert elo_ci["ci95"][0] <= elo_ci["mean"] <= elo_ci["ci95"][1]
    assert set(report["comparisons"]["test"]) == {"m5_minus_m1", "m5_minus_elo"}

    validation = report["metrics"]["validation"]
    assert gate["passed"] == (validation["m5"]["log_loss"] < validation["m1"]["log_loss"])
    assert gate["m5_log_loss"] == validation["m5"]["log_loss"]
    assert gate["m1_log_loss"] == validation["m1"]["log_loss"]

    assert list(frame.columns) == FRAME_COLUMNS
    assert set(frame["model"]) == {"m5", "m5_oracle", "m1", "elo"}
    assert set(frame["split"]) == {"tuning", "validation", "test"}


def test_the_choice_is_consistent_with_the_grid(
    full_run: tuple[dict[str, Any], pd.DataFrame],
) -> None:
    report, _ = full_run
    chosen = report["chosen"]
    losses = report["grid"]["candidates"]
    assert min(losses.values()) <= chosen["tuning_log_loss"] <= min(losses.values()) + 0.0005
    platt = chosen["platt"]
    assert platt["kept"] == (platt["log_loss_with"] < platt["log_loss_without"])
    assert platt["log_loss_without"] == chosen["tuning_log_loss"]
    assert bool(platt["coefficients"]) == platt["kept"]
    total = chosen["total"]
    crps = total["tuning_crps"]
    assert (total["variant"] == "total_m1_rest") == (crps["total_m1_rest"] < crps["total_m1"])
    assert chosen["choice"]["form"] in report["other_variants"]
    assert report["other_variants"][chosen["choice"]["form"]]["chosen"]


def test_every_model_is_scored_on_the_same_games(
    full_run: tuple[dict[str, Any], pd.DataFrame],
) -> None:
    report, frame = full_run
    for split in ("tuning", "validation", "test"):
        rows = frame[frame["split"] == split]
        ids = {model: sorted(rows[rows["model"] == model]["game_id"]) for model in MODELS[:3]}
        assert ids["m5"] == ids["m1"] == ids["elo"]
        assert len(ids["m5"]) == len(set(ids["m5"])) == report["games_per_split"][split]["scored"]
        sizes = {report["metrics"][split][model]["n"] for model in MODELS}
        assert sizes == {len(ids["m5"])}
        oracle = set(rows[rows["model"] == "m5_oracle"]["game_id"])
        assert oracle <= set(ids["m5"]) and report["oracle"][split]["n"] == len(oracle)
        actual = rows.groupby("model")["actual_margin"].apply(lambda s: s.tolist())
        assert actual["m5"] == actual["m1"] == actual["elo"]
    games = report["games_per_split"]
    assert all(g["rated"] == g["scored"] + g["dropped"] for g in games.values())
    assert games["tuning"]["rated"] == 111  # 112 games less the forfeit
    assert frame["p_home"].between(0.0, 1.0).all()


def test_the_synthetic_league_exercises_the_rest_and_segment_machinery(
    full_run: tuple[dict[str, Any], pd.DataFrame],
) -> None:
    report, _ = full_run
    segments = report["segments"]["tuning"]
    assert segments["short_rest_any"]["n"] > 0
    assert segments["other_comp_prev_any"]["n"] > 0
    assert segments["early_rounds"]["n"] > 0
    assert segments["pan_oly"]["n"] == 0  # no PAN or OLY in the synthetic league
    assert segments["greek_after_other"]["n"] == 0


def test_greek_segments_follow_the_greek_club_set(
    inputs: M5Inputs, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = _prepare(inputs, SPEC)
    mask = ctx.split["tuning"] & np.isfinite(ctx.m1.margin)
    monkeypatch.setattr(m5_backtest, "GREEK_CLUBS", frozenset({"GRE"}))
    segments = _segments(ctx, ctx.m1, mask)
    rounds = ctx.frame["round"].to_numpy()
    gre = (ctx.frame["home"] == "GRE") | (ctx.frame["away"] == "GRE")
    assert segments["pan_oly"]["n"] == int((mask & gre.to_numpy()).sum()) > 0
    # GRE follows its other-competition game by about a day in every even round
    assert (
        0
        < segments["greek_after_other"]["n"]
        <= int((mask & gre.to_numpy() & (rounds % 2 == 0)).sum())
    )
    assert segments["greek_after_other"]["n"] < segments["pan_oly"]["n"]


# --- The gate --------------------------------------------------------------------------------


def test_gate_block_uses_a_strict_point_rule() -> None:
    cis = {"log_loss_m5_minus_m1": {"mean": -0.01}}
    passing = gate_block("variant", 0.58, 0.59, cis)
    assert passing["passed"] is True
    assert passing["variant"] == "variant" and passing["log_loss_m5_minus_m1"] == {"mean": -0.01}
    assert gate_block("variant", 0.59, 0.59, cis)["passed"] is False  # a tie fails
    assert gate_block("variant", 0.60, 0.59, cis)["passed"] is False
    assert gate_block("variant", 0.590000001, 0.59, cis)["passed"] is False  # unrounded compare
    assert gate_block("variant", 0.589999999, 0.59, cis)["passed"] is True
    assert gate_block("variant", None, 0.59, cis)["passed"] is None  # no validation game
    assert "owner 2026-10-05" in passing["rule"]


# --- Discipline ------------------------------------------------------------------------------


def _mentions_a_held_out_split(node: Any, top: bool = True) -> list[str]:
    """Every key or string that names the validation or test split (the ``seasons`` block and the
    two ``*_scored`` flags aside)."""
    found = []
    if isinstance(node, dict):
        for key, value in node.items():
            if top and key in {"seasons", "validation_scored", "test_scored"}:
                continue
            if any(word in str(key) for word in ("validation", "test")):
                found.append(str(key))
            found += _mentions_a_held_out_split(value, top=False)
    elif isinstance(node, list):
        for value in node:
            found += _mentions_a_held_out_split(value, top=False)
    elif isinstance(node, str) and any(word in node for word in ("validation", "test")):
        found.append(node)
    return found


def test_tuning_only_puts_no_validation_or_test_anywhere(
    tuning_run: tuple[dict[str, Any], pd.DataFrame],
    full_run: tuple[dict[str, Any], pd.DataFrame],
) -> None:
    report, frame = tuning_run
    assert report["tuning_only"] is True
    assert not report["validation_scored"] and not report["test_scored"]
    assert _mentions_a_held_out_split(report) == []
    assert not {"gate", "comparisons", "gap"} & set(report)
    for block in ("games_per_split", "metrics", "oracle", "segments"):
        assert set(report[block]) - {"label"} == {"tuning"}
    assert set(frame["split"]) == {"tuning"}
    assert set(frame["season"]) <= set(SPEC.tuning)
    # the verdict (the choice and every tuning number) is the one the full run reports
    full, _ = full_run
    assert report["grid"] == full["grid"]
    # Platt (a, b) of a season are fitted on all earlier outcomes: the tuning-only report keeps
    # only warm-up and tuning seasons (the leak J5 found), the rest of the verdict is identical
    platt = report["chosen"]["platt"]
    full_platt = full["chosen"]["platt"]
    kept = {str(s) for s in (*SPEC.warmup, *SPEC.tuning)}
    assert platt["coefficients"] == {
        s: ab for s, ab in full_platt["coefficients"].items() if s in kept
    }
    assert {k: v for k, v in platt.items() if k != "coefficients"} == {
        k: v for k, v in full_platt.items() if k != "coefficients"
    }
    without_platt = {k: v for k, v in report["chosen"].items() if k != "platt"}
    assert without_platt == {k: v for k, v in full["chosen"].items() if k != "platt"}
    assert report["metrics"]["tuning"] == full["metrics"]["tuning"]
    assert report["oracle"]["tuning"] == full["oracle"]["tuning"]
    assert report["segments"]["tuning"] == full["segments"]["tuning"]


def test_without_score_test_there_is_no_test_number(
    validation_run: tuple[dict[str, Any], pd.DataFrame],
) -> None:
    report, frame = validation_run
    assert report["validation_scored"] and not report["test_scored"]
    assert set(report["metrics"]) == {"tuning", "validation"}
    assert set(report["games_per_split"]) == {"tuning", "validation"}
    assert set(report["gap"]) == {"validation"}
    assert set(report["comparisons"]) == {"validation"}
    assert set(frame["split"]) == {"tuning", "validation"}
    assert max(frame["season"]) == SPEC.validation[-1]
    assert not [m for m in _mentions_a_held_out_split(report) if "test" in m]


# --- predict_games ---------------------------------------------------------------------------

PREDICTION_COLUMNS = [
    "p_home",
    "exp_margin",
    "margin_sigma",
    "margin_df",
    "exp_total",
    "total_sigma",
]


def _assert_rows_reproduced(
    predictions: pd.DataFrame, frame: pd.DataFrame, model: str, inputs: M5Inputs
) -> None:
    rows = frame[frame["model"] == model].reset_index(drop=True)
    assert len(rows) > 0
    by_game = predictions.set_index("game_id")
    again = by_game.loc[rows["game_id"], PREDICTION_COLUMNS].reset_index(drop=True)
    pd.testing.assert_frame_equal(again, rows[PREDICTION_COLUMNS], check_exact=True)
    assert set(predictions["season"]) == set(range(SPEC.warmup[0], SPEC.test[-1] + 1))
    assert len(predictions) == int(inputs.games["season"].between(*_span()).sum())


def _span() -> tuple[int, int]:
    return SPEC.warmup[0], SPEC.test[-1]


def test_predict_games_reproduces_the_m5_rows(
    inputs: M5Inputs, full_run: tuple[dict[str, Any], pd.DataFrame]
) -> None:
    report, frame = full_run
    predictions = predict_games(
        inputs, spec=SPEC, player_part=fake_player_part, choice=_choice(report)
    )
    assert list(predictions.columns) == ["game_id", "season", *PREDICTION_COLUMNS]
    _assert_rows_reproduced(predictions, frame, "m5", inputs)
    tip_off = inputs.games.set_index("game_id").loc[predictions["game_id"], "tipoff_utc"]
    assert tip_off.is_monotonic_increasing  # the frame is in tip-off order


def test_predict_games_reproduces_the_oracle_rows(
    inputs: M5Inputs, full_run: tuple[dict[str, Any], pd.DataFrame]
) -> None:
    report, frame = full_run
    predictions = predict_games(
        inputs, spec=SPEC, player_part=fake_player_part, choice=_choice(report), oracle=True
    )
    _assert_rows_reproduced(predictions, frame, "m5_oracle", inputs)


def test_predict_games_follows_the_choice(inputs: M5Inputs) -> None:
    core = Choice("proj_hc", None, "core", 180.0, 40.0, None, False, "total_m1")
    other = Choice("proj_hc", None, "core", 180.0, 10.0, None, False, "total_m1")
    rest = Choice("proj_hc", None, "core_rest", 180.0, 40.0, 25.0, True, "total_m1_rest")
    frames = [
        predict_games(inputs, spec=SPEC, player_part=fake_player_part, choice=c)
        for c in (core, other, rest)
    ]
    assert not frames[0]["exp_margin"].equals(frames[1]["exp_margin"])
    assert not frames[0]["p_home"].equals(frames[2]["p_home"])
    totals = frames[2]["exp_total"].to_numpy()
    assert not np.array_equal(totals, frames[0]["exp_total"].to_numpy(), equal_nan=True)


# --- The comparison models ------------------------------------------------------------------


def test_the_m1_replay_equals_m3s_m1_margins(inputs: M5Inputs) -> None:
    ctx = _prepare(inputs, SPEC)
    expected = m1_margins_for(ctx.frame, inputs.games, inputs.team_games, inputs.tuned_m1)
    assert np.isfinite(expected).sum() > 300
    np.testing.assert_allclose(ctx.m1.margin, expected, rtol=0.0, atol=1e-12, equal_nan=True)
    assert np.isfinite(ctx.m1.total[np.isfinite(ctx.m1.margin)]).all()


def test_the_rest_total_starts_as_m1s_total(inputs: M5Inputs) -> None:
    ctx = _prepare(inputs, SPEC)
    first = ctx.season == SPEC.warmup[0]
    np.testing.assert_array_equal(ctx.total_rest[first], ctx.m1.total[first])
    later = (ctx.season > SPEC.warmup[0]) & np.isfinite(ctx.m1.total)
    assert not np.allclose(ctx.total_rest[later], ctx.m1.total[later])


def test_the_rest_features_reach_the_harness(inputs: M5Inputs) -> None:
    ctx = _prepare(inputs, SPEC)
    names = ["days_rest", "short_rest", "games_last_7d", "other_comp_prev"]
    home = pd.DataFrame(ctx.rest_home, columns=names)
    assert set(home["short_rest"]) == {0.0, 1.0}
    assert home["other_comp_prev"].sum() > 0
    assert (ctx.rest_diff == ctx.rest_home - ctx.rest_away).all()


# --- The choice rule -------------------------------------------------------------------------


def _candidate(shares: str, form: str) -> Choice:
    return Choice(
        shares,
        None if shares == "proj_hc" else 4.0,
        form,
        365.0,
        40.0,
        None if form == "core" else 100.0,
        False,
        "total_m1",
    )


def test_choose_candidate_breaks_ties_toward_the_simpler_candidate() -> None:
    tied = [
        (_candidate("proj_avail", "blend"), 0.6000),  # the lowest loss
        (_candidate("proj_decay", "core_rest"), 0.6002),
        (_candidate("proj_decay", "core"), 0.6004),  # within 0.0005 of the best: tied
        (_candidate("proj_hc", "core"), 0.6004),  # same form, simpler shares rule
        (_candidate("proj_hc", "core"), 0.6004),  # same everything: the earlier grid position
        (_candidate("proj_hc", "core"), 0.6006),  # more than 0.0005 above the best: out
    ]
    assert choose_candidate(tied, 0.0005) == 3
    assert choose_candidate(tied, 0.0) == 0  # no tolerance: the lowest loss
    assert choose_candidate(tied[:3], 0.0005) == 2  # simplest form beats simpler shares
    # outside the tolerance the lower loss wins whatever its complexity
    assert choose_candidate([(_candidate("proj_hc", "core"), 0.6006), tied[0]], 0.0005) == 1
    # shares rule decides within a form, then the lower loss (D10), not the grid position
    same_form = [
        (_candidate("proj_avail", "core"), 0.6),
        (_candidate("proj_decay", "core"), 0.6001),
        (_candidate("proj_decay", "core"), 0.6),
    ]
    assert choose_candidate(same_form, 0.0005) == 2
    # D10: within the simplest model, the lowest loss wins, not the earlier grid point
    same_model = [
        (_candidate("proj_hc", "core"), 0.6004),
        (_candidate("proj_hc", "core"), 0.6001),
        (_candidate("proj_avail", "core"), 0.6),
    ]
    assert choose_candidate(same_model, 0.0005) == 1
    # a non-finite loss never wins
    assert choose_candidate([(_candidate("proj_hc", "core"), float("nan")), tied[0]], 0.0) == 1
    with pytest.raises(ValueError, match="finite"):
        choose_candidate([(_candidate("proj_hc", "core"), float("nan"))], 0.0005)


def test_a_fixed_choice_is_scored_as_given(
    inputs: M5Inputs, tuning_run: tuple[dict[str, Any], pd.DataFrame]
) -> None:
    """J-g: the GBL scores the EuroLeague verdict. Fix a candidate the tuning did not choose,
    with the opposite Platt and totals decisions, and check it is scored exactly as given."""
    tuned, _ = tuning_run
    winner = _choice(tuned)
    other_key = next(
        k for k in tuned["grid"]["candidates"] if k != tuned["chosen"]["key"] and "|blend|" not in k
    )
    fixed = next(
        replace(c, platt=not winner.platt, total="total_m1_rest")
        for c in _all_choices()
        if candidate_key(c) == other_key
    )
    report, frame = run_m5_backtest(inputs, spec=SPEC, player_part=fake_player_part, fixed=fixed)
    assert report["chosen"]["key"] == other_key
    assert _choice(report) == fixed
    assert "EuroLeague verdict" in report["chosen"]["fixed"]
    assert "fixed" not in tuned["chosen"]  # an unfixed report keeps its format
    assert report["gate"]["gated"] is False  # J-g: the GBL comparison is reported, not gated
    expected = predict_games(inputs, spec=SPEC, player_part=fake_player_part, choice=fixed)
    m5 = frame[(frame["model"] == "m5") & (frame["split"] == "tuning")]
    by_id = expected.set_index("game_id").loc[m5["game_id"]]
    np.testing.assert_array_equal(m5["p_home"].to_numpy(), by_id["p_home"].to_numpy())
    missing = replace(fixed, residual_ridge=12345.0)
    with pytest.raises(ValueError, match="not a candidate"):
        run_m5_backtest(inputs, spec=SPEC, player_part=fake_player_part, fixed=missing)


def _all_choices() -> list[Choice]:
    """Every core/core_rest candidate of ``SPEC``'s grid (the blend's component is data-chosen)."""
    grid = SPEC.grid
    options = [("proj_hc", None)] + [
        (s, h) for s in ("proj_decay", "proj_avail") for h in grid.half_life_games
    ]
    out = []
    for shares, hl in options:
        for d in grid.residual_half_life_days:
            for r in grid.residual_ridge:
                out.append(Choice(shares, hl, "core", d, r, None, False, "total_m1"))
                out += [
                    Choice(shares, hl, "core_rest", d, r, k, False, "total_m1")
                    for k in grid.rest_ridge
                ]
    return out


def test_candidate_keys_are_readable_and_unique() -> None:
    keys = {
        candidate_key(_candidate(shares, form))
        for shares in ("proj_hc", "proj_decay", "proj_avail")
        for form in ("core", "core_rest", "blend")
    }
    assert len(keys) == 9
    assert candidate_key(_candidate("proj_decay", "core_rest")) == (
        "proj_decay@4|core_rest|d365|r40|k100"
    )


# --- Blend weights, rest coefficients, the missed-top-3 subset -------------------------------


def test_chosen_details_report_blend_weights_and_rest_coefficients(inputs: M5Inputs) -> None:
    ctx = _prepare(inputs, SPEC)
    shares = _projected(_Option("proj_hc", None), ctx.frame, inputs.player_games, SPEC)
    part = fake_player_part(ctx.frame, shares)
    blend = Choice("proj_hc", None, "blend", 180.0, 40.0, 25.0, False, "total_m1")
    details = _chosen_details(ctx, part, blend)
    assert set(details) == {"rest_coefficients", "blend_weights"}
    assert all(
        set(by_name) == {"days_rest", "short_rest", "games_last_7d", "other_comp_prev"}
        for by_name in details["rest_coefficients"].values()
    )
    assert set(details["blend_weights"]) == {"2017", "2018", "2019", "2020", "2021", "2022"}
    for weights in details["blend_weights"].values():
        assert set(weights) == {"core_rest", "m1", "elo"}
        assert all(w >= 0.0 for w in weights.values())
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-5)
    components, per_game = _blend_weights(ctx, np.zeros(len(ctx.frame)))
    assert components.shape == per_game.shape == (len(ctx.frame), 3)
    assert (
        _chosen_details(
            ctx, part, Choice("proj_hc", None, "core", 180.0, 40.0, None, False, "total_m1")
        )
        == {}
    )


def test_missed_top3_flags_games_with_a_top_player_unprojected() -> None:
    frame = pd.DataFrame({"game_id": ["g1", "g2", "g3"]})
    oracle = pd.DataFrame(
        [("g1", "home", p, s) for p, s in (("a", 0.9), ("b", 0.8), ("c", 0.7), ("d", 0.1))]
        + [("g1", "away", p, s) for p, s in (("x", 0.9), ("y", 0.8), ("z", 0.7))]
        + [("g2", "home", p, s) for p, s in (("a", 0.9), ("b", 0.8), ("c", 0.7), ("d", 0.1))]
        + [("g3", "away", p, s) for p, s in (("x", 0.9), ("y", 0.8), ("z", 0.7))],
        columns=["game_id", "side", "player_id", "share"],
    )
    projected = pd.DataFrame(
        [("g1", "home", p, 0.5) for p in "abc"]
        + [("g1", "away", "x", 0.5), ("g1", "away", "y", 0.5), ("g1", "away", "z", 0.0)]
        + [("g2", "home", p, 0.5) for p in "abc"]  # d (not a top 3 player) is missing: fine
        + [("g3", "away", "x", 0.5), ("g3", "away", "y", 0.5)],  # z has no row at all
        columns=["game_id", "side", "player_id", "share"],
    )
    assert _missed_top3(frame, projected, oracle).tolist() == [True, False, True]


# --- Reproducibility and the report writer ---------------------------------------------------


def test_two_runs_are_byte_identical(
    inputs: M5Inputs, validation_run: tuple[dict[str, Any], pd.DataFrame]
) -> None:
    report, frame = validation_run
    again, again_frame = run_m5_backtest(inputs, spec=SPEC, player_part=fake_player_part)
    assert json.dumps(report, sort_keys=True) == json.dumps(again, sort_keys=True)
    pd.testing.assert_frame_equal(frame, again_frame, check_exact=True)


def test_the_report_is_plain_json(full_run: tuple[dict[str, Any], pd.DataFrame]) -> None:
    report, _ = full_run
    assert json.loads(json.dumps(report, allow_nan=False)) == report


def test_the_frame_is_rounded_and_sorted(full_run: tuple[dict[str, Any], pd.DataFrame]) -> None:
    _, frame = full_run
    floats = frame.select_dtypes("float64")
    pd.testing.assert_frame_equal(floats, floats.round(6), check_exact=True)
    split_rank = frame["split"].map({"tuning": 0, "validation": 1, "test": 2})
    assert split_rank.is_monotonic_increasing
    for _, rows in frame.groupby(["split", "model"]):
        assert rows["game_id"].is_unique


def test_format_m5_table_runs_for_both_modes(
    full_run: tuple[dict[str, Any], pd.DataFrame],
    tuning_run: tuple[dict[str, Any], pd.DataFrame],
) -> None:
    full = format_m5_table(full_run[0])
    assert "gate (M5 vs M1" in full and ("PASS" in full or "FAIL" in full)
    assert all(name in full for name in ("m5", "m1", "elo", "b0", "oracle"))
    tuning = format_m5_table(tuning_run[0])
    assert "gate" not in tuning and "validation" not in tuning
    assert asdict(_choice(full_run[0]))["shares"] in full


# --- scripts/checks/m5_order.py ---------------------------------------------------------------


def _git(repo: Path, *args: str) -> str:
    run = subprocess.run(
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
    return run.stdout.strip()


def _commit(repo: Path, message: str, *, report: dict[str, bool], progress: list[str]) -> str:
    (repo / "reports").mkdir(exist_ok=True)
    (repo / "reports" / "backtest_m5.json").write_text(json.dumps(report), encoding="utf-8")
    (repo / "reports" / "week14-16_progress.md").write_text("\n".join(progress), encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _check(repo: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ORDER_SCRIPT)], cwd=repo, capture_output=True, text=True, check=False
    )


def _history(repo: Path, *, leak_validation_at_verdict: bool = False) -> dict[str, str]:
    """verdict (tuning only) -> validation scored -> test scored -> progress naming the test."""
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


def test_m5_order_accepts_verdict_then_validation_then_test(tmp_path: Path) -> None:
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
    run = _check(tmp_path)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "verdict < validation < test: yes" in run.stdout


def test_m5_order_needs_the_test_commit_once_test_is_scored(tmp_path: Path) -> None:
    shas = _history(tmp_path)
    _commit(
        tmp_path,
        "forget the test commit",
        report={"validation_scored": True, "test_scored": True},
        progress=[f"VERDICT {shas['verdict']}", f"VALIDATION {shas['validation']}"],
    )
    run = _check(tmp_path)
    assert run.returncode == 1
    assert "NO" in run.stdout


def test_m5_order_rejects_a_validation_number_at_the_verdict_commit(tmp_path: Path) -> None:
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


def test_m5_order_does_not_need_a_test_commit_before_test_is_scored(tmp_path: Path) -> None:
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
