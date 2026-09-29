"""M3 GBL SPM transfer (week 9-12 H8): leakage, determinism, and report shape."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from eurohoops import cli
from eurohoops.config import M3Backtest, M3Grid
from eurohoops.eval.m3_backtest import BaselineResult
from eurohoops.eval.m3_gbl_backtest import (
    ElSpm,
    choose_model,
    el_spm_models,
    run_m3_gbl_backtest,
    spm_transfer_diff,
)
from eurohoops.models import box_impact as bi
from eurohoops.models.minutes import projected_shares
from eurohoops.models.spm import SpmModel, fit_season_spm
from eurohoops.models.team_eff import DecayParams
from tests.m3_synthetic import SyntheticM3, make_synthetic_m3

runner = CliRunner()

GBL_SPEC = M3Backtest(
    report=None,  # type: ignore[arg-type]
    warmup=(2018, 2019),
    tuning=(2020, 2021),
    validation=(2022,),
    test=(2023, 2024),
    grid=M3Grid(half_life_days=(365.0,), ridge=(500.0,)),
)

CHOSEN: dict[str, Any] = {
    "variant": "rapm_spm",
    "target_half_life_days": 365.0,
    "target_ridge_o": 500.0,
    "target_ridge_d": 500.0,
    "k": 250.0,
    "alpha": 1.0,
}

TUNED_M1: dict[str, Any] = {
    "rating": asdict(DecayParams(half_life_days=365.0, carry=1.0, ridge=500.0)),
    "pace": asdict(DecayParams(half_life_days=365.0, carry=1.0, ridge=2.0)),
    "margin": {"variant": "normal_const", "scale": 10.0, "df": None, "ref_pace": None},
    "totals_sigma": 12.0,
}


def _zero_baseline(
    games: pd.DataFrame,
    player_games: pd.DataFrame,
    shares: pd.DataFrame,
    possessions: pd.Series,
    tuning: np.ndarray,
) -> BaselineResult:
    return BaselineResult(margin=np.zeros(len(games)), params={"kind": "zero"})


def _pir_baseline(
    games: pd.DataFrame,
    player_games: pd.DataFrame,
    shares: pd.DataFrame,
    possessions: pd.Series,
    tuning: np.ndarray,
) -> BaselineResult:
    return BaselineResult(margin=np.full(len(games), 0.5), params={"kind": "pir"})


@pytest.fixture(scope="module")
def el_data() -> SyntheticM3:
    return make_synthetic_m3(
        list(range(2011, 2026)),
        teams=4,
        games_per_season=4,
        stints_per_game=4,
        roster=5,
        seed=11,
    )


@pytest.fixture(scope="module")
def gbl_data() -> SyntheticM3:
    return make_synthetic_m3(
        list(range(2018, 2025)),
        teams=4,
        games_per_season=10,
        stints_per_game=6,
        roster=6,
        seed=22,
    )


@pytest.fixture(scope="module")
def el_spm(el_data: SyntheticM3) -> ElSpm:
    return el_spm_models(
        el_data.games,
        el_data.player_games,
        el_data.stints,
        el_data.checks,
        CHOSEN,
    )


def _run_gbl(gbl: SyntheticM3, el: ElSpm, *, score_test: bool = False) -> dict[str, Any]:
    return run_m3_gbl_backtest(
        gbl.games,
        gbl.team_games,
        gbl.player_games,
        spec=GBL_SPEC,
        el=el,
        tuned_m1=TUNED_M1,
        box_only_fn=_zero_baseline,
        pir_fn=_pir_baseline,
        box_pages_skipped=0,
        score_test=score_test,
    )


def test_choose_model_never_uses_an_el_model_fitted_after_the_cutoff() -> None:
    models = {
        2020: fit_season_spm(2020, np.zeros((1, 11)), np.ones(1), np.zeros(1), np.zeros(1)),
        2021: fit_season_spm(2021, np.zeros((1, 11)), np.ones(1), np.zeros(1), np.zeros(1)),
        2022: fit_season_spm(2022, np.zeros((1, 11)), np.ones(1), np.zeros(1), np.zeros(1)),
    }
    fit_time = {2020: 100.0, 2021: 200.0, 2022: 300.0}
    el = ElSpm(models=models, fit_time=fit_time, k=250.0, half_life_days=365.0)
    rng = np.random.default_rng(0)
    for _ in range(50):
        cutoff = float(rng.uniform(50.0, 400.0))
        chosen = choose_model(el, cutoff)
        eligible = [s for s, t in fit_time.items() if t <= cutoff]
        if not eligible:
            assert chosen is None
        else:
            season = max(eligible, key=lambda s: (fit_time[s], s))
            assert chosen is models[season]
            assert fit_time[season] <= cutoff


def test_editing_a_later_gbl_box_row_does_not_change_game_g(
    gbl_data: SyntheticM3, el_spm: ElSpm
) -> None:
    frame = gbl_data.games[gbl_data.games["season"].between(2018, 2024)].sort_values("tipoff_utc")
    g = len(frame) // 2
    shares_before = projected_shares(frame, gbl_data.player_games, GBL_SPEC.projection_games)
    diff_before = spm_transfer_diff(frame, gbl_data.player_games, shares_before, el_spm)
    edited = gbl_data.player_games.copy()
    game_id = str(frame.iloc[g]["game_id"])
    last_id = str(frame.iloc[-1]["game_id"])
    own = edited.index[edited["game_id"] == game_id][0]
    last = edited.index[edited["game_id"] == last_id][0]
    edited.loc[own, "pts"] = float(edited.loc[own, "pts"]) + 50.0
    edited.loc[last, "pts"] = float(edited.loc[last, "pts"]) + 99.0
    shares_after = projected_shares(frame, edited, GBL_SPEC.projection_games)
    diff_after = spm_transfer_diff(frame, edited, shares_after, el_spm)
    assert np.isfinite(diff_before[g])  # a real prediction, so the equality below is not NaN == NaN
    np.testing.assert_allclose(diff_before[g], diff_after[g], rtol=0, atol=0)
    pd.testing.assert_frame_equal(
        shares_before[shares_before["game_id"] == game_id],
        shares_after[shares_after["game_id"] == game_id],
    )


def _leaky_choose_model(el: ElSpm, cutoff_time: float) -> SpmModel | None:
    eligible = [s for s, t in el.fit_time.items() if t <= cutoff_time + 400 * 86_400]
    if not eligible:
        return None
    season = max(eligible, key=lambda s: (el.fit_time[s], s))
    return el.models[season]


def test_planted_el_leak_is_caught(gbl_data: SyntheticM3, el_spm: ElSpm) -> None:
    frame = gbl_data.games[gbl_data.games["season"].between(2018, 2024)].sort_values("tipoff_utc")
    g = 30
    shares = projected_shares(frame, gbl_data.player_games, GBL_SPEC.projection_games)
    cutoff = next(t for t, idx in bi.round_batches(frame) if g in idx)
    safe_season = min(el_spm.models)
    leak_season = max(el_spm.models)
    safe_model = el_spm.models[safe_season]
    leak_model = el_spm.models[leak_season]
    fit_time = {safe_season: cutoff - 1.0, leak_season: cutoff + 200.0 * 86_400.0}
    el_toy = ElSpm(
        models={safe_season: safe_model, leak_season: leak_model},
        fit_time=fit_time,
        k=el_spm.k,
        half_life_days=el_spm.half_life_days,
    )
    before = spm_transfer_diff(frame, gbl_data.player_games, shares, el_toy)[g]
    assert np.isfinite(before)
    bumped = SpmModel(
        season=leak_model.season,
        stats=leak_model.stats,
        mean=leak_model.mean,
        std=leak_model.std,
        o_intercept=leak_model.o_intercept + 40.0,
        o_coef=leak_model.o_coef,
        d_intercept=leak_model.d_intercept,
        d_coef=leak_model.d_coef,
        alpha=leak_model.alpha,
        n_players=leak_model.n_players,
    )
    leaked = ElSpm(
        models={safe_season: safe_model, leak_season: bumped},
        fit_time=fit_time,
        k=el_spm.k,
        half_life_days=el_spm.half_life_days,
    )
    assert choose_model(el_toy, cutoff) is safe_model
    assert _leaky_choose_model(leaked, cutoff) is bumped
    after_leak = spm_transfer_diff(
        frame, gbl_data.player_games, shares, leaked, choose=_leaky_choose_model
    )[g]
    after_real = spm_transfer_diff(frame, gbl_data.player_games, shares, leaked)[g]
    assert after_leak != before
    np.testing.assert_allclose(after_real, before, rtol=0, atol=0)


def test_gbl_transfer_never_uses_euroleague_data_from_the_same_season_or_later(
    el_data: SyntheticM3, gbl_data: SyntheticM3, el_spm: ElSpm
) -> None:
    frame = gbl_data.games[gbl_data.games["season"].between(2018, 2024)].sort_values("tipoff_utc")
    g = 25
    shares = projected_shares(frame, gbl_data.player_games, GBL_SPEC.projection_games)
    before = spm_transfer_diff(frame, gbl_data.player_games, shares, el_spm)[g]
    assert np.isfinite(before)
    edited_stints = el_data.stints.copy()
    later = edited_stints.index[edited_stints["season"] >= frame.iloc[g]["season"]]
    edited_stints.loc[later, "home_points"] = (
        edited_stints.loc[later, "home_points"].astype(int) + 9
    )
    edited_box = el_data.player_games.copy()
    edited_box.loc[edited_box["season"] >= frame.iloc[g]["season"], "pts"] += 20.0
    el_after = el_spm_models(
        el_data.games,
        edited_box,
        edited_stints,
        el_data.checks,
        CHOSEN,
    )
    after = spm_transfer_diff(frame, gbl_data.player_games, shares, el_after)[g]
    np.testing.assert_allclose(after, before, rtol=0, atol=0)


def test_two_runs_are_byte_identical(gbl_data: SyntheticM3, el_spm: ElSpm) -> None:
    args = (gbl_data.games, gbl_data.team_games, gbl_data.player_games)
    kwargs = {
        "spec": GBL_SPEC,
        "el": el_spm,
        "tuned_m1": TUNED_M1,
        "box_only_fn": _zero_baseline,
        "pir_fn": _pir_baseline,
        "box_pages_skipped": 0,
        "score_test": False,
    }
    first = json.dumps(run_m3_gbl_backtest(*args, **kwargs), sort_keys=True)
    second = json.dumps(run_m3_gbl_backtest(*args, **kwargs), sort_keys=True)
    assert first == second


def test_report_has_every_model_and_split(gbl_data: SyntheticM3, el_spm: ElSpm) -> None:
    report = _run_gbl(gbl_data, el_spm)
    models = ("spm_transfer", "spm_transfer_scaled", "box_only", "pir", "m1", "b0")
    for split in ("tuning", "validation"):
        assert set(report["metrics"][split]) == set(models)
    assert report["comparisons"]["reference_model"] == "spm_transfer"
    assert report["comparisons"]["label"].startswith("GBL evidence")


def test_score_test_adds_the_test_split(gbl_data: SyntheticM3, el_spm: ElSpm) -> None:
    report = _run_gbl(gbl_data, el_spm, score_test=True)
    assert "test" in report["metrics"]
    assert report["test_scored"] is True


def test_cli_rejects_tuning_only_for_gbl() -> None:
    result = runner.invoke(
        cli.app,
        ["backtest", "--model", "m3", "--competition", "gbl", "--tuning-only"],
    )
    assert result.exit_code == 1
