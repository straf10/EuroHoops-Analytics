"""M3 H7 tests: season-end player report, ``on_cutoff`` hook, pair covariances."""

from __future__ import annotations

import json

import numpy as np
import pytest

from eurohoops.config import M3Backtest, M3Grid
from eurohoops.eval.m3_backtest import TunedRapm, build_rapm_inputs, prepare_data
from eurohoops.eval.m3_players import season_end_players
from eurohoops.models.rapm import fit_walk_forward, plain_player_columns
from eurohoops.models.rapm_posterior import (
    posterior_from_normal_equations,
    posterior_with_pair_covariance,
)
from eurohoops.models.spm import build_prior_fn, build_season_spm
from tests.m3_synthetic import SyntheticM3, make_synthetic_m3

SPEC = M3Backtest(
    report=None,  # type: ignore[arg-type]
    warmup=(2019,),
    tuning=(2020,),
    validation=(2021,),
    test=(2021,),
    grid=M3Grid(half_life_days=(365.0,), ridge=(500.0,)),
)

TUNED_RAPM = TunedRapm(365.0, 500.0, 500.0, 0.0, {})
SPM_EXTRA = {
    "k": 250.0,
    "alpha": 1.0,
    "target_half_life_days": 365.0,
    "target_ridge_o": 500.0,
    "target_ridge_d": 500.0,
}
TUNED_SPM = TunedRapm(365.0, 500.0, 500.0, 0.0, {}, extra=SPM_EXTRA)

LOOKUP_ATOL = 1e-6


@pytest.fixture(scope="module")
def synth() -> SyntheticM3:
    return make_synthetic_m3(
        [2019, 2020, 2021],
        teams=8,
        games_per_season=40,
        stints_per_game=12,
        roster=9,
        seed=20261002,
    )


def _prepare(synth: SyntheticM3):
    data = prepare_data(
        synth.games,
        synth.team_games,
        synth.player_games,
        synth.stints,
        synth.checks,
        spec=SPEC,
    )
    inputs = build_rapm_inputs(synth.stints, synth.checks, data.games, synth.player_games)
    return data, inputs


def _last_round_game_indices(games, season: int) -> np.ndarray:
    g = games[games["season"] == season]
    last_round = int(g["round"].max())
    return g.index[g["round"] == last_round].to_numpy()


def test_on_cutoff_none_reproduces_fit_walk_forward_exactly(synth: SyntheticM3) -> None:
    data, inputs = _prepare(synth)
    columns = plain_player_columns(inputs.spell_index)
    kwargs = {
        "half_life_days": 365.0,
        "ridge_o": 500.0,
        "ridge_d": 500.0,
    }
    base = fit_walk_forward(data.games, inputs.rows, columns, **kwargs)
    explicit = fit_walk_forward(data.games, inputs.rows, columns, on_cutoff=None, **kwargs)

    def noop(_t: float, _s: int, _m: object, _f: np.ndarray, _p: np.ndarray | None) -> None:
        return None

    with_cb = fit_walk_forward(data.games, inputs.rows, columns, on_cutoff=noop, **kwargs)
    np.testing.assert_array_equal(base.home_coef, explicit.home_coef)
    np.testing.assert_array_equal(base.home_coef, with_cb.home_coef)
    for a, b, c in zip(base.lookups, explicit.lookups, with_cb.lookups, strict=True):
        if a is None:
            assert b is None and c is None
            continue
        assert b is not None and c is not None
        assert a.o == b.o == c.o
        assert a.d == b.d == c.d


@pytest.mark.parametrize("variant,tuned", [("rapm", TUNED_RAPM), ("rapm_spm", TUNED_SPM)])
def test_posterior_mean_equals_the_walk_forward_rating_at_the_snapshot(
    synth: SyntheticM3, variant: str, tuned: TunedRapm
) -> None:
    data, inputs = _prepare(synth)
    columns = plain_player_columns(inputs.spell_index)
    if variant == "rapm_spm":
        season_spm = build_season_spm(
            data.games,
            inputs.player_games,
            inputs,
            rapm_half_life_days=SPM_EXTRA["target_half_life_days"],
            rapm_ridge_o=SPM_EXTRA["target_ridge_o"],
            rapm_ridge_d=SPM_EXTRA["target_ridge_d"],
            k=SPM_EXTRA["k"],
            alpha=SPM_EXTRA["alpha"],
        )
        prior_fn = build_prior_fn(season_spm, columns, SPM_EXTRA["k"])
        wf = fit_walk_forward(
            data.games,
            inputs.rows,
            columns,
            half_life_days=tuned.half_life_days,
            ridge_o=tuned.ridge_o,
            ridge_d=tuned.ridge_d,
            prior_fn=prior_fn,
        )
    else:
        wf = fit_walk_forward(
            data.games,
            inputs.rows,
            columns,
            half_life_days=tuned.half_life_days,
            ridge_o=tuned.ridge_o,
            ridge_d=tuned.ridge_d,
        )
    report = season_end_players(
        data.games,
        inputs,
        synth.player_games,
        tuned,
        variant,
        SPEC,
        data.snapshot,
    )
    for season_str, block in report["seasons"].items():
        season = int(season_str)
        for g_idx in _last_round_game_indices(data.games, season):
            lookup = wf.lookups[g_idx]
            assert lookup is not None
            for row in block["players"]:
                if not row["seen"]:
                    continue
                pid = row["player_id"]
                box = synth.player_games[
                    (synth.player_games["season"] == season)
                    & (synth.player_games["player_id"].astype(str) == pid)
                ]
                if not len(box):
                    continue
                team = str(box.iloc[0]["team"])
                o, d = lookup.rating(pid, team, season)
                assert row["o"] == pytest.approx(o, abs=LOOKUP_ATOL)
                assert row["d"] == pytest.approx(d, abs=LOOKUP_ATOL)


def test_unseen_players_sit_at_the_prior_with_prior_sd(synth: SyntheticM3) -> None:
    data, inputs = _prepare(synth)
    report = season_end_players(
        data.games,
        inputs,
        synth.player_games,
        TUNED_RAPM,
        "rapm",
        SPEC,
        data.snapshot,
    )
    columns = plain_player_columns(inputs.spell_index)
    penalty = columns.penalty(TUNED_RAPM.ridge_o, TUNED_RAPM.ridge_d)
    noise = report["noise_variance"]["value"]
    for block in report["seasons"].values():
        for row in block["players"]:
            if row["seen"]:
                continue
            pid = row["player_id"]
            o_col = columns.o_start + columns.o_labels.index(pid)
            d_col = columns.d_start + columns.d_labels.index(pid)
            assert row["o"] == pytest.approx(0.0, abs=LOOKUP_ATOL)
            assert row["d"] == pytest.approx(0.0, abs=LOOKUP_ATOL)
            assert row["sd_o"] == pytest.approx(
                float(np.sqrt(noise / penalty[o_col])), rel=0, abs=1e-5
            )
            assert row["sd_d"] == pytest.approx(
                float(np.sqrt(noise / penalty[d_col])), rel=0, abs=1e-5
            )


def test_interval_width_shrinks_with_minutes(synth: SyntheticM3) -> None:
    data, inputs = _prepare(synth)
    report = season_end_players(
        data.games,
        inputs,
        synth.player_games,
        TUNED_RAPM,
        "rapm",
        SPEC,
        data.snapshot,
    )
    rho = report["summary"]["sd_total_minutes_spearman"]
    assert rho < 0.0


def test_snapshot_never_uses_the_last_round_or_later(synth: SyntheticM3) -> None:
    data, inputs = _prepare(synth)
    base = season_end_players(
        data.games,
        inputs,
        synth.player_games,
        TUNED_RAPM,
        "rapm",
        SPEC,
        data.snapshot,
    )
    target_season = 2020
    edited_stints = synth.stints.copy()
    g = synth.games[synth.games["season"] == target_season]
    last_round = int(g["round"].max())
    last_game_id = str(g.loc[g["round"] == last_round, "game_id"].iloc[0])
    row_idx = edited_stints.index[edited_stints["game_id"] == last_game_id][0]
    edited_stints.at[row_idx, "home_points"] = int(edited_stints.at[row_idx, "home_points"]) + 7
    prep = prepare_data(
        synth.games,
        synth.team_games,
        synth.player_games,
        edited_stints,
        synth.checks,
        spec=SPEC,
    )
    inp = build_rapm_inputs(edited_stints, synth.checks, prep.games, synth.player_games)
    after_last_round = season_end_players(
        prep.games,
        inp,
        synth.player_games,
        TUNED_RAPM,
        "rapm",
        SPEC,
        prep.snapshot,
    )
    assert base["seasons"][str(target_season)] == after_last_round["seasons"][str(target_season)]

    later_stints = edited_stints.copy()
    later_game = str(synth.games[synth.games["season"] == 2021].iloc[-1]["game_id"])
    lidx = later_stints.index[later_stints["game_id"] == later_game][0]
    later_stints.at[lidx, "away_points"] = int(later_stints.at[lidx, "away_points"]) + 5
    prep2 = prepare_data(
        synth.games,
        synth.team_games,
        synth.player_games,
        later_stints,
        synth.checks,
        spec=SPEC,
    )
    inp2 = build_rapm_inputs(later_stints, synth.checks, prep2.games, synth.player_games)
    after_later = season_end_players(
        prep2.games,
        inp2,
        synth.player_games,
        TUNED_RAPM,
        "rapm",
        SPEC,
        prep2.snapshot,
    )
    assert base["seasons"][str(target_season)] == after_later["seasons"][str(target_season)]


def test_two_runs_are_byte_identical(synth: SyntheticM3) -> None:
    data, inputs = _prepare(synth)
    a = season_end_players(
        data.games, inputs, synth.player_games, TUNED_RAPM, "rapm", SPEC, data.snapshot
    )
    b = season_end_players(
        data.games, inputs, synth.player_games, TUNED_RAPM, "rapm", SPEC, data.snapshot
    )
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_posterior_with_pair_covariance_matches_reference() -> None:
    rng = np.random.default_rng(3)
    p = 8
    a = rng.normal(size=(p, p))
    gram = a @ a.T + np.eye(p) * 0.5
    rhs = rng.normal(size=p)
    penalty = rng.uniform(0.5, 5.0, size=p)
    prior_mean = rng.normal(size=p)
    noise_var = 2.5
    pairs = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int64)

    ref = posterior_from_normal_equations(gram, rhs, penalty, prior_mean, noise_var)
    fit, pair_cov = posterior_with_pair_covariance(gram, rhs, penalty, prior_mean, noise_var, pairs)
    np.testing.assert_allclose(fit.mean, ref.mean, atol=1e-9)
    np.testing.assert_allclose(fit.sd, ref.sd, atol=1e-9)
    system = gram + np.diag(penalty)
    dense_inv = np.linalg.inv(system)
    expected = noise_var * dense_inv[pairs[:, 0], pairs[:, 1]]
    np.testing.assert_allclose(pair_cov, expected, atol=1e-9)
