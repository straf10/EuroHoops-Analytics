"""M3 RAPM backtest (week 9-12 H1): walk-forward tuning, the box-only gate, test (once).

Every RAPM hyperparameter (half-life, ridge) is chosen on the tuning seasons only, by
future-margin RMSE: every (half-life, shared ridge) pair, then a one-dimensional search of a
separate offense and defense ridge at the best half-life, each kept only if it lowers tuning
RMSE (D7). ``VARIANTS`` is a registry of declared RAPM variants (name -> its tuner and fit);
this module declares only the plain ``rapm`` variant. Subagents E and F (``rapm_dummy``,
``rapm_spm``) add their own entries and their own extra grid axis to this same registry, so the
tuning loop and the report stay generic over however many variants are declared. The chosen
variant is whichever has the lowest tuning RMSE.

Comparison models: ``m1`` (the committed, un-retuned ``reports/backtest_m1.json`` parameters
replayed through ``models.team_eff.forecast``), ``b0`` (the tuning-season home margin constant),
and ``box_only``/``pir`` from ``models.box_impact`` (H2), passed in by the caller with the
``BaselineFn`` call shape (the CLI binds box-only's ``BoxGrid``), so tests can use fakes.

A game is *scored* in a split when it is rated (played, not forfeit), in that split's seasons,
and every model in play produced a finite prediction for it (H-d): the same game set is used for
every model's metrics, the gate and the comparisons, so they are a fair head-to-head. The gate
(H-d): the chosen RAPM variant beats ``box_only`` on validation iff the game-level bootstrap 95%
CI of RMSE(chosen) - RMSE(box_only) has an upper bound < 0 (1,000 resamples, seed 20261001,
resampling game indices and recomputing both RMSEs per resample -- RMSE is not a mean of
per-game values, so ``eval.metrics.paired_bootstrap_ci`` does not apply here).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib.metadata import version
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import special

from eurohoops.config import M3Backtest, M3Grid
from eurohoops.eval.metrics import per_game_log_loss
from eurohoops.models.box_impact import BaselineResult
from eurohoops.models.elo import FloatArray
from eurohoops.models.minutes import expected_possessions, oracle_shares, projected_shares
from eurohoops.models.rapm import (
    DesignRows,
    SpellIndex,
    WalkForward,
    build_design_rows,
    build_spell_index,
    fit_walk_forward,
    plain_player_columns,
)
from eurohoops.models.team_eff import DecayParams, prepare_history
from eurohoops.models.team_eff import forecast as team_eff_forecast

BoolArray = npt.NDArray[np.bool_]
SPLITS = ("tuning", "validation", "test")


def _round(value: float | None) -> float | None:
    return None if value is None or not math.isfinite(value) else round(float(value), 6)


# --- The comparison-model protocol (box_only/pir: models.box_impact) ----------------------------


class BaselineFn(Protocol):
    """The call shape every comparison-model registry entry has once its own extra grid (e.g.
    box_only's shrinkage grid) is bound with ``functools.partial``."""

    def __call__(
        self,
        games: pd.DataFrame,
        player_games: pd.DataFrame,
        shares: pd.DataFrame,
        possessions: pd.Series,
        tuning: BoolArray,
    ) -> BaselineResult: ...


# --- Data preparation -----------------------------------------------------------------------


def _data_sha256(*frames: pd.DataFrame) -> str:
    """sha256 of every input table's canonical CSV, in the order given."""
    digest = hashlib.sha256()
    for frame in frames:
        digest.update(frame.to_csv(index=False, lineterminator="\n").encode())
    return digest.hexdigest()


@dataclass(frozen=True)
class Data:
    games: pd.DataFrame  # warmup..test seasons, sorted by tip-off
    margin: FloatArray
    home_won: FloatArray
    split: dict[str, BoolArray]  # tuning / validation / test, rated games only
    shares: pd.DataFrame  # projected_shares (H-c)
    oracle: pd.DataFrame  # oracle_shares (reported, never gated)
    possessions: pd.Series
    snapshot: str


def prepare_data(
    games: pd.DataFrame,
    team_games: pd.DataFrame,
    player_games: pd.DataFrame,
    stints: pd.DataFrame,
    checks: pd.DataFrame,
    *,
    spec: M3Backtest,
) -> Data:
    first, last = spec.warmup[0], spec.test[-1]
    frame = (
        games[games["season"].between(first, last)].sort_values("tipoff_utc").reset_index(drop=True)
    )
    rated = (frame["played"] & ~frame["forfeit"]).to_numpy()
    home = frame["home_score"].astype("float64").fillna(0.0).to_numpy()
    away = frame["away_score"].astype("float64").fillna(0.0).to_numpy()
    season = frame["season"].to_numpy()
    return Data(
        games=frame,
        margin=home - away,
        home_won=(home > away).astype(np.float64),
        split={
            name: rated & np.isin(season, seasons)
            for name, seasons in (
                ("tuning", spec.tuning),
                ("validation", spec.validation),
                ("test", spec.test),
            )
        },
        shares=projected_shares(frame, player_games, spec.projection_games),
        oracle=oracle_shares(frame, player_games),
        possessions=expected_possessions(frame, team_games),
        snapshot=_data_sha256(frame, team_games, player_games, stints, checks),
    )


# --- The rapm variant: base design, tuning, and margins from a WalkForward ----------------------


@dataclass(frozen=True)
class RapmFitInputs:
    """The base (spell-level) design, shared by every RAPM variant's grid search: building it
    once and reusing it across hyperparameter combinations (and, for E/F, across variants) is
    what keeps a full tuning grid affordable (H-j)."""

    spell_index: SpellIndex
    rows: DesignRows


def build_rapm_inputs(
    stints: pd.DataFrame, checks: pd.DataFrame, games: pd.DataFrame
) -> RapmFitInputs:
    spell_index = build_spell_index(stints)
    rows = build_design_rows(stints, checks, games, spell_index)
    return RapmFitInputs(spell_index, rows)


def rapm_margins(
    games: pd.DataFrame, wf: WalkForward, shares: pd.DataFrame, possessions: pd.Series
) -> FloatArray:
    """H-c's margin formula from a fitted ``WalkForward`` and a shares frame (projected or
    oracle): ``P/100 * (h * home_flag + sum_home f_p*r_p - sum_away f_p*r_p)``."""
    n = len(games)
    home_sum = np.zeros(n)
    away_sum = np.zeros(n)
    game_ids = games["game_id"].astype(str).to_numpy()
    index = {gid: i for i, gid in enumerate(game_ids)}
    season_of = dict(zip(game_ids, games["season"].astype(int).to_numpy(), strict=True))
    for game_id, side, team, player_id, share in zip(
        shares["game_id"].astype(str),
        shares["side"],
        shares["team"].astype(str),
        shares["player_id"].astype(str),
        shares["share"],
        strict=True,
    ):
        i = index.get(game_id)
        if i is None:
            continue
        lookup = wf.lookups[i]
        if lookup is None:
            continue
        o, d = lookup.rating(player_id, team, season_of[game_id])
        rating = float(share) * (o + d)
        if side == "home":
            home_sum[i] += rating
        else:
            away_sum[i] += rating
    home_flag = np.where(games["neutral"].to_numpy(), 0.0, 1.0)
    poss = possessions.reindex(game_ids).to_numpy(dtype=np.float64)
    margin: FloatArray = poss / 100.0 * (wf.home_coef * home_flag + home_sum - away_sum)
    return margin


def _finite_rmse(pred: FloatArray, actual: FloatArray, mask: BoolArray) -> float:
    sel = mask & np.isfinite(pred)
    return float(np.sqrt(np.mean((pred[sel] - actual[sel]) ** 2))) if sel.any() else math.nan


@dataclass(frozen=True)
class TunedRapm:
    half_life_days: float
    ridge_o: float
    ridge_d: float
    tuning_rmse: float
    grid: dict[str, Any]
    extra: dict[str, Any] = field(default_factory=dict)  # a variant's own chosen params


def tune_rapm(
    data: Data, spec: M3Backtest, inputs: RapmFitInputs, earlier: dict[str, TunedRapm]
) -> TunedRapm:
    """The ``rapm`` variant's grid search (D7). ``earlier`` (the variants tuned before this
    one) is unused: ``rapm`` is tuned first, and later variants search their own axis at its
    chosen half-life and ridge."""
    del earlier
    grid: M3Grid = spec.grid
    columns = plain_player_columns(inputs.spell_index)
    last_tuning_season = max(spec.tuning)
    sub_mask = data.games["season"].to_numpy() <= last_tuning_season
    games_tuning = data.games[sub_mask].reset_index(drop=True)
    tuning_mask = data.split["tuning"][sub_mask]
    actual = data.margin[sub_mask]
    # Precomputed once: every grid point re-solves the fit, but never needs shares/possessions
    # rows outside this truncated frame, so filtering here (not inside the closure) keeps the
    # per-game-lookup loop in rapm_margins from scanning validation/test rows on every combo.
    tuning_game_ids = set(games_tuning["game_id"].astype(str))
    shares_tuning = data.shares[data.shares["game_id"].astype(str).isin(tuning_game_ids)]

    def rmse_for(half_life: float, ridge_o: float, ridge_d: float) -> float:
        wf = fit_walk_forward(
            games_tuning,
            inputs.rows,
            columns,
            half_life_days=half_life,
            ridge_o=ridge_o,
            ridge_d=ridge_d,
        )
        pred = rapm_margins(games_tuning, wf, shares_tuning, data.possessions)
        return _finite_rmse(pred, actual, tuning_mask)

    shared = [
        (half_life, ridge, rmse_for(half_life, ridge, ridge))
        for half_life in grid.half_life_days
        for ridge in grid.ridge
    ]
    best_half_life, best_shared_ridge, shared_best_rmse = min(shared, key=lambda t: t[2])

    o_candidates = [(r, rmse_for(best_half_life, r, best_shared_ridge)) for r in grid.ridge]
    best_o_ridge, best_o_rmse = min(o_candidates, key=lambda t: t[1])
    o_improved = best_o_rmse < shared_best_rmse
    ridge_o = best_o_ridge if o_improved else best_shared_ridge

    d_candidates = [(r, rmse_for(best_half_life, best_shared_ridge, r)) for r in grid.ridge]
    best_d_ridge, best_d_rmse = min(d_candidates, key=lambda t: t[1])
    d_improved = best_d_rmse < shared_best_rmse
    ridge_d = best_d_ridge if d_improved else best_shared_ridge

    final_rmse = (
        rmse_for(best_half_life, ridge_o, ridge_d) if o_improved or d_improved else shared_best_rmse
    )
    grid_report = {
        "half_life_days": list(grid.half_life_days),
        "ridge": list(grid.ridge),
        "size": len(shared) + 2 * len(grid.ridge),
        "shared_best": {
            "half_life_days": best_half_life,
            "ridge": best_shared_ridge,
            "tuning_rmse": _round(shared_best_rmse),
        },
        "separate_o": {
            "ridge": best_o_ridge,
            "tuning_rmse": _round(best_o_rmse),
            "improved": o_improved,
        },
        "separate_d": {
            "ridge": best_d_ridge,
            "tuning_rmse": _round(best_d_rmse),
            "improved": d_improved,
        },
    }
    return TunedRapm(best_half_life, ridge_o, ridge_d, final_rmse, grid_report)


def fit_rapm(games: pd.DataFrame, inputs: RapmFitInputs, tuned: TunedRapm) -> WalkForward:
    """The plain ``rapm`` variant's walk-forward fit over ``games`` with tuned parameters."""
    return fit_walk_forward(
        games,
        inputs.rows,
        plain_player_columns(inputs.spell_index),
        half_life_days=tuned.half_life_days,
        ridge_o=tuned.ridge_o,
        ridge_d=tuned.ridge_d,
    )


@dataclass(frozen=True)
class Variant:
    """A declared RAPM variant: its tuner (tuning seasons only; receives the variants tuned
    before it, in ``VARIANTS`` order) and its walk-forward fit."""

    tune: Callable[[Data, M3Backtest, RapmFitInputs, dict[str, TunedRapm]], TunedRapm]
    fit: Callable[[pd.DataFrame, RapmFitInputs, TunedRapm], WalkForward]


VARIANTS: dict[str, Variant] = {"rapm": Variant(tune_rapm, fit_rapm)}


# --- Comparison models: m1, b0 ------------------------------------------------------------------


def m1_margins(
    games: pd.DataFrame, team_games: pd.DataFrame, tuned_m1: dict[str, Any]
) -> FloatArray:
    """M1's committed, un-retuned parameters (``reports/backtest_m1.json``'s ``tuned`` block),
    replayed through ``models.team_eff.forecast`` (D5: M1 itself is never touched)."""
    rating = DecayParams(**tuned_m1["rating"])
    pace = DecayParams(**tuned_m1["pace"])
    history = prepare_history(games, team_games)
    forecast = team_eff_forecast(history, rating, pace)
    return forecast.margin


def m1_margins_for(
    frame: pd.DataFrame,
    all_games: pd.DataFrame,
    team_games: pd.DataFrame,
    tuned_m1: dict[str, Any],
) -> FloatArray:
    """M1 margins for ``frame``'s games, replayed over every game of the competition (M1 warms
    up from 2007, before the M3 frame starts) and aligned by ``game_id``."""
    ordered = all_games.sort_values("tipoff_utc", kind="stable").reset_index(drop=True)
    margin = pd.Series(
        m1_margins(ordered, team_games, tuned_m1), index=ordered["game_id"].astype(str)
    )
    aligned: FloatArray = np.asarray(margin.reindex(frame["game_id"].astype(str)), dtype=np.float64)
    return aligned


def b0_margin(data: Data) -> tuple[FloatArray, float]:
    """The tuning-season home margin constant (0 at a neutral venue)."""
    fit_mask = data.split["tuning"] & ~data.games["neutral"].to_numpy()
    home_margin = float(data.margin[fit_mask].mean()) if fit_mask.any() else 0.0
    margin: FloatArray = np.where(data.games["neutral"].to_numpy(), 0.0, home_margin)
    return margin, home_margin


# --- Metrics, the gate and comparisons ------------------------------------------------------


def _sigma(residual: FloatArray, mask: BoolArray) -> float:
    sel = mask & np.isfinite(residual)
    return float(np.sqrt(np.mean(residual[sel] ** 2))) if sel.any() else math.nan


def _model_metrics(margin: FloatArray, data: Data, scored: dict[str, BoolArray]) -> dict[str, Any]:
    residual = data.margin - margin
    sigma = _sigma(residual, scored["tuning"])
    out: dict[str, Any] = {}
    for split in SPLITS:
        mask = scored.get(split)
        if mask is None or not mask.any():
            out[split] = {
                "n": 0,
                "rmse": None,
                "mae": None,
                "log_loss": None,
                "sigma": _round(sigma),
            }
            continue
        r = residual[mask]
        rmse = float(np.sqrt(np.mean(r**2)))
        mae = float(np.mean(np.abs(r)))
        p = np.clip(special.ndtr(margin[mask] / sigma), 1e-9, 1.0 - 1e-9)
        log_loss = float(per_game_log_loss(p, data.home_won[mask]).mean())
        out[split] = {
            "n": int(mask.sum()),
            "rmse": _round(rmse),
            "mae": _round(mae),
            "log_loss": _round(log_loss),
            "sigma": _round(sigma),
        }
    return out


def rmse_diff_bootstrap_ci(
    residual_a: FloatArray, residual_b: FloatArray, resamples: int, seed: int
) -> dict[str, Any] | None:
    """The 95% percentile CI of RMSE(a) - RMSE(b) over game-index resamples (H-d): RMSE is not
    a mean of per-game values, so ``eval.metrics.paired_bootstrap_ci`` (built for linear
    statistics) does not apply -- both RMSEs are recomputed on every resample instead."""
    n = len(residual_a)
    if not n:
        return None
    point = float(np.sqrt(np.mean(residual_a**2)) - np.sqrt(np.mean(residual_b**2)))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(resamples, n))
    rmse_a = np.sqrt((residual_a[idx] ** 2).mean(axis=1))
    rmse_b = np.sqrt((residual_b[idx] ** 2).mean(axis=1))
    low, high = np.percentile(rmse_a - rmse_b, [2.5, 97.5])
    return {
        "mean": _round(point),
        "ci95": [_round(float(low)), _round(float(high))],
        "resamples": resamples,
        "seed": seed,
    }


def model_version(tuned: TunedRapm, chosen: str) -> str:
    key = json.dumps(
        {
            "variant": chosen,
            "half_life_days": tuned.half_life_days,
            "ridge_o": tuned.ridge_o,
            "ridge_d": tuned.ridge_d,
            **tuned.extra,
        },
        sort_keys=True,
    )
    return f"{version('eurohoops')}+m3.{hashlib.sha256(key.encode()).hexdigest()[:8]}"


# --- The report -----------------------------------------------------------------------------


def run_m3_backtest(
    games: pd.DataFrame,
    team_games: pd.DataFrame,
    player_games: pd.DataFrame,
    stints: pd.DataFrame,
    checks: pd.DataFrame,
    *,
    spec: M3Backtest,
    tuned_m1: dict[str, Any],
    box_only_fn: BaselineFn,
    pir_fn: BaselineFn,
    tuning_only: bool = False,
    score_test: bool = False,
) -> dict[str, Any]:
    """The M3 report (module docstring). ``tuning_only``: only the tuning grid, chosen variant
    and tuning metrics are computed and written -- no validation or test metric anywhere, since
    the verdict (which variant is chosen) must be committed before validation is scored (H5).
    ``score_test``: also score the test seasons (only after the verdict commit)."""
    data = prepare_data(games, team_games, player_games, stints, checks, spec=spec)
    inputs = build_rapm_inputs(stints, checks, data.games)

    tuned: dict[str, TunedRapm] = {}
    for name, variant in VARIANTS.items():
        tuned[name] = variant.tune(data, spec, inputs, dict(tuned))
    chosen_name = min(tuned, key=lambda name: tuned[name].tuning_rmse)
    chosen = tuned[chosen_name]

    grid_report = {name: t.grid for name, t in tuned.items()}
    chosen_block = {
        "variant": chosen_name,
        "half_life_days": chosen.half_life_days,
        "ridge_o": chosen.ridge_o,
        "ridge_d": chosen.ridge_d,
        "tuning_rmse": _round(chosen.tuning_rmse),
        **chosen.extra,
    }
    report: dict[str, Any] = {
        "model": "m3",
        "model_version": model_version(chosen, chosen_name),
        "seasons": {
            "warmup": list(spec.warmup),
            "tuning": list(spec.tuning),
            "validation": list(spec.validation),
            "test": list(spec.test),
        },
        "data_sha256": data.snapshot,
        "tuning_only": tuning_only,
        "test_scored": score_test and not tuning_only,
        "validation_scored": not tuning_only,
        "grid": grid_report,
        "chosen": chosen_block,
    }

    # The full walk-forward fit and every comparison model: needed for tuning metrics even in
    # tuning_only mode (H5's verdict commit reports tuning numbers), validation/test are simply
    # not written to the report in that mode (no validation or test metric anywhere).
    wf = VARIANTS[chosen_name].fit(data.games, inputs, chosen)
    rapm_margin = rapm_margins(data.games, wf, data.shares, data.possessions)
    oracle_margin = rapm_margins(data.games, wf, data.oracle, data.possessions)
    m1_margin = m1_margins_for(data.games, games, team_games, tuned_m1)
    b0_margins, b0_home_margin = b0_margin(data)
    tuning_bool = data.split["tuning"]
    box_result = box_only_fn(data.games, player_games, data.shares, data.possessions, tuning_bool)
    pir_result = pir_fn(data.games, player_games, data.shares, data.possessions, tuning_bool)

    if tuning_only:
        splits: tuple[str, ...] = ("tuning",)
    else:
        splits = ("tuning", "validation", *(("test",) if score_test else ()))
    finite = {
        "rapm": np.isfinite(rapm_margin),
        "box_only": np.isfinite(box_result.margin),
        "pir": np.isfinite(pir_result.margin),
        "m1": np.isfinite(m1_margin),
        "b0": np.isfinite(b0_margins),
    }
    common_finite = np.logical_and.reduce(list(finite.values()))
    scored = {split: data.split[split] & common_finite for split in splits}

    report["games_per_split"] = {
        split: {
            "rated": int(data.split[split].sum()),
            "scored": int(scored[split].sum()),
            "dropped": int(data.split[split].sum()) - int(scored[split].sum()),
        }
        for split in splits
    }
    report["metrics"] = {
        split: {
            "rapm": _model_metrics(rapm_margin, data, scored)[split],
            "box_only": _model_metrics(box_result.margin, data, scored)[split],
            "pir": _model_metrics(pir_result.margin, data, scored)[split],
            "m1": _model_metrics(m1_margin, data, scored)[split],
            "b0": _model_metrics(b0_margins, data, scored)[split],
        }
        for split in splits
    }
    report["b0"] = {"home_margin": _round(b0_home_margin)}
    report["box_only_params"] = box_result.params
    report["pir_params"] = pir_result.params
    oracle_scored = {split: scored[split] & finite["rapm"] for split in splits}
    report["oracle_minutes"] = {
        "label": "oracle, not a forecast",
        **{split: _model_metrics(oracle_margin, data, oracle_scored)[split] for split in splits},
    }
    if tuning_only:
        return report  # no validation or test metric anywhere (H5: the verdict commit)

    validation_scored = scored["validation"]
    chosen_residual = (data.margin - rapm_margin)[validation_scored]
    box_residual = (data.margin - box_result.margin)[validation_scored]
    gate_ci = rmse_diff_bootstrap_ci(
        chosen_residual, box_residual, spec.bootstrap_resamples, spec.bootstrap_seed
    )
    passed = None if gate_ci is None else bool(gate_ci["ci95"][1] < 0.0)
    report["gate"] = {
        "rule": (
            "bootstrap 95% CI of RMSE(chosen RAPM) - RMSE(box_only) on validation, upper bound < 0"
        ),
        "metric": "validation margin RMSE",
        "variant": chosen_name,
        "reference": "box_only",
        "rmse_diff": gate_ci,
        "passed": passed,
    }
    report["comparisons"] = {
        model: {
            "rmse_diff": rmse_diff_bootstrap_ci(
                chosen_residual,
                (data.margin - other)[validation_scored],
                spec.bootstrap_resamples,
                spec.bootstrap_seed,
            )
        }
        for model, other in (("m1", m1_margin), ("pir", pir_result.margin))
    }
    return report


def format_m3_table(report: dict[str, Any]) -> str:
    if report.get("tuning_only"):
        chosen = report["chosen"]
        return f"chosen: {chosen['variant']} (tuning RMSE {chosen['tuning_rmse']})"
    header = f"{'split':<11}{'model':<9}{'n':>6}{'rmse':>8}{'mae':>8}{'logloss':>9}"
    lines = [header, "-" * len(header)]
    for split, models in report["metrics"].items():
        for name, m in models.items():
            n = m["n"]
            rmse = m["rmse"] if m["rmse"] is not None else float("nan")
            mae = m["mae"] if m["mae"] is not None else float("nan")
            log_loss = m["log_loss"] if m["log_loss"] is not None else float("nan")
            lines.append(f"{split:<11}{name:<9}{n:>6}{rmse:>8.3f}{mae:>8.3f}{log_loss:>9.4f}")
    gate = report["gate"]
    ci = gate["rmse_diff"]
    if ci is not None:
        lines.append(
            f"gate ({gate['variant']} vs {gate['reference']}): RMSE diff {ci['mean']:+.4f} "
            f"95% CI [{ci['ci95'][0]:+.4f}, {ci['ci95'][1]:+.4f}] -> "
            + ("PASS" if gate["passed"] else "FAIL")
        )
    return "\n".join(lines)
