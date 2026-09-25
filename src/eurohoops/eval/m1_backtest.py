"""M1 walk-forward backtest vs a comparison Elo and B0: tuning, validation gate, test (once).

Every hyperparameter is chosen on the tuning seasons only: the pace grid by tuning MSE of
possessions per 40 minutes, the rating grid by tuning log loss (Normal margin, tuning-RMS sigma),
then each declared margin distribution is fitted on tuning and the one with the best tuning
log loss is the gate variant. The comparison Elo is the existing grid re-tuned on the same
tuning seasons (E-b); it never sees validation either. The live Elo is not touched.

The gate (per competition): M1 validation log loss < comparison-Elo validation log loss, with
paired bootstrap CIs of the per-game differences in log loss, margin CRPS and totals CRPS.
Test seasons are scored only with ``score_test`` (after the verdict is committed).
"""

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from importlib.metadata import version
from itertools import product
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import special

from eurohoops.config import Backtest, DecayGrid, M1Backtest
from eurohoops.eval.backtest import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    grid_edges,
    run_backtest,
    totals_baseline,
    win_probabilities,
)
from eurohoops.eval.metrics import crps_normal, paired_bootstrap_ci, per_game_log_loss, score
from eurohoops.eval.tracking import data_hash
from eurohoops.models.elo import EloParams, FloatArray, prepare, replay
from eurohoops.models.team_eff import (
    MINUTES_PER_GAME,
    VARIANTS,
    DecayParams,
    Forecast,
    History,
    MarginModel,
    fit_margin_model,
    pace_fits,
    prepare_history,
    rating_fits,
)

LOPSIDED = 0.75  # comparison-Elo P(favourite) at or above this counts as a lopsided game
EARLY_ROUNDS = 5  # regular-season rounds 1-5 count as early season
SPLITS = ("tuning", "validation", "test")
BoolArray = npt.NDArray[np.bool_]


def _round(value: float) -> float:
    return round(float(value), 6)


def _grid(grid: DecayGrid) -> list[DecayParams]:
    return [DecayParams(*combo) for combo in product(grid.half_life_days, grid.carry, grid.ridge)]


def _decay_edges(grid: DecayGrid, best: DecayParams) -> list[str]:
    """Axes whose best value sits on the grid edge; carry = 1 (no extra shrinkage) is a
    natural bound, not an edge."""
    return [
        axis
        for axis, values in asdict(grid).items()
        if len(values) > 1
        and getattr(best, axis) in {max(values), min(values)}
        and not (axis == "carry" and getattr(best, axis) == 1.0)
    ]


def _ci(diff: FloatArray) -> dict[str, Any]:
    mean, low, high = paired_bootstrap_ci(diff, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED)
    return {
        "mean": _round(mean),
        "ci95": [_round(low), _round(high)],
        "resamples": BOOTSTRAP_RESAMPLES,
        "seed": BOOTSTRAP_SEED,
    }


@dataclass(frozen=True)
class Predictions:
    """One model's per-game forecasts on the scored games."""

    p_home: FloatArray
    margin: FloatArray
    margin_crps: FloatArray
    total: FloatArray
    total_crps: FloatArray


def _metrics(
    pred: Predictions, mask: BoolArray, margin: FloatArray, total: FloatArray
) -> dict[str, Any]:
    base = score(pred.p_home[mask], pred.margin[mask], margin[mask]).as_dict()
    n = int(mask.sum())
    return {
        **base,
        "margin_crps": _round(pred.margin_crps[mask].mean()) if n else None,
        "totals_mae": _round(np.abs(pred.total[mask] - total[mask]).mean()) if n else None,
        "totals_crps": _round(pred.total_crps[mask].mean()) if n else None,
    }


@dataclass(frozen=True)
class Data:
    games: pd.DataFrame  # every game of the backtest seasons, tip-off order
    history: History
    rated: BoolArray  # played and not forfeit
    has_rows: BoolArray  # the game has team_games rows
    margin: FloatArray
    total: FloatArray
    home_won: FloatArray
    poss40: FloatArray  # NaN without rows
    split: dict[str, BoolArray]
    snapshot: str  # sha256 of the game and team-game rows read


def prepare_data(games: pd.DataFrame, team_games: pd.DataFrame, spec: M1Backtest) -> Data:
    first, last = spec.warmup[0], spec.test[-1]
    frame = games[games["season"].between(first, last)].reset_index(drop=True)
    rows = team_games[team_games["game_id"].isin(set(frame["game_id"]))]
    history = prepare_history(frame, rows)
    rated = (frame["played"] & ~frame["forfeit"]).to_numpy()
    poss40 = rows.drop_duplicates("game_id").set_index("game_id")
    per40 = (poss40["poss_game"] * MINUTES_PER_GAME / poss40["minutes"]).reindex(frame["game_id"])
    home = frame["home_score"].astype("float64").fillna(0.0).to_numpy()
    away = frame["away_score"].astype("float64").fillna(0.0).to_numpy()
    season = frame["season"].to_numpy()
    return Data(
        games=frame,
        history=history,
        rated=rated,
        has_rows=per40.notna().to_numpy(),
        margin=home - away,
        total=home + away,
        home_won=(home > away).astype(np.float64),
        poss40=per40.to_numpy(dtype=np.float64),
        snapshot=data_hash(frame, rows),
        split={
            name: rated & np.isin(season, seasons)
            for name, seasons in (
                ("tuning", spec.tuning),
                ("validation", spec.validation),
                ("test", spec.test),
            )
        },
    )


def _normal_log_loss(margin: FloatArray, data: Data, mask: BoolArray) -> float:
    residual = data.margin[mask] - margin[mask]
    sigma = float(np.sqrt(np.mean(residual**2)))
    p = np.clip(special.ndtr(margin[mask] / sigma), 1e-9, 1 - 1e-9)
    return float(per_game_log_loss(p, data.home_won[mask]).mean())


@dataclass(frozen=True)
class Tuned:
    pace: DecayParams
    rating: DecayParams
    forecast: Forecast
    margin_models: dict[str, MarginModel]
    variant: str
    totals_sigma: float
    grids: dict[str, Any]


def tune(data: Data, spec: M1Backtest) -> Tuned:
    tuning = data.split["tuning"]
    fit_mask = tuning & data.has_rows
    pace_grid = _grid(spec.pace_grid)
    pace_mse = []
    for params in pace_grid:
        pace = pace_fits(data.history, params)
        pace_mse.append(float(np.mean((pace[fit_mask] - data.poss40[fit_mask]) ** 2)))
    best_pace = pace_grid[int(np.argmin(pace_mse))]
    pace = pace_fits(data.history, best_pace)

    rating_grid = _grid(spec.rating_grid)
    losses = []
    for params in rating_grid:
        home, away = rating_fits(data.history, params)
        losses.append(_normal_log_loss(pace * (home - away) / 100.0, data, tuning))
    best_rating = rating_grid[int(np.argmin(losses))]
    home, away = rating_fits(data.history, best_rating)
    forecast = Forecast(pace * home / 100.0, pace * away / 100.0, pace)

    models = {
        variant: fit_margin_model(
            variant, forecast.margin[tuning], forecast.pace[tuning], data.margin[tuning]
        )
        for variant in VARIANTS
    }
    variant_loss = {
        variant: float(
            per_game_log_loss(
                model.p_home(forecast.margin[tuning], forecast.pace[tuning]),
                data.home_won[tuning],
            ).mean()
        )
        for variant, model in models.items()
    }
    chosen = min(VARIANTS, key=lambda v: (variant_loss[v], VARIANTS.index(v)))
    totals_sigma = _round(np.sqrt(np.mean((data.total[tuning] - forecast.total[tuning]) ** 2)))
    grids = {
        "pace": {
            **{axis: list(values) for axis, values in asdict(spec.pace_grid).items()},
            "size": len(pace_grid),
            "objective": "tuning MSE of possessions per 40 minutes",
            "best": {**asdict(best_pace), "tuning_mse": _round(min(pace_mse))},
            "best_on_edge": _decay_edges(spec.pace_grid, best_pace),
        },
        "rating": {
            **{axis: list(values) for axis, values in asdict(spec.rating_grid).items()},
            "size": len(rating_grid),
            "objective": "tuning log loss, Normal margin with tuning-RMS sigma",
            "best": {**asdict(best_rating), "tuning_log_loss": _round(min(losses))},
            "best_on_edge": _decay_edges(spec.rating_grid, best_rating),
        },
        "variant_tuning_log_loss": {v: _round(loss) for v, loss in variant_loss.items()},
    }
    return Tuned(best_pace, best_rating, forecast, models, chosen, totals_sigma, grids)


def m1_predictions(tuned: Tuned, variant: str, data: Data) -> Predictions:
    model = tuned.margin_models[variant]
    f = tuned.forecast
    return Predictions(
        p_home=model.p_home(f.margin, f.pace),
        margin=f.margin,
        margin_crps=model.crps(f.margin, f.pace, data.margin),
        total=f.total,
        total_crps=crps_normal(f.total, tuned.totals_sigma, data.total),
    )


@dataclass(frozen=True)
class Baselines:
    elo: Predictions
    b0: Predictions
    elo_params: dict[str, Any]
    b0_params: dict[str, float]
    totals_sigma: float


def baselines(games: pd.DataFrame, data: Data, spec: M1Backtest) -> Baselines:
    """The comparison Elo (re-tuned on the M1 tuning seasons) and B0, on the same games."""
    elo_spec = Backtest(spec.report, spec.warmup, spec.tuning, spec.validation, grid=spec.elo_grid)
    tuned = run_backtest(games, elo_spec)["tuned"]
    params = EloParams(k=tuned["k"], hca=tuned["hca"], reversion=tuned["reversion"])
    diffs = replay(prepare(data.games), params)
    tuning = data.split["tuning"]
    scale, sigma = float(tuned["margin_scale"]), float(tuned["margin_sigma"])
    elo_margin = diffs / scale
    fit_b0 = data.rated & (data.games["season"].to_numpy() <= spec.tuning[-1])
    fit_b0 &= ~data.games["neutral"].to_numpy()
    b0_rate, b0_margin = float(data.home_won[fit_b0].mean()), float(data.margin[fit_b0].mean())
    neutral = data.games["neutral"].to_numpy()
    seasons = data.games["season"].to_numpy()
    baseline = {int(s): totals_baseline(games, int(s)) for s in np.unique(seasons)}
    total = np.array([baseline[int(s)] or np.nan for s in seasons], dtype=np.float64)
    totals_sigma = _round(np.sqrt(np.nanmean((data.total[tuning] - total[tuning]) ** 2)))
    total_crps = crps_normal(total, totals_sigma, data.total)
    b0_p = np.where(neutral, 0.5, b0_rate)
    b0_m = np.where(neutral, 0.0, b0_margin)
    return Baselines(
        elo=Predictions(
            p_home=win_probabilities(diffs),
            margin=elo_margin,
            margin_crps=crps_normal(elo_margin, sigma, data.margin),
            total=total,
            total_crps=total_crps,
        ),
        b0=Predictions(
            p_home=b0_p,
            margin=b0_m,
            margin_crps=crps_normal(b0_m, sigma, data.margin),
            total=total,
            total_crps=total_crps,
        ),
        elo_params={
            "k": params.k,
            "hca": params.hca,
            "reversion": params.reversion,
            "margin_scale": scale,
            "margin_sigma": sigma,
            "tuning_log_loss": tuned["tuning_log_loss"],
            "grid_size": len(spec.elo_grid.k)
            * len(spec.elo_grid.hca)
            * len(spec.elo_grid.reversion),
            "best_on_edge": grid_edges(spec.elo_grid, params),
        },
        b0_params={"home_win_rate": _round(b0_rate), "home_margin": _round(b0_margin)},
        totals_sigma=totals_sigma,
    )


def _comparison(m1: Predictions, elo: Predictions, data: Data, mask: BoolArray) -> dict[str, Any]:
    loss = per_game_log_loss(m1.p_home[mask], data.home_won[mask]) - per_game_log_loss(
        elo.p_home[mask], data.home_won[mask]
    )
    return {
        "log_loss_m1_minus_elo": _ci(loss),
        "brier_m1_minus_elo": _ci(
            (m1.p_home[mask] - data.home_won[mask]) ** 2
            - (elo.p_home[mask] - data.home_won[mask]) ** 2
        ),
        "margin_crps_m1_minus_elo": _ci(m1.margin_crps[mask] - elo.margin_crps[mask]),
        "totals_crps_m1_minus_elo": _ci(m1.total_crps[mask] - elo.total_crps[mask]),
    }


def _segments(m1: Predictions, elo: Predictions, data: Data, mask: BoolArray) -> dict[str, Any]:
    """Where M1 wins or loses against Elo: lopsided games, early rounds, phase."""
    games = data.games
    favourite = np.maximum(elo.p_home, 1.0 - elo.p_home)
    early = (games["phase"] == "RS").to_numpy() & (games["round"] <= EARLY_ROUNDS).to_numpy()
    groups = {
        "lopsided": favourite >= LOPSIDED,
        "balanced": favourite < LOPSIDED,
        "early_rounds": early,
        "later_regular_season": (games["phase"] == "RS").to_numpy() & ~early,
        "postseason": (games["phase"] != "RS").to_numpy(),
    }
    out = {}
    for name, group in groups.items():
        sel = mask & group
        n = int(sel.sum())
        out[name] = {
            "n": n,
            "m1_log_loss": _round(per_game_log_loss(m1.p_home[sel], data.home_won[sel]).mean())
            if n
            else None,
            "elo_log_loss": _round(per_game_log_loss(elo.p_home[sel], data.home_won[sel]).mean())
            if n
            else None,
            "m1_margin_crps": _round(m1.margin_crps[sel].mean()) if n else None,
            "elo_margin_crps": _round(elo.margin_crps[sel].mean()) if n else None,
        }
    return out


def heteroscedasticity(forecast: Forecast, data: Data, mask: BoolArray) -> dict[str, Any]:
    """R4: does the squared margin residual grow with expected pace or with the rating gap?

    OLS of r^2 on [1, P - mean P, |m| - mean |m|] with HC0 (White) standard errors, plus the
    Normal log-likelihood gain of a pace-scaled over a constant sigma (a likelihood ratio).
    """
    r2 = (data.margin[mask] - forecast.margin[mask]) ** 2
    pace = forecast.pace[mask] - forecast.pace[mask].mean()
    gap = np.abs(forecast.margin[mask]) - np.abs(forecast.margin[mask]).mean()
    x = np.column_stack([np.ones(len(r2)), pace, gap])
    xtx_inv = np.linalg.inv(x.T @ x)
    beta = xtx_inv @ x.T @ r2
    resid = r2 - x @ beta
    cov = xtx_inv @ (x.T * resid**2) @ x @ xtx_inv
    se = np.sqrt(np.diag(cov))

    def row(i: int) -> dict[str, float]:
        z = float(beta[i] / se[i])
        return {
            "coef": _round(beta[i]),
            "se": _round(se[i]),
            "z": _round(z),
            "p_value": _round(math.erfc(abs(z) / math.sqrt(2.0))),
        }

    raw = data.margin[mask] - forecast.margin[mask]
    sigma = float(np.sqrt(np.mean(raw**2)))
    rel = forecast.pace[mask] / forecast.pace[mask].mean()
    sigma0 = float(np.sqrt(np.mean(raw**2 / rel)))

    def normal_ll(scales: FloatArray) -> float:
        return float(np.sum(-np.log(scales) - 0.5 * (raw / scales) ** 2))

    lr = 2.0 * (normal_ll(sigma0 * np.sqrt(rel)) - normal_ll(np.full(len(raw), sigma)))
    return {
        "n": int(mask.sum()),
        "regression": "r^2 ~ 1 + (expected pace - mean) + (|expected margin| - mean), HC0 SE",
        "intercept": row(0),
        "expected_pace": row(1),
        "abs_expected_margin": row(2),
        "pace_scaled_sigma_log_likelihood_ratio": _round(lr),
        "pace_scaled_sigma_p_value": _round(math.erfc(math.sqrt(max(lr, 0.0) / 2.0))),
    }


def model_version(tuned: Tuned) -> str:
    """Short hash of every tuned value the live predictor reads, plus the package version."""
    key = json.dumps(tuned_block(tuned), sort_keys=True)
    return f"{version('eurohoops')}+m1.{hashlib.sha256(key.encode()).hexdigest()[:8]}"


def tuned_block(tuned: Tuned) -> dict[str, Any]:
    model = tuned.margin_models[tuned.variant]
    return {
        "rating": asdict(tuned.rating),
        "pace": asdict(tuned.pace),
        "margin": {
            "variant": model.variant,
            "scale": model.scale,
            "df": model.df,
            "ref_pace": model.ref_pace,
        },
        "totals_sigma": tuned.totals_sigma,
    }


def run_m1_backtest(
    games: pd.DataFrame,
    team_games: pd.DataFrame,
    spec: M1Backtest,
    *,
    score_test: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """The M1 report; the test split is filled only with ``score_test``."""
    say = progress or (lambda _: None)
    data = prepare_data(games, team_games, spec)
    say("tuning M1 (pace and rating grids)")
    tuned = tune(data, spec)
    say("comparison Elo and B0")
    base = baselines(games, data, spec)
    splits = ("tuning", "validation", *(("test",) if score_test else ()))
    chosen = m1_predictions(tuned, tuned.variant, data)
    metrics = {
        split: {
            "m1": _metrics(chosen, data.split[split], data.margin, data.total),
            "elo": _metrics(base.elo, data.split[split], data.margin, data.total),
            "b0": _metrics(base.b0, data.split[split], data.margin, data.total),
        }
        for split in splits
    }
    variants = {}
    for variant in VARIANTS:
        pred = m1_predictions(tuned, variant, data)
        model = tuned.margin_models[variant]
        variants[variant] = {
            "declared": True,
            "post_hoc": False,
            "chosen": variant == tuned.variant,
            "df": model.df,
            "scale": model.scale,
            "ref_pace": model.ref_pace,
            **{
                split: {
                    key: _metrics(pred, data.split[split], data.margin, data.total)[key]
                    for key in ("n", "log_loss", "brier", "ece", "margin_crps")
                }
                for split in splits
            },
            "validation_vs_elo": _comparison(pred, base.elo, data, data.split["validation"]),
        }
    gate = _comparison(chosen, base.elo, data, data.split["validation"])
    validation = metrics["validation"]
    passed = validation["m1"]["log_loss"] < validation["elo"]["log_loss"]
    seasons = data.games["season"]
    report: dict[str, Any] = {
        "model": "m1",
        "model_version": model_version(tuned),
        "seasons": {
            "warmup": list(spec.warmup),
            "tuning": list(spec.tuning),
            "validation": list(spec.validation),
            "test": list(spec.test),
        },
        "test_scored": score_test,
        "data_sha256": data.snapshot,
        "games_per_season": {
            str(s): int(n) for s, n in seasons[data.rated].value_counts().sort_index().items()
        },
        "games_without_box_lines": {
            split: int((data.split[split] & ~data.has_rows).sum()) for split in SPLITS
        },
        "grid": tuned.grids,
        "tuned": {**tuned_block(tuned), "variant": tuned.variant},
        "comparison_elo": base.elo_params,
        "b0": base.b0_params,
        "totals_baseline_sigma": base.totals_sigma,
        "metrics": metrics,
        "variants": variants,
        "gate": {
            "rule": "M1 validation log loss < comparison-Elo validation log loss",
            "variant": tuned.variant,
            "m1_log_loss": validation["m1"]["log_loss"],
            "elo_log_loss": validation["elo"]["log_loss"],
            "passed": bool(passed),
            **gate,
        },
        "validation_segments": _segments(chosen, base.elo, data, data.split["validation"]),
        "heteroscedasticity_tuning": heteroscedasticity(tuned.forecast, data, data.split["tuning"]),
    }
    if score_test:
        report["test_vs_elo"] = _comparison(chosen, base.elo, data, data.split["test"])
        report["test_segments"] = _segments(chosen, base.elo, data, data.split["test"])
    return report


def format_m1_table(report: dict[str, Any]) -> str:
    header = (
        f"{'split':<11}{'model':<5}{'n':>5}{'logloss':>9}{'brier':>8}{'acc':>7}{'ece':>7}"
        f"{'mCRPS':>7}{'tMAE':>7}{'tCRPS':>7}"
    )
    lines = [header, "-" * len(header)]
    for split, models in report["metrics"].items():
        for name, m in models.items():
            lines.append(
                f"{split:<11}{name:<5}{m['n']:>5}{m['log_loss']:>9.4f}{m['brier']:>8.4f}"
                f"{m['accuracy']:>7.3f}{m['ece']:>7.3f}{m['margin_crps']:>7.2f}"
                f"{m['totals_mae']:>7.2f}{m['totals_crps']:>7.2f}"
            )
    gate = report["gate"]
    ci = gate["log_loss_m1_minus_elo"]
    lines.append(
        f"gate ({gate['variant']}): M1 - Elo validation log loss {ci['mean']:+.4f} "
        f"95% CI [{ci['ci95'][0]:+.4f}, {ci['ci95'][1]:+.4f}] -> "
        + ("PASS" if gate["passed"] else "FAIL")
    )
    return "\n".join(lines)
