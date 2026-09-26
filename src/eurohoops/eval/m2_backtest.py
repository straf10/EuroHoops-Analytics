"""M2 backtest (F5): leave-one-season-out CV on development seasons, validation, then test.

Declared variants (F-c; reports/week7-10_progress.md): ``spline`` (knots x L2 grid chosen by
pooled LOSO CV log loss), ``spline_iso``, ``lgbm`` (the Optuna-tuned parameters stored in this
report) and ``lgbm_iso``. Isotonic calibration never sees the season it calibrates: for
development season s it is fitted on predictions for the other development seasons t made by
models trained without s *and* t (leave-two-seasons-out), so a season-s out-of-fold value
depends on no season-s shot; validation and test are calibrated on the development
out-of-fold predictions. LightGBM variants use the mean prediction of five seeds (F-l); the
per-seed numbers are kept.

The gate (exit-gate items 1-2): the best-on-CV challenger vs the best-on-CV baseline on
validation log loss, with game-level bootstrap CIs (F-g); the chosen M2 (the challenger if it
wins, else the spline baseline) must meet F-f.
"""

import hashlib
import json
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from eurohoops.config import M2Seasons
from eurohoops.eval.shot_metrics import (
    calibrated,
    cluster_bootstrap,
    ece,
    ece_diff_bootstrap,
    scores,
    shot_log_loss,
)
from eurohoops.models.elo import FloatArray
from eurohoops.models.feature_audit import outcome_coded_levels
from eurohoops.models.season_level import level_predictions
from eurohoops.models.xpts import fit_isotonic, fit_spline
from eurohoops.models.xpts_gbm import N_TRIALS, STUDY_SEED, fit_gbm, run_study
from eurohoops.parse.shot_table import BANDS

SEEDS = tuple(range(20261001, 20261006))
SPLINE_KNOTS = (4, 6, 8)
SPLINE_L2 = (1e-5, 1e-4, 1e-3)
VARIANTS = ("spline", "spline_iso", "lgbm", "lgbm_iso")
BASELINES = ("spline", "spline_iso")
CHALLENGERS = ("lgbm", "lgbm_iso")
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 20261001
END_OF_PERIOD_S = 24  # the last shot-clock of a period

Predictor = Callable[[pd.DataFrame], FloatArray]
Fitter = Callable[[pd.DataFrame], Predictor]
Progress = Callable[[str], None]


def _labels(shots: pd.DataFrame) -> FloatArray:
    return shots["made"].to_numpy(dtype=np.float64)


def _r(value: float) -> float:
    return round(float(value), 6)


LATER = -1  # target "the later seasons" in a fit job
FIT_WORKERS = 2  # LightGBM fits in parallel processes of NUM_THREADS each (12 logical cores)
Job = tuple[tuple[int, ...], tuple[int, ...]]  # (seasons left out, seasons / LATER predicted)


def fold_jobs(development: Sequence[int], pairs: bool, has_later: bool) -> list[Job]:
    """Every fit of a Folds: LOSO, then the leave-two-out pairs, then the development fit."""
    jobs: list[Job] = [((s,), (s,)) for s in development]
    if pairs:
        jobs += [((s, t), (t, s)) for s, t in combinations(development, 2)]
    if has_later:
        jobs.append(((), (LATER,)))
    return jobs


def run_job(
    fit: Fitter, shots: pd.DataFrame, later: pd.DataFrame, development: Sequence[int], job: Job
) -> list[FloatArray]:
    """Fit on the development seasons without ``job[0]``; predict each target of ``job[1]``."""
    season = shots["season"].to_numpy()
    keep = np.isin(season, development)
    for s in job[0]:
        keep &= season != s
    model = fit(shots.iloc[np.flatnonzero(keep)])
    return [
        model(later) if t == LATER else model(shots.iloc[np.flatnonzero(season == t)])
        for t in job[1]
    ]


_WORKER: dict[str, Any] = {}


def _init_worker(shots: pd.DataFrame, later: pd.DataFrame, development: tuple[int, ...]) -> None:
    _WORKER.update(shots=shots, later=later, development=development)


def _worker_job(task: tuple[Fitter, Job]) -> list[FloatArray]:
    fit, job = task
    return run_job(fit, _WORKER["shots"], _WORKER["later"], _WORKER["development"], job)


class FitPool:
    """Worker processes that hold the development and later shots once and run fit jobs.

    LightGBM with ``deterministic=True`` and a fixed ``num_threads`` gives byte-identical
    predictions whether fits run one at a time or in parallel processes (verified,
    reports/week7-10b_progress.md; ``test_parallel_fits_equal_sequential_fits``), so this is a
    number-preserving speed-up."""

    def __init__(
        self, shots: pd.DataFrame, later: pd.DataFrame, development: Sequence[int], workers: int
    ) -> None:
        self.shots, self.later = shots, later
        self.executor = ProcessPoolExecutor(
            workers, initializer=_init_worker, initargs=(shots, later, tuple(development))
        )

    def run(self, fit: Fitter, jobs: Sequence[Job]) -> list[list[FloatArray]]:
        return list(self.executor.map(_worker_job, [(fit, job) for job in jobs]))

    def __enter__(self) -> "FitPool":
        return self

    def __exit__(self, *_: object) -> None:
        self.executor.shutdown()


class Folds:
    """Out-of-fold predictions of one fitter: LOSO on development, leave-two-out pairs for the
    nested isotonic fit, and a development-wide fit for the later splits. With a ``pool`` the
    fits run in its worker processes (same jobs, same results)."""

    def __init__(
        self,
        shots: pd.DataFrame,
        development: Sequence[int],
        fit: Fitter,
        pairs: bool,
        later: pd.DataFrame,
        *,
        pool: FitPool | None = None,
    ) -> None:
        clock = time.perf_counter()
        season = shots["season"].to_numpy()
        self.index = {s: np.flatnonzero(season == s) for s in development}
        jobs = fold_jobs(development, pairs, bool(len(later)))
        if pool is None:
            results = [run_job(fit, shots, later, development, job) for job in jobs]
        else:
            if pool.shots is not shots or pool.later is not later:
                raise ValueError("the pool holds other shots than this Folds")
            results = pool.run(fit, jobs)
        self.oof = np.full(len(shots), np.nan)
        self.pair: dict[tuple[int, int], FloatArray] = {}  # (left out s, t) -> preds on t
        self.later = np.empty(0)
        for (left_out, targets), predictions in zip(jobs, results, strict=True):
            if targets == (LATER,):
                self.later = predictions[0]
            elif len(left_out) == 1:
                self.oof[self.index[left_out[0]]] = predictions[0]
            else:
                s, t = left_out
                self.pair[(s, t)], self.pair[(t, s)] = predictions
        self.seconds = {"fits": time.perf_counter() - clock, "n": float(len(jobs))}

    def timing(self) -> str:
        return f"{self.seconds['n']:.0f} fits, {self.seconds['fits']:.0f} s"


def mean_folds(folds: Sequence[Folds]) -> Folds:
    """The seed-mean predictions of several fits of one configuration."""
    out = object.__new__(Folds)
    out.index = folds[0].index
    out.oof = np.mean([f.oof for f in folds], axis=0)
    out.pair = {k: np.mean([f.pair[k] for f in folds], axis=0) for k in folds[0].pair}
    out.later = np.mean([f.later for f in folds], axis=0)
    out.seconds = {}
    return out


def calibrate(folds: Folds, y: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Isotonic-calibrated (out-of-fold development, later) predictions; see the module doc."""
    oof = np.full_like(folds.oof, np.nan)
    for s, rows in folds.index.items():
        others = [t for t in folds.index if t != s]
        p = np.concatenate([folds.pair[(s, t)] for t in others])
        target = np.concatenate([y[folds.index[t]] for t in others])
        oof[rows] = fit_isotonic(p, target)(folds.oof[rows])
    dev_rows = np.concatenate(list(folds.index.values()))
    later = fit_isotonic(folds.oof[dev_rows], y[dev_rows])(folds.later)
    return oof, later


def _scores_only(p: FloatArray, y: FloatArray) -> dict[str, Any]:
    return {k: v for k, v in scores(p, y).items() if k != "reliability"}


def contexts(shots: pd.DataFrame) -> dict[str, npt.NDArray[np.bool_]]:
    """Shot-context splits for the breakdowns (the feed's fastbreak/second-chance/off-turnover
    flags are outcome-coded, so they are not used): home, the last 24 s of a period, overtime."""
    return {
        "home": shots["home"].to_numpy(dtype=bool),
        "end_of_period": shots["seconds_left"].to_numpy() <= END_OF_PERIOD_S,
        "overtime": shots["period"].to_numpy() >= 5,
    }


def split_scores(p: FloatArray, shots: pd.DataFrame) -> dict[str, Any]:
    """Overall scores with reliability, per distance band and shot type (with reliability),
    and per shot context."""
    y = _labels(shots)
    band = shots["band"].to_numpy()
    value = shots["value"].to_numpy()
    flags = {
        name: {"on": _scores_only(p[on], y[on]), "off": _scores_only(p[~on], y[~on])}
        for name, on in contexts(shots).items()
    }
    return {
        **scores(p, y),
        "by_band": {b: scores(p[band == b], y[band == b]) for b in BANDS},
        "by_type": {f"{v}pt": scores(p[value == v], y[value == v]) for v in (2, 3)},
        "by_context": flags,
    }


def spline_fitter(n_knots: int, l2: float) -> Fitter:
    return lambda train: fit_spline(train, n_knots, l2).predict


@dataclass(frozen=True)
class GbmFitter:
    """A LightGBM fitter that can be sent to a worker process (a lambda cannot)."""

    params: dict[str, Any]
    seed: int

    def __call__(self, train: pd.DataFrame) -> Predictor:
        return fit_gbm(train, self.params, self.seed).predict


def gbm_fitter(params: dict[str, Any], seed: int) -> Fitter:
    return GbmFitter(params, seed)


def cv_log_loss(folds: Folds, y: FloatArray) -> float:
    rows = np.concatenate(list(folds.index.values()))
    return float(shot_log_loss(folds.oof[rows], y[rows]).mean())


def study_objective(
    shots: pd.DataFrame, development: Sequence[int]
) -> Callable[[dict[str, Any]], float]:
    """Pooled LOSO CV log loss on development seasons (LightGBM seed 20261001), per F-d."""
    dev = shots[shots["season"].isin(development)].reset_index(drop=True)
    y = _labels(dev)

    def objective(params: dict[str, Any]) -> float:
        folds = Folds(dev, development, gbm_fitter(params, SEEDS[0]), False, dev.iloc[:0])
        return cv_log_loss(folds, y)

    return objective


def search(
    shots: pd.DataFrame,
    development: Sequence[int],
    n_trials: int = N_TRIALS,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Run the declared Optuna study; its trials and best parameters, JSON-ready."""
    study = run_study(study_objective(shots, development), n_trials, STUDY_SEED, progress)
    return {
        "sampler": "TPESampler",
        "seed": STUDY_SEED,
        "n_trials": n_trials,
        "objective": "pooled LOSO CV log loss on development seasons, LightGBM seed 20261001",
        "development": list(development),
        "best_trial": study.best_trial.number,
        "best_value": round(float(study.best_value), 8),
        "best_params": study.best_params,
        "trials": [
            {"number": t.number, "value": round(float(t.value), 8), "params": t.params}
            for t in study.trials
        ],
    }


def data_sha256(shots: pd.DataFrame) -> str:
    text = shots.sort_values(["game_id", "event"]).to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(text.encode()).hexdigest()


def model_version(spline: dict[str, Any], params: dict[str, Any]) -> str:
    payload = json.dumps({"spline": spline, "lgbm": params, "seeds": SEEDS}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:8]


def _ece_diff_ci(
    p_a: FloatArray, p_b: FloatArray, y: FloatArray, games: npt.NDArray[Any]
) -> tuple[float, float]:
    """95% game-level bootstrap CI of ECE(a) - ECE(b)."""
    codes, inverse = np.unique(games, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    starts = np.searchsorted(inverse[order], np.arange(len(codes) + 1))
    members = [order[starts[g] : starts[g + 1]] for g in range(len(codes))]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    stats = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        rows = np.concatenate([members[g] for g in rng.integers(0, len(codes), len(codes))])
        stats.append(ece(p_a[rows], y[rows]) - ece(p_b[rows], y[rows]))
    low, high = np.percentile(stats, [2.5, 97.5])
    return float(low), float(high)


def _diff_ci(diff: FloatArray, games: npt.NDArray[Any]) -> dict[str, Any]:
    mean, low, high = cluster_bootstrap(diff, games, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED)
    return {"n": len(diff), "mean": _r(mean), "ci95": [_r(low), _r(high)]}


def gate(
    variants: dict[str, dict[str, Any]],
    split: str,
    preds: dict[str, FloatArray],
    per_seed: dict[str, dict[int, FloatArray]],
    shots: pd.DataFrame,
) -> dict[str, Any]:
    """Exit-gate items 1-2 on one held-out split, applied literally: the best-on-CV challenger
    vs the best-on-CV baseline by log loss, and F-f on the chosen model."""
    cv = {v: variants[v]["cv"]["log_loss"] for v in VARIANTS}
    challenger = min(CHALLENGERS, key=lambda v: cv[v])
    baseline = min(BASELINES, key=lambda v: cv[v])
    y = _labels(shots)
    games = shots["game_id"].to_numpy()
    base_loss = shot_log_loss(preds[baseline], y)
    diff = shot_log_loss(preds[challenger], y) - base_loss
    brier = (preds[challenger] - y) ** 2 - (preds[baseline] - y) ** 2
    ll = _diff_ci(diff, games)
    beats = ll["mean"] < 0.0
    chosen = challenger if beats else baseline
    flips = {
        str(seed): bool((float(shot_log_loss(p, y).mean()) < float(base_loss.mean())) != beats)
        for seed, p in per_seed[challenger].items()
    }
    groups: dict[str, dict[str, npt.NDArray[np.bool_]]] = {
        "by_band": {b: shots["band"].to_numpy() == b for b in BANDS},
        "by_type": {f"{v}pt": shots["value"].to_numpy() == v for v in (2, 3)},
    }
    for name, on in contexts(shots).items():
        groups[f"by_{name}"] = {"on": on, "off": ~on}
    breakdown = {
        name: {key: _diff_ci(diff[mask], games[mask]) for key, mask in masks.items() if mask.any()}
        for name, masks in groups.items()
    }
    ece_low, ece_high = _ece_diff_ci(preds[challenger], preds[baseline], y, games)
    ok = calibrated(variants[chosen][split])
    return {
        "split": split,
        "challenger": challenger,
        "baseline": baseline,
        "chosen": chosen,
        "log_loss_challenger_minus_baseline": ll,
        "brier_challenger_minus_baseline": _diff_ci(brier, games),
        "ece_challenger_minus_baseline": {
            "mean": _r(variants[challenger][split]["ece"] - variants[baseline][split]["ece"]),
            "ci95": [_r(ece_low), _r(ece_high)],
        },
        "single_seed_would_flip": flips,
        "any_seed_flips": any(flips.values()),
        "diff_breakdown": breakdown,
        "calibrated": ok,
        "beats_baseline": bool(beats),
        "passed": bool(ok and beats),
    }


LEVEL_BASES = {"lgbm_level": "lgbm", "spline_level": "spline"}
POST_HOC = "post-hoc, not a clean hold-out"
LEVEL_DECLARATION = "reports/week7-10b_progress.md, LEVEL_DECLARATION"


def calibration_in_the_large(p: FloatArray, shots: pd.DataFrame) -> dict[str, float]:
    """Per season: sum of xPTS over actual FG points."""
    value = shots["value"].to_numpy(dtype=np.float64)
    frame = pd.DataFrame(
        {"season": shots["season"].to_numpy(), "x": p * value, "a": _labels(shots) * value}
    )
    sums = frame.groupby("season")[["x", "a"]].sum()
    return {str(s): _r(row["x"] / row["a"]) for s, row in sums.iterrows()}


def _vs_base(p: FloatArray, base: FloatArray, shots: pd.DataFrame) -> dict[str, Any]:
    """Level variant minus its base: log loss and Brier with game-level bootstrap CIs, and
    the ECE difference with its CI (games resampled as weights)."""
    y = _labels(shots)
    games = shots["game_id"].to_numpy()
    diff, low, high = ece_diff_bootstrap(
        p, base, y, games, resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED
    )
    return {
        "log_loss": _diff_ci(shot_log_loss(p, y) - shot_log_loss(base, y), games),
        "brier": _diff_ci((p - y) ** 2 - (base - y) ** 2, games),
        "ece": {"mean": _r(diff), "ci95": [_r(low), _r(high)]},
    }


def level_variants(
    bases: dict[str, Folds],
    dev: pd.DataFrame,
    later: pd.DataFrame,
    val: npt.NDArray[np.bool_],
    *,
    tipoff: pd.Series,
    development: Sequence[int],
    score_test: bool,
) -> dict[str, Any]:
    """The declared season-level variants (G5), every number post-hoc (see LEVEL_DECLARATION)."""
    y_dev, y_later = _labels(dev), _labels(later)
    ticks = pd.to_datetime(tipoff, utc=True).dt.tz_convert(None).astype("int64")
    tip_dev = np.asarray(dev["game_id"].map(ticks), dtype=np.int64)
    tip_later = np.asarray(later["game_id"].map(ticks), dtype=np.int64)
    later_season: npt.NDArray[np.int64] = later["season"].to_numpy(dtype=np.int64)
    out: dict[str, Any] = {}
    for name, base in LEVEL_BASES.items():
        folds = bases[base]
        lp = level_predictions(
            folds,
            y_dev,
            tip_dev,
            later_season=later_season,
            y_later=y_later,
            tip_later=tip_later,
            development=development,
        )
        block: dict[str, Any] = {
            "base": base,
            "post_hoc": True,
            "label": POST_HOC,
            "cv": split_scores(lp.oof, dev),
            "cv_per_season": {
                str(s): _r(shot_log_loss(lp.oof[rows], y_dev[rows]).mean())
                for s, rows in folds.index.items()
            },
            "validation": split_scores(lp.later[val], later[val]),
            "test": split_scores(lp.later[~val], later[~val]) if score_test else None,
            "calibration_in_the_large": {
                "level": calibration_in_the_large(np.r_[lp.oof, lp.later], pd.concat([dev, later])),
                "base": calibration_in_the_large(
                    np.r_[folds.oof, folds.later], pd.concat([dev, later])
                ),
            },
            "minus_base": {
                "cv": _vs_base(lp.oof, folds.oof, dev),
                "validation": _vs_base(lp.later[val], folds.later[val], later[val]),
                "test": _vs_base(lp.later[~val], folds.later[~val], later[~val])
                if score_test
                else None,
            },
            "shrinkage": lp.shrinkage,
            "priors": lp.priors,
        }
        block["meets_f_f"] = {
            "validation": calibrated(block["validation"]),
            "test": calibrated(block["test"]) if score_test else None,
        }
        out[name] = block
    lgbm_cv = float(shot_log_loss(bases["lgbm"].oof, y_dev).mean())
    level = out["lgbm_level"]
    condition = {
        "meets_f_f_on_validation": level["meets_f_f"]["validation"],
        "cv_log_loss_below_lgbm": bool(level["cv"]["log_loss"] < _r(lgbm_cv)),
    }
    return {
        "declaration": LEVEL_DECLARATION,
        "post_hoc": True,
        "label": POST_HOC,
        "variants": out,
        "g_g_condition": {**condition, "holds": all(condition.values())},
    }


class Lap:
    """Wall time of each backtest phase, reported as ``TIMING <phase>: <s> s`` progress lines."""

    def __init__(self, progress: Progress) -> None:
        self.progress = progress
        self.clock = time.perf_counter()

    def __call__(self, label: str) -> None:
        now = time.perf_counter()
        self.progress(f"TIMING {label}: {now - self.clock:.1f} s")
        self.clock = now


def spline_search(
    dev: pd.DataFrame,
    later: pd.DataFrame,
    development: Sequence[int],
    progress: Progress,
    lap: Lap,
) -> tuple[list[dict[str, Any]], dict[str, Any], Folds]:
    """The declared knots x L2 grid by pooled LOSO CV log loss (ties: fewer knots, smaller
    L2), then the chosen configuration's folds with the leave-two-out pairs."""
    y_dev = _labels(dev)
    grid = []
    for n_knots in SPLINE_KNOTS:
        for l2 in SPLINE_L2:
            folds = Folds(dev, development, spline_fitter(n_knots, l2), False, later[:0])
            grid.append({"knots": n_knots, "l2": l2, "cv_log_loss": _r(cv_log_loss(folds, y_dev))})
            progress(f"spline knots {n_knots} l2 {l2}: CV log loss {grid[-1]['cv_log_loss']}")
            lap(f"spline grid knots {n_knots} l2 {l2} (12 LOSO fits)")
    best = min(grid, key=lambda g: (g["cv_log_loss"], g["knots"], g["l2"]))
    spline = Folds(
        dev, development, spline_fitter(int(best["knots"]), float(best["l2"])), True, later
    )
    progress(f"spline knots {best['knots']} l2 {best['l2']}: leave-two-out fits done")
    lap(f"spline chosen ({spline.timing()})")
    return grid, best, spline


def seed_robustness(
    seeds: dict[int, Folds], y_dev: FloatArray, y_later: FloatArray, val: npt.NDArray[np.bool_]
) -> tuple[dict[str, dict[int, FloatArray]], dict[str, Any]]:
    """F-l: each seed's held-out predictions (raw and isotonic) and its CV and validation log
    loss, with the mean and sd over seeds."""
    seed_held: dict[str, dict[int, FloatArray]] = {"lgbm": {}, "lgbm_iso": {}}
    per_seed: dict[str, Any] = {}
    for seed, folds in seeds.items():
        iso_oof, iso_later = calibrate(folds, y_dev)
        seed_held["lgbm"][seed], seed_held["lgbm_iso"][seed] = folds.later, iso_later
        per_seed[str(seed)] = {
            variant: {
                "cv_log_loss": _r(shot_log_loss(o, y_dev).mean()),
                "validation_log_loss": _r(shot_log_loss(h[val], y_later[val]).mean()),
            }
            for variant, o, h in (
                ("lgbm", folds.oof, folds.later),
                ("lgbm_iso", iso_oof, iso_later),
            )
        }
    summary = {
        variant: {
            key: {
                "mean_of_seeds": _r(np.mean([per_seed[str(s)][variant][key] for s in seeds])),
                "sd": _r(np.std([per_seed[str(s)][variant][key] for s in seeds], ddof=1)),
            }
            for key in ("cv_log_loss", "validation_log_loss")
        }
        for variant in CHALLENGERS
    }
    return seed_held, {"seeds": list(seeds), "per_seed": per_seed, "summary": summary}


def run_m2_backtest(
    shots: pd.DataFrame,
    seasons: M2Seasons,
    study: dict[str, Any],
    score_test: bool,
    progress: Progress,
    *,
    tipoff: pd.Series,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """The report and the chosen M2's out-of-fold P(make) per shot (development LOSO,
    validation, and test when scored). ``study`` is the stored Optuna result; ``tipoff`` maps
    game_id to tip-off time (the season-level variants use earlier games only)."""
    lap = Lap(progress)
    used = shots[shots["validated_season"]]
    dev = used[used["season"].isin(seasons.development)].reset_index(drop=True)
    coded = outcome_coded_levels(dev)
    if coded:
        raise ValueError(f"outcome-coded M2 feature levels on development shots: {coded}")
    lap("split + outcome-coding audit")
    later_seasons = [*seasons.validation, *(seasons.test if score_test else ())]
    later = used[used["season"].isin(later_seasons)].reset_index(drop=True)
    y_dev, y_later = _labels(dev), _labels(later)
    val = later["season"].isin(seasons.validation).to_numpy()
    params = study["best_params"]

    grid, best, spline = spline_search(dev, later, seasons.development, progress, lap)
    seeds: dict[int, Folds] = {}
    with FitPool(dev, later, seasons.development, FIT_WORKERS) as pool:
        for seed in SEEDS:
            fit = gbm_fitter(params, seed)
            seeds[seed] = Folds(dev, seasons.development, fit, True, later, pool=pool)
            progress(f"lgbm seed {seed}: done")
            lap(f"lgbm seed {seed} ({seeds[seed].timing()}, {FIT_WORKERS} processes)")
    gbm = mean_folds(list(seeds.values()))

    oof: dict[str, FloatArray] = {"spline": spline.oof, "lgbm": gbm.oof}
    held: dict[str, FloatArray] = {"spline": spline.later, "lgbm": gbm.later}
    oof["spline_iso"], held["spline_iso"] = calibrate(spline, y_dev)
    oof["lgbm_iso"], held["lgbm_iso"] = calibrate(gbm, y_dev)
    lap("isotonic calibration, spline + lgbm seed mean")
    seed_held, robustness = seed_robustness(seeds, y_dev, y_later, val)
    lap("per-seed isotonic calibration and scores")

    variants: dict[str, Any] = {}
    for variant in VARIANTS:
        variants[variant] = {
            "post_hoc": False,
            "cv": split_scores(oof[variant], dev),
            "cv_per_season": {
                str(s): _r(shot_log_loss(oof[variant][rows], y_dev[rows]).mean())
                for s, rows in spline.index.items()
            },
            "validation": split_scores(held[variant][val], later[val]),
            "test": split_scores(held[variant][~val], later[~val]) if score_test else None,
        }
    lap("variant scores (overall, bands, types, contexts)")
    verdict = gate(
        variants,
        "validation",
        {v: held[v][val] for v in VARIANTS},
        {v: {s: p[val] for s, p in seed_held[v].items()} for v in CHALLENGERS},
        later[val],
    )
    lap("gate on validation (bootstraps)")
    test_gate = (
        gate(
            variants,
            "test",
            {v: held[v][~val] for v in VARIANTS},
            {v: {s: p[~val] for s, p in seed_held[v].items()} for v in CHALLENGERS},
            later[~val],
        )
        if score_test
        else None
    )
    lap("gate on test (bootstraps)")
    levels = level_variants(
        {"lgbm": gbm, "spline": spline},
        dev,
        later,
        val,
        tipoff=tipoff,
        development=seasons.development,
        score_test=score_test,
    )
    lap("season-level variants (offsets, scores, bootstraps)")
    chosen = verdict["chosen"]
    keys = ["game_id", "event", "season"]
    xpts = pd.concat(
        [
            dev[keys].assign(split="development", p_make=oof[chosen]),
            later[keys].assign(split=np.where(val, "validation", "test"), p_make=held[chosen]),
        ],
        ignore_index=True,
    ).assign(variant=chosen)
    report = {
        "model_version": model_version(best, params),
        "seasons": {
            "development": list(seasons.development),
            "validation": list(seasons.validation),
            "test": list(seasons.test),
        },
        "data_sha256": data_sha256(pd.concat([dev, later], ignore_index=True)),
        "shots": {"development": len(dev), "validation": int(val.sum()), "test": int((~val).sum())},
        "declared_variants": list(VARIANTS),
        "spline_grid": grid,
        "spline_chosen": {"knots": best["knots"], "l2": best["l2"]},
        "optuna_study": study,
        "lightgbm_params": params,
        "seed_robustness": robustness,
        "variants": variants,
        "gate": verdict,
        "test_scored": score_test,
        "test_gate": test_gate,
        "level_variants": levels,
    }
    lap("xPTS table + report (data hash)")
    return report, xpts
