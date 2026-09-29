"""M3 SPM box-score prior (week 9-12 H6, H-h/D2): a per-season regression of plain-``rapm``
ratings on decayed, shrunk per-100 box rates, used as the ridge target ``rapm_spm`` shrinks
toward instead of 0.

**Target and features (H-h design).** For season ``s``, the target is the plain ``rapm``
variant's O/D ratings (``earlier["rapm"]``'s chosen half-life and ridges) at the first round
cutoff of season ``s`` -- i.e. fitted on every stint strictly before season ``s`` (D2's expanding
window: never season ``s``'s own stints, stricter than the task text's leave-one-season-out).
Features are that same cutoff's decayed, shrunk per-100 rate of ``models.box_impact``'s 11 H-e
stats (``STAT_COLUMNS``) minus the league rate, reusing ``box_impact``'s own walk-forward
machinery through its public aliases (``Accumulator``, ``round_snapshots``, ``features``, ...) so
box-only's numbers and behaviour are untouched -- the rate decay uses ``rapm``'s own half-life,
the shrinkage uses ``SPM_K_MINUTES`` (declared below, not tuned: shrinking a *prior* itself, so a
second layer of tuned shrinkage on top would be circular). The weight is the player's decayed
minutes (the same ``weight`` ``box_impact``'s accumulator already tracks). Two independent
weighted ridge regressions (O, D), each with an unpenalised intercept and a shared ridge penalty
``SPM_ALPHA`` (declared, not tuned, for the same reason as ``k``) on *standardised* features
(weighted mean/std, so ``alpha`` is comparable across differently-scaled stats) -- ``SpmModel``
keeps the standardisation alongside the coefficients so ``predict`` is self-contained.

**Prior per cutoff (D2).** At every round cutoff of season ``s`` (not just its first), the prior
for player ``p``'s O/D columns is season ``s``'s ``SpmModel`` applied to ``p``'s box features as
of *that* cutoff -- fresher within-season evidence, fixed coefficients. This is wired into
``rapm.fit_walk_forward`` through its new ``prior_fn`` hook (``build_prior_fn``); a season with no
earlier rapm rating (the very first season any ``games`` frame here starts from) has no
``SpmModel``, and ``prior_fn`` returns ``None`` for it -- prior mean 0, plain ``rapm`` behaviour,
per D2's stated fallback.

**Lookup semantics (H6 done-when).** A player whose column was solved this round reports the
solved ``theta`` (unchanged from plain ``rapm``). A player whose column exists but was never
touched by a row up to this cutoff (``model.model_seen`` false) is *filled* with the prior instead
of the solver's default 0 (the posterior equals the prior with no data) -- ``fit_walk_forward``
does this substitution itself when a ``prior_fn`` is given, so ``rapm_margins``/``RatingLookup``
need no changes. A player with no box history either (missing from ``player_index``, or every box
feature 0 because his weight is 0) predicts through ``SpmModel.predict`` at a 0 feature row, which
is exactly the SPM of league-average rates (no special case needed: standardisation and the
fitted intercept do this correctly on their own).

**Reuse for GBL (H8, out of scope here).** ``SpmModel.predict`` takes a bare ``(n, n_stats)``
feature matrix, not a EuroLeague-specific frame, so H8 can build a GBL feature matrix in the same
11-stat order and reuse a fitted ``SpmModel`` directly.

**Why ``fit`` needs ``player_games`` (an ambiguity the task flagged and resolved here).**
``Variant.fit``'s call shape is ``(games, inputs, tuned)`` -- no ``Data``, so no
``Data.player_games`` -- yet rebuilding each round's box features needs the raw box lines. Rather
than widen ``Variant.fit``'s signature (shared with every other variant), ``player_games`` is
threaded through ``RapmFitInputs`` (``eval.m3_backtest``) too, alongside the ``Data`` field the
task named; see that module's ``RapmFitInputs`` docstring.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from eurohoops.models import box_impact as bi
from eurohoops.models.box_impact import STAT_COLUMNS, UNPENALISED
from eurohoops.models.elo import FloatArray
from eurohoops.models.minutes import round_cutoffs
from eurohoops.models.rapm import (
    ModelColumns,
    PriorFn,
    WalkForward,
    fit_walk_forward,
    plain_player_columns,
)

if TYPE_CHECKING:  # pragma: no cover -- typing only, avoids the runtime circular import
    from eurohoops.config import M3Backtest
    from eurohoops.eval.m3_backtest import Data, RapmFitInputs, TunedRapm

# Declared, not tuned (module docstring): shrinking a box-score rate toward the league mean to
# build a *prior* that RAPM itself then shrinks toward is already two layers of shrinkage: a
# third tuned layer (alpha) would be circular evidence-counting. 250 minutes matches
# ``box_impact.PIR_K_MINUTES``'s own declared (not grid-searched) shrinkage; alpha=1.0 is a light
# ridge on 11 standardised features (H-h: "declare alpha as a module constant ... your call").
SPM_K_MINUTES = 250.0
SPM_ALPHA = 1.0


def _round6(value: float) -> float:
    return round(float(value), 6)


def _round_dict(values: dict[str, float]) -> dict[str, float]:
    return {name: _round6(v) for name, v in values.items()}


BoolArray = npt.NDArray[np.bool_]


def _finite_rmse(pred: FloatArray, actual: FloatArray, mask: BoolArray) -> float:
    sel = mask & np.isfinite(pred)
    return float(np.sqrt(np.mean((pred[sel] - actual[sel]) ** 2))) if sel.any() else math.nan


# --- SpmModel: one season's fitted box-score SPM -------------------------------------------------


@dataclass(frozen=True)
class SpmModel:
    """A season's fitted SPM: two weighted ridge regressions (O, D) of plain-rapm ratings on
    standardised ``box_impact`` per-100 rate features, with an intercept each and a shared ridge
    ``alpha``. ``predict`` takes any matching feature matrix (module docstring: GBL reuse)."""

    season: int
    stats: tuple[str, ...]
    mean: FloatArray  # (n_stats,) weighted feature mean (standardisation)
    std: FloatArray  # (n_stats,) weighted feature std, floored away from 0
    o_intercept: float
    o_coef: FloatArray  # (n_stats,)
    d_intercept: float
    d_coef: FloatArray  # (n_stats,)
    alpha: float
    n_players: int  # players used to fit this season (real rapm target + real box features)

    def predict(self, features: FloatArray) -> tuple[FloatArray, FloatArray]:
        """``features``: ``(n, n_stats)`` raw (unstandardised) shrunk-rate-minus-league-rate
        rows, in ``self.stats`` order. A row of 0s (a player with no box history) predicts the
        SPM of league-average rates."""
        standardized = (features - self.mean) / self.std
        o: FloatArray = self.o_intercept + standardized @ self.o_coef
        d: FloatArray = self.d_intercept + standardized @ self.d_coef
        return o, d

    def to_json(self) -> dict[str, Any]:
        """JSON-ready, rounded to 6 decimals (H6: "coefficients reported")."""
        return {
            "n": self.n_players,
            "alpha": _round6(self.alpha),
            "o_intercept": _round6(self.o_intercept),
            "d_intercept": _round6(self.d_intercept),
            "o_coefficients": _round_dict(dict(zip(self.stats, self.o_coef.tolist(), strict=True))),
            "d_coefficients": _round_dict(dict(zip(self.stats, self.d_coef.tolist(), strict=True))),
            "mean": _round_dict(dict(zip(self.stats, self.mean.tolist(), strict=True))),
            "std": _round_dict(dict(zip(self.stats, self.std.tolist(), strict=True))),
        }


def _weighted_mean_std(x: FloatArray, w: FloatArray) -> tuple[FloatArray, FloatArray]:
    total = float(w.sum())
    if total <= 0.0:
        return np.zeros(x.shape[1]), np.ones(x.shape[1])
    mean = (w[:, None] * x).sum(axis=0) / total
    var = (w[:, None] * (x - mean) ** 2).sum(axis=0) / total
    std = np.sqrt(var)
    return mean, np.where(std > 1e-12, std, 1.0)


def _weighted_ridge(
    x: FloatArray, y: FloatArray, w: FloatArray, alpha: float
) -> tuple[float, FloatArray]:
    """Weighted ridge, ``x`` already standardised, intercept unpenalised (``UNPENALISED``,
    ``box_impact``'s own convention), weighted by ``sqrt(w)`` (``rapm.DecayedRidgeSparse.add``'s
    convention): normal-equations solve, matching ``box_impact._ridge_fit``'s pattern."""
    n, k = x.shape
    sw = np.sqrt(np.clip(w, 0.0, None))
    design = np.column_stack([np.ones(n), x]) * sw[:, None]
    target = y * sw
    penalty = np.concatenate([[UNPENALISED], np.full(k, alpha)])
    theta = np.linalg.solve(design.T @ design + np.diag(penalty), design.T @ target)
    return float(theta[0]), theta[1:]


def fit_season_spm(  # noqa: PLR0917 -- one weighted-ridge call per season, an exact call shape
    season: int,
    features: FloatArray,
    weight: FloatArray,
    target_o: FloatArray,
    target_d: FloatArray,
    alpha: float = SPM_ALPHA,
) -> SpmModel:
    """Fit one season's SPM (H-h): weighted ridge of ``target_o``/``target_d`` (plain-rapm O/D
    ratings at the season's cutoff) on ``features`` (this module's per-player box-rate features
    at the same cutoff), weight = decayed minutes. Standalone and pipeline-free -- the recovery
    test in ``tests/test_spm.py`` calls this directly with known synthetic coefficients."""
    mean, std = _weighted_mean_std(features, weight)
    standardized = (features - mean) / std
    o_intercept, o_coef = _weighted_ridge(standardized, target_o, weight, alpha)
    d_intercept, d_coef = _weighted_ridge(standardized, target_d, weight, alpha)
    return SpmModel(
        season=season,
        stats=STAT_COLUMNS,
        mean=mean,
        std=std,
        o_intercept=o_intercept,
        o_coef=o_coef,
        d_intercept=d_intercept,
        d_coef=d_coef,
        alpha=alpha,
        n_players=len(features),
    )


# --- Building every season's SpmModel from a games/player_games/rapm-target combination ----------


def _first_seen_time(rows: Any, columns: ModelColumns) -> FloatArray:
    """Epoch seconds a model (player) column's spell first appears in ``rows`` (rapm's base-
    column design rows, pre-sorted by time), +inf for a column no row ever touches. Rows are
    time-sorted, so the first occurrence of a column in the flattened column array is its
    earliest appearance -- a player's "seen by cutoff t" mask (H-h's SPM training set), without
    refitting RAPM once per season (D6-style but cheaper: one pass over the design rows)."""
    player_cols = columns.col_map[rows.cols[:, 2:]]  # exclude the intercept/home columns
    flat_cols = player_cols.ravel()
    flat_time = np.repeat(rows.time, player_cols.shape[1])
    first_time = np.full(columns.n_model, np.inf)
    if len(flat_cols):
        uniq, first_idx = np.unique(flat_cols, return_index=True)
        first_time[uniq] = flat_time[first_idx]
    return first_time


@dataclass(frozen=True)
class SeasonSpm:
    """Everything a per-cutoff prior needs from one ``(games, player_games, rapm target)``
    combination: every round's box feature snapshot and every fittable season's ``SpmModel``."""

    models: dict[int, SpmModel]
    snapshot_by_time: dict[float, bi.RoundSnapshot]
    player_index: dict[str, int]  # box player_id -> row position in a snapshot/feature array


def build_season_spm(
    games: pd.DataFrame,
    player_games: pd.DataFrame,
    inputs: RapmFitInputs,
    *,
    rapm_half_life_days: float,
    rapm_ridge_o: float,
    rapm_ridge_d: float,
    k: float = SPM_K_MINUTES,
    alpha: float = SPM_ALPHA,
) -> SeasonSpm:
    """Every season's SPM fittable from ``games`` (H-h + D2): season s's target is the plain
    ``rapm`` variant's O/D ratings (``rapm_half_life_days``/``rapm_ridge_o``/``rapm_ridge_d`` --
    ``earlier["rapm"]``'s chosen values) at the first round cutoff of season s, i.e. fitted on
    every stint strictly before it; features are that same cutoff's box rate features. A season
    with no earlier rapm rating (the very first season ``games`` starts from) has no model (D2:
    prior mean 0, plain rapm behaviour, left to the caller/``build_prior_fn``)."""
    columns = plain_player_columns(inputs.spell_index)
    wf = fit_walk_forward(
        games,
        inputs.rows,
        columns,
        half_life_days=rapm_half_life_days,
        ridge_o=rapm_ridge_o,
        ridge_d=rapm_ridge_d,
    )
    first_seen = _first_seen_time(inputs.rows, columns)

    box_index = bi.player_index(player_games, player_games)
    n_box_players = len(box_index)
    box_rows = bi.rows_from_player_games(games, player_games, box_index, STAT_COLUMNS)
    batches = bi.round_batches(games)
    times = [time for time, _ in batches]
    snapshots = bi.round_snapshots(
        games, box_rows, n_box_players, rapm_half_life_days, bi.RateSpec("poss", 100.0)
    )
    snapshot_by_time = dict(zip(times, snapshots, strict=True))

    box_players = sorted(box_index)  # box_index[box_players[i]] == i, by construction
    rapm_pos = {label: i for i, label in enumerate(columns.o_labels)}
    box_rapm_pos = np.array([rapm_pos.get(p, -1) for p in box_players], dtype=np.int64)

    cutoff = round_cutoffs(games).to_numpy()
    season_col = games["season"].astype(int).to_numpy()
    models: dict[int, SpmModel] = {}
    for season in sorted(set(season_col.tolist())):
        season_idx = np.flatnonzero(season_col == season)
        first_idx = int(season_idx[0])
        lookup = wf.lookups[first_idx]
        if lookup is None:
            continue  # D2's fallback: no earlier rapm rating yet
        cutoff_time = float(cutoff[first_idx])
        snapshot = snapshot_by_time.get(cutoff_time)
        if snapshot is None:
            continue
        theta_o = np.array([lookup.o.get(label, 0.0) for label in columns.o_labels])
        theta_d = np.array([lookup.d.get(label, 0.0) for label in columns.d_labels])
        present = box_rapm_pos >= 0
        idx = np.clip(box_rapm_pos, 0, None)
        target_o_box = np.where(present, theta_o[idx], 0.0)
        target_d_box = np.where(present, theta_d[idx], 0.0)
        box_first_seen = np.where(present, first_seen[columns.o_start + idx], np.inf)
        seen = box_first_seen < cutoff_time
        feature_matrix = bi.features(snapshot, k)
        models[season] = fit_season_spm(
            season,
            feature_matrix[seen],
            snapshot.weight[seen],
            target_o_box[seen],
            target_d_box[seen],
            alpha,
        )
    return SeasonSpm(models, snapshot_by_time, box_index)


# --- The per-cutoff prior (rapm.fit_walk_forward's prior_fn hook) --------------------------------


def build_prior_fn(
    season_spm: SeasonSpm, columns: ModelColumns, k: float = SPM_K_MINUTES
) -> PriorFn:
    """The per-cutoff prior D2/H-h needs: at a round's cutoff, season s's ``SpmModel`` predicted
    on that same cutoff's box features -- ``None`` (prior mean 0, plain rapm behaviour) for a
    season with no ``SpmModel`` yet, or a cutoff this ``SeasonSpm`` has no snapshot for."""
    box_pos_for_player = np.array(
        [season_spm.player_index.get(label, -1) for label in columns.o_labels], dtype=np.int64
    )
    n_model = columns.n_model
    n_players = len(columns.o_labels)

    def prior_fn(cutoff_time: float, season: int, cols: ModelColumns) -> FloatArray | None:
        del cols  # always ``columns`` above (the same rapm player universe every round)
        model = season_spm.models.get(season)
        if model is None:
            return None
        snapshot = season_spm.snapshot_by_time.get(cutoff_time)
        if snapshot is None:
            return None
        feature_matrix = bi.features(snapshot, k)
        o_pred, d_pred = model.predict(feature_matrix)
        o_zero, d_zero = model.predict(np.zeros((1, len(model.stats))))
        present = box_pos_for_player >= 0
        idx = np.clip(box_pos_for_player, 0, None)
        prior = np.zeros(n_model)
        prior[columns.o_start : columns.o_start + n_players] = np.where(
            present, o_pred[idx], float(o_zero[0])
        )
        prior[columns.d_start : columns.d_start + n_players] = np.where(
            present, d_pred[idx], float(d_zero[0])
        )
        return prior

    return prior_fn


# --- The declared variant: tune (grid search) and fit (final walk-forward) ----------------------


def tune_spm(
    data: Data, spec: M3Backtest, inputs: RapmFitInputs, earlier: dict[str, TunedRapm]
) -> TunedRapm:
    """``rapm_spm``'s tuner (H6 design): at ``earlier["rapm"]``'s chosen half-life, search the
    shared ridge over ``spec.grid.ridge`` (shrinking toward an informative prior may prefer a
    different lambda than plain rapm's own); ``k``/``alpha`` are declared module constants, not
    searched (module docstring). Same tuning-RMSE computation as ``m3_backtest.tune_rapm``."""
    # Lazy: ``m3_backtest`` imports this module at its own top level (``VARIANTS["rapm_spm"]``),
    # so a top-level import back would be a real import cycle; by the time this function runs,
    # ``m3_backtest`` has finished loading and both names exist.
    from eurohoops.eval.m3_backtest import TunedRapm, rapm_margins  # noqa: PLC0415

    rapm = earlier["rapm"]
    last_tuning_season = max(spec.tuning)
    sub_mask = data.games["season"].to_numpy() <= last_tuning_season
    games_tuning = data.games[sub_mask].reset_index(drop=True)
    tuning_mask = data.split["tuning"][sub_mask]
    actual = data.margin[sub_mask]
    tuning_game_ids = set(games_tuning["game_id"].astype(str))
    shares_tuning = data.shares[data.shares["game_id"].astype(str).isin(tuning_game_ids)]
    player_games_tuning = data.player_games[
        data.player_games["game_id"].astype(str).isin(tuning_game_ids)
    ]

    season_spm = build_season_spm(
        games_tuning,
        player_games_tuning,
        inputs,
        rapm_half_life_days=rapm.half_life_days,
        rapm_ridge_o=rapm.ridge_o,
        rapm_ridge_d=rapm.ridge_d,
    )
    columns = plain_player_columns(inputs.spell_index)
    prior_fn = build_prior_fn(season_spm, columns)

    def rmse_for(ridge: float) -> float:
        wf = fit_walk_forward(
            games_tuning,
            inputs.rows,
            columns,
            half_life_days=rapm.half_life_days,
            ridge_o=ridge,
            ridge_d=ridge,
            prior_fn=prior_fn,
        )
        pred = rapm_margins(games_tuning, wf, shares_tuning, data.possessions)
        return _finite_rmse(pred, actual, tuning_mask)

    candidates = [(ridge, rmse_for(ridge)) for ridge in spec.grid.ridge]
    best_ridge, best_rmse = min(candidates, key=lambda t: t[1])

    grid_report = {
        "half_life_days": rapm.half_life_days,  # fixed at rapm's chosen value (H6 design)
        "ridge": list(spec.grid.ridge),
        "size": len(candidates),
        "results": [{"ridge": r, "tuning_rmse": _round6(v)} for r, v in candidates],
    }
    extra = {
        "ridge": best_ridge,
        "alpha": SPM_ALPHA,
        "k": SPM_K_MINUTES,
        "target_half_life_days": rapm.half_life_days,
        "target_ridge_o": rapm.ridge_o,
        "target_ridge_d": rapm.ridge_d,
        "spm": {
            str(season): model.to_json() for season, model in sorted(season_spm.models.items())
        },
    }
    return TunedRapm(
        half_life_days=rapm.half_life_days,
        ridge_o=best_ridge,
        ridge_d=best_ridge,
        tuning_rmse=best_rmse,
        grid=grid_report,
        extra=extra,
    )


def fit_spm(games: pd.DataFrame, inputs: RapmFitInputs, tuned: TunedRapm) -> WalkForward:
    """``rapm_spm``'s walk-forward fit over ``games`` with tuned parameters. Rebuilds every
    season's ``SpmModel`` from ``inputs.player_games`` (module docstring: ``fit`` has no ``Data``
    to read ``player_games`` from directly) using the *target*-fit hyperparameters stashed in
    ``tuned.extra`` during tuning (``earlier["rapm"]``'s chosen values, D2), then fits the outer
    ridge-toward-SPM-prior walk-forward with the tuned shared ridge. Walk-forward safe over the
    full ``games`` frame (validation/test included): ``build_season_spm`` only ever uses stints
    and box rows strictly before each season's own cutoff, whatever seasons ``games`` spans."""
    season_spm = build_season_spm(
        games,
        inputs.player_games,
        inputs,
        rapm_half_life_days=tuned.extra["target_half_life_days"],
        rapm_ridge_o=tuned.extra["target_ridge_o"],
        rapm_ridge_d=tuned.extra["target_ridge_d"],
        k=tuned.extra["k"],
        alpha=tuned.extra["alpha"],
    )
    columns = plain_player_columns(inputs.spell_index)
    prior_fn = build_prior_fn(season_spm, columns, tuned.extra["k"])
    return fit_walk_forward(
        games,
        inputs.rows,
        columns,
        half_life_days=tuned.half_life_days,
        ridge_o=tuned.ridge_o,
        ridge_d=tuned.ridge_d,
        prior_fn=prior_fn,
    )
