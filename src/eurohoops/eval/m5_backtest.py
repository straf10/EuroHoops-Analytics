"""M5 backtest (week 14-16 J4): walk-forward tuning, the point-rule gate against M1, test (once).

M5's margin for a game is ``player_part + adjustment``. ``player_part`` is injected
(``PlayerPartFn``: M3's ``rapm_margins`` with a fitted RAPM for the EuroLeague, box-only margins
for the GBL), so the harness never knows which; ``adjustment`` is ``models.m5.fit_residual``'s
walk-forward team residual, with or without the rest effect.

Declared choices (the variant grid), all chosen on the tuning seasons only:

- **shares**: ``proj_hc``; ``proj_decay`` and ``proj_avail`` per ``half_life_games`` grid value.
  Each shares frame, and the player part computed from it, is built once.
- **form**: ``core`` (residual only), ``core_rest`` (residual plus the rest effect) and ``blend``
  (a probit blend by season of that shares option's best ``core_rest`` margin, M1's margin and
  the comparison Elo's margin, season s weighted on seasons before s).
- **distribution**: ``fit_margin_model`` on the candidate's own tuning-scored games (rated tuning
  games where its margin, M1's and Elo's are finite), in sample on tuning as M1 does (D6).

The candidate with the lowest log loss on the common tuning set wins (rated tuning games where
every candidate, M1 and Elo are finite); differences below ``tie_tolerance`` go to the simpler
form, then the simpler shares rule, then the earlier grid position. Platt recalibration (walk
forward by season) is kept only if it lowers that same log loss. The totals forecast is M1's
total or M1's plus a walk-forward ridge on the rest features, chosen by tuning totals CRPS.
The oracle (actual minutes shares) is reported, never chosen or gated.

A game is *scored* in a split when it is rated, in that split's seasons, and M5, M1, Elo and B0
all forecast it finitely (margin and total): every model's metrics, the gate and the comparisons
use that one game set.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from importlib.metadata import version
from itertools import product
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from eurohoops.config import M5Backtest
from eurohoops.eval.backtest import totals_baseline, win_probabilities
from eurohoops.eval.m1_backtest import EARLY_ROUNDS, Predictions, _metrics
from eurohoops.eval.m3_backtest import _data_sha256, rmse_diff_bootstrap_ci
from eurohoops.eval.metrics import crps_normal, paired_bootstrap_ci, per_game_log_loss
from eurohoops.models.elo import EloParams, FloatArray, prepare, replay
from eurohoops.models.m5 import apply_platt, blend_by_season, fit_platt, fit_residual
from eurohoops.models.minutes import oracle_shares
from eurohoops.models.rest import rest_features
from eurohoops.models.rotation import projected_shares_variant
from eurohoops.models.team_eff import (
    DecayParams,
    IntArray,
    MarginModel,
    fit_margin_model,
    forecast,
    prepare_history,
)

BoolArray = npt.NDArray[np.bool_]
PlayerPartFn = Callable[[pd.DataFrame, pd.DataFrame], FloatArray]

SPLITS = ("tuning", "validation", "test")
FORMS = ("core", "core_rest", "blend")  # simplest first (the tie-break order)
SHARES = ("proj_hc", "proj_decay", "proj_avail")  # simplest first
TOTALS = ("total_m1", "total_m1_rest")
MODELS = ("m5", "m5_oracle", "m1", "elo")  # the returned frame's model order
REST_COLUMNS = ("days_rest", "short_rest", "games_last_7d", "other_comp_prev")
BLEND_COMPONENTS = ("core_rest", "m1", "elo")
GREEK_CLUBS = frozenset({"PAN", "OLY", "00000001", "00000002"})
AFTER_OTHER_DAYS = 3.0  # greek_after_other: previous game in the other competition within 3 days
ORACLE_LABEL = "oracle, not a forecast"
GATE_RULE = (
    "M5 validation log loss < M1 validation log loss (point rule, owner 2026-10-05); "
    "paired bootstrap 95% CI reported"
)
FRAME_COLUMNS = (
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
)


def _round(value: float | None) -> float | None:
    return None if value is None or not math.isfinite(value) else round(float(value), 6)


# --- Inputs and the declared choice -----------------------------------------------------------


@dataclass(frozen=True)
class M5Inputs:
    """Everything a run reads. ``games``, ``team_games`` and ``player_games`` are the target
    competition's marts over every season (M1 and Elo warm up before the frame starts);
    ``other_games`` is the other competition's ``games`` mart (or ``None``); ``club_map`` maps its
    team codes to the target's. ``tuned_m1`` is ``reports/backtest_m1.json``'s ``tuned`` block and
    ``elo`` its ``comparison_elo`` block (k, hca, reversion, margin_scale, margin_sigma)."""

    games: pd.DataFrame
    team_games: pd.DataFrame
    player_games: pd.DataFrame
    other_games: pd.DataFrame | None
    club_map: Mapping[str, str]
    tuned_m1: dict[str, Any]
    elo: dict[str, Any]


@dataclass(frozen=True)
class Choice:
    """One declared M5 variant. ``half_life_games`` is None for ``proj_hc``; ``rest_ridge`` is
    None for ``core`` (a ``blend`` carries its ``core_rest`` component's parameters)."""

    shares: str
    half_life_games: float | None
    form: str
    residual_half_life_days: float
    residual_ridge: float
    rest_ridge: float | None
    platt: bool
    total: str


@dataclass(frozen=True)
class _Option:
    """A projected-shares rule; its shares frame and player part are built once per run."""

    shares: str
    half_life_games: float | None


def candidate_key(choice: Choice) -> str:
    """A readable unique key of a candidate (the Platt and totals decisions are not part of it)."""
    shares = (
        choice.shares
        if choice.half_life_games is None
        else f"{choice.shares}@{choice.half_life_games:g}"
    )
    key = f"{shares}|{choice.form}|d{choice.residual_half_life_days:g}|r{choice.residual_ridge:g}"
    return key if choice.rest_ridge is None else f"{key}|k{choice.rest_ridge:g}"


def choose_candidate(table: Sequence[tuple[Choice, float]], tolerance: float) -> int:
    """Index of the winner of a table of (candidate, tuning log loss) in grid order: the lowest
    loss, or, among every candidate within ``tolerance`` of it (difference < tolerance), the
    simplest by form (``FORMS``), then shares rule (``SHARES``), then the lowest loss within that
    model, then grid position. Grid values (half-lives, ridges) are not a complexity order, so
    they never break a tie (D10)."""
    finite = [(i, loss) for i, (_, loss) in enumerate(table) if math.isfinite(loss)]
    if not finite:
        raise ValueError("no candidate has a finite tuning log loss")
    best = min(loss for _, loss in finite)
    tied = [i for i, loss in finite if loss == best or loss - best < tolerance]
    return min(
        tied,
        key=lambda i: (
            FORMS.index(table[i][0].form),
            SHARES.index(table[i][0].shares),
            table[i][1],
            i,
        ),
    )


def gate_block(
    variant: str,
    m5_log_loss: float | None,
    m1_log_loss: float | None,
    differences: dict[str, Any],
) -> dict[str, Any]:
    """The gate (J-h, point rule): M5 passes iff its validation log loss is strictly below M1's.
    The comparison uses the unrounded values; ``None`` (no validation game) leaves it undecided."""
    passed = None if m5_log_loss is None or m1_log_loss is None else bool(m5_log_loss < m1_log_loss)
    return {
        "rule": GATE_RULE,
        "variant": variant,
        "m5_log_loss": _round(m5_log_loss),
        "m1_log_loss": _round(m1_log_loss),
        "passed": passed,
        **differences,
    }


# --- The prepared data: the frame, rest features and the comparison models --------------------


@dataclass(frozen=True)
class _Context:
    frame: pd.DataFrame  # warmup..test seasons, tip-off order
    season: IntArray
    rated: BoolArray
    margin: FloatArray  # actual home margin (0 where unplayed)
    total: FloatArray
    home_won: FloatArray
    split: dict[str, BoolArray]  # rated games of the split's seasons
    rest_home: FloatArray  # (n, 4) REST_COLUMNS of the home side
    rest_away: FloatArray
    m1: Predictions
    m1_model: MarginModel
    m1_pace: FloatArray
    m1_total_sigma: float
    elo: Predictions
    elo_margin_sigma: float
    elo_total_sigma: float
    b0: Predictions
    total_rest: FloatArray  # M1 total plus the walk-forward rest regression

    @property
    def rest_diff(self) -> FloatArray:
        diff: FloatArray = self.rest_home - self.rest_away
        return diff

    @property
    def rest_sum(self) -> FloatArray:
        total: FloatArray = self.rest_home + self.rest_away
        return total

    @property
    def comparison_ok(self) -> BoolArray:
        """M1 and Elo both forecast the game's margin."""
        ok: BoolArray = np.isfinite(self.m1.margin) & np.isfinite(self.elo.margin)
        return ok


def _rms(values: FloatArray, mask: BoolArray) -> float:
    sel = mask & np.isfinite(values)
    return float(np.sqrt(np.mean(values[sel] ** 2))) if sel.any() else math.nan


def _rest_arrays(inputs: M5Inputs, frame: pd.DataFrame) -> tuple[FloatArray, FloatArray]:
    """Rest features of the home and away side of every frame game, as floats."""
    long = rest_features(inputs.games, inputs.other_games, inputs.club_map)
    long = long.set_index(["game_id", "side"])
    ids = frame["game_id"].astype(str)
    sides: list[FloatArray] = []
    for side in ("home", "away"):
        part = long.xs(side, level="side")[list(REST_COLUMNS)].astype("float64")
        values: FloatArray = np.asarray(part.reindex(ids), dtype=np.float64)
        sides.append(values)
    return sides[0], sides[1]


def _rest_total(
    actual_total: FloatArray,
    m1_total: FloatArray,
    *,
    rated: BoolArray,
    season: IntArray,
    rest_sum: FloatArray,
    ridge: float,
) -> FloatArray:
    """M1's total plus a ridge regression of (actual total - M1 total) on the rest sums (home +
    away of each feature), intercept unpenalised, fitted for season s on rated games of seasons
    before s (M1's total unchanged for the first season)."""
    design = np.column_stack([np.ones(len(m1_total)), rest_sum])
    penalty = np.diag([0.0, *([ridge] * rest_sum.shape[1])])
    gap = actual_total - m1_total
    out = m1_total.copy()
    for s in np.unique(season):
        train = rated & (season < s) & np.isfinite(gap)
        if not train.any():
            continue
        x = design[train]
        theta = np.linalg.solve(x.T @ x + penalty, x.T @ gap[train])
        now = season == s
        out[now] = m1_total[now] + design[now] @ theta
    return out


def _prepare(inputs: M5Inputs, spec: M5Backtest) -> _Context:
    games = inputs.games
    first, last = spec.warmup[0], spec.test[-1]
    frame = (
        games[games["season"].between(first, last)].sort_values("tipoff_utc").reset_index(drop=True)
    )
    ids = frame["game_id"].astype(str)
    rated = (frame["played"] & ~frame["forfeit"]).to_numpy()
    home = frame["home_score"].astype("float64").fillna(0.0).to_numpy()
    away = frame["away_score"].astype("float64").fillna(0.0).to_numpy()
    season = frame["season"].to_numpy(dtype=np.int64)
    margin, total = home - away, home + away
    home_won = (home > away).astype(np.float64)
    split = {
        name: rated & np.isin(season, seasons)
        for name, seasons in (
            ("tuning", spec.tuning),
            ("validation", spec.validation),
            ("test", spec.test),
        )
    }
    tuning = split["tuning"]

    # M1 and Elo are replayed over every game of the competition and aligned by game_id.
    ordered = games.sort_values("tipoff_utc", kind="stable").reset_index(drop=True)
    ordered_ids = ordered["game_id"].astype(str)

    def aligned(values: FloatArray) -> FloatArray:
        out: FloatArray = np.asarray(
            pd.Series(values, index=ordered_ids).reindex(ids), dtype=np.float64
        )
        return out

    tuned = inputs.tuned_m1
    history = prepare_history(ordered, inputs.team_games)
    f = forecast(history, DecayParams(**tuned["rating"]), DecayParams(**tuned["pace"]))
    m1_margin, m1_total, m1_pace = aligned(f.margin), aligned(f.total), aligned(f.pace)
    m1_model = MarginModel(**tuned["margin"])
    m1_total_sigma = float(tuned["totals_sigma"])
    m1 = Predictions(
        p_home=m1_model.p_home(m1_margin, m1_pace),
        margin=m1_margin,
        margin_crps=m1_model.crps(m1_margin, m1_pace, margin),
        total=m1_total,
        total_crps=crps_normal(m1_total, m1_total_sigma, total),
    )

    elo_in = inputs.elo
    params = EloParams(k=elo_in["k"], hca=elo_in["hca"], reversion=elo_in["reversion"])
    diffs = aligned(replay(prepare(ordered), params))
    elo_margin = diffs / float(elo_in["margin_scale"])
    elo_sigma = float(elo_in["margin_sigma"])
    baseline = {int(s): totals_baseline(games, int(s)) for s in np.unique(season)}
    elo_total = np.array([baseline[int(s)] or np.nan for s in season], dtype=np.float64)
    elo_total_sigma = _rms(total - elo_total, tuning)
    total_crps = crps_normal(elo_total, elo_total_sigma, total)
    elo = Predictions(
        p_home=win_probabilities(diffs),
        margin=elo_margin,
        margin_crps=crps_normal(elo_margin, elo_sigma, margin),
        total=elo_total,
        total_crps=total_crps,
    )

    neutral = frame["neutral"].to_numpy(dtype=bool)
    fit_b0 = rated & (season <= spec.tuning[-1]) & ~neutral
    b0_rate = float(home_won[fit_b0].mean()) if fit_b0.any() else 0.5
    b0_home_margin = float(margin[fit_b0].mean()) if fit_b0.any() else 0.0
    b0_margin = np.where(neutral, 0.0, b0_home_margin)
    b0 = Predictions(
        p_home=np.where(neutral, 0.5, b0_rate),
        margin=b0_margin,
        margin_crps=crps_normal(b0_margin, elo_sigma, margin),
        total=elo_total,
        total_crps=total_crps,
    )

    rest_home, rest_away = _rest_arrays(inputs, frame)
    return _Context(
        frame=frame,
        season=season,
        rated=rated,
        margin=margin,
        total=total,
        home_won=home_won,
        split=split,
        rest_home=rest_home,
        rest_away=rest_away,
        m1=m1,
        m1_model=m1_model,
        m1_pace=m1_pace,
        m1_total_sigma=m1_total_sigma,
        elo=elo,
        elo_margin_sigma=elo_sigma,
        elo_total_sigma=elo_total_sigma,
        b0=b0,
        total_rest=_rest_total(
            total,
            m1_total,
            rated=rated,
            season=season,
            rest_sum=rest_home + rest_away,
            ridge=spec.total_rest_ridge,
        ),
    )


# --- Margins: the player part plus the team residual ------------------------------------------


def _options(spec: M5Backtest) -> list[_Option]:
    hc = [_Option("proj_hc", None)]
    decay = [_Option(rule, hl) for rule in SHARES[1:] for hl in spec.grid.half_life_games]
    return [*hc, *decay]


def _option_of(choice: Choice) -> _Option:
    return _Option(choice.shares, choice.half_life_games)


def _projected(
    option: _Option, frame: pd.DataFrame, player_games: pd.DataFrame, spec: M5Backtest
) -> pd.DataFrame:
    extra = {} if option.half_life_games is None else {"half_life_games": option.half_life_games}
    return projected_shares_variant(
        option.shares,
        frame,
        player_games,
        n_games=spec.projection_games,
        absent_games=spec.absent_games,
        **extra,
    )


def _core_margin(
    ctx: _Context,
    player_part: FloatArray,
    half_life_days: float,
    ridge: float,
    rest_ridge: float | None,
) -> tuple[FloatArray, FloatArray]:
    """``player_part + adjustment`` (NaN where either is) and the rest coefficients per game."""
    target = np.where(ctx.rated, ctx.margin - player_part, np.nan)
    fit = fit_residual(
        ctx.frame,
        target,
        None if rest_ridge is None else ctx.rest_diff,
        half_life_days=half_life_days,
        residual_ridge=ridge,
        rest_ridge=0.0 if rest_ridge is None else rest_ridge,
    )
    return player_part + fit.adjustment, fit.rest_coef


def _blend_weights(ctx: _Context, core: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Components (core_rest, M1, Elo margins) and each game's blend weights by season."""
    components = np.column_stack([core, ctx.m1.margin, ctx.elo.margin])
    weights = blend_by_season(components, ctx.home_won, ctx.season, ctx.rated)
    return components, weights


def _blend(ctx: _Context, core: FloatArray) -> FloatArray:
    components, weights = _blend_weights(ctx, core)
    blended: FloatArray = (weights * components).sum(axis=1)
    return blended


def _margin(ctx: _Context, player_part: FloatArray, choice: Choice) -> FloatArray:
    """The margin of ``choice`` over every frame game."""
    rest = None if choice.form == "core" else choice.rest_ridge
    margin, _ = _core_margin(
        ctx, player_part, choice.residual_half_life_days, choice.residual_ridge, rest
    )
    return _blend(ctx, margin) if choice.form == "blend" else margin


@dataclass(frozen=True)
class _Candidate:
    choice: Choice
    margin: FloatArray


def _tuning_common(ctx: _Context, margins: Sequence[FloatArray]) -> BoolArray:
    """Rated tuning games where every margin in ``margins``, M1 and Elo are finite."""
    common = ctx.split["tuning"] & ctx.comparison_ok
    for margin in margins:
        common = common & np.isfinite(margin)
    return common


def _fit_mask(ctx: _Context, margin: FloatArray) -> BoolArray:
    """A candidate's own tuning-scored games: the games its distribution and totals sigma fit on."""
    mask: BoolArray = ctx.split["tuning"] & ctx.comparison_ok & np.isfinite(margin)
    return mask


def _distribution(
    ctx: _Context, spec: M5Backtest, margin: FloatArray
) -> tuple[MarginModel, FloatArray]:
    """The margin distribution fitted on the candidate's tuning games, and p_home for every game."""
    fit = _fit_mask(ctx, margin)
    if not fit.any():
        raise ValueError("no tuning game with finite forecasts to fit the margin model on")
    model = fit_margin_model(
        spec.margin_variant, margin[fit], np.ones(int(fit.sum())), ctx.margin[fit]
    )
    return model, model.p_home(margin, np.ones(len(margin)))


def _mean_log_loss(ctx: _Context, p_home: FloatArray, mask: BoolArray) -> float:
    return float(per_game_log_loss(p_home[mask], ctx.home_won[mask]).mean())


def _option_candidates(
    ctx: _Context, spec: M5Backtest, option: _Option, player_part: FloatArray
) -> list[_Candidate]:
    """Every ``core`` and ``core_rest`` grid point of a shares option, then its ``blend``."""
    grid = spec.grid
    base = (option.shares, option.half_life_games)
    core = [
        _Candidate(
            Choice(*base, "core", hl, ridge, None, False, TOTALS[0]),
            _core_margin(ctx, player_part, hl, ridge, None)[0],
        )
        for hl, ridge in product(grid.residual_half_life_days, grid.residual_ridge)
    ]
    rest = [
        _Candidate(
            Choice(*base, "core_rest", hl, ridge, rest_ridge, False, TOTALS[0]),
            _core_margin(ctx, player_part, hl, ridge, rest_ridge)[0],
        )
        for hl, ridge, rest_ridge in product(
            grid.residual_half_life_days, grid.residual_ridge, grid.rest_ridge
        )
    ]
    # The blend's core_rest component: this option's best grid point by log loss on the games
    # where all of its core_rest points, M1 and Elo are finite (tuning only; first on a tie).
    local = _tuning_common(ctx, [c.margin for c in rest])
    if not local.any():
        raise ValueError(f"no tuning game is finite for every core_rest point of {option}")
    losses = [_mean_log_loss(ctx, _distribution(ctx, spec, c.margin)[1], local) for c in rest]
    best = rest[int(np.argmin(losses))]
    blend = _Candidate(replace(best.choice, form="blend"), _blend(ctx, best.margin))
    return [*core, *rest, blend]


# --- Distribution, Platt and totals of one candidate ------------------------------------------


@dataclass(frozen=True)
class _Forecast:
    pred: Predictions
    model: MarginModel
    total_sigma: float
    platt: dict[str, list[float | None]]  # season -> (a, b), only for seasons with earlier games


def _platt_by_season(
    ctx: _Context, p: FloatArray
) -> tuple[FloatArray, dict[str, list[float | None]]]:
    """Season s is recalibrated by (a, b) fitted on rated games of earlier seasons (identity
    when there are none)."""
    out = p.copy()
    coefficients: dict[str, list[float | None]] = {}
    for s in np.unique(ctx.season):
        train = ctx.rated & (ctx.season < s) & np.isfinite(p)
        if not train.any():
            continue
        a, b = fit_platt(p[train], ctx.home_won[train])
        now = ctx.season == s
        out[now] = apply_platt(p[now], a, b)
        coefficients[str(s)] = [_round(a), _round(b)]
    return out, coefficients


def _forecast(ctx: _Context, spec: M5Backtest, choice: Choice, margin: FloatArray) -> _Forecast:
    """The full forecast of ``choice``: margin distribution (fit on tuning), Platt if chosen,
    and the chosen totals variant with its tuning-RMS sigma."""
    model, p_home = _distribution(ctx, spec, margin)
    platt: dict[str, list[float | None]] = {}
    if choice.platt:
        p_home, platt = _platt_by_season(ctx, p_home)
    total = ctx.m1.total if choice.total == TOTALS[0] else ctx.total_rest
    total_sigma = _rms(ctx.total - total, _fit_mask(ctx, margin))
    ones = np.ones(len(margin))
    pred = Predictions(
        p_home=p_home,
        margin=margin,
        margin_crps=model.crps(margin, ones, ctx.margin),
        total=total,
        total_crps=crps_normal(total, total_sigma, ctx.total),
    )
    return _Forecast(pred, model, total_sigma, platt)


def _finite(pred: Predictions) -> BoolArray:
    ok: BoolArray = (
        np.isfinite(pred.p_home)
        & np.isfinite(pred.margin)
        & np.isfinite(pred.margin_crps)
        & np.isfinite(pred.total)
        & np.isfinite(pred.total_crps)
    )
    return ok


def _frame_of(ctx: _Context, forecast_: _Forecast) -> pd.DataFrame:
    """The per-game predictions of every frame game (what ``predict_games`` returns)."""
    pred, model = forecast_.pred, forecast_.model
    out = pd.DataFrame(
        {
            "game_id": ctx.frame["game_id"].astype(str).to_numpy(),
            "season": ctx.season,
            "p_home": pred.p_home,
            "exp_margin": pred.margin,
            "margin_sigma": model.scales(np.ones(len(pred.margin))),
            "margin_df": np.full(len(pred.margin), np.nan if model.df is None else model.df),
            "exp_total": pred.total,
            "total_sigma": forecast_.total_sigma,
        }
    )
    return out.round(6)


def predict_games(
    inputs: M5Inputs,
    *,
    spec: M5Backtest,
    player_part: PlayerPartFn,
    choice: Choice,
    oracle: bool = False,
) -> pd.DataFrame:
    """M5's predictions for every game of the frame (seasons ``warmup[0]`` to ``test[-1]``, tip-off
    order) under ``choice``: game_id, season, p_home, exp_margin, margin_sigma, margin_df,
    exp_total, total_sigma. The margin distribution and the total sigma are fit on the tuning
    seasons only (D6). ``oracle``: the actual minutes shares instead of projected ones (not a
    forecast)."""
    ctx = _prepare(inputs, spec)
    if oracle:
        shares = oracle_shares(ctx.frame, inputs.player_games)
    else:
        shares = _projected(_option_of(choice), ctx.frame, inputs.player_games, spec)
    margin = _margin(ctx, player_part(ctx.frame, shares), choice)
    return _frame_of(ctx, _forecast(ctx, spec, choice, margin))


# --- Metrics, differences, segments and the oracle gap ----------------------------------------


def _ci(diff: FloatArray, spec: M5Backtest) -> dict[str, Any] | None:
    """Mean and paired bootstrap 95% CI; None for an empty set."""
    if not len(diff):
        return None
    mean, low, high = paired_bootstrap_ci(diff, spec.bootstrap_resamples, spec.bootstrap_seed)
    return {
        "mean": _round(mean),
        "ci95": [_round(low), _round(high)],
        "resamples": spec.bootstrap_resamples,
        "seed": spec.bootstrap_seed,
    }


def _model_metrics(ctx: _Context, pred: Predictions, mask: BoolArray) -> dict[str, Any]:
    n = int(mask.sum())
    rmse = float(np.sqrt(np.mean((pred.margin[mask] - ctx.margin[mask]) ** 2))) if n else None
    return {**_metrics(pred, mask, ctx.margin, ctx.total), "margin_rmse": _round(rmse)}


def _differences(
    ctx: _Context, a: Predictions, b: Predictions, mask: BoolArray, *, suffix: str, spec: M5Backtest
) -> dict[str, Any]:
    """Paired CIs of a - b per game: log loss, Brier, margin CRPS and totals CRPS."""
    won = ctx.home_won[mask]
    loss = per_game_log_loss(a.p_home[mask], won) - per_game_log_loss(b.p_home[mask], won)
    return {
        f"log_loss_{suffix}": _ci(loss, spec),
        f"brier_{suffix}": _ci((a.p_home[mask] - won) ** 2 - (b.p_home[mask] - won) ** 2, spec),
        f"margin_crps_{suffix}": _ci(a.margin_crps[mask] - b.margin_crps[mask], spec),
        f"totals_crps_{suffix}": _ci(a.total_crps[mask] - b.total_crps[mask], spec),
    }


def _segments(
    ctx: _Context, m5: Predictions, mask: BoolArray
) -> dict[str, dict[str, float | int | None]]:
    """Where M5 and M1 differ: the Greek clubs, tight rest, the other competition, early season."""
    frame = ctx.frame
    greek_home = frame["home"].astype(str).isin(GREEK_CLUBS).to_numpy()
    greek_away = frame["away"].astype(str).isin(GREEK_CLUBS).to_numpy()
    short = REST_COLUMNS.index("short_rest")
    other = REST_COLUMNS.index("other_comp_prev")
    days = REST_COLUMNS.index("days_rest")
    after_other = np.zeros(len(frame), dtype=bool)
    for greek, rest in ((greek_home, ctx.rest_home), (greek_away, ctx.rest_away)):
        after_other |= greek & (rest[:, other] > 0.0) & (rest[:, days] <= AFTER_OTHER_DAYS)
    groups = {
        "pan_oly": greek_home | greek_away,
        "short_rest_any": (ctx.rest_home[:, short] > 0.0) | (ctx.rest_away[:, short] > 0.0),
        "other_comp_prev_any": (ctx.rest_home[:, other] > 0.0) | (ctx.rest_away[:, other] > 0.0),
        "early_rounds": (frame["phase"] == "RS").to_numpy()
        & (frame["round"] <= EARLY_ROUNDS).to_numpy(),
        "greek_after_other": after_other,
    }
    out: dict[str, dict[str, float | int | None]] = {}
    for name, group in groups.items():
        sel = mask & group
        n = int(sel.sum())
        out[name] = {
            "n": n,
            "m5_log_loss": _round(_mean_log_loss(ctx, m5.p_home, sel)) if n else None,
            "m1_log_loss": _round(_mean_log_loss(ctx, ctx.m1.p_home, sel)) if n else None,
            "m5_margin_crps": _round(float(m5.margin_crps[sel].mean())) if n else None,
            "m1_margin_crps": _round(float(ctx.m1.margin_crps[sel].mean())) if n else None,
        }
    return out


def _missed_top3(frame: pd.DataFrame, projected: pd.DataFrame, oracle: pd.DataFrame) -> BoolArray:
    """Games where, for either side, one of the three players with the highest oracle share has
    no projected-share row (or a projected share of 0)."""
    keys = ["game_id", "side", "player_id"]
    ranked = oracle.sort_values(
        ["game_id", "side", "share", "player_id"], ascending=[True, True, False, True]
    )
    top3 = ranked.groupby(["game_id", "side"]).head(3)
    seen = projected[projected["share"] > 0.0][keys].assign(seen=True)
    merged = top3[keys].merge(seen, on=keys, how="left")
    missed = set(merged[merged["seen"].isna()]["game_id"].astype(str))
    flagged: BoolArray = frame["game_id"].astype(str).isin(missed).to_numpy()
    return flagged


def _gap(
    ctx: _Context,
    projected: Predictions,
    oracle: Predictions,
    *,
    mask: BoolArray,
    missed: BoolArray,
    spec: M5Backtest,
) -> dict[str, Any]:
    """Projected minus oracle: per-game log loss (paired CI) and margin RMSE (bootstrap CI)."""

    def diffs(sel: BoolArray) -> dict[str, Any]:
        won = ctx.home_won[sel]
        loss = per_game_log_loss(projected.p_home[sel], won) - per_game_log_loss(
            oracle.p_home[sel], won
        )
        return {
            "n": int(sel.sum()),
            "log_loss_projected_minus_oracle": _ci(loss, spec),
            "rmse_projected_minus_oracle": rmse_diff_bootstrap_ci(
                ctx.margin[sel] - projected.margin[sel],
                ctx.margin[sel] - oracle.margin[sel],
                spec.bootstrap_resamples,
                spec.bootstrap_seed,
            ),
        }

    return {**diffs(mask), "missed_top3": diffs(mask & missed)}


# --- Selection: the candidates, the choice, Platt and the totals variant -----------------------


@dataclass(frozen=True)
class _Selection:
    candidates: list[_Candidate]
    table: list[tuple[Choice, float]]  # (candidate, log loss on the common tuning set)
    common: BoolArray
    index: int  # the winner
    shares: dict[_Option, pd.DataFrame]
    player_part: dict[_Option, FloatArray]

    @property
    def chosen(self) -> _Candidate:
        return self.candidates[self.index]


def _fixed_index(table: Sequence[tuple[Choice, float]], fixed: Choice) -> int:
    """Index of the candidate that ``fixed`` names (Platt and totals are decided separately)."""
    keys = [candidate_key(choice) for choice, _ in table]
    key = candidate_key(fixed)
    if key not in keys:
        raise ValueError(f"the fixed choice {key} is not a candidate of this grid")
    return keys.index(key)


def _select(
    ctx: _Context,
    spec: M5Backtest,
    inputs: M5Inputs,
    player_part: PlayerPartFn,
    fixed: Choice | None = None,
) -> _Selection:
    """Every shares frame and player part once, every candidate margin once, then the choice
    (``fixed``: the named candidate instead of the tuning winner)."""
    shares: dict[_Option, pd.DataFrame] = {}
    parts: dict[_Option, FloatArray] = {}
    candidates: list[_Candidate] = []
    for option in _options(spec):
        shares[option] = _projected(option, ctx.frame, inputs.player_games, spec)
        parts[option] = player_part(ctx.frame, shares[option])
        candidates += _option_candidates(ctx, spec, option, parts[option])
    common = _tuning_common(ctx, [c.margin for c in candidates])
    if not common.any():
        raise ValueError("no rated tuning game is finite for every candidate, M1 and Elo")
    table = [
        (c.choice, _mean_log_loss(ctx, _distribution(ctx, spec, c.margin)[1], common))
        for c in candidates
    ]
    index = (
        choose_candidate(table, spec.tie_tolerance) if fixed is None else _fixed_index(table, fixed)
    )
    return _Selection(candidates, table, common, index, shares, parts)


@dataclass(frozen=True)
class _Decision:
    """The final choice: the winner plus the Platt and totals decisions and their evidence."""

    choice: Choice
    platt_loss: dict[str, float]  # tuning log loss "with" and "without" Platt
    totals_crps: dict[str, float]  # tuning CRPS of each totals variant
    totals_sigma: dict[str, float]


def _decide(
    ctx: _Context, spec: M5Backtest, selection: _Selection, fixed: Choice | None = None
) -> _Decision:
    """Platt is kept iff it lowers the common-tuning-set log loss; the totals variant is the one
    with the lower tuning CRPS (Normal, tuning-RMS sigma) on the winner's tuning games. With
    ``fixed`` both decisions are taken from it; the evidence is still computed and reported."""
    chosen = selection.chosen
    losses = {
        name: _mean_log_loss(
            ctx,
            _forecast(ctx, spec, replace(chosen.choice, platt=platt), chosen.margin).pred.p_home,
            selection.common,
        )
        for name, platt in (("without", False), ("with", True))
    }
    fit = _fit_mask(ctx, chosen.margin) & np.isfinite(ctx.m1.total) & np.isfinite(ctx.total_rest)
    totals = dict(zip(TOTALS, (ctx.m1.total, ctx.total_rest), strict=True))
    sigma = {name: _rms(ctx.total - total, fit) for name, total in totals.items()}
    crps = {
        name: float(crps_normal(total, sigma[name], ctx.total)[fit].mean())
        for name, total in totals.items()
    }
    choice = replace(
        chosen.choice,
        platt=losses["with"] < losses["without"],
        total=TOTALS[1] if crps[TOTALS[1]] < crps[TOTALS[0]] else TOTALS[0],
    )
    if fixed is not None:
        choice = replace(chosen.choice, platt=fixed.platt, total=fixed.total)
    return _Decision(choice, losses, crps, sigma)


# --- The report -------------------------------------------------------------------------------


def model_version(choice: Choice) -> str:
    key = json.dumps(asdict(choice), sort_keys=True)
    return f"{version('eurohoops')}+m5.{hashlib.sha256(key.encode()).hexdigest()[:8]}"


def _edge_flags(spec: M5Backtest, choice: Choice) -> dict[str, bool]:
    """Per grid axis: the chosen value is on the grid edge (False for an axis it does not use)."""
    axes = {
        "half_life_games": spec.grid.half_life_games,
        "residual_half_life_days": spec.grid.residual_half_life_days,
        "residual_ridge": spec.grid.residual_ridge,
        "rest_ridge": spec.grid.rest_ridge,
    }
    flags = {}
    for axis, values in axes.items():
        value = getattr(choice, axis)
        flags[axis] = value is not None and len(values) > 1 and value in (min(values), max(values))
    return flags


def _season_rows(ctx: _Context, values: FloatArray, *, last: bool) -> dict[str, FloatArray]:
    """The first (or last) row of ``values`` with finite entries, per season."""
    out: dict[str, FloatArray] = {}
    for s in np.unique(ctx.season):
        rows = np.flatnonzero((ctx.season == s) & np.isfinite(values).all(axis=1))
        if len(rows):
            out[str(s)] = values[rows[-1] if last else rows[0]]
    return out


def _chosen_details(ctx: _Context, player_part: FloatArray, choice: Choice) -> dict[str, Any]:
    """Blend weights per season and the rest coefficients at each season's last cutoff."""
    details: dict[str, Any] = {}
    if choice.form == "core":
        return details
    core, coef = _core_margin(
        ctx, player_part, choice.residual_half_life_days, choice.residual_ridge, choice.rest_ridge
    )
    details["rest_coefficients"] = {
        season: {name: _round(float(v)) for name, v in zip(REST_COLUMNS, row, strict=True)}
        for season, row in _season_rows(ctx, coef, last=True).items()
    }
    if choice.form == "blend":
        _, weights = _blend_weights(ctx, core)
        details["blend_weights"] = {
            season: {name: _round(float(w)) for name, w in zip(BLEND_COMPONENTS, row, strict=True)}
            for season, row in _season_rows(ctx, weights, last=False).items()
        }
    return details


@dataclass(frozen=True)
class _Scoring:
    """The forecasts compared in one run and the games each split scores them on."""

    ctx: _Context
    spec: M5Backtest
    splits: tuple[str, ...]
    key: str
    m5: _Forecast
    oracle: _Forecast
    scored: dict[str, BoolArray]  # M5, M1, Elo and B0 all forecast the game
    oracle_scored: dict[str, BoolArray]  # ... and the oracle does too

    @property
    def models(self) -> dict[str, Predictions]:
        return {"m5": self.m5.pred, "m1": self.ctx.m1, "elo": self.ctx.elo, "b0": self.ctx.b0}


def _score(
    ctx: _Context,
    spec: M5Backtest,
    *,
    splits: tuple[str, ...],
    key: str,
    m5: _Forecast,
    oracle: _Forecast,
) -> _Scoring:
    everyone = np.logical_and.reduce([_finite(p) for p in (m5.pred, ctx.m1, ctx.elo, ctx.b0)])
    scored = {split: ctx.split[split] & everyone for split in splits}
    oracle_scored = {split: scored[split] & _finite(oracle.pred) for split in splits}
    return _Scoring(ctx, spec, splits, key, m5, oracle, scored, oracle_scored)


def _split_blocks(sc: _Scoring) -> dict[str, Any]:
    """Games per split, every model's metrics, the oracle and the segments of the scored splits."""
    ctx = sc.ctx
    return {
        "games_per_split": {
            split: {
                "rated": int(ctx.split[split].sum()),
                "scored": int(sc.scored[split].sum()),
                "dropped": int(ctx.split[split].sum()) - int(sc.scored[split].sum()),
            }
            for split in sc.splits
        },
        "metrics": {
            split: {
                name: _model_metrics(ctx, pred, sc.scored[split])
                for name, pred in sc.models.items()
            }
            for split in sc.splits
        },
        "oracle": {
            "label": ORACLE_LABEL,
            **{
                split: _model_metrics(ctx, sc.oracle.pred, sc.oracle_scored[split])
                for split in sc.splits
            },
        },
        "segments": {split: _segments(ctx, sc.m5.pred, sc.scored[split]) for split in sc.splits},
    }


def _other_variants(sc: _Scoring, selection: _Selection, decision: _Decision) -> dict[str, Any]:
    """The best candidate of each form (the chosen totals, no Platt) on the same scored games."""
    out: dict[str, Any] = {}
    for form in FORMS:
        members = [i for i, (c, _) in enumerate(selection.table) if c.form == form]
        pick = members[
            choose_candidate([selection.table[i] for i in members], sc.spec.tie_tolerance)
        ]
        choice = replace(selection.candidates[pick].choice, total=decision.choice.total)
        pred = _forecast(sc.ctx, sc.spec, choice, selection.candidates[pick].margin).pred
        out[form] = {
            "choice": asdict(choice),
            "key": candidate_key(choice),
            "chosen": pick == selection.index,
            "tuning_log_loss": _round(selection.table[pick][1]),
            **{
                split: _model_metrics(sc.ctx, pred, sc.scored[split] & _finite(pred))
                for split in sc.splits
            },
        }
    return out


def _validation_blocks(
    sc: _Scoring, projected: pd.DataFrame, oracle_frame: pd.DataFrame, *, score_test: bool
) -> dict[str, Any]:
    """The gate, the comparisons and the projected-minus-oracle gap (validation, test)."""
    ctx, spec, m5 = sc.ctx, sc.spec, sc.m5.pred
    validation = sc.scored["validation"]
    n = int(validation.sum())
    m5_loss = _mean_log_loss(ctx, m5.p_home, validation) if n else None
    m1_loss = _mean_log_loss(ctx, ctx.m1.p_home, validation) if n else None
    comparisons = {
        "validation": {
            "m5_minus_elo": _differences(
                ctx, m5, ctx.elo, validation, suffix="m5_minus_elo", spec=spec
            )
        }
    }
    if score_test:
        test = sc.scored["test"]
        comparisons["test"] = {
            "m5_minus_m1": _differences(ctx, m5, ctx.m1, test, suffix="m5_minus_m1", spec=spec),
            "m5_minus_elo": _differences(ctx, m5, ctx.elo, test, suffix="m5_minus_elo", spec=spec),
        }
    missed = _missed_top3(ctx.frame, projected, oracle_frame)
    return {
        "gate": gate_block(
            sc.key,
            m5_loss,
            m1_loss,
            _differences(ctx, m5, ctx.m1, validation, suffix="m5_minus_m1", spec=spec),
        ),
        "comparisons": comparisons,
        "gap": {
            split: _gap(
                ctx, m5, sc.oracle.pred, mask=sc.oracle_scored[split], missed=missed, spec=spec
            )
            for split in sc.splits
            if split != "tuning"
        },
    }


def _scored_frame(sc: _Scoring) -> pd.DataFrame:
    """Per-game predictions of the scored splits: split, then model, then tip-off order."""
    ctx = sc.ctx
    ones = np.ones(len(ctx.frame))
    elo_model = MarginModel("elo", ctx.elo_margin_sigma)
    rows = {
        "m5": (sc.key, sc.m5.pred, sc.m5.model, ones, sc.m5.total_sigma),
        "m5_oracle": (sc.key, sc.oracle.pred, sc.oracle.model, ones, sc.oracle.total_sigma),
        "m1": (ctx.m1_model.variant, ctx.m1, ctx.m1_model, ctx.m1_pace, ctx.m1_total_sigma),
        "elo": ("elo", ctx.elo, elo_model, ones, ctx.elo_total_sigma),
    }
    parts = []
    for split in sc.splits:
        for model in MODELS:
            variant, pred, margin_model, pace, total_sigma = rows[model]
            mask = sc.oracle_scored[split] if model == "m5_oracle" else sc.scored[split]
            idx = np.flatnonzero(mask)
            margin_df = np.nan if margin_model.df is None else margin_model.df
            parts.append(
                pd.DataFrame(
                    {
                        "game_id": ctx.frame["game_id"].astype(str).to_numpy()[idx],
                        "season": ctx.season[idx],
                        "split": split,
                        "model": model,
                        "variant": variant,
                        "p_home": pred.p_home[idx],
                        "exp_margin": pred.margin[idx],
                        "margin_sigma": margin_model.scales(pace)[idx],
                        "margin_df": np.full(len(idx), margin_df),
                        "exp_total": pred.total[idx],
                        "total_sigma": total_sigma,
                        "actual_margin": ctx.margin[idx],
                        "actual_total": ctx.total[idx],
                    },
                    columns=list(FRAME_COLUMNS),
                )
            )
    return pd.concat(parts, ignore_index=True).round(6)


def run_m5_backtest(
    inputs: M5Inputs,
    *,
    spec: M5Backtest,
    player_part: PlayerPartFn,
    tuning_only: bool = False,
    score_test: bool = False,
    fixed: Choice | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """The M5 report and the per-game predictions of the scored splits (module docstring).
    ``tuning_only``: the grid, the choice and tuning metrics only -- no validation or test number
    anywhere, since the verdict must be committed before validation is scored. ``score_test``:
    also score the test seasons (only after the validation commit). ``fixed``: score this
    choice (Platt and totals included) instead of choosing one (J-g: the GBL is scored with the
    EuroLeague verdict, no GBL-specific choice)."""
    ctx = _prepare(inputs, spec)
    selection = _select(ctx, spec, inputs, player_part, fixed)
    decision = _decide(ctx, spec, selection, fixed)
    final = decision.choice
    option = _option_of(final)
    m5 = _forecast(ctx, spec, final, selection.chosen.margin)
    oracle_frame = oracle_shares(ctx.frame, inputs.player_games)
    oracle_margin = _margin(ctx, player_part(ctx.frame, oracle_frame), final)
    splits: tuple[str, ...] = ("tuning",)
    if not tuning_only:
        splits = (*splits, "validation", *(("test",) if score_test else ()))
    # Platt (a, b) of season s are fitted on every earlier season's outcomes: report only the
    # seasons of the scored splits, so the verdict commit holds nothing fitted on validation.
    reported = {str(s) for split in splits for s in getattr(spec, split)} | {
        str(s) for s in spec.warmup
    }
    platt_reported = {s: ab for s, ab in m5.platt.items() if s in reported}
    sc = _score(
        ctx,
        spec,
        splits=splits,
        key=candidate_key(final),
        m5=m5,
        oracle=_forecast(ctx, spec, final, oracle_margin),
    )
    report: dict[str, Any] = {
        "model": "m5",
        "model_version": model_version(final),
        "seasons": {
            "warmup": list(spec.warmup),
            "tuning": list(spec.tuning),
            "validation": list(spec.validation),
            "test": list(spec.test),
        },
        "data_sha256": _data_sha256(
            ctx.frame,
            inputs.team_games,
            inputs.player_games,
            *([] if inputs.other_games is None else [inputs.other_games]),
        ),
        "tuning_only": tuning_only,
        "validation_scored": not tuning_only,
        "test_scored": score_test and not tuning_only,
        "grid": {
            "objective": "mean log loss on the common tuning set",
            "size": len(selection.table),
            "common_tuning_games": int(selection.common.sum()),
            "candidates": {candidate_key(c): _round(loss) for c, loss in selection.table},
            "best_on_edge": _edge_flags(spec, final),
        },
        "chosen": {
            "choice": asdict(final),
            "key": sc.key,
            **({"fixed": "the EuroLeague verdict (J-g), not chosen here"} if fixed else {}),
            "tuning_log_loss": _round(selection.table[selection.index][1]),
            "platt": {
                "kept": final.platt,
                "log_loss_with": _round(decision.platt_loss["with"]),
                "log_loss_without": _round(decision.platt_loss["without"]),
                "coefficients": platt_reported,
            },
            "total": {
                "variant": final.total,
                "tuning_crps": {name: _round(v) for name, v in decision.totals_crps.items()},
                "tuning_sigma": {name: _round(v) for name, v in decision.totals_sigma.items()},
            },
            "margin_model": {
                "variant": m5.model.variant,
                "scale": m5.model.scale,
                "df": m5.model.df,
            },
            **_chosen_details(ctx, selection.player_part[option], final),
        },
        **_split_blocks(sc),
        "other_variants": _other_variants(sc, selection, decision),
    }
    if not tuning_only:
        report.update(
            _validation_blocks(sc, selection.shares[option], oracle_frame, score_test=score_test)
        )
        if fixed is not None:  # J-g: the same comparison, reported only
            report["gate"]["gated"] = False
    return report, _scored_frame(sc)


def format_m5_table(report: dict[str, Any]) -> str:
    def cell(value: float | None, width: int, digits: int) -> str:
        return f"{'-' if value is None else f'{value:.{digits}f}':>{width}}"

    chosen = report["chosen"]
    platt = "Platt" if chosen["platt"]["kept"] else "no Platt"
    lines = [
        f"chosen: {chosen['key']} ({platt}, {chosen['choice']['total']}); "
        f"tuning log loss {chosen['tuning_log_loss']}"
    ]
    header = (
        f"{'split':<11}{'model':<10}{'n':>5}{'logloss':>9}{'brier':>8}{'rmse':>8}"
        f"{'mCRPS':>7}{'tMAE':>7}{'tCRPS':>7}"
    )
    lines += [header, "-" * len(header)]
    for split, by_model in report["metrics"].items():
        for name, m in {**by_model, "oracle": report["oracle"][split]}.items():
            lines.append(
                f"{split:<11}{name:<10}{m['n']:>5}{cell(m['log_loss'], 9, 4)}"
                f"{cell(m['brier'], 8, 4)}{cell(m['margin_rmse'], 8, 3)}"
                f"{cell(m['margin_crps'], 7, 2)}{cell(m['totals_mae'], 7, 2)}"
                f"{cell(m['totals_crps'], 7, 2)}"
            )
    gate = report.get("gate")
    if gate is not None:
        ci = gate["log_loss_m5_minus_m1"]
        ci_text = (
            ""
            if ci is None
            else f" diff {ci['mean']:+.4f} 95% CI [{ci['ci95'][0]:+.4f}, {ci['ci95'][1]:+.4f}]"
        )
        verdict = {True: "PASS", False: "FAIL", None: "n/a"}[gate["passed"]]
        lines.append(f"gate (M5 vs M1, validation log loss):{ci_text} -> {verdict}")
    return "\n".join(lines)
