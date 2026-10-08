"""M3 GBL SPM transfer backtest (week 9-12 H8, H-i/D4).

EuroLeague ``rapm_spm`` season models (from the committed ``reports/backtest_m3.json`` ``chosen``
block's target RAPM hyperparameters) are applied to Greek Basket League box rates. At each GBL round
cutoff the latest EL season whose first-round cutoff is at or before that GBL cutoff supplies the
``SpmModel``; that model was fitted only on EuroLeague stints and box rows strictly before its own
first-round cutoff, so no EL data at or after the GBL cutoff reaches a GBL prediction (no GBL
RAPM: GBL stints pass only 56.1% of 2018-19 games, below the 95% rule, and there is no GBL
play-by-play for 2023-25). Transfer margins reuse H-c's projected minutes and expected
possessions; ``spm_transfer`` and ``spm_transfer_scaled`` fit only the home coefficient (and an
EL-scale factor ``c``) on GBL tuning games. Comparisons are labelled GBL evidence, not the
EuroLeague PLAN exit gate.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from eurohoops.config import M3, M3Backtest
from eurohoops.eval.m3_backtest import (
    BaselineFn,
    Data,
    _data_sha256,
    _model_metrics,
    _round,
    b0_margin,
    build_rapm_inputs,
    m1_margins_for,
    rmse_diff_bootstrap_ci,
)
from eurohoops.models import box_impact as bi
from eurohoops.models.box_impact import STAT_COLUMNS
from eurohoops.models.elo import FloatArray
from eurohoops.models.minutes import (
    expected_possessions,
    oracle_shares,
    projected_shares,
    round_cutoffs,
)
from eurohoops.models.spm import SpmModel, build_season_spm

BoolArray = npt.NDArray[np.bool_]

NO_GBL_RAPM_REASON = "GBL stints pass 56.1% of 2018-19 games (below 95%); no GBL PBP for 2023-25"
COMPARISONS_LABEL = (
    "GBL evidence, not the PLAN weeks 9-12 exit gate "
    "(that gate is EuroLeague rapm_spm vs box_only on validation, already PASS)"
)


@dataclass(frozen=True)
class ElSpm:
    """EuroLeague season SPM models for GBL transfer (EL data strictly before each EL season)."""

    models: dict[int, SpmModel]
    fit_time: dict[int, float]  # epoch seconds: first round cutoff of EL season s
    k: float
    half_life_days: float


def el_spm_models(
    el_games: pd.DataFrame,
    el_player_games: pd.DataFrame,
    stints: pd.DataFrame,
    checks: pd.DataFrame,
    chosen: dict[str, Any],
    *,
    last: int = M3.test[-1],
) -> ElSpm:
    """Build EL ``SpmModel`` per season on the M3 EuroLeague frame (same seasons as ``M3``;
    ``last`` extends it, e.g. to the live season for M6's live SPM: each season's model still
    reads only the seasons before it)."""
    first = M3.warmup[0]
    frame = (
        el_games[el_games["season"].between(first, last)]
        .sort_values("tipoff_utc")
        .reset_index(drop=True)
    )
    inputs = build_rapm_inputs(stints, checks, frame, el_player_games)
    season_spm = build_season_spm(
        frame,
        el_player_games,
        inputs,
        rapm_half_life_days=float(chosen["target_half_life_days"]),
        rapm_ridge_o=float(chosen["target_ridge_o"]),
        rapm_ridge_d=float(chosen["target_ridge_d"]),
        k=float(chosen["k"]),
        alpha=float(chosen["alpha"]),
    )
    cutoff = round_cutoffs(frame)
    season_col = frame["season"].astype(int).to_numpy()
    fit_time: dict[int, float] = {}
    for season in sorted(season_spm.models):
        idx = np.flatnonzero(season_col == season)
        fit_time[season] = float(cutoff.iloc[idx].min())
    return ElSpm(
        models=season_spm.models,
        fit_time=fit_time,
        k=float(chosen["k"]),
        half_life_days=float(chosen["target_half_life_days"]),
    )


def choose_model(el: ElSpm, cutoff_time: float) -> SpmModel | None:
    """Latest EL season with ``fit_time[season] <= cutoff_time``; ``None`` if none qualify."""
    eligible = [s for s, t in el.fit_time.items() if t <= cutoff_time]
    if not eligible:
        return None
    season = max(eligible, key=lambda s: (el.fit_time[s], s))
    return el.models[season]


def _shares_lookup(
    shares: pd.DataFrame, player_index: dict[str, int]
) -> dict[tuple[str, str], tuple[npt.NDArray[np.int64], FloatArray]]:
    out: dict[tuple[str, str], tuple[npt.NDArray[np.int64], FloatArray]] = {}
    for (game_id, side), rows in shares.groupby(["game_id", "side"]):
        idx = np.asarray(rows["player_id"].astype(str).map(player_index), dtype=np.int64)
        weight = rows["share"].to_numpy(dtype=np.float64)
        out[(str(game_id), str(side))] = (idx, weight)
    return out


def spm_transfer_diff(
    gbl_games: pd.DataFrame,
    gbl_player_games: pd.DataFrame,
    shares: pd.DataFrame,
    el: ElSpm,
    *,
    choose: Callable[[ElSpm, float], SpmModel | None] = choose_model,
) -> FloatArray:
    """Per-game per-100 rating difference from EL SPM on GBL box features (NaN if no EL model)."""
    n = len(gbl_games)
    diff = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return diff
    index = bi.player_index(gbl_player_games, shares)
    rows = bi.rows_from_player_games(gbl_games, gbl_player_games, index, STAT_COLUMNS)
    snapshots = bi.round_snapshots(
        gbl_games, rows, len(index), el.half_life_days, bi.RateSpec("poss", 100.0)
    )
    batches = bi.round_batches(gbl_games)
    game_ids = gbl_games["game_id"].astype(str).to_numpy()
    lookup = _shares_lookup(shares, index)
    for (cutoff_time, _), snapshot in zip(batches, snapshots, strict=True):
        model = choose(el, cutoff_time)
        if model is None:
            continue
        feature = bi.features(snapshot, el.k)
        o, d = model.predict(feature)
        rating = o + d
        for g in snapshot.games_idx:
            gid = str(game_ids[g])
            empty_idx = np.zeros(0, dtype=np.int64)
            empty_w = np.zeros(0, dtype=np.float64)
            home_idx, home_w = lookup.get((gid, "home"), (empty_idx, empty_w))
            away_idx, away_w = lookup.get((gid, "away"), (empty_idx, empty_w))
            diff[g] = float(home_w @ rating[home_idx] - away_w @ rating[away_idx])
    return diff


def _home_flag(games: pd.DataFrame) -> FloatArray:
    return np.where(games["neutral"].to_numpy(), 0.0, 1.0)


def _fit_transfer_h(
    margin: FloatArray,
    diff: FloatArray,
    possessions: FloatArray,
    home_flag: FloatArray,
    mask: BoolArray,
) -> float:
    """``margin = P/100 * (h * home_flag + diff)`` on ``mask``; unpenalised ``h``."""
    sel = mask & np.isfinite(margin) & np.isfinite(diff) & np.isfinite(possessions)
    if not sel.any():
        return 0.0
    p = possessions[sel] / 100.0
    x = p * home_flag[sel]
    y = margin[sel] - p * diff[sel]
    denom = float(np.dot(x, x))
    return float(np.dot(x, y) / denom) if denom > 0 else 0.0


def _fit_transfer_hc(
    margin: FloatArray,
    diff: FloatArray,
    possessions: FloatArray,
    home_flag: FloatArray,
    mask: BoolArray,
) -> tuple[float, float]:
    """``margin = P/100 * (h * home_flag + c * diff)`` by least squares on ``mask``."""
    sel = mask & np.isfinite(margin) & np.isfinite(diff) & np.isfinite(possessions)
    if not sel.any():
        return 0.0, 1.0
    p = possessions[sel] / 100.0
    design = np.column_stack([p * home_flag[sel], p * diff[sel]])
    coef, _, _, _ = np.linalg.lstsq(design, margin[sel], rcond=None)
    return float(coef[0]), float(coef[1])


def _transfer_margin(
    diff: FloatArray,
    possessions: FloatArray,
    home_flag: FloatArray,
    h: float,
    c: float = 1.0,
) -> FloatArray:
    p = possessions / 100.0
    margin: FloatArray = p * (h * home_flag + c * diff)
    margin[~np.isfinite(diff)] = np.nan
    return margin


def _games_without_box_lines(
    games: pd.DataFrame, player_games: pd.DataFrame, split_masks: dict[str, BoolArray]
) -> dict[str, int]:
    with_rows = set(player_games["game_id"].astype(str))
    game_ids = games["game_id"].astype(str).to_numpy()
    return {
        split: int((mask & ~np.isin(game_ids, list(with_rows))).sum())
        for split, mask in split_masks.items()
    }


def run_m3_gbl_backtest(
    gbl_games: pd.DataFrame,
    gbl_team_games: pd.DataFrame,
    gbl_player_games: pd.DataFrame,
    *,
    spec: M3Backtest,
    el: ElSpm,
    tuned_m1: dict[str, Any],
    box_only_fn: BaselineFn,
    pir_fn: BaselineFn,
    box_pages_skipped: int,
    score_test: bool,
) -> dict[str, Any]:
    """GBL transfer report (module docstring)."""
    first, last = spec.warmup[0], spec.test[-1]
    frame = (
        gbl_games[gbl_games["season"].between(first, last)]
        .sort_values("tipoff_utc")
        .reset_index(drop=True)
    )
    rated = (frame["played"] & ~frame["forfeit"]).to_numpy()
    home = frame["home_score"].astype("float64").fillna(0.0).to_numpy()
    away = frame["away_score"].astype("float64").fillna(0.0).to_numpy()
    season = frame["season"].to_numpy()
    split_masks = {
        name: rated & np.isin(season, seasons)
        for name, seasons in (
            ("tuning", spec.tuning),
            ("validation", spec.validation),
            ("test", spec.test),
        )
    }
    shares = projected_shares(frame, gbl_player_games, spec.projection_games)
    oracle = oracle_shares(frame, gbl_player_games)
    possessions = expected_possessions(frame, gbl_team_games)
    snapshot = _data_sha256(frame, gbl_team_games, gbl_player_games)
    data = Data(
        games=frame,
        margin=home - away,
        home_won=(home > away).astype(np.float64),
        split=split_masks,
        shares=shares,
        oracle=oracle,
        possessions=possessions,
        snapshot=snapshot,
        player_games=gbl_player_games,
    )
    poss_arr = possessions.reindex(frame["game_id"].astype(str)).to_numpy(dtype=np.float64)
    home_flag = _home_flag(frame)
    diff = spm_transfer_diff(frame, gbl_player_games, shares, el)
    diff_oracle = spm_transfer_diff(frame, gbl_player_games, oracle, el)
    tuning_mask = data.split["tuning"]
    h = _fit_transfer_h(data.margin, diff, poss_arr, home_flag, tuning_mask)
    h_scaled, c = _fit_transfer_hc(data.margin, diff, poss_arr, home_flag, tuning_mask)
    spm_margin = _transfer_margin(diff, poss_arr, home_flag, h, 1.0)
    spm_scaled_margin = _transfer_margin(diff, poss_arr, home_flag, h_scaled, c)
    m1_margin = m1_margins_for(frame, gbl_games, gbl_team_games, tuned_m1)
    b0_margins, b0_home_margin = b0_margin(data)
    box_result = box_only_fn(frame, gbl_player_games, shares, possessions, tuning_mask)
    pir_result = pir_fn(frame, gbl_player_games, shares, possessions, tuning_mask)
    oracle_margin = _transfer_margin(diff_oracle, poss_arr, home_flag, h, 1.0)

    splits: tuple[str, ...] = ("tuning", "validation", *(("test",) if score_test else ()))
    finite = {
        "spm_transfer": np.isfinite(spm_margin),
        "spm_transfer_scaled": np.isfinite(spm_scaled_margin),
        "box_only": np.isfinite(box_result.margin),
        "pir": np.isfinite(pir_result.margin),
        "m1": np.isfinite(m1_margin),
        "b0": np.isfinite(b0_margins),
    }
    common_finite = np.logical_and.reduce(list(finite.values()))
    scored = {name: data.split[name] & common_finite for name in splits}

    metrics_models = {
        "spm_transfer": spm_margin,
        "spm_transfer_scaled": spm_scaled_margin,
        "box_only": box_result.margin,
        "pir": pir_result.margin,
        "m1": m1_margin,
        "b0": b0_margins,
    }
    metrics = {
        split: {
            model: _model_metrics(margin, data, scored)[split]
            for model, margin in metrics_models.items()
        }
        for split in splits
    }
    oracle_scored = {split: scored[split] & finite["spm_transfer"] for split in splits}
    oracle_minutes = {
        "label": "oracle, not a forecast",
        **{split: _model_metrics(oracle_margin, data, oracle_scored)[split] for split in splits},
    }
    comparisons: dict[str, Any] = {
        "label": COMPARISONS_LABEL,
        "reference_model": "spm_transfer",
    }
    ref_residual = data.margin - spm_margin
    for comp_split in ("validation", *(("test",) if score_test else ())):
        mask = scored[comp_split]
        res_ref = ref_residual[mask]
        comparisons[comp_split] = {
            other: rmse_diff_bootstrap_ci(
                res_ref,
                (data.margin - metrics_models[other])[mask],
                spec.bootstrap_resamples,
                spec.bootstrap_seed,
            )
            for other in ("box_only", "pir", "m1")
        }

    return {
        "model": "m3_gbl",
        "competition": "gbl",
        "seasons": {
            "warmup": list(spec.warmup),
            "tuning": list(spec.tuning),
            "validation": list(spec.validation),
            "test": list(spec.test),
        },
        "data_sha256": snapshot,
        "test_scored": score_test,
        "validation_scored": True,
        "design": {
            "el_spm_source": "reports/backtest_m3.json chosen block",
            "rule": (
                "EL SpmModel of the latest EL season whose fit time (its first round cutoff; "
                "data strictly before it) is at or before the GBL round cutoff"
            ),
            "k": _round(el.k),
            "half_life_days": _round(el.half_life_days),
            "no_gbl_rapm_reason": NO_GBL_RAPM_REASON,
        },
        "el_spm_seasons_fitted": sorted(el.models),
        "transfer_params": {
            "spm_transfer": {"h": _round(h)},
            "spm_transfer_scaled": {"h": _round(h_scaled), "c": _round(c)},
        },
        "box_only_params": box_result.params,
        "pir_params": pir_result.params,
        "b0": {"home_margin": _round(b0_home_margin)},
        "games_per_split": {
            split: {
                "rated": int(data.split[split].sum()),
                "scored": int(scored[split].sum()),
                "dropped": int(data.split[split].sum()) - int(scored[split].sum()),
            }
            for split in splits
        },
        "games_without_box_lines": _games_without_box_lines(frame, gbl_player_games, split_masks),
        "box_pages_skipped": box_pages_skipped,
        "metrics": metrics,
        "oracle_minutes": oracle_minutes,
        "comparisons": comparisons,
    }


def format_m3_gbl_table(report: dict[str, Any]) -> str:
    header = f"{'split':<11}{'model':<22}{'n':>6}{'rmse':>8}{'mae':>8}{'logloss':>9}"
    lines = [header, "-" * len(header)]
    for split, models in report["metrics"].items():
        for name, m in models.items():
            n = m["n"]
            rmse = m["rmse"] if m["rmse"] is not None else float("nan")
            mae = m["mae"] if m["mae"] is not None else float("nan")
            log_loss = m["log_loss"] if m["log_loss"] is not None else float("nan")
            lines.append(f"{split:<11}{name:<22}{n:>6}{rmse:>8.3f}{mae:>8.3f}{log_loss:>9.4f}")
    return "\n".join(lines)
