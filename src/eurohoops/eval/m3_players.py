"""M3 H7: per-player-season RAPM ratings and uncertainty at each season's last-round cutoff.

Snapshot semantics (owner decision, fixed): one row per player-season for every EuroLeague season
in the M3 frame (2011-2025), carrying the chosen variant's rating *at that season's last round
cutoff* — the walk-forward state fitted on every stint that tipped off before that round's first
tip-off (exactly the object H-c predicts that round with), so nothing from that round or later
feeds it.

Noise variance sigma^2 follows H-g: estimated once from the tuning fit's residuals at the last
tuning season's snapshot (``noise_variance`` on rows from the first tuning season through that
cutoff, ``dof`` = used columns at that snapshot). Unseen players (column not yet used at the
snapshot) are reported at the prior mean with sd ``sqrt(sigma^2 / penalty)`` and zero O/D
covariance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy import special, stats

from eurohoops.config import M3Backtest
from eurohoops.eval.m3_backtest import RapmFitInputs, TunedRapm, fit_rapm
from eurohoops.models.elo import FloatArray
from eurohoops.models.minutes import round_cutoffs
from eurohoops.models.rapm import (
    AggregatedSystem,
    DecayedRidgeSparse,
    DesignRows,
    ModelColumns,
    plain_player_columns,
)
from eurohoops.models.rapm_posterior import noise_variance, posterior_with_pair_covariance
from eurohoops.models.spm import fit_spm

SNAPSHOT_SENTENCE = (
    "Each season's rating is the walk-forward fit using only stints before that season's "
    "last round's first tip-off."
)


def _round6(value: float) -> float:
    return round(float(value), 6)


def _round_tree(obj: Any) -> Any:
    if isinstance(obj, float):
        return _round6(obj) if math.isfinite(obj) else obj
    if isinstance(obj, dict):
        return {k: _round_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_tree(v) for v in obj]
    return obj


def _last_cutoff_by_season(games: pd.DataFrame) -> dict[int, float]:
    cutoffs = round_cutoffs(games)
    grouped = pd.DataFrame({"season": games["season"].to_numpy(), "cutoff": cutoffs.to_numpy()})
    out: dict[int, float] = {}
    for season_key, cutoff in grouped.groupby("season")["cutoff"].max().items():
        out[int(cast(Any, season_key))] = float(cutoff)
    return out


def _row_predictions(
    cols: np.ndarray, vals: np.ndarray, col_map: np.ndarray, theta: FloatArray
) -> FloatArray:
    model_cols = col_map[cols]
    return np.sum(vals * theta[model_cols], axis=1)


@dataclass
class _SeasonSnapshot:
    system: AggregatedSystem
    filled_theta: FloatArray
    prior: FloatArray | None


def _tuning_noise_variance(
    rows: DesignRows,
    col_map: np.ndarray,
    spec: M3Backtest,
    snapshot: _SeasonSnapshot,
    last_cutoffs: dict[int, float],
) -> tuple[float, float, int]:
    tuning_first = min(spec.tuning)
    tuning_last = max(spec.tuning)
    first_tip = float(np.min(rows.time[rows.season == tuning_first]))
    tuning_cutoff = last_cutoffs[tuning_last]
    mask = (rows.time >= first_tip) & (rows.time < tuning_cutoff)
    if not mask.any():
        raise ValueError("no design rows in the tuning residual window")
    pred = _row_predictions(rows.cols[mask], rows.vals[mask], col_map, snapshot.filled_theta)
    residual = rows.target[mask] - pred
    weight = rows.weight[mask]
    dof = float(len(snapshot.system.used))
    sigma2 = noise_variance(residual, weight, dof)
    return sigma2, dof, int(mask.sum())


def _player_season_box(player_games: pd.DataFrame, season: int) -> pd.DataFrame:
    pg = player_games[player_games["season"] == season]
    active = pg[pg["sec"].astype("float64") > 0.0]
    return active


def _cutoff_iso(frame: pd.DataFrame, season: int, cutoff_epoch: float) -> str:
    games = frame[frame["season"] == season]
    cutoffs = round_cutoffs(games).to_numpy()
    tipoffs = games["tipoff_utc"]
    match = games.loc[cutoffs == cutoff_epoch, "tipoff_utc"]
    ts = match.iloc[0] if len(match) else tipoffs.iloc[np.abs(cutoffs - cutoff_epoch).argmin()]
    return pd.Timestamp(ts).tz_convert("UTC").isoformat()


def _season_player_rows(  # noqa: PLR0915, PLR0917 -- one JSON row per active player
    season: int,
    snap: _SeasonSnapshot,
    columns: ModelColumns,
    tuned: TunedRapm,
    player_games: pd.DataFrame,
    noise_var: float,
    z90: float,
) -> list[dict[str, Any]]:
    active = _player_season_box(player_games, season)
    if not len(active):
        return []
    label_index = {label: i for i, label in enumerate(snap.system.labels)}
    players = sorted(active["player_id"].astype(str).unique())
    pairs: list[list[int]] = []
    pair_players: list[str] = []
    for pid in players:
        o_label, d_label = f"O:{pid}", f"D:{pid}"
        if o_label in label_index and d_label in label_index:
            pairs.append([label_index[o_label], label_index[d_label]])
            pair_players.append(pid)
    pairs_arr = np.asarray(pairs, dtype=np.int64) if pairs else np.empty((0, 2), dtype=np.int64)
    post, pair_cov = posterior_with_pair_covariance(
        snap.system.gram,
        snap.system.rhs,
        snap.system.penalty,
        snap.system.prior_mean,
        noise_var,
        pairs_arr,
    )
    pair_cov_map = dict(zip(pair_players, pair_cov, strict=True))
    penalty_full = columns.penalty(tuned.ridge_o, tuned.ridge_d)
    prior_full = (
        snap.prior if snap.prior is not None else np.zeros(columns.n_model, dtype=np.float64)
    )
    rows_out: list[dict[str, Any]] = []
    for pid in players:
        box = active[active["player_id"].astype(str) == pid]
        minutes = float(box["sec"].astype("float64").sum()) / 60.0
        games_n = len(box)
        teams = sorted(box["team"].astype(str).unique())
        o_label, d_label = f"O:{pid}", f"D:{pid}"
        seen = o_label in label_index and d_label in label_index
        if seen:
            o_i, d_i = label_index[o_label], label_index[d_label]
            o_val = float(post.mean[o_i])
            d_val = float(post.mean[d_i])
            sd_o = float(post.sd[o_i])
            sd_d = float(post.sd[d_i])
            cov = float(pair_cov_map[pid])
            sd_total = math.sqrt(sd_o**2 + sd_d**2 + 2.0 * cov)
            ci90_o = (o_val - z90 * sd_o, o_val + z90 * sd_o)
            ci90_d = (d_val - z90 * sd_d, d_val + z90 * sd_d)
            ci90_total = (
                (o_val + d_val) - z90 * sd_total,
                (o_val + d_val) + z90 * sd_total,
            )
        else:
            try:
                o_col = columns.o_start + columns.o_labels.index(pid)
            except ValueError:
                o_col = None
            try:
                d_col = columns.d_start + columns.d_labels.index(pid)
            except ValueError:
                d_col = None
            o_val = float(prior_full[o_col]) if o_col is not None else 0.0
            d_val = float(prior_full[d_col]) if d_col is not None else 0.0
            sd_o = (
                math.sqrt(noise_var / penalty_full[o_col])
                if o_col is not None
                else math.sqrt(noise_var / tuned.ridge_o)
            )
            sd_d = (
                math.sqrt(noise_var / penalty_full[d_col])
                if d_col is not None
                else math.sqrt(noise_var / tuned.ridge_d)
            )
            sd_total = math.sqrt(sd_o**2 + sd_d**2)
            ci90_o = (o_val - z90 * sd_o, o_val + z90 * sd_o)
            ci90_d = (d_val - z90 * sd_d, d_val + z90 * sd_d)
            ci90_total = (
                (o_val + d_val) - z90 * sd_total,
                (o_val + d_val) + z90 * sd_total,
            )
        rows_out.append(
            {
                "player_id": pid,
                "o": o_val,
                "d": d_val,
                "total": o_val + d_val,
                "sd_o": sd_o,
                "sd_d": sd_d,
                "sd_total": sd_total,
                "ci90_o": list(ci90_o),
                "ci90_d": list(ci90_d),
                "ci90_total": list(ci90_total),
                "minutes": minutes,
                "games": games_n,
                "teams": teams,
                "seen": seen,
            }
        )
    return rows_out


def season_end_players(  # noqa: PLR0917 -- report API matches the task spec
    frame: pd.DataFrame,
    inputs: RapmFitInputs,
    player_games: pd.DataFrame,
    tuned: TunedRapm,
    variant: str,
    spec: M3Backtest,
    snapshot: str,
) -> dict[str, Any]:
    """Build the H7 player-season report for ``variant`` (``rapm`` or ``rapm_spm``)."""
    if variant not in ("rapm", "rapm_spm"):
        raise ValueError(f"unsupported variant {variant!r}")
    columns = plain_player_columns(inputs.spell_index)
    col_map = columns.col_map
    last_cutoffs = _last_cutoff_by_season(frame)
    season_snaps: dict[int, _SeasonSnapshot] = {}

    def on_cutoff(
        cutoff_time: float,
        season: int,
        model: DecayedRidgeSparse,
        filled_theta: FloatArray,
        prior: FloatArray | None,
    ) -> None:
        if last_cutoffs.get(season) != cutoff_time:
            return
        system = model.aggregated_system(tuned.ridge_o, tuned.ridge_d, prior)
        prior_copy = None if prior is None else prior.copy()
        season_snaps[season] = _SeasonSnapshot(system, filled_theta.copy(), prior_copy)

    if variant == "rapm":
        fit_rapm(frame, inputs, tuned, on_cutoff=on_cutoff)
    else:
        fit_spm(frame, inputs, tuned, on_cutoff=on_cutoff)

    tuning_last = max(spec.tuning)
    if tuning_last not in season_snaps:
        raise RuntimeError(f"missing tuning-season snapshot for {tuning_last}")
    noise_var, dof, n_rows = _tuning_noise_variance(
        inputs.rows, col_map, spec, season_snaps[tuning_last], last_cutoffs
    )
    z90 = float(special.ndtri(0.95))
    seasons_out: dict[str, Any] = {}
    all_seen_sd: list[float] = []
    all_seen_minutes: list[float] = []
    player_seasons = 0
    players_set: set[str] = set()
    for season in sorted(frame["season"].unique()):
        season_int = int(season)
        if season_int not in season_snaps:
            continue
        snap = season_snaps[season_int]
        player_rows = _season_player_rows(
            season_int, snap, columns, tuned, player_games, noise_var, z90
        )
        for row in player_rows:
            if row["seen"]:
                all_seen_sd.append(row["sd_total"])
                all_seen_minutes.append(row["minutes"])
        player_seasons += len(player_rows)
        players_set.update(r["player_id"] for r in player_rows)
        n_seen = sum(1 for r in player_rows if r["seen"])
        median_sd = (
            float(np.median([r["sd_total"] for r in player_rows if r["seen"]])) if n_seen else None
        )
        seasons_out[str(season_int)] = {
            "cutoff_utc": _cutoff_iso(frame, season_int, last_cutoffs[season_int]),
            "n_players": len(player_rows),
            "n_seen": n_seen,
            "median_sd_total": median_sd,
            "players": player_rows,
        }
    if len(all_seen_sd) >= 2:
        rho, _ = stats.spearmanr(all_seen_sd, all_seen_minutes)
        spearman = float(rho)
    else:
        spearman = float("nan")
    base_extra = {k: v for k, v in tuned.extra.items() if k != "spm"}
    params: dict[str, Any] = {
        "half_life_days": tuned.half_life_days,
        "ridge_o": tuned.ridge_o,
        "ridge_d": tuned.ridge_d,
        **base_extra,
    }
    report: dict[str, Any] = {
        "model": "m3",
        "variant": variant,
        "params": params,
        "data_sha256": snapshot,
        "units": "points per 100 possessions (O + D = net)",
        "interval": 0.9,
        "snapshot": SNAPSHOT_SENTENCE,
        "noise_variance": {
            "value": noise_var,
            "dof": dof,
            "n_rows": n_rows,
            "seasons": [min(spec.tuning), max(spec.tuning)],
        },
        "seasons": seasons_out,
        "summary": {
            "player_seasons": player_seasons,
            "players": len(players_set),
            "sd_total_minutes_spearman": spearman,
        },
    }
    return cast(dict[str, Any], _round_tree(report))
