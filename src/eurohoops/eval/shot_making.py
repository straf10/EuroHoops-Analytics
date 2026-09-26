"""Player shot-making (F7, PLAN R9-R11): points above expected per 100 shots, shrunk, and whether
it is stable enough to show (R10, the F-k rule, fixed before this code ran on real data).

- Raw shot-making of a player-season = 100 * Σ(actual points - xPTS) / FGA, with out-of-fold
  xPTS only (development seasons LOSO, later seasons a development-fitted model).
- Sampling variance of the raw value: game-level bootstrap (F-g; 1,000 resamples of the
  player's games, seed 20261001).
- Empirical Bayes (R11): prior N(0, τ²) with τ² = var(raw) - mean(sampling variance) over
  development player-seasons with ≥ ``SHOWN_MIN_FGA`` FGA (method of moments, floored at 0);
  shrunk = raw * τ² / (τ² + v), posterior sd = sqrt(τ² v / (τ² + v)), 90% interval ± 1.645 sd.
- Stability (F-k): year-to-year Pearson correlation of shrunk shot-making for players with
  ≥ ``STABILITY_MIN_FGA`` FGA in consecutive development seasons, with a 90% bootstrap CI over
  player pairs (the pair is the unit of that correlation); split-half = Pearson correlation of
  raw shot-making on odd vs even games (by tip-off) within development player-seasons with
  ≥ 200 FGA, not Spearman-Brown corrected. Stable iff CI low ≥ 0.20 and split-half ≥ 0.30.
"""

from typing import Any, cast

import numpy as np
import pandas as pd

from eurohoops.models.elo import FloatArray

SHOWN_MIN_FGA = 100
STABILITY_MIN_FGA = 200
Y2Y_CI_LOW_MIN = 0.20
SPLIT_HALF_MIN = 0.30
Z90 = 1.6448536269514722
RESAMPLES = 1000
SEED = 20261001


def stability_verdict(y2y_ci_low: float, split_half: float) -> str:
    """The F-k rule."""
    stable = y2y_ci_low >= Y2Y_CI_LOW_MIN and split_half >= SPLIT_HALF_MIN
    return "stable enough to show" if stable else "not stable enough"


def _pearson(a: FloatArray, b: FloatArray) -> float:
    return float(np.corrcoef(a, b)[0, 1])


def _bootstrap_variance(residual: FloatArray, games: np.ndarray, rng: np.random.Generator) -> float:
    """Variance of 100 * mean residual over resampled games (game-level bootstrap)."""
    codes, inverse = np.unique(games, return_inverse=True)
    sums = np.bincount(inverse, weights=residual)
    counts = np.bincount(inverse).astype(np.float64)
    draws = rng.integers(0, len(codes), size=(RESAMPLES, len(codes)))
    stats = 100.0 * sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return float(stats.var(ddof=1))


def player_seasons(shots: pd.DataFrame) -> pd.DataFrame:
    """One row per shooter-season: FGA, points, xPTS, raw shot-making and its sampling variance.

    ``shots`` needs shooter, season, game_id, made, value, xpts.
    """
    rng = np.random.default_rng(SEED)
    shots = shots.assign(residual=shots["made"] * shots["value"] - shots["xpts"])
    rows = []
    for key, group in shots.groupby(["shooter", "season"], sort=True):
        shooter, season = cast(tuple[str, int], key)
        residual = group["residual"].to_numpy(dtype=np.float64)
        made = group["made"].to_numpy(dtype=np.float64)
        three = (group["value"].to_numpy() == 3).astype(np.float64)
        rows.append(
            {
                "shooter": shooter,
                "season": int(season),
                "fga": len(group),
                "points": float((group["made"] * group["value"]).sum()),
                "xpts": float(group["xpts"].sum()),
                "raw": 100.0 * float(residual.mean()),
                "variance": _bootstrap_variance(residual, group["game_id"].to_numpy(), rng)
                if (group["game_id"] != group["game_id"].iloc[0]).any()
                else np.nan,
                "fg_pct": float(made.mean()),
                "efg_pct": float((made * (1.0 + 0.5 * three)).mean()),
            }
        )
    return pd.DataFrame(rows)


def prior_variance(table: pd.DataFrame, development: tuple[int, ...]) -> float:
    use = table[table["season"].isin(development) & (table["fga"] >= SHOWN_MIN_FGA)]
    return max(float(use["raw"].var(ddof=1) - use["variance"].mean()), 0.0)


def shrink(table: pd.DataFrame, tau2: float) -> pd.DataFrame:
    v = table["variance"]
    weight = tau2 / (tau2 + v) if tau2 > 0 else 0.0 * v
    sd = np.sqrt(tau2 * v / (tau2 + v)) if tau2 > 0 else 0.0 * v
    shrunk = table["raw"] * weight
    return table.assign(
        shrunk=shrunk, shrink_weight=weight, low90=shrunk - Z90 * sd, high90=shrunk + Z90 * sd
    )


def correlation_ci(a: FloatArray, b: FloatArray) -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(a), size=(RESAMPLES, len(a)))
    stats = np.array([_pearson(a[d], b[d]) for d in draws])
    low, high = np.percentile(stats, [5.0, 95.0])
    return {
        "n": len(a),
        "r": round(_pearson(a, b), 6),
        "ci90": [round(float(low), 6), round(float(high), 6)],
    }


def _split_half(
    scored: pd.DataFrame, tipoff: pd.Series, development: tuple[int, ...]
) -> dict[str, Any]:
    """Raw shot-making on odd vs even games (by tip-off) of development player-seasons."""
    dev = scored[scored["season"].isin(development)].assign(
        tipoff=scored["game_id"].map(tipoff), residual=scored["points"] - scored["xpts"]
    )
    halves_a, halves_b = [], []
    for _, group in dev.groupby(["shooter", "season"], sort=True):
        if len(group) < STABILITY_MIN_FGA:
            continue
        order = group[["game_id", "tipoff"]].drop_duplicates().sort_values(["tipoff", "game_id"])
        odd = set(order["game_id"].iloc[0::2])
        in_odd = group["game_id"].isin(odd)
        halves_a.append(100.0 * float(group.loc[in_odd, "residual"].mean()))
        halves_b.append(100.0 * float(group.loc[~in_odd, "residual"].mean()))
    a, b = np.array(halves_a), np.array(halves_b)
    return {"player_seasons": len(a), "r": round(_pearson(a, b), 6)}


def _year_to_year(table: pd.DataFrame, development: tuple[int, ...]) -> dict[str, Any]:
    eligible = table[table["season"].isin(development) & (table["fga"] >= STABILITY_MIN_FGA)]
    nxt = eligible.assign(season=eligible["season"] - 1)
    pairs = eligible.merge(nxt, on=["shooter", "season"], suffixes=("", "_next"))
    out: dict[str, Any] = {"pairs": len(pairs)}
    for name, column in (
        ("shrunk_shot_making", "shrunk"),
        ("raw_shot_making", "raw"),
        ("raw_efg_pct", "efg_pct"),
        ("raw_fg_pct", "fg_pct"),
    ):
        out[name] = correlation_ci(
            pairs[column].to_numpy(dtype=np.float64),
            pairs[f"{column}_next"].to_numpy(dtype=np.float64),
        )
    return out


def player_report(
    scored: pd.DataFrame,
    tipoff: pd.Series,
    names: dict[str, str],
    development: tuple[int, ...],
    split_of: dict[int, str],
) -> dict[str, Any]:
    """``scored``: shots with out-of-fold xpts and points; ``tipoff``: game_id -> tip-off."""
    table = player_seasons(scored)
    tau2 = prior_variance(table, development)
    table = shrink(table, tau2)
    y2y = _year_to_year(table, development)
    split_half = _split_half(scored, tipoff, development)
    shown_dev = table[table["season"].isin(development) & (table["fga"] >= SHOWN_MIN_FGA)]
    mean_noise = float(shown_dev["variance"].mean())
    verdict = stability_verdict(y2y["shrunk_shot_making"]["ci90"][0], split_half["r"])
    shown = table[table["fga"] >= SHOWN_MIN_FGA].sort_values(
        ["season", "shrunk"], ascending=[True, False]
    )
    return {
        "units": "points above expected per 100 field-goal attempts",
        "xpts_source": "shot_xpts mart (out of fold)",
        "shown_min_fga": SHOWN_MIN_FGA,
        "prior": {
            "mean": 0.0,
            "variance": round(tau2, 6),
            "sd": round(float(np.sqrt(tau2)), 6),
            "estimated_on": list(development),
            "player_seasons": len(shown_dev),
        },
        "discrimination": {
            "signal_share": round(tau2 / (tau2 + mean_noise), 6) if tau2 + mean_noise else 0.0,
            "sd_raw": round(float(shown_dev["raw"].std(ddof=1)), 6),
            "sd_shrunk": round(float(shown_dev["shrunk"].std(ddof=1)), 6),
            "mean_sampling_sd": round(float(np.sqrt(mean_noise)), 6),
        },
        "stability": {
            "rule": "stable iff year-to-year r (shrunk, >= 200 FGA both seasons) 90% CI low "
            f">= {Y2Y_CI_LOW_MIN} and split-half r >= {SPLIT_HALF_MIN} (F-k)",
            "year_to_year": y2y,
            "split_half": split_half,
            "verdict": verdict,
        },
        "players": [
            {
                "shooter": r["shooter"],
                "name": names.get(r["shooter"], ""),
                "season": int(r["season"]),
                "split": split_of.get(int(r["season"]), ""),
                "fga": int(r["fga"]),
                "points": round(float(r["points"]), 1),
                "xpts": round(float(r["xpts"]), 2),
                "raw": round(float(r["raw"]), 4),
                "shrunk": round(float(r["shrunk"]), 4),
                "ci90": [round(float(r["low90"]), 4), round(float(r["high90"]), 4)],
            }
            for r in shown.to_dict("records")
        ],
    }
