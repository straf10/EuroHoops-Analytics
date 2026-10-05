"""M5 rest report (week 14-16 J7, ``reports/m5_rest.json``): how much rest explains beyond M5.

The chosen M5 has no rest term (its tuning choice preferred ``core`` to ``core_rest``), so the
report asks the question directly: regress M5's per-game margin residual (actual - expected,
from the committed per-game predictions of each scored split) on the home - away difference of
each rest feature, by OLS with HC0 (White) standard errors and 90% intervals, per split and per
tuning season. It also records each competition's rest-day distribution and how often the two
Greek EuroLeague clubs play within 3 days of a game in the other competition. Nothing is fitted
for forecasting: this is a description of committed predictions.
"""

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from eurohoops.models.elo import FloatArray
from eurohoops.models.rest import rest_features

FEATURES = ("days_rest", "short_rest", "games_last_7d", "other_comp_prev")
Z90 = 1.6448536269514722  # two-sided 90% Normal quantile
GREEK_WINDOW_DAYS = 3.0


def _round(value: float) -> float | None:
    return round(float(value), 6) if math.isfinite(value) else None


def rest_diff(
    games: pd.DataFrame, other: pd.DataFrame | None, club_map: Mapping[str, str]
) -> pd.DataFrame:
    """Home - away of every rest feature, one row per game of ``games`` (index game_id)."""
    long = rest_features(games, other, club_map)
    home, away = (
        long[long["side"] == side].set_index("game_id")[list(FEATURES)].astype("float64")
        for side in ("home", "away")
    )
    return home - away


def ols_hc0(x: FloatArray, y: FloatArray) -> dict[str, Any]:
    """OLS of ``y`` on [1, x] with HC0 standard errors and 90% intervals (``x`` columns =
    ``FEATURES``). A feature with no variation gets null estimates (it cannot be identified)."""
    varying = [j for j in range(x.shape[1]) if np.ptp(x[:, j]) > 0]
    design = np.column_stack([np.ones(len(y)), x[:, varying]])
    xtx_inv = np.linalg.inv(design.T @ design)
    beta = xtx_inv @ design.T @ y
    resid = y - design @ beta
    cov = xtx_inv @ (design.T * resid**2) @ design @ xtx_inv
    se = np.sqrt(np.diag(cov))
    out: dict[str, Any] = {"n": len(y)}
    for j, name in enumerate(FEATURES):
        if j not in varying:
            out[name] = {"coef": None, "se": None, "ci90": None}
            continue
        k = 1 + varying.index(j)
        out[name] = {
            "coef": _round(beta[k]),
            "se": _round(se[k]),
            "ci90": [_round(beta[k] - Z90 * se[k]), _round(beta[k] + Z90 * se[k])],
        }
    return out


def _distribution(diff_long: pd.DataFrame, seasons: tuple[int, ...]) -> dict[str, Any]:
    rows = diff_long[diff_long["season"].isin(seasons)]
    days = rows["days_rest"]
    return {
        "team_games": len(rows),
        "days_rest_quantiles": {
            str(q): _round(float(days.quantile(q))) for q in (0.05, 0.25, 0.5, 0.75, 0.95)
        },
        "short_rest_share": _round(float(rows["short_rest"].mean())),
        "games_last_7d_mean": _round(float(rows["games_last_7d"].mean())),
    }


def rest_report(
    games: pd.DataFrame,
    other: pd.DataFrame | None,
    club_map: Mapping[str, str],
    predictions: pd.DataFrame,
    *,
    tuning: tuple[int, ...],
    greek: tuple[str, ...],
) -> dict[str, Any]:
    """One competition's block. ``predictions``: the committed per-game CSV of the M5 backtest
    (rows ``model == "m5"``); ``greek``: this competition's codes of the two Greek clubs."""
    m5 = predictions[predictions["model"] == "m5"].set_index("game_id")
    diff = rest_diff(games, other, club_map)
    long = rest_features(games, other, club_map).merge(games[["game_id", "season"]], on="game_id")
    played = set(games.loc[games["played"], "game_id"].astype(str))
    long = long[long["game_id"].isin(played)]

    def fit(rows: pd.DataFrame) -> dict[str, Any]:
        x = diff.loc[rows.index].to_numpy(dtype=np.float64)
        y = (rows["actual_margin"] - rows["exp_margin"]).to_numpy(dtype=np.float64)
        return ols_hc0(x, y)

    by_split = {
        split: fit(rows) for split, rows in m5.groupby("split", sort=False) if len(rows) > 10
    }
    tuning_rows = m5[m5["split"] == "tuning"]
    by_season = {
        str(season): fit(rows) for season, rows in tuning_rows.groupby("season") if len(rows) > 10
    }
    seasons = tuple(sorted({int(s) for s in m5["season"]}))
    greek_rows = long[
        long["team"].isin(greek)
        & long["other_comp_prev"]
        & (long["days_rest"] <= GREEK_WINDOW_DAYS)
    ]
    return {
        "residual_on_rest_diff": {
            "model": "M5 margin residual (actual - expected) ~ 1 + home-away rest differences, "
            "OLS, HC0 SE, 90% Normal intervals",
            "by_split": by_split,
            "by_tuning_season": by_season,
        },
        "distribution": {
            "all_scored_seasons": _distribution(long, seasons),
            "tuning": _distribution(long, tuning),
        },
        "greek_within_3_days_of_other_competition": {
            str(season): len(rows) for season, rows in greek_rows.groupby("season")
        },
    }
