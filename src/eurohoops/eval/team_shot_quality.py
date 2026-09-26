"""Team shot quality and total shot value (F6), from out-of-fold xPTS only.

Per team-game (mart ``team_shot_quality``): the team's FGA, actual FG points and xPTS
(offence), the same for its opponent's shots (defence allowed), actual and expected FT points
(F2), and total shot value = xPTS + expected FT points, for and against. xPTS comes from the
``shot_xpts`` mart: development seasons out of fold (LOSO), later seasons from the
development-fitted model, so no shot's value comes from a model that saw its season.

Per team-season (``reports/m2_teams.json``): per-shot quality for offence and defence, actual
minus expected per shot, and total shot value per game, each with a game-level bootstrap 95%
interval (F-g). Calibration in the large: per development season, Σ xPTS vs Σ actual FG points.
"""

from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import pandas as pd

from eurohoops.eval.shot_metrics import cluster_bootstrap
from eurohoops.parse.free_throws import out_of_fold_expected

LARGE_TOLERANCE = 0.005  # |Σ xPTS / Σ actual FG points - 1| per development season (F6)
RESAMPLES = 1000
SEED = 20261001


def shots_with_xpts(shots: pd.DataFrame, xpts: pd.DataFrame) -> pd.DataFrame:
    """The shots that have an out-of-fold P(make), with xpts = P(make) * value."""
    joined = shots.merge(
        xpts[["game_id", "event", "split", "p_make"]], on=["game_id", "event"], how="inner"
    )
    return joined.assign(
        xpts=joined["p_make"] * joined["value"], points=joined["made"] * joined["value"]
    )


def team_games(
    scored: pd.DataFrame,
    ft: pd.DataFrame,
    development: Sequence[int],
    later: Sequence[int],
) -> pd.DataFrame:
    """One row per team-game with shots scored out of fold (see the module doc)."""
    offence = scored.groupby(
        ["season", "split", "game_id", "team", "opponent"], as_index=False
    ).agg(fga=("points", "size"), fg_points=("points", "sum"), xpts=("xpts", "sum"))
    free = ft[ft["season"].isin([*development, *later])]
    free = free.assign(expected_ft=out_of_fold_expected(free, development, later))
    offence = offence.merge(
        free[["game_id", "team", "ftm", "expected_ft"]], on=["game_id", "team"], how="left"
    )
    against = offence[["game_id", "team", "fga", "fg_points", "xpts", "ftm", "expected_ft"]]
    against = against.set_axis(
        [
            "game_id",
            "opponent",
            "fga_allowed",
            "fg_points_allowed",
            "xpts_allowed",
            "ftm_allowed",
            "expected_ft_allowed",
        ],
        axis=1,
    )
    out = offence.merge(against, on=["game_id", "opponent"], how="left")
    return out.assign(
        total_shot_value=out["xpts"] + out["expected_ft"],
        total_shot_value_allowed=out["xpts_allowed"] + out["expected_ft_allowed"],
    ).sort_values(["game_id", "team"], ignore_index=True)


def calibration_in_the_large(scored: pd.DataFrame, seasons: Sequence[int]) -> dict[str, Any]:
    out = {}
    for season in seasons:
        rows = scored[scored["season"] == season]
        actual, expected = float(rows["points"].sum()), float(rows["xpts"].sum())
        ratio = expected / actual
        out[str(season)] = {
            "fg_points": int(actual),
            "xpts": round(expected, 3),
            "ratio": round(ratio, 6),
            "within_tolerance": abs(ratio - 1.0) <= LARGE_TOLERANCE,
        }
    return out


def _interval(values: pd.Series, games: pd.Series) -> dict[str, Any]:
    mean, low, high = cluster_bootstrap(
        values.to_numpy(dtype=np.float64), games.to_numpy(), RESAMPLES, SEED
    )
    return {"mean": round(mean, 4), "ci95": [round(low, 4), round(high, 4)]}


def _per_shot(scored: pd.DataFrame, column: str) -> dict[str, Any]:
    """Per-shot mean of ``column`` over a team-season's shots, bootstrapping whole games."""
    return _interval(scored[column], scored["game_id"])


def team_seasons(scored: pd.DataFrame, games: pd.DataFrame) -> list[dict[str, Any]]:
    """Per team-season: offence and defence per-shot quality and total shot value per game."""
    scored = scored.assign(residual=scored["points"] - scored["xpts"])
    rows = []
    for key, tg in games.groupby(["season", "team"], sort=True):
        season, team = cast(tuple[int, str], key)
        own = scored[(scored["season"] == season) & (scored["team"] == team)]
        allowed = scored[(scored["season"] == season) & (scored["opponent"] == team)]
        rows.append(
            {
                "season": int(season),
                "team": str(team),
                "split": str(tg["split"].iloc[0]),
                "games": len(tg),
                "offence": {
                    "fga": len(own),
                    "xpts_per_shot": _per_shot(own, "xpts"),
                    "points_per_shot": round(float(own["points"].mean()), 4),
                    "actual_minus_expected_per_shot": _per_shot(own, "residual"),
                },
                "defence": {
                    "fga": len(allowed),
                    "xpts_allowed_per_shot": _per_shot(allowed, "xpts"),
                    "points_allowed_per_shot": round(float(allowed["points"].mean()), 4),
                    "actual_minus_expected_per_shot": _per_shot(allowed, "residual"),
                },
                "per_game": {
                    "expected_ft_points": round(float(tg["expected_ft"].mean()), 4),
                    "ft_points": round(float(tg["ftm"].mean()), 4),
                    "total_shot_value": _interval(tg["total_shot_value"], tg["game_id"]),
                    "total_shot_value_allowed": _interval(
                        tg["total_shot_value_allowed"], tg["game_id"]
                    ),
                    "actual_points_from_shots_and_ft": round(
                        float((tg["fg_points"] + tg["ftm"]).mean()), 4
                    ),
                },
            }
        )
    return rows


def team_report(
    scored: pd.DataFrame, games: pd.DataFrame, development: Sequence[int], variant: str
) -> dict[str, Any]:
    large = calibration_in_the_large(scored, development)
    return {
        "xpts_source": "shot_xpts mart: development LOSO, later seasons development-fitted",
        "chosen_variant": variant,
        "calibration_in_the_large": large,
        "calibration_in_the_large_tolerance": LARGE_TOLERANCE,
        "calibration_in_the_large_ok": all(s["within_tolerance"] for s in large.values()),
        "team_seasons": team_seasons(scored, games),
    }
