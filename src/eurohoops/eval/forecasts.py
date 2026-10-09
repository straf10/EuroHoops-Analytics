"""Season forecast record: every model's provable calls, scored per model, per round and per game.

The site's Forecasts view reads this. A model's calls are the earliest row per game that was
stamped before tip-off and public before it (the scorecard's own ``not_provable`` rule), joined
with the game's result. There is no backfill: a model has entries only for the games it logged,
so it is compared on its own games, and again on the games every compared model logged.
Codes here are the source's; the site shows display codes (``publish.py``).
"""

from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eurohoops.eval.metrics import per_game_log_loss
from eurohoops.eval.scorecard import not_provable, read_log
from eurohoops.logs import TIME_FORMAT

REGULAR_SEASON = "RS"
WITHIN_POINTS = (5, 10)  # margin error bands, in points
_NUMERIC = ("p_home", "exp_margin", "exp_total")


def _provable(log: pd.DataFrame, manual_pushes: Sequence[datetime]) -> pd.DataFrame:
    """The pre-registered row per game: earliest provable row stamped before its tip-off."""
    valid = log[
        pd.to_datetime(log["predicted_at_utc"], utc=True)
        < pd.to_datetime(log["tipoff_utc"], utc=True)
    ]
    kept = valid[~not_provable(valid, manual_pushes)]
    return kept.sort_values("predicted_at_utc").drop_duplicates("game_id", keep="first")


def _calls(
    log_path: Path, games: pd.DataFrame, season: int, manual_pushes: Sequence[datetime]
) -> tuple[pd.DataFrame, int | None]:
    """A model's scored games (with the model's columns) and its first logged round."""
    log = read_log(log_path)
    if "exp_total" not in log.columns:
        log = log.assign(exp_total=np.nan)
    rows = _provable(log, manual_pushes)
    rows = rows[rows["season"] == season]
    first_round = int(rows["round"].min()) if len(rows) else None
    finished = games.loc[
        games["played"] & ~games["forfeit"] & (games["season"] == season),
        ["game_id", "round", "phase", "tipoff_utc", "home", "away", "home_score", "away_score"],
    ]
    scored = rows[["game_id", *_NUMERIC]].astype(dict.fromkeys(_NUMERIC, "float64"))
    scored = scored.merge(finished, on="game_id")
    scored = scored.assign(margin=(scored["home_score"] - scored["away_score"]).astype("float64"))
    return scored.sort_values(["tipoff_utc", "game_id"], ignore_index=True), first_round


def _pick_right(p_home: pd.Series, margin: pd.Series) -> pd.Series:
    """True/False for a pick; None where p_home is 0.5 (no pick)."""
    right = pd.Series(np.where(p_home > 0.5, margin > 0, margin < 0), index=p_home.index)
    return right.astype(object).where(p_home != 0.5, None)


def _round(x: float, digits: int = 6) -> float:
    return round(float(x), digits)


def metrics(calls: pd.DataFrame) -> dict[str, Any]:
    """The season-scorecard metrics of one model's scored games."""
    n = len(calls)
    out: dict[str, Any] = {
        "n": n,
        "picks": 0,
        "right": 0,
        "right_pct": None,
        "log_loss": None,
        "brier": None,
        "margin_mae": None,
        "within": {str(w): None for w in WITHIN_POINTS},
        "totals_mae": None,
    }
    if not n:
        return out
    p = calls["p_home"].to_numpy(dtype=np.float64)
    margin = calls["margin"].to_numpy(dtype=np.float64)
    home_won = (margin > 0).astype(np.float64)
    picked = p != 0.5
    right = int((((p > 0.5) == (margin > 0)) & picked).sum())
    error = np.abs(calls["exp_margin"].to_numpy(dtype=np.float64) - margin)
    out |= {
        "picks": int(picked.sum()),
        "right": right,
        "right_pct": _round(right / picked.sum()) if picked.any() else None,
        "log_loss": _round(per_game_log_loss(p, home_won).mean()),
        "brier": _round(np.mean((p - home_won) ** 2)),
        "margin_mae": _round(error.mean()),
        "within": {str(w): _round((error <= w).mean()) for w in WITHIN_POINTS},
    }
    if calls["exp_total"].notna().any():
        total = (calls["home_score"] + calls["away_score"]).to_numpy(dtype=np.float64)
        out["totals_mae"] = _round(
            np.abs(calls["exp_total"].to_numpy(dtype=np.float64) - total).mean()
        )
    return out


def _group_key(phase: str, rnd: int) -> tuple[str, int | None]:
    """Regular-season games group by round number; each postseason phase is one group."""
    return (phase, rnd) if phase == REGULAR_SEASON else (phase, None)


def _rounds(
    per_model: Mapping[str, pd.DataFrame], labels: Mapping[str, str]
) -> list[dict[str, Any]]:
    frames = [c.assign(model=key) for key, c in per_model.items() if len(c)]
    if not frames:
        return []
    allc = pd.concat(frames, ignore_index=True)
    allc["group"] = [
        _group_key(p, int(r)) for p, r in zip(allc["phase"], allc["round"], strict=True)
    ]
    first_tip = allc.groupby("group")["tipoff_utc"].min().sort_values()
    out = []
    for group in first_tip.index:
        phase, rnd = group
        rows = allc[allc["group"] == group]
        out.append(
            {
                "phase": phase,
                "round": rnd,
                "games": int(rows["game_id"].nunique()),
                "models": {
                    key: metrics(rows[rows["model"] == key])
                    for key in labels
                    if (rows["model"] == key).any()
                },
            }
        )
    return out


def _games_list(per_model: Mapping[str, pd.DataFrame]) -> list[dict[str, Any]]:
    frames = []
    for key, calls in per_model.items():
        if len(calls):
            frames.append(
                calls.assign(model=key, right=_pick_right(calls["p_home"], calls["margin"]))
            )
    if not frames:
        return []
    allc = pd.concat(frames, ignore_index=True)
    out = []
    for game_id, rows in allc.groupby("game_id", sort=False):
        g = rows.iloc[0]
        out.append(
            {
                "game_id": str(game_id),
                "round": int(g["round"]),
                "phase": str(g["phase"]),
                "tipoff_utc": g["tipoff_utc"].strftime(TIME_FORMAT),
                "home": str(g["home"]),
                "away": str(g["away"]),
                "home_score": int(g["home_score"]),
                "away_score": int(g["away_score"]),
                "models": {
                    str(r["model"]): {
                        "p_home": float(r["p_home"]),
                        "exp_margin": float(r["exp_margin"]),
                        "right": None if r["right"] is None else bool(r["right"]),
                    }
                    for r in rows.to_dict("records")
                },
            }
        )
    out.sort(key=lambda g: (g["tipoff_utc"], g["game_id"]), reverse=True)
    return out


def build_forecasts(
    logs: Mapping[str, Path | None],
    labels: Mapping[str, str],
    games: pd.DataFrame,
    season: int,
    manual_pushes: Sequence[datetime] = (),
) -> dict[str, Any]:
    """The season record of the models in ``logs`` (key -> log path; ``None`` skips a model).

    ``labels`` fixes the models' order and names. Games are newest first; ``shared`` scores
    the models that have scored games on the games all of them logged.
    """
    per_model: dict[str, pd.DataFrame] = {}
    first_round: dict[str, int | None] = {}
    for key in labels:
        path = logs.get(key)
        if path is not None:
            per_model[key], first_round[key] = _calls(path, games, season, manual_pushes)
    used = {k: labels[k] for k in per_model}
    active = [k for k, calls in per_model.items() if len(calls)]
    compared = active if len(active) > 1 else []
    shared_ids = (
        set.intersection(*(set(per_model[k]["game_id"]) for k in compared)) if compared else set()
    )
    return {
        "models": [{"key": k, "label": v, "first_round": first_round[k]} for k, v in used.items()],
        "season": [{"model": k, **metrics(per_model[k])} for k in used],
        "shared": {
            "n": len(shared_ids),
            "rows": [
                {"model": k, **metrics(per_model[k][per_model[k]["game_id"].isin(shared_ids)])}
                for k in compared
            ],
        },
        "rounds": _rounds(per_model, used),
        "games": _games_list(per_model),
    }
