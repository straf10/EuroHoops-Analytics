"""Live scorecard: the prediction log joined with results, Elo vs B0."""

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eurohoops.eval.backtest import TunedModel, totals_scores
from eurohoops.eval.metrics import score
from eurohoops.predict import LOG_COLUMNS


def read_log(log_path: Path) -> pd.DataFrame:
    return (
        pd.read_csv(log_path, dtype={"game_id": str, "p_home": float, "exp_margin": float})
        if log_path.exists()
        else pd.DataFrame(columns=list(LOG_COLUMNS))
    )


def public_at(predicted_at: pd.Series, manual_pushes: Sequence[datetime]) -> pd.Series:
    """When each row became public: the first manual push at or after its stamp, else its stamp."""
    public = predicted_at.copy()
    for pushed in sorted(manual_pushes, reverse=True):
        public = public.where(predicted_at > pd.Timestamp(pushed), pd.Timestamp(pushed))
    return public


def not_provable(log: pd.DataFrame, manual_pushes: Sequence[datetime]) -> pd.Series:
    """Rows that were stamped before tip-off but only became public at or after it."""
    predicted_at = pd.to_datetime(log["predicted_at_utc"], utc=True)
    tipoff = pd.to_datetime(log["tipoff_utc"], utc=True)
    return (predicted_at < tipoff) & (public_at(predicted_at, manual_pushes) >= tipoff)


def _metrics(rows: pd.DataFrame, games: pd.DataFrame, model: TunedModel) -> dict[str, Any]:
    """Score each game's earliest row against its result, Elo vs B0, plus the totals baseline."""
    first = rows.sort_values("predicted_at_utc").drop_duplicates("game_id", keep="first")
    scored = first[["game_id", "p_home", "exp_margin"]].merge(
        games.loc[
            games["played"] & ~games["forfeit"],
            ["game_id", "season", "home_score", "away_score", "neutral"],
        ],
        on="game_id",
    )
    margin = (scored["home_score"] - scored["away_score"]).to_numpy(dtype=np.float64)
    p_b0, exp_b0 = model.b0(scored["neutral"].to_numpy(dtype=np.float64))
    elo = score(
        scored["p_home"].to_numpy(dtype=np.float64),
        scored["exp_margin"].to_numpy(dtype=np.float64),
        margin,
        model.margin_sigma,
    )
    return {
        "elo": elo.as_dict(),
        "b0": score(p_b0, exp_b0, margin, model.margin_sigma).as_dict(),
        "totals": totals_scores(scored, model.totals_baseline),
    }


def build_scorecard(
    log_path: Path,
    games: pd.DataFrame,
    model: TunedModel,
    manual_pushes: Sequence[datetime] = (),
) -> dict[str, Any]:
    """Score logged games that have finished; a missing log or no finished games gives n=0.

    Only rows stamped strictly before tip-off count; if a game was logged by several model
    versions, its earliest valid row is the pre-registered one. The headline (``elo``/``b0``)
    also drops rows that became public only after tip-off; ``all_rows`` keeps them.
    """
    log = read_log(log_path)
    valid = log[
        pd.to_datetime(log["predicted_at_utc"], utc=True)
        < pd.to_datetime(log["tipoff_utc"], utc=True)
    ]
    hidden = not_provable(valid, manual_pushes)
    provable_games = set(valid.loc[~hidden, "game_id"])
    return {
        "rows_in_log": len(log),
        "rows_excluded_late": len(log) - len(valid),
        "rows_not_provable": int(hidden.sum()),
        "games_not_provable": sorted(set(valid["game_id"]) - provable_games),
        **_metrics(valid[~hidden], games, model),
        "all_rows": _metrics(valid, games, model),
    }
