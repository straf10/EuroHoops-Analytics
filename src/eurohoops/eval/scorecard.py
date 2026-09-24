"""Live scorecard: the prediction log joined with results, Elo vs B0."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eurohoops.eval.backtest import TunedModel
from eurohoops.eval.metrics import score
from eurohoops.predict import LOG_COLUMNS


def build_scorecard(log_path: Path, games: pd.DataFrame, model: TunedModel) -> dict[str, Any]:
    """Score logged games that have finished; a missing log or no finished games gives n=0.

    Only rows stamped strictly before tip-off count; if a game was logged by several model
    versions, its earliest valid row is the pre-registered one.
    """
    predictions = (
        pd.read_csv(log_path, dtype={"game_id": str, "p_home": float, "exp_margin": float})
        if log_path.exists()
        else pd.DataFrame(columns=list(LOG_COLUMNS))
    )
    predicted_at = pd.to_datetime(predictions["predicted_at_utc"], utc=True)
    valid = predictions[predicted_at < pd.to_datetime(predictions["tipoff_utc"], utc=True)]
    first = valid.sort_values("predicted_at_utc").drop_duplicates("game_id", keep="first")
    scored = first[["game_id", "p_home", "exp_margin"]].merge(
        games.loc[games["played"], ["game_id", "home_score", "away_score", "neutral"]],
        on="game_id",
    )

    margin = (scored["home_score"] - scored["away_score"]).to_numpy(dtype=np.float64)
    p_b0, exp_b0 = model.b0(scored["neutral"].to_numpy(dtype=np.float64))
    elo = score(
        scored["p_home"].to_numpy(dtype=np.float64),
        scored["exp_margin"].to_numpy(dtype=np.float64),
        margin,
    )
    return {
        "rows_in_log": len(predictions),
        "rows_excluded_late": len(predictions) - len(valid),
        "elo": elo.as_dict(),
        "b0": score(p_b0, exp_b0, margin).as_dict(),
    }
