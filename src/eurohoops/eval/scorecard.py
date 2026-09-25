"""Live scorecard: the prediction log joined with results, Elo vs B0."""

from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eurohoops.eval.backtest import TunedModel, totals_scores
from eurohoops.eval.metrics import per_game_log_loss, score
from eurohoops.predict import LOG_COLUMNS, TIME_FORMAT

ROLLING_WINDOW = 50  # scored games per point of the rolling log loss (PLAN §7 monitoring)
RESULT_GRACE = timedelta(hours=48)  # a logged game without a result after this is flagged


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


def _scored(rows: pd.DataFrame, games: pd.DataFrame, model: TunedModel) -> pd.DataFrame:
    """Each game's earliest row joined with its result, in tip-off order, with B0 alongside."""
    first = rows.sort_values("predicted_at_utc").drop_duplicates("game_id", keep="first")
    scored = first[["game_id", "p_home", "exp_margin"]].merge(
        games.loc[
            games["played"] & ~games["forfeit"],
            ["game_id", "season", "tipoff_utc", "home_score", "away_score", "neutral"],
        ],
        on="game_id",
    )
    scored = scored.sort_values(["tipoff_utc", "game_id"], ignore_index=True)
    p_b0, exp_b0 = model.b0(scored["neutral"].to_numpy(dtype=np.float64))
    return scored.assign(
        margin=(scored["home_score"] - scored["away_score"]).astype("float64"),
        p_b0=p_b0,
        exp_b0=exp_b0,
    )


def _metrics(scored: pd.DataFrame, model: TunedModel) -> dict[str, Any]:
    """Elo vs B0 on the scored games, plus the totals baseline."""
    margin = scored["margin"].to_numpy(dtype=np.float64)
    sigma = model.margin_sigma
    return {
        "elo": score(
            scored["p_home"].to_numpy(dtype=np.float64),
            scored["exp_margin"].to_numpy(dtype=np.float64),
            margin,
            sigma,
        ).as_dict(),
        "b0": score(
            scored["p_b0"].to_numpy(dtype=np.float64),
            scored["exp_b0"].to_numpy(dtype=np.float64),
            margin,
            sigma,
        ).as_dict(),
        "totals": totals_scores(scored, model.totals_baseline),
    }


def rolling_log_loss(scored: pd.DataFrame, window: int = ROLLING_WINDOW) -> list[dict[str, Any]]:
    """Mean log loss of Elo and B0 over each run of ``window`` consecutive scored games.

    One point per game from the ``window``-th on (empty below ``window`` games), labelled with
    the window's last game.
    """
    home_won = (scored["margin"].to_numpy(dtype=np.float64) > 0).astype(np.float64)
    losses = pd.DataFrame(
        {
            "elo": per_game_log_loss(scored["p_home"].to_numpy(dtype=np.float64), home_won),
            "b0": per_game_log_loss(scored["p_b0"].to_numpy(dtype=np.float64), home_won),
        }
    )
    means = losses.rolling(window).mean().round(6)
    ids, tipoffs = scored["game_id"].tolist(), scored["tipoff_utc"].tolist()
    elo, b0 = means["elo"].tolist(), means["b0"].tolist()
    return [
        {
            "game_id": ids[i],
            "tipoff_utc": tipoffs[i].strftime(TIME_FORMAT),
            "n": i + 1,
            "elo": elo[i],
            "b0": b0[i],
        }
        for i in range(window - 1, len(scored))
    ]


def monitoring_warnings(
    rolling: list[dict[str, Any]], log: pd.DataFrame, games: pd.DataFrame, now: datetime
) -> list[dict[str, Any]]:
    """Monitoring alerts: Elo losing to B0 over the latest window, and results gone missing."""
    alerts: list[dict[str, Any]] = []
    if rolling and rolling[-1]["elo"] > rolling[-1]["b0"]:
        alerts.append(
            {
                "code": "elo_worse_than_b0",
                "message": f"Elo is worse than B0 over the last {ROLLING_WINDOW} scored games",
            }
        )
    logged = games[games["game_id"].isin(set(log["game_id"]))]
    stale = logged[~logged["played"] & (logged["tipoff_utc"] < now - RESULT_GRACE)]
    if not stale.empty:
        alerts.append(
            {
                "code": "missing_result",
                "message": (
                    f"{len(stale)} logged game(s) have no result "
                    f"{RESULT_GRACE.total_seconds() / 3600:.0f} h after tip-off"
                ),
                "games": sorted(stale["game_id"]),
            }
        )
    return alerts


def market_scores(scored: pd.DataFrame, odds_path: Path) -> dict[str, Any]:
    """Log loss of the de-vigged market P(home) on scored games with a pre-tip-off odds row.

    Each game uses its latest snapshot fetched before tip-off (the closest to a closing line);
    Elo and B0 are scored on the same games so the three numbers compare like for like.
    """
    odds = (
        pd.read_csv(odds_path, dtype={"game_id": str})
        if odds_path.exists()
        else pd.DataFrame(columns=["game_id", "fetched_at_utc", "p_home"])
    )
    odds = odds.loc[odds["p_home"].notna(), ["game_id", "fetched_at_utc", "p_home"]]
    joined = scored.merge(odds.rename(columns={"p_home": "p_market"}), on="game_id")
    joined = joined[pd.to_datetime(joined["fetched_at_utc"], utc=True) < joined["tipoff_utc"]]
    latest = joined.sort_values("fetched_at_utc").drop_duplicates("game_id", keep="last")
    home_won = (latest["margin"].to_numpy(dtype=np.float64) > 0).astype(np.float64)

    def mean_loss(column: str) -> float | None:
        if latest.empty:
            return None
        loss = per_game_log_loss(latest[column].to_numpy(dtype=np.float64), home_won)
        return round(float(loss.mean()), 6)

    return {
        "n": len(latest),
        "log_loss": mean_loss("p_market"),
        "elo_log_loss": mean_loss("p_home"),
        "b0_log_loss": mean_loss("p_b0"),
    }


def build_scorecard(
    log_path: Path,
    games: pd.DataFrame,
    model: TunedModel,
    now: datetime,
    *,
    manual_pushes: Sequence[datetime] = (),
    odds_path: Path | None = None,
) -> dict[str, Any]:
    """Score logged games that have finished; a missing log or no finished games gives n=0.

    Only rows stamped strictly before tip-off count; if a game was logged by several model
    versions, its earliest valid row is the pre-registered one. The headline (``elo``/``b0``)
    also drops rows that became public only after tip-off; ``all_rows`` keeps them. The rolling
    window, the warnings and the market column (only with ``odds_path``) use the headline rows.
    """
    log = read_log(log_path)
    valid = log[
        pd.to_datetime(log["predicted_at_utc"], utc=True)
        < pd.to_datetime(log["tipoff_utc"], utc=True)
    ]
    hidden = not_provable(valid, manual_pushes)
    provable_games = set(valid.loc[~hidden, "game_id"])
    headline = _scored(valid[~hidden], games, model)
    rolling = rolling_log_loss(headline)
    return {
        "rows_in_log": len(log),
        "rows_excluded_late": len(log) - len(valid),
        "rows_not_provable": int(hidden.sum()),
        "games_not_provable": sorted(set(valid["game_id"]) - provable_games),
        **_metrics(headline, model),
        "all_rows": _metrics(_scored(valid, games, model), model),
        "rolling": {"window": ROLLING_WINDOW, "series": rolling},
        "warnings": monitoring_warnings(rolling, log, games, now),
        "market": None if odds_path is None else market_scores(headline, odds_path),
    }
