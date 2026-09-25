"""Append-only, idempotent log of pre-tip-off predictions for the live season."""

import csv
import logging
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from eurohoops.eval.backtest import TunedModel
from eurohoops.models.elo import prepare, replay, win_probability

LOG_COLUMNS = (
    "game_id",
    "season",
    "round",
    "phase",
    "tipoff_utc",
    "home",
    "away",
    "p_home",
    "exp_margin",
    "model",
    "model_version",
    "predicted_at_utc",
)
TIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

log = logging.getLogger(__name__)


class LatePredictionError(RuntimeError):
    """A prediction would be stamped at or after its game's tip-off."""


def logged_keys(log_path: Path) -> set[tuple[str, str]]:
    if not log_path.exists():
        return set()
    with log_path.open(newline="", encoding="utf-8") as fh:
        return {(row["game_id"], row["model_version"]) for row in csv.DictReader(fh)}


def predict_upcoming(
    games: pd.DataFrame,
    model: TunedModel,
    *,
    log_path: Path,
    season: int,
    replay_from: int,
    window: timedelta,
    clock: Callable[[], datetime],
) -> int:
    """Log confirmed, unplayed ``season`` games tipping off within ``window``; return rows added.

    Ratings come from replaying Elo over every completed game from season ``replay_from`` on
    (the tuned model's first warm-up season, so extra history cannot shift live ratings).
    The selection time and the ``predicted_at_utc`` stamp are separate clock reads; if the stamp
    is not strictly before a game's tip-off, nothing is written and LatePredictionError is raised.
    """
    now = clock()
    games = games[games["season"] >= replay_from]
    diffs = replay(prepare(games), model.params)
    upcoming = (
        (games["season"] == season)
        & ~games["played"]
        & games["confirmed_date"]
        & (games["tipoff_utc"] > now)
        & (games["tipoff_utc"] <= now + window)
    ).to_numpy()
    model_version = model.version()
    logged = logged_keys(log_path)
    targets = [
        (game, float(diff))
        for game, diff in zip(games[upcoming].to_dict("records"), diffs[upcoming], strict=True)
        if (game["game_id"], model_version) not in logged
    ]
    predicted_at = clock()
    late = [game["game_id"] for game, _ in targets if predicted_at >= game["tipoff_utc"]]
    if late:
        raise LatePredictionError(f"predicted_at {predicted_at} is not before tip-off of {late}")
    rows = [
        {
            "game_id": game["game_id"],
            "season": game["season"],
            "round": game["round"],
            "phase": game["phase"],
            "tipoff_utc": game["tipoff_utc"].strftime(TIME_FORMAT),
            "home": game["home"],
            "away": game["away"],
            "p_home": f"{win_probability(diff):.4f}",
            "exp_margin": f"{diff / model.margin_scale:.2f}",
            "model": "elo",
            "model_version": model_version,
            "predicted_at_utc": predicted_at.strftime(TIME_FORMAT),
        }
        for game, diff in targets
    ]
    is_new = not log_path.exists()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=LOG_COLUMNS, lineterminator="\n")
        if is_new:
            writer.writeheader()
        writer.writerows(rows)
    log.info(
        "%d upcoming games in window, %d already logged, %d rows appended",
        int(upcoming.sum()),
        int(upcoming.sum()) - len(rows),
        len(rows),
    )
    return len(rows)
