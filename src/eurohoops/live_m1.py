"""Live M1 (E-g): append-only pre-tip-off log of the team model, next to (never in) the Elo log.

M1 goes live only for a competition whose M1 report says the validation gate passed. Its
parameters are the frozen ``tuned`` block of that report (as Elo reads its own report); MLflow
is never read here. Ratings come from the same walk-forward as the backtest, fitted on every
game with box lines from season ``replay_from`` (the live Elo's first warm-up season) that
tipped off before the game's round opened.
"""

import csv
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from eurohoops.models.elo import FloatArray
from eurohoops.models.team_eff import DecayParams, MarginModel, forecast, prepare_history
from eurohoops.predict import TIME_FORMAT, LatePredictionError, logged_keys

M1_LOG_COLUMNS = (
    "game_id",
    "season",
    "round",
    "phase",
    "tipoff_utc",
    "home",
    "away",
    "p_home",
    "exp_margin",
    "exp_total",
    "margin_sigma",
    "margin_df",
    "total_sigma",
    "model",
    "model_version",
    "predicted_at_utc",
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class LiveM1:
    rating: DecayParams
    pace: DecayParams
    margin: MarginModel
    totals_sigma: float
    version: str
    gate_passed: bool


def load_m1(report_path: Path) -> LiveM1 | None:
    """The frozen M1 parameters of a backtest report; None when there is no report."""
    if not report_path.exists():
        return None
    report = json.loads(report_path.read_text(encoding="utf-8"))
    tuned, margin = report["tuned"], report["tuned"]["margin"]
    return LiveM1(
        rating=DecayParams(**tuned["rating"]),
        pace=DecayParams(**tuned["pace"]),
        margin=MarginModel(margin["variant"], margin["scale"], margin["df"], margin["ref_pace"]),
        totals_sigma=float(tuned["totals_sigma"]),
        version=str(report["model_version"]),
        gate_passed=bool(report["gate"]["passed"]),
    )


@dataclass(frozen=True)
class M1Forecasts:
    games: pd.DataFrame  # the games forecast, tip-off order
    p_home: FloatArray
    margin: FloatArray
    total: FloatArray
    margin_sigma: FloatArray


def m1_forecasts(
    games: pd.DataFrame, team_games: pd.DataFrame, model: LiveM1, replay_from: int
) -> M1Forecasts:
    """Walk-forward M1 forecasts of every game from season ``replay_from`` (NaN: no data yet)."""
    frame = games[games["season"] >= replay_from].reset_index(drop=True)
    result = forecast(prepare_history(frame, team_games), model.rating, model.pace)
    return M1Forecasts(
        games=frame,
        p_home=model.margin.p_home(result.margin, result.pace),
        margin=result.margin,
        total=result.total,
        margin_sigma=model.margin.scales(result.pace),
    )


def predict_upcoming_m1(
    games: pd.DataFrame,
    team_games: pd.DataFrame,
    model: LiveM1,
    *,
    log_path: Path,
    season: int,
    replay_from: int,
    window: timedelta,
    clock: Callable[[], datetime],
) -> int:
    """Like the Elo predictor: confirmed, unplayed ``season`` games within ``window`` that are
    not logged at this model version yet; games without a forecast (no box lines) are skipped.
    """
    if not model.gate_passed:
        log.info("M1 did not pass its validation gate: not logged")
        return 0
    now = clock()
    fc = m1_forecasts(games, team_games, model, replay_from)
    g = fc.games
    upcoming = (
        (g["season"] == season)
        & ~g["played"]
        & g["confirmed_date"]
        & (g["tipoff_utc"] > now)
        & (g["tipoff_utc"] <= now + window)
    ).to_numpy()
    missing = upcoming & np.isnan(fc.margin)
    if missing.any():
        log.warning("M1: %d upcoming games have no forecast (no box lines)", int(missing.sum()))
    logged = logged_keys(log_path)
    records = g.to_dict("records")
    targets = [
        int(i)
        for i in np.flatnonzero(upcoming & ~missing)
        if (str(records[i]["game_id"]), model.version) not in logged
    ]
    predicted_at = clock()
    late = [records[i]["game_id"] for i in targets if predicted_at >= records[i]["tipoff_utc"]]
    if late:
        raise LatePredictionError(f"predicted_at {predicted_at} is not before tip-off of {late}")
    rows = [
        {
            "game_id": records[i]["game_id"],
            "season": records[i]["season"],
            "round": records[i]["round"],
            "phase": records[i]["phase"],
            "tipoff_utc": records[i]["tipoff_utc"].strftime(TIME_FORMAT),
            "home": records[i]["home"],
            "away": records[i]["away"],
            "p_home": f"{fc.p_home[i]:.4f}",
            "exp_margin": f"{fc.margin[i]:.2f}",
            "exp_total": f"{fc.total[i]:.2f}",
            "margin_sigma": f"{fc.margin_sigma[i]:.4f}",
            "margin_df": "" if model.margin.df is None else f"{model.margin.df:g}",
            "total_sigma": f"{model.totals_sigma:.4f}",
            "model": "m1",
            "model_version": model.version,
            "predicted_at_utc": predicted_at.strftime(TIME_FORMAT),
        }
        for i in targets
    ]
    is_new = not log_path.exists()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=M1_LOG_COLUMNS, lineterminator="\n")
        if is_new:
            writer.writeheader()
        writer.writerows(rows)
    log.info("M1: %d rows appended to %s", len(rows), log_path)
    return len(rows)
