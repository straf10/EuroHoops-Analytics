"""Live M5 (EuroLeague only): an append-only pre-tip-off shadow log of the roster-aware predictor,
next to (never in) the Elo and M1 logs.

M5 goes live only if its committed report (``reports/backtest_m5.json``) says the validation gate
passed; its parameters are that report's frozen ``chosen.choice`` (MLflow is never read). The
forecast is the backtest's own: ``m5_backtest.predict_games`` over every game of the competition
with the live season appended to the frame, so each game is forecast with exactly the walk-forward
the backtest scores -- the player part from RAPM fitted on stints that ended before the game's round
opened, the team residual and the rest effect fitted on games that tipped off before it, and the
*projected* roster (the team's recent rotation, ``models.rotation``), never the game's own minutes.
The margin distribution and the totals sigma are fitted on the tuning seasons only, so appending
the live season moves none of them. The live season only enters the frame, not the report's splits.
"""

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from eurohoops.config import M5, M5Backtest
from eurohoops.eval.m5_backtest import Choice, M5Inputs, PlayerPartFn, predict_games
from eurohoops.live_m1 import M1_LOG_COLUMNS
from eurohoops.logs import TIME_FORMAT, append_rows
from eurohoops.predict import logged_keys, refuse_late, upcoming_games

M5_LOG_COLUMNS = M1_LOG_COLUMNS  # the same shape as M1's: M5 needs nothing more

log = logging.getLogger(__name__)

# What a run needs once there is something to forecast: the (heavy) marts, rosters and RAPM.
Build = Callable[[], tuple[M5Inputs, PlayerPartFn]]


@dataclass(frozen=True)
class LiveM5:
    choice: Choice
    version: str
    gate_passed: bool


def load_m5(report_path: Path) -> LiveM5 | None:
    """The frozen M5 choice of a backtest report; None when there is no report."""
    if not report_path.exists():
        return None
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return LiveM5(
        choice=Choice(**report["chosen"]["choice"]),
        version=str(report["model_version"]),
        gate_passed=report.get("gate", {}).get("passed") is True,
    )


def live_spec(spec: M5Backtest, season: int) -> M5Backtest:
    """``spec`` with the live season appended to the test seasons, so the harness frame reaches it.
    The tuning seasons, which fit the margin distribution and the totals sigma, are unchanged."""
    return spec if season <= spec.test[-1] else replace(spec, test=(*spec.test, season))


def m5_forecasts(
    inputs: M5Inputs, model: LiveM5, player_part: PlayerPartFn, spec: M5Backtest, season: int
) -> pd.DataFrame:
    """M5's forecast of every game of the frame: ``predict_games`` under the frozen choice."""
    return predict_games(
        inputs, spec=live_spec(spec, season), player_part=player_part, choice=model.choice
    )


def predict_upcoming_m5(
    games: pd.DataFrame,
    model: LiveM5,
    build: Build,
    *,
    log_path: Path,
    season: int,
    window: timedelta,
    clock: Callable[[], datetime],
    spec: M5Backtest = M5,
) -> int:
    """Like the M1 predictor: confirmed, unplayed ``season`` games within ``window`` that are not
    logged at this model version yet. ``games`` is the competition's games mart; ``build`` makes
    the harness inputs and the player part, and is called only when there is something to log.
    Games without a finite forecast are skipped with a warning."""
    if not model.gate_passed:
        log.info("M5 did not pass its validation gate: not logged")
        return 0
    now = clock()
    upcoming = upcoming_games(games, season, now, window)
    logged = logged_keys(log_path)
    fresh = np.array([(str(g), model.version) not in logged for g in games["game_id"]], dtype=bool)
    wanted = games[upcoming & fresh]
    if wanted.empty:
        return 0
    inputs, player_part = build()
    fc = m5_forecasts(inputs, model, player_part, spec, season).set_index("game_id")
    rows_in = wanted.to_dict("records")
    ok = [
        r
        for r in rows_in
        if str(r["game_id"]) in fc.index
        and np.isfinite(fc.loc[str(r["game_id"]), ["p_home", "exp_margin", "exp_total"]]).all()
    ]
    if len(ok) < len(rows_in):
        log.warning("M5: %d upcoming games have no forecast", len(rows_in) - len(ok))
    predicted_at = clock()
    refuse_late(predicted_at, [(r["game_id"], r["tipoff_utc"]) for r in ok])
    rows = []
    for r in ok:
        f = fc.loc[str(r["game_id"])]
        rows.append(
            {
                "game_id": r["game_id"],
                "season": r["season"],
                "round": r["round"],
                "phase": r["phase"],
                "tipoff_utc": r["tipoff_utc"].strftime(TIME_FORMAT),
                "home": r["home"],
                "away": r["away"],
                "p_home": f"{f['p_home']:.4f}",
                "exp_margin": f"{f['exp_margin']:.2f}",
                "exp_total": f"{f['exp_total']:.2f}",
                "margin_sigma": f"{f['margin_sigma']:.4f}",
                "margin_df": "" if np.isnan(f["margin_df"]) else f"{f['margin_df']:g}",
                "total_sigma": f"{f['total_sigma']:.4f}",
                "model": "m5",
                "model_version": model.version,
                "predicted_at_utc": predicted_at.strftime(TIME_FORMAT),
            }
        )
    append_rows(log_path, M5_LOG_COLUMNS, rows)
    log.info("M5: %d rows appended to %s", len(rows), log_path)
    return len(rows)
