"""Walk-forward Elo backtest: warm-up, tune on one season, score the test season once.

Elo is replayed chronologically from the first warm-up game, so every prediction uses only
games that tipped off before it. The grid search is |grid| independent O(N) passes.
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from importlib.metadata import version
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eurohoops.config import TEST_SEASON, TUNING_SEASON, WARMUP_SEASON
from eurohoops.eval.metrics import paired_bootstrap_ci, per_game_log_loss, score
from eurohoops.models.elo import (
    EloParams,
    FloatArray,
    fit_margin_scale,
    prepare,
    replay,
    win_probability,
)

GRID_K = (15.0, 20.0, 30.0, 40.0)
GRID_HCA = (50.0, 70.0, 90.0, 110.0, 130.0)
GRID_REVERSION = (0.25, 0.5, 0.75)
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 20260924

win_probabilities = np.vectorize(win_probability, otypes=[np.float64])


@dataclass(frozen=True)
class TunedModel:
    params: EloParams
    margin_scale: float
    b0_home_win_rate: float
    b0_home_margin: float

    def version(self) -> str:
        """Short hash of the tuned parameters plus the package (code) version."""
        key = json.dumps({**asdict(self.params), "margin_scale": self.margin_scale}, sort_keys=True)
        digest = hashlib.sha256(key.encode()).hexdigest()[:8]
        return f"{version('eurohoops')}+{digest}"

    def b0(self, neutral: FloatArray) -> tuple[FloatArray, FloatArray]:
        """B0: constant home-win rate and home margin; neutral-venue games get 0.5 and 0."""
        p_home = np.where(neutral == 1.0, 0.5, self.b0_home_win_rate)
        exp_margin = np.where(neutral == 1.0, 0.0, self.b0_home_margin)
        return p_home, exp_margin


def load_tuned_model(report_path: Path) -> TunedModel:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    tuned = report["tuned"]
    return TunedModel(
        params=EloParams(k=tuned["k"], hca=tuned["hca"], reversion=tuned["reversion"]),
        margin_scale=tuned["margin_scale"],
        b0_home_win_rate=report["b0"]["home_win_rate"],
        b0_home_margin=report["b0"]["home_margin"],
    )


def _log_loss(diffs: FloatArray, home_won: FloatArray) -> float:
    return float(per_game_log_loss(win_probabilities(diffs), home_won).mean())


def run_backtest(games: pd.DataFrame) -> dict[str, Any]:
    history = games[games["played"] & games["season"].between(WARMUP_SEASON, TEST_SEASON)]
    arrays = prepare(history)
    season = history["season"].to_numpy()
    margin = (history["home_score"] - history["away_score"]).to_numpy(dtype=np.float64)
    home_won = (margin > 0).astype(np.float64)
    neutral = history["neutral"].to_numpy(dtype=np.float64)
    tuning, test = season == TUNING_SEASON, season == TEST_SEASON

    grid = [EloParams(*combo) for combo in product(GRID_K, GRID_HCA, GRID_REVERSION)]
    losses = [_log_loss(replay(arrays, params)[tuning], home_won[tuning]) for params in grid]
    best = grid[int(np.argmin(losses))]
    diffs = replay(arrays, best)

    fit_b0 = (season <= TUNING_SEASON) & (neutral == 0.0)
    model = TunedModel(
        params=best,
        margin_scale=round(fit_margin_scale(diffs[tuning], margin[tuning]), 6),
        b0_home_win_rate=round(float(home_won[fit_b0].mean()), 6),
        b0_home_margin=round(float(margin[fit_b0].mean()), 6),
    )
    p_elo = win_probabilities(diffs)
    exp_elo = diffs / model.margin_scale
    p_b0, exp_b0 = model.b0(neutral)

    metrics = {
        name: {
            "elo": score(p_elo[mask], exp_elo[mask], margin[mask]).as_dict(),
            "b0": score(p_b0[mask], exp_b0[mask], margin[mask]).as_dict(),
        }
        for name, mask in (("tuning", tuning), ("test", test))
    }
    loss_diff = per_game_log_loss(p_elo[test], home_won[test]) - per_game_log_loss(
        p_b0[test], home_won[test]
    )
    mean, low, high = paired_bootstrap_ci(loss_diff, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED)
    return {
        "model": "elo",
        "model_version": model.version(),
        "seasons": {"warmup": WARMUP_SEASON, "tuning": TUNING_SEASON, "test": TEST_SEASON},
        "games_per_season": {
            str(s): int(n) for s, n in history["season"].value_counts().sort_index().items()
        },
        "grid": {
            "k": list(GRID_K),
            "hca": list(GRID_HCA),
            "reversion": list(GRID_REVERSION),
            "size": len(grid),
        },
        "tuned": {
            **asdict(best),
            "margin_scale": model.margin_scale,
            "tuning_log_loss": round(min(losses), 6),
            "home_win_prob_equal_ratings": round(win_probability(best.hca), 6),
        },
        "b0": {"home_win_rate": model.b0_home_win_rate, "home_margin": model.b0_home_margin},
        "metrics": metrics,
        "test_log_loss_diff_elo_minus_b0": {
            "mean": round(mean, 6),
            "ci95": [round(low, 6), round(high, 6)],
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
        },
    }


def format_table(report: dict[str, Any]) -> str:
    header = f"{'split':<8}{'model':<6}{'n':>5}{'logloss':>9}{'brier':>8}{'acc':>7}{'mae':>7}"
    lines = [header, "-" * len(header)]
    for split, models in report["metrics"].items():
        for name, m in models.items():
            lines.append(
                f"{split:<8}{name:<6}{m['n']:>5}{m['log_loss']:>9.4f}{m['brier']:>8.4f}"
                f"{m['accuracy']:>7.3f}{m['margin_mae']:>7.2f}"
            )
    tuned, ci = report["tuned"], report["test_log_loss_diff_elo_minus_b0"]
    lines.append(
        f"tuned: K={tuned['k']:g} HCA={tuned['hca']:g} reversion={tuned['reversion']:g} "
        f"s={tuned['margin_scale']:.2f} (home win at equal ratings "
        f"{tuned['home_win_prob_equal_ratings']:.3f})"
    )
    lines.append(
        f"test logloss Elo-B0: {ci['mean']:+.4f} "
        f"95% CI [{ci['ci95'][0]:+.4f}, {ci['ci95'][1]:+.4f}]"
    )
    return "\n".join(lines)
