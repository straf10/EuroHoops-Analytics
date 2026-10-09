"""M7 gate v2 (docs/models/m7.md, "Gate v2", declared 2026-10-09 before any run; PLAN 8.1 R5).

Rolling origin, expanding window. The seasons are ``spec.tuning`` (the first is a selection
season only; every later one is scored). For each scored season (the origin) the variant
(``sim_full``, ``sim_net``, ``sim_inflate_*``) is the one with the lowest pooled RPS of the final
place on all EARLIER seasons of the list, then that season is scored with it. Checkpoints,
targets, simulation counts and seeds are v1's (``run_m7_backtest`` over the same season list:
the plan, hence every seed, is v1's full-run plan); M1's parameters are the committed ones.

Pass requires all three, on the scored team-checkpoints pooled over the origins:

1. non-inferiority to ``point_sim``: the season-cluster bootstrap 95% CI (whole seasons
   resampled with replacement, ``spec.bootstrap_resamples`` resamples, ``spec.bootstrap_seed``)
   of the mean per-row Brier difference of the direct cut (chosen - point_sim) has an upper
   bound below ``margin``;
2. pooled Brier of the chosen variants below ``standings_now``'s;
3. pooled Spiegelhalter |z| of the chosen variants' direct-cut probabilities below 1.96.

The choice has no tie tolerance (the card says "lowest"); an exact tie goes to the simpler.
"""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from eurohoops.config import GATE_V2_MARGIN, M7Backtest
from eurohoops.eval.m7_backtest import (
    BASELINES,
    Z_LIMIT,
    M7Inputs,
    _metrics,
    _round,
    _scores,
    cluster_bootstrap_ci,
    run_m7_backtest,
    variants,
)
from eurohoops.sim.formats import season_format
from eurohoops.standings import Format

GATE_V2_RULE = (
    "rolling origin, expanding window: per scored season the variant with the lowest pooled "
    "final-place RPS on all earlier seasons; pass = (a) season-cluster bootstrap 95% CI upper "
    "bound of the pooled direct-cut Brier difference (chosen - point_sim) below the margin, "
    "(b) pooled Brier below standings_now, (c) pooled Spiegelhalter |z| < 1.96"
)
KEY = ["season", "checkpoint", "team"]


def earlier_seasons(seasons: Sequence[int], origin: int) -> list[int]:
    """The seasons a choice at ``origin`` may see: strictly before it."""
    return [s for s in seasons if s < origin]


@dataclass(frozen=True)
class Origin:
    """One scored season: the seasons the choice saw, the pooled RPS of every candidate on them
    (candidate order) and the chosen key."""

    season: int
    seen: tuple[int, ...]
    rps: dict[str, float]
    chosen: str


def pooled_rps(rows: pd.DataFrame, model: str, seasons: Sequence[int]) -> float:
    """Mean RPS of the model's team-checkpoint rows in ``seasons``; NaN without any."""
    mine = rows[(rows["model"] == model) & rows["season"].isin(list(seasons))]
    return float(mine["rps"].mean()) if len(mine) else math.nan


def origin_choices(
    rows: pd.DataFrame,
    seasons: Sequence[int],
    candidates: Sequence[str],
    seen: Callable[[Sequence[int], int], list[int]] | None = None,
) -> list[Origin]:
    """The variant chosen at every scored season (``seasons[1:]``): the lowest pooled RPS over
    ``seen(seasons, origin)`` (default: the earlier seasons), the simplest (first listed) on an
    exact tie. ``seen`` is the one place a leak could enter, so tests replace it."""
    look = earlier_seasons if seen is None else seen
    out = []
    for origin in seasons[1:]:
        eligible = look(seasons, origin)
        table = {c: pooled_rps(rows, c, eligible) for c in candidates}
        if not all(math.isfinite(v) for v in table.values()):
            raise ValueError(f"no finite pooled RPS for every variant before season {origin}")
        best = min(table.values())
        chosen = next(c for c in candidates if table[c] == best)
        out.append(Origin(origin, tuple(eligible), table, chosen))
    return out


def chosen_rows(rows: pd.DataFrame, origins: Sequence[Origin]) -> pd.DataFrame:
    """The rows of each origin's chosen variant in that season, as one frame sorted by key."""
    parts = [rows[(rows["model"] == o.chosen) & (rows["season"] == o.season)] for o in origins]
    return (
        pd.concat(parts, ignore_index=True).sort_values(KEY, kind="stable").reset_index(drop=True)
    )


def _aligned(rows: pd.DataFrame, model: str, seasons: Sequence[int]) -> pd.DataFrame:
    mine = rows[(rows["model"] == model) & rows["season"].isin(list(seasons))]
    return mine.sort_values(KEY, kind="stable").reset_index(drop=True)


def season_cluster_ci(
    mine: pd.DataFrame, other: pd.DataFrame, resamples: int, seed: int
) -> dict[str, Any]:
    """Pooled mean per-row Brier difference of the direct cut (mine - other) with the percentile
    95% CI over resamples of WHOLE seasons (every checkpoint and team of a season together)."""
    if not mine[KEY].equals(other[KEY]):
        raise ValueError("the models' rows are not aligned")
    y = mine["made_direct"].to_numpy(dtype=np.float64)
    diff = (mine["p_direct"].to_numpy() - y) ** 2 - (other["p_direct"].to_numpy() - y) ** 2
    cluster = pd.factorize(mine["season"])[0].astype(np.int64)
    mean, low, high = cluster_bootstrap_ci(diff, cluster, resamples, seed)
    return {
        "mean": mean,
        "ci95": [low, high],
        "clusters": int(cluster.max()) + 1,
        "resamples": resamples,
        "seed": seed,
    }


def gate_v2_block(
    *,
    ci_high: float,
    brier_chosen: float,
    brier_standings_now: float,
    z_pooled: float | None,
    margin: float = GATE_V2_MARGIN,
) -> dict[str, Any]:
    """The three criteria as declared, from unrounded values; ``passed`` needs all three."""
    non_inferior = bool(ci_high < margin)
    beats_table = bool(brier_chosen < brier_standings_now)
    calibrated = bool(z_pooled is not None and abs(z_pooled) < Z_LIMIT)
    return {
        "a_non_inferior_to_point_sim": non_inferior,
        "b_beats_standings_now": beats_table,
        "c_calibrated": calibrated,
        "margin": margin,
        "z_limit": Z_LIMIT,
        "passed": non_inferior and beats_table and calibrated,
    }


def evaluate(
    rows: pd.DataFrame, seasons: Sequence[int], spec: M7Backtest
) -> tuple[list[Origin], dict[str, Any]]:
    """Origins and the ``gate_v2`` report block from the rows of every model and season."""
    candidates = [v.key for v in variants(spec)]
    origins = origin_choices(rows, seasons, candidates)
    scored = [o.season for o in origins]
    mine = chosen_rows(rows, origins)
    point = _aligned(rows, "point_sim", scored)
    table = _aligned(rows, "standings_now", scored)
    vs_point = season_cluster_ci(mine, point, spec.bootstrap_resamples, spec.bootstrap_seed)
    vs_table = season_cluster_ci(mine, table, spec.bootstrap_resamples, spec.bootstrap_seed)
    pooled = _scores(mine)
    criteria = gate_v2_block(
        ci_high=vs_point["ci95"][1],
        brier_chosen=pooled["brier"],
        brier_standings_now=_scores(table)["brier"],
        z_pooled=pooled["z"],
    )

    def rounded(diff: dict[str, Any]) -> dict[str, Any]:
        return {
            **diff,
            "mean": _round(diff["mean"]),
            "ci95": [_round(x) for x in diff["ci95"]],
        }

    block: dict[str, Any] = {
        "rule": GATE_V2_RULE,
        "selection_seasons": [int(seasons[0])],
        "scored_seasons": scored,
        "n_team_checkpoints": len(mine),
        "origins": [
            {
                "season": o.season,
                "seen_seasons": list(o.seen),
                "chosen": o.chosen,
                "pooled_rps": {k: _round(v) for k, v in o.rps.items()},
            }
            for o in origins
        ],
        "pooled": {
            "chosen": _metrics(mine),
            **{m: _metrics(_aligned(rows, m, scored)) for m in BASELINES},
            **{c: _metrics(_aligned(rows, c, scored)) for c in candidates},
        },
        "per_season": {
            str(o.season): {
                "chosen": o.chosen,
                "chosen_metrics": _metrics(mine[mine["season"] == o.season]),
                "point_sim": _metrics(point[point["season"] == o.season]),
                "standings_now": _metrics(table[table["season"] == o.season]),
            }
            for o in origins
        },
        "brier_vs_point_sim": rounded(vs_point),
        "brier_vs_standings_now": rounded(vs_table),
        "spiegelhalter_z_pooled": _round(pooled["z"]),
        "brier_chosen": _round(pooled["brier"]),
        "brier_point_sim": _round(_scores(point)["brier"]),
        "brier_standings_now": _round(_scores(table)["brier"]),
        "criteria": criteria,
        "passed": criteria["passed"],
    }
    return origins, block


def relabel_splits(rows: pd.DataFrame, first: int) -> pd.DataFrame:
    """The CSV rows with the split column ``selection`` (the first season) or ``scored``."""
    out = rows.copy()
    out["split"] = np.where(out["season"] == first, "selection", "scored")
    return out


def run_gate_v2(
    inputs: M7Inputs,
    *,
    spec: M7Backtest,
    formats: Callable[[str, int], Format] = season_format,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Simulate every season of ``spec.tuning`` with every model (v1's harness, its tuning-only
    mode), then evaluate gate v2. Returns the report and the per team-checkpoint rows."""
    seasons = sorted(spec.tuning)
    if len(seasons) < 2 or spec.validation or spec.test:
        raise ValueError("gate v2 needs a selection season, scored seasons and no other split")
    base, rows = run_m7_backtest(inputs, spec=spec, formats=formats, tuning_only=True)
    origins, block = evaluate(rows, seasons, spec)
    report: dict[str, Any] = {
        "model": "m7",
        "gate": "v2",
        "model_version": "m7-gate-v2",
        "competition": inputs.competition,
        "seasons": seasons,
        "checkpoints": base["checkpoints"],
        "n_sims": base["n_sims"],
        "seeds": base["seeds"],
        "formats": base["formats"],
        "data_sha256": base["data_sha256"],
        "candidates": [v.key for v in variants(spec)],
        "tie_tolerance": None,
        "chosen_per_origin": {str(o.season): o.chosen for o in origins},
        "gate_v2": block,
        "selection_season_metrics": {
            m: _metrics(rows[(rows["season"] == seasons[0]) & (rows["model"] == m)])
            for m in (*[v.key for v in variants(spec)], *BASELINES)
        },
    }
    return report, relabel_splits(rows, seasons[0])


def format_gate_v2(report: Mapping[str, Any]) -> str:
    """A short console summary of a gate v2 report."""
    g = report["gate_v2"]
    c = g["criteria"]

    def mark(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    lines = [f"gate v2 over {g['scored_seasons']} ({g['n_team_checkpoints']} team-checkpoints)"]
    lines += [
        f"  origin {o['season']}: {o['chosen']} (saw {o['seen_seasons']})" for o in g["origins"]
    ]
    d = g["brier_vs_point_sim"]
    lines += [
        f"  (a) chosen - point_sim {d['mean']} CI {d['ci95']} < {c['margin']}: "
        f"{mark(c['a_non_inferior_to_point_sim'])}",
        f"  (b) Brier {g['brier_chosen']} < standings_now {g['brier_standings_now']}: "
        f"{mark(c['b_beats_standings_now'])}",
        f"  (c) pooled z {g['spiegelhalter_z_pooled']}: {mark(c['c_calibrated'])}",
        f"  verdict: {mark(g['passed'])}",
    ]
    return "\n".join(lines)
