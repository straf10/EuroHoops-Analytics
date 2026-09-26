"""Outcome-coding audit of both M2 feature builders (weeks 7-10b, G1).

The feed's ``FASTBREAK``, ``SECOND_CHANCE`` and ``POINTS_OFF_TURNOVER`` flags looked like shot
context but are set only on made shots from 2015-16 (scoring tags, i.e. the outcome); they
were dropped from M2 for good (user decision, 2026-09-26). A name guard keeps those three out;
this audit catches the next one by its values, whatever its name.

Every column either builder produces (the spline's raw design, before whitening, and the
LightGBM feature frame) that takes at most ``MAX_LEVELS`` distinct values is a flag or a set of
levels: a 0/1 column is checked where it is set (1), any other such column level by level (a
one-hot level of it). A level fails when it covers at least ``MIN_SET_SHOTS`` shots and its
make rate is ``>= MAX_MAKE_RATE`` (made-only tags) or ``<= 1 - MAX_MAKE_RATE`` (missed-only
tags, e.g. a blocked-shot code). Continuous columns (distance, the spline basis, the clock,
the margin) have no level where a tag could hide and are not checked.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from eurohoops.models.elo import FloatArray
from eurohoops.models.xpts import make_spec, raw_design
from eurohoops.models.xpts_gbm import features

MAX_MAKE_RATE = 0.99
MIN_SET_SHOTS = 100  # fewer shots than this cannot show a tag apart from chance
MAX_LEVELS = 32
AUDIT_KNOTS = 4  # the spline basis is continuous and unchecked, so the knot count is moot


@dataclass(frozen=True)
class Level:
    builder: str
    column: str
    level: float
    n: int
    make_rate: float

    @property
    def outcome_coded(self) -> bool:
        return self.n >= MIN_SET_SHOTS and (
            self.make_rate >= MAX_MAKE_RATE or self.make_rate <= 1.0 - MAX_MAKE_RATE
        )


def builder_columns(shots: pd.DataFrame) -> dict[tuple[str, str], FloatArray]:
    """Every column of both feature builders on ``shots``, keyed by (builder, column)."""
    spec = make_spec(shots, AUDIT_KNOTS)
    raw, names = raw_design(shots, spec.knots_two, spec.knots_three, spec.zones)
    out = {("spline", name): raw[:, i] for i, name in enumerate(names)}
    frame = features(shots)
    out |= {("lgbm", str(c)): frame[c].to_numpy(dtype=np.float64) for c in frame.columns}
    return out


def levels(columns: dict[tuple[str, str], FloatArray], made: FloatArray) -> list[Level]:
    """Make rate of every checked level (see the module doc)."""
    out = []
    for (builder, column), values in columns.items():
        distinct = np.unique(values)
        if len(distinct) > MAX_LEVELS:
            continue
        checked = [1.0] if set(distinct.tolist()) <= {0.0, 1.0} else distinct.tolist()
        for level in checked:
            on = values == level
            n = int(on.sum())
            if n:
                out.append(Level(builder, column, level, n, float(made[on].mean())))
    return out


def outcome_coded_levels(shots: pd.DataFrame) -> list[Level]:
    """The levels of either builder that look like outcome tags on ``shots``."""
    made = shots["made"].to_numpy(dtype=np.float64)
    return [lv for lv in levels(builder_columns(shots), made) if lv.outcome_coded]
