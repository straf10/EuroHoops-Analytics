"""Weeks 7-10b G1: the outcome-coding audit of both M2 feature builders catches a made-only or
missed-only tag in either builder, whatever its name, and passes the declared features."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest

from eurohoops.config import M2Seasons
from eurohoops.eval import m2_backtest
from eurohoops.models import feature_audit
from eurohoops.models.feature_audit import MIN_SET_SHOTS, Level, levels, outcome_coded_levels
from tests.test_xpts import synthetic_shots


def shots(n: int = 20_000, seed: int = 5) -> pd.DataFrame:
    frame = synthetic_shots(n, seed)
    rng = np.random.default_rng(seed + 1)
    p = 1.0 / (1.0 + np.exp(-(0.9 - 0.25 * frame["distance"])))
    return frame.assign(made=rng.random(n) < p, event=np.arange(n), validated_season=True)


def _tag(frame: pd.DataFrame, made_only: bool) -> np.ndarray:
    """A 'transition' tag set on a tenth of the makes (or of the misses) only."""
    target = frame["made"].to_numpy() == made_only
    return (target & (frame["event"].to_numpy() % 10 == 0)).astype(np.float64)


def test_the_declared_features_have_no_outcome_coded_level() -> None:
    assert outcome_coded_levels(shots()) == []


@pytest.mark.parametrize("made_only", [True, False])
def test_a_tag_in_the_lightgbm_builder_is_caught(
    monkeypatch: pytest.MonkeyPatch, made_only: bool
) -> None:
    original = feature_audit.features

    def with_tag(frame: pd.DataFrame) -> pd.DataFrame:
        return original(frame).assign(transition=_tag(frame, made_only))

    monkeypatch.setattr(feature_audit, "features", with_tag)
    found = outcome_coded_levels(shots())
    assert [(lv.builder, lv.column, lv.level) for lv in found] == [("lgbm", "transition", 1.0)]
    assert found[0].make_rate == (1.0 if made_only else 0.0)


def test_a_tag_in_the_spline_builder_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    original: Callable[..., tuple[np.ndarray, tuple[str, ...]]] = feature_audit.raw_design

    def with_tag(frame: pd.DataFrame, *args: Any) -> tuple[np.ndarray, tuple[str, ...]]:
        raw, names = original(frame, *args)
        return np.column_stack([raw, _tag(frame, True)]), (*names, "putback")

    monkeypatch.setattr(feature_audit, "raw_design", with_tag)
    found = outcome_coded_levels(shots())
    assert [(lv.builder, lv.column) for lv in found] == [("spline", "putback")]


def test_levels_rules() -> None:
    made = np.array([1.0] * 150 + [0.0] * 50)
    few = np.zeros(200)
    few[: MIN_SET_SHOTS - 1] = 1.0  # all makes, but too few shots to call it a tag
    columns = {
        ("x", "few"): few,
        ("x", "many"): np.r_[np.ones(150), np.zeros(50)],  # every make, no miss
        ("x", "continuous"): np.arange(200.0),  # more than MAX_LEVELS values: not checked
        ("x", "levels"): np.r_[np.full(150, 2.0), np.full(50, 3.0)],  # both levels checked
    }
    found = {(lv.column, lv.level): lv for lv in levels(columns, made)}
    assert set(found) == {("few", 1.0), ("many", 1.0), ("levels", 2.0), ("levels", 3.0)}
    assert not found[("few", 1.0)].outcome_coded
    assert found[("many", 1.0)].outcome_coded
    assert found[("levels", 2.0)].outcome_coded  # 150 makes only
    assert not found[("levels", 3.0)].outcome_coded  # misses only, but 50 < MIN_SET_SHOTS
    assert Level("x", "y", 1.0, 1000, 0.985).outcome_coded is False


def test_the_backtest_refuses_an_outcome_coded_feature(monkeypatch: pytest.MonkeyPatch) -> None:
    original = feature_audit.features
    monkeypatch.setattr(
        feature_audit, "features", lambda f: original(f).assign(transition=_tag(f, True))
    )
    seasons = M2Seasons(development=(2011, 2012), validation=(2013,), test=(2014,))
    frame = shots().assign(season=lambda f: 2011 + f["event"] % 4, band="rim")
    with pytest.raises(ValueError, match="outcome-coded"):
        m2_backtest.run_m2_backtest(frame, seasons, {}, False, lambda _: None)
