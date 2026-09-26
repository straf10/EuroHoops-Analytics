"""F9 leakage tests for M2: out-of-fold values never depend on their own season, and nothing
fitted on development seasons depends on validation or test shots; guards prove the edits
reach what they should."""

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from eurohoops.eval.m2_backtest import Fitter, Folds, calibrate, gbm_fitter, spline_fitter
from tests.test_m2_backtest import TINY, labelled

DEVELOPMENT = (2011, 2012, 2013, 2014)
LATER = (2015, 2016)


def _data() -> tuple[pd.DataFrame, pd.DataFrame]:
    shots = labelled(4_000, 21, (*DEVELOPMENT, *LATER))
    dev = shots[shots["season"].isin(DEVELOPMENT)].reset_index(drop=True)
    later = shots[shots["season"].isin(LATER)].reset_index(drop=True)
    return dev, later


def _flip(frame: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    return frame.assign(made=np.where(mask, ~frame["made"], frame["made"]))


def _fitters() -> list[tuple[str, Fitter]]:
    out = [("spline", spline_fitter(4, 1e-4))]
    if pytest.importorskip("lightgbm", reason="dev dependency") is not None:
        out.append(("lgbm", gbm_fitter(TINY, 20261001)))
    return out


def _run(fit: Fitter, dev: pd.DataFrame, later: pd.DataFrame) -> dict[str, np.ndarray]:
    folds = Folds(dev, DEVELOPMENT, fit, True, later)
    y = dev["made"].to_numpy(dtype=np.float64)
    iso_oof, iso_later = calibrate(folds, y)
    return {"oof": folds.oof, "later": folds.later, "iso_oof": iso_oof, "iso_later": iso_later}


@pytest.mark.parametrize(("name", "fit"), _fitters())
def test_editing_a_season_changes_none_of_its_own_out_of_fold_values(
    name: str, fit: Fitter
) -> None:
    dev, later = _data()
    base = _run(fit, dev, later)
    for season in (2011, 2013):
        in_season = dev["season"] == season
        edited = _run(fit, _flip(dev, in_season & (dev["event"] % 3 == 0)), later)
        own = in_season.to_numpy()
        for key in ("oof", "iso_oof"):
            np.testing.assert_array_equal(edited[key][own], base[key][own], err_msg=f"{name} {key}")
            # guard: the edit does move the other seasons' out-of-fold values
            assert not np.array_equal(edited[key][~own], base[key][~own]), f"{name} {key} guard"


@pytest.mark.parametrize(("name", "fit"), _fitters())
def test_editing_validation_or_test_shots_changes_nothing_fitted_on_development(
    name: str, fit: Fitter
) -> None:
    dev, later = _data()
    base = _run(fit, dev, later)
    moved_later = later.assign(made=~later["made"])  # every validation/test label flipped
    edited = _run(fit, dev, moved_later)
    for key in base:
        np.testing.assert_array_equal(edited[key], base[key], err_msg=f"{name} {key}")
    # guard: a development edit does reach the development-fitted later predictions
    dev_edit = _run(fit, _flip(dev, dev["event"] % 2 == 0), later)
    assert not np.array_equal(dev_edit["later"], base["later"])
    assert not np.array_equal(dev_edit["iso_later"], base["iso_later"])


def test_later_predictions_follow_their_own_features() -> None:
    """Guard for the test above: a later shot's own features do change its prediction."""
    dev, later = _data()
    fit: Callable[[pd.DataFrame], Callable[[pd.DataFrame], np.ndarray]] = spline_fitter(4, 1e-4)
    folds = Folds(dev, DEVELOPMENT, fit, False, later)
    moved = Folds(dev, DEVELOPMENT, fit, False, later.assign(distance=later["distance"] + 1.0))
    assert not np.array_equal(folds.later, moved.later)
