"""F9 leakage tests for M2: out-of-fold values never depend on their own season, and nothing
fitted on development seasons depends on validation or test shots; guards prove the edits
reach what they should."""

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from eurohoops.eval.m2_backtest import Fitter, Folds, calibrate, gbm_fitter, spline_fitter
from eurohoops.models.season_level import LevelPredictions, level_predictions
from tests.test_m2_backtest import TINY, labelled, tipoffs

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


def _level(fit: Fitter, dev: pd.DataFrame, later: pd.DataFrame) -> LevelPredictions:
    """The G5 season-level predictions of one base, as the backtest computes them."""
    folds = Folds(dev, DEVELOPMENT, fit, True, later)
    ticks = pd.to_datetime(tipoffs(pd.concat([dev, later])), utc=True)
    ticks = ticks.dt.tz_convert(None).astype("int64")
    return level_predictions(
        folds,
        dev["made"].to_numpy(dtype=np.float64),
        dev["game_id"].map(ticks).to_numpy(dtype=np.int64),
        later_season=later["season"].to_numpy(dtype=np.int64),
        y_later=later["made"].to_numpy(dtype=np.float64),
        tip_later=later["game_id"].map(ticks).to_numpy(dtype=np.int64),
        development=DEVELOPMENT,
    )


def _with_season_levels(frame: pd.DataFrame) -> pd.DataFrame:
    """Outcomes redrawn with a season-level logit shift (+-0.4 alternating), so the seasons
    differ in level beyond noise and k is finite (else the offset is the prior alone)."""
    rng = np.random.default_rng(99)
    shift = np.where(frame["season"].to_numpy() % 2 == 0, 0.4, -0.4)
    p = 1.0 / (1.0 + np.exp(-(0.9 - 0.25 * frame["distance"].to_numpy() + shift)))
    return frame.assign(made=rng.random(len(frame)) < p)


def _game(frame: pd.DataFrame) -> pd.Series:
    return frame["game_id"].str.split("_").str[1].astype(int)


@pytest.mark.parametrize(("name", "fit"), _fitters())
def test_no_level_offset_uses_its_own_game_or_a_later_one(name: str, fit: Fitter) -> None:
    """G5: flipping every shot of game g and later in a season changes no season-level
    prediction (nor offset) of that season's games up to g, through any route: the season's own
    earlier games, the prior (the previous season under a model without this season) or k."""
    dev, later = (_with_season_levels(f) for f in _data())
    base = _level(fit, dev, later)
    g = 20
    for season in (2012, 2014):
        edit = (dev["season"] == season) & (_game(dev) >= g)
        moved = _level(fit, _flip(dev, edit), later)
        keep = ((dev["season"] == season) & (_game(dev) <= g)).to_numpy()
        for key in ("oof", "offsets_oof"):
            np.testing.assert_array_equal(
                getattr(moved, key)[keep], getattr(base, key)[keep], err_msg=f"{name} {key}"
            )
        after = ((dev["season"] == season) & (_game(dev) > g)).to_numpy()
        assert not np.array_equal(moved.offsets_oof[after], base.offsets_oof[after]), "guard"
    edit = (later["season"] == 2016) & (_game(later) >= g)
    moved = _level(fit, dev, _flip(later, edit))
    keep = ((later["season"] == 2016) & (_game(later) <= g)).to_numpy()
    np.testing.assert_array_equal(moved.later[keep], base.later[keep], err_msg=f"{name} later")
    after = ((later["season"] == 2016) & (_game(later) > g)).to_numpy()
    assert not np.array_equal(moved.offsets_later[after], base.offsets_later[after]), "guard"
