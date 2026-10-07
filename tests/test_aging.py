"""L2: aging curve: planted-curve recovery, survivor correction, privacy, leakage, ``apply``."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from eurohoops.models.aging import MIN_AGE_PAIRS, AgingCurve, aging_curve, apply
from eurohoops.models.player_seasons import COUNT_COLUMNS, PROJECTED_STATS

N_PLAYERS = 2500
CAREER = 4  # seasons per player
FIRST_SEASON = 2010
AGE_LOW, AGE_HIGH = 19, 33  # first-season ages
MU, TALENT_SD, CHANGE_SD = (
    30.0,
    4.0,
    2.5,
)  # ast per 100: league mean, talent spread, spread of true changes
DROP_BELOW = -1.2 * CHANGE_SD  # a true change below this leaves the player a short next season
FULL_POSS, SHORT_POSS = 1200.0, 450.0
CUTOFF = FIRST_SEASON + CAREER  # all four seasons of every player are complete before it


def planted(age: np.ndarray) -> np.ndarray:
    """True expected change of ast per 100 from age a to a + 1: convex, falling, then negative."""
    return 2.0 * np.exp(-(age - 19.0) / 4.0) - 0.8


def _population(seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    start_age = rng.integers(AGE_LOW, AGE_HIGH + 1, N_PLAYERS)
    talent = MU + TALENT_SD * rng.standard_normal(N_PLAYERS)
    rows = []
    ages = []
    short = np.zeros(N_PLAYERS, dtype=bool)
    for k in range(CAREER):
        age = start_age + k
        poss = np.where(short, SHORT_POSS, FULL_POSS) * np.ones(N_PLAYERS)
        ast = rng.poisson(poss * talent / 100.0)
        fg2a = rng.poisson(poss * 0.10)
        fg3a = rng.poisson(poss * 0.05)
        fta = rng.poisson(poss * 0.04)
        fg2m, fg3m, ftm = (rng.binomial(n, p) for n, p in ((fg2a, 0.5), (fg3a, 0.35), (fta, 0.75)))
        frame = pd.DataFrame(
            {
                "person_id": [f"P:{i}" for i in range(N_PLAYERS)],
                "competition": "euroleague",
                "season": FIRST_SEASON + k,
                "partial": False,
                "mapped": True,
                "team": "AAA",
                "games": 20,
                "minutes": poss / 2.0,
                "poss": poss,
                **{c: 0 for c in COUNT_COLUMNS},
                "debut_season": FIRST_SEASON,
            }
        )
        frame["ast"] = ast
        frame["dreb"] = rng.poisson(poss * 0.08)
        frame["fg2a"], frame["fg2m"] = fg2a, fg2m
        frame["fg3a"], frame["fg3m"] = fg3a, fg3m
        frame["fta"], frame["ftm"] = fta, ftm
        frame["pts"] = 2 * fg2m + 3 * fg3m + ftm
        rows.append(frame)
        ages.append(
            pd.DataFrame(
                {
                    "person_id": frame["person_id"],
                    "season": FIRST_SEASON + k,
                    "age": age + rng.uniform(-0.3, 0.3, N_PLAYERS),  # synthetic, fractional
                }
            )
        )
        spread = CHANGE_SD * rng.standard_normal(N_PLAYERS)  # the player's own deviation
        talent = talent + planted(age.astype(float)) + spread
        short = spread < DROP_BELOW
    return pd.concat(rows, ignore_index=True), pd.concat(ages, ignore_index=True)


@pytest.fixture(scope="module")
def population() -> tuple[pd.DataFrame, pd.DataFrame]:
    return _population()


@pytest.fixture(scope="module")
def curves(population: tuple[pd.DataFrame, pd.DataFrame]) -> dict[bool, AgingCurve]:
    history, ages = population
    return {on: aging_curve(history, ages, CUTOFF, survivor_correction=on) for on in (True, False)}


def _signed_error(curve: AgingCurve) -> tuple[np.ndarray, np.ndarray]:
    ok = curve.n_pairs["ast"] >= 100
    err = curve.delta["ast"] - planted(curve.ages)
    return err[ok], curve.se["ast"][ok]


def test_planted_curve_is_recovered_with_the_correction(
    curves: dict[bool, AgingCurve],
) -> None:
    curve = curves[True]
    assert curve.ages[0] >= AGE_LOW - 1 and curve.ages[-1] <= AGE_HIGH + CAREER
    err, se = _signed_error(curve)
    assert len(err) >= 10
    # Every age with >= 100 pairs within 3 SE of the planted curve (a 0.3% miss each under the
    # normal; 8 other seeds peak at 2.0 SE), and the mean error inside 3 SE of the mean SE (the
    # spline pools several ages, so the mean error's own SE is smaller than the mean SE).
    assert np.all(np.abs(err) < 3 * se)
    assert abs(err.mean()) < 3 * se.mean()
    # The stat with no planted change stays at zero within its band.
    ok = curve.n_pairs["dreb"] >= 100
    assert np.all(np.abs(curve.delta["dreb"][ok]) < 3 * curve.se["dreb"][ok])


def test_dropping_survivors_biases_the_curve_upward_and_the_correction_removes_it(
    curves: dict[bool, AgingCurve],
) -> None:
    err_on, se_on = _signed_error(curves[True])
    err_off, se_off = _signed_error(curves[False])
    # Dropping the players whose true rate fell the most pushes the mean change up, by more
    # than the band, at essentially every well-sampled age.
    assert err_off.mean() > 3 * se_off.mean()
    assert np.mean(err_off > 2 * se_off) > 0.8
    # With the correction (pairs weighted by their season-s exposure, not the outcome's) the
    # bias is a fraction of that and inside the band.
    assert abs(err_on.mean()) < 0.5 * err_off.mean()
    assert abs(err_on.mean()) < 3 * se_on.mean()


def test_curve_holds_ages_and_deltas_only(curves: dict[bool, AgingCurve]) -> None:
    curve = curves[True]
    assert [f.name for f in dataclasses.fields(curve)] == [
        "cutoff_season",
        "ages",
        "delta",
        "se",
        "n_pairs",
    ]
    assert np.array_equal(curve.ages, np.round(curve.ages))
    assert np.all(np.diff(curve.ages) == 1.0)
    assert np.sum(curve.n_pairs["pts"][[0, -1]] >= MIN_AGE_PAIRS) == 2
    for mapping in (curve.delta, curve.se, curve.n_pairs):
        assert set(mapping) == set(PROJECTED_STATS) - {"spm", "brapm"}
        # One value per grid age, not per person.
        assert all(a.shape == curve.ages.shape for a in mapping.values())
    assert curve.ages.shape[0] < N_PLAYERS


def test_rows_at_or_after_the_cutoff_and_partial_rows_change_nothing(
    population: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    history, ages = population
    cutoff = CUTOFF - 1
    base = aging_curve(history, ages, cutoff)
    rng = np.random.default_rng(3)
    late = history["season"] >= cutoff
    noisy = history.copy()
    for column in ("ast", "pts", "poss", "dreb"):
        noisy.loc[late, column] = rng.permutation(noisy.loc[late, column].to_numpy()) + 1.0
    noisy["poss"] = noisy["poss"].astype("float64")
    flagged = history.copy()
    flagged.loc[flagged["season"] == cutoff - 1, ["partial", "ast"]] = [True, 0]
    for other in (noisy, pd.concat([history[~late], noisy[late]])):
        again = aging_curve(other, ages, cutoff)
        for stat in base.delta:
            assert np.array_equal(again.delta[stat], base.delta[stat])
            assert np.array_equal(again.se[stat], base.se[stat])
    # A partial season (cut at a checkpoint) is not a pair member: marking one changes the fit
    # only by removing it, so it must equal the fit without those rows.
    without = aging_curve(history[history["season"] < cutoff - 1], ages, cutoff)
    flagged_fit = aging_curve(flagged, ages, cutoff)
    assert np.array_equal(flagged_fit.delta["ast"], without.delta["ast"])
    assert np.array_equal(flagged_fit.se["ast"], without.se["ast"])
    # An earlier pair does matter.
    early = history.copy()
    early.loc[early["season"] == FIRST_SEASON, "ast"] += 40
    assert not np.array_equal(aging_curve(early, ages, cutoff).delta["ast"], base.delta["ast"])


def _hand_curve() -> AgingCurve:
    ages = np.array([20.0, 21.0, 22.0, 23.0])
    delta = np.array([3.0, 2.0, 1.0, -1.0])
    zero = np.zeros(4)
    return AgingCurve(2020, ages, {"x": delta}, {"x": zero}, {"x": np.full(4, 50, dtype="int64")})


def test_apply_hand_examples() -> None:
    curve = _hand_curve()

    def go(a: list[float], b: list[float]) -> np.ndarray:
        return apply(curve, "x", np.array(a), np.array(b))

    # Integer ages: sums of the deltas at the ages passed.
    assert go([20.0, 21.0, 20.0], [21.0, 23.0, 23.0]) == pytest.approx([3.0, 3.0, 6.0])
    # Fractional ages: linear between grid ages (20 -> 20.5 is half of the 20 -> 21 change
    # under a piecewise-linear cumulative curve; 20.5 -> 21.5 is half of each of the two).
    assert go([20.0, 20.5], [20.5, 21.5]) == pytest.approx([1.5, 2.5])
    # Beyond the ends the end delta continues: below 20 the first delta, above 23 the last.
    assert go([18.0, 24.0, 18.0], [20.0, 26.0, 26.0]) == pytest.approx([6.0, -2.0, 6.0 + 6.0 - 3.0])
    # Equal ages give zero; reversed order is the negative of the forward sum.
    assert go([21.3, 25.0], [21.3, 25.0]) == pytest.approx([0.0, 0.0])
    assert go([23.0, 20.5], [20.0, 18.0]) == pytest.approx([-6.0, -7.5])


def test_impact_stat_with_a_planted_linear_age_effect_is_recovered(
    population: tuple[pd.DataFrame, pd.DataFrame], curves: dict[bool, AgingCurve]
) -> None:
    history, ages = population
    assert "spm" not in curves[True].delta and "brapm" not in curves[True].delta
    rng = np.random.default_rng(11)
    noise_sd = 1.0
    start = ages[ages["season"] == FIRST_SEASON].set_index("person_id")["age"].round()
    level = pd.Series(rng.standard_normal(len(start)) * 2.0, index=start.index)
    frames = []
    for k in range(CAREER):
        frames.append(
            pd.DataFrame(
                {
                    "person_id": start.index,
                    "competition": "euroleague",
                    "season": FIRST_SEASON + k,
                    "stat": "spm",
                    "value": level.to_numpy() + noise_sd * rng.standard_normal(len(start)),
                    "sd": noise_sd,
                }
            )
        )
        level = level + (1.0 - 0.1 * (start + k - 19.0))  # planted: linear in age
    curve = aging_curve(history, ages, CUTOFF, impact=pd.concat(frames, ignore_index=True))
    ok = curve.n_pairs["spm"] >= 100
    err = curve.delta["spm"] - (1.0 - 0.1 * (curve.ages - 19.0))
    # Within 3 SE at every well-sampled age; the spline's SE is about noise_sd * sqrt(2) / sqrt(n).
    assert ok.sum() >= 10
    assert np.all(np.abs(err[ok]) < 3 * curve.se["spm"][ok])
    assert np.all(curve.se["spm"][ok] < 0.3)
    assert "brapm" not in curve.delta
