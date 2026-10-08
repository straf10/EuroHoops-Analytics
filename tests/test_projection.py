"""M6 projections (weeks 16-18 L1): calibration on synthetic leagues with known true rates,
shrinkage limits, the prior-only limit, the leakage rule, and hand examples for translation,
aging, impact and the variants. Synthetic data only; ages are synthetic."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pandera.pandas as pa
import pytest
from scipy import special

from eurohoops.models.player_seasons import (
    COUNT_COLUMNS,
    COUNT_STATS,
    IMPACT_STATS,
    PCT_STATS,
    PLAYER_SEASONS_SCHEMA,
    PROJECTED_STATS,
    rate_table,
)
from eurohoops.models.projection import (
    FLAGS,
    IMPACT_SCHEMA,
    PROJECTIONS_SCHEMA,
    TARGET_SCHEMA,
    VARIANTS,
    ProjectionParams,
    Translation,
    fit_drift,
    project,
    variant_params,
)
from eurohoops.parse.schemas import validated

Z80 = float(special.ndtri(0.9))

# stat -> (league mean, between-player sd, drift sd per season) of the synthetic true rates
COUNT_PRIOR = {
    "pts": (20.0, 4.0, 1.0),
    "fg3a": (7.0, 2.5, 0.6),
    "fta": (6.0, 2.0, 0.5),
    "ast": (10.0, 3.0, 0.7),
    "tov": (6.0, 1.5, 0.4),
    "oreb": (4.0, 1.5, 0.4),
    "dreb": (9.0, 2.5, 0.5),
    "stl": (2.5, 0.7, 0.25),
    "blk": (1.5, 0.7, 0.2),
    "fg2a": (12.0, 3.0, 0.7),
    "pf": (8.0, 1.5, 0.3),
}
PCT_PRIOR = {"fg3": (0.35, 0.04, 0.015), "ft": (0.75, 0.06, 0.02)}
POSS_RANGE = (1500.0, 3000.0)
# per possession counts of a hand-built row
DEFAULT_RATES = {c: m / 100.0 for c, (m, _, _) in COUNT_PRIOR.items()}


def synth_league(
    rng: np.random.Generator,
    *,
    players: int,
    seasons: range,
    competition: str = "euroleague",
    prefix: str = "p",
) -> pd.DataFrame:
    """I1 rows for ``players`` over ``seasons``: true rates drawn from the prior (per stat),
    a random walk of true rates (the drift), counts Poisson at the season's possessions and made
    shots binomial in the attempts: exactly the model the projection assumes."""
    n_seasons = len(seasons)
    poss = rng.uniform(*POSS_RANGE, size=(players, n_seasons))

    def walk(mean: float, sd: float, step: float) -> np.ndarray:
        start = mean + sd * rng.standard_normal((players, 1))
        return start + np.cumsum(
            np.hstack(
                [np.zeros((players, 1)), step * rng.standard_normal((players, n_seasons - 1))]
            ),
            axis=1,
        )

    counts = {}
    for stat, (mean, sd, step) in COUNT_PRIOR.items():
        rate = np.maximum(walk(mean, sd, step), 0.05)
        counts[stat] = rng.poisson(rate * poss / 100.0)
    for stat, made in (("fg3", "fg3m"), ("ft", "ftm")):
        mean, sd, step = PCT_PRIOR[stat]
        p = np.clip(walk(mean, sd, step), 0.05, 0.95)
        attempts = counts["fg3a" if stat == "fg3" else "fta"]
        counts[made] = rng.binomial(attempts, p)
    counts["fg2m"] = rng.binomial(counts["fg2a"], 0.5)
    ids = np.array([f"{prefix}{i:05d}" for i in range(players)])
    frame = pd.DataFrame(
        {
            "person_id": np.repeat(ids, n_seasons),
            "competition": competition,
            "season": np.tile(np.array(list(seasons), dtype="int64"), players),
            "partial": False,
            "mapped": True,
            "team": "AAA",
            "games": 30,
            "minutes": poss.ravel() / 2.0,
            "poss": poss.ravel(),
            **{c: counts[c].ravel().astype("int64") for c in COUNT_COLUMNS},
            "debut_season": int(seasons[0]),
        }
    )
    return frame


def hand_rows(
    person: str, competition: str, season: int, poss: float, *, partial: bool = False, **counts: int
) -> pd.DataFrame:
    """One I1 row with default per-possession counts, overridden by ``counts``."""
    row = {c: round(DEFAULT_RATES.get(c, 0.0) * poss) for c in COUNT_COLUMNS}
    row["fg3m"] = round(0.35 * row["fg3a"])
    row["ftm"] = round(0.75 * row["fta"])
    row["fg2m"] = round(0.5 * row["fg2a"])
    row.update(counts)
    frame = pd.DataFrame(
        [
            {
                "person_id": person,
                "competition": competition,
                "season": season,
                "partial": partial,
                "mapped": True,
                "team": "AAA",
                "games": 10,
                "minutes": poss / 2.0,
                "poss": float(poss),
                **row,
                "debut_season": season,
            }
        ]
    )
    return validated(frame, PLAYER_SEASONS_SCHEMA)


def concat(*frames: pd.DataFrame) -> pd.DataFrame:
    out = pd.concat(frames, ignore_index=True)
    out["debut_season"] = out.groupby("person_id")["season"].transform("min")
    return validated(out, PLAYER_SEASONS_SCHEMA)


def targets_for(
    people: list[str], competition: str, season: int, *, checkpoint: float = 0.0, exposure=2000.0
) -> pd.DataFrame:
    return validated(
        pd.DataFrame(
            {
                "person_id": people,
                "competition": competition,
                "season": season,
                "checkpoint": checkpoint,
                "exposure": exposure,
            }
        ),
        TARGET_SCHEMA,
    )


def row_of(frame: pd.DataFrame, person: str, stat: str) -> pd.Series:
    return frame[(frame["person_id"] == person) & (frame["stat"] == stat)].iloc[0]


def zero_drift() -> dict[str, float]:
    return dict.fromkeys(PROJECTED_STATS, 0.0)


def recovered_blend(out: pd.DataFrame, person: str, stat: str) -> float:
    """The player's decayed rate m, recovered from ``mean = w m + (1 - w) prior_mean``."""
    row = out[(out["person_id"] == person) & (out["stat"] == stat)].iloc[0]
    assert row["weight"] > 0.5
    return (row["mean"] - (1.0 - row["weight"]) * row["prior_mean"]) / row["weight"]


def background(competition: str = "euroleague", seasons: range = range(2020, 2024)) -> pd.DataFrame:
    return synth_league(
        np.random.default_rng(7), players=60, seasons=seasons, competition=competition, prefix="bg"
    )


# --- calibration -------------------------------------------------------------------------------


def test_intervals_cover_the_observed_target_rate_on_synthetic_leagues():
    """200 leagues of 60 players: the 80% interval holds the target season's observed rate in
    77-83% of cases, per stat. Players pooled: n = 12000 per stat, binomial SE of the coverage
    sqrt(0.8 * 0.2 / n) = 0.0037, so the +-3 point band is 8 SE wide: a correct method fails it
    with probability ~1e-15 per stat; a mis-scaled variance (e.g. missing the target sampling
    term or the drift) does not pass."""
    rng = np.random.default_rng(20261201)
    params = variant_params("proj_shrunk", half_life=2.0)
    stats = ("pts", "ast", "dreb", "fta", "fg3a", "fg3", "ft")
    hits = dict.fromkeys(stats, 0)
    total = dict.fromkeys(stats, 0)
    for _ in range(200):
        full = synth_league(rng, players=60, seasons=range(2020, 2025))
        history = full[full["season"] < 2024]
        last = full[full["season"] == 2024]
        observed = rate_table(last).set_index("person_id")
        target = targets_for(
            last["person_id"].tolist(), "euroleague", 2024, exposure=last["poss"].to_numpy()
        )
        out = project(history, target, params, drift=fit_drift(history, 2024))
        for stat in stats:
            rows = out[out["stat"] == stat].set_index("person_id")
            seen = observed.loc[rows.index, stat]
            ok = seen.notna()
            inside = (seen[ok] >= rows.loc[ok, "lo80"]) & (seen[ok] <= rows.loc[ok, "hi80"])
            hits[stat] += int(inside.sum())
            total[stat] += int(ok.sum())
    for stat in stats:
        n = total[stat]
        se = math.sqrt(0.8 * 0.2 / n)
        assert 0.03 / se > 7.0  # the band is more than 7 SE wide, as documented above
        assert 0.77 <= hits[stat] / n <= 0.83, (stat, hits[stat] / n)


def test_fit_drift_recovers_the_true_step_variance():
    """3000 players, 5 seasons: the changes between consecutive seasons have variance D plus
    two sampling variances; D is recovered within 5 SE of a variance estimate from n = 3000
    independent players (SE = sqrt(2 / n) * (D + sampling), consecutive pairs overlap, so n is
    conservative)."""
    rng = np.random.default_rng(3)
    history = synth_league(rng, players=3000, seasons=range(2018, 2023))
    drift = fit_drift(history, 2023)
    for stat in ("pts", "ast", "dreb"):
        mean, _, step = COUNT_PRIOR[stat]
        sampling = 2 * 100.0 * mean * math.log(2) / POSS_RANGE[0]  # 2 * u * E[1/poss]
        se = math.sqrt(2 / 3000) * (step**2 + sampling)
        assert abs(drift[stat] - step**2) < 5 * se, (stat, drift[stat], step**2)


def test_fit_drift_of_impact_ratings_recovers_the_true_step_variance():
    """2000 players, 4 seasons of a random walk with step variance 1 observed with sd 0.5 (two
    sampling variances of 0.25 in a change): D within 5 SE, SE = sqrt(2 / n) * (D + 0.5) with
    n = 2000 independent players."""
    rng = np.random.default_rng(8)
    history = synth_league(rng, players=2000, seasons=range(2018, 2022))
    truth = np.cumsum(rng.standard_normal((2000, 4)), axis=1)
    impact = pd.DataFrame(
        {
            "person_id": np.repeat(history["person_id"].unique(), 4),
            "competition": "euroleague",
            "season": np.tile(np.arange(2018, 2022), 2000),
            "stat": "spm",
            "value": truth.ravel() + 0.5 * rng.standard_normal(8000),
            "sd": 0.5,
        }
    )
    assert "spm" not in fit_drift(history, 2022)
    drift = fit_drift(history, 2022, impact)
    se = math.sqrt(2 / 2000) * (1.0 + 0.5)
    assert abs(drift["spm"] - 1.0) < 5 * se


def test_fit_drift_uses_only_complete_seasons_before_the_cutoff():
    rng = np.random.default_rng(4)
    history = synth_league(rng, players=300, seasons=range(2018, 2024))
    base = fit_drift(history, 2022)
    later = history.copy()
    later.loc[later["season"] >= 2022, "pts"] += 500  # rows at or after the cutoff
    assert fit_drift(later, 2022) == base
    # a partial season is not a complete one: with 2021 partial (and altered) the pairs through
    # 2021 are gone, exactly as if the season were absent
    cut = later[later["season"] == 2021].copy()
    cut["partial"] = True
    cut["pts"] += 900
    changed = pd.concat([later[later["season"] != 2021], cut])
    absent = fit_drift(history[history["season"] != 2021], 2022)
    assert fit_drift(changed, 2022) == absent
    assert absent != base
    earlier = history.copy()
    earlier.loc[earlier["season"] == 2019, "pts"] += 300
    assert fit_drift(earlier, 2022)["pts"] != base["pts"]


# --- shrinkage ---------------------------------------------------------------------------------


@pytest.mark.parametrize("stat", COUNT_STATS + PCT_STATS)
def test_weight_goes_to_one_with_exposure_and_to_zero_without(stat):
    """w = tau2 / (tau2 + S) with S = u / exposure: more exposure never lowers it, it is below
    1e-3 at 1e-3 possessions (counts) and above 1 - 1e-3 at 1e9."""
    league = background()
    exposures = [1e-3, 1.0, 1e2, 1e4, 1e6, 1e9]
    rows = []
    for i, e in enumerate(exposures):
        att = {"fg3a": max(1, round(0.07 * e)), "fta": max(1, round(0.06 * e))}
        att.update({"fg2a": max(1, round(0.12 * e))})
        made = {"fg3m": att["fg3a"] // 3, "ftm": att["fta"] * 3 // 4}
        counts = {c: max(1, round(DEFAULT_RATES.get(c, 0.1) * e)) for c in COUNT_COLUMNS}
        counts.update(att)
        counts.update(made)
        rows.append(hand_rows(f"probe{i}", "euroleague", 2023, e, **counts))
    history = concat(league, *rows)
    target = targets_for([f"probe{i}" for i in range(len(exposures))], "euroleague", 2024)
    out = project(history, target, variant_params("proj_shrunk", 2.0), drift=zero_drift())
    w = out[out["stat"] == stat].set_index("person_id")["weight"].to_numpy()
    assert (np.diff(w) >= 0).all()
    # a percentage has at least one attempt even at ~0 possessions: w <= tau2 / (tau2 + u) with
    # tau2 / u ~ 0.02 for these spreads, so it is bounded by 0.05 there
    assert w[0] < (1e-3 if stat in COUNT_STATS else 0.05)
    assert w[-1] > 1 - 1e-3


def test_a_player_without_history_gets_the_league_mean_and_prior_sd():
    """No usable row, drift 0 and a huge target exposure isolate the prior: the interval is the
    possession-weighted league mean of seasons t-3..t-1 +- z90 * prior sd. The prior sd is the
    true between-player sd within 5 SE of a variance estimate (SE_sd = sd * sqrt(1 / 2n),
    n = 3000 players)."""
    rng = np.random.default_rng(11)
    league = synth_league(rng, players=3000, seasons=range(2021, 2024))
    out = project(
        league,
        targets_for(["nobody"], "euroleague", 2024, exposure=1e12),
        variant_params("proj_shrunk", 2.0),
        drift=zero_drift(),
    )
    rates = rate_table(league)
    for stat in ("pts", "ast", "ft"):
        row = out[out["stat"] == stat].iloc[0]
        seen = rates[rates[stat].notna()]
        mean = (seen[stat] * seen[f"{stat}_n"]).sum() / seen[f"{stat}_n"].sum()
        assert row["mean"] == pytest.approx(mean, rel=1e-12)
        assert row["weight"] == 0.0
        assert row["flags"] == "no_history"
        assert row["n_seasons"] == 0
        assert row["hi80"] - row["mean"] == pytest.approx(Z80 * row["sd"], rel=1e-9)
        _, tau, step = (COUNT_PRIOR | PCT_PRIOR)[stat]
        true_var = tau**2 + step**2  # the true spread in season s is tau^2 + s * step^2, s = 0..2
        noise = (
            (
                100.0 * COUNT_PRIOR[stat][0]
                if stat in COUNT_STATS
                else PCT_PRIOR[stat][0] * (1 - PCT_PRIOR[stat][0]) * 100.0 / COUNT_PRIOR["fta"][0]
            )
            * math.log(2)
            / POSS_RANGE[0]
        )  # mean sampling variance of a row: u * E[1 / exposure]
        # the variance estimate has SE sqrt(2 / n) * (true + noise); sd has half the relative SE
        tolerance = 5 * 0.5 * math.sqrt(2 / 3000) * (1 + noise / true_var)
        assert abs(row["sd"] / math.sqrt(true_var) - 1) < tolerance, stat


# --- leakage -----------------------------------------------------------------------------------


def project_with(history, target, **kwargs):
    return project(
        history, target, variant_params("proj_shrunk", 2.0), drift=zero_drift(), **kwargs
    )


def future_rows() -> pd.DataFrame:
    """Rows a 2024 target may not use: 2025-26, the whole 2024 season, a partial 2024 season of
    the other competition, for known players and for new ones."""
    return pd.concat(
        [
            hand_rows("bg00003", "euroleague", 2025, 2500, pts=900),
            hand_rows("bg00003", "euroleague", 2026, 2500, pts=1000),
            hand_rows("bg00003", "euroleague", 2024, 2500, pts=800),
            hand_rows("bg00004", "gbl", 2024, 900, partial=True, pts=400),
            hand_rows("newcomer", "euroleague", 2024, 2500, pts=700),
            hand_rows("newcomer", "euroleague", 2025, 2500, pts=700),
        ],
        ignore_index=True,
    )


def future_impact() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "person_id": ["bg00003", "bg00003", "bg00005"],
            "competition": "euroleague",
            "season": [2024, 2025, 2025],
            "stat": "spm",
            "value": [9.0, 9.0, 9.0],
            "sd": [1.0, 1.0, 1.0],
        }
    )


def impact_league(history: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    rows = history[["person_id", "competition", "season"]].copy()
    parts = []
    for stat in IMPACT_STATS:
        part = rows.assign(
            stat=stat,
            value=rng.normal(0.0, 2.0, len(rows)),
            sd=rng.uniform(0.5, 1.5, len(rows)),
        )
        parts.append(part)
    return validated(pd.concat(parts, ignore_index=True), IMPACT_SCHEMA)


def test_rows_at_or_after_the_cutoff_change_nothing():
    base_history = background()
    base_impact = impact_league(base_history)
    people = ["bg00003", "bg00004", "bg00005", "newcomer"]
    future = future_rows()
    extra_impact = validated(future_impact(), IMPACT_SCHEMA)
    target = targets_for(people, "euroleague", 2024, checkpoint=0.0)
    base = project_with(base_history, target, impact=base_impact)
    leaked = project_with(
        concat(base_history, future),
        target,
        impact=pd.concat([base_impact, extra_impact], ignore_index=True),
    )
    pd.testing.assert_frame_equal(leaked, base, check_exact=True)
    # a partial season-t row of the target's competition is ignored at checkpoint 0 ...
    partial = hand_rows("bg00003", "euroleague", 2024, 600, partial=True, pts=300)
    pd.testing.assert_frame_equal(
        project_with(concat(base_history, partial), target), project_with(base_history, target)
    )
    # ... and used (changing the output, flagged) at a checkpoint
    mid = targets_for(people, "euroleague", 2024, checkpoint=0.5)
    before = project_with(base_history, mid)
    after = project_with(concat(base_history, partial), mid)
    changed = after[(after["person_id"] == "bg00003") & (after["stat"] == "pts")].iloc[0]
    unchanged = before[(before["person_id"] == "bg00003") & (before["stat"] == "pts")].iloc[0]
    assert changed["mean"] != unchanged["mean"]
    assert "partial_season" in changed["flags"].split("|")
    assert "partial_season" not in unchanged["flags"].split("|")
    # at a checkpoint the full season-t row, later seasons and the other league's partial row
    # still change nothing
    pd.testing.assert_frame_equal(
        project_with(concat(base_history, future), mid), before, check_exact=True
    )
    # an earlier row does change the projection
    earlier = base_history.copy()
    earlier.loc[(earlier["person_id"] == "bg00003") & (earlier["season"] == 2022), "pts"] += 200
    moved = project_with(earlier, target)
    assert not moved.equals(base)
    row = lambda frame: frame[(frame["person_id"] == "bg00003") & (frame["stat"] == "pts")]  # noqa: E731
    assert row(moved)["mean"].iloc[0] != row(base)["mean"].iloc[0]


# --- hooks and hand examples -------------------------------------------------------------------


def translation_for(season: int = 2024) -> Translation:
    return Translation(
        delta=dict.fromkeys(COUNT_STATS, 0.2) | {"ast": -0.1},
        c=dict.fromkeys(COUNT_STATS, 2.0) | {"ast": 1.0},
        target_season=season,
    )


@pytest.mark.parametrize(
    ("target_comp", "other_comp", "sign"), [("euroleague", "gbl", 1.0), ("gbl", "euroleague", -1.0)]
)
def test_translation_moves_only_other_league_rows_and_only_when_asked(
    target_comp, other_comp, sign
):
    """Hand example: rows in the other league (2023, 2000 poss, 25 pts per 100) and the
    target's league (2022, 3000 poss, 18); half-life 1 gives decay 0.5 and 0.25, so the weights
    are 1000 and 750. M4's form: to EuroLeague (r + c) exp(delta) - c, to GBL exp(-delta)."""
    league = background(target_comp)
    history = concat(
        league,
        hand_rows("mover", other_comp, 2023, 2000, pts=500, ast=100),
        hand_rows("mover", target_comp, 2022, 3000, pts=540, ast=300),
    )
    target = targets_for(["mover"], target_comp, 2024)
    t = translation_for()
    full = project(
        history, target, variant_params("proj_full", 1.0), drift=zero_drift(),
        aging=lambda s, a, b: np.zeros_like(a), translation=t,
    )  # fmt: skip
    plain = project(
        history, target, variant_params("proj_shrunk", 1.0), drift=zero_drift(), translation=t
    )
    shifted = max((25.0 + 2.0) * math.exp(sign * 0.2) - 2.0, 0.0)
    assert recovered_blend(full, "mover", "pts") == pytest.approx(
        (1000 * shifted + 750 * 18.0) / 1750, rel=1e-9
    )
    assert recovered_blend(plain, "mover", "pts") == pytest.approx(
        (1000 * 25.0 + 750 * 18.0) / 1750, rel=1e-9
    )
    shifted_ast = max((5.0 + 1.0) * math.exp(sign * -0.1) - 1.0, 0.0)
    assert recovered_blend(full, "mover", "ast") == pytest.approx(
        (1000 * shifted_ast + 750 * 10.0) / 1750, rel=1e-9
    )
    flags = lambda out, stat: out[(out["stat"] == stat)].iloc[-1]["flags"]  # noqa: E731
    assert "translated" in flags(full, "pts").split("|")
    assert "translated" not in flags(plain, "pts").split("|")
    # percentages are not translated
    assert recovered_blend(full, "mover", "ft") == pytest.approx(
        recovered_blend(plain, "mover", "ft"), rel=1e-12
    )
    assert "translated" not in flags(full, "ft").split("|")


def test_translation_is_required_for_other_league_rows_and_may_not_be_from_the_future():
    history = concat(background(), hand_rows("mover", "gbl", 2023, 2000))
    target = targets_for(["mover"], "euroleague", 2024)
    params = variant_params("proj_full", 2.0)
    zero_age = lambda s, a, b: np.zeros_like(a)  # noqa: E731
    with pytest.raises(ValueError, match="M4"):
        project(history, target, params, drift=zero_drift(), aging=zero_age)
    with pytest.raises(ValueError, match="walk-forward"):
        project(
            history, target, params, drift=zero_drift(), aging=zero_age,
            translation=translation_for(2025),
        )  # fmt: skip
    # no other-league row, no factors needed
    league_only = targets_for(["bg00001"], "euroleague", 2024)
    project(history, league_only, params, drift=zero_drift(), aging=zero_age)
    with pytest.raises(ValueError, match="aging"):
        project(history, league_only, params, drift=zero_drift())


def test_aging_moves_every_row_to_the_target_age_through_the_callable():
    """Hand example: age 23.0 at 1 Oct 2023 (synthetic), so 22.0 in 2022 and 24.0 in 2024. The
    stub adds 0.5 pts per 100 per year; rows 2023 (decay 0.5) and 2022 (decay 0.25) of 2000 and
    3000 possessions weigh 1000 and 750."""
    history = concat(
        background(),
        hand_rows("vet", "euroleague", 2023, 2000, pts=500),
        hand_rows("vet", "euroleague", 2022, 3000, pts=540),
        hand_rows("ageless", "euroleague", 2023, 2000, pts=500),
    )
    ages = pd.DataFrame({"person_id": ["vet"], "season": [2023], "age": [23.0]})
    calls = []

    def stub(stat, age_from, age_to):
        calls.append(stat)
        return 0.5 * (age_to - age_from) if stat == "pts" else np.zeros_like(age_from)

    target = targets_for(["vet", "ageless"], "euroleague", 2024)
    out = project(
        history, target, variant_params("proj_age", 1.0), drift=zero_drift(), ages=ages, aging=stub
    )
    assert recovered_blend(out, "vet", "pts") == pytest.approx(
        (1000 * (25.0 + 0.5) + 750 * (18.0 + 1.0)) / 1750, rel=1e-9
    )
    assert recovered_blend(out, "ageless", "pts") == pytest.approx(25.0, rel=1e-9)
    flag = lambda who: out[(out["person_id"] == who) & (out["stat"] == "pts")]["flags"].iloc[0]  # noqa: E731
    assert flag("ageless") == "no_age"
    assert flag("vet") == ""
    # the shrunk variant never calls the curve
    calls.clear()
    plain = project(
        history, target, variant_params("proj_shrunk", 1.0), drift=zero_drift(), ages=ages,
        aging=stub,
    )  # fmt: skip
    assert not calls
    assert recovered_blend(plain, "vet", "pts") == pytest.approx(
        (1000 * 25.0 + 750 * 18.0) / 1750, rel=1e-9
    )
    assert "no_age" not in plain["flags"].str.cat(sep="|")


def test_a_partial_current_season_has_decay_one_and_a_flag():
    """Hand example at checkpoint 0.5, half-life 1: partial 2024 (600 poss, 20 pts/100, decay 1)
    and 2023 (2000 poss, 25, decay 0.5): weights 600 and 1000."""
    history = concat(
        background(),
        hand_rows("p", "euroleague", 2023, 2000, pts=500),
        hand_rows("p", "euroleague", 2024, 600, partial=True, pts=120),
    )
    target = targets_for(["p"], "euroleague", 2024, checkpoint=0.5)
    out = project(history, target, variant_params("proj_shrunk", 1.0), drift=zero_drift())
    assert recovered_blend(out, "p", "pts") == pytest.approx(
        (600 * 20.0 + 1000 * 25.0) / 1600, rel=1e-9
    )
    assert out[out["stat"] == "pts"]["flags"].iloc[0] == "partial_season"
    assert out[out["stat"] == "pts"]["n_seasons"].iloc[0] == 2


def test_the_drift_widens_the_interval_by_the_seasons_since_the_newest_row():
    """One row, g seasons old: variance grows by w^2 * D * g (w from the sampling alone)."""
    history = concat(background(), hand_rows("old", "euroleague", 2021, 2000, pts=500))
    params = variant_params("proj_shrunk", 2.0)
    target = targets_for(["old"], "euroleague", 2024)
    flat = project(history, target, params, drift=zero_drift())
    drifting = project(history, target, params, drift=dict.fromkeys(PROJECTED_STATS, 0.25))
    a = flat[flat["stat"] == "pts"].iloc[0]
    b = drifting[drifting["stat"] == "pts"].iloc[0]
    assert b["sd"] ** 2 - a["sd"] ** 2 == pytest.approx(a["weight"] ** 2 * 0.25 * 3, rel=1e-9)
    assert b["mean"] == a["mean"]


def test_impact_rows_blend_by_inverse_variance_and_a_missing_one_gives_the_prior():
    """Hand example, half-life 2: spm 3.0 (sd 1.0) a season ago and 1.0 (sd 0.5) two seasons
    ago: weights decay / sd^2 = 2^-0.5 / 1 and 0.5 / 0.25."""
    league = background()
    history = concat(
        league,
        hand_rows("star", "euroleague", 2023, 2000),
        hand_rows("star", "euroleague", 2022, 2000),
        hand_rows("blank", "euroleague", 2023, 2000),
    )
    impact = impact_league(history)
    impact = impact[~impact["person_id"].isin(["star", "blank"])]
    star = pd.DataFrame(
        {
            "person_id": "star",
            "competition": "euroleague",
            "season": [2023, 2022],
            "stat": "spm",
            "value": [3.0, 1.0],
            "sd": [1.0, 0.5],
        }
    )
    impact = pd.concat([impact, star], ignore_index=True)
    target = targets_for(["star", "blank"], "euroleague", 2024)
    out = project(
        history, target, variant_params("proj_shrunk", 2.0), drift=zero_drift(), impact=impact
    )
    g1, g2 = 2**-0.5 / 1.0, 0.5 / 0.25
    got = out[(out["person_id"] == "star") & (out["stat"] == "spm")].iloc[0]
    m = (got["mean"] - (1 - got["weight"]) * got["prior_mean"]) / got["weight"]
    assert m == pytest.approx((g1 * 3.0 + g2 * 1.0) / (g1 + g2), rel=1e-9)
    assert got["n_seasons"] == 2
    assert "no_impact_input" not in got["flags"]
    blank = out[(out["person_id"] == "blank") & (out["stat"] == "spm")].iloc[0]
    assert blank["mean"] == blank["prior_mean"]
    assert blank["weight"] == 0.0
    assert "no_impact_input" in blank["flags"].split("|")
    # impact has no sampling term: the predictive sd of a blank is the prior sd
    assert blank["sd"] > 0
    # a GBL target has no impact prior: the impact stats are simply not projected
    gbl_history = concat(background("gbl"), hand_rows("g", "gbl", 2023, 2000))
    gbl = project(
        gbl_history,
        targets_for(["g"], "gbl", 2024),
        variant_params("proj_shrunk", 2.0),
        drift=zero_drift(),
        impact=impact,
    )
    assert set(gbl["stat"]) == set(COUNT_STATS + PCT_STATS)


def test_the_spm_prior_replaces_the_league_mean_of_brapm_by_the_projected_spm():
    league = background()
    history = concat(league, hand_rows("star", "euroleague", 2023, 2000))
    impact = impact_league(history)
    target = targets_for(["star", "bg00001"], "euroleague", 2024)
    league_prior = project(
        history, target, variant_params("proj_full", 2.0), drift=zero_drift(), impact=impact,
        aging=lambda s, a, b: np.zeros_like(a), translation=translation_for(),
    )  # fmt: skip
    spm_prior = project(
        history, target, variant_params("proj_full_spm", 2.0), drift=zero_drift(), impact=impact,
        aging=lambda s, a, b: np.zeros_like(a), translation=translation_for(),
    )  # fmt: skip
    for who in ("star", "bg00001"):
        spm = row_of(spm_prior, who, "spm")
        assert row_of(spm_prior, who, "brapm")["prior_mean"] == spm["mean"]
        assert row_of(league_prior, who, "brapm")["prior_mean"] != spm["mean"]
        assert row_of(league_prior, who, "spm")["mean"] == spm["mean"]


# --- variants and schemas ----------------------------------------------------------------------


def test_variant_params_for_all_four_variants():
    assert variant_params("proj_shrunk", 1.0) == ProjectionParams(1.0, False, False, "league")
    assert variant_params("proj_age", 2.0) == ProjectionParams(2.0, True, False, "league")
    assert variant_params("proj_full", 3.0) == ProjectionParams(3.0, True, True, "league")
    assert variant_params("proj_full_spm", 2.0) == ProjectionParams(2.0, True, True, "spm")
    assert variant_params("proj_full", 2.0).interval == 0.8
    assert set(VARIANTS) == {"proj_shrunk", "proj_age", "proj_full", "proj_full_spm"}
    with pytest.raises(ValueError, match="unknown"):
        variant_params("proj_magic", 1.0)


def test_output_follows_the_projections_schema():
    history = concat(
        background(), hand_rows("a", "euroleague", 2023, 2000), hand_rows("b", "gbl", 2023, 800)
    )
    impact = impact_league(history)
    target = pd.concat(
        [
            targets_for(["a", "b", "ghost"], "euroleague", 2024),
            targets_for(["a"], "euroleague", 2024, checkpoint=0.25),
        ],
        ignore_index=True,
    )
    out = project(
        history, target, variant_params("proj_shrunk", 2.0), drift=zero_drift(), impact=impact
    )
    PROJECTIONS_SCHEMA.validate(out)
    assert len(out) == 4 * len(PROJECTED_STATS)
    first = out[(out["person_id"] == "a") & (out["checkpoint"] == 0.0)]
    assert first["stat"].tolist() == list(PROJECTED_STATS)
    assert (out["lo80"] <= out["mean"]).all() or (out["lo80"] == 0).any()
    assert (out["hi80"] >= out["mean"]).all()
    assert (out[out["stat"].isin(COUNT_STATS + PCT_STATS)]["lo80"] >= 0).all()
    assert out["flags"].map(lambda f: set(f.split("|")) - {""} <= set(FLAGS)).all()
    ghost = out[out["person_id"] == "ghost"]
    assert (ghost["flags"].str.contains("no_history")).all()
    assert (ghost["exposure"] == 0).all()
    wide = out[(out["person_id"] == "ghost") & (out["stat"] == "pts")].iloc[0]
    narrow = out[(out["person_id"] == "a") & (out["stat"] == "pts") & (out["checkpoint"] == 0)]
    assert wide["sd"] > narrow["sd"].iloc[0]
    with pytest.raises(pa.errors.SchemaError):
        validated(target.assign(checkpoint=1.0), TARGET_SCHEMA)
    with pytest.raises(pa.errors.SchemaError):
        validated(target.assign(exposure=0.0), TARGET_SCHEMA)
    with pytest.raises(pa.errors.SchemaError):
        validated(impact.assign(stat="orapm"), IMPACT_SCHEMA)


# --- impact measurement noise (D14) ------------------------------------------------------------

IMPACT_SPM_B = 1500.0  # spm truth noise variance b / exposure
IMPACT_BRAPM_A = 0.6  # brapm truth noise variance a
IMPACT_NOISE = {"spm": (0.0, IMPACT_SPM_B), "brapm": (IMPACT_BRAPM_A, 0.0)}


def impact_world(rng: np.random.Generator, players: int = 40) -> dict:
    """One league whose true impact (spread 2, drift 0.5 per season) is seen in the history
    through noisy measurements (SPM sd sqrt(1500 / poss), BRAPM sd 0.7) and in the target season
    as a measurement with variance ``a + b / exposure`` (known a, b): what the harness scores."""
    full = synth_league(rng, players=players, seasons=range(2020, 2025))
    history = full[full["season"] < 2024]
    last = full[full["season"] == 2024].reset_index(drop=True)
    ids = full["person_id"].unique()
    poss = full["poss"].to_numpy().reshape(players, 5)
    parts, observed = [], {}
    for stat in IMPACT_STATS:
        truth = 2.0 * rng.standard_normal((players, 1)) + np.cumsum(
            np.hstack([np.zeros((players, 1)), 0.5 * rng.standard_normal((players, 4))]), axis=1
        )
        sd = np.sqrt(IMPACT_SPM_B / poss) if stat == "spm" else np.full((players, 5), 0.7)
        seen = truth + sd * rng.standard_normal((players, 5))
        parts.append(
            pd.DataFrame(
                {
                    "person_id": np.repeat(ids, 4),
                    "competition": "euroleague",
                    "season": np.tile(np.arange(2020, 2024), players),
                    "stat": stat,
                    "value": seen[:, :4].ravel(),
                    "sd": sd[:, :4].ravel(),
                }
            )
        )
        a, b = IMPACT_NOISE[stat]
        noise = np.sqrt(a + b / poss[:, 4])
        observed[stat] = truth[:, 4] + noise * rng.standard_normal(players)
    return {
        "history": history,
        "target": targets_for(ids.tolist(), "euroleague", 2024, exposure=last["poss"].to_numpy()),
        "impact": pd.concat(parts, ignore_index=True),
        "observed": observed,
        "noise": {s: IMPACT_NOISE[s][0] + IMPACT_NOISE[s][1] / poss[:, 4] for s in IMPACT_STATS},
    }


@pytest.fixture(scope="module")
def impact_coverage() -> dict:
    """Coverage of the observed impact truth over 200 leagues with ``impact_noise`` and, on the
    first 40 leagues, without it (with the coverage theory expects there)."""
    rng = np.random.default_rng(20261208)
    params = variant_params("proj_shrunk", half_life=2.0)
    hits = {"with": dict.fromkeys(IMPACT_STATS, 0), "without": dict.fromkeys(IMPACT_STATS, 0)}
    expected = dict.fromkeys(IMPACT_STATS, 0.0)
    variance = dict.fromkeys(IMPACT_STATS, 0.0)
    for league in range(200):
        world = impact_world(rng)
        drift = fit_drift(world["history"], 2024, world["impact"])
        runs = {"with": IMPACT_NOISE} | ({"without": None} if league < 40 else {})
        for name, noise in runs.items():
            proj = project(
                world["history"], world["target"], params, drift=drift,
                impact=world["impact"], impact_noise=noise,
            )  # fmt: skip
            for stat in IMPACT_STATS:
                rows = proj[proj["stat"] == stat]
                seen = world["observed"][stat]
                hits[name][stat] += int(((seen >= rows["lo80"]) & (seen <= rows["hi80"])).sum())
                if name == "without":
                    # the interval covers the truth with variance V; the observation adds noise n
                    v = rows["sd"].to_numpy() ** 2
                    p = 2 * special.ndtr(Z80 * np.sqrt(v / (v + world["noise"][stat]))) - 1
                    expected[stat] += float(p.sum())
                    variance[stat] += float((p * (1 - p)).sum())
    return {"hits": hits, "expected": expected, "variance": variance}


def test_impact_intervals_with_the_noise_term_cover_the_observed_truth(impact_coverage):
    """200 leagues of 40 players: n = 8000 per stat, binomial SE sqrt(0.8 * 0.2 / n) = 0.0045, so
    the +-3 point band is 6.7 SE wide."""
    n = 200 * 40
    assert 0.03 / math.sqrt(0.8 * 0.2 / n) > 6.5
    for stat in IMPACT_STATS:
        assert 0.77 <= impact_coverage["hits"]["with"][stat] / n <= 0.83, stat


def test_impact_intervals_without_the_noise_term_undercover_as_theory_says(impact_coverage):
    """Without the term the interval is that of the true impact, z90 * sqrt(V); the observation
    adds noise n, so a player is covered with probability 2 Phi(z90 sqrt(V / (V + n))) - 1 (V from
    the interval itself, n known). The sum of those probabilities is the expected hit count; the
    observed count lies within 5 SD of it (Poisson-binomial variance), and the expected coverage
    is under the band."""
    n = 40 * 40
    for stat in IMPACT_STATS:
        expected = impact_coverage["expected"][stat]
        sd = math.sqrt(impact_coverage["variance"][stat])
        assert abs(impact_coverage["hits"]["without"][stat] - expected) < 5 * sd, stat
        assert expected / n < 0.77, stat


def test_no_impact_noise_is_the_old_output_bit_for_bit_and_the_term_is_exact():
    world = impact_world(np.random.default_rng(2), players=30)
    params = variant_params("proj_shrunk", 2.0)
    args = (world["history"], world["target"], params)
    drift = fit_drift(world["history"], 2024, world["impact"])
    base = project(*args, drift=drift, impact=world["impact"])
    for none in (None, {}):
        same = project(*args, drift=drift, impact=world["impact"], impact_noise=none)
        pd.testing.assert_frame_equal(same, base, check_exact=True)
    noisy = project(*args, drift=drift, impact=world["impact"], impact_noise=IMPACT_NOISE)
    for stat in IMPACT_STATS:
        old, new = base[base["stat"] == stat], noisy[noisy["stat"] == stat]
        np.testing.assert_allclose(
            new["sd"] ** 2, old["sd"] ** 2 + world["noise"][stat], rtol=1e-12
        )
        assert (new["mean"].to_numpy() == old["mean"].to_numpy()).all()
    other = base["stat"].isin(COUNT_STATS + PCT_STATS)
    pd.testing.assert_frame_equal(
        noisy[other].reset_index(drop=True), base[other].reset_index(drop=True), check_exact=True
    )
    # a stat without an entry keeps the old variance
    only = project(*args, drift=drift, impact=world["impact"], impact_noise={"brapm": (0.6, 0.0)})
    pd.testing.assert_frame_equal(
        only[only["stat"] == "spm"].reset_index(drop=True),
        base[base["stat"] == "spm"].reset_index(drop=True),
        check_exact=True,
    )
    for bad in ({"spm": (-0.1, 0.0)}, {"spm": (0.0, -1.0)}, {"orapm": (1.0, 1.0)}):
        with pytest.raises(ValueError, match="impact_noise"):
            project(*args, drift=drift, impact=world["impact"], impact_noise=bad)
