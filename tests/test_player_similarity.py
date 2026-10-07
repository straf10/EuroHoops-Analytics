"""M6 "plays like" similarity (weeks 16-18 L4): synthetic leagues with a persistent style per
player plus season-to-season noise. Nothing here is a real player. (``test_similarity.py`` is
the cross-league name matcher's test.)"""

import numpy as np
import pandas as pd
import pytest

from eurohoops.models.player_seasons import PLAYER_SEASONS_SCHEMA
from eurohoops.models.similarity import (
    BOX_FEATURES,
    SHOT_FEATURES,
    SIMILAR_SCHEMA,
    embed,
    neighbours,
    raw_features,
    self_retrieval_rate,
)
from eurohoops.parse.schemas import validated
from eurohoops.stats.twins import ZONES

POOL = (2019, 2020, 2021, 2022)
RATED = ("fg2a", "fg3a", "fta", "oreb", "dreb", "ast", "stl", "blk", "tov", "pf")
# representative (band, x, y) of each twins zone: angle = atan2(x, y) from the basket's axis
SPOT = {
    "rim": ("rim", 0.0, 0.8),
    "short_l": ("short", -2.2, 0.8),
    "short_c": ("short", 0.0, 2.2),
    "short_r": ("short", 2.2, 0.8),
    "mid_l": ("mid", -4.0, 1.0),
    "mid_c": ("mid", 0.0, 4.0),
    "mid_r": ("mid", 4.0, 1.0),
    "long2_l": ("long2", -5.5, 1.0),
    "long2_c": ("long2", 0.0, 5.5),
    "long2_r": ("long2", 5.5, 1.0),
    "corner3_l": ("three", -6.6, 0.5),
    "three_c": ("three", 0.0, 6.9),
    "corner3_r": ("three", 6.6, 0.5),
    "deep3": ("deep3", 0.0, 9.0),
}


def _style(rng: np.random.Generator) -> dict[str, float]:
    """A player's persistent per-100 rates and shooting percentages."""
    spread = {
        "fg2a": (25.0, 0.35),
        "fg3a": (15.0, 0.5),
        "fta": (10.0, 0.4),
        "oreb": (3.0, 0.5),
        "dreb": (8.0, 0.4),
        "ast": (8.0, 0.6),
        "stl": (2.0, 0.4),
        "blk": (1.0, 0.7),
        "tov": (6.0, 0.3),
        "pf": (7.0, 0.25),
    }
    style = {k: m * float(np.exp(s * rng.standard_normal())) for k, (m, s) in spread.items()}
    style["fg2"] = float(np.clip(0.5 + 0.05 * rng.standard_normal(), 0.3, 0.7))
    style["fg3"] = float(np.clip(0.35 + 0.04 * rng.standard_normal(), 0.2, 0.5))
    style["ft"] = float(np.clip(0.75 + 0.07 * rng.standard_normal(), 0.5, 0.95))
    return style


def _season_row(
    rng: np.random.Generator, person: str, season: int, style: dict[str, float]
) -> dict[str, object]:
    poss = float(rng.uniform(800.0, 2500.0))
    wobble = float(np.exp(0.1 * rng.standard_normal()))  # a season's form
    n = {k: int(rng.poisson(style[k] * wobble * poss / 100.0)) for k in RATED}
    fg2m = int(rng.binomial(n["fg2a"], style["fg2"]))
    fg3m = int(rng.binomial(n["fg3a"], style["fg3"]))
    ftm = int(rng.binomial(n["fta"], style["ft"]))
    return {
        "person_id": person,
        "competition": "euroleague",
        "season": season,
        "partial": False,
        "mapped": True,
        "team": "AAA",
        "games": 30,
        "minutes": poss / 2.0,
        "poss": poss,
        "debut_season": season,
        **n,
        "fg2m": fg2m,
        "fg3m": fg3m,
        "ftm": ftm,
        "pts": 2 * fg2m + 3 * fg3m + ftm,
    }


def _league(
    seed: int,
    n_players: int,
    seasons: tuple[int, ...] = POOL,
    competition: str = "euroleague",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_players):
        style = _style(rng)
        for s in seasons:
            row = _season_row(rng, f"P:{competition[0]}{i}", s, style)
            rows.append({**row, "competition": competition})
    frame = pd.DataFrame(rows)
    frame["debut_season"] = frame.groupby("person_id")["season"].transform("min")
    return validated(frame[list(PLAYER_SEASONS_SCHEMA.columns)], PLAYER_SEASONS_SCHEMA)


def _shots(
    seed: int, seasons: pd.DataFrame, favourite: dict[str, str] | None = None
) -> pd.DataFrame:
    """Shots of every row: a persistent zone preference (a Dirichlet draw, or most of the mass
    on the person's ``favourite`` zone) and a persistent shot-making skill in points per shot."""
    rng = np.random.default_rng(seed)
    persons = sorted(set(seasons["person_id"]))
    prefs = dict(zip(persons, rng.dirichlet(np.ones(len(ZONES)), size=len(persons)), strict=True))
    skill = dict(zip(persons, rng.normal(0.0, 0.08, len(persons)), strict=True))
    out = []
    for p, s in zip(seasons["person_id"], seasons["season"], strict=True):
        pref = prefs[p]
        if favourite is not None:
            pref = 0.2 * pref
            pref[ZONES.index(favourite[p])] += 0.8
        zones = rng.choice(ZONES, size=250, p=pref / pref.sum())
        for z in zones:
            band, x, y = SPOT[str(z)]
            value = 3 if band in ("three", "deep3") else 2
            p_make = (0.35 if value == 3 else 0.5) + skill[p]
            out.append((p, s, bool(rng.random() < p_make), value, x, y, band, value * 0.45))
    return pd.DataFrame(
        out, columns=["person_id", "season", "made", "value", "x", "y", "band", "xpts"]
    )


@pytest.fixture(scope="module")
def league() -> pd.DataFrame:
    return _league(11, 200)


def test_features_follow_the_interface_definitions() -> None:
    row = _league(3, 1, seasons=(2020,)).iloc[[0]]
    raw = raw_features(row, None).iloc[0]
    poss = float(row["poss"].iloc[0])
    r = row.iloc[0]
    assert raw["usage"] == pytest.approx(100 * (r.fg2a + r.fg3a + 0.44 * r.fta + r.tov) / poss)
    assert raw["fg2a"] == pytest.approx(100 * r.fg2a / poss)
    assert raw["pf"] == pytest.approx(100 * r.pf / poss)
    assert raw["ts"] == pytest.approx(r.pts / (2 * (r.fg2a + r.fg3a + 0.44 * r.fta)))
    assert raw["ft"] == pytest.approx(r.ftm / r.fta)
    assert raw[list(SHOT_FEATURES)].isna().all()


def test_self_retrieval_beats_a_random_ranking(league: pd.DataFrame) -> None:
    """A player's season s finds his own season s+1 in the top 10 of the whole pool more often
    than a random ranking. Under random ranking every trial is a Bernoulli with probability
    10 / (n - 1) (the row itself left out), so the hits have mean sum(p) and variance
    sum(p (1 - p)); require hits above mean + 4 standard errors (one-sided p < 4e-5)."""
    emb = embed(league, None, pool_seasons=POOL)
    result = self_retrieval_rate(emb, k=10)
    assert result.trials == 200 * (len(POOL) - 1)
    assert result.chance == pytest.approx(result.trials * 10 / (len(league) - 1))
    assert result.hits > result.chance + 4 * np.sqrt(result.variance)
    assert result.z() > 4.0


def test_min_poss_filters_the_pool(league: pd.DataFrame) -> None:
    emb = embed(league, None, pool_seasons=POOL, min_poss=1500.0)
    kept = league[league["poss"] >= 1500.0]
    assert 0 < len(emb.keys) == len(kept)
    assert emb.matrix.shape == (len(kept), len(BOX_FEATURES))
    with pytest.raises(ValueError, match="empty"):
        embed(league, None, pool_seasons=POOL, min_poss=1e9)


def test_a_person_never_matches_any_of_his_own_seasons(league: pd.DataFrame) -> None:
    emb = embed(league, None, pool_seasons=POOL)
    out = neighbours(raw_features(league, None), emb, k=10)
    assert len(out) == len(league) * 10
    assert (out["person_id"] != out["match_person_id"]).all()
    grouped = out.groupby(["person_id", "competition", "season"])
    assert grouped["rank"].apply(lambda r: list(r) == list(range(1, 11))).all()
    assert grouped["distance"].apply(lambda d: d.is_monotonic_increasing).all()
    assert out["score"].between(-1.0, 1.0).all()
    # the evaluation helper does rank own seasons; the product output never does
    assert self_retrieval_rate(emb).hits > 0


def test_standardisation_uses_the_pool_rows_only(league: pd.DataFrame) -> None:
    extended = pd.concat([league, _league(12, 50, seasons=(2023, 2024))], ignore_index=True)
    base = embed(league, None, pool_seasons=POOL)
    with_later = embed(extended, None, pool_seasons=POOL)
    changed = extended.copy()
    late = changed["season"] >= 2023
    changed.loc[late, ["pts", "ast", "fg3a"]] *= 7  # wild non-pool rows
    wild = embed(changed, None, pool_seasons=POOL)
    for other in (with_later, wild):
        assert np.array_equal(base.mean, other.mean)
        assert np.array_equal(base.sd, other.sd)
        assert np.array_equal(base.matrix, other.matrix)
        pd.testing.assert_frame_equal(base.keys, other.keys)


def test_a_pool_before_a_cutoff_ignores_later_seasons(league: pd.DataFrame) -> None:
    cutoff = 2021
    early = [s for s in POOL if s < cutoff]
    alone = embed(league[league["season"] < cutoff], None, pool_seasons=early)
    inside = embed(league, None, pool_seasons=early)
    assert np.array_equal(alone.mean, inside.mean)
    assert np.array_equal(alone.sd, inside.sd)
    assert np.array_equal(alone.matrix, inside.matrix)
    assert inside.keys["season"].max() < cutoff


def test_a_query_outside_the_pool_is_standardised_with_the_pool(league: pd.DataFrame) -> None:
    emb = embed(league, None, pool_seasons=POOL[:3])
    live = raw_features(league[league["season"] == 2022], None)
    out = neighbours(live, emb, k=5)
    assert (out["season"] == 2022).all() and (out["match_season"] <= 2021).all()
    # the same stat line is the same point in standardised space whatever its season label
    inside = raw_features(league[league["season"] == 2021], None)
    swapped = inside.assign(season=2099)
    a = neighbours(inside, emb, k=5).drop(columns="season")
    b = neighbours(swapped, emb, k=5).drop(columns="season")
    pd.testing.assert_frame_equal(a, b)


def test_zero_attempt_percentages_take_the_pool_mean() -> None:
    rows = _league(5, 40, seasons=(2020,))
    rows.loc[0, ["fta", "ftm"]] = 0
    emb = embed(rows, None, pool_seasons=(2020,))
    ft = BOX_FEATURES.index("ft")
    first = emb.keys.index[emb.keys["person_id"] == rows.loc[0, "person_id"]][0]
    assert emb.matrix[first, ft] == 0.0
    rest = raw_features(rows.iloc[1:], None)["ft"].to_numpy()
    assert emb.mean[ft] == pytest.approx(np.nanmean(rest))
    assert np.isfinite(emb.matrix).all()


def test_ties_break_by_the_match_keys() -> None:
    rows = _league(6, 30, seasons=(2020,))
    clones = pd.concat(
        [rows.iloc[[0]].assign(person_id=f"P:z{c}") for c in "cab"], ignore_index=True
    )
    pool = pd.concat([rows.iloc[1:], clones], ignore_index=True)
    emb = embed(pool, None, pool_seasons=(2020,))
    out = neighbours(raw_features(rows.iloc[[0]], None), emb, k=3)
    assert list(out["match_person_id"]) == ["P:za", "P:zb", "P:zc"]
    assert np.ptp(out["distance"].to_numpy()) == 0.0


def test_shot_features_are_used_when_both_sides_have_them() -> None:
    """Three archetypes differ only in their favourite zone; the box lines are exchangeable. Of
    a query's neighbours the same archetype is over-represented: 120 queries x 10 neighbours at
    chance 1/3 gives mean 400, sd sqrt(1200 * 1/3 * 2/3) = 16.3; require mean + 4 sd."""
    seasons = _league(21, 60, seasons=(2020, 2021))
    persons = sorted(set(seasons["person_id"]))
    kinds = ["rim", "corner3_l", "mid_c"]
    favourite = {p: kinds[i % 3] for i, p in enumerate(persons)}
    shots = _shots(22, seasons, favourite)
    emb = embed(seasons, shots, pool_seasons=(2020, 2021))
    assert emb.features == (*BOX_FEATURES, *SHOT_FEATURES)
    out = neighbours(raw_features(seasons, shots), emb, k=10)
    same = (out["person_id"].map(favourite) == out["match_person_id"].map(favourite)).sum()
    draws = len(out)
    assert draws == 1200
    assert same > draws / 3 + 4 * np.sqrt(draws * (1 / 3) * (2 / 3))


def test_gbl_queries_use_box_features_over_the_whole_pool() -> None:
    el = _league(31, 40, seasons=(2020, 2021))
    gbl = _league(32, 40, seasons=(2020, 2021), competition="gbl")
    both = pd.concat([el, gbl], ignore_index=True)
    shots = _shots(33, el)
    with_shots = embed(both, shots, pool_seasons=(2020, 2021))
    box_only = embed(both, None, pool_seasons=(2020, 2021))
    assert with_shots.features == (*BOX_FEATURES, *SHOT_FEATURES)
    assert box_only.features == BOX_FEATURES
    gbl_query = raw_features(gbl, None)
    mixed = neighbours(gbl_query, with_shots, k=10)
    pd.testing.assert_frame_equal(mixed, neighbours(gbl_query, box_only, k=10))
    assert {"euroleague", "gbl"} <= set(mixed["match_competition"])
    # a EuroLeague query with shot features only meets rows that have them
    el_out = neighbours(raw_features(el, shots), with_shots, k=10)
    assert (el_out["match_competition"] == "euroleague").all()
    # a query without shot columns falls back to the box features
    plain = raw_features(el, None)
    pd.testing.assert_frame_equal(
        neighbours(plain, with_shots, k=10), neighbours(plain, box_only, k=10)
    )


def test_shot_features_need_enough_shots() -> None:
    seasons = _league(41, 5, seasons=(2020,))
    shots = _shots(42, seasons)
    thin = pd.concat(
        [shots[shots["person_id"] != "P:e0"], shots[shots["person_id"] == "P:e0"].head(10)],
        ignore_index=True,
    )
    raw = raw_features(seasons, thin)
    cols = list(SHOT_FEATURES)
    assert raw.loc[raw["person_id"] == "P:e0", cols].isna().all(axis=None)
    assert raw.loc[raw["person_id"] != "P:e0", cols].notna().all(axis=None)
    zones = raw[[f"zone_{z}" for z in ZONES]].dropna()
    assert np.allclose((zones**2).sum(axis=1), 1.0)  # square roots of shares


def test_output_columns_are_the_interface_ones() -> None:
    assert tuple(SIMILAR_SCHEMA.columns) == (
        "person_id",
        "competition",
        "season",
        "rank",
        "match_person_id",
        "match_competition",
        "match_season",
        "distance",
        "score",
    )
