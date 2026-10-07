"""Tests for the M6 over/under-performance board (``models/board.py``): hand-built players,
planted-skill synthetic data, leakage and the 3P% prior's walk-forward fit. Synthetic data only."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from eurohoops.models import board as b
from eurohoops.models.player_seasons import PLAYER_SEASONS_SCHEMA
from eurohoops.parse.schemas import validated


def _obs(rows: list[tuple[str, float, float, float, float]], season: int = 2024) -> pd.DataFrame:
    """Observations from ``(person, observed, expected, se, n)``."""
    return b.validated(
        pd.DataFrame(
            {
                "person_id": [r[0] for r in rows],
                "competition": "euroleague",
                "season": np.int64(season),
                "observed": [r[1] for r in rows],
                "expected": [r[2] for r in rows],
                "se": [r[3] for r in rows],
                "n": [r[4] for r in rows],
            }
        ),
        b.OBSERVATION_SCHEMA,
    )


def test_interface_constants() -> None:
    assert b.DIMENSIONS == ("shot_making", "fg3_pct", "on_off")
    assert b.LABELS == ("likely regression", "likely real", "too few attempts")
    assert b.N_MIN == {"shot_making": 200, "fg3_pct": 50, "on_off": 1000}


def test_labels_follow_the_formula_on_hand_built_players() -> None:
    # r = 0.5 and mean_n = 100 give k = 100 * (1 - 0.5) / 0.5 = 100 and persist = n / (n + 100),
    # which is exactly 0.5 at n = 100 and 1/3 at n = 50 (the 3P% minimum).
    stab = b.Stability("fg3_pct", 0.5, 100.0, 20)
    players = [
        # id, observed, expected, se, n   -> expected label
        ("too_few_big_z", 0.60, 0.35, 0.01, 49),  # n < 50: too few however large z is
        ("boundary_n", 0.60, 0.35, 0.01, 50),  # n = N_MIN counts; persist 1/3 < 0.5 -> regression
        ("real_over", 0.37, 0.35, 0.01, 100),  # persist 0.5, z = +2 -> real
        ("real_under", 0.33, 0.35, 0.01, 100),  # persist 0.5, z = -2 -> real (negative gap)
        ("z_at_cut", 1.645, 0.0, 1.0, 100),  # |z| = 1.645 exactly -> real
        ("z_below_cut", 1.644, 0.0, 1.0, 100),  # |z| < 1.645 -> regression
        ("persist_below", 0.45, 0.35, 0.01, 99),  # persist 99/199 < 0.5, z = 10 -> regression
        ("big_n_over", 0.40, 0.35, 0.02, 1000),  # persist 10/11, z = 2.5 -> real
        ("big_n_small_z", 0.355, 0.35, 0.02, 1000),  # z = 0.25 -> regression
        ("neg_small_z", 0.34, 0.35, 0.02, 400),  # z = -0.5 -> regression
    ]
    want = {
        "too_few_big_z": "too few attempts",
        "boundary_n": "likely regression",
        "real_over": "likely real",
        "real_under": "likely real",
        "z_at_cut": "likely real",
        "z_below_cut": "likely regression",
        "persist_below": "likely regression",
        "big_n_over": "likely real",
        "big_n_small_z": "likely regression",
        "neg_small_z": "likely regression",
    }
    out = b.board({"fg3_pct": _obs(players)}, {"fg3_pct": stab}).set_index("person_id")
    assert out["label"].to_dict() == want
    for pid, observed, expected, se, n in players:
        row = out.loc[pid]
        persist = n / (n + 100.0)
        assert row["gap"] == pytest.approx(observed - expected)
        assert row["z"] == pytest.approx((observed - expected) / se)
        assert row["persist"] == pytest.approx(persist)
        assert row["expected_next"] == pytest.approx(expected + persist * (observed - expected))
        assert row["stability"] == 0.5
    assert out["gap"].lt(0).any() and out["gap"].gt(0).any()


def test_n_min_is_per_dimension() -> None:
    stab = {d: b.Stability(d, 0.9, 100.0, 9) for d in b.DIMENSIONS}
    obs = {
        d: _obs([("below", 5.0, 0.0, 1.0, b.N_MIN[d] - 1), ("at", 5.0, 0.0, 1.0, b.N_MIN[d])])
        for d in b.DIMENSIONS
    }
    out = b.board(obs, stab)
    for d in b.DIMENSIONS:
        labels = out[out["dimension"] == d].set_index("person_id")["label"]
        assert labels["below"] == "too few attempts"
        assert labels["at"] != "too few attempts"


def test_persist_limits_and_board_schema() -> None:
    n = np.array([10.0, 100.0])
    assert b.Stability("on_off", -0.2, 50.0, 5).persist(n).tolist() == [0.0, 0.0]
    assert b.Stability("on_off", 1.0, 50.0, 5).persist(n).tolist() == [1.0, 1.0]
    bad = b.board(
        {"on_off": _obs([("p", 1.0, 0.0, 1.0, 2000)])},
        {"on_off": b.Stability("on_off", 0.3, 800.0, 9)},
    )
    broken = bad.assign(persist=1.5)
    with pytest.raises(Exception, match="persist"):
        b.BOARD_SCHEMA.validate(broken)
    with pytest.raises(Exception, match="se"):
        b.BOARD_SCHEMA.validate(bad.assign(se=0.0))
    with pytest.raises(ValueError, match="unknown dimension"):
        b.board({"height": _obs([("p", 1.0, 0.0, 1.0, 5)])}, {})


def test_stability_pairs_consecutive_tuning_seasons_only() -> None:
    rows = []
    for i in range(6):
        rows.append((f"p{i}", float(i), 0.0, 1.0, 300.0))
    s1 = _obs(rows, 2020)
    s2 = _obs([(p, 2.0 * o, 0.0, 1.0, 500.0) for p, o, *_ in rows], 2021)  # gap doubles: r = 1
    s3 = _obs([(p, -o, 0.0, 1.0, 500.0) for p, o, *_ in rows], 2022)  # anti-correlated later season
    thin = _obs([("thin", 99.0, 0.0, 1.0, 10.0)], 2020)  # below N_MIN: excluded
    thin2 = _obs([("thin", -99.0, 0.0, 1.0, 10.0)], 2021)
    obs = pd.concat([s1, s2, s3, thin, thin2], ignore_index=True)
    st = b.stability(obs, "shot_making", (2020, 2021))
    assert st.r == pytest.approx(1.0)
    assert st.pairs == 6
    assert st.mean_n == pytest.approx(400.0)  # (300 + 500) / 2 for each pair
    both = b.stability(obs, "shot_making", (2020, 2021, 2022))
    assert both.pairs == 12 and both.r < 1.0  # 2021 -> 2022 pairs are in once asked for
    with pytest.raises(ValueError, match="pairs"):
        b.stability(obs, "shot_making", (2020,))


# ---- the three observation builders on hand-computed inputs ----


def _xwalk() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "competition": ["euroleague", "euroleague", "gbl"],
            "source_id": ["P1", "P2", "9"],
            "person_id": ["person_1", "person_2", "person_g"],
        }
    )


def test_shot_making_observation_by_hand() -> None:
    shots = pd.DataFrame(
        {
            "competition": "euroleague",
            "season": [2024, 2024, 2024, 2024, 2024, 2023],
            "game_id": ["g1"] * 6,
            "event": [1, 2, 3, 4, 5, 6],
            "shooter": ["P1", "P1", "P1", "P1", "P3", "P1"],
            "made": [True, False, True, False, True, True],
            "value": [2, 3, 2, 2, 3, 2],
        }
    )
    xpts = pd.DataFrame(
        {
            "game_id": ["g1"] * 12,
            "event": list(range(1, 7)) * 2,
            "p_make": [0.5, 0.3, 0.6, 0.5, 0.4, 0.9] + [0.1] * 6,
            "variant": ["m2"] * 6 + ["other"] * 6,
        }
    )
    out = b.shot_making_observations(shots, xpts, _xwalk(), 2024, "m2").set_index("person_id")
    points, expected = 4.0, 2 * 0.5 + 3 * 0.3 + 2 * 0.6 + 2 * 0.5
    var = 4 * 0.25 + 9 * 0.21 + 4 * 0.24 + 4 * 0.25
    p1 = out.loc["person_1"]
    assert p1["n"] == 4
    assert p1["observed"] == pytest.approx(100 * (points - expected) / 4)
    assert p1["expected"] == 0.0
    assert p1["se"] == pytest.approx(100 * math.sqrt(var) / 4)
    assert out.loc["P:P3", "n"] == 1  # an unmapped shooter keeps the crosswalk's scheme
    assert set(out.index) == {"person_1", "P:P3"}  # the 2023 shot of P1 is not read


def test_shot_making_rejects_ambiguous_xpts() -> None:
    xpts = pd.DataFrame(
        {"game_id": ["g", "g"], "event": [1, 1], "p_make": [0.4, 0.5], "variant": "m2"}
    )
    shots = pd.DataFrame(
        {
            "competition": "euroleague",
            "season": [2024],
            "game_id": ["g"],
            "event": [1],
            "shooter": ["P1"],
            "made": [True],
            "value": [2],
        }
    )
    with pytest.raises(ValueError, match="several rows"):
        b.shot_making_observations(shots, xpts, _xwalk(), 2024, "m2")


def _seasons(rows: list[dict[str, object]]) -> pd.DataFrame:
    base = {c: 0 for c in PLAYER_SEASONS_SCHEMA.columns}
    base.update(
        competition="euroleague",
        partial=False,
        mapped=True,
        team="AAA",
        games=10,
        minutes=300.0,
        poss=600.0,
    )
    frame = pd.DataFrame([{**base, **r} for r in rows])
    return validated(frame, PLAYER_SEASONS_SCHEMA)


def _fit_rows(
    season: int, intercept: float, slope: float, count: int = 30
) -> list[dict[str, object]]:
    """Rows whose 3P% is a line of FT% (rounded to whole makes at 2,000 3PA)."""
    rows = []
    for i in range(count):
        ft = 0.55 + 0.4 * i / (count - 1)
        rows.append(
            {
                "person_id": f"s{season}_{i}",
                "season": season,
                "fg3a": 2000,
                "fg3m": round(2000 * (intercept + slope * ft)),
                "fta": 1000,
                "ftm": round(1000 * ft),
                "debut_season": season,
            }
        )
    return rows


def test_fg3_prior_uses_only_seasons_before_the_cutoff() -> None:
    early = _fit_rows(2021, 0.20, 0.30) + _fit_rows(2022, 0.20, 0.30)
    later = _fit_rows(2023, 0.60, -0.30) + _fit_rows(2024, 0.60, -0.30)  # a different relation
    seasons = _seasons(early + later)
    intercept, slope, _ = b.fg3_fit(seasons, 2023, "euroleague")
    # The planted line is recovered to the rounding of whole makes (<= 0.5 / 2000 = 0.00025 in 3P%
    # per row); 0.005 is 20x that and far below the 0.6 slope gap to the later relation.
    assert slope == pytest.approx(0.30, abs=0.005)
    assert intercept == pytest.approx(0.20, abs=0.005)
    # Fitting at 2025 sees both relations (a blend), so the 2023 fit really excluded the later ones.
    blend = b.fg3_fit(seasons, 2025, "euroleague")
    assert abs(blend[1] - 0.30) > 0.1
    # A season-2023 row is judged against the early line only.
    rows = _seasons(
        [
            *early,
            {
                "person_id": "x",
                "season": 2023,
                "fg3a": 100,
                "fg3m": 40,
                "fta": 100,
                "ftm": 80,
                "debut_season": 2021,
            },
        ]
    )
    out = b.fg3_observations(rows, 2023).set_index("person_id")
    assert out.loc["x", "expected"] == pytest.approx(0.20 + 0.30 * 0.80, abs=0.005)
    assert out.loc["x", "observed"] == 0.4
    assert out.loc["x", "se"] == pytest.approx(
        math.sqrt(out.loc["x", "expected"] * (1 - out.loc["x", "expected"]) / 100)
    )


def test_fg3_few_ft_attempts_take_the_fitted_mean_and_gbl_without_history_is_skipped() -> None:
    early = _fit_rows(2022, 0.20, 0.30)
    mean_ft = 0.55 + 0.4 / 2  # equal weights, evenly spaced FT%
    rows = _seasons(
        [
            *early,
            {
                "person_id": "few_ft",
                "season": 2023,
                "fg3a": 80,
                "fg3m": 30,
                "fta": 5,
                "ftm": 5,
                "debut_season": 2022,
            },
            {
                "person_id": "gbl_new",
                "season": 2023,
                "competition": "gbl",
                "fg3a": 80,
                "fg3m": 30,
                "fta": 50,
                "ftm": 40,
                "debut_season": 2023,
            },
            {
                "person_id": "no_3pa",
                "season": 2023,
                "fg3a": 0,
                "fg3m": 0,
                "fta": 50,
                "ftm": 40,
                "debut_season": 2022,
            },
        ]
    )
    out = b.fg3_observations(rows, 2023).set_index("person_id")
    assert set(out.index) == {"few_ft"}
    assert out.loc["few_ft", "expected"] == pytest.approx(0.20 + 0.30 * mean_ft, abs=0.005)
    with pytest.raises(ValueError, match="rows before"):
        b.fg3_fit(rows, 2022, "euroleague")


def _hand_stints() -> tuple[pd.DataFrame, pd.DataFrame]:
    def stint(game: str, hp: list[str], ap: list[str], hpts: int, apts: int) -> dict[str, object]:
        return {
            "game_id": game,
            "season": 2024,
            "home": "AAA",
            "away": "BBB",
            "home_players": hp,
            "away_players": ap,
            "home_points": hpts,
            "away_points": apts,
            "home_poss": 10,
            "away_poss": 10,
        }

    stints = pd.DataFrame(
        [
            stint("g1", ["a1", "a2"], ["b1", "b2"], 10, 5),
            stint("g1", ["a1", "a3"], ["b1", "b2"], 4, 8),
            stint("g1", ["a2", "a3"], ["b1", "b2"], 6, 6),
            stint("bad", ["a1", "a2"], ["b1", "b2"], 90, 0),  # failing game: never read
            {**stint("g0", ["a1"], ["b1"], 90, 0), "season": 2023},  # another season: never read
        ]
    )
    checks = pd.DataFrame({"game_id": ["g1", "bad", "g0"], "passed": [True, False, True]})
    return stints, checks


def test_on_off_by_hand() -> None:
    stints, checks = _hand_stints()
    xwalk = pd.DataFrame(
        {
            "competition": "euroleague",
            "source_id": ["a1", "a2", "a3"],
            "person_id": ["pa1", "pa2", "pa3"],
        }
    )
    brapm = pd.DataFrame(
        {
            "player_id": ["a1", "a2", "b1", "a3"],
            "season": [2024, 2024, 2024, 2023],
            "total": [2.0, -1.0, 4.0, 9.0],
            "sd_total": [3.0, 1.0, 1.0, 1.0],
        }
    )
    out = b.on_off_observations(stints, checks, brapm, xwalk, 2024).set_index("person_id")
    # b1 is on court for every possession of his team: no off-court sample, no row. a3 has no
    # 2024 BRAPM row. Two players remain.
    assert set(out.index) == {"pa1", "pa2"}
    # points per possession of the six stint sides, each of 10 possessions
    ppp = np.array([1.0, 0.4, 0.6, 0.5, 0.8, 0.6])
    s2 = float((10 * (ppp - ppp.mean()) ** 2).sum() / 5)
    # a1: on stints 1-2 (A 14 for, 13 against in 20 poss), off stint 3 (6-6): +5 vs 0
    a1 = out.loc["pa1"]
    assert a1["observed"] == pytest.approx(100 * (14 / 20 - 13 / 20) - 0.0)
    assert a1["expected"] == 2.0
    assert a1["n"] == 20
    se_diff = 100 * math.sqrt(2 * s2 * (1 / 20 + 1 / 10))
    assert a1["se"] == pytest.approx(math.hypot(se_diff, 3.0))
    # a2: on stints 1 and 3 (16 for, 11 against), off stint 2 (4 for, 8 against)
    a2 = out.loc["pa2"]
    assert a2["observed"] == pytest.approx(100 * (16 / 20 - 11 / 20) - 100 * (4 / 10 - 8 / 10))
    assert a2["expected"] == -1.0


# ---- planted skill: labels separate who keeps the gap ----


def _planted(n_players: int, seasons: tuple[int, ...], seed: int) -> pd.DataFrame:
    """Observed = 0.35 + skill + sampling noise (sd sqrt(.35 * .65 / n)) at fixed n per player."""
    rng = np.random.default_rng(seed)
    skill = rng.normal(0.0, 0.03, n_players)
    n = rng.integers(50, 401, n_players).astype(np.float64)
    se = np.sqrt(0.35 * 0.65 / n)
    frames = []
    for s in seasons:
        frames.append(
            b.validated(
                pd.DataFrame(
                    {
                        "person_id": [f"p{i}" for i in range(n_players)],
                        "competition": "euroleague",
                        "season": np.int64(s),
                        "observed": 0.35 + skill + rng.normal(0.0, 1.0, n_players) * se,
                        "expected": 0.35,
                        "se": se,
                        "n": n,
                    }
                ),
                b.OBSERVATION_SCHEMA,
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_likely_regression_players_regress_more_than_likely_real() -> None:
    # Seasons 1-2 are the tuning seasons (stability), 3 is the board season, 4 the next season.
    data = _planted(4000, (1, 2, 3, 4), seed=7)
    stab = b.stability(data, "fg3_pct", (1, 2))
    now = data[data["season"] == 3]
    out = b.board({"fg3_pct": now}, {"fg3_pct": stab})
    nxt = data[data["season"] == 4]
    next_gap = pd.DataFrame(
        {
            "person_id": nxt["person_id"],
            "competition": nxt["competition"],
            "dimension": "fg3_pct",
            "gap": nxt["observed"] - nxt["expected"],
        }
    )
    result = b.retention_backtest(out, next_gap)
    # Both groups must be big enough for the standard errors to mean something.
    assert result.count["likely real"] >= 100
    assert result.count["likely regression"] >= 100
    assert result.slope["likely real"] > result.slope["likely regression"]
    # Statistic: contrast_z, the difference of the two groups' retention slopes in standard errors
    # (slopes of next gap on gap, HC0 errors, disjoint players). Under "no difference in
    # retention" it is ~ N(0, 1), so the threshold 3 is a one-sided false-failure rate of 0.13%;
    # the planted effect is
    # large: theory retains tau^2 / (tau^2 + sigma^2) of a gap, 0.12 at 50 attempts (sigma .067)
    # to 0.60 at 400 (sigma .024) for tau = .03, and the 'real' group is the high-n, high-|z| end.
    assert result.contrast_z >= 3.0


def test_retention_backtest_needs_both_groups() -> None:
    stab = b.Stability("fg3_pct", 0.5, 100.0, 9)
    out = b.board(
        {"fg3_pct": _obs([("a", 0.36, 0.35, 0.01, 100), ("b", 0.351, 0.35, 0.01, 100)])},
        {"fg3_pct": stab},
    )
    nxt = pd.DataFrame(
        {
            "person_id": ["a", "b"],
            "competition": "euroleague",
            "dimension": "fg3_pct",
            "gap": [0.01, 0.0],
        }
    )
    with pytest.raises(ValueError, match="both"):
        b.retention_backtest(out, nxt.iloc[:1])


# ---- leakage ----


def _world() -> dict[str, pd.DataFrame]:
    """A small synthetic world, seasons 2020-2023, for all three builders."""
    rng = np.random.default_rng(11)
    people = [f"P{i}" for i in range(12)]
    xwalk = pd.DataFrame(
        {
            "competition": "euroleague",
            "source_id": people,
            "person_id": [f"pp{i}" for i in range(12)],
        }
    )
    shots = []
    for season in (2020, 2021, 2022, 2023):
        for k, who in enumerate(people):
            for j in range(220):
                shots.append(
                    {
                        "competition": "euroleague",
                        "season": season,
                        "game_id": f"E{season}_{k}",
                        "event": j,
                        "shooter": who,
                        "made": bool(rng.random() < 0.5),
                        "value": int(rng.choice([2, 3])),
                    }
                )
    shots_df = pd.DataFrame(shots)
    xpts = shots_df[["game_id", "event"]].assign(
        p_make=rng.uniform(0.2, 0.7, len(shots_df)), variant="m2"
    )
    rows = []
    for season in (2019, 2020, 2021, 2022, 2023):
        for i in range(40):
            ft = rng.uniform(0.55, 0.95)
            rows.append(
                {
                    "person_id": f"s{i}",
                    "season": season,
                    "fg3a": int(rng.integers(60, 300)),
                    "fg3m": 0,
                    "fta": int(rng.integers(30, 200)),
                    "ftm": 0,
                    "debut_season": 2019,
                    "_ft": ft,
                }
            )
    frame = pd.DataFrame(rows)
    frame["ftm"] = (frame["fta"] * frame["_ft"]).round().astype(int)
    frame["fg3m"] = (
        (frame["fg3a"] * (0.2 + 0.2 * frame["_ft"] + rng.normal(0, 0.03, len(frame))))
        .round()
        .astype(int)
    )
    seasons = _seasons(frame.drop(columns="_ft").to_dict("records"))
    stint_rows = []
    for season in (2020, 2021, 2022, 2023):
        for g in range(6):
            for _ in range(40):
                hp = list(rng.choice(people[:7], 5, replace=False))
                ap = list(rng.choice(people[5:], 5, replace=False))
                stint_rows.append(
                    {
                        "game_id": f"E{season}_{g}",
                        "season": season,
                        "home": "AAA",
                        "away": "BBB",
                        "home_players": hp,
                        "away_players": ap,
                        "home_points": int(rng.integers(0, 12)),
                        "away_points": int(rng.integers(0, 12)),
                        "home_poss": 8,
                        "away_poss": 8,
                    }
                )
    stints = pd.DataFrame(stint_rows)
    checks = pd.DataFrame({"game_id": stints["game_id"].unique(), "passed": True})
    brapm = pd.DataFrame(
        [
            {"player_id": p, "season": s, "total": float(rng.normal(0, 2)), "sd_total": 2.0}
            for s in (2020, 2021, 2022, 2023)
            for p in people
        ]
    )
    return {
        "xwalk": xwalk,
        "shots": shots_df,
        "xpts": xpts,
        "seasons": seasons,
        "stints": stints,
        "checks": checks,
        "brapm": brapm,
    }


def _builders(w: dict[str, pd.DataFrame], season: int) -> dict[str, pd.DataFrame]:
    return {
        "shot_making": b.shot_making_observations(w["shots"], w["xpts"], w["xwalk"], season, "m2"),
        "fg3_pct": b.fg3_observations(w["seasons"], season),
        "on_off": b.on_off_observations(w["stints"], w["checks"], w["brapm"], w["xwalk"], season),
    }


def _exactly_equal(a: dict[str, pd.DataFrame], c: dict[str, pd.DataFrame]) -> bool:
    return all(a[d].equals(c[d]) for d in a)  # DataFrame.equals compares values bit for bit


def test_leakage_later_seasons_change_nothing_earlier_rows_do() -> None:
    world = _world()
    base = _builders(world, 2022)
    assert all(len(v) > 0 for v in base.values())

    # Delete every later-season (2023) row of every input, then scramble them: the cutoff's (2022)
    # rows are bit-identical.
    later_gone = {
        **world,
        "shots": world["shots"][world["shots"]["season"] <= 2022],
        "seasons": world["seasons"][world["seasons"]["season"] <= 2022],
        "stints": world["stints"][world["stints"]["season"] <= 2022],
        "brapm": world["brapm"][world["brapm"]["season"] <= 2022],
    }
    assert _exactly_equal(base, _builders(later_gone, 2022))
    scrambled = {k: v.copy() for k, v in world.items()}
    scrambled["shots"].loc[scrambled["shots"]["season"] > 2022, "made"] = True
    scrambled["seasons"].loc[scrambled["seasons"]["season"] > 2022, ["fg3m", "ftm"]] = [1, 1]
    scrambled["stints"].loc[scrambled["stints"]["season"] > 2022, "home_points"] = 99
    scrambled["brapm"].loc[scrambled["brapm"]["season"] > 2022, "total"] = 50.0
    assert _exactly_equal(base, _builders(scrambled, 2022))

    # Changing an earlier row changes the 3P% rows (the prior is fitted on earlier seasons).
    earlier = {k: v.copy() for k, v in world.items()}
    earlier["seasons"].loc[earlier["seasons"]["season"] == 2020, "fg3m"] = 0
    changed = _builders(earlier, 2022)
    assert not changed["fg3_pct"].equals(base["fg3_pct"])
    # Shot-making and on/off read only their own season by design, so earlier rows leave them alone.
    assert changed["shot_making"].equals(base["shot_making"])
    assert changed["on_off"].equals(base["on_off"])

    # The finished board: stabilities come from earlier tuning seasons, so a change there moves it,
    # a change in later seasons does not.
    def board_at_2022(w: dict[str, pd.DataFrame]) -> pd.DataFrame:
        stabs = {}
        obs = {}
        for d in b.DIMENSIONS:
            tuning = pd.concat([_builders(w, s)[d] for s in (2020, 2021)], ignore_index=True)
            stabs[d] = b.stability(tuning, d, (2020, 2021))
            obs[d] = _builders(w, 2022)[d]
        return b.board(obs, stabs)

    reference = board_at_2022(world)
    assert board_at_2022(later_gone).equals(reference)
    assert board_at_2022(scrambled).equals(reference)
    assert not board_at_2022(earlier).equals(reference)
    assert set(reference["dimension"]) == set(b.DIMENSIONS)
    b.BOARD_SCHEMA.validate(reference)
