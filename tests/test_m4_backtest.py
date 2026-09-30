import json

import numpy as np
import pandas as pd
import pytest

from eurohoops.config import M4Backtest
from eurohoops.eval.m4_backtest import (
    MODELS,
    net_offset,
    person_bootstrap,
    run_m4_backtest,
    stat_sds,
    team_ratios,
    weighted_loss,
)
from eurohoops.models.box_impact import STAT_COLUMNS

CLUBS = {"00000001": "PAN"}
SPEC = M4Backtest(tuning=(2019, 2020, 2021, 2022), validation=(2023,), test=(2024, 2025))
BASE = dict(
    zip(STAT_COLUMNS, (20.0, 12.0, 9.0, 6.0, 3.0, 8.0, 5.0, 2.0, 1.0, 3.5, 5.0), strict=True)
)


def rates_row(
    person: str, comp: str, season: int, scale: float, minutes: float
) -> dict[str, object]:
    return {
        "person_id": person,
        "competition": comp,
        "season": season,
        "team": "PAN" if comp == "euroleague" else "00000001",
        "minutes": minutes,
        "poss": minutes * 2.0,
        **{f"rate_{k}": v * scale for k, v in BASE.items()},
    }


def synthetic(
    n_movers: int, seed: int = 3
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    rng = np.random.default_rng(seed)
    pairs, rates = [], []
    for s in range(2016, 2026):
        for i in range(12):
            person = f"D{s}_{i}"
            skill = float(rng.lognormal(0, 0.2))
            gbl, el = skill * 1.2, skill * 0.9 * float(rng.lognormal(0, 0.1))
            rates += [
                rates_row(person, "gbl", s, gbl, 600),
                rates_row(person, "euroleague", s, el, 500),
            ]
            pairs.append(("dual", person, s, s, gbl, el))
        for i in range(n_movers):
            person = f"M{s}_{i}"
            skill = float(rng.lognormal(0, 0.2))
            gbl, el = skill * 1.2, skill * 0.85 * float(rng.lognormal(0, 0.1))
            rates += [
                rates_row(person, "gbl", s - 1, gbl, 700),
                rates_row(person, "euroleague", s, el, 400),
            ]
            pairs.append(("gbl_to_el", person, s - 1, s, gbl, el))
    frame = pd.DataFrame(
        [
            {
                "person_id": person,
                "pair_type": kind,
                "season_gbl": sg,
                "season_el": se,
                "minutes_gbl": 600.0,
                "minutes_el": 450.0,
                "team_gbl": "00000001",
                "team_el": "PAN",
                "team_gap": 0.0,
                "later_season": max(sg, se),
                **{f"rate_gbl_{k}": v * g for k, v in BASE.items()},
                **{f"rate_el_{k}": v * e for k, v in BASE.items()},
            }
            for kind, person, sg, se, g, e in pairs
        ]
    )
    games = {
        comp: pd.DataFrame(
            [
                {"season": s, "team": team, "poss": 1000.0, **{k: v * f for k, v in BASE.items()}}
                for s in range(2016, 2026)
            ]
        )
        for comp, team, f in (("gbl", "00000001", 12.0), ("euroleague", "PAN", 9.0))
    }
    return frame, pd.DataFrame(rates), games


def test_pooled_rule_with_few_validation_movers_and_report_shape() -> None:
    pairs, rates, games = synthetic(n_movers=2)
    report, translation = run_m4_backtest(
        pairs, rates, games, spec=SPEC, clubs=CLUBS, score_test=True
    )
    assert report["chosen"]["variant"] == "translate"
    assert report["chosen"]["rule"].startswith("declared")
    gate = report["gate"]
    assert gate["pooled"] and gate["seasons"] == [2019, 2020, 2021, 2022, 2023]
    assert gate["n_movers"] == 10
    assert set(report["metrics"]) == {"tuning", "validation", "test"}
    assert set(report["metrics"]["tuning"]["loss"]) == set(MODELS)
    # the synthetic truth is a pure per-stat scale, so translation must beat same stats
    assert (
        report["metrics"]["tuning"]["loss"]["translate"]
        < report["metrics"]["tuning"]["loss"]["same_stats"]
    )
    assert gate["passed"]
    assert set(translation["fits_by_target_season"]) == {str(s) for s in range(2019, 2026)}
    json.dumps(report)


def test_choice_on_tuning_when_validation_has_enough_movers() -> None:
    pairs, rates, games = synthetic(n_movers=21)
    report, _ = run_m4_backtest(pairs, rates, games, spec=SPEC, clubs=CLUBS)
    assert report["chosen"]["rule"] == "lowest tuning loss of translate and translate_team"
    assert not report["gate"]["pooled"] and report["gate"]["seasons"] == [2023]
    assert not report["test_scored"] and "test" not in report["metrics"]


def test_tuning_only_scores_no_validation_and_is_walk_forward() -> None:
    pairs, rates, games = synthetic(n_movers=2)
    report, _ = run_m4_backtest(pairs, rates, games, spec=SPEC, clubs=CLUBS, tuning_only=True)
    assert report["gate"] is None and not report["validation_scored"]
    later = pairs["later_season"] >= 2023
    edited = pairs.copy()
    rate_cols = [c for c in pairs.columns if c.startswith("rate_")]
    edited.loc[later, rate_cols] = edited.loc[later, rate_cols] * 3.0
    edited_games = {
        c: g.assign(**{k: np.where(g["season"] >= 2023, g[k] * 3, g[k]) for k in STAT_COLUMNS})
        for c, g in games.items()
    }
    again, _ = run_m4_backtest(
        edited, rates, edited_games, spec=SPEC, clubs=CLUBS, tuning_only=True
    )
    assert again["metrics"]["tuning"] == report["metrics"]["tuning"]
    with pytest.raises(ValueError, match="exclude"):
        run_m4_backtest(
            pairs, rates, games, spec=SPEC, clubs=CLUBS, tuning_only=True, score_test=True
        )


def test_runs_are_identical() -> None:
    pairs, rates, games = synthetic(n_movers=2)
    a = run_m4_backtest(pairs, rates, games, spec=SPEC, clubs=CLUBS, score_test=True)
    b = run_m4_backtest(pairs, rates, games, spec=SPEC, clubs=CLUBS, score_test=True)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_helpers_on_hand_values() -> None:
    table = pd.DataFrame(
        {
            "person_id": ["a", "b", "b"],
            "minutes_el": [100.0, 300.0, 100.0],
            "x": [1.0, 3.0, 5.0],
            "y": [1.0, 1.0, 1.0],
        }
    )
    assert weighted_loss(table, "x") == pytest.approx((100 + 900 + 500) / 500)
    diff = person_bootstrap(table, "x", "y", 200, 1)
    assert diff["mean"] == pytest.approx((0 * 100 + 2 * 300 + 4 * 100) / 500)
    assert diff["ci95"][0] <= diff["mean"] <= diff["ci95"][1]
    assert person_bootstrap(table.iloc[0:0], "x", "y", 10, 1)["mean"] is None
    _, rates, games = synthetic(n_movers=0)
    sds = stat_sds(rates, (2019, 2020), 300.0, "euroleague")
    assert all(v > 0 for v in sds.values())
    ratios = team_ratios(games, CLUBS, before=2020)
    assert ratios == pytest.approx(np.full(len(STAT_COLUMNS), 9.0 / 12.0))
    assert team_ratios(games, CLUBS, before=2000) == pytest.approx(np.ones(len(STAT_COLUMNS)))


def test_net_offset_is_gbl_minus_el_relative_net() -> None:
    net = pd.DataFrame(
        [
            ("gbl", 2023, "00000001", 20.0),
            ("euroleague", 2023, "PAN", 2.0),
            ("gbl", 2024, "00000001", 16.0),
            ("euroleague", 2024, "PAN", 4.0),
            ("gbl", 2025, "00000001", 10.0),
        ],
        columns=["competition", "season", "team", "net"],
    )
    out = net_offset(net, CLUBS, 100, 7)
    assert out["mean"] == 15.0
    assert [r["gap"] for r in out["club_seasons"]] == [18.0, 12.0]
    assert net_offset(net.iloc[0:0], CLUBS, 10, 7)["mean"] is None
