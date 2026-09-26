"""F6-F8: team shot quality from out-of-fold xPTS, player shot-making (shrinkage, stability and
the F-k verdict) and the shot-chart court geometry."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eurohoops.eval import shot_making
from eurohoops.eval.shot_making import (
    correlation_ci,
    player_report,
    player_seasons,
    prior_variance,
    shrink,
    stability_verdict,
)
from eurohoops.eval.team_shot_quality import (
    calibration_in_the_large,
    shots_with_xpts,
    team_games,
    team_report,
)
from eurohoops.parse import shots as court
from eurohoops.parse.free_throws import COUNT_COLUMNS

SEASONS = (2011, 2012, 2013)


def _shots(n_games: int = 24, seed: int = 1, skill: float = 0.0) -> pd.DataFrame:
    """Shots of two teams (AAA, BBB) and eight shooters; shooter P0 is a true shot-maker."""
    rng = np.random.default_rng(seed)
    rows = []
    for season in SEASONS:
        for g in range(n_games):
            game_id = f"E{season}_{g}"
            for team, opponent in (("AAA", "BBB"), ("BBB", "AAA")):
                for k in range(60):
                    shooter = f"P{(k % 4) + (0 if team == 'AAA' else 4)}"
                    p = 0.5 + (skill if shooter == "P0" else 0.0)
                    rows.append(
                        {
                            "season": season,
                            "game_id": game_id,
                            "event": len(rows),
                            "team": team,
                            "opponent": opponent,
                            "shooter": shooter,
                            "value": 2 if k % 3 else 3,
                            "made": bool(rng.random() < p),
                            "x": rng.uniform(-6, 6),
                            "y": rng.uniform(0, 8),
                        }
                    )
    return pd.DataFrame(rows)


def _xpts(shots: pd.DataFrame, p: float = 0.5) -> pd.DataFrame:
    return shots[["game_id", "event"]].assign(split="development", p_make=p, variant="spline")


def _ft(shots: pd.DataFrame) -> pd.DataFrame:
    keys = shots[["season", "game_id", "team"]].drop_duplicates()
    ft = keys.assign(**{c: 0 for c in COUNT_COLUMNS})
    ft["fga"], ft["fga_rim"], ft["other_trips"], ft["other_ftm"] = 60, 60, 10, 15
    ft["trips"], ft["ftm"] = 10, 15
    return ft.reset_index(drop=True)


def test_team_games_pair_offence_with_the_defence_it_allowed() -> None:
    shots = _shots()
    scored = shots_with_xpts(shots, _xpts(shots))
    tg = team_games(scored, _ft(shots), (2011, 2012), (2013,))
    assert len(tg) == 2 * 24 * 3
    first = tg[tg["game_id"] == "E2011_0"].set_index("team")
    assert first.loc["AAA", "xpts_allowed"] == first.loc["BBB", "xpts"]
    assert first.loc["AAA", "fg_points_allowed"] == first.loc["BBB", "fg_points"]
    assert (tg["expected_ft"] == 15.0).all()  # every season has the same FT rates
    np.testing.assert_allclose(tg["total_shot_value"], tg["xpts"] + tg["expected_ft"])
    report = team_report(scored, tg, (2011, 2012), "spline")
    season = report["team_seasons"][0]
    assert (
        season["offence"]["xpts_per_shot"]["ci95"][0] <= season["offence"]["xpts_per_shot"]["mean"]
    )
    assert report["chosen_variant"] == "spline"


def test_calibration_in_the_large_flags_a_biased_season() -> None:
    shots = _shots()
    scored = shots_with_xpts(shots, _xpts(shots))
    rate = scored.groupby("season").apply(lambda s: s["points"].sum() / s["value"].sum())
    fair = scored.assign(xpts=scored["value"] * scored["season"].map(rate))
    assert all(v["within_tolerance"] for v in calibration_in_the_large(fair, SEASONS).values())
    biased = fair.assign(xpts=fair["xpts"] * np.where(fair["season"] == 2012, 1.01, 1.0))
    large = calibration_in_the_large(biased, SEASONS)
    assert not large["2012"]["within_tolerance"] and large["2011"]["within_tolerance"]


def test_only_out_of_fold_xpts_reach_team_numbers() -> None:
    """Leakage (F6): shots without an out-of-fold xPTS row are dropped, never filled in, and
    the team numbers read only the xPTS the out-of-fold table holds."""
    shots = _shots()
    xpts = _xpts(shots)
    partial = xpts[xpts["game_id"] != "E2012_3"]
    scored = shots_with_xpts(shots, partial)
    assert "E2012_3" not in set(scored["game_id"])
    moved = shots_with_xpts(
        shots, xpts.assign(p_make=np.where(xpts["game_id"] == "E2011_0", 0.9, 0.5))
    )
    tg = team_games(moved, _ft(shots), (2011, 2012), (2013,)).set_index(["game_id", "team"])
    base = team_games(shots_with_xpts(shots, xpts), _ft(shots), (2011, 2012), (2013,))
    base = base.set_index(["game_id", "team"])
    changed = tg["xpts"] != base["xpts"]
    assert set(changed[changed].index.get_level_values(0)) == {"E2011_0"}


def test_shrinkage_pulls_small_samples_harder_and_intervals_cover() -> None:
    table = pd.DataFrame(
        {
            "shooter": ["A", "B"],
            "season": [2011, 2011],
            "fga": [100, 400],
            "raw": [10.0, 10.0],
            "variance": [100.0, 25.0],
        }
    )
    out = shrink(table, tau2=25.0)
    assert out.loc[0, "shrunk"] == pytest.approx(10.0 * 25 / 125)
    assert out.loc[1, "shrunk"] == pytest.approx(5.0)
    assert (out["low90"] < out["shrunk"]).all() and (out["high90"] > out["shrunk"]).all()
    none = shrink(table, tau2=0.0)
    assert (none["shrunk"] == 0.0).all()


def test_prior_variance_finds_real_skill_and_not_noise() -> None:
    noise = player_seasons(_shots(seed=2).assign(xpts=lambda s: 0.5 * s["value"]))
    skill = player_seasons(_shots(seed=2, skill=0.12).assign(xpts=lambda s: 0.5 * s["value"]))
    assert prior_variance(skill, SEASONS) > prior_variance(noise, SEASONS)


def test_stability_verdict_is_the_f_k_rule() -> None:
    assert stability_verdict(0.20, 0.30) == "stable enough to show"
    assert stability_verdict(0.19, 0.90) == "not stable enough"
    assert stability_verdict(0.50, 0.29) == "not stable enough"
    a = np.arange(50.0)
    ci = correlation_ci(a, a + np.random.default_rng(3).normal(0, 5, 50))
    assert ci["ci90"][0] <= ci["r"] <= ci["ci90"][1]


def test_player_report_verdict_matches_its_own_numbers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shot_making, "STABILITY_MIN_FGA", 100)
    shots = _shots(seed=4, skill=0.1)
    scored = shots.assign(points=shots["made"] * shots["value"], xpts=0.5 * shots["value"])
    tipoff = pd.Series(
        pd.to_datetime([f"{g.split('_')[0][1:]}-10-01" for g in scored["game_id"].unique()])
        + pd.to_timedelta([int(g.split("_")[1]) for g in scored["game_id"].unique()], unit="D"),
        index=scored["game_id"].unique(),
    )
    report = player_report(scored, tipoff, {"P0": "SHOOTER, ZERO"}, SEASONS, {2011: "development"})
    stability = report["stability"]
    expected = stability_verdict(
        stability["year_to_year"]["shrunk_shot_making"]["ci90"][0], stability["split_half"]["r"]
    )
    assert stability["verdict"] == expected
    top = report["players"][0]
    assert top["shooter"] == "P0" and top["name"] == "SHOOTER, ZERO"
    assert all(p["fga"] >= 100 for p in report["players"])


def test_chart_court_geometry_is_the_parse_shots_constants() -> None:
    """F8 Done-when: the drawing uses exactly the FIBA constants of parse/shots.py."""
    from eurohoops.eval.shot_charts import court_geometry  # noqa: PLC0415

    g = court_geometry()
    assert g["three_radius"] == court.THREE_RADIUS_M == 6.75
    assert g["corner_three_x"] == court.CORNER_THREE_M == 6.60
    assert g["corner_end_y"] == court.CORNER_END_Y_M
    assert g["baseline_y"] == court.BASELINE_Y_M == -1.575
    assert g["sideline_x"] == court.SIDELINE_X_M == 7.5
    assert g["key_half_width"] == court.KEY_HALF_WIDTH_M == 2.45
    assert g["free_throw_line_y"] == court.FREE_THROW_LINE_Y_M
    assert g["restricted_radius"] == court.RESTRICTED_AREA_RADIUS_M == 1.25
    # the corner segment meets the arc: sqrt(6.75^2 - 6.60^2) is where the arc starts
    assert np.hypot(g["corner_three_x"], g["corner_end_y"]) == pytest.approx(
        g["three_radius"], abs=0.01
    )


def test_charts_render(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from eurohoops.eval.shot_charts import residual_chart, xpts_surface  # noqa: PLC0415

    shots = _shots(n_games=4).assign(xpts=lambda s: 0.5 * s["value"])
    shots = shots.assign(points=shots["made"] * shots["value"])
    xpts_surface(shots, "surface", tmp_path / "a.png")
    residual_chart(shots, "residual", tmp_path / "b.png")
    assert (tmp_path / "a.png").stat().st_size > 10_000
    assert (tmp_path / "b.png").read_bytes()[:4] == b"\x89PNG"
