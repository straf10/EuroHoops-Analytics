"""M7 gate v2 (docs/models/m7.md, "Gate v2"): rolling-origin choice, season-cluster bootstrap and
the three pass criteria, on synthetic leagues only.

* Each origin's variant choice sees earlier seasons only: editing the rows (unit) or the games
  (integration) of the origin's season and later leaves its choice bit-identical, and a planted
  leak (the choice also reading the origin's own season) is caught by the very same helper.
* The bootstrap resamples whole seasons.
* The criteria are evaluated as declared.
"""

import json
from dataclasses import replace
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from eurohoops import cli
from eurohoops.config import GATE_V2_MARGIN, M7, M7_V2
from eurohoops.eval import m7_gate_v2
from eurohoops.eval.m7_backtest import (
    BASELINES,
    Z_LIMIT,
    M7Inputs,
    cluster_bootstrap_ci,
    spiegelhalter_z,
)
from eurohoops.eval.m7_gate_v2 import (
    chosen_rows,
    earlier_seasons,
    evaluate,
    format_gate_v2,
    gate_v2_block,
    origin_choices,
    run_gate_v2,
    season_cluster_ci,
)
from eurohoops.logs import write_json
from tests.m7_synthetic import build_inputs, small_spec, synthetic_formats

SEASONS = (2017, 2018, 2019, 2020, 2021)
SPEC = replace(
    small_spec(),
    tuning=SEASONS,
    validation=(),
    test=(),
    n_sims=50,
    bootstrap_resamples=400,
)
CANDIDATES = ["sim_full", "sim_net", "sim_inflate_1.5", "sim_inflate_2"]
Run = tuple[dict[str, Any], pd.DataFrame]


@cache
def _run(inputs_key: str = "base") -> Run:
    return run_gate_v2(_inputs(inputs_key), spec=SPEC, formats=synthetic_formats)


@cache
def _inputs(key: str) -> M7Inputs:
    inputs = build_inputs()
    if key == "base":
        return inputs
    season = int(key)  # every game of this season and later gets other scores
    games = inputs.games.copy()
    late = games["season"] >= season
    home, away = games["home_score"].copy(), games["away_score"].copy()
    new_home = away + 17 + ((away + 17) == home).astype("Int64")
    games.loc[late, "home_score"] = new_home[late]
    games.loc[late, "away_score"] = home[late]
    team_games = inputs.team_games.copy()
    rows = team_games["game_id"].isin(set(games.loc[late, "game_id"]))
    by_game = games.set_index("game_id")
    keys = team_games.loc[rows, "game_id"]
    home_points = by_game["home_score"].reindex(keys).to_numpy(dtype=np.int64)
    away_points = by_game["away_score"].reindex(keys).to_numpy(dtype=np.int64)
    team_games.loc[rows, "points"] = np.where(
        team_games.loc[rows, "home"], home_points, away_points
    )
    return replace(inputs, games=games, team_games=team_games)


# --- A hand-made frame of rows --------------------------------------------------------------------


def frame(
    rps: dict[tuple[str, int], float],
    *,
    models: tuple[str, ...] = tuple(CANDIDATES),
    per_season: int = 4,
) -> pd.DataFrame:
    """Rows of every model in the seasons of ``rps`` ((model, season) -> RPS; absent keys 0.5)."""
    out = []
    for season in sorted({s for _, s in rps} | set(SEASONS)):
        for model in models:
            for i in range(per_season):
                out.append(
                    {
                        "season": season,
                        "checkpoint": 0.25 * (1 + i % 3),
                        "team": f"T{i:02d}",
                        "model": model,
                        "rps": rps.get((model, season), 0.5),
                        "p_direct": 0.5,
                        "made_direct": i % 2,
                        "p_top10": np.nan,
                    }
                )
    return pd.DataFrame(out)


def _planted() -> pd.DataFrame:
    """sim_net is best in 2017, sim_inflate_2 in 2018 and 2019, sim_full from 2020 on."""
    rps = {(m, s): 0.5 for m in CANDIDATES for s in SEASONS}
    rps[("sim_net", 2017)] = 0.1
    rps[("sim_inflate_2", 2018)] = 0.05
    rps[("sim_inflate_2", 2019)] = 0.1
    rps[("sim_full", 2020)] = 0.0
    rps[("sim_full", 2021)] = 0.0
    return frame(rps)


def _choices(rows: pd.DataFrame) -> dict[int, str]:
    return {o.season: o.chosen for o in origin_choices(rows, SEASONS, CANDIDATES)}


# --- (i) each origin sees earlier seasons only ---------------------------------------------------


def test_the_first_season_is_a_selection_season_only() -> None:
    origins = origin_choices(_planted(), SEASONS, CANDIDATES)
    assert [o.season for o in origins] == [2018, 2019, 2020, 2021]
    assert origins[0].seen == (2017,)
    assert origins[3].seen == (2017, 2018, 2019, 2020)


def test_the_choice_is_the_lowest_pooled_rps_on_earlier_seasons() -> None:
    assert _choices(_planted()) == {
        2018: "sim_net",  # 2017 only
        2019: "sim_inflate_2",  # pooled over 2017-18: 0.275 against sim_net's 0.3
        2020: "sim_inflate_2",
        2021: "sim_inflate_2",  # sim_full's 0.0 in 2020 pools to 0.375, above inflate_2's 0.2875
    }


def test_the_choice_pools_rows_not_seasons() -> None:
    """Pooled over team-checkpoints: a season with more rows weighs more."""
    rows = frame({("sim_full", 2017): 0.2})
    extra = rows[(rows["season"] == 2018) & (rows["model"] == "sim_net")].copy()
    extra["team"] = [f"X{i}" for i in range(len(extra))]
    extra["rps"] = 0.1
    heavy = pd.concat([rows, extra, extra.assign(team=lambda d: d["team"] + "b")])
    # 2017-18 pooled: full (4*.2 + 4*.5)/8 = 0.35, net (4*.5 + 4*.5 + 8*.1)/16 = 0.3; by season
    # means it would be full 0.35 against net 0.367
    assert _choices(heavy)[2019] == "sim_net"


def test_an_exact_tie_goes_to_the_simpler_variant() -> None:
    assert set(_choices(frame({})).values()) == {"sim_full"}


def test_editing_the_origins_season_or_later_never_moves_its_choice() -> None:
    base = _planted()
    for origin in SEASONS[1:]:
        edited = base.copy()
        late = edited["season"] >= origin
        edited.loc[late, "rps"] = np.where(edited.loc[late, "model"] == "sim_inflate_1.5", 0.0, 0.9)
        before, after = _choices(base), _choices(edited)
        assert {s: c for s, c in after.items() if s <= origin} == {
            s: c for s, c in before.items() if s <= origin
        }
    # the helper can fail: the last origin's later seasons do matter to a later origin
    edited = base.copy()
    edited.loc[edited["season"] == 2018, "rps"] = np.where(
        edited.loc[edited["season"] == 2018, "model"] == "sim_inflate_1.5", 0.0, 0.9
    )
    assert _choices(edited)[2019] != _choices(base)[2019]


def test_a_planted_leak_into_the_origins_own_season_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _planted()
    edited = base.copy()
    mask = edited["season"] == 2019
    edited.loc[mask, "rps"] = np.where(edited.loc[mask, "model"] == "sim_inflate_1.5", 0.0, 9.0)
    assert _choices(base)[2019] == _choices(edited)[2019]  # clean
    monkeypatch.setattr(m7_gate_v2, "earlier_seasons", lambda seasons, origin: list(seasons))
    assert _choices(base)[2019] != _choices(edited)[2019]  # the leak shows up


def test_the_choice_never_reads_a_later_season_in_the_real_flow() -> None:
    """Rescoring every game of 2021 (a season only the last origin scores) moves no origin's
    pooled RPS table or choice, 2021's included (its choice sees up to 2020); rescoring 2020 and
    later leaves the origins up to 2020 alone."""
    base, _ = _run()
    for late, kept in (("2021", 2021), ("2020", 2020)):
        edited, _ = _run(late)
        same = [
            (a["season"], a["chosen"], a["pooled_rps"])
            for a in base["gate_v2"]["origins"]
            if a["season"] <= kept
        ]
        got = [
            (a["season"], a["chosen"], a["pooled_rps"])
            for a in edited["gate_v2"]["origins"]
            if a["season"] <= kept
        ]
        assert got == same


def test_a_planted_leak_in_the_real_flow_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    def leaky(seasons: Any, origin: int) -> list[int]:
        return [s for s in seasons if s <= origin]

    monkeypatch.setattr(m7_gate_v2, "earlier_seasons", leaky)
    base = run_gate_v2(_inputs("base"), spec=SPEC, formats=synthetic_formats)[0]
    edited = run_gate_v2(_inputs("2021"), spec=SPEC, formats=synthetic_formats)[0]
    leaked = [a["pooled_rps"] for a in base["gate_v2"]["origins"] if a["season"] == 2021]
    moved = [a["pooled_rps"] for a in edited["gate_v2"]["origins"] if a["season"] == 2021]
    assert leaked != moved


def test_earlier_seasons_is_strict() -> None:
    assert earlier_seasons(SEASONS, 2019) == [2017, 2018]
    assert earlier_seasons(SEASONS, 2017) == []


def test_a_variant_without_earlier_rows_raises() -> None:
    rows = _planted()
    rows = rows[~((rows["model"] == "sim_net") & (rows["season"] == 2017))]
    with pytest.raises(ValueError, match="no finite pooled RPS"):
        origin_choices(rows, SEASONS, CANDIDATES)


# --- (ii) the bootstrap resamples whole seasons --------------------------------------------------


def _diff_frames(per_season: int, diffs: dict[int, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rows whose per-row Brier difference is ``diffs[season]`` for every row of the season."""
    rows = []
    for season, d in diffs.items():
        for i in range(per_season):
            rows.append(
                {
                    "season": season,
                    "checkpoint": 0.25 * (1 + i % 3),
                    "team": f"T{i:02d}{i // 3}",
                    "made_direct": 1,
                    # (p - 1)^2 differences: mine 1 - (1 - d) ... set directly via p
                    "p_direct": 1.0 - np.sqrt(d) if d > 0 else 1.0,
                    "q": 0.0,
                }
            )
    mine = pd.DataFrame(rows)
    other = mine.assign(p_direct=1.0)
    return mine.drop(columns="q"), other.drop(columns="q")


def test_the_bootstrap_resamples_whole_seasons() -> None:
    """Two seasons of 10 rows with differences 0 and 1: a row bootstrap would give a CI inside
    (0.2, 0.8); resampling whole seasons gives means in {0, 0.5, 1} and the CI [0, 1]."""
    mine, other = _diff_frames(10, {2018: 0.0, 2019: 1.0})
    out = season_cluster_ci(mine, other, 2000, 7)
    assert out["clusters"] == 2
    assert out["mean"] == pytest.approx(0.5)
    assert out["ci95"] == pytest.approx([0.0, 1.0])
    assert (out["resamples"], out["seed"]) == (2000, 7)


def test_every_resample_mean_is_a_mix_of_whole_seasons() -> None:
    diff = np.array([0.0] * 6 + [1.0] * 2 + [3.0] * 4)
    cluster = np.array([0] * 6 + [1] * 2 + [2] * 4)
    # all-season-1 resample is the only way to reach exactly 1.0, all-season-2 exactly 3.0
    mean, low, high = cluster_bootstrap_ci(diff, cluster, 3000, 1)
    assert mean == pytest.approx(diff.mean())
    assert 0.0 <= low < high <= 3.0
    assert high == pytest.approx(3.0)  # P(all three draws = season 2) = 1/27 > 2.5%
    assert low == pytest.approx(0.0)


def test_the_season_cluster_ignores_team_and_checkpoint_clusters() -> None:
    """Seven seasons give seven clusters whatever the number of teams and checkpoints."""
    mine, other = _diff_frames(12, {s: 0.1 * i for i, s in enumerate(range(2018, 2025))})
    assert season_cluster_ci(mine, other, 100, 3)["clusters"] == 7


def test_misaligned_rows_are_refused() -> None:
    mine, other = _diff_frames(6, {2018: 0.0, 2019: 1.0})
    with pytest.raises(ValueError, match="not aligned"):
        season_cluster_ci(mine, other.iloc[::-1].reset_index(drop=True), 10, 1)


# --- (iii) the three criteria as declared ---------------------------------------------------------


def block(**kwargs: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "ci_high": 0.004,
        "brier_chosen": 0.1,
        "brier_standings_now": 0.2,
        "z_pooled": 0.5,
    } | kwargs
    return gate_v2_block(**values)


def test_all_three_criteria_pass() -> None:
    out = block()
    assert out["passed"] is True
    assert (out["a_non_inferior_to_point_sim"], out["b_beats_standings_now"]) == (True, True)
    assert out["c_calibrated"] is True


def test_the_margin_is_five_thousandths_and_strict() -> None:
    assert block(ci_high=0.0049999)["passed"] is True
    assert block(ci_high=0.005)["a_non_inferior_to_point_sim"] is False
    assert block(ci_high=0.005)["passed"] is False
    assert block(ci_high=0.02)["passed"] is False
    assert block()["margin"] == 0.005


def test_the_chosen_must_beat_standings_now_strictly() -> None:
    assert block(brier_chosen=0.2)["b_beats_standings_now"] is False
    assert block(brier_chosen=0.2001)["passed"] is False
    assert block(brier_chosen=0.1999)["passed"] is True


def test_calibration_is_two_sided_and_strict() -> None:
    assert block(z_pooled=1.959)["passed"] is True
    assert block(z_pooled=-1.959)["passed"] is True
    assert block(z_pooled=Z_LIMIT)["c_calibrated"] is False
    assert block(z_pooled=-Z_LIMIT)["passed"] is False
    assert block(z_pooled=None)["passed"] is False


def test_each_criterion_alone_fails_the_gate() -> None:
    for broken in ({"ci_high": 0.01}, {"brier_chosen": 0.3}, {"z_pooled": 2.5}):
        assert block(**broken)["passed"] is False


def test_evaluate_recomputes_the_criteria_from_the_rows() -> None:
    """The report's numbers follow from its CSV rows, evaluated as declared."""
    report, rows = _run()
    g = report["gate_v2"]
    scored = rows[rows["season"] != SEASONS[0]]
    mine = pd.concat(
        [
            scored[(scored["season"] == int(season)) & (scored["model"] == key)]
            for season, key in report["chosen_per_origin"].items()
        ]
    ).sort_values(["season", "checkpoint", "team"])
    point = scored[scored["model"] == "point_sim"].sort_values(["season", "checkpoint", "team"])
    table = scored[scored["model"] == "standings_now"].sort_values(["season", "checkpoint", "team"])
    assert len(mine) == g["n_team_checkpoints"] == len(point) == 4 * 3 * 12
    y = mine["made_direct"].to_numpy(dtype=float)
    p = mine["p_direct"].to_numpy()
    diff = (p - y) ** 2 - (point["p_direct"].to_numpy() - y) ** 2
    cluster = pd.factorize(mine["season"])[0]
    mean, low, high = cluster_bootstrap_ci(diff, cluster, 400, SPEC.bootstrap_seed)
    assert g["brier_vs_point_sim"]["mean"] == pytest.approx(mean, abs=2e-6)
    assert g["brier_vs_point_sim"]["ci95"] == pytest.approx([low, high], abs=2e-6)
    assert g["brier_chosen"] == pytest.approx(np.mean((p - y) ** 2), abs=2e-6)
    assert g["brier_standings_now"] == pytest.approx(
        np.mean((table["p_direct"].to_numpy() - y) ** 2), abs=2e-6
    )
    assert g["spiegelhalter_z_pooled"] == pytest.approx(spiegelhalter_z(p, y), abs=2e-6)
    c = g["criteria"]
    assert c["a_non_inferior_to_point_sim"] == (high < 0.005)
    assert c["b_beats_standings_now"] == (g["brier_chosen"] < g["brier_standings_now"])
    assert c["c_calibrated"] == (abs(g["spiegelhalter_z_pooled"]) < 1.96)
    assert g["passed"] == (
        c["a_non_inferior_to_point_sim"] and c["b_beats_standings_now"] and c["c_calibrated"]
    )


def test_evaluate_on_planted_rows_follows_the_declared_rule() -> None:
    """Chosen = point_sim exactly (difference 0, upper bound 0 < margin), a perfect table beaten
    by nobody: criterion (b) fails, so does the gate."""
    rows = _planted()
    keep = rows[rows["model"].isin(CANDIDATES)]
    extra = []
    for base in ("point_sim", "standings_now"):
        extra.append(keep[keep["model"] == "sim_full"].assign(model=base))
    both = pd.concat([keep, *extra], ignore_index=True)
    both.loc[both["model"] == "standings_now", "p_direct"] = both["made_direct"].astype(float)
    spec = replace(SPEC, bootstrap_resamples=50)
    _, out = evaluate(both, SEASONS, spec)
    assert out["brier_vs_point_sim"]["ci95"][1] < 0.005
    assert out["criteria"]["a_non_inferior_to_point_sim"] is True
    assert out["criteria"]["b_beats_standings_now"] is False
    assert out["passed"] is False


# --- The run ------------------------------------------------------------------------------------


def test_the_report_and_rows_of_a_run() -> None:
    report, rows = _run()
    g = report["gate_v2"]
    assert report["seasons"] == list(SEASONS)
    assert g["selection_seasons"] == [2017]
    assert g["scored_seasons"] == [2018, 2019, 2020, 2021]
    assert [o["season"] for o in g["origins"]] == [2018, 2019, 2020, 2021]
    assert all(o["chosen"] in CANDIDATES for o in g["origins"])
    assert all(o["seen_seasons"] == [s for s in SEASONS if s < o["season"]] for o in g["origins"])
    assert set(g["pooled"]) == {"chosen", *BASELINES, *CANDIDATES}
    assert {m for m in rows["model"]} == {*CANDIDATES, *BASELINES}
    assert set(rows.loc[rows["season"] == 2017, "split"]) == {"selection"}
    assert set(rows.loc[rows["season"] != 2017, "split"]) == {"scored"}
    assert report["tie_tolerance"] is None
    assert json.loads(json.dumps(report)) == report  # plain JSON


def test_the_run_is_deterministic() -> None:
    first, rows_first = _run()
    second, rows_second = run_gate_v2(_inputs("base"), spec=SPEC, formats=synthetic_formats)
    assert first == second
    pd.testing.assert_frame_equal(rows_first, rows_second)


def test_chosen_rows_are_each_origins_variant_in_its_season() -> None:
    _, rows = _run()
    origins = origin_choices(rows, SEASONS, CANDIDATES)
    mine = chosen_rows(rows, origins)
    for o in origins:
        got = mine[mine["season"] == o.season]
        assert len(got) == 3 * 12
    assert set(mine["season"]) == {2018, 2019, 2020, 2021}


def test_a_run_needs_a_selection_season_and_no_other_split() -> None:
    with pytest.raises(ValueError, match="selection season"):
        run_gate_v2(_inputs("base"), spec=replace(SPEC, tuning=(2018,)), formats=synthetic_formats)
    with pytest.raises(ValueError, match="selection season"):
        run_gate_v2(_inputs("base"), spec=replace(SPEC, test=(2021,)), formats=synthetic_formats)


def test_the_summary_prints_every_criterion() -> None:
    report, _ = _run()
    text = format_gate_v2(report)
    for needle in ("(a)", "(b)", "(c)", "verdict:", "origin 2018", "origin 2021"):
        assert needle in text


def test_the_real_config_is_the_declared_one() -> None:
    assert M7_V2.tuning == (2016, 2017, 2018, 2020, 2022, 2023, 2024, 2025)
    assert (M7_V2.validation, M7_V2.test) == ((), ())
    assert M7_V2.tuning == tuple(sorted({*M7.tuning, *M7.validation, *M7.test}))
    assert M7_V2.bootstrap_resamples == 10000
    assert (M7_V2.checkpoints, M7_V2.n_sims, M7_V2.seed) == (
        M7.checkpoints,
        M7.n_sims,
        M7.seed,
    )
    assert (M7_V2.inflate, M7_V2.noise_floor) == (M7.inflate, M7.noise_floor)
    assert GATE_V2_MARGIN == 0.005


def test_the_cli_refuses_a_gbl_gate_v2_run_and_other_models() -> None:
    runner = CliRunner()
    gbl = runner.invoke(cli.app, ["backtest", "--model", "m7", "--competition", "gbl", "--gate-v2"])
    other = runner.invoke(cli.app, ["backtest", "--model", "m1", "--gate-v2"])
    assert (gbl.exit_code, other.exit_code) == (1, 1)


def test_the_cli_writes_the_report_and_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report, rows = _run()
    spec = replace(cli.M7_V2, report=tmp_path / "r.json", teams_report=tmp_path / "t.csv")
    monkeypatch.setattr(cli, "M7_V2", spec)
    monkeypatch.setattr(cli, "_m7_inputs", lambda *_: _inputs("base"))
    monkeypatch.setattr(cli, "run_gate_v2", lambda *_, **__: (report, rows))
    result = CliRunner().invoke(cli.app, ["backtest", "--model", "m7", "--gate-v2"])
    assert result.exit_code == 0, result.output
    assert json.loads(spec.report.read_text(encoding="utf-8")) == report
    assert len(pd.read_csv(spec.teams_report)) == len(rows)
    assert "verdict:" in result.output
    write_json(tmp_path / "x.json", report)  # the report is writable as is
