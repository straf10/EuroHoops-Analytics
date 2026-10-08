"""Live M6 (weeks 16-18 L9): the live block equals the harness's test-target projections, the live
season's own games change nothing at checkpoint 0, newcomers, the frozen interval scale, the
undervalued rule, the report shapes the read model checks, and no age or birth date anywhere."""

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import special

from eurohoops.api.readmodel import (
    BOARD_ROWS_SCHEMA,
    PROJECTION_ROWS_SCHEMA,
    SIMILAR_ROWS_SCHEMA,
    UNDERVALUED_ROWS_SCHEMA,
)
from eurohoops.config import M6_BOARD, M6_PROJECTIONS, M6_SIMILARITY, SIM_UNGATED
from eurohoops.eval.m6_backtest import (
    M6Inputs,
    World,
    prepare,
    project_cell,
    scale_intervals,
    target_inputs,
)
from eurohoops.live_m6 import (
    UNDERVALUED_MAX_MPG,
    UNDERVALUED_MAX_SEASONS,
    inputs_digest,
    live_checkpoint,
    live_people,
    live_projections,
    person_names,
    personal_fields,
    player_rows,
    projections_report,
    query_rows,
    translate_el,
    undervalued,
)
from eurohoops.models.player_seasons import COUNT_STATS, PROJECTED_STATS
from eurohoops.models.projection import Translation
from eurohoops.parse.schemas import validated
from tests.m6_synthetic import build_inputs, small_spec, synthetic_rounds

SPEC = small_spec()
LIVE = SPEC.test[-1]  # the synthetic world's last season plays the live one
CHOSEN = {"variant": "proj_shrunk", "half_life": 1.0}
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def inputs() -> M6Inputs:
    return build_inputs()


@pytest.fixture(scope="module")
def world(inputs: M6Inputs) -> World:
    return prepare(inputs)


def _live(inputs: M6Inputs, world: World, people: list[str], scale: dict[str, float]) -> Any:
    rows, ti = live_projections(
        inputs,
        "euroleague",
        LIVE,
        0.0,
        people,
        chosen=CHOSEN,
        scale=scale,
        spec=SPEC,
        translation=inputs.translations.get(LIVE),
        rounds=synthetic_rounds("euroleague", LIVE),
        world=world,
    )
    return rows, ti


# --- The checkpoint -------------------------------------------------------------------------------


def _regular(played_rounds: int, rounds: int = 20, partial_next: bool = False) -> pd.DataFrame:
    rows = []
    for r in range(1, rounds + 1):
        for g in range(2):
            played = r <= played_rounds or (partial_next and r == played_rounds + 1 and g == 0)
            rows.append({"round": r, "played": played})
    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    ("played", "expected"),
    [(0, 0.0), (4, 0.0), (5, 0.25), (9, 0.25), (10, 0.5), (14, 0.5), (15, 0.75), (20, 0.75)],
)
def test_the_live_checkpoint_is_the_largest_declared_one_whose_rounds_are_complete(
    played: int, expected: float
) -> None:
    # R = 20: floor(0.25 R) = 5, floor(0.5 R) = 10, floor(0.75 R) = 15 completed rounds
    assert live_checkpoint(_regular(played), 20, (0.25, 0.5, 0.75)) == expected


def test_a_round_with_an_unplayed_game_is_not_complete() -> None:
    assert live_checkpoint(_regular(4, partial_next=True), 20, (0.25, 0.5, 0.75)) == 0.0


# --- The live block -------------------------------------------------------------------------------


def test_the_live_block_projects_exactly_what_the_harness_projects_for_a_test_target(
    inputs: M6Inputs, world: World
) -> None:
    scored = target_inputs(
        inputs, "euroleague", LIVE, 0.0, spec=SPEC, rounds=synthetic_rounds, world=world
    )
    people = list(scored.targets["person_id"])
    harness = project_cell(scored, CHOSEN["variant"], CHOSEN["half_life"])
    live, _ = _live(inputs, world, people, {})
    keys = ["person_id", "stat"]
    pd.testing.assert_frame_equal(
        live.sort_values(keys).reset_index(drop=True)[["person_id", "stat", "mean", "sd"]],
        harness.sort_values(keys).reset_index(drop=True)[["person_id", "stat", "mean", "sd"]],
    )


def test_the_live_seasons_own_games_change_nothing_at_checkpoint_0(inputs: M6Inputs) -> None:
    people = live_people(prepare(inputs).complete, "euroleague", LIVE)
    before, _ = _live(inputs, prepare(inputs), people, {})
    lines = inputs.player_games["euroleague"]
    doped = lines.assign(pts=np.where(lines["season"] == LIVE, lines["pts"] * 3, lines["pts"]))
    changed = replace(inputs, player_games={**inputs.player_games, "euroleague": doped})
    after, _ = _live(changed, prepare(changed), people, {})
    pd.testing.assert_frame_equal(before, after)


def test_a_newcomer_takes_the_floor_exposure_and_the_no_history_flag(
    inputs: M6Inputs, world: World
) -> None:
    people = [*live_people(world.complete, "euroleague", LIVE)[:3], "P:NEWCOMER"]
    rows, ti = _live(inputs, world, people, {})
    target = ti.targets.set_index("person_id")
    assert target.loc["P:NEWCOMER", "exposure"] == SPEC.min_poss
    new = rows[rows["person_id"] == "P:NEWCOMER"]
    assert (new["flags"].str.contains("no_history")).all()
    assert (new["n_seasons"] == 0).all()


def test_the_frozen_scale_multiplies_the_sd_by_its_square_root(
    inputs: M6Inputs, world: World
) -> None:
    people = live_people(world.complete, "euroleague", LIVE)[:5]
    raw, _ = _live(inputs, world, people, {})
    scaled, _ = _live(inputs, world, people, {"pts": 4.0})
    pts, other = raw["stat"] == "pts", raw["stat"] == "ast"
    np.testing.assert_allclose(scaled.loc[pts, "sd"], 2.0 * raw.loc[pts, "sd"])
    np.testing.assert_allclose(scaled.loc[other, "sd"], raw.loc[other, "sd"])
    np.testing.assert_allclose(scaled["mean"], raw["mean"])
    z = special.ndtri(0.9)
    expect = np.maximum(scaled.loc[pts, "mean"] - z * scaled.loc[pts, "sd"], 0.0)
    np.testing.assert_allclose(scaled.loc[pts, "lo80"], expect)


def test_scale_intervals_keeps_negative_impact_bounds() -> None:
    rows = pd.DataFrame({"stat": ["spm", "pts"], "mean": [-1.0, 1.0], "sd": [1.0, 1.0]})
    out = scale_intervals(rows, {}, 0.8)
    assert out.loc[0, "lo80"] < -1.0  # an impact stat is not clipped at 0
    assert out.loc[1, "lo80"] == 0.0


def test_the_digest_moves_with_the_history_only(inputs: M6Inputs, world: World) -> None:
    people = live_people(world.complete, "euroleague", LIVE)[:5]
    _, a = _live(inputs, world, people, {})
    _, b = _live(inputs, world, people, {})
    assert inputs_digest([a]) == inputs_digest([b])
    history = a.history.assign(pts=a.history["pts"] + 1)
    assert inputs_digest([replace(a, history=history)]) != inputs_digest([a])


# --- Report rows ----------------------------------------------------------------------------------


def test_player_rows_match_the_read_models_schema(inputs: M6Inputs, world: World) -> None:
    people = live_people(world.complete, "euroleague", LIVE)
    rows, _ = _live(inputs, world, people, {})
    live = world.complete[
        (world.complete["season"] == LIVE) & (world.complete["competition"] == "euroleague")
    ]
    out = player_rows(rows, live, {people[0]: "A Name"})
    assert [r["person_id"] for r in out] == sorted(people)
    validated(pd.DataFrame(out), PROJECTION_ROWS_SCHEMA)
    first = out[0]
    assert first["name"] == "A Name"
    assert set(first["stats"]) <= set(PROJECTED_STATS)
    for stat in first["stats"].values():
        assert stat["lo80"] <= stat["mean"] <= stat["hi80"]
    report = projections_report(
        chosen=CHOSEN,
        gate_passed=True,
        season=LIVE,
        checkpoints={"euroleague": 0.0, "gbl": 0.25},
        digest="x",
        players=out,
        undervalued_rows=[],
    )
    assert report["checkpoint"] == 0.0
    assert report["stats"] == list(PROJECTED_STATS)


def test_query_rows_take_each_persons_newest_qualifying_complete_season(world: World) -> None:
    people = live_people(world.complete, "euroleague", LIVE)
    query = query_rows(world.complete, people, LIVE, SPEC.min_poss)
    assert query["person_id"].is_unique
    assert (query["season"] < LIVE).all() and (query["poss"] >= SPEC.min_poss).all()
    done = world.complete[(world.complete["season"] < LIVE) & (world.complete["poss"] >= 500)]
    newest = done.groupby("person_id")["season"].max()
    assert (query.set_index("person_id")["season"] == newest.loc[query["person_id"]]).all()


def test_person_names_prefer_the_euroleague_spelling() -> None:
    xwalk = pd.DataFrame(
        {"competition": ["gbl", "euroleague"], "source_id": ["G1", "E1"], "person_id": ["P:E1"] * 2}
    )
    el = pd.DataFrame({"player_id": ["E1"], "player": ["DOE, JOHN"]})
    gbl = pd.DataFrame(
        {
            "source_id": ["G1", "G2"],
            "season": [2024, 2025],
            "surname_raw": ["DOE", "ROE"],
            "first_raw": ["JON", "RICHARD"],
        }
    )
    assert person_names(el, gbl, xwalk) == {"P:E1": "John Doe", "G:G2": "Richard Roe"}


# --- The undervalued list -------------------------------------------------------------------------


def test_translate_el_is_m4s_formula_floored_at_0() -> None:
    t = Translation(
        delta={"pts": math.log(2.0), "tov": -5.0}, c={"pts": 1.0, "tov": 3.0}, target_season=1
    )
    out = translate_el(pd.DataFrame({"pts": [9.0], "tov": [1.0]}), t)
    assert out.loc[0, "pts"] == pytest.approx(19.0)  # (9 + 1) * 2 - 1
    assert out.loc[0, "tov"] == 0.0  # (1 + 3) * exp(-5) - 3 < 0


def _gbl_case() -> tuple[pd.DataFrame, pd.DataFrame, Translation]:
    """Five GBL players, newest complete season 2025: a (young, low minutes, high), b (young,
    high minutes), c (old), d (young, low minutes, low), e (rotation reference only)."""
    people = ["G:a", "G:b", "G:c", "G:d", "G:e"]
    level = {"G:a": 30.0, "G:b": 30.0, "G:c": 30.0, "G:d": 5.0, "G:e": 10.0}
    complete = pd.DataFrame(
        {
            "person_id": people,
            "competition": "gbl",
            "season": 2025,
            "team": "T",
            "games": 20,
            "minutes": [15.0 * 20, 30.0 * 20, 15.0 * 20, 15.0 * 20, 30.0 * 20],
            "poss": 600.0,
            "fg2a": 60,
            "pf": 18,
            "debut_season": [2024, 2023, 2015, 2024, 2020],
        }
    )
    projections = pd.DataFrame(
        [
            {"person_id": p, "stat": s, "mean": level[p] if s == "pts" else 1.0}
            for p in people
            for s in COUNT_STATS
        ]
    )
    stats = (*COUNT_STATS, "fg2a", "pf")
    identity = Translation(
        delta=dict.fromkeys(stats, 0.0), c=dict.fromkeys(stats, 0.0), target_season=2025
    )
    return complete, projections, identity


def test_the_undervalued_rule_needs_young_low_minutes_and_the_top_quarter() -> None:
    complete, projections, identity = _gbl_case()

    def spm(frame: pd.DataFrame, before: int) -> pd.Series:
        assert before == 2026 and (frame["competition"] == "euroleague").all()
        return frame["pts"] / 10.0  # 3.0 for a, b, c; 0.5 for d; 1.0 for e

    rows = undervalued(
        projections,
        complete,
        2026,
        translation=identity,
        spm=spm,
        names={"G:a": "A"},
        teams={},
        min_poss=500.0,
    )
    validated(pd.DataFrame(rows), UNDERVALUED_ROWS_SCHEMA)
    # the top-quarter cut of (3, 3, 3, 0.5, 1) is 3: only a is young, below the minutes and in it
    assert [r["person_id"] for r in rows] == ["G:a"]
    a = rows[0]
    assert a["name"] == "A" and a["seasons_since_debut"] == 2
    assert a["seasons_since_debut"] <= UNDERVALUED_MAX_SEASONS
    assert a["minutes"] == 15.0 < UNDERVALUED_MAX_MPG
    assert a["projected_spm"] == 3.0 and a["projected_brapm"] is None


# --- Privacy and the committed reports ------------------------------------------------------------


def test_the_scan_finds_a_planted_age_and_birth_date() -> None:
    planted = {"players": [{"person_id": "x", "age": 24}, {"stats": {"birth_date": "2001-01-01"}}]}
    assert len(personal_fields(planted)) == 3  # the age key, the birth key and its date value
    # a time stamp is a tip-off or a cutoff, and "usage" is not an age
    assert personal_fields({"cutoff_utc": "2026-10-07T18:45:00Z", "f": ["usage"]}) == []
    assert personal_fields({"note": ["1999-04-02"]}) == ["$.note[0] = 1999-04-02"]


@pytest.mark.parametrize("path", [M6_PROJECTIONS, M6_BOARD, M6_SIMILARITY])
def test_the_committed_m6_reports_hold_no_age_and_no_birth_date(path: Path) -> None:
    assert personal_fields(json.loads((ROOT / path).read_text(encoding="utf-8"))) == []


@pytest.mark.parametrize("name", ["euroleague", "gbl"])
def test_the_ungated_reports_are_labelled_not_gated(name: str) -> None:
    report = json.loads((ROOT / SIM_UNGATED[name]).read_text(encoding="utf-8"))
    assert report["gated"] is False
    assert report["gate"]["passed"] is False and "not gated" in report["gate"]["reason"]
    assert report["simulated_at_utc"] == report["cutoff_utc"]  # deterministic: no wall clock


def test_the_committed_report_rows_match_the_read_models_schemas() -> None:
    projections = json.loads((ROOT / M6_PROJECTIONS).read_text(encoding="utf-8"))
    validated(pd.DataFrame(projections["players"]), PROJECTION_ROWS_SCHEMA)
    if projections["undervalued"]:
        validated(pd.DataFrame(projections["undervalued"]), UNDERVALUED_ROWS_SCHEMA)
    board = json.loads((ROOT / M6_BOARD).read_text(encoding="utf-8"))
    validated(pd.DataFrame(board["rows"]), BOARD_ROWS_SCHEMA)
    similar = json.loads((ROOT / M6_SIMILARITY).read_text(encoding="utf-8"))
    for rows in similar["players"].values():
        validated(pd.DataFrame(rows), SIMILAR_ROWS_SCHEMA)
