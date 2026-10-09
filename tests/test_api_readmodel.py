"""Read model of the API on a small repository tree (``tests/api_tree.py``).

Expected values are hand-read from the raw box scores and the committed stats fixtures (named in
each test), or computed by the pipeline's own public function that the read model delegates to
(then the test shows the delegation, not a re-implementation). Nothing here fits a model, so there
is no walk-forward state to leak; the read-model functions only read.
"""

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pandera.errors
import pytest

from eurohoops.api import readmodel as rm
from eurohoops.config import EUROLEAGUE, GBL, M6_PROJECTIONS, M7, MART_PATH
from eurohoops.eval.backtest import load_tuned_model
from eurohoops.marts import read_games
from eurohoops.models.elo import prepare, season_ratings
from eurohoops.stats.box import build_box_games
from eurohoops.stats.export import PLAYER_FIELDS, load_cached_games
from tests.api_tree import EL_PERSON, GBL_ONLY_PERSON, NOW, build_tree
from tests.conftest import FIXTURES


@pytest.fixture(scope="module")
def root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tree = tmp_path_factory.mktemp("api") / "tree"
    tree.mkdir()
    build_tree(tree)
    return tree


@pytest.fixture(scope="module")
def store(root: Path) -> rm.Store:
    return rm.Store(root)


@pytest.fixture
def fresh(tmp_path: Path) -> Path:
    """A tree a test may damage."""
    build_tree(tmp_path)
    return tmp_path


def test_jsonable_unwraps_numpy_and_pandas_scalars() -> None:
    value = {
        "a": np.float64("nan"),
        "b": np.int64(3),
        "c": np.bool_(True),
        "d": [pd.Timestamp("2026-10-01T18:00:00Z"), pd.NaT, float("inf")],
        "e": np.array([1.5, np.nan]),
        "f": (1, "x"),
    }
    out = rm.jsonable(value)
    assert out == {
        "a": None,
        "b": 3,
        "c": True,
        "d": ["2026-10-01T18:00:00Z", None, None],
        "e": [1.5, None],
        "f": [1, "x"],
    }
    assert type(out["b"]) is int and type(out["c"]) is bool
    json.dumps(out, allow_nan=False)


def test_a_store_is_a_snapshot_that_loads_each_thing_once(root: Path) -> None:
    fresh_store = rm.Store(root)
    calls: list[int] = []
    assert fresh_store.memo("k", lambda: calls.append(1) or 7) == 7
    assert fresh_store.memo("k", lambda: calls.append(1) or 8) == 7
    assert calls == [1]
    assert rm.Store(root) == fresh_store  # equality ignores the memo


def test_site_is_the_publish_payload(store: rm.Store) -> None:
    payload = rm.site(store, NOW)
    assert payload["generated_at_utc"] == "2026-10-08T08:00:00Z"
    (el,) = payload["competitions"]  # the GBL is not on the site
    assert el["key"] == "euroleague"
    assert el["season"] == "2026-27"
    # E2026_2 is logged twice and counted once; the round-2 rows tip off after now.
    assert el["logged"] == 5
    assert [g["game_id"] for g in el["upcoming"]] == ["E2026_4", "E2026_6"]
    assert el["upcoming"][0]["away"] == {"code": "FBT", "name": "Fenerbahce Beko Istanbul"}
    assert len(el["ratings"]) == 6


def test_site_without_its_inputs_is_not_found(fresh: Path) -> None:
    (fresh / "reports" / "live_scorecard.json").unlink()
    with pytest.raises(rm.NotFound, match=r"live_scorecard\.json not found"):
        rm.site(rm.Store(fresh), NOW)


def test_stats_come_from_the_marts_and_from_the_cache_identically(
    store: rm.Store, root: Path
) -> None:
    cached = rm.Store(root, from_cache=True)
    stats_now = datetime(2026, 9, 26, 8, tzinfo=UTC)
    assert rm.stats_payloads(cached, stats_now) == rm.stats_payloads(store, stats_now)
    index = rm.stats_index(store, stats_now)["files"]
    committed = sorted(
        p.relative_to(FIXTURES / "stats").as_posix() for p in (FIXTURES / "stats").rglob("*.json")
    )
    assert index == committed  # the file list of the committed export of the same raw slice
    meta = rm.stats_file(store, "meta.json", stats_now)
    assert [s["season"] for s in meta["seasons"]] == [2010, 2024]
    assert meta["generated_at"] == "2026-09-26T08:00Z"  # the build time, stamped once per Store


def test_stats_build_stamps_the_first_now_and_reuses_it(root: Path) -> None:
    local = rm.Store(root, from_cache=True)
    first = rm.stats_file(local, "meta.json", NOW)["generated_at"]
    later = rm.stats_file(local, "meta.json", datetime(2027, 1, 1, tzinfo=UTC))["generated_at"]
    assert first == later == "2026-10-08T08:00Z"


def test_unknown_stats_file_names_the_listing(store: rm.Store) -> None:
    with pytest.raises(
        rm.NotFound, match=r"no stats file 'seasons/1999/players\.json'; GET /stats"
    ):
        rm.stats_file(store, "seasons/1999/players.json", NOW)


def test_stats_from_the_cache_need_no_marts(root: Path, tmp_path: Path) -> None:
    shutil.copytree(root / "data" / "raw", tmp_path / "data" / "raw")
    local = rm.Store(tmp_path, from_cache=True)
    assert "seasons/2024/teams.json" in rm.stats_index(local, NOW)["files"]
    with pytest.raises(rm.NotFound, match="the marts are not built"):
        rm.upcoming_games(local, "euroleague", NOW)


def test_upcoming_games_hold_the_pre_registered_forecast(store: rm.Store) -> None:
    out = rm.upcoming_games(store, "euroleague", NOW)
    assert (out["competition"], out["season"], out["as_of"]) == (
        "euroleague",
        2026,
        "2026-10-08T08:00:00Z",
    )
    games = out["games"]
    # make_games: round r tips off 7 days apart from 2026-10-01 18:00, three games 15 minutes apart
    assert [g["game_id"] for g in games[:3]] == ["E2026_4", "E2026_5", "E2026_6"]
    assert len(games) == 27 and all(g["tipoff_utc"] > "2026-10-08T08:00:00Z" for g in games)
    first = games[0]
    assert first["tipoff_utc"] == "2026-10-08T18:00:00Z"
    assert first["away"] == {
        "code": "ULK",
        "display_code": "FBT",
        "name": "Fenerbahce Beko Istanbul",
    }
    assert (first["prediction"]["p_home"], first["prediction"]["exp_margin"]) == (0.61, 3.5)
    assert games[1]["prediction"] is None  # E2026_5 is not logged
    assert games[2]["home"]["code"] == "MAD" and games[2]["prediction"]["p_home"] == 0.4


def test_upcoming_games_of_the_other_competition_and_unknown_ones(store: rm.Store) -> None:
    gbl = rm.upcoming_games(store, "gbl", NOW)["games"]
    assert gbl[0]["game_id"] == "GBL2026_4" and gbl[0]["prediction"]["p_home"] == 0.52
    with pytest.raises(rm.NotFound, match="unknown competition 'nba'; expected one of"):
        rm.upcoming_games(store, "nba", NOW)


def test_a_log_row_outside_its_schema_is_an_error_not_a_404(fresh: Path) -> None:
    log = fresh / EUROLEAGUE.prediction_log
    log.write_text(log.read_text(encoding="utf-8").replace("0.6100", "1.6100"), encoding="utf-8")
    with pytest.raises(pandera.errors.SchemaError):
        rm.upcoming_games(rm.Store(fresh), "euroleague", NOW)


def test_prediction_lists_the_pre_registered_row_and_the_later_ones(store: rm.Store) -> None:
    out = rm.prediction(store, "E2026_2")
    assert (out["competition"], out["home"], out["away"], out["round"]) == (
        "euroleague",
        "BAR",
        "OLY",
        1,
    )
    assert out["result"] == {"home_score": 90, "away_score": 80, "forfeit": False}
    elo = out["models"]["elo"]
    assert elo["pre_registered"]["p_home"] == 0.5 and elo["pre_registered"]["late"] is False
    assert [r["p_home"] for r in elo["later"]] == [0.58]
    assert set(out["models"]) == {"elo"}


def test_prediction_marks_a_row_stamped_after_tip_off_as_late(store: rm.Store) -> None:
    only = rm.prediction(store, "E2026_3")["models"]["elo"]
    assert only["pre_registered"]["late"] is True and only["later"] == []


def test_prediction_gathers_every_model_and_an_unplayed_game_has_no_result(
    store: rm.Store,
) -> None:
    out = rm.prediction(store, "E2026_4")
    assert out["result"] is None
    assert set(out["models"]) == {"elo", "m1"}
    m1 = out["models"]["m1"]["pre_registered"]
    assert (m1["p_home"], m1["exp_total"], m1["margin_df"]) == (0.63, 160.5, 7)
    assert rm.prediction(store, "GBL2026_4")["competition"] == "gbl"


def test_prediction_of_an_unlogged_game_is_not_found(store: rm.Store) -> None:
    with pytest.raises(rm.NotFound, match="no prediction is logged for game 'E2026_5'"):
        rm.prediction(store, "E2026_5")


def _elo_table(root: Path, code: str, season: int) -> dict[str, tuple[float, float]]:
    comp = {"euroleague": EUROLEAGUE, "gbl": GBL}[code]
    games = read_games(root / MART_PATH, comp.name)
    params = load_tuned_model(root / comp.live_backtest.report).params
    window = games[games["season"].between(comp.live_backtest.warmup[0], season)]
    return season_ratings(prepare(window), params, season)


def test_team_ratings_are_the_published_elo(store: rm.Store, root: Path) -> None:
    out = rm.team_ratings(store, "euroleague", "BAR")
    published = {r["code"]: r for r in rm.site(store, NOW)["competitions"][0]["ratings"]}["BAR"]
    assert out["elo"]["rating"] == published["rating"]
    assert out["elo"]["change"] == published["change"]
    now, start = _elo_table(root, "euroleague", 2026)["BAR"]
    assert out["elo"]["rating"] == round(now, 1) and out["elo"]["change"] == round(now - start, 1)
    # BAR won its three round-1 games and the other five teams won none
    assert (out["elo"]["rank"], out["elo"]["teams"]) == (1, 6)
    assert out["team"] == {"code": "BAR", "display_code": "BAR", "name": "FC Barcelona"}
    assert out["m1"]["available"] is False
    assert rm.team_ratings(store, "euroleague", "ULK")["team"]["display_code"] == "FBT"


def test_team_ratings_unknown_team_and_a_team_without_a_live_game(store: rm.Store) -> None:
    with pytest.raises(rm.NotFound, match="unknown team 'XXX' in euroleague"):
        rm.team_ratings(store, "euroleague", "XXX")
    # ASV played in 2024 only: known to the competition, no game in the live season
    with pytest.raises(rm.NotFound, match="'ASV' has no game in the euroleague 2026 season"):
        rm.team_ratings(store, "euroleague", "ASV")


def test_four_factors_of_a_euroleague_season_from_the_box_totals(store: rm.Store) -> None:
    out = rm.team_factors(store, "euroleague", "LJU", 2010)
    # Union Olimpija's only 2010-11 line in the fixtures (boxscore/E2010/12.json.gz, totr):
    # FG2 26/48, FG3 8/30, FTA 25, OREB 16, TOV 14; opponent IST DREB 27.
    # possessions = FGA - OREB + TOV + 0.42 FTA = 78 - 16 + 14 + 10.5
    f = out["factors"]
    assert f["efg_pct"]["value"] == pytest.approx((26 + 1.5 * 8) / 78)
    assert f["tov_pct"]["value"] == pytest.approx(14 / 86.5)
    assert f["orb_pct"]["value"] == pytest.approx(16 / (16 + 27))
    assert f["ft_rate"]["value"] == pytest.approx(25 / 78)
    assert out["games"] == 1 and out["available_seasons"] == [2010]
    assert all(v["reason"] is None for v in f.values())


def test_a_season_sums_games_before_the_ratios(store: rm.Store) -> None:
    games, _ = load_cached_games(FIXTURES / "stats_raw")
    teams = build_box_games(FIXTURES / "stats_raw", games).teams
    mine = teams[(teams["team"] == "BER") & (teams["season"] == 2024)]
    against = teams[(teams["opponent"] == "BER") & (teams["season"] == 2024)]
    out = rm.team_factors(store, "euroleague", "BER", 2024)
    assert out["games"] == len(mine) == len(against) == 6  # rounds 1-6
    fga = mine["fg2a"].sum() + mine["fg3a"].sum()
    f = out["factors"]
    assert f["efg_pct"]["value"] == pytest.approx(
        (mine["fg2m"].sum() + 1.5 * mine["fg3m"].sum()) / fga
    )
    assert f["tov_pct"]["value"] == pytest.approx(mine["tov"].sum() / mine["poss"].sum())
    assert f["orb_pct"]["value"] == pytest.approx(
        mine["oreb"].sum() / (mine["oreb"].sum() + against["dreb"].sum())
    )
    assert f["ft_rate"]["value"] == pytest.approx(mine["fta"].sum() / fga)
    per_game = (mine["fta"] / (mine["fg2a"] + mine["fg3a"])).mean()
    assert abs(f["ft_rate"]["value"] - per_game) > 1e-4  # the sum of games, not the mean of ratios


def test_schedule_strength_is_the_mean_opponent_rating(store: rm.Store, root: Path) -> None:
    out = rm.team_factors(store, "euroleague", "BAR", 2026)
    games = read_games(root / MART_PATH, "euroleague")
    mine = games[(games["season"] == 2026) & ((games["home"] == "BAR") | (games["away"] == "BAR"))]
    opponents = [a if h == "BAR" else h for h, a in zip(mine["home"], mine["away"], strict=True)]
    table = _elo_table(root, "euroleague", 2026)
    strength = out["schedule_strength"]
    assert strength["games"] == len(opponents) == 10
    assert strength["value"] == pytest.approx(np.mean([table[o][0] for o in opponents]))
    # no 2026 box scores exist: with no season given, the latest box season (2024) is used
    latest = rm.team_factors(store, "euroleague", "BAR", None)
    assert latest["season"] == 2024 and latest["available_seasons"] == [2010, 2024]
    assert latest["schedule_strength"]["games"] == 6


def test_a_missing_input_gives_a_null_factor_and_its_reason(store: rm.Store) -> None:
    out = rm.team_factors(store, "euroleague", "BAR", 2026)
    assert out["season"] == 2026 and out["games"] == 0
    for factor in out["factors"].values():
        assert factor == {"value": None, "reason": "no box-score games for 'BAR' in 2026"}
    assert out["schedule_strength"]["value"] is not None
    early = rm.team_factors(store, "euroleague", "LJU", 2010)["schedule_strength"]
    assert early["value"] is None
    assert early["reason"] == "the euroleague Elo is replayed from 2023; no rating for 2010"


def test_gbl_factors_take_counts_from_team_games_and_makes_from_the_player_lines(
    store: rm.Store, root: Path
) -> None:
    out = rm.team_factors(store, "gbl", "AAAA0001", 2025)
    # box_8FC479F6.html, first team, hand-read: totals row FGA 51 + 23, FTA 27, OREB 18, TOV 18;
    # player lines sum to FG2M 24 and FG3M 7 (24 * 2 + 7 * 3 + 24 FTM = 93 points);
    # the opponent's DREB is 25; possessions 74 - 18 + 18 + 0.42 * 27
    f = out["factors"]
    assert f["efg_pct"]["value"] == pytest.approx((24 + 1.5 * 7) / 74)
    assert f["tov_pct"]["value"] == pytest.approx(18 / 85.34)
    assert f["orb_pct"]["value"] == pytest.approx(18 / (18 + 25))
    assert f["ft_rate"]["value"] == pytest.approx(27 / 74)
    # its only 2025 game is against AAAA0002, which won it: above the 1500 start
    assert out["schedule_strength"]["games"] == 1
    assert out["schedule_strength"]["value"] == pytest.approx(
        _elo_table(root, "gbl", 2025)["AAAA0002"][0]
    )
    assert out["schedule_strength"]["value"] > 1500.0
    other = rm.team_factors(store, "gbl", "AAA", 2025)
    assert other["factors"]["efg_pct"]["reason"] == "no box-score games for 'AAA' in 2025"


def test_gbl_factors_without_cached_box_pages_say_so(fresh: Path) -> None:
    shutil.rmtree(fresh / GBL.raw_dir)
    out = rm.team_factors(rm.Store(fresh), "gbl", "AAAA0001", 2025)
    assert out["factors"]["efg_pct"] == {
        "value": None,
        "reason": "no gbl box-score lines are cached",
    }
    assert out["available_seasons"] == []
    assert out["schedule_strength"]["value"] is not None


def test_factors_of_an_unknown_team_or_competition(store: rm.Store) -> None:
    with pytest.raises(rm.NotFound, match="unknown team 'XXX'"):
        rm.team_factors(store, "gbl", "XXX", None)
    with pytest.raises(rm.NotFound, match="unknown competition"):
        rm.team_factors(store, "acb", "BAR", None)


def test_a_player_has_his_euroleague_seasons_and_the_impact_reports(store: rm.Store) -> None:
    out = rm.player(store, EL_PERSON)
    assert (out["name"], out["debut_season"]) == ("Levi Randolph", 2024)
    (el,) = out["seasons"]  # his GBL season is not published
    assert (el["competition"], el["season"], el["team"], el["games"]) == (
        "euroleague",
        2024,
        "TEL",
        6,
    )
    # the committed stats export of the same raw slice (independent code path)
    fixture = json.loads((FIXTURES / "stats/seasons/2024/players.json").read_text(encoding="utf-8"))
    row = next(p for p in fixture["players"] if p["id"] == "P013382")["totals"]["season"]
    for column, value in el["totals"].items():
        assert value == row[PLAYER_FIELDS.index(column)], column
    assert el["minutes"] == pytest.approx(row[PLAYER_FIELDS.index("sec")] / 60)
    assert el["poss"] == pytest.approx(row[PLAYER_FIELDS.index("poss")], abs=0.05)
    assert el["per100"]["pts"] == pytest.approx(100 * 69 / el["poss"])
    assert el["pct"]["ts"] == pytest.approx(69 / (2 * (43 + 13 + 0.44 * 10)))
    assert [(s["season"], s["total"]) for s in out["impact"]["seasons"]] == [
        (2023, 1.5),
        (2024, 2.5),
    ]
    assert out["impact"]["units"].startswith("points per 100 possessions")
    shot = out["shot_making"]["seasons"]
    assert [(s["season"], s["shot_making"], s["split"]) for s in shot] == [
        (2023, 2.5, "validation"),
        (2024, 3.0, "test"),
    ]


def test_players_outside_the_crosswalk_get_the_source_scheme_and_other_names(
    store: rm.Store,
) -> None:
    with pytest.raises(rm.NotFound, match="unknown person_id"):  # GBL-only: not on the site
        rm.player(store, GBL_ONLY_PERSON)
    unmapped = rm.player(store, "P:P006835")  # an EuroLeague id the crosswalk does not cover
    assert unmapped["name"] == "Jaylen Hoard"
    assert [s["total"] for s in unmapped["impact"]["seasons"]] == [1.0, 2.5]
    assert [s["shot_making"] for s in unmapped["shot_making"]["seasons"]] == [-1.0]


def test_unknown_person_and_a_store_without_box_scores(store: rm.Store, tmp_path: Path) -> None:
    with pytest.raises(rm.NotFound, match="unknown person_id 'P:NOBODY'"):
        rm.player(store, "P:NOBODY")
    empty = tmp_path / "data" / "raw" / "euroleague"
    empty.mkdir(parents=True)
    with pytest.raises(rm.NotFound, match="no box scores are cached"):
        rm.player(rm.Store(tmp_path, from_cache=True), EL_PERSON)


def test_player_payloads_carry_no_birth_date_or_age(store: rm.Store) -> None:
    text = json.dumps([rm.player(store, EL_PERSON), rm.player_projection(store, EL_PERSON)]).lower()
    assert "birth" not in text and '"age"' not in text


def test_projection_similarity_and_scouting_serve_the_committed_reports(store: rm.Store) -> None:
    projection = rm.player_projection(store, EL_PERSON)
    assert (projection["model"], projection["variant"], projection["season"]) == (
        "m6",
        "proj_full",
        2026,
    )
    assert projection["gated"] is True and projection["gate_passed"] is False
    assert projection["projections"][0]["stats"]["pts"] == {
        "mean": 20.5,
        "lo80": 15.0,
        "hi80": 26.0,
    }
    assert "pts" in projection["stats"]

    similar = rm.player_similar(store, EL_PERSON)
    assert [(r["rank"], r["person_id"]) for r in similar["similar"]] == [(1, "P:P006835")]
    assert list(similar["features"]) == ["euroleague"]
    assert similar["pool_seasons"] == [2023, 2024, 2025]

    board = rm.scouting_board(store)
    assert board["rows"][0]["z"] == 1.5 and board["dimensions"]["shot_making"]["n_min"] == 100
    assert {r["competition"] for r in board["rows"]} == {"euroleague"}


def test_projection_and_similarity_not_found_say_why(store: rm.Store, fresh: Path) -> None:
    with pytest.raises(
        rm.NotFound, match=r"no M6 projection for person_id 'P:X' .*2026 projected set"
    ):
        rm.player_projection(store, "P:X")
    with pytest.raises(rm.NotFound, match="no similarity list for person_id 'G:ABCD1234'"):
        rm.player_similar(store, "G:ABCD1234")
    (fresh / M6_PROJECTIONS).unlink()
    with pytest.raises(rm.NotFound, match=r"m6_projections\.json not found: the M6 projections"):
        rm.player_projection(rm.Store(fresh), EL_PERSON)


def test_a_malformed_report_is_an_error_not_a_404(fresh: Path) -> None:
    path = fresh / M6_PROJECTIONS
    report = json.loads(path.read_text(encoding="utf-8"))
    del report["players"][0]["exposure"]
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(pandera.errors.SchemaError):
        rm.player_projection(rm.Store(fresh), EL_PERSON)


def test_the_ungated_simulation_is_labelled_with_the_m7_verdict(store: rm.Store) -> None:
    out = rm.simulation_latest(store, "euroleague")
    assert out["gated"] is False and out["teams"][0]["team"] == "BAR"
    assert out["gate"]["passed"] is False
    # fixture gate: chosen Brier 0.091951 is not below point_sim's 0.088277; the other rules pass
    assert out["gate"]["reason"] == (
        "M7 failed its validation gate: validation Brier 0.0920 is not below the point rule's "
        "0.0883"
    )


def test_a_passed_gate_v2_gates_the_euroleague_simulation_only(fresh: Path) -> None:
    v2 = fresh / "reports" / "backtest_m7_v2.json"
    v2.write_text(json.dumps({"gate_v2": {"passed": True}}), encoding="utf-8")
    el = rm.simulation_latest(rm.Store(fresh), "euroleague")
    assert el["gated"] is True and el["gate"]["passed"] is True
    assert "gate v2" in el["gate"]["reason"]
    (fresh / "reports" / "sim_ungated_gbl.json").write_text(
        (fresh / "reports" / "sim_ungated_euroleague.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    gbl = rm.simulation_latest(rm.Store(fresh), "gbl")
    assert gbl["gated"] is False and "GBL is not gated" in gbl["gate"]["reason"]
    v2.write_text(json.dumps({"gate_v2": {"passed": False}}), encoding="utf-8")
    assert rm.simulation_latest(rm.Store(fresh), "euroleague")["gated"] is False


def test_without_an_ungated_report_the_404_states_the_verdict(fresh: Path) -> None:
    (fresh / "reports" / "sim_ungated_euroleague.json").unlink()
    with pytest.raises(rm.NotFound) as caught:
        rm.simulation_latest(rm.Store(fresh), "euroleague")
    assert caught.value.reason.startswith("no ungated simulation report for euroleague: M7 failed")
    assert caught.value.extra["gated"] is False
    assert caught.value.extra["gate"]["passed"] is False
    with pytest.raises(rm.NotFound, match="no ungated simulation report for gbl"):
        rm.simulation_latest(rm.Store(fresh), "gbl")


def _m7_store(tmp_path: Path, gate: dict[str, Any] | None) -> rm.Store:
    if gate is not None:
        (tmp_path / M7.report).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / M7.report).write_text(json.dumps({"gate": gate}), encoding="utf-8")
    return rm.Store(tmp_path)


GATE = {
    "brier_chosen": 0.09,
    "brier_point_sim": 0.10,
    "brier_standings_now": 0.15,
    "spiegelhalter_z_pooled": 0.5,
    "passed": False,
}


def test_the_m7_verdict_names_each_rule_that_was_missed(tmp_path: Path) -> None:
    assert rm.m7_verdict(_m7_store(tmp_path, {**GATE, "passed": True})) == {
        "passed": True,
        "reason": "M7 passed its validation gate",
    }
    # every number below is a bound from the gate's rule: Brier below both references, |z| < 1.96
    both = _m7_store(tmp_path, {**GATE, "brier_chosen": 0.2, "spiegelhalter_z_pooled": -2.5})
    reason = rm.m7_verdict(both)["reason"]
    assert reason == (
        "M7 failed its validation gate: validation Brier 0.2000 is not below the point rule's "
        "0.1000; validation Brier 0.2000 is not below standings-now's 0.1500; "
        "pooled Spiegelhalter |z| 2.50 is not below 1.96"
    )
    z_only = rm.m7_verdict(_m7_store(tmp_path, {**GATE, "spiegelhalter_z_pooled": 1.96}))
    assert z_only["reason"].endswith("pooled Spiegelhalter |z| 1.96 is not below 1.96")
    unexplained = _m7_store(tmp_path, {**GATE})
    assert rm.m7_verdict(unexplained)["reason"] == "M7 did not pass its validation gate"


def test_the_m7_verdict_without_a_backtest_report(tmp_path: Path) -> None:
    verdict = rm.m7_verdict(_m7_store(tmp_path, None))
    assert verdict["passed"] is False and "backtest_m7.json not found" in verdict["reason"]


def test_live_metrics_carry_the_euroleague_scorecard_and_the_shadow_blocks(
    store: rm.Store, fresh: Path
) -> None:
    out = rm.metrics_live(store)
    assert [c["competition"] for c in out["competitions"]] == ["euroleague"]
    el = out["competitions"][0]["scorecard"]
    assert el["elo"]["log_loss"] == 0.61 and "m1" in el and "m5" in el
    (fresh / "reports" / "live_scorecard.json").unlink()
    with pytest.raises(rm.NotFound, match="no live scorecard has been written"):
        rm.metrics_live(rm.Store(fresh))
