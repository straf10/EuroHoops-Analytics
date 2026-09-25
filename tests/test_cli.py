import dataclasses
import json
from collections.abc import Iterator
from datetime import UTC, datetime
from itertools import product
from pathlib import Path

import httpx
import pandas as pd
import pytest
from typer.testing import CliRunner

from eurohoops import cli
from eurohoops.config import (
    BOX_INVARIANTS_REPORT,
    EUROLEAGUE,
    GBL,
    GBL_BOX_FILL,
    GBL_PBP,
    MART_PATH,
    ODDS_TEAMS,
    POSSESSION_REPORT,
    SITE_DATA,
    SQL_DIR,
    STINT_REPORT,
    STINTS_MART_REPORT,
)
from eurohoops.eval.backtest import load_tuned_model
from eurohoops.marts import read_games, write_tables
from tests.conftest import REPO, make_games, make_team_games, write_pipeline
from tests.test_gbl_ingest import FakeEsake
from tests.test_gbl_pbp import GAME, FakeBasketHotel, export
from tests.test_ingest import FakeApi
from tests.test_m1_backtest import SPEC as M1_SPEC
from tests.test_stints import game, write_game
from tests.test_stints_mart import full_box

runner = CliRunner()
NOW = datetime(2026, 10, 1, 8, 0, tzinfo=UTC)


@pytest.fixture
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    (tmp_path / SQL_DIR).mkdir()
    for sql in (REPO / SQL_DIR).glob("*.sql"):
        (tmp_path / SQL_DIR / sql.name).write_text(sql.read_text(encoding="utf-8"))
    return tmp_path


@pytest.fixture
def pipeline(workdir: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Staged + built synthetic data covering every backtest split of both competitions."""
    write_pipeline(
        make_games({s: True for s in range(2007, 2026)} | {2026: False}),
        make_games({s: True for s in range(2018, 2026)} | {2026: False}, gbl_like=True),
    )
    monkeypatch.setattr(cli, "utc_now", lambda: NOW)
    return workdir


def invoke(*args: str) -> str:
    result = runner.invoke(cli.app, list(args))
    assert result.exit_code == 0, result.output
    return result.output


@pytest.mark.usefixtures("workdir", "no_sleep")
def test_ingest_both_competitions_then_build(monkeypatch: pytest.MonkeyPatch) -> None:
    el_api, esake = FakeApi(), FakeEsake()
    transports = iter([httpx.MockTransport(el_api), httpx.MockTransport(esake)])
    monkeypatch.setattr(cli, "make_client", lambda: httpx.Client(transport=next(transports)))
    invoke("ingest", "--seasons", "2024", "2026", "--details")
    assert len(el_api.calls) == 2 + 3 * 3
    invoke("ingest", "--competition", "gbl", "--seasons", "2018", "--details")
    assert len(esake.calls) == 4 + 4
    output = invoke("build")
    assert "gbl box scores 2018: 0/4 pass" in output  # one recorded box served for every game
    report = json.loads(BOX_INVARIANTS_REPORT.read_text())
    assert report["seasons"]["2018"]["games"] == 4
    assert list(read_games(MART_PATH, "euroleague")["season"].unique()) == [2024, 2026]
    assert read_games(MART_PATH, "gbl")["forfeit"].sum() == 1


@pytest.mark.usefixtures("workdir")
def test_ingest_rejects_junk_arguments() -> None:
    result = runner.invoke(cli.app, ["ingest", "--seasons", "2024", "--detials"])
    assert result.exit_code != 0


@pytest.mark.usefixtures("pipeline")
def test_backtest_predict_score_publish_end_to_end() -> None:
    output = invoke("backtest")
    assert "backtest_elo_history.json" in output
    for spec in (EUROLEAGUE.live_backtest, *EUROLEAGUE.history_backtests):
        grid = set(product(spec.grid.k, spec.grid.hca, spec.grid.reversion))
        report = json.loads(spec.report.read_text())
        tuned = report["tuned"]
        assert (tuned["k"], tuned["hca"], tuned["reversion"]) in grid
        assert not tuned["frozen"]
        assert {k: report["grid"]["best"][k] for k in ("k", "hca", "reversion")} == {
            k: tuned[k] for k in ("k", "hca", "reversion")
        }
        assert report["grid"]["best_minus_tuned_log_loss"]["test"]["mean"] == 0.0
        assert report["seasons"]["test"] == list(spec.test)
        assert report["model_version"] == load_tuned_model(spec.report).version()
    output = invoke("backtest", "--competition", "gbl")
    assert "[frozen]" in output
    gbl_report = json.loads(GBL.live_backtest.report.read_text())
    assert GBL.live_backtest.frozen is not None
    assert load_tuned_model(GBL.live_backtest.report).params == GBL.live_backtest.frozen
    assert gbl_report["tuned"]["frozen"]
    assert gbl_report["grid"]["size"] == 9 * 10 * 5
    assert gbl_report["seasons"]["tuning"] == [2022, 2023]
    ci = gbl_report["test_margin_abs_error_diff_elo_minus_b0"]
    assert ci["ci95"][0] <= ci["mean"] <= ci["ci95"][1]

    invoke("predict")
    el_log = EUROLEAGUE.prediction_log.read_bytes()
    invoke("predict", "--competition", "gbl")
    invoke("predict", "--competition", "gbl")
    assert EUROLEAGUE.prediction_log.read_bytes() == el_log  # a GBL run never touches it
    gbl_log = pd.read_csv(GBL.prediction_log)
    assert len(gbl_log) == 3
    assert gbl_log["game_id"].str.startswith("GBL2026_").all()

    invoke("score")
    invoke("score", "--competition", "gbl")
    assert json.loads(GBL.scorecard.read_text())["rows_in_log"] == 3
    invoke("publish")
    data = json.loads(SITE_DATA.read_text(encoding="utf-8"))
    el, gbl_data = data["competitions"]
    assert (el["key"], gbl_data["key"]) == ("euroleague", "gbl")
    assert len(gbl_data["upcoming"]) == 3
    assert any(r["name"] == "Team AAA & Co" for r in el["ratings"])


@pytest.mark.usefixtures("pipeline")
def test_predict_exits_non_zero_when_stamp_is_not_before_tipoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invoke("backtest")
    ticks: Iterator[datetime] = iter([NOW, datetime(2026, 10, 1, 18, 0, tzinfo=UTC)])
    monkeypatch.setattr(cli, "utc_now", lambda: next(ticks))
    result = runner.invoke(cli.app, ["predict"])
    assert result.exit_code == 1
    assert not EUROLEAGUE.prediction_log.exists()


def odds_api(status: int) -> httpx.MockTransport:
    events = [
        {
            "commence_time": "2026-10-01T18:00:00Z",
            "home_team": "Alpha",
            "away_team": "Bravo",
            "bookmakers": [
                {
                    "key": "a",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Alpha", "price": 1.5},
                                {"name": "Bravo", "price": 2.5},
                            ],
                        }
                    ],
                }
            ],
        }
    ]
    body = events if status == 200 else {"error_code": "INVALID_KEY"}
    headers = {"x-requests-last": "3", "x-requests-remaining": "400"}
    return httpx.MockTransport(lambda request: httpx.Response(status, json=body, headers=headers))


@pytest.mark.usefixtures("pipeline")
def test_odds_records_a_consensus_row(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ODDS_API_KEY", "fake-key")
    ODDS_TEAMS.parent.mkdir(parents=True, exist_ok=True)
    ODDS_TEAMS.write_text("odds_api_name,team,verified\nAlpha,AAA,no\nBravo,BBB,no\n")
    monkeypatch.setattr(cli, "make_client", lambda: httpx.Client(transport=odds_api(200)))
    output = invoke("odds")
    assert "1 consensus rows from 1 events" in output
    assert "requests remaining 400" in output
    assert EUROLEAGUE.odds_log is not None
    [row] = pd.read_csv(EUROLEAGUE.odds_log).to_dict("records")
    assert (row["game_id"], row["p_home"]) == ("E2026_1", 0.625)
    invoke("backtest")
    invoke("backtest", "--competition", "gbl")
    invoke("score")
    assert json.loads(EUROLEAGUE.scorecard.read_text())["market"]["n"] == 0  # not played yet
    invoke("score", "--competition", "gbl")
    assert json.loads(GBL.scorecard.read_text())["market"] is None


@pytest.mark.usefixtures("pipeline")
def test_odds_fails_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    result = runner.invoke(cli.app, ["odds"])
    assert result.exit_code == 1
    monkeypatch.setenv("ODDS_API_KEY", "fake-key")
    monkeypatch.setattr(cli, "make_client", lambda: httpx.Client(transport=odds_api(401)))
    result = runner.invoke(cli.app, ["odds"])
    assert result.exit_code == 1
    assert "fake-key" not in result.output


@pytest.mark.usefixtures("pipeline")
def test_build_writes_team_games_and_possessions_validates_them() -> None:
    failed = runner.invoke(cli.app, ["possessions"])
    assert failed.exit_code == 1  # the marts have no team_games before `build`
    assert "team_games: 0 games" in invoke("build")  # synthetic games have no box scores
    output = invoke("possessions")
    assert "0 points mismatches; no EuroLeague play-by-play cached" in output
    report = json.loads(POSSESSION_REPORT.read_text(encoding="utf-8"))
    assert (
        report["coverage"]["gbl"]["2025"]["missing"]
        == report["coverage"]["gbl"]["2025"]["rated_games"]
    )
    assert report["missing_games"][0]["reason"] == "no cached box score (ingest --details)"


@pytest.mark.usefixtures("workdir")
def test_stints_writes_a_reproducible_report() -> None:
    write_game(EUROLEAGUE.raw_dir, 2024, 5, *game())
    write_game(EUROLEAGUE.raw_dir, 2025, 9, *game(drop_in=True))
    output = invoke("stints")
    assert "2 games: all checks 50%" in output
    first = STINT_REPORT.read_bytes()
    invoke("stints")
    assert STINT_REPORT.read_bytes() == first
    assert json.loads(first)["failing_games"].keys() == {"E2025_9"}


@pytest.mark.usefixtures("pipeline")
def test_stints_mart_builds_tables_and_a_reproducible_report() -> None:
    assert runner.invoke(cli.app, ["stints", "--mart"]).exit_code == 1  # no team_games yet
    pbp, box = game()
    write_game(EUROLEAGUE.raw_dir, 2024, 1, pbp, full_box(box))  # E2024_1 is AAA vs BBB
    invoke("build")
    output = invoke("stints", "--mart")
    assert "pass rate 2011-14 no games, 2015+ 100.0%" in output
    first = STINTS_MART_REPORT.read_bytes()
    invoke("stints", "--mart")
    assert STINTS_MART_REPORT.read_bytes() == first
    assert b"NaN" not in first
    report = json.loads(first)
    assert report["games"] == 1
    assert report["pbp_vs_box_possessions"]["2011_2014"] is None
    assert (
        len(report["not_cached"])
        == len(read_games(MART_PATH, "euroleague").query("played and season >= 2011")) - 1
    )


@pytest.fixture
def small_m1(pipeline: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The pipeline plus team_games, with the small test grids in both M1 specs."""
    comps = {}
    for comp in (EUROLEAGUE, GBL):
        assert comp.m1 is not None
        spec = dataclasses.replace(
            comp.m1,
            elo_grid=M1_SPEC.elo_grid,
            rating_grid=M1_SPEC.rating_grid,
            pace_grid=M1_SPEC.pace_grid,
        )
        comps[comp.name] = dataclasses.replace(comp, m1=spec)
    monkeypatch.setattr(cli, "COMPETITIONS", comps)
    return pipeline


def test_backtest_m1_writes_reproducible_reports_for_both_competitions(small_m1: Path) -> None:
    failed = runner.invoke(cli.app, ["backtest", "--model", "m1"])
    assert failed.exit_code == 1  # no team_games in the marts yet
    rows = pd.concat(
        [make_team_games(read_games(MART_PATH, c)) for c in ("euroleague", "gbl")],
        ignore_index=True,
    )
    write_tables(MART_PATH, {"team_games": rows})
    output = invoke("backtest", "--model", "m1")
    assert "gate (" in output
    assert "MLflow parent run" in output
    assert (small_m1 / "mlruns" / "mlflow.db").exists()  # the working directory, not the repo
    assert EUROLEAGUE.m1 is not None and GBL.m1 is not None
    first = EUROLEAGUE.m1.report.read_bytes()
    invoke("backtest", "--model", "m1")
    assert EUROLEAGUE.m1.report.read_bytes() == first
    assert "test" not in json.loads(first)["metrics"]
    invoke("backtest", "--model", "m1", "--competition", "gbl", "--score-test")
    gbl = json.loads(GBL.m1.report.read_text(encoding="utf-8"))
    assert gbl["test_scored"] and gbl["metrics"]["test"]["m1"]["n"] > 0
    assert gbl["seasons"]["validation"] == [2023]


class AnyGameHotel(FakeBasketHotel):
    """Serves the recorded export for any game id."""

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request.url)
        if request.url.path.endswith("/show"):
            return httpx.Response(200, text='url += "&game_id=" + 4453393;')
        return httpx.Response(200, content=export(GAME))


@pytest.mark.usefixtures("workdir", "no_sleep")
def test_ingest_gbl_pbp_caches_exports_and_keeps_every_season(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    esake, hotel = FakeEsake(), AnyGameHotel()

    def route(request: httpx.Request) -> httpx.Response:
        return (hotel if request.url.host == "widgets.baskethotel.com" else esake)(request)

    monkeypatch.setattr(
        cli, "make_client", lambda: httpx.Client(transport=httpx.MockTransport(route))
    )
    invoke("ingest", "--competition", "gbl", "--seasons", "2018", "--pbp")
    assert len(hotel.calls) == 2 * 4  # widget + export for each played, non-forfeit game
    staged = pd.read_parquet(GBL.staging_games)
    assert set(staged["season"]) > {2018}  # --pbp never narrows the staging tables
    assert GBL_PBP.exists()
    fill = pd.read_parquet(GBL_BOX_FILL)
    assert not fill.empty  # the fake's one box never matches the fake results
    invoke("ingest", "--competition", "gbl", "--seasons", "2018", "--pbp")
    assert len(hotel.calls) == 2 * 4  # all cached


@pytest.mark.usefixtures("workdir")
def test_pbp_flag_is_gbl_only() -> None:
    result = runner.invoke(cli.app, ["ingest", "--pbp"])
    assert result.exit_code != 0
