import json
from datetime import UTC, datetime

import pandas as pd
from pytest import MonkeyPatch

from eurohoops.eval.backtest import TunedModel
from eurohoops.models.elo import EloParams
from eurohoops.predict import LOG_COLUMNS
from eurohoops.publish import DISPLAY_CODES, Section, section_data, site_data
from tests.conftest import make_games

NOW = datetime(2026, 10, 8, 8, 0, tzinfo=UTC)
CARD = {
    "elo": {"n": 3, "log_loss": 0.61, "brier": 0.21, "accuracy": 2 / 3, "margin_mae": 9.5},
    "b0": {"n": 3, "log_loss": 0.66, "brier": 0.23, "accuracy": 0.6, "margin_mae": 9.9},
    "rolling": {"window": 50, "series": []},
}
REPORT = {
    "seasons": {"warmup": [2024], "tuning": [2025], "test": [2025]},
    "tuned": {"k": 20.0, "hca": 90.0, "reversion": 0.25},
    "metrics": {"test": CARD},
    "test_log_loss_diff_elo_minus_b0": {"mean": -0.05, "ci95": [-0.08, -0.02]},
}
MODEL = TunedModel(EloParams(k=20.0, hca=90.0, reversion=0.25), 25.0, 0.63, 3.7)


def section(card: dict[str, object] = CARD, logged_rounds: int = 2) -> Section:
    games = make_games({2025: True, 2026: False})
    played = (games["season"] == 2026) & (games["round"] == 1)
    games.loc[played, ["home_score", "away_score", "played"]] = [90, 80, True]
    live = games[(games["season"] == 2026) & (games["round"] <= logged_rounds)]
    log = pd.DataFrame(
        [
            [g.game_id, 2026, g.round, "RS", "", g.home, g.away, 0.4, -2.0, "elo", "v1", stamp]
            for g in live.itertuples()
            for stamp in ("2026-09-30T08:00:00Z", "2026-09-30T09:00:00Z")  # a re-log
        ],
        columns=list(LOG_COLUMNS),
    )
    names = {"AAA": "Ολυμπιακός & <Co>"}
    return Section("gbl", "Greek Basket League", log, card, REPORT, games, names, MODEL, 2026, 2025)


def test_upcoming_results_and_scorecard() -> None:
    data = section_data(section(), NOW)
    assert data["season"] == "2026-27"
    assert data["logged"] == 6  # first row per game only
    assert len(data["upcoming"]) == 3
    assert len(data["results"]) == 3
    result = data["results"][0]
    assert (result["home_score"], result["away_score"]) == (90, 80)
    assert not result["hit"]  # p_home 0.4 but the home team won
    assert result["provable"]
    assert result["predicted_at_utc"] == "2026-09-30T08:00:00Z"
    assert data["scorecard"]["elo"]["log_loss"] == 0.61
    assert data["scorecard"]["not_provable"] == 0
    assert data["backtest"]["params"] == {"k": 20.0, "hca": 90.0, "reversion": 0.25}
    assert data["backtest"]["log_loss_diff"]["ci95"] == [-0.08, -0.02]
    assert data["next_tipoff_utc"] == data["upcoming"][0]["tipoff_utc"]


def test_ratings_cover_the_live_season_and_keep_raw_names() -> None:
    ratings = section_data(section(), NOW)["ratings"]
    assert len(ratings) == 6
    assert [r["rating"] for r in ratings] == sorted((r["rating"] for r in ratings), reverse=True)
    assert any(r["name"] == "Ολυμπιακός & <Co>" for r in ratings)  # escaping is the page's job
    assert all(isinstance(r["change"], float) for r in ratings)


def test_unprovable_games_are_flagged() -> None:
    plain = section()
    first = plain.games.loc[
        (plain.games["season"] == 2026) & (plain.games["round"] == 1), "game_id"
    ].iloc[0]
    data = section_data(section({**CARD, "games_not_provable": [first]}), NOW)
    assert data["scorecard"]["not_provable"] == 1
    assert [r["provable"] for r in data["results"]].count(False) == 1


def test_site_without_a_log_is_json_ready() -> None:
    empty = Section(
        "euroleague",
        "EuroLeague",
        pd.DataFrame(),
        CARD,
        REPORT,
        make_games({2026: False}),
        {},
        MODEL,
        2026,
        2026,
    )
    data = site_data([empty], NOW)
    json.dumps(data)
    comp = data["competitions"][0]
    assert comp["upcoming"] == comp["results"] == []
    assert comp["logged"] == 0
    assert data["generated_at_utc"] == "2026-10-08T08:00:00Z"


def test_display_codes_rename_only_what_the_site_shows(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setitem(DISPLAY_CODES, "gbl", {"AAA": "OLY"})
    data = section_data(section(), NOW)
    rated = {r["code"]: r["name"] for r in data["ratings"]}
    assert rated["OLY"] == "Ολυμπιακός & <Co>"  # the name is still looked up by the source code
    assert "AAA" not in rated
    teams = {t["code"] for g in data["upcoming"] + data["results"] for t in (g["home"], g["away"])}
    assert "AAA" not in teams


def test_display_codes_are_unique_per_competition() -> None:
    for codes in DISPLAY_CODES.values():
        assert len(set(codes.values())) == len(codes)
