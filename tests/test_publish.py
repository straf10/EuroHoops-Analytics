from datetime import UTC, datetime
from pathlib import Path

from eurohoops.predict import LOG_COLUMNS
from eurohoops.publish import Section, render_page
from tests.conftest import make_games

NOW = datetime(2026, 10, 8, 8, 0, tzinfo=UTC)
CARD = {
    "elo": {"n": 3, "log_loss": 0.61, "brier": 0.21, "accuracy": 2 / 3, "margin_mae": 9.5},
    "b0": {"n": 3, "log_loss": 0.66, "brier": 0.23, "accuracy": 0.6, "margin_mae": 9.9},
}


def section(tmp_path: Path) -> Section:
    games = make_games({2026: False})
    played = games["round"] == 1
    games.loc[played, ["home_score", "away_score", "played"]] = [90, 80, True]
    log = tmp_path / "log.csv"
    rows = [
        f"{g.game_id},2026,{g.round},RS,{g.tipoff_utc:%Y-%m-%dT%H:%M:%SZ},{g.home},{g.away},"
        f"0.4,-2.0,elo,v1,2026-09-30T08:00:00Z"
        for g in games[games["round"] <= 2].itertuples()
    ]
    log.write_text(",".join(LOG_COLUMNS) + "\n" + "\n".join(rows) + "\n")
    names = {"AAA": "Ολυμπιακός & <Co>"}
    return Section("Greek Basket League", log, CARD, games, names)


def test_page_lists_upcoming_and_recent_games_with_escaped_names(tmp_path: Path) -> None:
    page = render_page([section(tmp_path)], NOW)
    upcoming = page.split("<h3>Upcoming</h3>")[1].split("<h3>Recent results</h3>")[0]
    recent = page.split("<h3>Recent results</h3>")[1].split("<h3>Scorecard")[0]
    assert upcoming.count("<tr>") == 1 + 3  # header + the three round-2 games
    assert recent.count("<tr>") == 1 + 3
    assert "90-80" in recent
    assert "miss" in recent  # p_home 0.4 but the home team won
    assert "Ολυμπιακός &amp; &lt;Co&gt;" in page
    assert "<Co>" not in page
    assert "Athens time" in page
    assert "not betting advice" in page
    assert "66.7%" in page
    assert "Not scored" not in page


def test_unprovable_games_are_marked_and_explained(tmp_path: Path) -> None:
    plain = section(tmp_path)
    first = plain.games.loc[plain.games["round"] == 1, "game_id"].iloc[0]
    card = {**CARD, "games_not_provable": [first]}
    page = render_page([Section(plain.title, plain.log_path, card, plain.games, {})], NOW)
    recent = page.split("<h3>Recent results</h3>")[1].split("<h3>Scorecard")[0]
    assert recent.count("*") == 1
    assert "* Not scored: 1 game whose prediction" in page


def test_page_without_a_log(tmp_path: Path) -> None:
    empty = Section("EuroLeague", tmp_path / "none.csv", CARD, make_games({2026: False}), {})
    page = render_page([empty], NOW)
    assert "No logged games in the next window." in page
    assert "No logged game has finished yet." in page
