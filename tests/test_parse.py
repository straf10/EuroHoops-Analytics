import pandas as pd

from eurohoops.parse.games import (
    build_games_table,
    build_team_seasons,
    build_teams_table,
    parse_schedule,
)
from tests.conftest import load_fixture


def test_schedule_fixture_parses_to_expected_rows() -> None:
    games = parse_schedule(2024, load_fixture("schedule_E2024.json"))
    first = games.iloc[0]
    assert list(games["game_id"]) == ["E2024_1", "E2024_326", "E2024_327"]
    assert list(games["phase"]) == ["RS", "PO", "FF"]
    assert (first["home"], first["away"], first["home_score"], first["away_score"]) == (
        "BER",
        "PAN",
        77,
        87,
    )
    assert first["round"] == 1
    assert games["played"].all()


def test_tipoff_comes_from_utcdate_not_local_date() -> None:
    games = parse_schedule(2024, load_fixture("schedule_E2024.json"))
    # Fixture: date "2024-10-03T18:45:00" (local) vs utcDate "2024-10-03T16:45:00Z".
    assert games.loc[0, "tipoff_utc"] == pd.Timestamp("2024-10-03T16:45:00Z")
    assert str(games["tipoff_utc"].dtype) == "datetime64[ns, UTC]"


def test_neutral_venue_flag_is_respected() -> None:
    games = parse_schedule(2024, load_fixture("schedule_E2024.json"))
    assert games.set_index("phase")["neutral"].to_dict() == {"RS": False, "PO": False, "FF": True}


def test_unplayed_games_have_null_scores_not_zero() -> None:
    games = parse_schedule(2026, load_fixture("schedule_E2026.json"))
    assert not games["played"].any()
    assert games["home_score"].isna().all()
    assert games["away_score"].isna().all()


def test_games_table_is_validated_and_sorted_by_tipoff() -> None:
    table = build_games_table(
        {2026: load_fixture("schedule_E2026.json"), 2024: load_fixture("schedule_E2024.json")}
    )
    assert len(table) == 5
    assert table["tipoff_utc"].is_monotonic_increasing
    # Same tip-off time: ties broken by game code.
    assert list(table.loc[table["season"] == 2026, "game_code"]) == [2, 3]


def test_final_four_is_neutral_even_when_the_api_flag_says_otherwise() -> None:
    games = load_fixture("schedule_E2024.json")
    final_four = next(g for g in games if g["phaseType"]["code"] == "FF")
    final_four["isNeutralVenue"] = False  # as in all 2023-24 Final Four games
    parsed = parse_schedule(2024, games).set_index("phase")
    assert parsed.loc["FF", "neutral"]
    assert not parsed.loc["PO", "neutral"]


def test_round_label_and_no_forfeits_in_euroleague() -> None:
    games = parse_schedule(2024, load_fixture("schedule_E2024.json"))
    assert games.loc[0, "round_label"] == "Round 1"
    assert not games["forfeit"].any()


def test_teams_table_keeps_the_latest_name() -> None:
    older = load_fixture("schedule_E2024.json")
    newer = [{**older[0], "local": {**older[0]["local"], "club": {"code": "BER", "name": "New"}}}]
    teams = build_teams_table({2025: newer, 2024: older}).set_index("team")["name"]
    assert teams["BER"] == "New"
    assert teams["PAN"] == "Panathinaikos AKTOR Athens"


def test_team_seasons_keep_each_seasons_name() -> None:
    older = load_fixture("schedule_E2024.json")
    newer = [{**older[0], "local": {**older[0]["local"], "club": {"code": "BER", "name": "New"}}}]
    rows = build_team_seasons({2025: newer, 2024: older}).set_index(["season", "team"])["name"]
    assert rows[2025, "BER"] == "New"
    assert rows[2024, "BER"] != "New"
    assert rows[2024, "PAN"] == "Panathinaikos AKTOR Athens"
