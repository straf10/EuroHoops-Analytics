import pandas as pd

from eurohoops.parse.games import build_games_table, parse_schedule
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
