import pandas as pd

from eurohoops.parse.continuity import continuity, continuity_report


def seasons(rows: list[tuple[int, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["season", "team", "name"])


TEAM_SEASONS = seasons(
    [
        (2022, "OLY", "Olympiacos"),
        (2022, "MCO", "AS Monaco"),
        (2022, "PAM", "Valencia Basket"),
        (2023, "OLY", "Olympiacos Piraeus"),
        (2023, "MCO", "AS Monaco"),
        (2024, "OLY", "Olympiacos Piraeus"),
        (2024, "PAM", "Power Electronics Valencia"),  # back after a season away
        (2024, "BES", "Besiktas Istanbul"),  # first season in the data
        # MCO left after 2023; a different club taking over the code shares no name word:
        (2025, "OLY", "Olympiacos Piraeus"),
        (2025, "PAM", "Power Electronics Valencia"),
        (2025, "BES", "Besiktas JK Istanbul"),
        (2025, "MCO", "Hapoel Tel Aviv"),
    ]
)


def test_new_returning_left_and_renamed_codes() -> None:
    report = continuity(TEAM_SEASONS)
    s = report["seasons"]
    assert s["2022"] == {"teams": 3, "first_season_in_data": True}
    assert s["2023"]["left"] == [{"team": "PAM", "name": "Valencia Basket"}]
    assert s["2023"]["renamed"] == [
        {"team": "OLY", "from": "Olympiacos", "to": "Olympiacos Piraeus"}
    ]
    assert s["2024"]["new"] == [{"team": "BES", "name": "Besiktas Istanbul"}]
    assert s["2024"]["returning"] == [
        {
            "team": "PAM",
            "name": "Power Electronics Valencia",
            "last_season": 2022,
            "name_then": "Valencia Basket",
        }
    ]
    assert s["2024"]["left"] == [{"team": "MCO", "name": "AS Monaco"}]
    assert s["2025"]["returning"][0]["team"] == "MCO"
    assert s["2025"]["new"] == []


def test_only_name_changes_without_a_shared_word_need_review() -> None:
    """Sponsor renames that keep a word pass; a code taken over by another club is flagged."""
    review = continuity(TEAM_SEASONS)["review_name_changes"]
    assert review == [{"season": 2025, "team": "MCO", "from": "AS Monaco", "to": "Hapoel Tel Aviv"}]


def test_report_has_one_block_per_competition() -> None:
    both = pd.concat(
        [
            TEAM_SEASONS.assign(competition="euroleague"),
            seasons([(2024, "A1", "Aris"), (2025, "A1", "Aris Thessaloniki")]).assign(
                competition="gbl"
            ),
        ]
    )
    report = continuity_report(both)
    assert list(report) == ["euroleague", "gbl"]
    assert report["gbl"]["seasons"]["2025"]["renamed"] == [
        {"team": "A1", "from": "Aris", "to": "Aris Thessaloniki"}
    ]
    assert report["gbl"]["review_name_changes"] == []
