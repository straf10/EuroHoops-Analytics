"""EuroLeague titles: champions from the Final Four games, and the all-time totals with the seed."""

import pandas as pd

from eurohoops.publish import DISPLAY_CODES
from eurohoops.titles import all_time, champions, load_seed, titles_payload
from tests.conftest import FIXTURES

# The 18 champions since 2007-08 (season = start year), source codes.
CHAMPIONS = {
    2007: "CSK", 2008: "PAN", 2009: "BAR", 2010: "PAN", 2011: "OLY", 2012: "OLY",
    2013: "TEL", 2014: "MAD", 2015: "CSK", 2016: "ULK", 2017: "MAD", 2018: "CSK",
    2020: "IST", 2021: "IST", 2022: "MAD", 2023: "PAN", 2024: "ULK", 2025: "OLY",
}  # fmt: skip

# All-time totals (seed plus data), source codes; the uncoded clubs are in the sum only.
ALL_TIME = {
    "MAD": 11, "CSK": 8, "PAN": 7, "TEL": 6, "OLY": 4, "MIL": 3, "BAR": 2, "VIR": 2,
    "ULK": 2, "IST": 2, "CTU": 2, "CIB": 2, "JOV": 1, "ZAL": 1, "PAR": 1, "ROM": 1, "LMG": 1,
}  # fmt: skip


def final_four() -> pd.DataFrame:
    games = pd.read_csv(FIXTURES / "el_final_four.csv", keep_default_na=False)
    games["tipoff_utc"] = pd.to_datetime(games["tipoff_utc"], utc=True)
    return games.drop(columns="competition")


def test_the_data_champions_are_the_eighteen_finals() -> None:
    champs = champions(final_four())
    assert dict(zip(champs["season"], champs["champion"], strict=True)) == CHAMPIONS
    assert len(champs) == 18 and 2019 not in set(champs["season"])


def test_the_final_is_the_last_played_game_not_the_earlier_third_place() -> None:
    def game(game_id: str, tipoff: str, teams: tuple[str, str], scores: tuple[int, int]) -> dict:
        home, away = teams
        return {
            "game_id": game_id, "season": 2010, "phase": "FF", "played": True,
            "tipoff_utc": pd.Timestamp(tipoff), "home": home, "away": away,
            "home_score": scores[0], "away_score": scores[1],
        }  # fmt: skip

    games = pd.DataFrame(
        [
            # earlier the same day: the third-place game, won by SIE
            game("third", "2011-05-08T14:30:00+03:00", ("MAD", "SIE"), (62, 80)),
            game("final", "2011-05-08T17:30:00+03:00", ("TEL", "PAN"), (70, 78)),
            game("ahead", "2011-05-09T17:30:00+03:00", ("PAN", "TEL"), (90, 10)),
        ]
    )
    games.loc[games["game_id"] == "ahead", "played"] = False  # FF but not played
    # a regular-season game after the final must not count
    regular = {**game("rs", "2011-05-10T17:30:00+03:00", ("CSK", "OLY"), (1, 99)), "phase": "RS"}
    champs = champions(pd.concat([games, pd.DataFrame([regular])]))
    assert champs.to_dict("records") == [{"season": 2010, "champion": "PAN"}]


def test_seed_is_the_owner_list_of_51_titles_with_source_codes() -> None:
    seed = load_seed()
    assert seed["titles"].sum() == 51
    coded = seed.dropna(subset=["code"])
    assert set(coded["code"]) <= set(DISPLAY_CODES["euroleague"]) | set(ALL_TIME)
    assert coded["code"].is_unique
    assert seed.loc[seed["club"] == "Pallacanestro Varese", "code"].isna().all()


def test_seed_plus_data_gives_the_all_time_totals() -> None:
    seed = load_seed()
    totals = all_time(seed, champions(final_four()))
    assert dict(totals) == ALL_TIME
    # the clubs with no code (not in our data) are counted in the 69 as well
    assert seed.loc[seed["code"].isna(), "titles"].sum() == 13
    assert int(totals.sum()) + 13 == 69


def test_the_payload_is_keyed_by_display_codes_and_lists_clubs_with_a_title() -> None:
    payload = titles_payload(final_four(), DISPLAY_CODES["euroleague"])
    assert payload["since"] == 1958
    titles = payload["titles"]
    assert titles["RMB"] == 11 and titles["PAO"] == 7 and titles["OLY"] == 4
    assert titles["FBT"] == 2 and titles["EFS"] == 2 and titles["MTA"] == 6
    assert "MAD" not in titles and "ULK" not in titles and "TEL" not in titles
    assert len(titles) == len(ALL_TIME)
    assert all(n >= 1 for n in titles.values())


def test_no_games_means_the_seed_alone() -> None:
    payload = titles_payload(None, DISPLAY_CODES["euroleague"])
    assert payload["titles"]["RMB"] == 8 and payload["titles"]["PAO"] == 4
    assert "FBT" not in payload["titles"]  # Fenerbahce has no title before 2007-08
