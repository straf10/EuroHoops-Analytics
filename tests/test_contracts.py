import pandas as pd
import pandera.errors
import pytest

from eurohoops.parse.games import GAMES_SCHEMA, parse_schedule
from eurohoops.parse.team_box import TEAM_GAMES_SCHEMA
from tests.conftest import load_fixture


@pytest.fixture
def valid() -> pd.DataFrame:
    return parse_schedule(2024, load_fixture("schedule_E2024.json"))


def test_valid_table_passes(valid: pd.DataFrame) -> None:
    GAMES_SCHEMA.validate(valid)


def test_rejects_duplicate_season_game_code(valid: pd.DataFrame) -> None:
    dup = pd.concat([valid, valid.iloc[[0]].assign(game_id="other")], ignore_index=True)
    with pytest.raises(pandera.errors.SchemaError):
        GAMES_SCHEMA.validate(dup)


def test_rejects_duplicate_game_id(valid: pd.DataFrame) -> None:
    dup = pd.concat([valid, valid.iloc[[0]].assign(game_code=999)], ignore_index=True)
    with pytest.raises(pandera.errors.SchemaError):
        GAMES_SCHEMA.validate(dup)


def test_rejects_naive_datetimes(valid: pd.DataFrame) -> None:
    naive = valid.assign(tipoff_utc=valid["tipoff_utc"].dt.tz_localize(None))
    with pytest.raises(pandera.errors.SchemaError):
        GAMES_SCHEMA.validate(naive)


def test_rejects_played_game_without_scores(valid: pd.DataFrame) -> None:
    broken = valid.copy()
    broken.loc[0, "home_score"] = pd.NA
    with pytest.raises(pandera.errors.SchemaError):
        GAMES_SCHEMA.validate(broken)


def test_rejects_unplayed_game_with_scores() -> None:
    games = parse_schedule(2026, load_fixture("schedule_E2026.json"))
    games.loc[0, ["home_score", "away_score"]] = [0, 0]
    with pytest.raises(pandera.errors.SchemaError):
        GAMES_SCHEMA.validate(games)


@pytest.mark.parametrize(
    ("column", "value"), [("away", "BER"), ("phase", "XX"), ("away_score", 77)]
)
def test_rejects_self_play_unknown_phase_and_draws(
    valid: pd.DataFrame, column: str, value: object
) -> None:
    broken = valid.copy()
    broken.loc[0, column] = value
    with pytest.raises(pandera.errors.SchemaError):
        GAMES_SCHEMA.validate(broken)


def test_rejects_forfeit_that_was_not_played() -> None:
    games = parse_schedule(2026, load_fixture("schedule_E2026.json"))
    games.loc[0, "forfeit"] = True
    with pytest.raises(pandera.errors.SchemaError):
        GAMES_SCHEMA.validate(games)


def test_accepts_top_16_phase(valid: pd.DataFrame) -> None:
    GAMES_SCHEMA.validate(valid.assign(phase="TS"))


# --- team_games (E1): one contract for both competitions' team box lines --------------------


@pytest.fixture
def team_games() -> pd.DataFrame:
    rows = [
        ("E2023_1", "RED", "ASV", True, 94, 71, 13, 17, 23, 10, 69.72),
        ("E2023_1", "ASV", "RED", False, 73, 54, 21, 9, 25, 16, 70.24),
    ]
    frame = pd.DataFrame(
        rows,
        columns=[
            *("game_id", "team", "opponent", "home", "points"),
            *("fga", "fta", "oreb", "dreb", "tov", "poss_raw"),
        ],
    )
    frame.insert(0, "season", 2023)
    frame.insert(0, "competition", "euroleague")
    return TEAM_GAMES_SCHEMA.validate(
        frame.assign(minutes=40.0, poss_game=69.98, source="euroleague_box")[
            list(TEAM_GAMES_SCHEMA.columns)
        ]
    )


def test_team_games_contract_accepts_a_real_game(team_games: pd.DataFrame) -> None:
    assert len(team_games) == 2


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("poss_game", 70.0),  # not the mean of both teams
        ("tov", -1),
        ("source", "guess"),
        ("minutes", 35.0),
        ("opponent", "RED"),  # RED plays itself
        ("competition", "nba"),
    ],
)
def test_team_games_contract_rejects(team_games: pd.DataFrame, column: str, value: object) -> None:
    broken = team_games.copy()
    broken.loc[0, column] = value
    with pytest.raises(pandera.errors.SchemaError):
        TEAM_GAMES_SCHEMA.validate(broken)


def test_team_games_contract_needs_both_teams(team_games: pd.DataFrame) -> None:
    with pytest.raises(pandera.errors.SchemaError):
        TEAM_GAMES_SCHEMA.validate(team_games.iloc[[0]].assign(poss_game=69.72))
