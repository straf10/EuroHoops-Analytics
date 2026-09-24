import pandas as pd
import pandera.errors
import pytest

from eurohoops.parse.games import GAMES_SCHEMA, parse_schedule
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
