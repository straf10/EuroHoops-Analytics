"""Raw EuroLeague schedules -> the validated ``games`` staging table."""

from pathlib import Path

import pandas as pd
import pandera.pandas as pa

from eurohoops.ingest.euroleague import RawGame

PHASES = ("RS", "PI", "PO", "FF")


def _scores_match_played(df: pd.DataFrame) -> pd.Series:
    played = df["played"]
    return (df["home_score"].notna() == played) & (df["away_score"].notna() == played)


def _no_draws(df: pd.DataFrame) -> pd.Series:
    return ~df["played"] | (df["home_score"] != df["away_score"]).fillna(value=False)


GAMES_SCHEMA = pa.DataFrameSchema(
    {
        "game_id": pa.Column(str, unique=True),
        "season": pa.Column("int64"),
        "game_code": pa.Column("int64", pa.Check.gt(0)),
        "phase": pa.Column(str, pa.Check.isin(PHASES)),
        "round": pa.Column("int64", pa.Check.gt(0)),
        "tipoff_utc": pa.Column(pd.DatetimeTZDtype("ns", "UTC")),
        "home": pa.Column(str),
        "away": pa.Column(str),
        "home_score": pa.Column("Int64", pa.Check.ge(0), nullable=True),
        "away_score": pa.Column("Int64", pa.Check.ge(0), nullable=True),
        "played": pa.Column(bool),
        "neutral": pa.Column(bool),
        "confirmed_date": pa.Column(bool),
    },
    checks=[
        pa.Check(lambda df: df["home"] != df["away"], error="home team equals away team"),
        pa.Check(_scores_match_played, error="scores must be non-null iff played"),
        pa.Check(_no_draws, error="a played game cannot end in a draw"),
    ],
    unique=["season", "game_code"],
    strict=True,
)


def parse_schedule(season: int, games: list[RawGame]) -> pd.DataFrame:
    """Parse one season. Tip-off comes from ``utcDate``; ``date`` is local time and never used."""
    rows = [
        {
            "game_id": game["identifier"],
            "season": season,
            "game_code": game["gameCode"],
            "phase": game["phaseType"]["code"],
            "round": game["round"],
            "tipoff_utc": game["utcDate"],
            "home": game["local"]["club"]["code"],
            "away": game["road"]["club"]["code"],
            # Unplayed games carry score 0 in the API; they must be null here.
            "home_score": game["local"]["score"] if game["played"] else None,
            "away_score": game["road"]["score"] if game["played"] else None,
            "played": game["played"],
            "neutral": game["isNeutralVenue"],
            "confirmed_date": game["confirmedDate"],
        }
        for game in games
    ]
    df = pd.DataFrame(rows, columns=list(GAMES_SCHEMA.columns))
    return df.astype(
        {
            "game_id": str,
            "season": "int64",
            "game_code": "int64",
            "phase": str,
            "round": "int64",
            "home": str,
            "away": str,
            "home_score": "Int64",
            "away_score": "Int64",
            "played": bool,
            "neutral": bool,
            "confirmed_date": bool,
        }
    ).assign(
        tipoff_utc=pd.to_datetime(df["tipoff_utc"], format="ISO8601", utc=True).astype(
            "datetime64[ns, UTC]"
        )
    )


def build_games_table(schedules: dict[int, list[RawGame]]) -> pd.DataFrame:
    frames = [parse_schedule(season, games) for season, games in schedules.items()]
    games = pd.concat(frames, ignore_index=True).sort_values(["tipoff_utc", "game_code"])
    return GAMES_SCHEMA.validate(games.reset_index(drop=True))


def write_games(games: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    games.to_parquet(path, engine="pyarrow", index=False)


def read_games(path: Path) -> pd.DataFrame:
    return GAMES_SCHEMA.validate(pd.read_parquet(path, engine="pyarrow"))
