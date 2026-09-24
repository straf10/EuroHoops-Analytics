"""Raw EuroLeague schedules and GBL round pages -> validated ``games`` and ``teams`` tables.

Both competitions share one schema so the Elo backtest, predictor and scorecard are
competition-agnostic. GBL-driven additions: ``round_label`` (ESAKE round titles such as "QF1"
are not numbers) and ``forfeit`` (a forfeit is played, scored 20-0, but carries no rating
information). EuroLeague adds phase ``TS`` (Top 16, 2007-08 to 2015-16).
"""

from datetime import UTC
from pathlib import Path
from typing import Any

import pandas as pd
import pandera.pandas as pa

from eurohoops.ingest.euroleague import RawGame
from eurohoops.ingest.gbl import PHASES as GBL_PHASES
from eurohoops.ingest.gbl import RoundPage

PHASES = ("RS", "TS", "PI", "PO", "FF")
ATHENS = "Europe/Athens"
DTYPES: dict[str, Any] = {
    "game_id": str,
    "season": "int64",
    "game_code": "int64",
    "phase": str,
    "round": "int64",
    "round_label": str,
    "home": str,
    "away": str,
    "home_score": "Int64",
    "away_score": "Int64",
    "played": bool,
    "forfeit": bool,
    "neutral": bool,
    "confirmed_date": bool,
}


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
        "round_label": pa.Column(str),
        "tipoff_utc": pa.Column(pd.DatetimeTZDtype("ns", "UTC")),
        "home": pa.Column(str),
        "away": pa.Column(str),
        "home_score": pa.Column("Int64", pa.Check.ge(0), nullable=True),
        "away_score": pa.Column("Int64", pa.Check.ge(0), nullable=True),
        "played": pa.Column(bool),
        "forfeit": pa.Column(bool),
        "neutral": pa.Column(bool),
        "confirmed_date": pa.Column(bool),
    },
    checks=[
        pa.Check(lambda df: df["home"] != df["away"], error="home team equals away team"),
        pa.Check(_scores_match_played, error="scores must be non-null iff played"),
        pa.Check(_no_draws, error="a played game cannot end in a draw"),
        pa.Check(lambda df: ~df["forfeit"] | df["played"], error="a forfeit must be played"),
    ],
    unique=["season", "game_code"],
    strict=True,
)

TEAMS_SCHEMA = pa.DataFrameSchema(
    {"team": pa.Column(str, unique=True), "name": pa.Column(str)}, strict=True
)


def conform(df: pd.DataFrame) -> pd.DataFrame:
    """Cast to the schema dtypes, sort chronologically and validate."""
    typed = df.astype(DTYPES).assign(
        tipoff_utc=pd.to_datetime(df["tipoff_utc"], utc=True).astype("datetime64[ns, UTC]")
    )
    ordered = typed[list(GAMES_SCHEMA.columns)].sort_values(["tipoff_utc", "game_code"])
    return GAMES_SCHEMA.validate(ordered.reset_index(drop=True))


def parse_schedule(season: int, games: list[RawGame]) -> pd.DataFrame:
    """Parse one season. Tip-off comes from ``utcDate``; ``date`` is local time and never used.

    Final Four games are neutral whatever ``isNeutralVenue`` says: the API sets it
    inconsistently (0 of 4 Final Four games in 2023-24, 2 of 4 in 2024-25).
    """
    rows = [
        {
            "game_id": game["identifier"],
            "season": season,
            "game_code": game["gameCode"],
            "phase": game["phaseType"]["code"],
            "round": game["round"],
            "round_label": game["roundAlias"],
            "tipoff_utc": pd.to_datetime(game["utcDate"], format="ISO8601", utc=True),
            "home": game["local"]["club"]["code"],
            "away": game["road"]["club"]["code"],
            # Unplayed games carry score 0 in the API; they must be null here.
            "home_score": game["local"]["score"] if game["played"] else None,
            "away_score": game["road"]["score"] if game["played"] else None,
            "played": game["played"],
            "forfeit": False,
            "neutral": game["isNeutralVenue"] or game["phaseType"]["code"] == "FF",
            "confirmed_date": game["confirmedDate"],
        }
        for game in games
    ]
    return conform(pd.DataFrame(rows, columns=list(GAMES_SCHEMA.columns)))


def build_games_table(schedules: dict[int, list[RawGame]]) -> pd.DataFrame:
    return conform(
        pd.concat(
            [parse_schedule(season, games) for season, games in schedules.items()],
            ignore_index=True,
        )
    )


def build_teams_table(schedules: dict[int, list[RawGame]]) -> pd.DataFrame:
    """Club code -> name as of the latest season it appears in."""
    names = {
        side["club"]["code"]: side["club"]["name"]
        for _, games in sorted(schedules.items())
        for game in games
        for side in (game["local"], game["road"])
    }
    return _teams(names)


def _teams(names: dict[str, str]) -> pd.DataFrame:
    df = pd.DataFrame({"team": list(names), "name": list(names.values())}, dtype=str)
    return TEAMS_SCHEMA.validate(df.sort_values("team", ignore_index=True))


def build_gbl_tables(rounds: list[RoundPage]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """GBL ``games`` (tip-offs converted from Athens time to UTC) and ``teams`` tables."""
    rows = [
        {
            "game_id": f"GBL{r.season}_{game.idgame}",
            "season": r.season,
            "game_code": int(game.idgame, 16),
            "phase": GBL_PHASES[r.phase],
            "round": int(r.code),
            "round_label": game.round_label,
            "tipoff_utc": pd.Timestamp(game.tipoff_local).tz_localize(ATHENS).tz_convert(UTC),
            "home": game.home,
            "away": game.away,
            "home_score": game.home_score,
            "away_score": game.away_score,
            "played": game.home_score is not None,
            "forfeit": game.forfeit,
            "neutral": False,
            "confirmed_date": game.confirmed_date,
        }
        for r in sorted(rounds, key=lambda r: r.season)
        for game in r.page.games
    ]
    names = {
        team: name
        for r in sorted(rounds, key=lambda r: r.season)
        for game in r.page.games
        for team, name in ((game.home, game.home_name), (game.away, game.away_name))
    }
    games = conform(pd.DataFrame(rows, columns=list(GAMES_SCHEMA.columns)))
    return games, _teams(names)


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", index=False)
