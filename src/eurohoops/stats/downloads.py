"""The Downloads page's files: per-season gzipped CSVs of games, team lines, player lines, shots.

``build_downloads`` is pure (same ``Inputs`` and ``now`` -> the same bytes): the window is the one
``build_payloads`` publishes (the latest ``SITE_SEASONS`` seasons), team codes are the display
codes, dates the Athens local date the site shows. ``COLUMNS`` is the one place the columns and
their meanings are written; the schemas that check each frame and ``columns.json`` (the page's data
dictionary) both come from it. ``write_downloads`` replaces the directory's files.

Files, relative to ``web/public/downloads``: ``<dataset>-<season>.csv.gz``, ``manifest.json``
(``built`` and one entry per file with rows, bytes and sha256) and ``columns.json``.
"""

import gzip
import hashlib
import io
import json
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pandera.pandas as pa

from eurohoops.config import SITE_SEASONS
from eurohoops.parse.games import ATHENS
from eurohoops.stats.box import STATS
from eurohoops.stats.export import Inputs, _player_frame, display_name, season_label
from eurohoops.stats.shots import BANDS

DOWNLOADS_DIR = Path("web/public/downloads")  # a build output, served at /downloads/
DATASETS = ("games", "team_games", "player_box", "shots")

Column = tuple[str, str, str]  # name, dtype, meaning

_SEASON: tuple[Column, Column] = (
    ("season", "int64", "Season, named by the year it starts (2024 is 2024-25)."),
    ("season_label", "str", "Season as written on the site, such as 2024-25."),
)
_DATE: Column = ("date", "str", "Local date of the game in Athens, YYYY-MM-DD.")
_GAME: Column = ("game_id", "str", "EuroLeague game identifier.")
_ROUND: Column = ("round", "int64", "Round number within the season.")
_PHASE: Column = (
    "phase",
    "str",
    "Part of the season: RS regular season, PI play-in, PO playoffs, FF Final Four.",
)
_TEAM_LINE: tuple[Column, ...] = (
    ("team", "str", "Team code as shown on the site."),
    ("opponent", "str", "Opponent's team code."),
    ("venue", "str", "home, away or neutral, for this team."),
    ("won", "int64", "1 if this team won, else 0."),
)
_STAT_MEANING = {
    "pts": "Points.",
    "fg2m": "Two-point field goals made.",
    "fg2a": "Two-point field goals attempted.",
    "fg3m": "Three-point field goals made.",
    "fg3a": "Three-point field goals attempted.",
    "ftm": "Free throws made.",
    "fta": "Free throws attempted.",
    "oreb": "Offensive rebounds.",
    "dreb": "Defensive rebounds.",
    "ast": "Assists.",
    "stl": "Steals.",
    "tov": "Turnovers.",
    "blk": "Shots blocked.",
    "blka": "Own shots blocked.",
    "pf": "Personal fouls committed.",
    "fd": "Fouls drawn.",
    "pir": "Performance Index Rating, the EuroLeague's own valuation figure.",
}
_STATS: tuple[Column, ...] = tuple((s, "int64", _STAT_MEANING[s]) for s in STATS)

COLUMNS: dict[str, tuple[Column, ...]] = {
    "games": (
        *_SEASON,
        _GAME,
        _DATE,
        _ROUND,
        _PHASE,
        ("home", "str", "Home team code."),
        ("away", "str", "Away team code."),
        ("home_score", "int64", "Home team's final score."),
        ("away_score", "int64", "Away team's final score."),
        ("neutral", "int64", "1 if played on neutral ground, else 0."),
    ),
    "team_games": (
        *_SEASON,
        _GAME,
        _DATE,
        _ROUND,
        _PHASE,
        *_TEAM_LINE,
        ("game_sec", "int64", "Seconds of game time recorded for the team."),
        ("poss", "float64", "Possessions, estimated from the box score, averaged over both teams."),
        *_STATS,
    ),
    "player_box": (
        *_SEASON,
        _GAME,
        _DATE,
        *_TEAM_LINE,
        ("player_id", "str", "EuroLeague player identifier."),
        ("player", "str", "Player's name, First Last."),
        ("dorsal", "str", "Shirt number worn in the game."),
        ("starter", "int64", "1 if in the starting five, else 0."),
        ("sec", "int64", "Seconds played."),
        *_STATS,
        ("pm", "int64", "Plus-minus: point margin while the player was on court."),
        ("poss", "float64", "Team possessions while the player was on court, estimated."),
    ),
    "shots": (
        *_SEASON,
        _GAME,
        _DATE,
        ("period", "int64", "Period: 1-4 are quarters, 5 and up overtime."),
        ("team", "str", "Shooter's team code as shown on the site."),
        ("opponent", "str", "Defending team's code."),
        ("player_id", "str", "EuroLeague player identifier of the shooter."),
        ("player", "str", "Shooter's name, First Last."),
        ("value", "int64", "Points the shot is worth if made: 2 or 3."),
        ("made", "int64", "1 if made, else 0."),
        ("x", "float64", "Metres across the court, the basket at 0."),
        ("y", "float64", "Metres out from the basket along the court."),
        ("band", "str", f"Distance band: {', '.join(BANDS)}."),
        ("fastbreak", "int64", "1 if a fast-break shot, else 0."),
        ("second_chance", "int64", "1 if taken after an offensive rebound, else 0."),
    ),
}
SCHEMAS = {
    name: pa.DataFrameSchema({c: pa.Column(t) for c, t, _ in cols}, strict=True, ordered=True)
    for name, cols in COLUMNS.items()
}


def _date(tipoff: pd.Series) -> pd.Series:
    return pd.Series(tipoff.dt.tz_convert(ATHENS).dt.strftime("%Y-%m-%d"), index=tipoff.index)


def _frames(inputs: Inputs) -> dict[str, pd.DataFrame]:
    """The four datasets, every window season together, in file column order and row order."""
    career = _player_frame(inputs)
    window = sorted(int(s) for s in career["season"].unique())[-SITE_SEASONS:]
    labels = {s: season_label(s) for s in window}

    def code(col: pd.Series) -> pd.Series:
        return col.map(lambda t: inputs.codes.get(t, t))

    def finish(name: str, frame: pd.DataFrame, order: list[str]) -> pd.DataFrame:
        frame = frame.assign(season_label=frame["season"].map(labels))
        frame = frame.sort_values(order, kind="stable").reset_index(drop=True)
        return SCHEMAS[name].validate(frame[[c for c, _, _ in COLUMNS[name]]])

    def team_cols(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.assign(
            team=code(frame["team"]), opponent=code(frame["opponent"]), won=frame["won"].astype(int)
        )

    games = inputs.games[inputs.games["season"].isin(window) & inputs.games["played"]]
    games = games.assign(date=_date(games["tipoff_utc"]))
    when = games[["game_id", "date", "round", "phase"]]
    played = games.assign(
        home=code(games["home"]),
        away=code(games["away"]),
        neutral=games["neutral"].astype(int),
        home_score=games["home_score"].astype("int64"),
        away_score=games["away_score"].astype("int64"),
    )
    out = {"games": finish("games", played, ["season", "date", "game_id"])}

    teams = inputs.box.teams[inputs.box.teams["season"].isin(window)]
    teams = team_cols(teams.merge(when, on="game_id", validate="many_to_one"))
    out["team_games"] = finish("team_games", teams, ["season", "date", "game_id", "team"])

    names = dict(zip(career["player_id"], career["player"].map(display_name), strict=True))
    lines = career[career["season"].isin(window)]
    lines = team_cols(lines.merge(when, on="game_id", validate="many_to_one")).assign(
        player=lambda f: f["player_id"].map(names), starter=lambda f: f["starter"].astype(int)
    )
    order = ["season", "date", "game_id", "team", "player_id"]
    out["player_box"] = finish("player_box", lines, order)

    shots = inputs.shots.table[inputs.shots.table["season"].isin(window)]
    shots = shots.merge(when[["game_id", "date"]], on="game_id", validate="many_to_one")
    shots = shots.assign(
        team=code(shots["team"]),
        opponent=code(shots["opponent"]),
        player=shots["player_id"].map(names).fillna(shots["player_id"]),
        made=shots["made"].astype(int),
        fastbreak=shots["fastbreak"].astype(int),
        second_chance=shots["second_chance"].astype(int),
        x=shots["x"].round(2),
        y=shots["y"].round(2),
    )
    out["shots"] = finish("shots", shots, ["season", "date", "game_id"])  # feed order in a game
    return out


def csv_gz(frame: pd.DataFrame) -> bytes:
    """CSV with LF line ends in a gzip with no timestamp or name: the same frame, the same bytes."""
    text = frame.to_csv(index=False, lineterminator="\n", float_format="%.2f")
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, compresslevel=9, mtime=0) as gz:
        gz.write(text.encode("utf-8"))
    return buffer.getvalue()


def _json(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=1) + "\n").encode("utf-8")


def build_downloads(inputs: Inputs, now: datetime) -> dict[str, bytes]:
    """Relative path (``downloads/...`` for CSVs, bare for the two JSON files) -> bytes."""
    files: dict[str, bytes] = {}
    entries: list[dict[str, Any]] = []
    for name, frame in _frames(inputs).items():
        for season, part in frame.groupby("season", sort=True):
            number = int(str(season))
            path = f"downloads/{name}-{number}.csv.gz"
            data = csv_gz(part)
            files[path] = data
            entries.append(
                {
                    "dataset": name,
                    "season": number,
                    "path": path,
                    "rows": len(part),
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
    entries.sort(key=lambda e: (-e["season"], DATASETS.index(e["dataset"])))
    files["manifest.json"] = _json({"built": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "files": entries})
    dictionary = {
        name: [{"name": c, "meaning": m} for c, _, m in cols] for name, cols in COLUMNS.items()
    }
    files["columns.json"] = _json({"datasets": dictionary})
    return files


def write_downloads(out_dir: Path, files: Mapping[str, bytes]) -> None:
    """Replace the files in ``out_dir`` (flat): the built ones are written, any other is removed."""
    out_dir.mkdir(parents=True, exist_ok=True)
    keep = {Path(p).name for p in files}
    for old in out_dir.iterdir():
        if old.is_file() and old.name not in keep:
            old.unlink()
    for relative, data in files.items():
        (out_dir / Path(relative).name).write_bytes(data)
