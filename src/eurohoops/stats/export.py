"""Player, team and shot tables -> the JSON files the stats pages read (``web/src/data/stats``).

Layout (every number a raw total; the site derives per-game, per-36 and per-100 rates):

- ``meta.json``: seasons with their coverage, field lists, distance bands, hex size, team names.
- ``players.json``: every player once (id, display name, URL slug, seasons played).
- ``splits.json``: one row per player, season and club (the number he wore most there and his
  totals), for career, best-season and club views.
- ``twins.json``: the Shot Profile Twin pool (player-seasons) and, per player, the closest pool
  rows to his last 5/10/20 games and latest season (``stats/twins.py``).
- ``seasons/{season}/players.json``: one row per player-season: totals for the season and for
  his last 5/10/20 games (a window is left out when he played no more games than it holds),
  and attempts/makes per distance band.
- ``seasons/{season}/teams.json``: team and opponent totals, record, bands taken and allowed.
- ``seasons/{season}/games.json``: the games and every player's game log.
- ``seasons/{season}/shots.json``: hex bins for the league, each team (taken and allowed) and
  each player.
- ``seasons/{season}/attempts.json``: every located attempt (court centimetres, packed flags for
  make, value, fastbreak, second chance, band and period, player and team indices) for the
  Shots explorer's filters.

Team codes are the source codes up to here; ``codes`` renames them for display (DISPLAY_CODES).
"""

import json
import re
import shutil
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eurohoops.ingest.cache import read_cached
from eurohoops.ingest.euroleague import RawGame
from eurohoops.parse.games import ATHENS, build_games_table, build_teams_table
from eurohoops.stats.box import STATS, BoxGames
from eurohoops.stats.shots import BANDS, HEX_RADIUS_M, Shots, band_counts, band_table, hexbins
from eurohoops.stats.twins import twins_payload

STATS_DIR = Path("web/src/data/stats")  # read by the Astro build in web/
WINDOWS = (5, 10, 20)
TEAM_TOTALS = ("tm_fga", "tm_fta", "tm_tov", "tm_poss")
PLAYER_FIELDS = ("gp", "gs", "sec", *STATS, "pm", "poss", "game_sec", *TEAM_TOTALS)
TEAM_FIELDS = ("gp", *STATS, "poss", "game_sec")
LOG_FIELDS = ("game_id", "team", "starter", "sec", *STATS, "pm")
GAME_FIELDS = ("date", "round", "phase", "home", "away", "home_score", "away_score")
# attempts.json ``flags``: bit -> meaning (``band`` takes three bits, ``period`` the rest)
ATTEMPT_BITS = {"made": 0, "three": 1, "fastbreak": 2, "second_chance": 3, "band": 4, "period": 7}


@dataclass(frozen=True)
class Inputs:
    games: pd.DataFrame  # games table (one competition)
    names: Mapping[str, str]  # source team code -> club name
    box: BoxGames
    shots: Shots
    codes: Mapping[str, str]  # source team code -> display code
    live_season: int


def load_cached_games(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Games and teams tables from the cached schedules (no marts; nothing is fetched)."""
    schedules: dict[int, list[RawGame]] = {
        int(path.name.removeprefix("E").split(".")[0]): json.loads(read_cached(path))["data"]
        for path in sorted((raw_dir / "schedule").glob("E*.json.gz"))
    }
    return build_games_table(schedules), build_teams_table(schedules)


def display_name(raw: str) -> str:
    """``"DE COLO, NANDO"`` -> ``"Nando De Colo"``; ``Mc``, ``O'``, ``II``-``IV`` stay capital."""
    last, _, first = raw.partition(",")
    name = " ".join(part.strip() for part in (first, last) if part.strip()).title()
    name = re.sub(r"\b(Ii|Iii|Iv)\b", lambda m: m.group(1).upper(), name)
    return re.sub(r"\bMc([a-z])", lambda m: "Mc" + m.group(1).upper(), name)


def slugify(text: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", ascii_text.lower()).strip("-")


def player_slugs(names: Mapping[str, str]) -> dict[str, str]:
    """Player id -> URL slug from the display name; ids sharing a name all get the id appended."""
    base = {pid: slugify(name) or pid.lower() for pid, name in names.items()}
    taken = pd.Series(base).value_counts()
    return {
        pid: slug if taken[slug] == 1 else f"{slug}-{pid.lower()}" for pid, slug in base.items()
    }


def season_label(season: int) -> str:
    return f"{season}-{(season + 1) % 100:02d}"


def _num(value: Any) -> int | float:
    number = float(value)
    return int(number) if number.is_integer() else round(number, 1)


def _row(values: np.ndarray[Any, Any]) -> list[int | float]:
    return [_num(v) for v in values]


def _player_frame(inputs: Inputs) -> pd.DataFrame:
    """Player game lines with tip-off, starter flag and his team's totals, in game order."""
    teams = inputs.box.teams.assign(tm_fga=lambda t: t["fg2a"] + t["fg3a"]).rename(
        columns={"fta": "tm_fta", "tov": "tm_tov", "poss": "tm_poss"}
    )[["game_id", "team", *TEAM_TOTALS]]
    frame = inputs.box.players.merge(teams, on=["game_id", "team"], validate="many_to_one")
    frame = frame.merge(
        inputs.games[["game_id", "tipoff_utc"]], on="game_id", validate="many_to_one"
    )
    return frame.assign(gp=1, gs=frame["starter"].astype(int)).sort_values(
        ["tipoff_utc", "game_id"], kind="stable"
    )


def _windows(values: np.ndarray[Any, Any]) -> dict[str, list[int | float]]:
    out = {"season": _row(values.sum(axis=0))}
    for n in WINDOWS:
        if len(values) > n:
            out[f"last{n}"] = _row(values[-n:].sum(axis=0))
    return out


def _players_payload(
    season: int, frame: pd.DataFrame, shots: pd.DataFrame, inputs: Inputs, slugs: Mapping[str, str]
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    bands = band_table(shots, "player_id")
    for key, lines in frame.groupby("player_id", sort=False):
        pid = str(key)
        rows.append(
            {
                "id": pid,
                "name": display_name(str(lines["player"].iloc[-1])),
                "slug": slugs[pid],
                "teams": [inputs.codes.get(t, t) for t in dict.fromkeys(lines["team"])],
                "dorsal": str(lines["dorsal"].iloc[-1]),
                "totals": _windows(lines[list(PLAYER_FIELDS)].to_numpy(dtype=float)),
                "bands": bands.get(pid, {b: [0, 0] for b in BANDS}),
            }
        )
    rows.sort(key=lambda r: (-r["totals"]["season"][PLAYER_FIELDS.index("pts")], r["id"]))
    return {"season": season, "fields": list(PLAYER_FIELDS), "bands": list(BANDS), "players": rows}


def _splits_payload(frame: pd.DataFrame, inputs: Inputs) -> dict[str, Any]:
    """One row per player, season and club: the number he wore most there, and his totals."""
    rows: list[list[Any]] = []
    for (pid, season, team), lines in frame.groupby(["player_id", "season", "team"], sort=False):
        worn = lines["dorsal"].value_counts(sort=False)  # first-worn order breaks ties
        rows.append(
            [
                str(pid),
                int(str(season)),
                inputs.codes.get(str(team), str(team)),
                str(worn.idxmax()),
                _row(lines[list(PLAYER_FIELDS)].to_numpy(dtype=float).sum(axis=0)),
            ]
        )
    rows.sort(key=lambda r: (r[0], r[1]))  # stable: clubs stay in the order he joined them
    return {"fields": list(PLAYER_FIELDS), "rows": rows}


def _teams_payload(
    season: int, teams: pd.DataFrame, shots: pd.DataFrame, inputs: Inputs
) -> dict[str, Any]:
    table = teams.assign(gp=1)
    opp = table[["game_id", "team", *TEAM_FIELDS]].rename(columns={"team": "opponent"})
    paired = table.merge(
        opp, on=["game_id", "opponent"], suffixes=("", "_opp"), validate="one_to_one"
    )
    rows: list[dict[str, Any]] = []
    taken, allowed = band_table(shots, "team"), band_table(shots, "opponent")
    zero = {b: [0, 0] for b in BANDS}
    for key, lines in paired.groupby("team"):
        code = str(key)
        rows.append(
            {
                "code": inputs.codes.get(code, code),
                "name": inputs.names.get(code, code),
                "w": int(lines["won"].sum()),
                "l": int((~lines["won"]).sum()),
                "totals": _row(lines[list(TEAM_FIELDS)].to_numpy(dtype=float).sum(axis=0)),
                "opp": _row(
                    lines[[f"{f}_opp" for f in TEAM_FIELDS]].to_numpy(dtype=float).sum(axis=0)
                ),
                "bands": taken.get(code, zero),
                "bands_allowed": allowed.get(code, zero),
            }
        )
    rows.sort(key=lambda r: (-r["w"], r["l"], r["code"]))
    league = _row(table[list(TEAM_FIELDS)].to_numpy(dtype=float).sum(axis=0))
    return {
        "season": season,
        "fields": list(TEAM_FIELDS),
        "bands": list(BANDS),
        "league": {"totals": league, "bands": band_counts(shots)},
        "teams": rows,
    }


def _games_payload(
    season: int, frame: pd.DataFrame, games: pd.DataFrame, inputs: Inputs
) -> dict[str, Any]:
    played = games[games["played"]]
    index = {
        str(g["game_id"]): [
            g["tipoff_utc"].tz_convert(ATHENS).strftime("%Y-%m-%d"),
            int(g["round"]),
            str(g["phase"]),
            inputs.codes.get(str(g["home"]), str(g["home"])),
            inputs.codes.get(str(g["away"]), str(g["away"])),
            int(g["home_score"]),
            int(g["away_score"]),
        ]
        for g in played.to_dict("records")
    }
    logs: dict[str, list[list[Any]]] = {}
    lines = frame.assign(
        team=frame["team"].map(lambda t: inputs.codes.get(t, t)),
        starter=frame["starter"].astype(int),
    )
    for pid, game_id, team, *numbers in lines[["player_id", *LOG_FIELDS]].itertuples(index=False):
        logs.setdefault(str(pid), []).append([game_id, team, *(int(v) for v in numbers)])
    return {
        "season": season,
        "game_fields": list(GAME_FIELDS),
        "log_fields": list(LOG_FIELDS),
        "games": index,
        "logs": logs,
    }


def _shots_payload(season: int, shots: pd.DataFrame, inputs: Inputs) -> dict[str, Any]:
    taken, allowed = hexbins(shots, "team"), hexbins(shots, "opponent")
    return {
        "season": season,
        "hex_radius_m": HEX_RADIUS_M,
        "cell_fields": ["q", "r", "att", "made", "pts"],
        "league": hexbins(shots).get("", []),
        "teams": {
            inputs.codes.get(t, t): {"taken": taken[t], "allowed": allowed.get(t, [])}
            for t in sorted(taken)
        },
        "players": hexbins(shots, "player_id"),
    }


def attempt_flags(shots: pd.DataFrame) -> pd.Series:
    """Each shot packed into one integer (``ATTEMPT_BITS``), so the explorer can filter it."""
    parts = {
        "made": shots["made"],
        "three": shots["value"] == 3,
        "fastbreak": shots["fastbreak"],
        "second_chance": shots["second_chance"],
        "band": shots["band"].map({b: i for i, b in enumerate(BANDS)}),
        "period": shots["period"],
    }
    packed = pd.Series(0, index=shots.index, dtype="int64")
    for name, values in parts.items():
        packed += values.astype("int64") * 2 ** ATTEMPT_BITS[name]
    return packed


def _attempts_payload(season: int, shots: pd.DataFrame, inputs: Inputs) -> dict[str, Any]:
    """Every located attempt as parallel columns: coordinates in centimetres, flags, indices."""
    players = sorted(set(shots["player_id"]))
    teams = sorted({inputs.codes.get(t, t) for t in (*shots["team"], *shots["opponent"])})
    player_at = {p: i for i, p in enumerate(players)}
    team_at = {t: i for i, t in enumerate(teams)}

    def team_index(codes: pd.Series) -> list[int]:
        return [team_at[inputs.codes.get(t, t)] for t in codes]

    return {
        "season": season,
        "bits": ATTEMPT_BITS,
        "players": players,
        "teams": teams,
        "x": [int(v) for v in (shots["x"] * 100).round()],
        "y": [int(v) for v in (shots["y"] * 100).round()],
        "flags": [int(v) for v in attempt_flags(shots)],
        "player": [player_at[p] for p in shots["player_id"]],
        "team": team_index(shots["team"]),
        "opp": team_index(shots["opponent"]),
    }


def build_payloads(inputs: Inputs, now: datetime) -> dict[str, dict[str, Any]]:
    """Relative path -> JSON payload of every file in the stats directory."""
    frame = _player_frame(inputs)
    names = dict(zip(frame["player_id"], frame["player"].map(display_name), strict=True))
    slugs = player_slugs(names)  # the last spelling of a name wins, as in the season rows
    files: dict[str, dict[str, Any]] = {}
    seasons: list[dict[str, Any]] = []
    missing = inputs.box.missing.groupby("season").size()
    coverage = {int(r["season"]): r for r in inputs.shots.coverage.to_dict("records")}
    for season in sorted(int(s) for s in frame["season"].unique()):
        lines = frame[frame["season"] == season]
        shots = inputs.shots.table[inputs.shots.table["season"] == season]
        games = inputs.games[inputs.games["season"] == season]
        base = f"seasons/{season}"
        files[f"{base}/players.json"] = _players_payload(season, lines, shots, inputs, slugs)
        files[f"{base}/teams.json"] = _teams_payload(
            season, inputs.box.teams[inputs.box.teams["season"] == season], shots, inputs
        )
        files[f"{base}/games.json"] = _games_payload(season, lines, games, inputs)
        files[f"{base}/shots.json"] = _shots_payload(season, shots, inputs)
        files[f"{base}/attempts.json"] = _attempts_payload(season, shots, inputs)
        cover = coverage.get(season)
        seasons.append(
            {
                "season": season,
                "label": season_label(season),
                "live": season == inputs.live_season,
                "games": int(lines["game_id"].nunique()),
                "games_without_box": int(missing.get(season, 0)),
                "games_with_shots": 0 if cover is None else int(cover["games"]),
                "shots": 0 if cover is None else int(cover["shots"]),
                "shots_unplaced": 0 if cover is None else int(cover["unplaced"]),
                "coords_validated": bool(cover is not None and cover["validated"]),
                "players": int(lines["player_id"].nunique()),
            }
        )
    played = frame.groupby("player_id", sort=False)["season"].unique()
    files["players.json"] = {
        "players": sorted(
            (
                {
                    "id": str(pid),
                    "name": names[str(pid)],
                    "slug": slugs[str(pid)],
                    "seasons": [int(s) for s in ss],
                }
                for pid, ss in played.items()
            ),
            key=lambda p: (p["slug"], p["id"]),
        )
    }
    files["splits.json"] = _splits_payload(frame, inputs)
    files["twins.json"] = twins_payload(frame, inputs.shots.table, inputs.codes)
    codes = sorted(set(inputs.box.teams["team"]))
    files["meta.json"] = {
        "generated_at": now.strftime("%Y-%m-%dT%H:%MZ"),
        "hex_radius_m": HEX_RADIUS_M,
        "bands": list(BANDS),
        "windows": [f"last{n}" for n in WINDOWS],
        "player_fields": list(PLAYER_FIELDS),
        "team_fields": list(TEAM_FIELDS),
        "teams": {inputs.codes.get(c, c): inputs.names.get(c, c) for c in codes},
        "seasons": seasons,
    }
    return files


def write_stats(out_dir: Path, files: Mapping[str, Mapping[str, Any]]) -> None:
    """Replace the stats directory's contents with ``files`` (compact JSON, LF endings)."""
    if (out_dir / "seasons").exists():
        shutil.rmtree(out_dir / "seasons")
    for relative, payload in files.items():
        path = out_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        path.write_text(text + "\n", encoding="utf-8", newline="\n")
