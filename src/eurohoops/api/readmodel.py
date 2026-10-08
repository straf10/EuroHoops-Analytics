"""Read model of the read-only API: the marts, prediction logs and committed reports as JSON-ready
dicts (weeks 16-18, L5). ``api/app.py`` only serves what these functions return.

Rules
-----
- Nothing is fitted or simulated here. A route reads a committed report, a log, a mart table, or
  calls a pure function the pipeline already owns (``publish.site_data``, ``stats.export
  .build_payloads``, ``models.player_seasons.build_player_seasons``, Elo's ``season_ratings``).
- Read-only: DuckDB is only opened through ``marts.read_games`` / ``read_teams`` / ``read_table``
  (``read_only=True``); no function writes a file.
- A ``Store`` is a snapshot. It holds the root of a data tree (repo-root-relative paths from
  ``eurohoops.config``) and memoises the expensive loads (marts, box scores, shots, the stats
  payloads, the player-season frame) for its lifetime, so one process builds each once. Logs and
  small reports are read on every call. Start a new ``Store`` to see newer marts.
- ``NotFound`` carries the reason a route answers 404 with; a malformed committed report is a bug
  and raises its own error (pandera / KeyError), not a 404.
- Source team codes everywhere (``team.code``); ``team.display_code`` is the site's real-life code
  (``publish.DISPLAY_CODES``). No birth date or age is read or returned.

Team factors (``team_factors``)
-------------------------------
Per season, from the team's box-score sums: eFG% = (fg2m + 1.5 fg3m) / fga, TOV% = tov / poss,
ORB% = oreb / (oreb + opponent dreb), FT rate = fta / fga. EuroLeague sums the cached box scores'
team totals rows. GBL takes fga, fta, oreb, dreb, tov and possessions from the ``team_games`` mart
and the made field goals from the player lines (player rows sum to the totals row for shots, not
for team rebounds and turnovers). Schedule strength is the mean opponent rating over the team's
scheduled games of the season (played and remaining), the rating being Elo as published (the
live backtest's frozen parameters, replayed from its first warm-up season) at the end of the
replay through that season. An input that is missing gives ``value: null`` and a ``reason``.
"""

import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any, TypeVar, cast

import numpy as np
import pandas as pd
import pandera.pandas as pa

from eurohoops.config import (
    COMPETITIONS,
    EUROLEAGUE,
    GBL,
    LIVE_SEASON,
    M2_PLAYERS_REPORT,
    M3_PLAYERS_REPORT,
    M4,
    M6_BOARD,
    M6_PROJECTIONS,
    M6_SIMILARITY,
    M7,
    MART_PATH,
    PLAYER_XWALK_FILE,
    SIM_UNGATED,
    Competition,
)
from eurohoops.eval.backtest import TunedModel, load_tuned_model
from eurohoops.live_m6 import person_names
from eurohoops.logs import TIME_FORMAT
from eurohoops.marts import read_games, read_table, read_teams
from eurohoops.models.elo import prepare, season_ratings
from eurohoops.models.player_seasons import (
    COUNT_COLUMNS,
    COUNT_STATS,
    PCT_STATS,
    build_player_seasons,
    person_ids,
    rate_table,
    read_xwalk_file,
)
from eurohoops.parse.gbl_box_lines import build_gbl_player_games
from eurohoops.parse.schemas import validated
from eurohoops.publish import DISPLAY_CODES, Section, site_data
from eurohoops.stats.box import BoxGames, build_box_games
from eurohoops.stats.export import Inputs, build_payloads, load_cached_games
from eurohoops.stats.shots import build_shots

T = TypeVar("T")

SPIEGELHALTER_LIMIT = (
    1.96  # the M7 gate's bound on the pooled |z| (config.M7, reports/m7_progress.md)
)

PREDICTION_LOG_SCHEMA = pa.DataFrameSchema(
    {
        "game_id": pa.Column(str),
        "season": pa.Column("int64"),
        "round": pa.Column("int64"),
        "phase": pa.Column(str),
        "tipoff_utc": pa.Column(str),
        "home": pa.Column(str),
        "away": pa.Column(str),
        "p_home": pa.Column("float64", pa.Check.in_range(0.0, 1.0)),
        "exp_margin": pa.Column("float64"),
        "model": pa.Column(str),
        "model_version": pa.Column(str),
        "predicted_at_utc": pa.Column(str),
    },
    strict=False,  # M1 and M5 rows carry extra forecast columns
)
M3_ROWS_SCHEMA = pa.DataFrameSchema(
    {
        "player_id": pa.Column(str),
        "season": pa.Column("int64"),
        "total": pa.Column("float64"),
        "sd_total": pa.Column("float64"),
        "minutes": pa.Column("float64", pa.Check.ge(0.0)),
        "games": pa.Column("int64", pa.Check.ge(0)),
    },
    strict=False,
)
M2_ROWS_SCHEMA = pa.DataFrameSchema(
    {
        "shooter": pa.Column(str),
        "season": pa.Column("int64"),
        "split": pa.Column(str),
        "fga": pa.Column("int64", pa.Check.ge(0)),
        "shrunk": pa.Column("float64"),
    },
    strict=False,
)
PROJECTION_ROWS_SCHEMA = pa.DataFrameSchema(
    {
        "person_id": pa.Column(str),
        "competition": pa.Column(str, pa.Check.isin(list(COMPETITIONS))),
        "team": pa.Column(str),
        "name": pa.Column(str),
        "debut_season": pa.Column("int64"),
        "exposure": pa.Column("float64", pa.Check.ge(0.0)),
        "n_seasons": pa.Column("int64", pa.Check.ge(0)),
    },
    strict=False,
)
UNDERVALUED_ROWS_SCHEMA = pa.DataFrameSchema(
    {
        "person_id": pa.Column(str),
        "team": pa.Column(str),
        "name": pa.Column(str),
        "seasons_since_debut": pa.Column("int64", pa.Check.ge(0)),
        "minutes": pa.Column("float64", pa.Check.ge(0.0)),
        "projected_spm": pa.Column("float64"),
        "reason": pa.Column(str),
    },
    strict=False,
)
BOARD_ROWS_SCHEMA = pa.DataFrameSchema(
    {
        "person_id": pa.Column(str),
        "competition": pa.Column(str, pa.Check.isin(list(COMPETITIONS))),
        "season": pa.Column("int64"),
        "dimension": pa.Column(str),
        "n": pa.Column("float64", pa.Check.ge(0.0)),
        "name": pa.Column(str),
        "team": pa.Column(str),
    },
    strict=False,  # observed, expected, gap, se, z, ... belong to models/board.py's schema
)
SIMILAR_ROWS_SCHEMA = pa.DataFrameSchema(
    {
        "rank": pa.Column("int64", pa.Check.ge(1)),
        "person_id": pa.Column(str),
        "competition": pa.Column(str, pa.Check.isin(list(COMPETITIONS))),
        "season": pa.Column("int64"),
        "name": pa.Column(str),
        "score": pa.Column("float64"),
    },
    strict=False,
)
TEAM_LINES_SCHEMA = pa.DataFrameSchema(
    {
        "season": pa.Column("int64"),
        "game_id": pa.Column(str),
        "team": pa.Column(str),
        "opponent": pa.Column(str),
        **{
            c: pa.Column("int64", pa.Check.ge(0))
            for c in ("fg2m", "fg3m", "fga", "fta", "oreb", "dreb", "tov")
        },
        "poss": pa.Column("float64", pa.Check.gt(0.0)),
    },
    strict=True,
)


class NotFound(LookupError):
    """A route's 404: ``reason`` is the body's ``detail``; ``extra`` is merged into the body."""

    def __init__(self, reason: str, **extra: Any) -> None:
        super().__init__(reason)
        self.reason = reason
        self.extra = extra


@dataclass(frozen=True)
class Store:
    """Where the data lives, and what has been loaded from it.

    ``root`` holds the repository layout (``data/``, ``reports/``, ``predictions/``); tests point
    it at a temporary tree. With ``from_cache`` the ``/stats`` routes read games from the cached
    schedules and never open DuckDB (the daily workflow's mode); other routes still read the marts.
    """

    root: Path = Path()
    from_cache: bool = False
    _memo: dict[str, Any] = field(default_factory=dict, init=False, repr=False, compare=False)
    _lock: Any = field(default_factory=RLock, init=False, repr=False, compare=False)

    def path(self, relative: Path) -> Path:
        return self.root / relative

    def memo(self, key: str, make: Callable[[], T]) -> T:
        with self._lock:
            if key not in self._memo:
                self._memo[key] = make()
            return cast(T, self._memo[key])


def jsonable(value: Any) -> Any:
    """Plain Python for ``json.dumps``: numpy and pandas scalars unwrapped, NaN and NaT -> None."""
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple | np.ndarray):
        return [jsonable(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.tz_convert("UTC").strftime(TIME_FORMAT)
    if isinstance(value, float | np.floating):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, np.generic):
        return value.item()
    return None if value is pd.NaT else value


def _stamp(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime(TIME_FORMAT)


def _competition(name: str) -> Competition:
    if name not in COMPETITIONS:
        raise NotFound(f"unknown competition '{name}'; expected one of {', '.join(COMPETITIONS)}")
    return COMPETITIONS[name]


def _mart(store: Store) -> Path:
    path = store.path(MART_PATH)
    if not path.exists():
        raise NotFound(f"the marts are not built: {MART_PATH.as_posix()} is missing")
    return path


def _report(store: Store, relative: Path) -> dict[str, Any] | None:
    path = store.path(relative)
    if not path.exists():
        return None
    return store.memo(f"report:{relative}", lambda: _load(path))  # read once per Store


def _load(path: Path) -> dict[str, Any]:
    report: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return report


def _required(store: Store, relative: Path, what: str) -> dict[str, Any]:
    report = _report(store, relative)
    if report is None:
        raise NotFound(f"{relative.as_posix()} not found: {what} has not been written")
    return report


def _check(rows: list[Any], schema: pa.DataFrameSchema) -> None:
    """Validate a report's row list against ``schema`` (the rows themselves are served as is)."""
    validated(pd.DataFrame(rows, columns=list(schema.columns)), schema)


# ---- marts, logs ----


def _games(store: Store, comp: Competition) -> pd.DataFrame:
    return store.memo(f"games:{comp.name}", lambda: read_games(_mart(store), comp.name))


def _names(store: Store, comp: Competition) -> dict[str, str]:
    return store.memo(
        f"names:{comp.name}",
        lambda: dict(read_teams(_mart(store), comp.name).itertuples(index=False)),
    )


def _team(comp: Competition, code: str, names: Mapping[str, str]) -> dict[str, str]:
    shown = DISPLAY_CODES.get(comp.name, {}).get(code, code)
    return {"code": code, "display_code": shown, "name": names.get(code, code)}


def _read_log(store: Store, relative: Path | None) -> pd.DataFrame:
    """A prediction log (Elo, M1 or M5); empty when the file is missing or has no rows."""
    path = None if relative is None else store.path(relative)
    if path is None or not path.exists() or not path.stat().st_size:
        return validated(
            pd.DataFrame(columns=list(PREDICTION_LOG_SCHEMA.columns)), PREDICTION_LOG_SCHEMA
        )
    return validated(pd.read_csv(path, dtype={"game_id": str}), PREDICTION_LOG_SCHEMA)


def _ordered(log: pd.DataFrame) -> pd.DataFrame:
    """Rows with ``late`` (stamped at or after tip-off); the pre-registered row of a game is its
    first on-time row, else its first row (the scorecard's rule)."""
    late = pd.to_datetime(log["predicted_at_utc"], utc=True) >= pd.to_datetime(
        log["tipoff_utc"], utc=True
    )
    return log.assign(late=late).sort_values(["late", "predicted_at_utc"], kind="stable")


def _forecast(row: Mapping[Any, Any]) -> dict[str, Any]:
    """A log row without the game's own columns."""
    skip = {"game_id", "season", "round", "phase", "tipoff_utc", "home", "away"}
    return {str(k): jsonable(v) for k, v in row.items() if k not in skip}


def _live_model(store: Store, comp: Competition) -> TunedModel:
    report = comp.live_backtest.report
    if not store.path(report).exists():
        raise NotFound(
            f"{report.as_posix()} not found: the {comp.name} Elo has no tuned parameters"
        )
    return load_tuned_model(store.path(report))


def _elo(store: Store, comp: Competition, season: int) -> dict[str, tuple[float, float]]:
    """Each ``season`` team's (rating, rating at the season's start) as ``publish`` shows them:
    the live Elo replayed from the backtest's first warm-up season through ``season``."""

    def make() -> dict[str, tuple[float, float]]:
        first = comp.live_backtest.warmup[0]
        if season < first:
            raise NotFound(f"the {comp.name} Elo is replayed from {first}; no rating for {season}")
        games = _games(store, comp)
        window = games[games["season"].between(first, season)]
        return season_ratings(prepare(window), _live_model(store, comp).params, season)

    return store.memo(f"elo:{comp.name}:{season}", make)


# ---- site ----


def site(store: Store, now: datetime) -> dict[str, Any]:
    """``site.json``: the payload ``publish.site_data`` builds from the logs, scorecards, tuned
    Elo parameters and games of both competitions (as ``eurohoops publish`` assembles it)."""
    sections = []
    for title, comp in (("EuroLeague", EUROLEAGUE), ("Greek Basket League", GBL)):
        scorecard = _required(store, comp.scorecard, f"the {comp.name} live scorecard")
        backtest = _required(store, comp.live_backtest.report, f"the {comp.name} Elo backtest")
        log = (
            pd.read_csv(store.path(comp.prediction_log), dtype={"game_id": str})
            if store.path(comp.prediction_log).exists()
            else pd.DataFrame()
        )
        sections.append(
            Section(
                key=comp.name,
                title=title,
                log=log,
                scorecard=scorecard,
                backtest=backtest,
                games=_games(store, comp),
                names=_names(store, comp),
                model=_live_model(store, comp),
                season=LIVE_SEASON,
                replay_from=comp.live_backtest.warmup[0],
            )
        )
    return site_data(sections, now)


# ---- stats ----


def _el_games(store: Store) -> tuple[pd.DataFrame, pd.DataFrame]:
    """EuroLeague games and teams: the cached schedules (``from_cache``) or the marts."""

    def make() -> tuple[pd.DataFrame, pd.DataFrame]:
        if store.from_cache:
            return load_cached_games(store.path(EUROLEAGUE.raw_dir))
        return _games(store, EUROLEAGUE), read_teams(_mart(store), EUROLEAGUE.name)

    return store.memo("el_games", make)


def _el_box(store: Store) -> BoxGames:
    return store.memo(
        "el_box",
        lambda: build_box_games(store.path(EUROLEAGUE.raw_dir), _el_games(store)[0]),
    )


def _stats_inputs(store: Store) -> Inputs:
    games, teams = _el_games(store)
    return Inputs(
        games=games,
        names=dict(teams.itertuples(index=False)),
        box=_el_box(store),
        shots=build_shots(store.path(EUROLEAGUE.raw_dir), games),
        codes=DISPLAY_CODES[EUROLEAGUE.name],
        live_season=LIVE_SEASON,
    )


def stats_payloads(store: Store, now: datetime) -> dict[str, dict[str, Any]]:
    """Every stats file by relative path (``stats.export.build_payloads``), built once per Store;
    ``meta.json`` carries the time of that build."""
    return store.memo("stats", lambda: build_payloads(_stats_inputs(store), now))


def stats_index(store: Store, now: datetime) -> dict[str, Any]:
    return {"files": sorted(stats_payloads(store, now))}


def stats_file(store: Store, path: str, now: datetime) -> dict[str, Any]:
    files = stats_payloads(store, now)
    if path not in files:
        raise NotFound(f"no stats file '{path}'; GET /stats lists the {len(files)} files")
    return files[path]


# ---- games, predictions ----


def upcoming_games(store: Store, competition: str, now: datetime) -> dict[str, Any]:
    """The live season's unplayed games that tip off after ``now``, each with its pre-registered
    Elo forecast when one is logged."""
    comp = _competition(competition)
    games = _games(store, comp)
    names = _names(store, comp)
    log = _read_log(store, comp.prediction_log)
    first = _ordered(log).drop_duplicates("game_id", keep="first").set_index("game_id")
    ahead = games[(games["season"] == LIVE_SEASON) & ~games["played"] & (games["tipoff_utc"] > now)]
    rows = []
    for g in ahead.to_dict("records"):
        logged = first.loc[g["game_id"]].to_dict() if g["game_id"] in first.index else None
        rows.append(
            {
                "game_id": g["game_id"],
                "round": int(g["round"]),
                "phase": g["phase"],
                "tipoff_utc": jsonable(g["tipoff_utc"]),
                "tipoff_time_confirmed": bool(g["confirmed_date"]),
                "home": _team(comp, g["home"], names),
                "away": _team(comp, g["away"], names),
                "prediction": None if logged is None else _forecast(logged),
            }
        )
    return {
        "competition": comp.name,
        "season": LIVE_SEASON,
        "as_of": _stamp(now),
        "games": rows,
    }


def prediction(store: Store, game_id: str) -> dict[str, Any]:
    """Every logged forecast of one game, per model: the pre-registered row (the first stamped
    before tip-off; ``late`` when none was) and the rows logged after it."""
    for comp in COMPETITIONS.values():
        logs = {
            "elo": comp.prediction_log,
            "m1": comp.m1_prediction_log,
            "m5": comp.m5_prediction_log,
        }
        models: dict[str, Any] = {}
        head: Mapping[Any, Any] | None = None
        for model, relative in logs.items():
            log = _read_log(store, relative)
            rows = _ordered(log[log["game_id"] == game_id])
            if rows.empty:
                continue
            records = rows.to_dict("records")
            head = head or records[0]
            models[model] = {
                "pre_registered": {**_forecast(records[0]), "late": bool(records[0]["late"])},
                "later": [{**_forecast(r), "late": bool(r["late"])} for r in records[1:]],
            }
        if head is not None:
            return {
                "game_id": game_id,
                "competition": comp.name,
                "season": int(head["season"]),
                "round": int(head["round"]),
                "phase": head["phase"],
                "tipoff_utc": head["tipoff_utc"],
                "home": head["home"],
                "away": head["away"],
                "result": _result(store, comp, game_id),
                "models": models,
            }
    raise NotFound(f"no prediction is logged for game '{game_id}'")


def _result(store: Store, comp: Competition, game_id: str) -> dict[str, Any] | None:
    if not store.path(MART_PATH).exists():
        return None
    games = _games(store, comp)
    played = games[(games["game_id"] == game_id) & games["played"]]
    if played.empty:
        return None
    g = played.iloc[0]
    return {
        "home_score": int(g["home_score"]),
        "away_score": int(g["away_score"]),
        "forfeit": bool(g["forfeit"]),
    }


# ---- teams ----


def _known_team(store: Store, comp: Competition, code: str) -> dict[str, str]:
    games = _games(store, comp)
    if code not in {*games["home"], *games["away"]}:
        raise NotFound(f"unknown team '{code}' in {comp.name}")
    return _team(comp, code, _names(store, comp))


def team_ratings(store: Store, competition: str, team: str) -> dict[str, Any]:
    """The team's published Elo (as ``site.json`` lists it): rating, change since the season's
    start and rank among the live season's teams. M1's strengths are not published."""
    comp = _competition(competition)
    info = _known_team(store, comp, team)
    table = _elo(store, comp, LIVE_SEASON)
    if team not in table:
        raise NotFound(f"'{team}' has no game in the {comp.name} {LIVE_SEASON} season")
    order = sorted(table, key=lambda t: (-round(table[t][0], 1), t))
    rating, start = table[team]
    return {
        "competition": comp.name,
        "team": info,
        "season": LIVE_SEASON,
        "elo": {
            "rating": round(rating, 1),
            "change": round(rating - start, 1),
            "rank": order.index(team) + 1,
            "teams": len(order),
        },
        "m1": {
            "available": False,
            "reason": "M1's team strengths are not published; its forecasts are under "
            "/predictions/{game_id}",
        },
    }


def _gbl_player_games(store: Store) -> pd.DataFrame | None:
    """GBL player games from the cached box pages; None without the marts or the cache."""

    def make() -> pd.DataFrame | None:
        raw = store.path(GBL.raw_dir)
        if not store.path(MART_PATH).exists() or not (raw / "boxscore").exists():
            return None
        team_games = read_table(_mart(store), "team_games", GBL.name)
        if team_games is None:
            return None
        return build_gbl_player_games(raw, _games(store, GBL), team_games).table

    return store.memo("gbl_player_games", make)


def _team_lines(store: Store, comp: Competition) -> tuple[pd.DataFrame | None, str]:
    """Team box lines (``TEAM_LINES_SCHEMA``) of a competition, or None and why not."""

    def make() -> tuple[pd.DataFrame | None, str]:
        if comp is EUROLEAGUE:
            teams = _el_box(store).teams
            lines = teams.assign(fga=teams["fg2a"] + teams["fg3a"])
            return validated(lines[list(TEAM_LINES_SCHEMA.columns)], TEAM_LINES_SCHEMA), ""
        players = _gbl_player_games(store)
        team_games = (
            read_table(_mart(store), "team_games", comp.name)
            if store.path(MART_PATH).exists()
            else None
        )
        if players is None or team_games is None or team_games.empty:
            return None, f"no {comp.name} box-score lines are cached"
        made = players.groupby(["game_id", "team"], as_index=False)[["fg2m", "fg3m"]].sum()
        lines = team_games.merge(made, on=["game_id", "team"], validate="one_to_one").rename(
            columns={"poss_raw": "poss"}
        )
        return validated(lines[list(TEAM_LINES_SCHEMA.columns)], TEAM_LINES_SCHEMA), ""

    return store.memo(f"team_lines:{comp.name}", make)


def _factor(numerator: float, denominator: float, what: str) -> dict[str, Any]:
    if denominator <= 0:
        return {"value": None, "reason": f"no {what} in the season"}
    return {"value": float(numerator) / float(denominator), "reason": None}


def _unavailable(reason: str) -> dict[str, Any]:
    return {"value": None, "reason": reason}


def _four_factors(lines: pd.DataFrame, team: str, season: int) -> dict[str, Any]:
    season_lines = lines[lines["season"] == season]
    mine = season_lines[season_lines["team"] == team]
    opponents = season_lines[["game_id", "team", "dreb"]].rename(
        columns={"team": "opponent", "dreb": "opp_dreb"}
    )
    paired = mine.merge(opponents, on=["game_id", "opponent"], how="left")
    sums = paired[["fg2m", "fg3m", "fga", "fta", "oreb", "tov", "poss", "opp_dreb"]].sum()
    orb = (
        _unavailable("an opponent's box line is missing for one of the games")
        if paired["opp_dreb"].isna().any()
        else _factor(sums["oreb"], sums["oreb"] + sums["opp_dreb"], "rebounds available")
    )
    return {
        "games": len(paired),
        "factors": {
            "efg_pct": _factor(
                sums["fg2m"] + 1.5 * sums["fg3m"], sums["fga"], "field-goal attempt"
            ),
            "tov_pct": _factor(sums["tov"], sums["poss"], "possession"),
            "orb_pct": orb,
            "ft_rate": _factor(sums["fta"], sums["fga"], "field-goal attempt"),
        },
    }


def _schedule_strength(store: Store, comp: Competition, team: str, season: int) -> dict[str, Any]:
    out: dict[str, Any] = {
        "rating": "Elo as published, end of the replay through the season; mean over the team's "
        "scheduled games (played and remaining)",
    }
    try:
        ratings = _elo(store, comp, season)
        games = _games(store, comp)
    except NotFound as missing:
        return {"value": None, "reason": missing.reason, **out}
    mine = games[(games["season"] == season) & ((games["home"] == team) | (games["away"] == team))]
    opponents = [a if h == team else h for h, a in zip(mine["home"], mine["away"], strict=True)]
    if not opponents:
        return {"value": None, "reason": f"no scheduled games in {season}", **out}
    return {
        "value": float(np.mean([ratings[o][0] for o in opponents])),
        "reason": None,
        "games": len(opponents),
        **out,
    }


def team_factors(store: Store, competition: str, team: str, season: int | None) -> dict[str, Any]:
    """The four factors and schedule strength of a team's season (latest with box scores when
    ``season`` is None); a missing input gives ``value: null`` and a ``reason``."""
    comp = _competition(competition)
    info = _known_team(store, comp, team)
    lines, why = _team_lines(store, comp)
    own = None if lines is None else lines[lines["team"] == team]
    seasons = [] if own is None else sorted(int(s) for s in own["season"].unique())
    chosen = season if season is not None else (seasons[-1] if seasons else LIVE_SEASON)
    if own is None or chosen not in seasons:
        reason = why or f"no box-score games for '{team}' in {chosen}"
        factors = {
            name: _unavailable(reason) for name in ("efg_pct", "tov_pct", "orb_pct", "ft_rate")
        }
        box: dict[str, Any] = {"games": 0, "factors": factors}
    else:
        assert lines is not None
        box = _four_factors(lines, team, chosen)
    return {
        "competition": comp.name,
        "team": info,
        "season": chosen,
        "available_seasons": seasons,
        **box,
        "schedule_strength": _schedule_strength(store, comp, team, chosen),
    }


# ---- players ----


def _xwalk(store: Store) -> pd.DataFrame:
    """The committed crosswalk file (D25; what CI has), else the ``player_xwalk`` mart."""

    def make() -> pd.DataFrame:
        if store.path(PLAYER_XWALK_FILE).exists():
            return read_xwalk_file(store.path(PLAYER_XWALK_FILE))
        found = read_table(_mart(store), "player_xwalk") if store.path(MART_PATH).exists() else None
        if found is None:
            return pd.DataFrame(columns=["person_id", "competition", "source_id"])
        return found

    return store.memo("xwalk", make)


def _player_games(store: Store) -> dict[str, pd.DataFrame]:
    def make() -> dict[str, pd.DataFrame]:
        games = {"euroleague": _el_box(store).players}
        gbl = _gbl_player_games(store)
        if gbl is not None:
            games["gbl"] = gbl
        return games

    return store.memo("player_games", make)


def _player_seasons(store: Store) -> pd.DataFrame:
    def make() -> pd.DataFrame:
        try:
            return build_player_seasons(_player_games(store), _xwalk(store))
        except ValueError as empty:
            raise NotFound(f"no box scores are cached: {empty}") from empty

    return store.memo("player_seasons", make)


def _person_names(store: Store) -> dict[str, str]:
    """Person id -> display name: the EuroLeague box-score spelling, else the GBL name table's."""

    def make() -> dict[str, str]:
        gbl = None
        if store.path(MART_PATH).exists():
            gbl = read_table(_mart(store), "player_names", "gbl")
        return person_names(
            _el_box(store).players, pd.DataFrame() if gbl is None else gbl, _xwalk(store)
        )

    return store.memo("person_names", make)


@dataclass(frozen=True)
class Impact:
    units: str
    rows: dict[str, list[dict[str, Any]]]  # person id -> one dict per season


def _m3_index(store: Store) -> Impact | None:
    def make() -> Impact | None:
        report = _report(store, M3_PLAYERS_REPORT)
        if report is None:
            return None
        records = [
            {"season": int(season), **p}
            for season, snapshot in report["seasons"].items()
            for p in snapshot["players"]
        ]
        frame = validated(
            pd.DataFrame(records, columns=list(M3_ROWS_SCHEMA.columns)), M3_ROWS_SCHEMA
        )
        ids = person_ids(frame["player_id"], "euroleague", _xwalk(store))["person_id"]
        keep = ("season", "o", "d", "total", "sd_total", "ci90_total", "minutes", "games", "seen")
        rows: dict[str, list[dict[str, Any]]] = {}
        for pid, record in zip(ids, records, strict=True):
            rows.setdefault(pid, []).append({k: record[k] for k in keep})
        return Impact(str(report["units"]), rows)

    return store.memo("m3_index", make)


def _m2_index(store: Store) -> Impact | None:
    def make() -> Impact | None:
        report = _report(store, M2_PLAYERS_REPORT)
        if report is None:
            return None
        records = report["players"]
        frame = validated(
            pd.DataFrame(records, columns=list(M2_ROWS_SCHEMA.columns)), M2_ROWS_SCHEMA
        )
        ids = person_ids(frame["shooter"], "euroleague", _xwalk(store))["person_id"]
        rows: dict[str, list[dict[str, Any]]] = {}
        for pid, r in zip(ids, records, strict=True):
            rows.setdefault(pid, []).append(
                {
                    "season": r["season"],
                    "split": r["split"],
                    "fga": r["fga"],
                    "shot_making": r["shrunk"],
                    "ci90": r["ci90"],
                }
            )
        return Impact(str(report["units"]), rows)

    return store.memo("m2_index", make)


def player(store: Store, person_id: str) -> dict[str, Any]:
    """One person's seasons (box totals, per-100 rates, shooting percentages), his M3 impact
    snapshots and M2 shot-making; the impact blocks are null for a person they do not cover."""
    frame = _player_seasons(store)
    rows = frame[frame["person_id"] == person_id]
    if rows.empty:
        raise NotFound(f"unknown person_id '{person_id}'")
    rows = rows.sort_values(["season", "competition"])
    rates = rate_table(rows)
    seasons = []
    for row, rate in zip(rows.to_dict("records"), rates.to_dict("records"), strict=True):
        seasons.append(
            {
                "competition": row["competition"],
                "season": int(row["season"]),
                "partial": bool(row["partial"]),
                "team": row["team"],
                "games": int(row["games"]),
                "minutes": float(row["minutes"]),
                "poss": float(row["poss"]),
                "totals": {c: int(row[c]) for c in COUNT_COLUMNS},
                "per100": {s: rate[s] for s in COUNT_STATS},
                "pct": {s: rate[s] for s in PCT_STATS},
            }
        )
    m3, m2 = _m3_index(store), _m2_index(store)
    out: dict[str, Any] = jsonable(
        {
            "person_id": person_id,
            "name": _person_names(store).get(person_id, person_id),
            "debut_season": int(rows["debut_season"].iloc[0]),
            "seasons": seasons,
            "impact": None
            if m3 is None or person_id not in m3.rows
            else {"model": "m3", "units": m3.units, "seasons": m3.rows[person_id]},
            "shot_making": None
            if m2 is None or person_id not in m2.rows
            else {"model": "m2", "units": m2.units, "seasons": m2.rows[person_id]},
        }
    )
    return out


def player_projection(store: Store, person_id: str) -> dict[str, Any]:
    report = _required(store, M6_PROJECTIONS, "the M6 projections")
    # the whole list is validated once per Store, not once per person (the export asks for each)
    store.memo("checked:m6_projections", lambda: _check(report["players"], PROJECTION_ROWS_SCHEMA))
    rows = [p for p in report["players"] if p["person_id"] == person_id]
    if not rows:
        raise NotFound(
            f"no M6 projection for person_id '{person_id}' "
            f"(not in the {report['season']} projected set)"
        )
    keys = (
        "model",
        "variant",
        "half_life",
        "gated",
        "gate_passed",
        "season",
        "checkpoint",
        "data_sha256",
        "stats",
    )
    return {**{k: report[k] for k in keys}, "projections": rows}


def player_similar(store: Store, person_id: str) -> dict[str, Any]:
    report = _required(store, M6_SIMILARITY, "the M6 similarity lists")
    if person_id not in report["players"]:
        raise NotFound(f"no similarity list for person_id '{person_id}'")
    rows = report["players"][person_id]
    _check(rows, SIMILAR_ROWS_SCHEMA)
    return {
        "model": report["model"],
        "season": report["season"],
        "person_id": person_id,
        "features": report["features"],
        "pool_seasons": report["pool_seasons"],
        "similar": rows,
    }


# ---- scouting ----


def scouting_board(store: Store) -> dict[str, Any]:
    report = _required(store, M6_BOARD, "the M6 over/under board")
    _check(report["rows"], BOARD_ROWS_SCHEMA)
    return {k: report[k] for k in ("model", "season", "dimensions", "rows")}


def scouting_undervalued(store: Store) -> dict[str, Any]:
    report = _required(store, M6_PROJECTIONS, "the M6 projections")
    _check(report["undervalued"], UNDERVALUED_ROWS_SCHEMA)
    keys = ("model", "variant", "gated", "gate_passed", "season", "checkpoint")
    return {**{k: report[k] for k in keys}, "undervalued": report["undervalued"]}


def scouting_translation(store: Store) -> dict[str, Any]:
    """M4's league translation factors with the gate verdict of its backtest."""
    translation = _required(store, M4.translation_report, "the M4 translation factors")
    gate = _required(store, M4.report, "the M4 backtest")["gate"]
    return {
        "model": "m4",
        "gate": gate,
        "gate_passed": bool(gate["passed"]),
        "fits_by_target_season": translation["fits_by_target_season"],
        "team_offset": translation["team_offset"],
    }


# ---- simulations, metrics ----


def m7_verdict(store: Store) -> dict[str, Any]:
    """M7's gate verdict from its backtest report: ``passed`` and, when it failed, the rules it
    missed in one line."""
    report = _report(store, M7.report)
    if report is None:
        return {"passed": False, "reason": f"{M7.report.as_posix()} not found: M7 has no verdict"}
    gate = report["gate"]
    if gate["passed"]:
        return {"passed": True, "reason": "M7 passed its validation gate"}
    missed = []
    if gate["brier_chosen"] >= gate["brier_point_sim"]:
        missed.append(
            f"validation Brier {gate['brier_chosen']:.4f} is not below the point rule's "
            f"{gate['brier_point_sim']:.4f}"
        )
    if gate["brier_chosen"] >= gate["brier_standings_now"]:
        missed.append(
            f"validation Brier {gate['brier_chosen']:.4f} is not below standings-now's "
            f"{gate['brier_standings_now']:.4f}"
        )
    if abs(gate["spiegelhalter_z_pooled"]) >= SPIEGELHALTER_LIMIT:
        missed.append(
            f"pooled Spiegelhalter |z| {abs(gate['spiegelhalter_z_pooled']):.2f} is not below "
            f"{SPIEGELHALTER_LIMIT}"
        )
    return {
        "passed": False,
        "reason": "M7 failed its validation gate: " + "; ".join(missed)
        if missed
        else "M7 did not pass its validation gate",
    }


def simulation_latest(store: Store, competition: str) -> dict[str, Any]:
    """The latest ungated season simulation, labelled ``gated: false`` with M7's verdict."""
    comp = _competition(competition)
    verdict = m7_verdict(store)
    report = _report(store, SIM_UNGATED[comp.name])
    if report is None:
        raise NotFound(
            f"no ungated simulation report for {comp.name}: {verdict['reason']}",
            gated=False,
            gate=verdict,
        )
    return {**report, "gated": False, "gate": verdict}


def metrics_live(store: Store) -> dict[str, Any]:
    """The live scorecards (Elo, B0, and the M1 and M5 shadow blocks they carry) per competition."""
    cards = []
    for comp in COMPETITIONS.values():
        card = _report(store, comp.scorecard)
        if card is not None:
            cards.append({"competition": comp.name, "scorecard": card})
    if not cards:
        raise NotFound("no live scorecard has been written")
    return {"competitions": cards}
