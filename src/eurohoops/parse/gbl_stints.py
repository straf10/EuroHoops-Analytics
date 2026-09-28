"""GBL (ESAKE) stints mart: 5-on-5 stints with points and possessions, and per-game checks.

Mirrors ``parse.stints_mart`` (the EuroLeague stints mart) on the GBL play-by-play built by
``parse.gbl_pbp`` (``config.GBL_PBP``). A stint is a stretch of one period in which neither
team's five changes (a substitution by either team closes it). Points and possession ends
count for the stint open when their row is logged.

Method (H-i, weeks 9-12 §4 H3):

- **Starting fives:** the players logged as ``in`` before any other player-side event of
  period 1 (the fixture logs the 10 starters as "entered the court" rows at ``00:00``, before
  "Start of game"/"Start of quarter 1").
- **Lineup changes:** ``in``/``out`` rows in log (``seq``) order. Lineups carry over between
  periods: the fixture logs a period's closing substitutions as ordinary ``in``/``out`` rows at
  the boundary clock, tagged with the *old* period (``gbl_pbp.parse_export`` only advances
  ``period`` on the row that reads "Start of quarter N"), so processing periods in order with
  one running lineup dict reproduces the carry-over without special-casing it.
- **Player identity:** ``"{number}:{player}"`` per side (jersey number and name as parsed; ESAKE
  id resolution is out of scope here).
- **Points:** ``ft_made`` (1) and ``fg_made`` (``value``) rows, credited to the stint open when
  logged.
- **Possessions:** a made field goal, a defensive rebound after the other side's miss, a
  turnover, or the last free throw of a trip (made) end a possession; a miss with no rebound
  before the period ends counts as a possession end at the period's last event -- the same
  rules as ``parse.possessions``, adapted to GBL's free-text vocabulary:

  1. GBL has no typed rebound/turnover play codes. Offensive/defensive rebounds and turnovers
     are matched with the same free-text patterns ``parse.gbl_pbp`` already uses for its team
     box counts (``OREB``, ``DREB``, ``TURNOVER``), not re-derived here.
  2. **Dead-ball fouls** (technical, unsportsmanlike, disqualifying, flagrant): their free
     throws do not end the fouled team's possession (it keeps the ball). EuroLeague's PBP marks
     these by a typed play code at the same clock as the free throws it produces, so
     ``parse.possessions`` matches them by exact (period, clock). The GBL fixture logs the
     foul-commit sentence one second *before* its free throws (an unsportsmanlike foul at
     ``07:41``, its free throws at ``07:42``, ``tests/fixtures/gbl_stints/17144DAC.xlsx.gz``),
     so an exact-clock match would miss it. Instead, a dead-ball foul sets a pending flag that
     the next free-throw trip consumes (regardless of its clock), verified against that exact
     fixture sequence (the fouled team keeps the ball and scores again before either side loses
     it).
  3. **And-one** free throws (a single free throw whose trip starts at the same elapsed second
     as a made field goal by the same side) are detected by the same exact-second match
     ``parse.possessions`` uses; unverified against a real GBL and-one (none in the fixture).
  4. A team or bench technical foul logged in the shared "Game Actions" column (no side) is
     invisible here, the same as it is to ``parse.gbl_pbp.team_counts`` -- an existing
     limitation of ``classify``, not a new one.
  5. A steal is its own free-text row, separate from the turnover it causes, and carries no
     possession-ending signal of its own: the turnover phrase logged on the ball-handler's own
     side is what ends the possession (matching how ``team_counts`` already reconciles against
     ESAKE's turnover totals).

Checks per game (``CHECKS``; nothing is dropped, every result is stored): ``five_on_court``
(every positive-length stint has exactly 5 players per side), ``points`` (each side's stint
points sum to the game's final score), ``possessions`` (each side's stint possession count is
within ``POSSESSION_TOLERANCE`` of that side's ``team_games.poss_raw``; a game without
``team_games`` rows fails it with a reason). ``passed`` is all three.

Only games with a cached play-by-play export are processed (``parse.gbl_pbp.build_pbp_table``
already restricts ``config.GBL_PBP`` to those); as of weeks 9-12 that is 2018 and 2019 only
(D4, ``reports/week9-12_progress.md``).
"""

import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import pandera.pandas as pa

from eurohoops.parse.gbl_pbp import DREB, OREB, TURNOVER
from eurohoops.parse.schemas import validated
from eurohoops.parse.stints import QUARTERS, period_length, period_start
from eurohoops.parse.stints_mart import digest

SIDES = ("home", "away")
CHECKS = ("five_on_court", "points", "possessions")
POSSESSION_TOLERANCE = 2.0  # H-i: within +/-2 of team_games.poss_raw
H_I_THRESHOLD = 0.95  # H-i: GBL RAPM attempted only if >= 95% of games pass
DEAD_BALL_FOUL = re.compile(r"\b(?:technical|unsportsmanlike|disqualifying|flagrant)\b", re.I)

STINT_COLUMNS = (
    "game_id",
    "season",
    "period",
    "start_s",
    "end_s",
    "home",
    "away",
    "home_players",
    "away_players",
    "home_points",
    "away_points",
    "home_poss",
    "away_poss",
)

GBL_STINT_SCHEMA = pa.DataFrameSchema(
    {
        "game_id": pa.Column(str),
        "season": pa.Column("int64"),
        "period": pa.Column("int64", pa.Check.ge(1)),
        "start_s": pa.Column("int64", pa.Check.ge(0)),
        "end_s": pa.Column("int64", pa.Check.ge(0)),
        "home": pa.Column(str),
        "away": pa.Column(str),
        "home_players": pa.Column(object),
        "away_players": pa.Column(object),
        "home_points": pa.Column("int64", pa.Check.ge(0)),
        "away_points": pa.Column("int64", pa.Check.ge(0)),
        "home_poss": pa.Column("int64", pa.Check.ge(0)),
        "away_poss": pa.Column("int64", pa.Check.ge(0)),
    },
    checks=[
        pa.Check(lambda df: df["end_s"] >= df["start_s"], error="end_s before start_s"),
        pa.Check(lambda df: df["home"] != df["away"], error="home team equals away team"),
        pa.Check(
            lambda df: df["home_players"].map(lambda v: isinstance(v, list)).all(),
            error="home_players must be a list",
        ),
        pa.Check(
            lambda df: df["away_players"].map(lambda v: isinstance(v, list)).all(),
            error="away_players must be a list",
        ),
    ],
    strict=True,
)

GBL_STINT_CHECK_SCHEMA = pa.DataFrameSchema(
    {
        "game_id": pa.Column(str),
        "season": pa.Column("int64"),
        "five_on_court": pa.Column(bool),
        "points": pa.Column(bool),
        "possessions": pa.Column(bool),
        "passed": pa.Column(bool),
        "reasons": pa.Column(str),
    },
    unique=["game_id"],
    strict=True,
)


@dataclass(frozen=True)
class GameStint:
    period: int
    start: int
    end: int
    players: tuple[tuple[str, ...], tuple[str, ...]]  # home five, away five (sorted)
    points: tuple[int, int]
    possessions: tuple[int, int]


@dataclass
class _OpenStint:
    period: int
    start: int
    points: dict[str, int] = field(default_factory=dict)
    poss: dict[str, int] = field(default_factory=dict)

    def close(self, at: int, lineups: dict[str, set[str]], out: list[GameStint]) -> None:
        out.append(
            GameStint(
                self.period,
                self.start,
                at,
                (tuple(sorted(lineups["home"])), tuple(sorted(lineups["away"]))),
                (self.points.get("home", 0), self.points.get("away", 0)),
                (self.poss.get("home", 0), self.poss.get("away", 0)),
            )
        )
        self.start, self.points, self.poss = at, {}, {}


def starting_fives(events: pd.DataFrame) -> dict[str, set[str]]:
    """The players logged as ``in`` before any other player-side event of the game."""
    fives: dict[str, set[str]] = {"home": set(), "away": set()}
    for side, number, player, action in zip(
        events["side"], events["number"], events["player"], events["action"], strict=True
    ):
        if side not in SIDES:
            continue
        if action != "in":
            break
        fives[side].add(f"{number}:{player}")
    return fives


@dataclass
class _Trip:
    side: str
    clock_s: int
    last_made: bool
    ends_possession: bool
    last_index: int


class _Counter:
    """Possession-end state machine over one game's events, in ``seq`` order."""

    def __init__(self) -> None:
        self.ends: list[tuple[int, str]] = []
        self.pending_miss: str | None = None
        self.trip: _Trip | None = None
        self.last_made: tuple[str, int] | None = None  # (side, clock_s) of the last made shot
        self.dead_ball_pending = False

    def _end(self, index: int, side: str) -> None:
        self.ends.append((index, side))
        self.pending_miss = None

    def _close_trip(self) -> None:
        trip, self.trip = self.trip, None
        if trip is None or not trip.ends_possession:
            return
        if trip.last_made:
            self._end(trip.last_index, trip.side)
        else:
            self.pending_miss = trip.side

    def close_period(self, index: int) -> None:
        self._close_trip()
        if self.pending_miss is not None:
            self._end(index, self.pending_miss)
        self.last_made = None
        self.dead_ball_pending = False

    def _free_throw(self, index: int, side: str, clock_s: int, made: bool) -> None:
        trip = self.trip
        if trip is not None and (trip.side, trip.clock_s) == (side, clock_s):
            trip.last_made, trip.last_index = made, index
            return
        self._close_trip()
        and_one = self.last_made == (side, clock_s)
        dead_ball, self.dead_ball_pending = self.dead_ball_pending, False
        self.trip = _Trip(side, clock_s, made, not (and_one or dead_ball), index)

    def step(self, index: int, side: str, action: str, clock_s: int, text: str) -> None:
        if action in {"ft_made", "ft_missed"}:
            self._free_throw(index, side, clock_s, action == "ft_made")
        elif action == "fg_made":
            self._close_trip()
            self._end(index, side)
            self.last_made = (side, clock_s)
        elif action == "fg_missed":
            self._close_trip()
            self.pending_miss = side
        elif action == "other" and DEAD_BALL_FOUL.search(text):
            self.dead_ball_pending = True
        elif action == "other" and OREB.search(text):
            self._close_trip()
            self.pending_miss = None
        elif action == "other" and DREB.search(text):
            self._close_trip()
            other = SIDES[1 - SIDES.index(side)]
            if self.pending_miss == other:
                self._end(index, other)
            self.pending_miss = None
        elif action == "other" and TURNOVER.search(text):
            self._close_trip()
            self._end(index, side)


def _possession_ends(events: pd.DataFrame) -> dict[int, list[str]]:
    """Position (0-based, matching ``events``' row order) -> sides whose possession ended there."""
    counter = _Counter()
    period: int | None = None
    for index, (side, action, clock_s, text, ev_period) in enumerate(
        zip(
            events["side"],
            events["action"],
            events["clock_s"],
            events["text"],
            events["period"],
            strict=True,
        )
    ):
        if ev_period != period:
            if period is not None:
                counter.close_period(index - 1)
            period = ev_period
        if side in SIDES:
            counter.step(index, side, action, int(clock_s), text)
    if period is not None:
        counter.close_period(len(events) - 1)
    ends: dict[int, list[str]] = {}
    for index, side in counter.ends:
        ends.setdefault(index, []).append(side)
    return ends


def build_game_stints(
    events: pd.DataFrame, starters: dict[str, set[str]], poss_ends: dict[int, list[str]]
) -> list[GameStint]:
    """5-on-5 stints in time order (``events``: one game, sorted by ``seq``)."""
    lineups = {side: set(starters.get(side, set())) for side in SIDES}
    n_periods = max(int(events["period"].max()), len(QUARTERS)) if len(events) else len(QUARTERS)
    by_period: dict[int, list[int]] = {}
    for pos, period in enumerate(events["period"]):
        by_period.setdefault(int(period), []).append(pos)
    sides = events["side"].to_numpy()
    numbers = events["number"].to_numpy()
    players = events["player"].to_numpy()
    actions = events["action"].to_numpy()
    values = events["value"].to_numpy()
    clocks = events["clock_s"].to_numpy()
    out: list[GameStint] = []
    for period in range(1, n_periods + 1):
        stint = _OpenStint(period, period_start(period))
        for pos in by_period.get(period, []):
            side = sides[pos]
            if side in SIDES:
                action = actions[pos]
                clock = int(clocks[pos])
                if action in {"in", "out"}:
                    if clock > stint.start:
                        stint.close(clock, lineups, out)
                    key = f"{numbers[pos]}:{players[pos]}"
                    if action == "in":
                        lineups[side].add(key)
                    else:
                        lineups[side].discard(key)
                elif action == "fg_made":
                    stint.points[side] = stint.points.get(side, 0) + int(values[pos])
                elif action == "ft_made":
                    stint.points[side] = stint.points.get(side, 0) + 1
            for ended in poss_ends.get(pos, []):
                stint.poss[ended] = stint.poss.get(ended, 0) + 1
        stint.close(period_start(period) + period_length(period), lineups, out)
    return out


def check_game(
    stints: list[GameStint],
    home: str,
    away: str,
    *,
    home_score: int,
    away_score: int,
    team_poss: dict[str, float] | None,
) -> dict[str, list[str]]:
    """Failure reasons per check (an empty list means the check passed)."""
    reasons: dict[str, list[str]] = {check: [] for check in CHECKS}
    for side_index, code in enumerate((home, away)):
        wrong = [s for s in stints if (s.end - s.start) > 0 and len(s.players[side_index]) != 5]
        if wrong:
            sizes = sorted({len(s.players[side_index]) for s in wrong})
            reasons["five_on_court"].append(
                f"{code}: {len(wrong)} stints "
                f"({sum(s.end - s.start for s in wrong)} s) with {sizes} players"
            )
    scored = {home: sum(s.points[0] for s in stints), away: sum(s.points[1] for s in stints)}
    finals = {home: home_score, away: away_score}
    for code in (home, away):
        if scored[code] != finals[code]:
            reasons["points"].append(f"{code}: {scored[code]} from stints, {finals[code]} final")
    if team_poss is None:
        reasons["possessions"].append("no team_games rows for this game")
    else:
        counted = {
            home: sum(s.possessions[0] for s in stints),
            away: sum(s.possessions[1] for s in stints),
        }
        for code in (home, away):
            box = team_poss.get(code)
            if box is None:
                reasons["possessions"].append(f"{code}: not in team_games")
            elif abs(counted[code] - box) > POSSESSION_TOLERANCE:
                reasons["possessions"].append(
                    f"{code}: {counted[code]} counted, {box:.2f} poss_raw"
                )
    return reasons


@dataclass(frozen=True)
class GameResult:
    game_id: str
    season: int
    stints: list[GameStint]
    home: str
    away: str
    reasons: dict[str, list[str]]


def game_stints(
    game_id: str,
    season: int,
    events: pd.DataFrame,
    home: str,
    away: str,
    *,
    home_score: int,
    away_score: int,
    team_poss: dict[str, float] | None,
) -> GameResult:
    """Stints and check results of one game (``events``: its ``gbl_pbp`` rows, any order)."""
    ordered = events.sort_values("seq").reset_index(drop=True)
    starters = starting_fives(ordered)
    ends = _possession_ends(ordered)
    stints = build_game_stints(ordered, starters, ends)
    reasons = check_game(
        stints, home, away, home_score=home_score, away_score=away_score, team_poss=team_poss
    )
    return GameResult(game_id, season, stints, home, away, reasons)


@dataclass(frozen=True)
class GblStintsMart:
    stints: pd.DataFrame
    checks: pd.DataFrame


def build_gbl_stints_mart(
    pbp: pd.DataFrame, games: pd.DataFrame, team_games: pd.DataFrame
) -> GblStintsMart:
    """Stints and per-game checks of every played, non-forfeit GBL game with a cached export.

    ``pbp``: ``config.GBL_PBP`` (already restricted to games with a cached export whose score
    reconciles, by ``parse.gbl_pbp.build_pbp_table``). ``games``: ``marts.read_games(...,
    "gbl")``. ``team_games``: ``marts.read_table(..., "team_games", "gbl")``.
    """
    by_game = {str(k): v for k, v in pbp.groupby("game_id")} if len(pbp) else {}
    poss_by_game: dict[str, dict[str, float]] = {}
    if len(team_games):
        for game_id, rows in team_games.groupby("game_id"):
            poss_by_game[str(game_id)] = dict(zip(rows["team"], rows["poss_raw"], strict=True))
    played = games[games["played"] & ~games["forfeit"]]
    stint_rows: list[dict[str, Any]] = []
    check_rows: list[dict[str, Any]] = []
    for game_id, season, home, away, home_score, away_score in zip(
        played["game_id"],
        played["season"],
        played["home"],
        played["away"],
        played["home_score"],
        played["away_score"],
        strict=True,
    ):
        events = by_game.get(str(game_id))
        if events is None or events.empty:
            continue
        result = game_stints(
            str(game_id),
            int(season),
            events,
            str(home),
            str(away),
            home_score=int(home_score),
            away_score=int(away_score),
            team_poss=poss_by_game.get(str(game_id)),
        )
        for stint in result.stints:
            stint_rows.append(
                {
                    "game_id": result.game_id,
                    "season": result.season,
                    "period": stint.period,
                    "start_s": stint.start,
                    "end_s": stint.end,
                    "home": result.home,
                    "away": result.away,
                    "home_players": list(stint.players[0]),
                    "away_players": list(stint.players[1]),
                    "home_points": stint.points[0],
                    "away_points": stint.points[1],
                    "home_poss": stint.possessions[0],
                    "away_poss": stint.possessions[1],
                }
            )
        passed = not any(result.reasons[c] for c in CHECKS)
        check_rows.append(
            {
                "game_id": result.game_id,
                "season": result.season,
                **{c: not result.reasons[c] for c in CHECKS},
                "passed": passed,
                "reasons": "; ".join(f"{c}: {why}" for c in CHECKS for why in result.reasons[c]),
            }
        )
    stints = pd.DataFrame(stint_rows, columns=list(STINT_COLUMNS))
    checks = pd.DataFrame(check_rows, columns=["game_id", "season", *CHECKS, "passed", "reasons"])
    return GblStintsMart(
        validated(stints, GBL_STINT_SCHEMA), validated(checks, GBL_STINT_CHECK_SCHEMA)
    )


def mart_report(mart: GblStintsMart) -> dict[str, Any]:
    """Pass rates per check and season, the H-i (>=95%) verdict, and both tables' digests."""
    checks = mart.checks
    seasons: dict[str, Any] = {}
    for season, rows in checks.groupby("season"):
        seasons[str(season)] = {
            "games": len(rows),
            "passed": int(rows["passed"].sum()),
            "pass_rate": round(float(rows["passed"].mean()), 4),
            "pass_rate_by_check": {c: round(float(rows[c].mean()), 4) for c in CHECKS},
            "failures_by_check": {c: int((~rows[c]).sum()) for c in CHECKS},
            "failed_games": {
                str(r["game_id"]): str(r["reasons"])
                for r in rows[~rows["passed"]].to_dict("records")
            },
        }
    overall = round(float(checks["passed"].mean()), 4) if len(checks) else None
    return {
        "possession_tolerance": POSSESSION_TOLERANCE,
        "games": len(checks),
        "stints": len(mart.stints),
        "overall_pass_rate": overall,
        "h_i_threshold": H_I_THRESHOLD,
        "h_i_rule_holds": overall is not None and overall >= H_I_THRESHOLD,
        "seasons_with_play_by_play": sorted(int(s) for s in checks["season"].unique()),
        "seasons": seasons,
        "table_sha256": {
            "gbl_stints": digest(mart.stints),
            "gbl_stint_game_checks": digest(checks),
        },
    }
