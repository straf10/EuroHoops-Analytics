"""Possessions (PLAN R1): the box-score estimate for both competitions and a EuroLeague PBP count.

Box estimate per team: ``FGA - OREB + TOV + 0.42 * FTA``; the game's possession count is the
mean of both teams' estimates. R1's 0.44 is an NBA value; 0.42 is the weight that makes the
mean box estimate equal the mean play-by-play count on EuroLeague games (M1 v2, see
docs/data/possessions.md). The play-by-play count ends a possession at a made field goal,
a defensive rebound after a miss, a turnover, or the last free throw of a trip (made; a missed
last free throw waits for its rebound). Free throws after an and-one field goal, a technical,
unsportsmanlike or disqualifying foul, or a bench/coach foul end nothing. A miss that no
rebound follows before the period ends counts as a possession end at the buzzer.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from eurohoops.parse.stints import Event

FT_WEIGHT = 0.42
MADE_FG = frozenset({"2FGM", "3FGM", "LAYUPMD", "DUNK"})
MISSED_FG = frozenset({"2FGA", "3FGA", "2FGAB", "3FGAB", "LAYUPATT"})
FREE_THROWS = frozenset({"FTM", "FTA"})
NO_POSSESSION_FOULS = frozenset({"CMT", "CMU", "CMD", "B", "C"})


def box_possessions(fga: float, oreb: float, tov: float, fta: float) -> float:
    return fga - oreb + tov + FT_WEIGHT * fta


@dataclass
class _Trip:
    team: str
    period: int
    elapsed: int | None
    last_made: bool
    ends_possession: bool
    last_index: int  # the trip's last free throw so far


@dataclass(frozen=True)
class PossessionEnd:
    index: int  # the event (in log order) that ends the possession
    team: str  # the team whose possession it was


class _Counter:
    def __init__(self, teams: Sequence[str]) -> None:
        self.teams = tuple(teams)
        self.ends: list[PossessionEnd] = []
        self.pending_miss: str | None = None
        self.trip: _Trip | None = None
        self.last_fg: tuple[str, int, int | None] | None = None  # team, period, elapsed
        self.dead_ball_fouls: set[tuple[int, int | None]] = set()

    def other(self, team: str) -> str | None:
        rest = [t for t in self.teams if t != team]
        return rest[0] if len(rest) == 1 else None

    def end(self, index: int, team: str) -> None:
        self.ends.append(PossessionEnd(index, team))
        self.pending_miss = None

    def close_trip(self) -> None:
        trip, self.trip = self.trip, None
        if trip is None or not trip.ends_possession:
            return
        if trip.last_made:
            self.end(trip.last_index, trip.team)
        else:
            self.pending_miss = trip.team

    def close_period(self, index: int) -> None:
        self.close_trip()
        if self.pending_miss is not None:
            self.end(index, self.pending_miss)
        self.last_fg = None
        self.dead_ball_fouls.clear()

    def free_throw(self, index: int, event: Event) -> None:
        trip = self.trip
        if trip and (trip.team, trip.period, trip.elapsed) == (
            event.team,
            event.period,
            event.elapsed,
        ):
            trip.last_made = event.kind == "FTM"
            trip.last_index = index
            return
        self.close_trip()
        and_one = self.last_fg == (event.team, event.period, event.elapsed)
        dead_ball = (event.period, event.elapsed) in self.dead_ball_fouls
        self.trip = _Trip(
            event.team,
            event.period,
            event.elapsed,
            event.kind == "FTM",
            not (and_one or dead_ball),
            index,
        )

    def step(self, index: int, event: Event) -> None:
        kind, team = event.kind, event.team
        if kind in FREE_THROWS:
            self.free_throw(index, event)
        elif kind in NO_POSSESSION_FOULS:
            self.dead_ball_fouls.add((event.period, event.elapsed))
        elif kind in MADE_FG:
            self.close_trip()
            self.end(index, team)
            self.last_fg = (team, event.period, event.elapsed)
        elif kind in MISSED_FG:
            self.close_trip()
            self.pending_miss = team
        elif kind == "D":
            self.close_trip()
            shooter = self.other(team)
            if shooter is not None and self.pending_miss == shooter:
                self.end(index, shooter)
            self.pending_miss = None
        elif kind == "O":
            self.close_trip()
            self.pending_miss = None
        elif kind == "TO":
            self.close_trip()
            self.end(index, team)


def possession_ends(game_events: Sequence[Event], teams: Sequence[str]) -> list[PossessionEnd]:
    """Every possession end of the game, in log order (``teams``: the two team codes).

    A made last free throw ends the possession at that free throw; a miss left without a
    rebound ends it at the period's last event.
    """
    counter = _Counter(teams)
    period: int | None = None
    for index, event in enumerate(game_events):
        if event.period != period:
            if period is not None:
                counter.close_period(index - 1)
            period = event.period
        if event.team in counter.teams or not event.team:
            counter.step(index, event)
    if period is not None:
        counter.close_period(len(game_events) - 1)
    return counter.ends


def count_possessions(game_events: Sequence[Event], teams: Sequence[str]) -> dict[str, int]:
    ends = possession_ends(game_events, teams)
    return {team: sum(end.team == team for end in ends) for team in teams}
