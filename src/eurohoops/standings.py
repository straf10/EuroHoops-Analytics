"""2026-27 competition formats and regular-season standings with official tie-breaks.

EuroLeague: EuroLeague Bylaws 2026-27, Articles 18 (competition system) and 19 (tie breakers),
https://ftpserver.euroleague.net/general/2026_27_EuroLeague_Bylaws.pdf (accessed 2026-09-25).
GBL: ESAKE's regulations defer tie-breaks to each season's competition notice (Προκήρυξη),
which could not be found for 2026-27; the GBL uses the same head-to-head procedure, marked
unverified (reports/week3_closeout.md, 7b).
"""

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations


@dataclass(frozen=True)
class Result:
    """A played regular-season game: the final score decides the winner, but overtime points
    count for no tie-break (Art. 19.4), so score criteria use the regulation score."""

    home: str
    away: str
    home_points: int
    away_points: int
    regulation: tuple[int, int] | None = None  # (home, away) after 40 minutes if it went to OT

    @property
    def winner(self) -> str:
        return self.home if self.home_points > self.away_points else self.away

    def tiebreak_points(self) -> tuple[int, int]:
        return self.regulation or (self.home_points, self.away_points)


@dataclass(frozen=True)
class Series:
    name: str
    best_of: int  # 1 = single game


@dataclass(frozen=True)
class Format:
    competition: str
    season: str
    teams: int
    regular_season_rounds: int
    playoffs_direct: tuple[int, ...]  # regular-season positions that go straight to playoffs
    play_in: tuple[int, ...]
    eliminated: tuple[int, ...]  # positions out after the regular season
    relegated: tuple[int, ...]
    series: tuple[Series, ...]
    sources: tuple[str, ...]
    unverified: tuple[str, ...] = ()  # rules not confirmed by an official source
    # Standings points deducted before the season (team id, points). The GBL awards 2 points
    # per win and 1 per loss, so 2 points weigh one win in rank(deducted_wins=...).
    points_deducted: tuple[tuple[str, int], ...] = ()

    def deducted_wins(self) -> dict[str, int]:
        return {team: points // 2 for team, points in self.points_deducted}


EUROLEAGUE_2026 = Format(
    competition="euroleague",
    season="2026-27",
    teams=20,
    regular_season_rounds=38,  # Art. 18.1.2: double round robin
    playoffs_direct=tuple(range(1, 7)),  # Art. 18.1.3
    play_in=(7, 8, 9, 10),
    eliminated=tuple(range(11, 21)),
    relegated=(),
    series=(
        # Art. 18.2: A = 7 v 8, B = 9 v 10, C = loser A v winner B, at the better-placed team;
        # winner A is the 7th seed (plays 2nd), winner C the 8th seed (plays 1st).
        Series("play-in", 1),
        # Art. 18.3: 1 v 8, 4 v 5, 3 v 6, 2 v 7; games 1, 2, 5 at the higher seed.
        Series("playoffs", 5),
        # Art. 18.4: semifinals 1/8-4/5 and 2/7-3/6, then the final, single games.
        Series("final four", 1),
    ),
    sources=("https://ftpserver.euroleague.net/general/2026_27_EuroLeague_Bylaws.pdf Art. 18-19",),
)

GBL_2026 = Format(
    competition="gbl",
    season="2026-27",
    teams=14,
    regular_season_rounds=26,  # 14 teams home and away, as in the published ESAKE schedule
    playoffs_direct=tuple(range(1, 9)),
    play_in=(),
    eliminated=(9, 10, 11, 12, 13),
    relegated=(14,),
    series=(Series("quarterfinals", 3), Series("semifinals", 3), Series("final", 5)),
    sources=(
        "ESAKE schedule 2026-27 (14 teams, 26 rounds), esake.gr results pages",
        "https://www.esake.gr/31B83429 (2025-26 general assembly: QF/SF best of 3, final best"
        " of 5; last place relegated)",
    ),
    unverified=(
        "playoff positions (1-8 assumed) and series lengths: official 2025-26 decision, no"
        " 2026-27 competition notice found",
        "relegation of the 14th team",
        "tie-breaks: EuroLeague-style head-to-head procedure assumed; it reproduces the"
        " 2024-25 and 2025-26 final tables (Wikipedia)",
    ),
    # Olympiacos and Panathinaikos start on -2: sanction for the altercation between players
    # in the 2025-26 finals (confirmed by the user 2026-09-25; shown on ESAKE's live table).
    points_deducted=(("00000002", 2), ("00000001", 2)),
)


def _stats(team: str, results: Sequence[Result]) -> tuple[int, int, int, float]:
    """Wins, score difference, points scored and goal average (sum of per-game quotients)."""
    wins = diff = points = 0
    quotient = 0.0
    for r in results:
        if team not in (r.home, r.away):
            continue
        home, away = r.tiebreak_points()
        scored, allowed = (home, away) if r.home == team else (away, home)
        wins += r.winner == team
        diff += scored - allowed
        points += scored
        quotient += scored / allowed if allowed else float(scored)
    return wins, diff, points, round(quotient, 5)  # Art. 19.5.3: 1/100,000 precision


def _met_twice(group: Sequence[str], h2h: Sequence[Result]) -> bool:
    games = Counter(frozenset((r.home, r.away)) for r in h2h)
    return all(games[frozenset(pair)] == 2 for pair in combinations(group, 2))


def _resolve(group: list[str], results: Sequence[Result]) -> list[str]:
    """Order teams tied on wins (Art. 19.5).

    Criteria are applied one at a time; as soon as one separates the group, every sub-group
    that is still tied starts again from the first criterion with only its own teams
    (19.5.2 II a, d, e). A tie that survives every criterion is left in alphabetical order.
    """
    if len(group) == 1:
        return group
    group_set = set(group)
    h2h = [r for r in results if r.home in group_set and r.away in group_set]
    overall: list[Callable[[str], float]] = [
        lambda t: _stats(t, results)[1],
        lambda t: _stats(t, results)[2],
        lambda t: _stats(t, results)[3],
    ]
    criteria = overall
    if _met_twice(group, h2h):  # 19.5.2; otherwise 19.5.1 uses only the overall criteria
        criteria = [lambda t: _stats(t, h2h)[0], lambda t: _stats(t, h2h)[1], *overall]
    for criterion in criteria:
        keys = {team: criterion(team) for team in group}
        if len(set(keys.values())) > 1:
            ordered: list[str] = []
            for value in sorted(set(keys.values()), reverse=True):
                ordered += _resolve(sorted(t for t in group if keys[t] == value), results)
            return ordered
    return sorted(group)


def rank(results: Sequence[Result], deducted_wins: Mapping[str, int] | None = None) -> list[str]:
    """Regular-season standings: most wins first, ties broken by Art. 19.5.

    A team with wins deducted by the disciplinary bodies is last among the teams it is tied
    with (19.1). Not modelled: teams with fewer games (19.2-19.3) and the 20-0 forfeit
    exclusion (19.6).
    """
    deducted = deducted_wins or {}
    teams = sorted({r.home for r in results} | {r.away for r in results})
    wins = {team: _stats(team, results)[0] - deducted.get(team, 0) for team in teams}
    ordered: list[str] = []
    for value in sorted(set(wins.values()), reverse=True):
        tied = [t for t in teams if wins[t] == value]
        clean = [t for t in tied if not deducted.get(t)]
        sanctioned = [t for t in tied if deducted.get(t)]
        ordered += (_resolve(clean, results) if clean else []) + (
            _resolve(sanctioned, results) if sanctioned else []
        )
    return ordered
