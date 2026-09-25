"""EuroLeague stints from play-by-play: the five players on court per team, and a validator.

A stint is a stretch of one period during which a team's five on court does not change. The
starting fives come from the box score (``IsStarter``); ``IN``/``OUT`` rows then change the
lineup in log order, and every substitution and period boundary closes the team's stint.
Lineups carry over between periods, because the log records period-start changes as ordinary
``IN``/``OUT`` rows at 10:00 (5:00 in overtime).

This is the PLAN §12 item 6 validation spike (D-h): the stints mart itself is M1 work.
"""

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from eurohoops.ingest.cache import read_cached

QUARTERS = ("FirstQuarter", "SecondQuarter", "ThirdQuarter", "ForthQuarter")  # sic, API keys
QUARTER_S = 600
OVERTIME_S = 300
# Every play type that moves the running score (checked over all cached games, 2007-2026);
# before 2015-16 made layups and dunks have their own codes.
POINTS = {"2FGM": 2, "LAYUPMD": 2, "DUNK": 2, "3FGM": 3, "FTM": 1}
MINUTES_TOLERANCE_S = 60
SAMPLE_SEASONS = tuple(range(2015, 2027))
SAMPLE_SIZE = 50
SAMPLE_SEED = 20260925
CHECKS = ("five_on_court", "seconds", "minutes", "points")


@dataclass(frozen=True)
class Event:
    period: int  # 1-4, then 5, 6, ... for overtimes
    elapsed: int | None  # game seconds since tip-off; None for rows without a clock
    kind: str
    team: str
    player: str


@dataclass(frozen=True)
class Stint:
    team: str
    period: int
    start: int  # game seconds
    end: int
    players: frozenset[str]
    points_for: int
    points_against: int

    @property
    def seconds(self) -> int:
        return self.end - self.start


def period_start(period: int) -> int:
    if period <= len(QUARTERS):
        return (period - 1) * QUARTER_S
    return len(QUARTERS) * QUARTER_S + (period - len(QUARTERS) - 1) * OVERTIME_S


def period_length(period: int) -> int:
    return QUARTER_S if period <= len(QUARTERS) else OVERTIME_S


def _clock(marker: str | None) -> int | None:
    """``"mm:ss"`` remaining in the period -> seconds, or None when the row has no clock."""
    if not marker or ":" not in marker:
        return None
    minutes, seconds = marker.split(":")
    return int(minutes) * 60 + int(seconds)


def _overtime_period(minute: int) -> int:
    """PBP ``MINUTE`` counts on through overtime: 41-45 is the first, 46-50 the second, ..."""
    return len(QUARTERS) + 1 + (max(minute, 41) - 41) // 5


def events(pbp: dict[str, Any]) -> list[Event]:
    """Every PBP row in log order, with its period and game clock."""
    tagged = [(q + 1, row) for q, key in enumerate(QUARTERS) for row in pbp.get(key) or []]
    tagged += [(_overtime_period(row["MINUTE"]), row) for row in pbp.get("ExtraTime") or []]
    out = []
    for period, row in tagged:
        remaining = _clock(row.get("MARKERTIME"))
        elapsed = (
            None if remaining is None else period_start(period) + period_length(period) - remaining
        )
        out.append(
            Event(
                period=period,
                elapsed=elapsed,
                kind=(row.get("PLAYTYPE") or "").strip(),
                team=(row.get("CODETEAM") or "").strip(),
                player=(row.get("PLAYER_ID") or "").strip(),
            )
        )
    return out


@dataclass
class _Open:
    """A team's current, not yet closed stint."""

    lineup: set[str]
    start: int
    points_for: int = 0
    points_against: int = 0
    closed: list[Stint] = field(default_factory=list)

    def close(self, team: str, period: int, at: int) -> None:
        self.closed.append(
            Stint(
                team,
                period,
                self.start,
                at,
                frozenset(self.lineup),
                self.points_for,
                self.points_against,
            )
        )
        self.start, self.points_for, self.points_against = at, 0, 0


def _apply(event: Event, teams: dict[str, _Open]) -> None:
    """A substitution closes the team's open stint (unless it has no length yet); points add up."""
    state = teams.get(event.team)
    if state is None:
        return
    if event.kind in {"IN", "OUT"} and event.elapsed is not None:
        if event.elapsed > state.start:
            state.close(event.team, event.period, event.elapsed)
        if event.kind == "IN":
            state.lineup.add(event.player)
        else:
            state.lineup.discard(event.player)
    elif event.kind in POINTS:
        state.points_for += POINTS[event.kind]
        for team, rival in teams.items():
            if team != event.team:
                rival.points_against += POINTS[event.kind]


def build_stints(
    game_events: Iterable[Event], starters: dict[str, set[str]], n_overtimes: int
) -> list[Stint]:
    """Stints of both teams over 4 quarters plus ``n_overtimes``, in time order per team.

    Several substitutions at the same instant give one stint boundary, not zero-length stints.
    Points count for the stint that is open when the scoring row is logged.
    """
    teams = {team: _Open(set(players), 0) for team, players in starters.items()}
    by_period: dict[int, list[Event]] = {}
    for event in game_events:
        by_period.setdefault(event.period, []).append(event)
    for period in range(1, len(QUARTERS) + n_overtimes + 1):
        for state in teams.values():
            state.start = period_start(period)
        for event in by_period.get(period, []):
            _apply(event, teams)
        end = period_start(period) + period_length(period)
        for team, state in teams.items():
            state.close(team, period, end)
    return [stint for state in teams.values() for stint in state.closed]


@dataclass(frozen=True)
class BoxFacts:
    starters: dict[str, set[str]]
    seconds: dict[str, dict[str, int]]  # team -> player -> seconds played
    points: dict[str, int]
    overtimes: int


def _box_seconds(minutes: str) -> int:
    if ":" not in minutes:  # "DNP"
        return 0
    mm, ss = minutes.split(":")
    return int(mm) * 60 + int(ss)


class EmptyGameError(ValueError):
    """The cached box score has no players (the API's ``N/D`` placeholder)."""


def box_facts(box: dict[str, Any]) -> BoxFacts:
    if not all(side["PlayersStats"] for side in box["Stats"]):
        raise EmptyGameError("box score has no players")
    starters: dict[str, set[str]] = {}
    seconds: dict[str, dict[str, int]] = {}
    points: dict[str, int] = {}
    for side in box["Stats"]:
        players = side["PlayersStats"]
        team = players[0]["Team"].strip()
        starters[team] = {p["Player_ID"].strip() for p in players if p["IsStarter"] == 1}
        seconds[team] = {p["Player_ID"].strip(): _box_seconds(p["Minutes"]) for p in players}
        points[team] = int(side["totr"]["Points"])
    return BoxFacts(starters, seconds, points, overtimes(box["ByQuarter"]))


def overtimes(by_quarter: list[dict[str, Any]]) -> int:
    """Overtime periods in which at least one team scored.

    Some box scores (13 in 2015-16) carry ``Extra1: 0`` for both teams in games decided in
    regulation; a period where neither team scores in five minutes is not a real overtime.
    """
    periods = {k for side in by_quarter for k in side if k.startswith("Extra")}
    return sum(any(side.get(k) or 0 for side in by_quarter) for k in periods)


def validate(stints: list[Stint], facts: BoxFacts) -> dict[str, list[str]]:
    """Failure reasons per check (an empty list means the check passed)."""
    expected_s = len(QUARTERS) * QUARTER_S + facts.overtimes * OVERTIME_S
    reasons: dict[str, list[str]] = {check: [] for check in CHECKS}
    for team in sorted(facts.starters):
        own = [s for s in stints if s.team == team]
        wrong = [s for s in own if s.seconds > 0 and len(s.players) != 5]
        if wrong:
            sizes = sorted({len(s.players) for s in wrong})
            reasons["five_on_court"].append(
                f"{team}: {len(wrong)} stints ({sum(s.seconds for s in wrong)} s) with "
                f"{sizes} players"
            )
        total = sum(s.seconds for s in own)
        negative = [s for s in own if s.seconds < 0]
        if total != expected_s or negative:
            reasons["seconds"].append(
                f"{team}: {total} s, expected {expected_s}; {len(negative)} negative stints"
            )
        on_court: dict[str, int] = {}
        for stint in own:
            for player in stint.players:
                on_court[player] = on_court.get(player, 0) + stint.seconds
        players = set(facts.seconds[team]) | set(on_court)
        off = {
            p: on_court.get(p, 0) - facts.seconds[team].get(p, 0)
            for p in players
            if abs(on_court.get(p, 0) - facts.seconds[team].get(p, 0)) > MINUTES_TOLERANCE_S
        }
        if off:
            worst = max(off, key=lambda p: abs(off[p]))
            reasons["minutes"].append(
                f"{team}: {len(off)} players off by > {MINUTES_TOLERANCE_S} s "
                f"(worst {worst} {off[worst]:+d} s)"
            )
        scored = sum(s.points_for for s in own)
        if scored != facts.points[team]:
            reasons["points"].append(f"{team}: {scored} from stints, {facts.points[team]} final")
    return reasons


def sample_games(raw_dir: Path) -> list[tuple[int, int]]:
    """``SAMPLE_SIZE`` (season, game code) pairs with cached PBP and box, spread over seasons.

    Each season's games are shuffled with a fixed seed, then taken round-robin across
    ``SAMPLE_SEASONS``, so the sample is deterministic for a given cache.
    """
    rng = np.random.default_rng(SAMPLE_SEED)
    pools = []
    for season in SAMPLE_SEASONS:
        pbp = raw_dir / "playbyplay" / f"E{season}"
        box = raw_dir / "boxscore" / f"E{season}"
        codes = sorted(
            int(p.name.split(".")[0]) for p in pbp.glob("*.json.gz") if (box / p.name).exists()
        )
        pools.append([(season, codes[i]) for i in rng.permutation(len(codes))])
    picked: list[tuple[int, int]] = []
    while len(picked) < SAMPLE_SIZE and any(pools):
        for pool in pools:
            if pool and len(picked) < SAMPLE_SIZE:
                picked.append(pool.pop(0))
    return picked


def validate_sample(raw_dir: Path) -> dict[str, Any]:
    """Validate the sample; JSON-ready, without timestamps, so reruns are byte-identical."""
    results = {}
    for season, code in sample_games(raw_dir):
        name = f"{code}.json.gz"
        pbp = json.loads(read_cached(raw_dir / "playbyplay" / f"E{season}" / name))
        try:
            facts = box_facts(json.loads(read_cached(raw_dir / "boxscore" / f"E{season}" / name)))
        except EmptyGameError as exc:
            results[f"E{season}_{code}"] = {check: [str(exc)] for check in CHECKS}
            continue
        stints = build_stints(events(pbp), facts.starters, facts.overtimes)
        results[f"E{season}_{code}"] = validate(stints, facts)
    n = len(results)
    failing = {
        game_id: {check: why for check, why in reasons.items() if why}
        for game_id, reasons in results.items()
        if any(reasons.values())
    }
    return {
        "seed": SAMPLE_SEED,
        "seasons": [SAMPLE_SEASONS[0], SAMPLE_SEASONS[-1]],
        "games": n,
        "minutes_tolerance_s": MINUTES_TOLERANCE_S,
        "pass_rate": {
            check: round(sum(not r[check] for r in results.values()) / n, 4) for check in CHECKS
        },
        "all_checks_pass_rate": round((n - len(failing)) / n, 4),
        "sample": list(results),
        "failing_games": failing,
    }
