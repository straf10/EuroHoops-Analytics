"""A small synthetic two-league world for the M6 harness (weeks 16-18 L6).

Reused by ``test_m6_backtest.py`` and the leakage suite: ``build_inputs`` returns an
``M6Inputs`` (``seed`` fixes every random draw), ``small_spec`` the matching ``M6Backtest`` and
``synthetic_rounds`` the injectable ``rounds`` callable of ``run_m6_backtest``. The module-level
``linear_spm`` is the known SPM the world plants.

The world. Eight EuroLeague teams (``E01``...) and, from ``gbl_from``, six GBL teams (``G01``...)
play ``LEGS`` legs of a round robin each season (28 and 20 weekly rounds, tip-offs 15 minutes
apart within a round), plus one playoff game (phase ``PO``) with absurd box lines that a
regular-season-only harness must ignore. Rosters hold 11 (EuroLeague) and 10 (GBL) players;
careers last 2-8 seasons, and a continuing player changes league with probability ``MOVE_P`` (the
movers). Each player has latent log-rate offsets, a random walk of true talent (``DRIFT_SD`` per
season on the log scale of the counts, ``PCT_DRIFT_SD`` on the logit of the percentages) and a
planted aging curve: the log rate of a count is ``scale_k * strength * -0.006 * (age - 26)^2``
(``AGING_SCALE``), so players decline after 26 (``aging_strength`` 0 plants none). A GBL row has
the EuroLeague rate translated as ``(r + c) * exp(-delta) - c`` (``DELTA``, ``PSEUDO``), the
direction M4's factors move it back. Box lines are drawn from the true rates: Poisson counts per
100 possessions, binomial makes; possessions = team possessions x seconds / 2400.

Impact. ``linear_spm`` is the known SPM: a linear function of the per-100 rates (``SPM_COEF``).
A EuroLeague player-season's BRAPM snapshot (from ``brapm_from``) is the SPM of his true rates
plus a persistent player effect plus noise with sd ``BRAPM_SD * sqrt(1000 / poss)``.

Ages are synthetic (age at 1 October = ``offset + season``). One ``Translation`` per target season
from ``translations_from``. Source ids: EuroLeague ``P000001``..., GBL ``G000001``...; a player of
both leagues has two crosswalk rows and the EuroLeague person id ``P:<id>``.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from eurohoops.config import M6Backtest
from eurohoops.eval.m6_backtest import M6Inputs
from eurohoops.models.player_seasons import COUNT_COLUMNS
from eurohoops.models.projection import Translation

COMPETITIONS = ("euroleague", "gbl")
N_TEAMS = {"euroleague": 8, "gbl": 6}
ROSTER = {"euroleague": 11, "gbl": 10}
TEAM_POSS = {"euroleague": 72.0, "gbl": 70.0}
LEGS = 4
SEASONS = tuple(range(2010, 2021))
MOVE_P = 0.07
DRIFT_SD = 0.04
PCT_DRIFT_SD = 0.06
# EuroLeague league means per 100 possessions, and the sd of the player offsets on the log scale.
MU = {
    "fg2a": 22.0,
    "fg3a": 9.0,
    "fta": 8.0,
    "ast": 6.0,
    "tov": 4.5,
    "oreb": 3.0,
    "dreb": 8.0,
    "stl": 2.0,
    "blk": 1.0,
    "pf": 5.0,
}
OFFSET_SD = 0.3
P_MEAN = {"fg2": 0.5, "fg3": 0.35, "ft": 0.75}
P_SD = {"fg2": 0.25, "fg3": 0.3, "ft": 0.35}  # on the logit scale
AGING_SCALE = {
    "fg2a": 1.0,
    "fg3a": 1.0,
    "fta": 0.5,
    "ast": 0.5,
    "tov": -0.5,
    "oreb": 0.5,
    "dreb": 0.0,
    "stl": 1.0,
    "blk": 0.5,
    "pf": 0.0,
}
DELTA = {
    "pts": 0.08,
    "fg3a": 0.1,
    "fta": 0.05,
    "ast": 0.0,
    "tov": -0.05,
    "oreb": 0.05,
    "dreb": 0.0,
    "stl": 0.0,
    "blk": 0.1,
}
PSEUDO = {stat: 2.0 for stat in DELTA}
SPM_COEF = {
    "pts": 0.05,
    "fta": 0.02,
    "ast": 0.08,
    "oreb": 0.06,
    "dreb": 0.04,
    "stl": 0.15,
    "blk": 0.08,
    "tov": -0.12,
}
SPM_INTERCEPT = -3.0
BRAPM_SD = 2.5
BRAPM_EFFECT_SD = 0.8


def linear_spm(frame: pd.DataFrame, before_season: int) -> "pd.Series[float]":
    """The planted SPM of ``spm_frame`` rows: ``SPM_INTERCEPT`` plus ``SPM_COEF`` times the
    per-100 rates; it does not depend on ``before_season``."""
    value = np.full(len(frame), SPM_INTERCEPT)
    for stat, coef in SPM_COEF.items():
        value = value + coef * frame[stat].to_numpy(dtype=np.float64)
    return pd.Series(value, index=frame.index)


def small_spec(*, gbl: bool = False) -> M6Backtest:
    """Two to three tuning seasons, one validation, two test; the GBL's tuning starts later."""
    return M6Backtest(
        report=Path("unused.json"),
        players_report=Path("unused.csv"),
        tuning=(2016, 2017) if gbl else (2015, 2016, 2017),
        validation=(2018,),
        test=(2019, 2020),
        half_lives=(1.0, 2.0, 3.0),
        bootstrap_resamples=200,
    )


def synthetic_rounds(competition: str, season: int) -> int:
    """R of the synthetic format: ``LEGS`` legs of the round robin."""
    return LEGS * (N_TEAMS[competition] - 1)


def team_codes(competition: str) -> tuple[str, ...]:
    letter = "E" if competition == "euroleague" else "G"
    return tuple(f"{letter}{i:02d}" for i in range(1, N_TEAMS[competition] + 1))


def _legs(teams: tuple[str, ...]) -> list[list[tuple[str, str]]]:
    """The rounds of ``LEGS`` legs of a round robin by the circle method; every team plays once
    per round and the venue alternates between legs."""
    rotation = list(teams)
    half = len(teams) // 2
    first = []
    for r in range(len(teams) - 1):
        pairs = [(rotation[i], rotation[-1 - i]) for i in range(half)]
        first.append(pairs if r % 2 == 0 else [(b, a) for a, b in pairs])
        rotation = [rotation[0], rotation[-1], *rotation[1:-1]]
    swapped = [[(b, a) for a, b in pairs] for pairs in first]
    legs = [first, swapped] * (LEGS // 2)
    return [round_ for leg in legs for round_ in leg]


def _schedule(competition: str, season: int, *, playoff: bool) -> pd.DataFrame:
    """The season's games: regular season rounds, then optionally one playoff game."""
    letter = "E" if competition == "euroleague" else "G"
    start = pd.Timestamp(f"{season}-10-01T18:00:00Z") + pd.Timedelta(
        minutes=30 * (competition == "gbl")
    )
    teams = team_codes(competition)
    rows = []
    for r, pairs in enumerate(_legs(teams), start=1):
        for slot, (home, away) in enumerate(pairs):
            rows.append(
                (r, "RS", start + pd.Timedelta(days=7 * (r - 1), minutes=15 * slot), home, away)
            )
    if playoff:
        last = len(_legs(teams)) + 1
        rows.append((last, "PO", start + pd.Timedelta(days=7 * last), teams[0], teams[1]))
    frame = pd.DataFrame(rows, columns=["round", "phase", "tipoff_utc", "home", "away"])
    frame["game_code"] = np.arange(1, len(frame) + 1)
    frame["game_id"] = [f"{letter}{season}_{c}" for c in frame["game_code"]]
    frame["season"] = season
    frame["played"] = True
    frame["tipoff_utc"] = pd.to_datetime(frame["tipoff_utc"], utc=True)
    return frame[
        ["game_id", "season", "game_code", "phase", "round", "tipoff_utc", "home", "away", "played"]
    ]


@dataclass
class _Person:
    """A synthetic player: latent log-rate offsets ``z``, the random-walk state ``walk`` (both
    per count stat and, on the logit scale, per percentage), age offset and base minutes."""

    league: str
    offset: float
    left: int
    minutes: float
    z: dict[str, float]
    walk: dict[str, float]


def _entrant(rng: np.random.Generator, league: str, season: int) -> _Person:
    return _Person(
        league=league,
        offset=float(rng.uniform(19.0, 31.0)) - season,
        left=int(rng.integers(2, 9)),
        minutes=float(rng.uniform(10.0, 32.0)),
        z={
            **{k: float(rng.normal(0.0, OFFSET_SD)) for k in MU},
            **{k: float(rng.normal(0.0, P_SD[k])) for k in P_SD},
        },
        walk=dict.fromkeys([*MU, *P_SD], 0.0),
    )


def _population(rng: np.random.Generator, seasons: tuple[int, ...], gbl_from: int) -> pd.DataFrame:
    """One row per (player, season, competition, team): the rosters of every season with the
    player's latent offsets (``z_*``), age offset (age = offset + season), base minutes and
    random-walk state (``walk_*``) at that season."""
    people: list[_Person] = []
    active: list[int] = []
    rows = []
    for season in seasons:
        leagues = [c for c in COMPETITIONS if c == "euroleague" or season >= gbl_from]
        assigned: dict[str, list[int]] = {c: [] for c in leagues}
        for pid in rng.permutation(active).tolist():
            person = people[pid]
            if len(leagues) > 1 and season > gbl_from and rng.random() < MOVE_P:
                person.league = leagues[1 - leagues.index(person.league)]
            if len(assigned[person.league]) < N_TEAMS[person.league] * ROSTER[person.league]:
                assigned[person.league].append(pid)
        active = []
        for league in leagues:
            while len(assigned[league]) < N_TEAMS[league] * ROSTER[league]:
                people.append(_entrant(rng, league, season))
                assigned[league].append(len(people) - 1)
            teams = team_codes(league)
            for i, pid in enumerate(rng.permutation(assigned[league]).tolist()):
                person = people[pid]
                for k in person.walk:
                    person.walk[k] += float(rng.normal(0.0, DRIFT_SD if k in MU else PCT_DRIFT_SD))
                rows.append(
                    {
                        "pid": pid,
                        "season": season,
                        "competition": league,
                        "team": teams[i % len(teams)],
                        "offset": person.offset,
                        "minutes": person.minutes,
                        **{f"z_{k}": v for k, v in person.z.items()},
                        **{f"walk_{k}": v for k, v in person.walk.items()},
                    }
                )
                person.left -= 1
                if person.left > 0:
                    active.append(pid)
    return pd.DataFrame(rows)


def _true_rates(pop: pd.DataFrame, aging_strength: float) -> pd.DataFrame:
    """The true per-100 rates and percentages of each roster row (translated for the GBL)."""
    age = pop["offset"].to_numpy() + pop["season"].to_numpy()
    curve = -0.006 * (age - 26.0) ** 2 * aging_strength
    out = pop[["pid", "season", "competition", "team"]].copy()
    for k, mu in MU.items():
        out[k] = mu * np.exp(pop[f"z_{k}"] + pop[f"walk_{k}"] + AGING_SCALE[k] * curve)
    for k, mean in P_MEAN.items():
        logit = np.log(mean / (1.0 - mean)) + pop[f"z_{k}"] + pop[f"walk_{k}"]
        out[k] = 1.0 / (1.0 + np.exp(-logit))
    out["pts"] = (
        2.0 * out["fg2a"] * out["fg2"] + 3.0 * out["fg3a"] * out["fg3"] + out["fta"] * out["ft"]
    )
    gbl = (out["competition"] == "gbl").to_numpy()
    for k, delta in DELTA.items():
        moved = np.maximum((out[k] + PSEUDO[k]) * np.exp(-delta) - PSEUDO[k], 0.0)
        out[k] = np.where(gbl, moved, out[k])
    return out


def _lines(
    rng: np.random.Generator, rates: pd.DataFrame, pop: pd.DataFrame, schedule: pd.DataFrame
) -> pd.DataFrame:
    """The player lines of the regular-season games: every roster row against its team's games."""
    roster = rates.assign(minutes=pop["minutes"].to_numpy())
    sides = pd.concat(
        [
            schedule.rename(columns={"home": "team"})[["game_id", "season", "team", "phase"]],
            schedule.rename(columns={"away": "team"})[["game_id", "season", "team", "phase"]],
        ]
    )
    lines = roster.merge(sides, on=["season", "team"])
    lines = lines[lines["phase"] == "RS"].reset_index(drop=True)
    n = len(lines)
    sec = np.rint(lines["minutes"].to_numpy() * 60.0 * rng.uniform(0.85, 1.15, n)).astype("int64")
    sec[rng.random(n) < 0.04] = 0  # did not play
    lines["sec"] = sec
    poss = np.array([TEAM_POSS[c] for c in lines["competition"]]) * sec / 2400.0
    lines["poss"] = poss

    def poisson(rate: str) -> np.ndarray:
        return rng.poisson(lines[rate].to_numpy() * poss / 100.0).astype("int64")

    fg2a, fg3a, fta = poisson("fg2a"), poisson("fg3a"), poisson("fta")
    fg2m = rng.binomial(fg2a, lines["fg2"].to_numpy()).astype("int64")
    fg3m = rng.binomial(fg3a, lines["fg3"].to_numpy()).astype("int64")
    ftm = rng.binomial(fta, lines["ft"].to_numpy()).astype("int64")
    counts = {
        "pts": 2 * fg2m + 3 * fg3m + ftm,
        "fg2m": fg2m,
        "fg2a": fg2a,
        "fg3m": fg3m,
        "fg3a": fg3a,
        "ftm": ftm,
        "fta": fta,
        **{k: poisson(k) for k in ("oreb", "dreb", "ast", "stl", "blk", "tov", "pf")},
    }
    for column in COUNT_COLUMNS:
        lines[column] = counts[column]
    return lines


def _playoff_lines(
    competition: str, season: int, schedule: pd.DataFrame, rosters: pd.DataFrame
) -> pd.DataFrame:
    """Absurd lines of the playoff game: a harness that reads them is wrong by a mile."""
    game = schedule[schedule["phase"] == "PO"].iloc[0]
    rows = rosters[
        (rosters["season"] == season)
        & (rosters["competition"] == competition)
        & rosters["team"].isin([game["home"], game["away"]])
    ].copy()
    rows["game_id"] = game["game_id"]
    rows["sec"] = 1500
    rows["poss"] = 45.0
    for column in COUNT_COLUMNS:
        rows[column] = 40
    return rows


def build_inputs(
    *,
    seasons: tuple[int, ...] = SEASONS,
    seed: int = 0,
    aging_strength: float = 1.0,
    gbl_from: int = 2012,
    brapm_from: int = 2012,
    translations_from: int | None = None,
    playoffs: bool = True,
) -> M6Inputs:
    """The synthetic world of the module docstring, deterministic in ``seed``."""
    rng = np.random.default_rng(seed)
    pop = _population(rng, seasons, gbl_from)
    rates = _true_rates(pop, aging_strength)
    games, player_games = {}, {}
    for competition in COMPETITIONS:
        schedule = pd.concat(
            [
                _schedule(competition, s, playoff=playoffs)
                for s in seasons
                if competition == "euroleague" or s >= gbl_from
            ],
            ignore_index=True,
        )
        mask = (pop["competition"] == competition).to_numpy()
        lines = _lines(
            rng, rates[mask].reset_index(drop=True), pop[mask].reset_index(drop=True), schedule
        )
        if playoffs:
            extra = pd.concat(
                [
                    _playoff_lines(competition, s, schedule, rates)
                    for s in seasons
                    if competition == "euroleague" or s >= gbl_from
                ]
            )
            lines = pd.concat([lines, extra], ignore_index=True)
        lines["player_id"] = [
            f"{'P' if competition == 'euroleague' else 'G'}{p:06d}" for p in lines["pid"]
        ]
        games[competition] = schedule
        player_games[competition] = lines[
            ["season", "game_id", "team", "player_id", "sec", "poss", *COUNT_COLUMNS]
        ].reset_index(drop=True)
    ages = _ages(pop)
    brapm = _brapm(rng, rates, pop, player_games["euroleague"], brapm_from)
    first = translations_from if translations_from is not None else seasons[1]
    translations = {
        s: Translation(delta=DELTA, c=PSEUDO, target_season=s) for s in seasons if s >= first
    }
    return M6Inputs(
        games=games,
        player_games=player_games,
        xwalk=_xwalk(pop),
        ages=ages,
        brapm=brapm,
        spm=linear_spm,
        translations=translations,
    )


def _xwalk(pop: pd.DataFrame) -> pd.DataFrame:
    """One crosswalk row per (competition, source id); a player of both leagues shares the
    EuroLeague person id."""
    in_el = set(pop.loc[pop["competition"] == "euroleague", "pid"])
    rows = []
    for competition, prefix in (("euroleague", "P"), ("gbl", "G")):
        for pid in sorted(set(pop.loc[pop["competition"] == competition, "pid"])):
            person = f"P:P{pid:06d}" if pid in in_el else f"G:G{pid:06d}"
            rows.append((competition, f"{prefix}{pid:06d}", person))
    return pd.DataFrame(rows, columns=["competition", "source_id", "person_id"])


def _person(pop_pid: pd.Series, in_el: set[int]) -> pd.Series:
    return pop_pid.map(lambda pid: f"P:P{pid:06d}" if pid in in_el else f"G:G{pid:06d}")


def _ages(pop: pd.DataFrame) -> pd.DataFrame:
    in_el = set(pop.loc[pop["competition"] == "euroleague", "pid"])
    ages = pd.DataFrame(
        {
            "person_id": _person(pop["pid"], in_el).to_numpy(),
            "season": pop["season"].astype("int64").to_numpy(),
            "age": (pop["offset"] + pop["season"]).round(3).to_numpy(),
        }
    )
    return ages.drop_duplicates(["person_id", "season"]).reset_index(drop=True)


def _brapm(
    rng: np.random.Generator,
    rates: pd.DataFrame,
    pop: pd.DataFrame,
    lines: pd.DataFrame,
    brapm_from: int,
) -> pd.DataFrame:
    """Season-end BRAPM snapshots of the EuroLeague player-seasons: the planted SPM of the true
    rates, a persistent player effect and noise shrinking with the possessions."""
    in_el = set(pop.loc[pop["competition"] == "euroleague", "pid"])
    el = rates[(rates["competition"] == "euroleague") & (rates["season"] >= brapm_from)]
    poss = lines[lines["sec"] > 0].groupby(["player_id", "season"])["poss"].sum()
    keys = [f"P{p:06d}" for p in el["pid"]]
    exposure = pd.Series(
        [poss.get((k, s), 0.0) for k, s in zip(keys, el["season"], strict=True)], index=el.index
    )
    keep = (exposure > 0).to_numpy()
    el, exposure = el[keep], exposure[keep]
    effect = {pid: float(rng.normal(0.0, BRAPM_EFFECT_SD)) for pid in sorted(in_el)}
    truth = linear_spm(el, 0).to_numpy()
    sd = BRAPM_SD * np.sqrt(1000.0 / exposure.to_numpy())
    value = truth + el["pid"].map(effect).to_numpy() + rng.normal(0.0, 1.0, len(el)) * sd
    return pd.DataFrame(
        {
            "person_id": _person(el["pid"], in_el).to_numpy(),
            "competition": "euroleague",
            "season": el["season"].astype("int64").to_numpy(),
            "stat": "brapm",
            "value": value,
            "sd": sd,
        }
    )
