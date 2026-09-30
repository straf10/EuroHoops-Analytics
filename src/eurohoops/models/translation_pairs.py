"""M4 league-translation pair tables (weeks 12-14 I6).

Per person-season-competition per-100 rates through ``player_xwalk``, then dual and mover
pairs for the I-j translation fit.

Rules (I-h / I-i / I-j):

- Rates: ``rate_<stat> = 100 * sum(stat) / sum(poss)`` over a person's games in one
  competition-season; ``team`` is the team with the most seconds (ties -> smallest code).
- Dual: same person, same season, ``>= MIN_FROM`` minutes in both leagues.
- ``gbl_to_el``: ``>= MIN_FROM`` GBL minutes in ``s``, ``< MAX_OTHER`` EL minutes in ``s``
  (0 if absent), ``>= MIN_TO`` EL minutes in ``s+1``.
- ``el_to_gbl``: the same thresholds with the leagues swapped.
- ``later_season = max(season_gbl, season_el)`` (walk-forward key: a fit for EuroLeague target
  season ``t`` may only use pairs with ``later_season < t``).
- ``team_gap`` = prior-season relative net of the EL side's team minus the GBL side's
  (season before each side's season; a missing prior-season team row counts as 0).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import pandera.pandas as pa

from eurohoops.models.box_impact import STAT_COLUMNS
from eurohoops.parse.schemas import validated

MIN_FROM = 300.0
MAX_OTHER = 100.0
MIN_TO = 300.0
PAIR_TYPES = ("dual", "gbl_to_el", "el_to_gbl")
COMPETITIONS = ("euroleague", "gbl")

_RATE_COLUMNS = tuple(f"rate_{stat}" for stat in STAT_COLUMNS)
_RATE_GBL_COLUMNS = tuple(f"rate_gbl_{stat}" for stat in STAT_COLUMNS)
_RATE_EL_COLUMNS = tuple(f"rate_el_{stat}" for stat in STAT_COLUMNS)
PairType = Literal["dual", "gbl_to_el", "el_to_gbl"]

SEASON_RATES_SCHEMA = pa.DataFrameSchema(
    {
        "person_id": pa.Column(str),
        "competition": pa.Column(str, pa.Check.isin(COMPETITIONS)),
        "season": pa.Column("int64"),
        "team": pa.Column(str),
        "minutes": pa.Column("float64", pa.Check.ge(0.0)),
        "poss": pa.Column("float64", pa.Check.ge(0.0)),
        **{name: pa.Column("float64") for name in _RATE_COLUMNS},
    },
    unique=["person_id", "competition", "season"],
    strict=True,
)

TEAM_NET_SCHEMA = pa.DataFrameSchema(
    {
        "competition": pa.Column(str, pa.Check.isin(COMPETITIONS)),
        "season": pa.Column("int64"),
        "team": pa.Column(str),
        "net": pa.Column("float64"),
    },
    unique=["competition", "season", "team"],
    strict=True,
)

PAIRS_SCHEMA = pa.DataFrameSchema(
    {
        "person_id": pa.Column(str),
        "pair_type": pa.Column(str, pa.Check.isin(PAIR_TYPES)),
        "season_gbl": pa.Column("int64"),
        "season_el": pa.Column("int64"),
        "minutes_gbl": pa.Column("float64", pa.Check.ge(0.0)),
        "minutes_el": pa.Column("float64", pa.Check.ge(0.0)),
        "team_gbl": pa.Column(str),
        "team_el": pa.Column(str),
        "team_gap": pa.Column("float64"),
        "later_season": pa.Column("int64"),
        **{name: pa.Column("float64") for name in _RATE_GBL_COLUMNS},
        **{name: pa.Column("float64") for name in _RATE_EL_COLUMNS},
    },
    unique=["person_id", "pair_type", "season_gbl", "season_el"],
    strict=True,
)


def _map_person_ids(player_games: pd.DataFrame, competition: str, xwalk: pd.DataFrame) -> pd.Series:
    """``player_id`` -> ``person_id`` for ``competition``; raises on an unmapped id."""
    mapping = (
        xwalk.loc[xwalk["competition"] == competition, ["source_id", "person_id"]]
        .drop_duplicates("source_id")
        .set_index("source_id")["person_id"]
    )
    ids = player_games["player_id"].astype(str)
    mapped = ids.map(mapping)
    missing = ids[mapped.isna()].unique()
    if len(missing):
        raise ValueError(f"unmapped player_id {missing[0]!r} for competition {competition!r}")
    return mapped.astype(str)


def _primary_team(sec_by_team: pd.Series) -> str:
    """Team with the most seconds; ties -> smallest code."""
    ordered = sec_by_team.sort_values(ascending=False)
    top = float(ordered.iloc[0])
    tied = ordered[ordered == top]
    return str(sorted(tied.index.astype(str))[0])


def _person_seasons(indexed: pd.DataFrame, person_id: str) -> set[int]:
    mask = indexed.index.get_level_values(0) == person_id
    if not mask.any():
        return set()
    return {int(s) for s in indexed.index[mask].get_level_values(1)}


def _rate_row(indexed: pd.DataFrame, person_id: str, season: int) -> pd.Series:
    return cast(pd.Series, indexed.loc[(person_id, season)])


def _minutes_in(indexed: pd.DataFrame, person_id: str, season: int, seasons: set[int]) -> float:
    if season not in seasons:
        return 0.0
    return float(cast(Any, _rate_row(indexed, person_id, season)["minutes"]))


def season_rates(player_games: pd.DataFrame, competition: str, xwalk: pd.DataFrame) -> pd.DataFrame:
    """One row per (person_id, competition, season) with per-100 rates and primary team."""
    if competition not in COMPETITIONS:
        raise ValueError(f"competition must be one of {COMPETITIONS}, got {competition!r}")
    if player_games.empty:
        return validated(
            pd.DataFrame(columns=list(SEASON_RATES_SCHEMA.columns)), SEASON_RATES_SCHEMA
        )

    rows = player_games.copy()
    rows["person_id"] = _map_person_ids(rows, competition, xwalk)
    rows["competition"] = competition

    by_person = rows.groupby(["person_id", "competition", "season"], as_index=False).agg(
        sec=("sec", "sum"),
        poss=("poss", "sum"),
        **{stat: (stat, "sum") for stat in STAT_COLUMNS},
    )
    team_sec = rows.groupby(["person_id", "competition", "season", "team"])["sec"].sum()
    team_rows: list[dict[str, object]] = []
    for key, sec_by_team in team_sec.groupby(level=[0, 1, 2]):
        person_id, _comp, season = cast(tuple[object, object, object], key)
        team_rows.append(
            {
                "person_id": str(person_id),
                "competition": competition,
                "season": int(cast(Any, season)),
                "team": _primary_team(sec_by_team.droplevel([0, 1, 2])),
            }
        )
    team_frame = pd.DataFrame(team_rows)
    out = by_person.merge(team_frame, on=["person_id", "competition", "season"], how="left")
    out["minutes"] = out["sec"].astype("float64") / 60.0
    poss = out["poss"].to_numpy(dtype=np.float64)
    for stat in STAT_COLUMNS:
        totals = out[stat].to_numpy(dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            rate = 100.0 * totals / poss
        out[f"rate_{stat}"] = np.where(poss > 0, rate, 0.0)

    frame = out[list(SEASON_RATES_SCHEMA.columns)].sort_values(
        ["person_id", "competition", "season"]
    )
    return validated(frame.reset_index(drop=True), SEASON_RATES_SCHEMA)


def team_net(team_games: pd.DataFrame) -> pd.DataFrame:
    """Relative net rating per 100 possessions, one row per (competition, season, team).

    ``net = 100 * sum(points - opponent_points) / sum(poss_game)`` minus the competition-
    season mean of that quantity over teams (so league average is 0).
    """
    if team_games.empty:
        return validated(pd.DataFrame(columns=list(TEAM_NET_SCHEMA.columns)), TEAM_NET_SCHEMA)

    opp = team_games[["game_id", "team", "points"]].rename(
        columns={"team": "opponent", "points": "opp_points"}
    )
    rows = team_games.merge(opp, on=["game_id", "opponent"], how="inner")
    grouped = rows.groupby(["competition", "season", "team"], as_index=False).agg(
        points=("points", "sum"),
        opp_points=("opp_points", "sum"),
        poss=("poss_game", "sum"),
    )
    grouped["net_raw"] = 100.0 * (grouped["points"] - grouped["opp_points"]) / grouped["poss"]
    mean_net = grouped.groupby(["competition", "season"])["net_raw"].transform("mean")
    grouped["net"] = grouped["net_raw"] - mean_net
    frame = grouped[list(TEAM_NET_SCHEMA.columns)].sort_values(["competition", "season", "team"])
    return validated(frame.reset_index(drop=True), TEAM_NET_SCHEMA)


def _lookup_net(
    net_map: dict[tuple[str, int, str], float], competition: str, season: int, team: str
) -> float:
    return net_map.get((competition, season, team), 0.0)


def _pair_row(
    *,
    person_id: str,
    pair_type: PairType,
    gbl: pd.Series,
    el: pd.Series,
    season_gbl: int,
    season_el: int,
    net_map: dict[tuple[str, int, str], float],
) -> dict[str, object]:
    team_gbl = str(gbl["team"])
    team_el = str(el["team"])
    gap = _lookup_net(net_map, "euroleague", season_el - 1, team_el) - _lookup_net(
        net_map, "gbl", season_gbl - 1, team_gbl
    )
    row: dict[str, object] = {
        "person_id": person_id,
        "pair_type": pair_type,
        "season_gbl": season_gbl,
        "season_el": season_el,
        "minutes_gbl": float(gbl["minutes"]),
        "minutes_el": float(el["minutes"]),
        "team_gbl": team_gbl,
        "team_el": team_el,
        "team_gap": float(gap),
        "later_season": max(season_gbl, season_el),
    }
    for col in _RATE_GBL_COLUMNS:
        row[col] = float(gbl[col.replace("rate_gbl_", "rate_")])
    for col in _RATE_EL_COLUMNS:
        row[col] = float(el[col.replace("rate_el_", "rate_")])
    return row


@dataclass(frozen=True)
class _PersonRates:
    person_id: str
    gbl: pd.DataFrame
    el: pd.DataFrame
    gbl_seasons: set[int]
    el_seasons: set[int]
    net_map: dict[tuple[str, int, str], float]


def _duals_for_person(ctx: _PersonRates) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for season in sorted(ctx.gbl_seasons & ctx.el_seasons):
        g_row = _rate_row(ctx.gbl, ctx.person_id, season)
        e_row = _rate_row(ctx.el, ctx.person_id, season)
        if float(g_row["minutes"]) >= MIN_FROM and float(e_row["minutes"]) >= MIN_FROM:
            out.append(
                _pair_row(
                    person_id=ctx.person_id,
                    pair_type="dual",
                    gbl=g_row,
                    el=e_row,
                    season_gbl=int(season),
                    season_el=int(season),
                    net_map=ctx.net_map,
                )
            )
    return out


def _gbl_to_el_for_person(ctx: _PersonRates) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for season in sorted(ctx.gbl_seasons):
        if season + 1 not in ctx.el_seasons:
            continue
        g_row = _rate_row(ctx.gbl, ctx.person_id, season)
        if float(g_row["minutes"]) < MIN_FROM:
            continue
        if _minutes_in(ctx.el, ctx.person_id, season, ctx.el_seasons) >= MAX_OTHER:
            continue
        e_row = _rate_row(ctx.el, ctx.person_id, season + 1)
        if float(e_row["minutes"]) < MIN_TO:
            continue
        out.append(
            _pair_row(
                person_id=ctx.person_id,
                pair_type="gbl_to_el",
                gbl=g_row,
                el=e_row,
                season_gbl=int(season),
                season_el=int(season) + 1,
                net_map=ctx.net_map,
            )
        )
    return out


def _el_to_gbl_for_person(ctx: _PersonRates) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for season in sorted(ctx.el_seasons):
        if season + 1 not in ctx.gbl_seasons:
            continue
        e_row = _rate_row(ctx.el, ctx.person_id, season)
        if float(e_row["minutes"]) < MIN_FROM:
            continue
        if _minutes_in(ctx.gbl, ctx.person_id, season, ctx.gbl_seasons) >= MAX_OTHER:
            continue
        g_row = _rate_row(ctx.gbl, ctx.person_id, season + 1)
        if float(g_row["minutes"]) < MIN_TO:
            continue
        out.append(
            _pair_row(
                person_id=ctx.person_id,
                pair_type="el_to_gbl",
                gbl=g_row,
                el=e_row,
                season_gbl=int(season) + 1,
                season_el=int(season),
                net_map=ctx.net_map,
            )
        )
    return out


def build_pairs(rates: pd.DataFrame, net: pd.DataFrame) -> pd.DataFrame:
    """Dual and mover pairs from season rates and relative team nets."""
    empty = validated(pd.DataFrame(columns=list(PAIRS_SCHEMA.columns)), PAIRS_SCHEMA)
    if rates.empty:
        return empty

    gbl = rates[rates["competition"] == "gbl"].set_index(["person_id", "season"]).sort_index()
    el = rates[rates["competition"] == "euroleague"].set_index(["person_id", "season"]).sort_index()
    net_map = {
        (str(r.competition), int(cast(Any, r.season)), str(r.team)): float(cast(Any, r.net))
        for r in net.itertuples(index=False)
    }

    out: list[dict[str, object]] = []
    people = sorted(
        set(gbl.index.get_level_values(0).astype(str))
        | set(el.index.get_level_values(0).astype(str))
    )
    for person_id in people:
        ctx = _PersonRates(
            person_id=person_id,
            gbl=gbl,
            el=el,
            gbl_seasons=_person_seasons(gbl, person_id),
            el_seasons=_person_seasons(el, person_id),
            net_map=net_map,
        )
        out.extend(_duals_for_person(ctx))
        out.extend(_gbl_to_el_for_person(ctx))
        out.extend(_el_to_gbl_for_person(ctx))

    if not out:
        return empty
    frame = pd.DataFrame(out, columns=list(PAIRS_SCHEMA.columns))
    frame = frame.sort_values(["later_season", "person_id", "pair_type"]).reset_index(drop=True)
    return validated(frame, PAIRS_SCHEMA)
