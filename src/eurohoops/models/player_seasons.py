"""M6 player-season input frame (weeks 16-18 L0): the one table projections (L1), the aging
curve (L2), the over/under board (L3) and the similarity search (L4) all read.

One row per person, competition and season: box-score counts summed over the games passed in,
possessions as exposure (``poss`` = team possessions x minutes share, as in M3/M4), the team with
the most seconds, and the person's first season in either league among the rows passed.
Per-100 rates are ``100 * count / poss``; percentages are made / attempted
(``ts`` = pts / (2 * tsa), ``tsa`` = fga + 0.44 * fta).

Walk-forward: the frame is a pure function of the player games passed in. A caller that needs
the state at a cutoff filters the games first (``partial=True`` marks a season cut at a
checkpoint, so a projection can tell it from a finished season).

Person ids come from ``player_xwalk``; a source id the crosswalk does not cover (the live
season's newcomers: the crosswalk is built on 2007-2025) gets ``P:<id>`` (EuroLeague) or
``G:<id>`` (GBL), the crosswalk's own scheme, and is flagged by ``mapped=False``.

``player_ages`` turns birth dates into ages at 1 October of each season. Birth dates and exact
ages are model inputs only: they never reach a committed file, a report or the site (owner rule,
``ingest/bios.py``; L-e).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
import pandera.pandas as pa

from eurohoops.parse.schemas import validated

COMPETITIONS = ("euroleague", "gbl")
COUNT_COLUMNS = (
    "pts",
    "fg2m",
    "fg2a",
    "fg3m",
    "fg3a",
    "ftm",
    "fta",
    "oreb",
    "dreb",
    "ast",
    "stl",
    "blk",
    "tov",
    "pf",
)
# The projected stats (L-b): per-100 counts (exposure: possessions), percentages (exposure: their
# attempts) and impact ratings (points per 100, supplied by M3, not built from the box).
COUNT_STATS = ("pts", "fg3a", "fta", "ast", "tov", "oreb", "dreb", "stl", "blk")
PCT_STATS = ("ts", "fg3", "ft")
IMPACT_STATS = ("spm", "brapm")
PROJECTED_STATS = COUNT_STATS + PCT_STATS + IMPACT_STATS
TS_FTA_WEIGHT = 0.44
_PREFIX = {"euroleague": "P:", "gbl": "G:"}
AGE_DAY = (10, 1)  # ages are taken at 1 October of the season's start year
DAYS_PER_YEAR = 365.2425

PLAYER_SEASONS_SCHEMA = pa.DataFrameSchema(
    {
        "person_id": pa.Column(str),
        "competition": pa.Column(str, pa.Check.isin(COMPETITIONS)),
        "season": pa.Column("int64"),
        "partial": pa.Column(bool),
        "mapped": pa.Column(bool),
        "team": pa.Column(str),
        "games": pa.Column("int64", pa.Check.ge(1)),
        "minutes": pa.Column("float64", pa.Check.ge(0.0)),
        "poss": pa.Column("float64", pa.Check.ge(0.0)),
        **{c: pa.Column("int64", pa.Check.ge(0)) for c in COUNT_COLUMNS},
        "debut_season": pa.Column("int64"),
    },
    unique=["person_id", "competition", "season"],
    strict=True,
)

AGES_SCHEMA = pa.DataFrameSchema(
    {
        "person_id": pa.Column(str),
        "season": pa.Column("int64"),
        "age": pa.Column("float64", pa.Check.in_range(10.0, 60.0)),
    },
    unique=["person_id", "season"],
    strict=True,
)


def person_ids(player_ids: pd.Series, competition: str, xwalk: pd.DataFrame) -> pd.DataFrame:
    """``person_id`` and ``mapped`` for each source id; uncovered ids keep the xwalk's scheme."""
    mapping = (
        xwalk.loc[xwalk["competition"] == competition, ["source_id", "person_id"]]
        .drop_duplicates("source_id")
        .set_index("source_id")["person_id"]
    )
    ids = player_ids.astype(str)
    mapped = ids.map(mapping)
    return pd.DataFrame(
        {
            "person_id": mapped.fillna(_PREFIX[competition] + ids).astype(str),
            "mapped": mapped.notna().to_numpy(),
        },
        index=player_ids.index,
    )


def _primary_teams(lines: pd.DataFrame) -> pd.Series:
    """Per (person, season): the team with the most seconds; ties -> the smallest code."""
    secs = lines.groupby(["person_id", "season", "team"], as_index=False).agg(sec=("sec", "sum"))
    secs = secs.sort_values(
        ["person_id", "season", "sec", "team"], ascending=[True, True, False, True]
    )
    return secs.drop_duplicates(["person_id", "season"]).set_index(["person_id", "season"])["team"]


def build_player_seasons(
    player_games: Mapping[str, pd.DataFrame],
    xwalk: pd.DataFrame,
    *,
    partial: Mapping[str, int] | None = None,
) -> pd.DataFrame:
    """The player-season frame from each competition's player games (``stats.box`` /
    ``parse.gbl_box_lines`` rows); ``partial`` names, per competition, the season whose games
    were cut at a checkpoint. Players with no seconds in a season are left out."""
    frames = []
    for competition, games in player_games.items():
        if games.empty:
            continue
        lines = games[games["sec"] > 0].copy()
        ids = person_ids(lines["player_id"], competition, xwalk)
        lines["person_id"] = ids["person_id"]
        lines["mapped"] = ids["mapped"]
        sums = lines.groupby(["person_id", "season"], as_index=False).agg(
            mapped=("mapped", "all"),
            games=("game_id", "nunique"),
            sec=("sec", "sum"),
            poss=("poss", "sum"),
            **{c: (c, "sum") for c in COUNT_COLUMNS},
        )
        teams = _primary_teams(lines)
        sums["team"] = [
            teams[(p, s)] for p, s in zip(sums["person_id"], sums["season"], strict=True)
        ]
        sums["competition"] = competition
        cut = (partial or {}).get(competition)
        sums["partial"] = sums["season"] == cut if cut is not None else False
        frames.append(sums)
    if not frames:
        raise ValueError("no player games with seconds played")
    table = pd.concat(frames, ignore_index=True)
    table["minutes"] = table["sec"].astype("float64") / 60.0
    table["poss"] = table["poss"].astype("float64")
    table["debut_season"] = table.groupby("person_id")["season"].transform("min")
    for c in ("season", "games", *COUNT_COLUMNS, "debut_season"):
        table[c] = table[c].astype("int64")
    table = table.sort_values(["competition", "season", "person_id"]).reset_index(drop=True)
    return validated(table[list(PLAYER_SEASONS_SCHEMA.columns)], PLAYER_SEASONS_SCHEMA)


def player_ages(bios: pd.DataFrame, xwalk: pd.DataFrame, seasons: pd.DataFrame) -> pd.DataFrame:
    """Age in years at 1 October of each ``(person_id, season)`` in ``seasons``, for persons
    with a known birth date (``bios``: ``competition, source_id, birth_date``; the EuroLeague
    date wins over the GBL one). In-memory only: never written out (L-e)."""
    dated = bios.dropna(subset=["birth_date"])
    frames = []
    for competition in COMPETITIONS:
        rows = dated[dated["competition"] == competition]
        ids = person_ids(rows["source_id"], competition, xwalk)
        frames.append(pd.DataFrame({"person_id": ids["person_id"], "born": rows["birth_date"]}))
    born = pd.concat(frames, ignore_index=True).drop_duplicates("person_id")
    born_on = dict(zip(born["person_id"], pd.to_datetime(born["born"]), strict=True))
    keys = seasons[["person_id", "season"]].drop_duplicates()
    keys = keys[keys["person_id"].isin(born_on)]
    births = pd.to_datetime(keys["person_id"].map(born_on))
    start = pd.to_datetime(
        keys["season"].astype(str) + f"-{AGE_DAY[0]:02d}-{AGE_DAY[1]:02d}", format="%Y-%m-%d"
    )
    ages = (start - births).dt.days.to_numpy(dtype="float64") / DAYS_PER_YEAR
    out = pd.DataFrame(
        {
            "person_id": keys["person_id"].astype(str).to_numpy(),
            "season": keys["season"].astype("int64").to_numpy(),
            "age": ages,
        }
    )
    return validated(out.reset_index(drop=True), AGES_SCHEMA)


def rate_table(seasons: pd.DataFrame) -> pd.DataFrame:
    """Each row's projected box stats with their exposure: ``<stat>`` is the per-100 rate (counts)
    or the proportion (percentages; NaN without an attempt), ``<stat>_n`` the possessions or
    attempts behind it. Keys: person_id, competition, season, partial."""
    out = seasons[["person_id", "competition", "season", "partial"]].copy()
    poss = seasons["poss"].to_numpy(dtype="float64")
    for stat in COUNT_STATS:
        count = seasons[stat].to_numpy(dtype="float64")
        out[stat] = np.divide(100.0 * count, poss, out=np.full_like(poss, np.nan), where=poss > 0)
        out[f"{stat}_n"] = poss
    tsa = (seasons["fg2a"] + seasons["fg3a"]).to_numpy(dtype="float64") + TS_FTA_WEIGHT * seasons[
        "fta"
    ].to_numpy(dtype="float64")
    parts = {
        "ts": (seasons["pts"].to_numpy(dtype="float64") / 2.0, tsa),
        "fg3": (
            seasons["fg3m"].to_numpy(dtype="float64"),
            seasons["fg3a"].to_numpy(dtype="float64"),
        ),
        "ft": (seasons["ftm"].to_numpy(dtype="float64"), seasons["fta"].to_numpy(dtype="float64")),
    }
    for stat, (made, attempts) in parts.items():
        out[stat] = np.divide(
            made, attempts, out=np.full_like(attempts, np.nan), where=attempts > 0
        )
        out[f"{stat}_n"] = attempts
    return out.reset_index(drop=True)
