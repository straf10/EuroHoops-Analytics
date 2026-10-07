"""M6 "plays like": a player-season embedding and a k-nearest-neighbour search (weeks 16-18 L4).

Not ``entity/similarity.py`` (the cross-league name matcher) and not ``stats/twins.py`` (the
shot-profile twin, untouched).

Features, one vector per player-season (``raw_features``):

- ``BOX_FEATURES``: per-100-possession rates of pts, fg2a, fg3a, fta, oreb, dreb, ast, stl, blk,
  tov, pf; the percentages ts, fg3, ft (as in ``player_seasons.rate_table``); and usage =
  100 * (fg2a + fg3a + 0.44 fta + tov) / poss.
- ``SHOT_FEATURES`` (EuroLeague, needs the shot frame): the share of located attempts in each of
  ``stats.twins.ZONES`` (square roots, as the twins do, to steady small shares) and shot-making =
  100 * mean(made * value - xPTS), points above M2's expected points per 100 FGA. A player-season
  with fewer than ``MIN_SHOTS`` located shots has no shot features (NaN).

The shot frame is pre-mapped: the caller turns the shooter's source id into ``person_id`` with
``player_seasons.person_ids(shots["shooter"], "euroleague", xwalk)`` and joins ``shot_xpts``
(``xpts = p_make * value``) first. Columns needed: person_id, season, made, value, x, y, band,
xpts. Every shot is a EuroLeague shot.

Standardisation: ``embed`` keeps the rows with ``poss >= min_poss`` whose season is in
``pool_seasons`` (the pool: the only rows that are searched and the only rows that set the
mean and sd; a zero sd becomes 1). Rows of other seasons change nothing, so a pool of seasons
before a cutoff is a pure function of those seasons. Missing values: a zero-attempt percentage
(NaN) takes the pool mean, which standardises to 0. Shot features stay NaN when missing, and
the embedding carries them only if some pool row has them.

Search: ``neighbours`` standardises each query row with the pool's mean and sd (the query may
come from outside the pool, e.g. the live season) and ranks pool rows by Euclidean distance
(root mean square over the features used; equal feature weights), ties broken by the match's
(season, competition, person_id). ``score`` is the cosine similarity. A query row uses the shot
features only if it has them and the embedding carries them; then only pool rows that have
them are candidates. Otherwise (a GBL query, or a pool without shots) only BOX_FEATURES are
used, over the whole pool. A person never matches any of his own seasons.

``self_retrieval_rate`` exists only to check the embedding: it ranks the whole pool, own other
seasons included, and asks whether a player's season s finds his own season s+1 in its top k.
It is not used for the product output, where a person never matches himself.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
import pandera.pandas as pa

from eurohoops.models.elo import FloatArray
from eurohoops.models.player_seasons import COMPETITIONS, TS_FTA_WEIGHT, rate_table
from eurohoops.parse.schemas import validated
from eurohoops.stats.twins import ZONES, shot_zones

BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.intp]
KEYS = ["person_id", "competition", "season"]
BOX_FEATURES = (
    "pts",
    "fg2a",
    "fg3a",
    "fta",
    "oreb",
    "dreb",
    "ast",
    "stl",
    "blk",
    "tov",
    "pf",
    "ts",
    "fg3",
    "ft",
    "usage",
)
SHOT_FEATURES = (*(f"zone_{z}" for z in ZONES), "shot_making")
MIN_SHOTS = 50
SIMILAR_K = 10

SIMILAR_SCHEMA = pa.DataFrameSchema(
    {
        "person_id": pa.Column(str),
        "competition": pa.Column(str, pa.Check.isin(COMPETITIONS)),
        "season": pa.Column("int64"),
        "rank": pa.Column("int64", pa.Check.ge(1)),
        "match_person_id": pa.Column(str),
        "match_competition": pa.Column(str, pa.Check.isin(COMPETITIONS)),
        "match_season": pa.Column("int64"),
        "distance": pa.Column("float64", pa.Check.ge(0.0)),
        "score": pa.Column("float64", pa.Check.in_range(-1.0 - 1e-9, 1.0 + 1e-9)),
    },
    unique=["person_id", "competition", "season", "rank"],
    strict=True,
)


@dataclass(frozen=True)
class Embedding:
    keys: pd.DataFrame  # person_id, competition, season (row order of matrix)
    features: tuple[str, ...]
    mean: FloatArray  # from the reference pool only
    sd: FloatArray
    matrix: FloatArray  # standardised; shot features NaN where a row has none


@dataclass(frozen=True)
class SelfRetrieval:
    trials: int  # (person, season s) with his own season s+1 in the pool
    hits: int  # own season s+1 within the top k
    chance: float  # expected hits under a random ranking
    variance: float  # of the hit count under a random ranking

    def z(self) -> float:
        return (self.hits - self.chance) / float(np.sqrt(self.variance))


def _shot_features(shots: pd.DataFrame) -> pd.DataFrame:
    zone = shot_zones(shots)
    frame = pd.DataFrame(
        {
            "person_id": shots["person_id"].to_numpy(),
            "season": shots["season"].to_numpy(),
            "zone": zone.to_numpy(),
            "residual": (shots["made"] * shots["value"] - shots["xpts"]).to_numpy(dtype=float),
        }
    )
    counts = frame.pivot_table(
        index=["person_id", "season"],
        columns="zone",
        values="residual",
        aggfunc="size",
        fill_value=0,
    ).reindex(columns=list(ZONES), fill_value=0)
    total = counts.sum(axis=1)
    shares = pd.DataFrame(
        np.sqrt(counts.div(total, axis=0).to_numpy(dtype=float)),
        index=counts.index,
        columns=[f"zone_{z}" for z in ZONES],
    )
    making = frame.groupby(["person_id", "season"])["residual"].mean()
    shares["shot_making"] = 100.0 * making.reindex(counts.index).to_numpy(dtype=float)
    shares[(total < MIN_SHOTS).to_numpy()] = np.nan
    return shares.reset_index()


def raw_features(seasons: pd.DataFrame, shots: pd.DataFrame | None) -> pd.DataFrame:
    """KEYS, ``BOX_FEATURES`` and ``SHOT_FEATURES`` of each row of ``seasons`` (the
    ``PLAYER_SEASONS_SCHEMA`` frame). Unrated values are NaN; no standardisation."""
    rates = rate_table(seasons)
    out = seasons[KEYS].reset_index(drop=True)
    poss = seasons["poss"].to_numpy(dtype=float)

    def per100(count: pd.Series) -> FloatArray:
        rate: FloatArray = np.divide(
            100.0 * count.to_numpy(dtype=float),
            poss,
            out=np.full_like(poss, np.nan),
            where=poss > 0,
        )
        return rate

    attempts = seasons["fg2a"] + seasons["fg3a"] + TS_FTA_WEIGHT * seasons["fta"] + seasons["tov"]
    extra = {
        "fg2a": per100(seasons["fg2a"]),
        "pf": per100(seasons["pf"]),
        "usage": per100(attempts),
    }
    for name in BOX_FEATURES:
        out[name] = extra[name] if name in extra else rates[name].to_numpy(dtype=float)
    if shots is None or shots.empty:
        for name in SHOT_FEATURES:
            out[name] = np.nan
        return out
    profile = _shot_features(shots.assign(season=shots["season"].astype("int64")))
    keyed = out.assign(_lg=out["competition"] == "euroleague")
    merged = keyed.merge(profile, on=["person_id", "season"], how="left")
    merged.loc[~merged["_lg"], list(SHOT_FEATURES)] = np.nan
    return merged.drop(columns="_lg")


def _standardise(values: FloatArray, mean: FloatArray, sd: FloatArray, n_box: int) -> FloatArray:
    z = (values - mean) / sd
    z[:, :n_box] = np.nan_to_num(z[:, :n_box], nan=0.0)  # box NaN = pool mean
    return z


def embed(
    seasons: pd.DataFrame,
    shots: pd.DataFrame | None,
    *,
    pool_seasons: Sequence[int],
    min_poss: float = 500.0,
) -> Embedding:
    """The standardised embedding of the pool: rows of ``seasons`` in ``pool_seasons`` with at
    least ``min_poss`` possessions; ``shots`` as in the module doc (or None for box only)."""
    pool_mask = (seasons["poss"] >= min_poss) & seasons["season"].isin(list(pool_seasons))
    pool = seasons[pool_mask].sort_values(KEYS).reset_index(drop=True)
    if pool.empty:
        raise ValueError("empty similarity pool")
    only = set(pool["person_id"])
    pool_shots = None if shots is None else shots[shots["person_id"].isin(only)]
    raw = raw_features(pool, pool_shots)
    has_shots = bool(raw[list(SHOT_FEATURES)].notna().all(axis=1).any())
    features = (*BOX_FEATURES, *(SHOT_FEATURES if has_shots else ()))
    values = raw[list(features)].to_numpy(dtype=float)
    mean = np.nanmean(values, axis=0)
    sd = np.nanstd(values, axis=0)
    sd[~(sd > 0)] = 1.0
    return Embedding(
        pool[KEYS].reset_index(drop=True),
        features,
        mean,
        sd,
        _standardise(values, mean, sd, len(BOX_FEATURES)),
    )


def _rank(distance: FloatArray, keys: pd.DataFrame, candidates: BoolArray) -> IntArray:
    """Candidate row indices in order: distance, then match season, competition, person_id."""
    rows = np.flatnonzero(candidates)
    order = np.lexsort(
        (
            keys["person_id"].to_numpy(str)[rows],
            keys["competition"].to_numpy(str)[rows],
            keys["season"].to_numpy(np.int64)[rows],
            distance[rows],
        )
    )
    return rows[order]


def _cosine(query: FloatArray, pool: FloatArray) -> FloatArray:
    norm = np.linalg.norm(pool, axis=1) * np.linalg.norm(query)
    dots = pool @ query
    return np.divide(dots, norm, out=np.zeros_like(dots), where=norm > 0)


def neighbours(query: pd.DataFrame, emb: Embedding, k: int = SIMILAR_K) -> pd.DataFrame:
    """The ``k`` nearest pool rows of each query row (``SIMILAR_SCHEMA``). ``query``: KEYS plus
    ``raw_features`` columns (``BOX_FEATURES`` required; shot columns optional)."""
    n_box = len(BOX_FEATURES)
    has_shots = len(emb.features) > n_box
    shot_cols = list(SHOT_FEATURES)
    box = _standardise(
        query[list(BOX_FEATURES)].to_numpy(dtype=float), emb.mean[:n_box], emb.sd[:n_box], n_box
    )
    if has_shots and set(shot_cols) <= set(query.columns):
        shot = query[shot_cols].to_numpy(dtype=float)
        shot = (shot - emb.mean[n_box:]) / emb.sd[n_box:]
    else:
        shot = np.full((len(query), len(shot_cols)), np.nan)
    pool_has_shots = np.isfinite(emb.matrix[:, n_box:]).all(axis=1)
    pool_persons = emb.keys["person_id"].to_numpy(str)
    rows = []
    for i, (person, competition, season) in enumerate(
        zip(query["person_id"], query["competition"], query["season"], strict=True)
    ):
        use_shots = has_shots and bool(np.isfinite(shot[i]).all())
        vector = np.concatenate([box[i], shot[i]]) if use_shots else box[i]
        matrix = emb.matrix[:, : len(vector)]
        candidates = pool_persons != str(person)  # never one of his own seasons
        if use_shots:
            candidates = candidates & pool_has_shots
        distance = np.sqrt(((matrix - vector) ** 2).mean(axis=1))
        cosine = _cosine(vector, matrix)
        for rank, j in enumerate(_rank(distance, emb.keys, candidates)[:k], start=1):
            rows.append(
                {
                    "person_id": str(person),
                    "competition": str(competition),
                    "season": int(season),
                    "rank": rank,
                    "match_person_id": str(emb.keys["person_id"].iloc[j]),
                    "match_competition": str(emb.keys["competition"].iloc[j]),
                    "match_season": int(emb.keys["season"].iloc[j]),
                    "distance": float(distance[j]),
                    "score": float(cosine[j]),
                }
            )
    return validated(pd.DataFrame(rows, columns=list(SIMILAR_SCHEMA.columns)), SIMILAR_SCHEMA)


def self_retrieval_rate(emb: Embedding, k: int = SIMILAR_K) -> SelfRetrieval:
    """How often a pool row's own next season is in its top ``k`` over the whole pool (box
    features, the row itself left out, the person's other seasons allowed). Only a check on the
    embedding: ``neighbours`` never returns a person's own seasons.

    A trial is a pool row (person, competition, s) whose person has a pool row at s + 1 in the
    same competition. Under a random ranking its hit probability is ``k / (n - 1)`` and the hits
    are a sum of independent Bernoullis, so ``chance`` and ``variance`` are exact sums."""
    matrix = emb.matrix[:, : len(BOX_FEATURES)]
    keys = emb.keys
    index = {
        (p, c, s): i
        for i, (p, c, s) in enumerate(
            zip(keys["person_id"], keys["competition"], keys["season"], strict=True)
        )
    }
    n = len(keys)
    p_random = min(1.0, k / (n - 1))
    hits = trials = 0
    for (p, c, s), i in index.items():
        target = index.get((p, c, s + 1))
        if target is None:
            continue
        distance = np.sqrt(((matrix - matrix[i]) ** 2).mean(axis=1))
        candidates = np.arange(n) != i
        top = _rank(distance, keys, candidates)[:k]
        trials += 1
        hits += int(target in top)
    return SelfRetrieval(trials, hits, trials * p_random, trials * p_random * (1.0 - p_random))
