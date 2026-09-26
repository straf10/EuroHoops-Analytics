"""Shot Profile Twin: a player's recent shot profile matched against every player-season.

A profile sums a set of a player's games (one season, or his last 5/10/20 games, which may
cross seasons) into ``COUNT_FIELDS``, and reads three feature groups from the counts:

- where: the share of located attempts in each of ``ZONES`` (distance bands split by side,
  threes into corners, above the break and deep), as square roots to steady small shares;
- how well: makes minus the league's expected makes per band (league FG% of that band in
  each shot's season), over attempts plus ``PRIOR_ATTEMPTS``, so small samples shrink to 0;
- style: 3PA rate and FT rate (FTA per FGA) from the box score.

Features are standardised over the pool (player-seasons with at least ``MIN_POOL_ATT`` located
attempts). The distance is the ``WEIGHTS``-weighted mean squared difference per group, and a
twin's match is ``100 exp(-d^2 / 2)``: about 37 for two random pool rows, 100 for identical.
A window never matches the seasons its own games come from; his other seasons can.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from eurohoops.parse.shots import CORNER_END_Y_M
from eurohoops.stats.shots import BANDS

ZONES = (
    "rim",
    "short_l",
    "short_c",
    "short_r",
    "mid_l",
    "mid_c",
    "mid_r",
    "long2_l",
    "long2_c",
    "long2_r",
    "corner3_l",
    "three_c",
    "corner3_r",
    "deep3",
)
ZONE_BAND = np.array(
    [
        [BANDS.index(z.split("_")[0].replace("corner3", "three")) == b for b in range(len(BANDS))]
        for z in ZONES
    ],
    dtype=float,
)  # zones x bands
COUNT_FIELDS = (
    *(f"att_{z}" for z in ZONES),
    *(f"made_{b}" for b in BANDS),
    *(f"exp_{b}" for b in BANDS),
    "fga",
    "fg3a",
    "fta",
)
SIDE_DEG = 45.0  # twos wider than this from the basket's axis are left or right
PRIOR_ATTEMPTS = 10.0
WEIGHTS = {"where": 0.6, "how_well": 0.25, "style": 0.15}
MIN_POOL_ATT = 150
MIN_WINDOW_ATT = 25
WINDOW_GAMES = (5, 10, 20)
TOP = 5

N_ZONES, N_BANDS = len(ZONES), len(BANDS)
SHOT_COLS = N_ZONES + 2 * N_BANDS


def shot_zones(shots: pd.DataFrame) -> pd.Series:
    """Zone of each shot (``ZONES``); x < 0 is the left side."""
    angle = np.degrees(np.arctan2(shots["x"], shots["y"]))
    side = np.select([angle < -SIDE_DEG, angle > SIDE_DEG], ["l", "r"], "c")
    corner = np.where(shots["x"] < 0, "corner3_l", "corner3_r")
    band = shots["band"].to_numpy(str)
    zone = np.select(
        [
            np.isin(band, ["rim", "deep3"]),
            (band == "three") & (shots["y"] < CORNER_END_Y_M),
            band == "three",
        ],
        [band, corner, "three_c"],
        default=np.char.add(np.char.add(band, "_"), side.astype(str)),
    )
    return pd.Series(zone, index=shots.index, dtype=str)


def _shot_matrix(shots: pd.DataFrame) -> np.ndarray[Any, Any]:
    """Per shot: zone one-hot, made per band, league-expected makes per band."""
    league = shots.groupby(["season", "band"])["made"].mean()
    expected = league.reindex(pd.MultiIndex.from_frame(shots[["season", "band"]])).to_numpy()
    zone = shot_zones(shots).map({z: i for i, z in enumerate(ZONES)}).to_numpy()
    band = shots["band"].map({b: i for i, b in enumerate(BANDS)}).to_numpy()
    rows = np.arange(len(shots))
    out = np.zeros((len(shots), SHOT_COLS))
    out[rows, zone] = 1.0
    out[rows, N_ZONES + band] = shots["made"].to_numpy(float)
    out[rows, N_ZONES + N_BANDS + band] = expected
    return out


def game_counts(lines: pd.DataFrame, shots: pd.DataFrame) -> pd.DataFrame:
    """``COUNT_FIELDS`` per player game line, in the lines' (game) order."""
    per_shot = pd.DataFrame(_shot_matrix(shots), columns=list(COUNT_FIELDS[:SHOT_COLS]))
    per_shot[["player_id", "game_id"]] = shots[["player_id", "game_id"]].to_numpy()
    summed = per_shot.groupby(["player_id", "game_id"]).sum()
    keys = pd.MultiIndex.from_frame(lines[["player_id", "game_id"]])
    counts = summed.reindex(keys, fill_value=0.0).reset_index(drop=True)
    counts["fga"] = (lines["fg2a"] + lines["fg3a"]).to_numpy(float)
    counts["fg3a"] = lines["fg3a"].to_numpy(float)
    counts["fta"] = lines["fta"].to_numpy(float)
    meta = lines[["player_id", "season", "team"]].reset_index(drop=True)
    return pd.concat([meta, counts], axis=1)


def features(counts: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Rows of ``COUNT_FIELDS`` -> where (zones), how well (bands) and style (2) features."""
    zones = counts[:, :N_ZONES]
    made = counts[:, N_ZONES : N_ZONES + N_BANDS]
    expected = counts[:, N_ZONES + N_BANDS : SHOT_COLS]
    fga, fg3a, fta = counts[:, SHOT_COLS], counts[:, SHOT_COLS + 1], counts[:, SHOT_COLS + 2]
    att = zones.sum(axis=1, keepdims=True)
    where = np.sqrt(np.divide(zones, att, out=np.zeros_like(zones), where=att > 0))
    how_well = (made - expected) / (zones @ ZONE_BAND + PRIOR_ATTEMPTS)
    per_fga = np.where(fga > 0, 1.0 / np.where(fga > 0, fga, 1.0), 0.0)
    style = np.column_stack([fg3a * per_fga, fta * per_fga])
    return np.hstack([where, how_well, style])


GROUPS = {
    "where": slice(0, N_ZONES),
    "how_well": slice(N_ZONES, N_ZONES + N_BANDS),
    "style": slice(N_ZONES + N_BANDS, N_ZONES + N_BANDS + 2),
}


def distances(query: np.ndarray[Any, Any], pool: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Weighted distance from one standardised feature row to every standardised pool row."""
    diff2 = (pool - query) ** 2
    total = sum(w * diff2[:, GROUPS[g]].mean(axis=1) for g, w in WEIGHTS.items())
    return np.sqrt(np.asarray(total))


def match(distance: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    return 100.0 * np.exp(-(distance**2) / 2.0)


def _round(values: np.ndarray[Any, Any]) -> list[int | float]:
    return [int(v) if float(v).is_integer() else round(float(v), 1) for v in values]


def twins_payload(
    lines: pd.DataFrame,
    shots: pd.DataFrame,
    codes: Mapping[str, str],
    min_pool_att: int = MIN_POOL_ATT,
    min_window_att: int = MIN_WINDOW_ATT,
) -> dict[str, Any]:
    """Pool rows and, per player, each window's counts and closest ``TOP`` pool rows.

    ``lines``: player game lines in game order (``player_id game_id season team fg2a fg3a fta``).
    """
    games = game_counts(lines, shots)
    count_cols = list(COUNT_FIELDS)
    seasons = games.groupby(["player_id", "season"], sort=True)
    totals = seasons[count_cols].sum()
    played = seasons.size()
    team = seasons["team"].agg(lambda t: t.value_counts(sort=True).index[0])
    totals = totals[totals[count_cols[:N_ZONES]].sum(axis=1) >= min_pool_att]
    pool_keys = [(str(p), int(str(s))) for p, s in totals.index]
    pool_at = {k: i for i, k in enumerate(pool_keys)}
    pool_counts = totals.to_numpy()
    pool_feats = features(pool_counts)
    if pool_keys:
        mean, std = pool_feats.mean(axis=0), pool_feats.std(axis=0)
        std[std == 0] = 1.0
    else:  # nothing to match against (e.g. a small fixture); every window is left out
        mean, std = np.zeros(pool_feats.shape[1]), np.ones(pool_feats.shape[1])
    pool_z = (pool_feats - mean) / std

    def window(pid: str, rows: pd.DataFrame) -> dict[str, Any] | None:
        counts = rows[count_cols].to_numpy().sum(axis=0)
        if counts[:N_ZONES].sum() < min_window_att or not pool_keys:
            return None
        dist = distances((features(counts[None, :])[0] - mean) / std, pool_z)
        own = [pool_at[(pid, s)] for s in {int(s) for s in rows["season"]} if (pid, s) in pool_at]
        dist[own] = np.inf  # a window never matches the seasons its own games come from
        order = [int(i) for i in np.argsort(dist, kind="stable")[:TOP] if np.isfinite(dist[i])]
        return {
            "from": int(rows["season"].iloc[0]),
            "to": int(rows["season"].iloc[-1]),
            "games": len(rows),
            "counts": _round(counts),
            "twins": [[i, round(float(match(dist[i])), 1)] for i in order],
        }

    players: dict[str, dict[str, Any]] = {}
    for key, rows in games.groupby("player_id", sort=True):
        pid = str(key)
        spans = {f"last{n}": rows.tail(n) for n in WINDOW_GAMES if len(rows) > n}
        located = rows.groupby("season")[count_cols[:N_ZONES]].sum().sum(axis=1)
        enough = located.index[located >= min_window_att]
        if len(enough):  # his latest season with enough attempts (a new one starts small)
            spans["season"] = rows[rows["season"] == enough[-1]]
        found = {name: window(pid, span) for name, span in spans.items()}
        if any(found.values()):
            players[pid] = {name: w for name, w in found.items() if w is not None}
    return {
        "zones": list(ZONES),
        "bands": list(BANDS),
        "count_fields": count_cols,
        "windows": [*(f"last{n}" for n in WINDOW_GAMES), "season"],
        "min_pool_att": min_pool_att,
        "min_window_att": min_window_att,
        "pool_fields": ["id", "season", "team", "games", "counts"],
        "pool": [
            [p, s, codes.get(str(team[(p, s)]), str(team[(p, s)])), int(played[(p, s)]), _round(c)]
            for (p, s), c in zip(pool_keys, pool_counts, strict=True)
        ],
        "players": players,
    }
