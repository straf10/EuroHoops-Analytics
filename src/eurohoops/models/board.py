"""M6 over/under-performance board (weeks 16-18 L3, PLAN R10): which players are doing better or
worse than expected on the three dimensions that are stable year to year, and how much of the
gap should be expected to last.

Dimensions, each as observed vs expected with a standard error ``se`` (so ``z = gap / se``):

- ``shot_making``: points per 100 FGA above M2's xPTS (expected 0); ``se`` = 100 sqrt(sum
  p (1 - p) value^2) / FGA over the player's shots, the sampling sd of points minus xPTS if the
  xPTS probabilities ``p`` were true. EuroLeague only (the ``shots`` mart).
- ``fg3_pct``: 3P% vs a prior from FT%: a weighted least-squares line of 3P% on FT%, fitted per
  competition on complete seasons strictly before the board's season (rows with at least
  the dimension's ``n_min`` 3PA and ``FT_MIN_FTA`` FTA). A player with fewer than ``FT_MIN_FTA`` FTA
  takes the fitted mean FT%. ``se`` is binomial at the expected rate (the null sd; the prior's own
  uncertainty is not added).
- ``on_off``: on-court minus off-court net rating per 100 possessions from the ``stints`` mart
  (games that pass ``stint_game_checks``, as M3) vs the BRAPM total of the same season (EuroLeague
  only). The net rating is 100 (points for / possessions for - points against / possessions
  against); off-court = the player's team total minus his on-court stints, summed over his
  teams. Possession-based ``se``: ``100 sqrt(2 s2 (1/n_on + 1/n_off))`` with ``s2`` the variance
  of points per possession, estimated from the season's own stints (possession-weighted spread of
  stint points per possession about their mean), ``n`` = mean of possessions for and against. It
  treats possessions as independent and ignores teammates; the BRAPM ``sd_total`` is added in
  quadrature.

Persistence (R10): ``stability`` is the year-to-year Pearson r of the gap in consecutive tuning
seasons for players with at least ``n_min`` attempts in both, and ``mean_n`` their mean n. With
the reliability model r = n / (n + k), ``k = mean_n (1 - r) / r`` and a player with n attempts
keeps ``persist = n / (n + k)`` of his gap (0 when r <= 0, 1 when r >= 1), so
``expected_next = expected + persist * gap``.

``n_min`` is derived, not typed in (rule v2, docs/models/m6.md): per dimension, the median n of
EuroLeague rotation player-seasons (``ROTATION_GAMES`` games and ``ROTATION_MINUTES`` minutes a
game) with an observation, over the tuning seasons, rounded down to a multiple of 10
(``derive_n_min``). Labels, in this order: n < ``n_min`` is "too few attempts"; |z| < ``Z_REAL`` is
"within noise"; ``persist * |z| >= Z_REAL`` (the lasting part of the gap is itself significant)
is "likely real"; otherwise "likely regression". On/off has no label (its stability is ~0).

Walk-forward: a builder reads only the rows of its board season and (3P% only) earlier seasons
for the prior; the caller passes tuning seasons only to ``stability``. Names and teams are not
kept: the report writer adds them.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
import pandera.pandas as pa

from eurohoops.models.elo import FloatArray
from eurohoops.models.player_seasons import person_ids
from eurohoops.parse.schemas import validated

DIMENSIONS = ("shot_making", "fg3_pct", "on_off")
LABELS = ("likely regression", "likely real", "too few attempts", "within noise")
UNLABELLED = ("on_off",)  # dimensions whose gap does not repeat year to year: no label
Z_REAL = 1.645
ROTATION_GAMES = 15  # a rotation player-season: at least this many games ...
ROTATION_MINUTES = 15.0  # ... and this many minutes a game
N_MIN_STEP = 10
FT_MIN_FTA = 25  # FT% is used as a predictor only from this many attempts
MIN_FIT_ROWS = 10
MIN_PAIRS = 3
P_FLOOR = 0.01  # expected 3P% is kept inside [P_FLOOR, 1 - P_FLOOR] so that its binomial se > 0

OBSERVATION_SCHEMA = pa.DataFrameSchema(
    {
        "person_id": pa.Column(str),
        "competition": pa.Column(str),
        "season": pa.Column("int64"),
        "observed": pa.Column("float64"),
        "expected": pa.Column("float64"),
        "se": pa.Column("float64", pa.Check.gt(0.0)),
        "n": pa.Column("float64", pa.Check.gt(0.0)),
    },
    unique=["person_id", "competition", "season"],
    strict=True,
)

BOARD_SCHEMA = pa.DataFrameSchema(
    {
        "person_id": pa.Column(str),
        "competition": pa.Column(str),
        "season": pa.Column("int64"),
        "dimension": pa.Column(str, pa.Check.isin(DIMENSIONS)),
        "observed": pa.Column("float64"),
        "expected": pa.Column("float64"),
        "gap": pa.Column("float64"),
        "se": pa.Column("float64", pa.Check.gt(0.0)),
        "z": pa.Column("float64"),
        "stability": pa.Column("float64"),
        "persist": pa.Column("float64", pa.Check.in_range(0.0, 1.0)),
        "expected_next": pa.Column("float64"),
        "n": pa.Column("float64", pa.Check.gt(0.0)),
        "label": pa.Column(str, pa.Check.isin(LABELS), nullable=True),  # null: no label (on/off)
    },
    unique=["person_id", "competition", "season", "dimension"],
    strict=True,
)


@dataclass(frozen=True)
class Stability:
    """Year-to-year reliability of one dimension's gap (tuning seasons only)."""

    dimension: str
    r: float
    mean_n: float
    pairs: int
    n_min: int = 0  # attempts needed for a label (derived, see ``derive_n_min``)

    def persist(self, n: FloatArray) -> FloatArray:
        """Share of the gap expected to persist at ``n`` attempts: n / (n + k)."""
        if self.r <= 0.0:
            return np.zeros_like(n, dtype=np.float64)
        if self.r >= 1.0:
            return np.ones_like(n, dtype=np.float64)
        k = self.mean_n * (1.0 - self.r) / self.r
        return np.asarray(n / (n + k), dtype=np.float64)


def rotation_player_seasons(seasons: pd.DataFrame, tuning: tuple[int, ...]) -> pd.DataFrame:
    """``person_id, season`` of the EuroLeague rotation player-seasons among ``tuning`` seasons:
    complete (not partial) seasons with at least ``ROTATION_GAMES`` games and ``ROTATION_MINUTES``
    minutes a game. ``seasons``: the I1 frame."""
    rows = seasons[
        (seasons["competition"] == "euroleague")
        & seasons["season"].isin(tuning)
        & ~seasons["partial"]
        & (seasons["games"] >= ROTATION_GAMES)
        & (seasons["minutes"] >= ROTATION_MINUTES * seasons["games"])
    ]
    return rows[["person_id", "season"]].drop_duplicates().reset_index(drop=True)


def derive_n_min(n: pd.Series) -> int:
    """The median of ``n`` (a rotation player-season each), rounded down to a multiple of
    ``N_MIN_STEP``."""
    if n.empty:
        raise ValueError("no rotation player-seasons to derive N_MIN from")
    return int(np.floor(float(n.median()) / N_MIN_STEP)) * N_MIN_STEP


def rotation_n(observations: pd.DataFrame, rotation: pd.DataFrame) -> pd.Series:
    """The ``n`` of the EuroLeague ``observations`` that are rotation player-seasons (tuning
    seasons are the caller's choice: ``rotation`` holds only those)."""
    mine = observations[observations["competition"] == "euroleague"]
    return mine.merge(rotation, on=["person_id", "season"], how="inner")["n"]


def fg3_rotation_n(seasons: pd.DataFrame, rotation: pd.DataFrame) -> pd.Series:
    """3PA of the rotation player-seasons that attempted a 3 (the 3P% rows' n, read from the I1
    frame because the 3P% prior needs ``n_min`` before its observations exist)."""
    mine = seasons[(seasons["competition"] == "euroleague") & (seasons["fg3a"] > 0)]
    return mine.merge(rotation, on=["person_id", "season"], how="inner")["fg3a"].astype("float64")


def _empty_observations() -> pd.DataFrame:
    columns = {
        "person_id": pd.Series(dtype=str),
        "competition": pd.Series(dtype=str),
        "season": pd.Series(dtype="int64"),
        "observed": pd.Series(dtype="float64"),
        "expected": pd.Series(dtype="float64"),
        "se": pd.Series(dtype="float64"),
        "n": pd.Series(dtype="float64"),
    }
    return validated(pd.DataFrame(columns), OBSERVATION_SCHEMA)


def _finish(rows: pd.DataFrame) -> pd.DataFrame:
    """Drop rows without a usable positive se / n, sort, validate."""
    keep = np.isfinite(rows["se"]) & (rows["se"] > 0) & (rows["n"] > 0)
    rows = rows[keep].sort_values(["competition", "person_id"]).reset_index(drop=True)
    return validated(rows[list(OBSERVATION_SCHEMA.columns)], OBSERVATION_SCHEMA)


def shot_making_observations(
    shots: pd.DataFrame, xpts: pd.DataFrame, xwalk: pd.DataFrame, season: int, variant: str
) -> pd.DataFrame:
    """Shot-making of each shooter in ``season``: 100 (points - xPTS) / FGA, expected 0.

    ``xpts`` rows of ``variant`` join ``shots`` on ``game_id`` + ``event``; a shot with no xPTS
    is left out. Only ``shots`` of ``season`` are read."""
    use = xpts[xpts["variant"] == variant]
    if use.duplicated(["game_id", "event"]).any():
        raise ValueError(f"shot_xpts variant {variant!r} has several rows for one shot")
    mine = shots[shots["season"] == season]
    joined = mine.merge(use[["game_id", "event", "p_make"]], on=["game_id", "event"], how="inner")
    if joined.empty:
        return _empty_observations()
    value = joined["value"].to_numpy(dtype=np.float64)
    p = joined["p_make"].to_numpy(dtype=np.float64)
    joined = joined.assign(
        points=joined["made"].to_numpy(dtype=np.float64) * value,
        xpts=p * value,
        var=p * (1.0 - p) * value**2,
    )
    frames = []
    for competition, group in joined.groupby("competition", sort=True):
        sums = group.groupby("shooter", as_index=False).agg(
            fga=("points", "size"),
            points=("points", "sum"),
            xpts=("xpts", "sum"),
            var=("var", "sum"),
        )
        ids = person_ids(sums["shooter"], str(competition), xwalk)
        fga = sums["fga"].to_numpy(dtype=np.float64)
        frames.append(
            pd.DataFrame(
                {
                    "person_id": ids["person_id"].to_numpy(),
                    "competition": str(competition),
                    "season": np.int64(season),
                    "observed": 100.0 * (sums["points"] - sums["xpts"]).to_numpy() / fga,
                    "expected": 0.0,
                    "se": 100.0 * np.sqrt(sums["var"].to_numpy(dtype=np.float64)) / fga,
                    "n": fga,
                }
            )
        )
    return _finish(pd.concat(frames, ignore_index=True))


def fg3_fit(
    seasons: pd.DataFrame, season: int, competition: str, n_min: int
) -> tuple[float, float, float]:
    """``(intercept, slope, mean_ft)`` of 3P% on FT% for ``competition``, from complete seasons
    strictly before ``season`` (rows with enough 3PA and FTA), weighted by 3PA. ``mean_ft`` is the
    weighted mean FT%, the FT% of a player with too few FTA."""
    rows = seasons[
        (seasons["competition"] == competition)
        & (seasons["season"] < season)
        & ~seasons["partial"]
        & (seasons["fg3a"] >= n_min)
        & (seasons["fta"] >= FT_MIN_FTA)
    ]
    if len(rows) < MIN_FIT_ROWS:
        raise ValueError(f"{competition}: only {len(rows)} rows before {season} for the 3P% prior")
    ft = (rows["ftm"] / rows["fta"]).to_numpy(dtype=np.float64)
    fg3 = (rows["fg3m"] / rows["fg3a"]).to_numpy(dtype=np.float64)
    weight = rows["fg3a"].to_numpy(dtype=np.float64)
    slope, intercept = np.polyfit(ft, fg3, 1, w=np.sqrt(weight))
    return float(intercept), float(slope), float(np.average(ft, weights=weight))


def fg3_observations(seasons: pd.DataFrame, season: int, n_min: int) -> pd.DataFrame:
    """3P% of each player-season of ``season`` (the I1 frame; a partial row is a partial season)
    vs the FT%-informed prior fitted before ``season``. A competition with too few earlier rows to
    fit the prior has no rows."""
    frames = []
    for competition in sorted(seasons["competition"].unique()):
        rows = seasons[
            (seasons["competition"] == competition)
            & (seasons["season"] == season)
            & (seasons["fg3a"] > 0)
        ]
        try:
            intercept, slope, mean_ft = fg3_fit(seasons, season, str(competition), n_min)
        except ValueError:
            continue
        fta = rows["fta"].to_numpy(dtype=np.float64)
        ft = np.divide(
            rows["ftm"].to_numpy(dtype=np.float64),
            fta,
            out=np.full_like(fta, mean_ft),
            where=fta >= FT_MIN_FTA,
        )
        expected = np.clip(intercept + slope * ft, P_FLOOR, 1.0 - P_FLOOR)
        n = rows["fg3a"].to_numpy(dtype=np.float64)
        frames.append(
            pd.DataFrame(
                {
                    "person_id": rows["person_id"].to_numpy(),
                    "competition": str(competition),
                    "season": np.int64(season),
                    "observed": (rows["fg3m"] / rows["fg3a"]).to_numpy(dtype=np.float64),
                    "expected": expected,
                    "se": np.sqrt(expected * (1.0 - expected) / n),
                    "n": n,
                }
            )
        )
    if not frames:
        return _empty_observations()
    return _finish(pd.concat(frames, ignore_index=True))


def _stint_sides(stints: pd.DataFrame) -> pd.DataFrame:
    """One row per stint side: team, players, points and possessions for and against."""
    sides = []
    for team, players, pts_for, poss_for, pts_against, poss_against in (
        ("home", "home_players", "home_points", "home_poss", "away_points", "away_poss"),
        ("away", "away_players", "away_points", "away_poss", "home_points", "home_poss"),
    ):
        sides.append(
            pd.DataFrame(
                {
                    "team": stints[team].astype(str).to_numpy(),
                    "players": stints[players].to_numpy(),
                    "pts_for": stints[pts_for].to_numpy(dtype=np.float64),
                    "poss_for": stints[poss_for].to_numpy(dtype=np.float64),
                    "pts_against": stints[pts_against].to_numpy(dtype=np.float64),
                    "poss_against": stints[poss_against].to_numpy(dtype=np.float64),
                }
            )
        )
    return pd.concat(sides, ignore_index=True)


_SUMS = ("pts_for", "poss_for", "pts_against", "poss_against")


def _net(sums: pd.DataFrame) -> FloatArray:
    return np.asarray(
        100.0
        * (
            sums["pts_for"].to_numpy() / sums["poss_for"].to_numpy()
            - sums["pts_against"].to_numpy() / sums["poss_against"].to_numpy()
        ),
        dtype=np.float64,
    )


def on_off_observations(
    stints: pd.DataFrame,
    checks: pd.DataFrame,
    brapm: pd.DataFrame,
    xwalk: pd.DataFrame,
    season: int,
) -> pd.DataFrame:
    """On-court minus off-court net rating of each EuroLeague player in ``season`` vs his BRAPM
    total. ``brapm``: ``player_id`` (source id), ``season``, ``total``, ``sd_total``; only the
    ``season`` rows are read. A player on court (or off court) for no possession is left out."""
    passed = set(checks.loc[checks["passed"].astype(bool), "game_id"].astype(str))
    mine = stints[(stints["season"] == season) & stints["game_id"].astype(str).isin(passed)]
    sides = _stint_sides(mine)
    sides = sides[(sides["poss_for"] > 0) & (sides["poss_against"] > 0)]
    ratings = brapm[brapm["season"] == season]
    if sides.empty or ratings.empty:
        return _empty_observations()
    ppp = (sides["pts_for"] / sides["poss_for"]).to_numpy()
    weight = sides["poss_for"].to_numpy()
    mean_ppp = float(np.average(ppp, weights=weight))
    s2 = float((weight * (ppp - mean_ppp) ** 2).sum() / (len(ppp) - 1))

    team_totals = sides.groupby("team")[list(_SUMS)].sum()
    exploded = sides.explode("players")
    exploded = exploded[exploded["players"].notna()]
    on = exploded.groupby([exploded["players"].astype(str), "team"])[list(_SUMS)].sum()
    off = team_totals.reindex(on.index.get_level_values("team")).set_axis(on.index) - on
    on_player = on.groupby(level=0).sum()
    off_player = off.groupby(level=0).sum()
    usable = (on_player[["poss_for", "poss_against"]] > 0).all(axis=1) & (
        off_player[["poss_for", "poss_against"]] > 0
    ).all(axis=1)
    on_player, off_player = on_player[usable], off_player[usable]
    n_on = 0.5 * (on_player["poss_for"] + on_player["poss_against"]).to_numpy()
    n_off = 0.5 * (off_player["poss_for"] + off_player["poss_against"]).to_numpy()
    table = pd.DataFrame(
        {
            "player_id": on_player.index.astype(str),
            "diff": _net(on_player) - _net(off_player),
            "se_diff": 100.0 * np.sqrt(2.0 * s2 * (1.0 / n_on + 1.0 / n_off)),
            "n": n_on,
        }
    ).merge(
        ratings[["player_id", "total", "sd_total"]].astype({"player_id": str}),
        on="player_id",
        how="inner",
    )
    ids = person_ids(table["player_id"], "euroleague", xwalk)
    return _finish(
        pd.DataFrame(
            {
                "person_id": ids["person_id"].to_numpy(),
                "competition": "euroleague",
                "season": np.int64(season),
                "observed": table["diff"].to_numpy(),
                "expected": table["total"].to_numpy(dtype=np.float64),
                "se": np.sqrt(table["se_diff"].to_numpy() ** 2 + table["sd_total"].to_numpy() ** 2),
                "n": table["n"].to_numpy(),
            }
        )
    )


def stability(
    observations: pd.DataFrame, dimension: str, seasons: tuple[int, ...], n_min: int
) -> Stability:
    """Year-to-year r of the gap (observed - expected) over consecutive ``seasons`` (the caller
    passes tuning seasons only), for players with at least ``n_min`` attempts in both seasons of a
    pair. ``observations``: ``OBSERVATION_SCHEMA`` rows of those seasons."""
    use = observations[observations["season"].isin(seasons) & (observations["n"] >= n_min)]
    first = use.assign(gap=use["observed"] - use["expected"])
    second = first.assign(season=first["season"] - 1)
    pairs = first.merge(second, on=["person_id", "competition", "season"], suffixes=("", "_next"))
    if len(pairs) < MIN_PAIRS:
        raise ValueError(f"{dimension}: {len(pairs)} year-to-year pairs, need {MIN_PAIRS}")
    r = float(np.corrcoef(pairs["gap"], pairs["gap_next"])[0, 1])
    mean_n = float(0.5 * (pairs["n"] + pairs["n_next"]).mean())
    return Stability(dimension, r, mean_n, len(pairs), n_min)


def _labels(
    n: FloatArray, z: FloatArray, persist: FloatArray, n_min: int, dimension: str
) -> npt.NDArray[np.object_]:
    """Rule v2, in order: too few attempts, within noise, likely real (the lasting part of the gap,
    ``persist * |z|``, is itself significant), likely regression. No label where the gap does not
    repeat year to year."""
    if dimension in UNLABELLED:
        return np.full(len(n), None, dtype=object)
    az = np.abs(z)
    label = np.where(persist * az >= Z_REAL, "likely real", "likely regression")
    label = np.where(az < Z_REAL, "within noise", label)
    return np.asarray(np.where(n < n_min, "too few attempts", label), dtype=object)


def board(
    observations: Mapping[str, pd.DataFrame], stabilities: Mapping[str, Stability]
) -> pd.DataFrame:
    """``BOARD_SCHEMA`` rows from each dimension's observations (``OBSERVATION_SCHEMA``) and its
    stability. A dimension in ``observations`` needs an entry in ``stabilities``."""
    frames = []
    for dimension, rows in observations.items():
        if dimension not in DIMENSIONS:
            raise ValueError(f"unknown dimension {dimension!r}")
        if rows.empty:
            continue
        stab = stabilities[dimension]
        n = rows["n"].to_numpy(dtype=np.float64)
        gap = (rows["observed"] - rows["expected"]).to_numpy(dtype=np.float64)
        se = rows["se"].to_numpy(dtype=np.float64)
        z = gap / se
        persist = stab.persist(n)
        label = _labels(n, z, persist, stab.n_min, dimension)
        frames.append(
            pd.DataFrame(
                {
                    "person_id": rows["person_id"].to_numpy(),
                    "competition": rows["competition"].to_numpy(),
                    "season": rows["season"].to_numpy(),
                    "dimension": dimension,
                    "observed": rows["observed"].to_numpy(),
                    "expected": rows["expected"].to_numpy(),
                    "gap": gap,
                    "se": se,
                    "z": z,
                    "stability": stab.r,
                    "persist": persist,
                    "expected_next": rows["expected"].to_numpy() + persist * gap,
                    "n": n,
                    "label": label,
                }
            )
        )
    if not frames:
        raise ValueError("no observations to put on the board")
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["dimension", "competition", "person_id"]).reset_index(drop=True)
    return validated(out, BOARD_SCHEMA)


@dataclass(frozen=True)
class RetentionBacktest:
    """How much of the gap each label kept: slope of next season's gap on this season's gap
    (through the origin; 1 = fully kept, 0 = fully regressed), its HC0 standard error and the
    number of players, and the contrast "likely real" minus "likely regression" in standard
    errors (``contrast_z``)."""

    slope: Mapping[str, float]
    se: Mapping[str, float]
    count: Mapping[str, int]
    contrast_z: float


def retention_backtest(board_rows: pd.DataFrame, next_gap: pd.DataFrame) -> RetentionBacktest:
    """Retention of the gap by label. ``next_gap``: ``person_id, competition, dimension, gap`` of
    the following season; players missing from it are left out."""
    keys = ["person_id", "competition", "dimension"]
    both = board_rows.merge(next_gap[[*keys, "gap"]], on=keys, suffixes=("", "_next"))
    slope: dict[str, float] = {}
    se: dict[str, float] = {}
    count: dict[str, int] = {}
    for label in LABELS:
        sub = both[both["label"] == label]
        count[label] = len(sub)
        if sub.empty:
            continue
        g0 = sub["gap"].to_numpy(dtype=np.float64)
        g1 = sub["gap_next"].to_numpy(dtype=np.float64)
        denom = float((g0**2).sum())
        b = float((g0 * g1).sum()) / denom
        slope[label] = b
        se[label] = float(np.sqrt(((g0 * (g1 - b * g0)) ** 2).sum())) / denom
    real, regress = "likely real", "likely regression"
    if real not in slope or regress not in slope:
        raise ValueError("both 'likely real' and 'likely regression' players are needed")
    contrast = (slope[real] - slope[regress]) / float(np.hypot(se[real], se[regress]))
    return RetentionBacktest(slope, se, count, contrast)
