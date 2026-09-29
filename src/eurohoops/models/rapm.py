"""RAPM (week 9-12 H1): sparse, time-decayed adjusted plus-minus over stint sides.

Base columns are *spells* = ``(player_id, team, season)``. Every stint has two sides (the home
team on offense / away on defense, and the reverse); each side is one row: the offense five's
spell get ``+1`` in their O columns, the defense five's spell ``-1`` in their D columns, plus a
home column (``+1`` offense-is-home, ``-1`` offense-is-away, ``0`` at a neutral venue) and an
intercept, both effectively unpenalised. The target is ``100 * points / possessions`` of that
side, the weight ``possessions * 0.5 ** (age_days / half_life)``:

    100 * points / poss = mu + h * home + sum(O[offense five]) - sum(D[defense five])

The decayed normal equations are kept incrementally (``advance`` rescales, ``add`` adds rows),
exactly as ``models.team_eff.DecayedRidge`` does, but sparse: a season has ~9,200 spells, so a
dense Gram would be ~85M cells. At a cutoff, *model* columns are a sparse aggregation of base
columns (``ModelColumns.col_map``, a spell -> model-column lookup — the general aggregation is a
matrix ``M``, exposed as ``.M``, but every variant here is a pure grouping, so a lookup array is
enough and much faster): the plain ``rapm`` variant maps every spell to its player, one O and one
D column per player across teams and seasons (``plain_player_columns``). A future variant
(``rapm_dummy``, week 9-12 subagent E) replaces the mapping for low-minute spells with one
replacement column per team-season; ``rapm_spm`` (subagent F) leaves the mapping alone and
supplies a non-zero prior mean. Both need only build a different ``ModelColumns`` (or reuse
``plain_player_columns`` and pass a ``prior_mean``) and call ``DecayedRidgeSparse.solve`` /
``.aggregated_system``; neither needs to touch the incremental base-column state.

Because the aggregation is a pure grouping (no base column feeds two model columns), the
Gram/rhs are accumulated *both* at base-column granularity (``base_gram``/``base_rhs``, exposed
for inspection and for subagent D's posterior) and, in lockstep from the same row batches, at
model-column granularity (``model_gram``/``model_rhs`` -- always exactly ``M.T @ base_gram @ M``
by construction, never recomputed by a triple product). Solving reuses the previous cutoff's
theta as the conjugate-gradient warm start (model columns are a fixed index space for a given
variant, so a column's position never moves between cutoffs), which is what keeps a walk-forward
pass fast (H-j; the H1 benchmark, ``scripts/bench_m3_rapm.py``).

Margin convention: the row-level home column is +-1 (H-b, literally), so, exactly as in
``team_eff`` (whose own margin is ``2 * h * home_flag`` because the home and away ORtg rows each
carry +-h), the fitted home *column* is half of the margin-scale home effect. H-c's margin
formula writes a single ``h * home_flag``; we read that ``h`` as the margin-scale coefficient and
so use ``2 * theta[HOME_COL]`` when reporting/predicting with it, for consistency with the
team-level model this design mirrors (documented ambiguity, see the H1 report).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import cg

from eurohoops.models.elo import FloatArray
from eurohoops.models.minutes import round_cutoffs

IntArray = npt.NDArray[np.int64]

UNPENALISED = 1e-8  # matches team_eff.UNPENALISED: keeps intercept/home invertible early on
SECONDS_PER_DAY = 86_400.0
N_ON_COURT = 5
ROW_WIDTH = 2 + 2 * N_ON_COURT  # intercept, home, 5 offense O columns, 5 defense D columns
INTERCEPT_COL = 0
HOME_COL = 1
BASE_OFFSET = 2  # first spell O column in base-column space


def _epoch(times: pd.Series) -> FloatArray:
    """Epoch seconds of a tz-aware timestamp series (mirrors team_eff._epoch / minutes._epoch)."""
    epoch = pd.Timestamp(0, tz="UTC")
    seconds: FloatArray = (
        (times.dt.tz_convert("UTC") - epoch).dt.total_seconds().to_numpy(dtype=np.float64)
    )
    return seconds


# --- Spells: (player, team, season) base columns -----------------------------------------------


@dataclass(frozen=True)
class Spell:
    player_id: str
    team: str
    season: int


@dataclass(frozen=True)
class SpellIndex:
    """Every ``(player, team, season)`` a stints table mentions, sorted, each with an O and a D
    base column (``o_col``/``d_col``). Built once from the whole stints table: which spells
    *exist* is structural (a roster fact), not a stint outcome, so this is not leakage -- exactly
    as ``team_eff.prepare_history`` fixes its team index from every game up front."""

    spells: tuple[Spell, ...]
    positions: dict[Spell, int]  # spell -> 0-based position among `spells`
    lookup: pd.DataFrame  # columns player_id, team, season, pos -- for a vectorised join

    @property
    def n_spells(self) -> int:
        return len(self.spells)

    @property
    def n_base(self) -> int:
        return BASE_OFFSET + 2 * len(self.spells)

    def position(self, player_id: str, team: str, season: int) -> int:
        return self.positions[Spell(player_id, team, season)]

    def o_col(self, position: IntArray | int) -> IntArray | int:
        return BASE_OFFSET + position

    def d_col(self, position: IntArray | int) -> IntArray | int:
        return BASE_OFFSET + self.n_spells + position


def build_spell_index(stints: pd.DataFrame) -> SpellIndex:
    """Every ``(player, team, season)`` in ``stints``' ``home_players``/``away_players``."""
    sides = []
    for team_col, players_col in (("home", "home_players"), ("away", "away_players")):
        exploded = stints[[team_col, "season", players_col]].explode(players_col)
        # A stint of a failing game can list no players (E2015_23 has no starters flagged):
        # explode turns the empty list into a missing value, which is not a player.
        exploded = exploded[exploded[players_col].notna()]
        sides.append(
            pd.DataFrame(
                {
                    "player_id": exploded[players_col].astype(str),
                    "team": exploded[team_col].astype(str),
                    "season": exploded["season"].astype(np.int64),
                }
            )
        )
    combined = pd.concat(sides, ignore_index=True).drop_duplicates()
    combined = combined.sort_values(["player_id", "team", "season"]).reset_index(drop=True)
    spells = tuple(
        Spell(str(p), str(t), int(s))
        for p, t, s in zip(
            combined["player_id"].tolist(),
            combined["team"].tolist(),
            combined["season"].tolist(),
            strict=True,
        )
    )
    positions = {spell: i for i, spell in enumerate(spells)}
    lookup = combined.assign(pos=np.arange(len(combined), dtype=np.int64))
    return SpellIndex(spells, positions, lookup)


def _spell_positions(
    players_col: pd.Series, team: pd.Series, season: pd.Series, spell_index: SpellIndex
) -> IntArray:
    """``(n, 5)`` spell positions of every row's five listed players (vectorised join)."""
    n = len(players_col)
    lengths = players_col.map(len).to_numpy()
    if n and (lengths != N_ON_COURT).any():
        raise ValueError("every stint side must list exactly 5 players")
    flat = pd.DataFrame(
        {
            "player_id": np.concatenate([np.asarray(p, dtype=str) for p in players_col])
            if n
            else np.array([], dtype=str),
            "team": np.repeat(team.astype(str).to_numpy(), lengths),
            "season": np.repeat(season.astype(np.int64).to_numpy(), lengths),
            "row": np.repeat(np.arange(n), lengths),
        }
    )
    merged = flat.merge(spell_index.lookup, on=["player_id", "team", "season"], how="left")
    if merged["pos"].isna().any():
        missing = merged[merged["pos"].isna()].iloc[0]
        raise KeyError(f"no spell for {missing['player_id']}/{missing['team']}/{missing['season']}")
    if not merged["row"].equals(pd.Series(np.repeat(np.arange(n), lengths))):
        raise AssertionError("merge reordered rows: the vectorised join is no longer 1:1")
    return merged["pos"].to_numpy(dtype=np.int64).reshape(n, N_ON_COURT)


# --- Design rows: one per stint side --------------------------------------------------------


@dataclass(frozen=True)
class DesignRows:
    """Stint-side rows, sorted by ``time`` (game tip-off), ready for incremental fitting."""

    cols: IntArray  # (n_rows, ROW_WIDTH) base-column indices
    vals: FloatArray  # (n_rows, ROW_WIDTH) values
    target: FloatArray  # (n_rows,) 100 * points / possessions
    weight: FloatArray  # (n_rows,) possessions (undecayed)
    time: FloatArray  # (n_rows,) epoch seconds of the game's tip-off
    season: IntArray  # (n_rows,)
    game_id: npt.NDArray[np.str_]  # (n_rows,)

    @property
    def n_base(self) -> int:
        return int(self.cols.max()) + 1 if len(self.cols) else BASE_OFFSET


_SIDES = (
    # offense team col, defense team col, offense players col, defense players col,
    # points col, possessions col, home sign (of the offense team)
    ("home", "away", "home_players", "away_players", "home_points", "home_poss", 1.0),
    ("away", "home", "away_players", "home_players", "away_points", "away_poss", -1.0),
)


def build_design_rows(
    stints: pd.DataFrame, checks: pd.DataFrame, games: pd.DataFrame, spell_index: SpellIndex
) -> DesignRows:
    """One row per stint side of games whose ``stint_game_checks.passed`` (H-b); 0-possession
    sides dropped. ``games``: the mart rows (for tip-off time and neutral-venue flag)."""
    passed = set(checks.loc[checks["passed"].astype(bool), "game_id"].astype(str))
    fit = stints[stints["game_id"].astype(str).isin(passed)]
    info = games.set_index(games["game_id"].astype(str))
    neutral_map = info["neutral"]
    time_map = pd.Series(_epoch(info["tipoff_utc"]), index=info.index)

    blocks: list[DesignRows] = []
    for off_team, def_team, off_players, def_players, pts, poss, sign in _SIDES:
        sub = fit[fit[poss] > 0]
        n = len(sub)
        if not n:
            continue
        game_ids = sub["game_id"].astype(str).to_numpy()
        neutral = neutral_map.reindex(game_ids).to_numpy()
        season = sub["season"].astype(np.int64)
        off_pos = _spell_positions(sub[off_players], sub[off_team], season, spell_index)
        def_pos = _spell_positions(sub[def_players], sub[def_team], season, spell_index)
        cols = np.empty((n, ROW_WIDTH), dtype=np.int64)
        cols[:, INTERCEPT_COL] = INTERCEPT_COL
        cols[:, HOME_COL] = HOME_COL
        cols[:, 2 : 2 + N_ON_COURT] = spell_index.o_col(off_pos)
        cols[:, 2 + N_ON_COURT : ROW_WIDTH] = spell_index.d_col(def_pos)
        vals = np.empty((n, ROW_WIDTH))
        vals[:, INTERCEPT_COL] = 1.0
        vals[:, HOME_COL] = np.where(neutral, 0.0, sign)
        vals[:, 2 : 2 + N_ON_COURT] = 1.0
        vals[:, 2 + N_ON_COURT : ROW_WIDTH] = -1.0
        target = 100.0 * sub[pts].to_numpy(dtype=np.float64) / sub[poss].to_numpy(dtype=np.float64)
        weight = sub[poss].to_numpy(dtype=np.float64)
        time = time_map.reindex(game_ids).to_numpy(dtype=np.float64).astype(np.float64)
        blocks.append(
            DesignRows(
                cols,
                vals,
                target,
                weight,
                time,
                season.to_numpy(dtype=np.int64).astype(np.int64),
                game_ids.astype(str),
            )
        )
    if not blocks:
        return DesignRows(
            np.empty((0, ROW_WIDTH), dtype=np.int64),
            np.empty((0, ROW_WIDTH)),
            np.empty(0),
            np.empty(0),
            np.empty(0),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=str),
        )
    order = np.argsort(
        np.concatenate([b.time for b in blocks]), kind="stable"
    )  # stable: ties keep block/row order, so results never depend on dict/set iteration order
    return DesignRows(
        cols=np.concatenate([b.cols for b in blocks])[order],
        vals=np.concatenate([b.vals for b in blocks])[order],
        target=np.concatenate([b.target for b in blocks])[order],
        weight=np.concatenate([b.weight for b in blocks])[order],
        time=np.concatenate([b.time for b in blocks])[order],
        season=np.concatenate([b.season for b in blocks])[order],
        game_id=np.concatenate([b.game_id for b in blocks])[order],
    )


# --- Model columns: base -> model aggregation (the hook rapm_dummy / rapm_spm extend) -----------


@dataclass(frozen=True)
class ModelColumns:
    """A variant's base -> model column aggregation. Every variant here is a pure grouping (no
    base column feeds two model columns), so ``col_map`` (an array lookup) is the fast internal
    representation; ``.M`` derives the general sparse matrix the spec and subagent D's posterior
    expect, built lazily and only once (``M[i, col_map[i]] = 1``)."""

    name: str
    n_model: int
    col_map: IntArray  # (n_base,) base column -> model column
    o_labels: tuple[str, ...]  # model O column position -> player_id
    d_labels: tuple[str, ...]  # model D column position -> player_id
    o_start: int
    d_start: int

    def penalty(self, ridge_o: float, ridge_d: float) -> FloatArray:
        penalty = np.full(self.n_model, UNPENALISED)
        penalty[self.o_start : self.o_start + len(self.o_labels)] = ridge_o
        penalty[self.d_start : self.d_start + len(self.d_labels)] = ridge_d
        return penalty

    def labels(self) -> tuple[str, ...]:
        return (
            "intercept",
            "home",
            *(f"O:{p}" for p in self.o_labels),
            *(f"D:{p}" for p in self.d_labels),
        )

    @property
    def M(self) -> sparse.csr_matrix:  # matches the spec's own notation (M)
        n_base = len(self.col_map)
        rows = np.arange(n_base)
        vals = np.ones(n_base)
        return sparse.coo_matrix((vals, (rows, self.col_map)), shape=(n_base, self.n_model)).tocsr()


def plain_player_columns(spell_index: SpellIndex) -> ModelColumns:
    """The ``rapm`` variant: every spell maps to its player, one O and one D column each,
    across every team and season the player appears in."""
    players = sorted({s.player_id for s in spell_index.spells})
    player_pos = {p: i for i, p in enumerate(players)}
    n_players = len(players)
    o_start, d_start = BASE_OFFSET, BASE_OFFSET + n_players
    n_model = BASE_OFFSET + 2 * n_players
    col_map = np.empty(spell_index.n_base, dtype=np.int64)
    col_map[INTERCEPT_COL] = INTERCEPT_COL
    col_map[HOME_COL] = HOME_COL
    spell_player_pos = np.array(
        [player_pos[s.player_id] for s in spell_index.spells], dtype=np.int64
    )
    col_map[BASE_OFFSET : BASE_OFFSET + spell_index.n_spells] = o_start + spell_player_pos
    col_map[BASE_OFFSET + spell_index.n_spells :] = d_start + spell_player_pos
    return ModelColumns("rapm", n_model, col_map, tuple(players), tuple(players), o_start, d_start)


# --- Rating lookup: the per-cutoff hook other variants (and m3_backtest) call -------------------


@dataclass(frozen=True)
class RatingLookup:
    """A cutoff's fitted (O, D) per player; 0 for a player with no column yet (H-c). ``team``
    and ``season`` are accepted (and ignored here) so a future variant -- e.g. ``rapm_dummy``,
    which maps a low-minute or unseen player to his team-season replacement column instead --
    can key on them without changing this call shape."""

    o: dict[str, float]
    d: dict[str, float]

    def rating(self, player_id: str, team: str, season: int) -> tuple[float, float]:
        return self.o.get(player_id, 0.0), self.d.get(player_id, 0.0)


def rating_lookup(theta: FloatArray, columns: ModelColumns) -> RatingLookup:
    o = {label: float(theta[columns.o_start + i]) for i, label in enumerate(columns.o_labels)}
    d = {label: float(theta[columns.d_start + i]) for i, label in enumerate(columns.d_labels)}
    return RatingLookup(o, d)


# --- The decayed normal equations, sparse, kept at both base and model granularity --------------


@dataclass(frozen=True)
class AggregatedSystem:
    """The solved-system inputs at a cutoff, model-column space, restricted to columns some row
    has touched (subagent D calls ``posterior_from_normal_equations(gram, rhs, penalty,
    prior_mean, noise_var)`` on exactly this)."""

    gram: sparse.csr_matrix  # (n_used, n_used), no penalty added
    rhs: FloatArray  # (n_used,)
    penalty: FloatArray  # (n_used,)
    prior_mean: FloatArray  # (n_used,)
    labels: tuple[str, ...]  # (n_used,)
    used: IntArray  # (n_used,) model-column indices this restricts to


class DecayedRidgeSparse:
    """Time-decayed normal equations over spell (base) columns, kept incrementally and sparse
    (module docstring). ``advance(time)`` rescales everything added so far; ``add`` adds a batch
    of stint-side rows. ``solve``/``aggregated_system`` work in one variant's model-column space
    (``columns``); only columns some row has touched are solved, so a spell that first appears
    later cannot change an earlier cutoff's solution."""

    def __init__(self, n_base: int, half_life_days: float, columns: ModelColumns) -> None:
        self.half_life_days = half_life_days
        self.columns = columns
        self.base_gram = sparse.csr_matrix((n_base, n_base))
        self.base_rhs = np.zeros(n_base)
        self.base_seen = np.zeros(n_base, dtype=bool)
        self.model_gram = sparse.csr_matrix((columns.n_model, columns.n_model))
        self.model_rhs = np.zeros(columns.n_model)
        self.model_seen = np.zeros(columns.n_model, dtype=bool)
        self.time = 0.0
        self._started = False

    def decay(self, age_days: FloatArray) -> FloatArray:
        weights: FloatArray = np.power(0.5, age_days / self.half_life_days)
        return weights

    def advance(self, time: float) -> None:
        if self._started:
            factor = float(self.decay(np.array([(time - self.time) / SECONDS_PER_DAY]))[0])
            self.base_gram = self.base_gram * factor
            self.base_rhs = self.base_rhs * factor
            self.model_gram = self.model_gram * factor
            self.model_rhs = self.model_rhs * factor
        self.time = time
        self._started = True

    def add(
        self,
        cols: IntArray,
        vals: FloatArray,
        *,
        target: FloatArray,
        weight: FloatArray,
        time: FloatArray,
    ) -> None:
        """Rows with nonzeros ``vals[i, k]`` in ``cols[i, k]``, aged to the current time."""
        if not len(target):
            return
        w = weight * self.decay((self.time - time) / SECONDS_PER_DAY)
        n_rows, k = cols.shape
        row_idx = np.repeat(np.arange(n_rows), k)
        sw = np.sqrt(w)
        scaled = vals * sw[:, None]
        base_x = sparse.coo_matrix(
            (scaled.ravel(), (row_idx, cols.ravel())), shape=(n_rows, self.base_gram.shape[0])
        ).tocsr()
        self.base_gram = self.base_gram + (base_x.T @ base_x)
        self.base_rhs += np.asarray(base_x.T @ (sw * target)).ravel()
        self.base_seen[np.unique(cols)] = True

        model_cols = self.columns.col_map[cols]
        model_shape = (n_rows, self.model_gram.shape[0])
        model_x = sparse.coo_matrix(
            (scaled.ravel(), (row_idx, model_cols.ravel())), shape=model_shape
        ).tocsr()
        self.model_gram = self.model_gram + (model_x.T @ model_x)
        self.model_rhs += np.asarray(model_x.T @ (sw * target)).ravel()
        self.model_seen[np.unique(model_cols)] = True

    def aggregated_system(
        self, ridge_o: float, ridge_d: float, prior_mean: FloatArray | None = None
    ) -> AggregatedSystem:
        used = np.flatnonzero(self.model_seen)
        m = np.zeros(self.columns.n_model) if prior_mean is None else prior_mean
        penalty = self.columns.penalty(ridge_o, ridge_d)
        labels = self.columns.labels()
        return AggregatedSystem(
            gram=self.model_gram[used][:, used].tocsr(),
            rhs=self.model_rhs[used],
            penalty=penalty[used],
            prior_mean=m[used],
            labels=tuple(labels[i] for i in used),
            used=used,
        )

    def solve(
        self,
        ridge_o: float,
        ridge_d: float,
        prior_mean: FloatArray | None = None,
        x0: FloatArray | None = None,
    ) -> FloatArray:
        """The ridge solution in model-column space (0 for an untouched column)."""
        system = self.aggregated_system(ridge_o, ridge_d, prior_mean)
        theta = np.zeros(self.columns.n_model)
        if not len(system.used):
            return theta
        penalised = system.gram + sparse.diags(system.penalty)
        rhs = system.rhs + system.penalty * system.prior_mean
        diag = penalised.diagonal()
        diag = np.where(diag > 0, diag, 1.0)
        preconditioner = sparse.diags(1.0 / diag)
        x0_used = None if x0 is None else x0[system.used]
        solved, info = cg(
            penalised.tocsr(),
            rhs,
            x0=x0_used,
            rtol=1e-10,
            atol=0.0,
            M=preconditioner,
            maxiter=max(2000, 20 * len(system.used)),
        )
        if info != 0:
            n = len(system.used)
            raise RuntimeError(f"RAPM ridge CG did not converge (info={info}, n={n})")
        theta[system.used] = solved
        return theta


# --- Walk-forward driver (H-c) -------------------------------------------------------------------


@dataclass(frozen=True)
class WalkForward:
    """Round-by-round RAPM fit, aligned with ``games`` (tip-off order): ``home_coef[g]`` and
    ``lookups[g]`` are the model's state just before game ``g``'s round (NaN / ``None`` before
    any stint has been added -- no forecast yet, as ``team_eff.rating_fits`` also leaves NaN)."""

    home_coef: FloatArray  # (n_games,) 2 * the fitted home column (module docstring)
    lookups: list[RatingLookup | None]  # (n_games,)
    columns: ModelColumns
    model: DecayedRidgeSparse  # the state as of the last cutoff processed


def _round_groups(games: pd.DataFrame) -> list[tuple[float, int, IntArray]]:
    """(cutoff time, season, game indices) in non-decreasing cutoff order (team_eff._cutoffs)."""
    cutoff = round_cutoffs(games).to_numpy()
    season = games["season"].to_numpy()
    order = np.lexsort((season, cutoff))
    out: list[tuple[float, int, IntArray]] = []
    for i in order:
        key = (float(cutoff[i]), int(season[i]))
        if out and (out[-1][0], out[-1][1]) == key:
            out[-1] = (key[0], key[1], np.append(out[-1][2], i))
        else:
            out.append((key[0], key[1], np.array([i], dtype=np.int64)))
    return out


def fit_walk_forward(
    games: pd.DataFrame,
    rows: DesignRows,
    columns: ModelColumns,
    *,
    half_life_days: float,
    ridge_o: float,
    ridge_d: float,
    prior_mean: FloatArray | None = None,
) -> WalkForward:
    """Fit RAPM walk-forward by round (H-c): a game's rating comes only from stints of games
    that tipped off before the first tip-off of its (season, phase, round)."""
    n_base = rows.n_base if len(rows.cols) else BASE_OFFSET
    model = DecayedRidgeSparse(n_base, half_life_days, columns)
    n_games = len(games)
    home_coef = np.full(n_games, np.nan)
    lookups: list[RatingLookup | None] = [None] * n_games
    theta = np.zeros(columns.n_model)
    added = 0
    for time, _season, game_idx in _round_groups(games):
        model.advance(time)
        stop = int(np.searchsorted(rows.time, time, side="left"))
        if stop > added:
            span = slice(added, stop)
            model.add(
                rows.cols[span],
                rows.vals[span],
                target=rows.target[span],
                weight=rows.weight[span],
                time=rows.time[span],
            )
            added = stop
        if not added:
            continue
        theta = model.solve(ridge_o, ridge_d, prior_mean, x0=theta)
        coef = 2.0 * theta[HOME_COL]
        lookup = rating_lookup(theta, columns)
        home_coef[game_idx] = coef
        for g in game_idx:
            lookups[g] = lookup
    return WalkForward(home_coef, lookups, columns, model)


# --- Decayed on-court minutes per spell (D6: rapm_dummy's fitting-window threshold) --------------


@dataclass(frozen=True)
class MinutesRows:
    """One entry per (stint, on-court player): spell position, seconds, and the game's tip-off,
    for the same games and decay ``fit_walk_forward`` uses (only passed-check games)."""

    position: IntArray  # (n,) spell position (0-based, not a base column)
    seconds: FloatArray  # (n,)
    time: FloatArray  # (n,) epoch seconds of the game's tip-off


def build_minutes_rows(
    stints: pd.DataFrame, checks: pd.DataFrame, games: pd.DataFrame, spell_index: SpellIndex
) -> MinutesRows:
    passed = set(checks.loc[checks["passed"].astype(bool), "game_id"].astype(str))
    fit = stints[stints["game_id"].astype(str).isin(passed)]
    if not len(fit):
        return MinutesRows(np.empty(0, dtype=np.int64), np.empty(0), np.empty(0))
    info = games.set_index(games["game_id"].astype(str))
    time_map = pd.Series(_epoch(info["tipoff_utc"]), index=info.index)
    duration = (fit["end_s"] - fit["start_s"]).to_numpy(dtype=np.float64)
    ids = fit["game_id"].astype(str).to_numpy()
    time = time_map.reindex(ids).to_numpy(dtype=np.float64)
    season = fit["season"].astype(np.int64)
    home_pos = _spell_positions(fit["home_players"], fit["home"], season, spell_index)
    away_pos = _spell_positions(fit["away_players"], fit["away"], season, spell_index)
    position = np.concatenate([home_pos, away_pos], axis=1)  # (n, 10)
    seconds = np.repeat(duration[:, None], 2 * N_ON_COURT, axis=1)
    time_rep = np.repeat(time[:, None], 2 * N_ON_COURT, axis=1)
    position_flat = position.ravel()
    seconds_flat = seconds.ravel().astype(np.float64)
    time_flat = time_rep.ravel().astype(np.float64)
    # `fit_decayed_minutes` (and any walk-forward caller) uses `searchsorted` on `time`, which
    # needs ascending order; rows come out in `stints`' own row order above, not tip-off order
    # (D6/H6 known issue), so sort here once, stably (ties keep the stint's own row order).
    order = np.argsort(time_flat, kind="stable")
    return MinutesRows(position_flat[order], seconds_flat[order], time_flat[order])


class DecayedMinutes:
    """Time-decayed on-court seconds per spell, kept incrementally like ``DecayedRidgeSparse``
    but as a plain vector (no Gram): the same decay as the fitting rows (D6)."""

    def __init__(self, n_spells: int, half_life_days: float) -> None:
        self.half_life_days = half_life_days
        self.seconds = np.zeros(n_spells)
        self.time = 0.0
        self._started = False

    def decay(self, age_days: FloatArray) -> FloatArray:
        weights: FloatArray = np.power(0.5, age_days / self.half_life_days)
        return weights

    def advance(self, time: float) -> None:
        if self._started:
            factor = float(self.decay(np.array([(time - self.time) / SECONDS_PER_DAY]))[0])
            self.seconds = self.seconds * factor
        self.time = time
        self._started = True

    def add(self, position: IntArray, seconds: FloatArray, time: FloatArray) -> None:
        if not len(seconds):
            return
        weighted = seconds * self.decay((self.time - time) / SECONDS_PER_DAY)
        np.add.at(self.seconds, position, weighted)


def fit_decayed_minutes(
    games: pd.DataFrame, rows: MinutesRows, spell_index: SpellIndex, half_life_days: float
) -> FloatArray:
    """Decayed on-court seconds per spell, as of the last game's round cutoff (a convenience
    for a one-shot snapshot; a walk-forward caller advances/adds round by round itself, the
    same way ``fit_walk_forward`` does, to read the state at each cutoff)."""
    minutes = DecayedMinutes(spell_index.n_spells, half_life_days)
    added = 0
    for time, _season, _game_idx in _round_groups(games):
        minutes.advance(time)
        stop = int(np.searchsorted(rows.time, time, side="left"))
        if stop > added:
            span = slice(added, stop)
            minutes.add(rows.position[span], rows.seconds[span], rows.time[span])
            added = stop
    return minutes.seconds
