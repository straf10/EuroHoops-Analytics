"""M3 H6 (week 9-12 subagent E): ``rapm_dummy`` -- players under a minutes threshold in the
fitting window share one "replacement" O/D column per (team, season) instead of their own (H-f).

D6 fixes what "minutes in the fitting window" means: at a round cutoff, a *player's* minutes are
his time-decayed on-court minutes, summed across every one of his spells (every team and season
he has appeared in), over stints of games that passed ``stint_game_checks`` and tipped off before
the cutoff, with the same half-life as the RAPM rows (``rapm.build_minutes_rows`` /
``rapm.DecayedMinutes``, built by subagent A for exactly this hook). A player at or above the
chosen threshold keeps his own O/D columns, as plain ``rapm``; a player below it -- *each* of his
spells, individually -- maps instead to the O/D "replacement" column of that spell's own (team,
season). A brand-new player with no minutes yet is below every positive threshold and so is
pooled too.

Because a player's minutes change every round, which of his spells map to his own column and
which map to a replacement changes between cutoffs (unlike plain ``rapm``'s fixed grouping): a
player who starts a season sparingly used and later plays heavy minutes moves from the
replacement column to his own partway through. So, unlike ``rapm.DecayedRidgeSparse`` (which
accumulates the *model*-level Gram/rhs incrementally, batch by batch, under one fixed mapping),
this module keeps only the *base* (spell-level) Gram/rhs incrementally (``BaseAccumulator``,
literally the base half of ``DecayedRidgeSparse.advance``/``.add``) and, at every cutoff, builds
that cutoff's base -> model grouping matrix ``M`` fresh from the current decayed minutes
(``threshold_map``) and re-aggregates ``model_gram = M.T @ base_gram @ M`` (and ``rhs``
likewise) before solving (``solve_dummy_cutoff``) -- exactly the "aggregate the base-level normal
equations per cutoff" the design calls for. The *model*-column index space itself never changes
across cutoffs (one O/D column per player, fixed once from the whole ``SpellIndex``, plus one
O/D replacement column per (team, season): ``build_dummy_columns``), so a column's position is
always the same and the previous cutoff's solved ``theta`` is a valid CG warm start even though
the mapping into it moved.

A threshold of 0 always keeps every player at or above it (minutes are never negative), so it
reproduces plain ``rapm``'s own mapping and, given the same half-life/ridges, its fit -- the
tuner (``tune_rapm_dummy``) never retries 0, since ``tune_rapm`` already is that point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import cg

from eurohoops.config import M3Backtest
from eurohoops.models.elo import FloatArray
from eurohoops.models.rapm import (
    BASE_OFFSET,
    HOME_COL,
    INTERCEPT_COL,
    SECONDS_PER_DAY,
    DecayedMinutes,
    DecayedRidgeSparse,
    DesignRows,
    MinutesRows,
    ModelColumns,
    RatingLookup,
    SpellIndex,
    WalkForward,
    _round_groups,  # the same round-cutoff driver `rapm.fit_walk_forward` uses
)

if TYPE_CHECKING:
    # Type-only: `eval.m3_backtest` imports this module (to register the `rapm_dummy` variant),
    # so importing it back for real at module load time would be circular -- and, unlike a
    # plain name lookup on a partially-initialised module, genuinely broken regardless of import
    # order (whichever of the two modules a caller imports first, its own top-level import of
    # the other runs before that other's names exist). `tune_rapm_dummy` (the only function that
    # actually constructs a `TunedRapm` or calls `rapm_margins`/`_finite_rmse`/`_round`) imports
    # them locally instead, deferred until it is actually called, by which point both modules
    # have long finished loading however the caller reached them.
    from eurohoops.eval.m3_backtest import Data, RapmFitInputs, TunedRapm

IntArray = npt.NDArray[np.int64]


# --- The fixed model-column space: one O/D column per player, plus one O/D replacement column
# --- per (team, season) (module docstring). ------------------------------------------------------


@dataclass(frozen=True)
class DummyColumns:
    """``rapm_dummy``'s fixed model-column space, built once from a ``SpellIndex``. ``columns``
    is the generic ``rapm.ModelColumns`` container (``.penalty``/``.labels``/``.M``); its own
    ``col_map`` is only a placeholder (threshold-0's mapping, every spell to its own player) --
    the real, per-cutoff mapping comes from ``threshold_map``, never from this fixed one."""

    columns: ModelColumns
    player_pos: dict[str, int]  # player_id -> index within the player block
    team_season_pos: dict[tuple[str, int], int]  # (team, season) -> index within the repl. block
    n_players: int
    n_team_seasons: int
    spell_player_index: IntArray  # (n_spells,) spell position -> its player's index
    spell_ts_index: IntArray  # (n_spells,) spell position -> its (team, season)'s index


def build_dummy_columns(spell_index: SpellIndex) -> DummyColumns:
    """Every player (as ``rapm.plain_player_columns``) plus every (team, season) any spell
    belongs to, each with its own O and D column."""
    players = sorted({s.player_id for s in spell_index.spells})
    player_pos = {p: i for i, p in enumerate(players)}
    team_seasons = sorted({(s.team, s.season) for s in spell_index.spells})
    ts_pos = {ts: i for i, ts in enumerate(team_seasons)}
    n_players, n_team_seasons = len(players), len(team_seasons)
    o_start = BASE_OFFSET
    d_start = BASE_OFFSET + n_players + n_team_seasons
    n_model = BASE_OFFSET + 2 * (n_players + n_team_seasons)
    repl_labels = tuple(f"REPL:{t}:{s}" for t, s in team_seasons)
    o_labels = tuple(players) + repl_labels
    d_labels = tuple(players) + repl_labels

    spell_player_index = np.array(
        [player_pos[s.player_id] for s in spell_index.spells], dtype=np.int64
    )
    spell_ts_index = np.array(
        [ts_pos[(s.team, s.season)] for s in spell_index.spells], dtype=np.int64
    )

    col_map = np.empty(spell_index.n_base, dtype=np.int64)
    col_map[INTERCEPT_COL] = INTERCEPT_COL
    col_map[HOME_COL] = HOME_COL
    n = spell_index.n_spells
    col_map[BASE_OFFSET : BASE_OFFSET + n] = o_start + spell_player_index
    col_map[BASE_OFFSET + n :] = d_start + spell_player_index
    columns = ModelColumns("rapm_dummy", n_model, col_map, o_labels, d_labels, o_start, d_start)
    return DummyColumns(
        columns, player_pos, ts_pos, n_players, n_team_seasons, spell_player_index, spell_ts_index
    )


def _player_minutes(dummy: DummyColumns, decayed_minutes: FloatArray) -> FloatArray:
    """Per-player decayed minutes (D6): a player's spells summed across every team-season."""
    totals = np.zeros(dummy.n_players)
    np.add.at(totals, dummy.spell_player_index, decayed_minutes)
    return totals


def threshold_map(
    dummy: DummyColumns, decayed_minutes: FloatArray, threshold_minutes: float
) -> IntArray:
    """This cutoff's base -> model column mapping (H-f): a spell whose player's total decayed
    minutes (``_player_minutes``) are >= ``threshold_minutes`` maps to that player's own O/D
    column; below it, to the O/D replacement column of the spell's own (team, season).
    ``threshold_minutes`` == 0 always keeps every player's own column (minutes are never
    negative), reproducing ``rapm.plain_player_columns``'s own mapping."""
    totals = _player_minutes(dummy, decayed_minutes)
    above = totals[dummy.spell_player_index] >= threshold_minutes
    o_own = dummy.columns.o_start + dummy.spell_player_index
    d_own = dummy.columns.d_start + dummy.spell_player_index
    o_repl = dummy.columns.o_start + dummy.n_players + dummy.spell_ts_index
    d_repl = dummy.columns.d_start + dummy.n_players + dummy.spell_ts_index
    o_pos = np.where(above, o_own, o_repl)
    d_pos = np.where(above, d_own, d_repl)
    n = len(dummy.spell_player_index)
    col_map = np.empty(BASE_OFFSET + 2 * n, dtype=np.int64)
    col_map[INTERCEPT_COL] = INTERCEPT_COL
    col_map[HOME_COL] = HOME_COL
    col_map[BASE_OFFSET : BASE_OFFSET + n] = o_pos
    col_map[BASE_OFFSET + n :] = d_pos
    return col_map


# --- Base-only decayed normal equations (module docstring: the half of DecayedRidgeSparse's
# --- bookkeeping this variant needs, since its mapping is rebuilt every cutoff) ------------------


class BaseAccumulator:
    """Time-decayed normal equations at base (spell) granularity only -- a byte-for-byte copy of
    ``rapm.DecayedRidgeSparse``'s base-column bookkeeping (``advance``/``add``), without the
    model-level half: ``rapm_dummy`` has no single fixed model-level Gram to accumulate
    incrementally, since its base -> model mapping is rebuilt every cutoff (module docstring)."""

    def __init__(self, n_base: int, half_life_days: float) -> None:
        self.half_life_days = half_life_days
        self.gram = sparse.csr_matrix((n_base, n_base))
        self.rhs = np.zeros(n_base)
        self.time = 0.0
        self._started = False

    def decay(self, age_days: FloatArray) -> FloatArray:
        weights: FloatArray = np.power(0.5, age_days / self.half_life_days)
        return weights

    def advance(self, time: float) -> None:
        if self._started:
            factor = float(self.decay(np.array([(time - self.time) / SECONDS_PER_DAY]))[0])
            self.gram = self.gram * factor
            self.rhs = self.rhs * factor
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
        if not len(target):
            return
        w = weight * self.decay((self.time - time) / SECONDS_PER_DAY)
        n_rows, k = cols.shape
        row_idx = np.repeat(np.arange(n_rows), k)
        sw = np.sqrt(w)
        scaled = vals * sw[:, None]
        x = sparse.coo_matrix(
            (scaled.ravel(), (row_idx, cols.ravel())), shape=(n_rows, self.gram.shape[0])
        ).tocsr()
        self.gram = self.gram + (x.T @ x)
        self.rhs += np.asarray(x.T @ (sw * target)).ravel()


# --- One cutoff's pooled ridge solve --------------------------------------------------------


def _solve_penalised_system(
    model_gram: sparse.csr_matrix,
    model_rhs: FloatArray,
    penalty: FloatArray,
    prior_mean: FloatArray,
    x0: FloatArray,
) -> tuple[FloatArray, IntArray]:
    """The same penalised CG solve ``rapm.DecayedRidgeSparse.solve`` performs, restricted to
    touched (diagonal > 0) columns, done here on a freshly reprojected ``model_gram``/
    ``model_rhs`` (this cutoff's) rather than an incrementally-kept one."""
    n_model = model_gram.shape[0]
    diag_full = np.asarray(model_gram.diagonal())
    used = np.flatnonzero(diag_full > 0)
    theta = np.zeros(n_model)
    if not len(used):
        return theta, used
    gram = model_gram[used][:, used].tocsr()
    penalty_used = penalty[used]
    rhs = model_rhs[used] + penalty_used * prior_mean[used]
    penalised = gram + sparse.diags(penalty_used)
    diag = penalised.diagonal()
    diag = np.where(diag > 0, diag, 1.0)
    preconditioner = sparse.diags(1.0 / diag)
    solved, info = cg(
        penalised.tocsr(),
        rhs,
        x0=x0[used],
        rtol=1e-10,
        atol=0.0,
        M=preconditioner,
        maxiter=max(2000, 20 * len(used)),
    )
    if info != 0:
        raise RuntimeError(f"rapm_dummy ridge CG did not converge (info={info}, n={len(used)})")
    theta[used] = solved
    return theta, used


@dataclass(frozen=True)
class DummyCutoffSolve:
    """One cutoff's solved model-space ``theta`` (0 for an untouched column) and which model
    columns were touched (``used``, the CG-restricted subset)."""

    theta: FloatArray
    used: IntArray


def solve_dummy_cutoff(
    base_gram: sparse.csr_matrix,
    base_rhs: FloatArray,
    dummy: DummyColumns,
    decayed_minutes: FloatArray,
    threshold_minutes: float,
    *,
    ridge_o: float,
    ridge_d: float,
    prior_mean: FloatArray | None = None,
    x0: FloatArray | None = None,
) -> DummyCutoffSolve:
    """One cutoff's pooled ridge solve (H-f, module docstring): this cutoff's base -> model
    grouping (``threshold_map``), the aggregated normal equations ``M.T @ base_gram @ M`` /
    ``M.T @ base_rhs``, and the CG solve (``_solve_penalised_system``). The hand test in
    ``test_rapm_dummy.py`` checks this against an independent dense solve of the same pooled
    design; ``fit_walk_forward_dummy`` calls it once per round cutoff.

    ``ModelColumns.M`` (a plain ``scipy.sparse`` CSR-CSR product, the same building block
    ``rapm.DecayedRidgeSparse`` and its own ``aggregated_system`` use) turned out faster here
    than relabelling ``base_gram``'s (row, col) triplets by ``col_map`` and letting ``coo_matrix``
    deduplicate them -- an equally correct alternative that looks like it should avoid the
    matmul, but costs more in practice at this scale (measured with H6's own benchmark, ``scripts
    /bench_m3_rapm_dummy.py``: ~1.6x slower, because ``base_gram`` accumulates several million
    nonzeros over a full walk-forward and deduplicating that many triplets by sort costs more
    than SciPy's CSR-CSR multiply does)."""
    col_map = threshold_map(dummy, decayed_minutes, threshold_minutes)
    n_model = dummy.columns.n_model
    m = ModelColumns(
        dummy.columns.name,
        n_model,
        col_map,
        dummy.columns.o_labels,
        dummy.columns.d_labels,
        dummy.columns.o_start,
        dummy.columns.d_start,
    ).M
    model_gram = (m.T @ base_gram @ m).tocsr()
    model_rhs = np.asarray(m.T @ base_rhs).ravel()
    penalty = dummy.columns.penalty(ridge_o, ridge_d)
    prior = np.zeros(dummy.columns.n_model) if prior_mean is None else prior_mean
    x0_full = np.zeros(dummy.columns.n_model) if x0 is None else x0
    theta, used = _solve_penalised_system(model_gram, model_rhs, penalty, prior, x0_full)
    return DummyCutoffSolve(theta, used)


# --- Rating lookup: above threshold reads a player's own column, below reads his (team, season)
# --- replacement, unseen or not-yet-fitted reads 0 -------------------------------------------


@dataclass(frozen=True)
class DummyRatingLookup(RatingLookup):
    """``rapm_dummy``'s per-cutoff rating lookup (H-f). ``o``/``d`` (from ``RatingLookup``) hold
    only the at-or-above-threshold players' own O/D values -- an untouched player column behaves
    like plain ``rapm``'s (0, via ``.get(..., 0.0)``). ``repl_o``/``repl_d`` hold every touched
    (team, season) replacement column. A player below threshold, or never seen at all, reads the
    (team, season) replacement instead -- 0 if that replacement itself has not been fitted yet."""

    repl_o: dict[tuple[str, int], float]
    repl_d: dict[tuple[str, int], float]

    def rating(self, player_id: str, team: str, season: int) -> tuple[float, float]:
        if player_id in self.o or player_id in self.d:
            return self.o.get(player_id, 0.0), self.d.get(player_id, 0.0)
        key = (team, season)
        return self.repl_o.get(key, 0.0), self.repl_d.get(key, 0.0)


def dummy_rating_lookup(
    solve: DummyCutoffSolve,
    dummy: DummyColumns,
    decayed_minutes: FloatArray,
    threshold_minutes: float,
) -> DummyRatingLookup:
    """The ``RatingLookup`` a cutoff's ``DummyCutoffSolve`` implies (H-f's lookup semantics)."""
    totals = _player_minutes(dummy, decayed_minutes)
    above = totals >= threshold_minutes
    o = {
        p: float(solve.theta[dummy.columns.o_start + i])
        for p, i in dummy.player_pos.items()
        if above[i]
    }
    d = {
        p: float(solve.theta[dummy.columns.d_start + i])
        for p, i in dummy.player_pos.items()
        if above[i]
    }
    used_set = {int(i) for i in solve.used}
    repl_o: dict[tuple[str, int], float] = {}
    repl_d: dict[tuple[str, int], float] = {}
    for (team, season), i in dummy.team_season_pos.items():
        o_idx = dummy.columns.o_start + dummy.n_players + i
        if o_idx in used_set:
            repl_o[(team, season)] = float(solve.theta[o_idx])
        d_idx = dummy.columns.d_start + dummy.n_players + i
        if d_idx in used_set:
            repl_d[(team, season)] = float(solve.theta[d_idx])
    return DummyRatingLookup(o, d, repl_o, repl_d)


# --- Walk-forward driver (H-c, mirroring rapm.fit_walk_forward's cutoff loop) -----------------


def fit_walk_forward_dummy(
    games: pd.DataFrame,
    rows: DesignRows,
    minutes_rows: MinutesRows,
    spell_index: SpellIndex,
    dummy: DummyColumns,
    *,
    half_life_days: float,
    ridge_o: float,
    ridge_d: float,
    dummy_minutes: float,
    prior_mean: FloatArray | None = None,
) -> WalkForward:
    """Fit ``rapm_dummy`` walk-forward by round (H-c), exactly the same cutoffs as
    ``rapm.fit_walk_forward``: a game's rating comes only from stints and minutes of games that
    tipped off before the first tip-off of its (season, phase, round). Unlike plain ``rapm``, the
    aggregated system is rebuilt from the base Gram/rhs every cutoff (module docstring), since
    the base -> model mapping depends on that cutoff's decayed minutes."""
    n_base = spell_index.n_base
    base = BaseAccumulator(n_base, half_life_days)
    minutes = DecayedMinutes(spell_index.n_spells, half_life_days)
    n_games = len(games)
    home_coef = np.full(n_games, np.nan)
    lookups: list[RatingLookup | None] = [None] * n_games
    theta = np.zeros(dummy.columns.n_model)
    added_rows = 0
    added_minutes = 0
    for time, _season, game_idx in _round_groups(games):
        base.advance(time)
        minutes.advance(time)
        stop = int(np.searchsorted(rows.time, time, side="left"))
        if stop > added_rows:
            span = slice(added_rows, stop)
            base.add(
                rows.cols[span],
                rows.vals[span],
                target=rows.target[span],
                weight=rows.weight[span],
                time=rows.time[span],
            )
            added_rows = stop
        m_stop = int(np.searchsorted(minutes_rows.time, time, side="left"))
        if m_stop > added_minutes:
            m_span = slice(added_minutes, m_stop)
            minutes.add(
                minutes_rows.position[m_span],
                minutes_rows.seconds[m_span],
                minutes_rows.time[m_span],
            )
            added_minutes = m_stop
        if not added_rows:
            continue
        decayed_minutes = minutes.seconds / 60.0
        solve = solve_dummy_cutoff(
            base.gram,
            base.rhs,
            dummy,
            decayed_minutes,
            dummy_minutes,
            ridge_o=ridge_o,
            ridge_d=ridge_d,
            prior_mean=prior_mean,
            x0=theta,
        )
        theta = solve.theta
        coef = 2.0 * theta[HOME_COL]
        lookup = dummy_rating_lookup(solve, dummy, decayed_minutes, dummy_minutes)
        home_coef[game_idx] = coef
        for g in game_idx:
            lookups[g] = lookup
    # `WalkForward.model` is a `DecayedRidgeSparse`, plain rapm's own incremental state; this
    # variant keeps its own `BaseAccumulator` instead (module docstring), so an inert, never
    # advanced placeholder is held here only to satisfy that field's type -- nothing reads it
    # downstream (`eval.m3_backtest.rapm_margins` only reads `.lookups`/`.home_coef`).
    placeholder_model = DecayedRidgeSparse(dummy.columns.n_model, half_life_days, dummy.columns)
    return WalkForward(home_coef, lookups, dummy.columns, placeholder_model)


# --- The declared variant: tuning (D7/H-f) and the walk-forward fit --------------------------


def tune_rapm_dummy(
    data: Data, spec: M3Backtest, inputs: RapmFitInputs, earlier: dict[str, TunedRapm]
) -> TunedRapm:
    """H-f's grid (D7): every *nonzero* ``spec.grid.dummy_minutes`` threshold, at
    ``earlier["rapm"]``'s already-chosen half-life and ridges (0 is ``tune_rapm``'s own tuning
    point, and this variant's fit reproduces it exactly, so it is not retried). Tuning RMSE is
    computed exactly as ``tune_rapm`` does: the same tuning-season game subset, the same
    ``rapm_margins`` reader."""
    from eurohoops.eval.m3_backtest import (  # noqa: PLC0415 -- deferred: see module docstring
        TunedRapm,
        _finite_rmse,
        _round,
        rapm_margins,
    )

    base = earlier["rapm"]
    dummy = build_dummy_columns(inputs.spell_index)
    thresholds = [t for t in spec.grid.dummy_minutes if t > 0.0]
    last_tuning_season = max(spec.tuning)
    sub_mask = data.games["season"].to_numpy() <= last_tuning_season
    games_tuning = data.games[sub_mask].reset_index(drop=True)
    tuning_mask = data.split["tuning"][sub_mask]
    actual = data.margin[sub_mask]
    tuning_game_ids = set(games_tuning["game_id"].astype(str))
    shares_tuning = data.shares[data.shares["game_id"].astype(str).isin(tuning_game_ids)]

    def rmse_for(threshold: float) -> float:
        wf = fit_walk_forward_dummy(
            games_tuning,
            inputs.rows,
            inputs.minutes,
            inputs.spell_index,
            dummy,
            half_life_days=base.half_life_days,
            ridge_o=base.ridge_o,
            ridge_d=base.ridge_d,
            dummy_minutes=threshold,
        )
        pred = rapm_margins(games_tuning, wf, shares_tuning, data.possessions)
        return _finite_rmse(pred, actual, tuning_mask)

    if not thresholds:
        # spec.grid.dummy_minutes has only 0.0 (off): degenerate, but still a well-defined
        # variant -- it falls back to `rapm`'s own tuning point (0 reproduces it exactly).
        grid_report: dict[str, Any] = {"dummy_minutes": [], "candidates": [], "size": 0}
        return TunedRapm(
            base.half_life_days,
            base.ridge_o,
            base.ridge_d,
            base.tuning_rmse,
            grid_report,
            extra={"dummy_minutes": 0.0},
        )

    candidates = [(t, rmse_for(t)) for t in thresholds]
    best_threshold, best_rmse = min(candidates, key=lambda c: c[1])
    grid_report = {
        "dummy_minutes": list(thresholds),
        "candidates": [{"dummy_minutes": t, "tuning_rmse": _round(r)} for t, r in candidates],
        "size": len(candidates),
    }
    return TunedRapm(
        base.half_life_days,
        base.ridge_o,
        base.ridge_d,
        best_rmse,
        grid_report,
        extra={"dummy_minutes": best_threshold},
    )


def fit_rapm_dummy(games: pd.DataFrame, inputs: RapmFitInputs, tuned: TunedRapm) -> WalkForward:
    """The ``rapm_dummy`` variant's walk-forward fit over ``games`` with tuned parameters."""
    dummy = build_dummy_columns(inputs.spell_index)
    threshold = float(tuned.extra.get("dummy_minutes", 0.0))
    return fit_walk_forward_dummy(
        games,
        inputs.rows,
        inputs.minutes,
        inputs.spell_index,
        dummy,
        half_life_days=tuned.half_life_days,
        ridge_o=tuned.ridge_o,
        ridge_d=tuned.ridge_d,
        dummy_minutes=threshold,
    )
