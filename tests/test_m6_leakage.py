"""Leakage tests for the M6 walk-forward (weeks 16-18 L7): nothing a target's projections are a
function of may depend on a game at or after its cutoff (next-season targets: any game of the
target season or later; checkpoint targets: any game from the first tip-off of round
floor(f * R) + 1, decisions D6, D11, D12, D13 and D15 in reports/week16-18_progress.md).

What is guaranteed, for EuroLeague and GBL targets, next-season and at a checkpoint, in a tuning
season (drift refitted for the target) and a test season (drift frozen at the first validation
season):

* **Future edits** (the player lines at or after the cutoff get other counts or are deleted, in
  the target competition and, whole seasons, in the other one; later seasons likewise; BRAPM
  snapshots of the target season and later are moved or deleted; the ages and the M4 translations
  of later seasons are moved or deleted): the target's history, ages, aging curve (every array),
  drift, translation, impact rows, targets (exposure included), the projections of every cell
  (four variants x every half-life), the baselines, the D11 standardisation (``loss_sds``) and the
  SPM noise unit are bit-identical. ``truth`` is the scoring side and must move, which proves the
  edit took effect.
* **Deleting** the last round of the season and the playoffs: the scored set is a truth-side
  floor on the target window's possessions and would shrink with the deletion, so the suite lowers
  the floor (``min_poss`` 1); every frame is then compared whole. The baselines predict the stats
  the truth has, so a deleted target-season BRAPM snapshot (no truth for it) is compared without
  the ``brapm`` baseline rows.
* **Sensitivity**: an edit before the cutoff (the previous season, the games of the current season
  before the cutoff, an earlier BRAPM snapshot) does move the history, the aging curve and the
  projections, so the comparisons are not vacuous.
* **The similarity pool** (``embed``): the walk-forward pool (seasons before the query) is
  standardised only from those seasons.

Each check is a helper that returns the names of what changed (empty = clean). The real tests
assert it is empty; the planted-leak tests monkeypatch a leak into the harness (an aging curve
fitted through the target season, a prior that includes it, a similarity pool standardised with a
later season) and assert the very same helper reports it, which proves the helpers can fail.
"""

import dataclasses
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from functools import cache
from typing import Any

import numpy as np
import pandas as pd
import pytest

from eurohoops.config import M6Backtest
from eurohoops.eval import m6_backtest
from eurohoops.eval.m6_backtest import (
    M6Inputs,
    TargetInputs,
    baseline_predictions,
    cells,
    loss_sds,
    project_cell,
    spm_noise_unit,
    target_inputs,
)
from eurohoops.models import projection
from eurohoops.models.player_seasons import COUNT_COLUMNS, rate_table
from eurohoops.models.similarity import embed, neighbours, raw_features
from tests.m6_synthetic import build_inputs, small_spec, synthetic_rounds

# Heavy (whole synthetic leagues per test): CI runs it on PRs, nightly and model changes.
pytestmark = pytest.mark.slow

# Tuning 2015-16, validation 2017, test 2018-19; first tuning season 2015. The possessions floor of
# the scored set is lowered to 1 (a truth-side filter on the target window): a deletion would
# otherwise shrink the scored set, and the checks compare whole frames of the same players.
SPEC = replace(small_spec(), min_poss=1.0)
INPUTS = build_inputs()
SEASONS = (2016, 2018)  # a tuning season and a test season (frozen drift)
# EuroLeague (the costly league, with the impact stats): a tuning next-season target and a test
# checkpoint target; the GBL: all four.
CASES = [
    ("euroleague", 2016, 0.0),
    ("euroleague", 2018, 0.5),
    *[("gbl", s, f) for s in SEASONS for f in (0.0, 0.5)],
]

CELLS = cells(SPEC)
# The sensitivity checks run a subset of the cases (each costs a full target block): both leagues,
# both seasons, both kinds of target.
SENSITIVE = [
    ("euroleague", 2018, 0.0),
    ("euroleague", 2016, 0.5),
    ("gbl", 2016, 0.0),
    ("gbl", 2018, 0.5),
]

Select = Callable[[str, "pd.Series[Any]", "pd.Series[Any]"], "pd.Series[bool]"]


# --- Equality, snapshot and the check ----------------------------------------------------------


def _equal_frames(a: pd.DataFrame, b: Any) -> bool:
    if not isinstance(b, pd.DataFrame):
        return False
    try:
        pd.testing.assert_frame_equal(a, b, check_exact=True)
    except AssertionError:
        return False
    return True


def _equal_arrays(a: np.ndarray[Any, Any], b: Any) -> bool:
    return (
        isinstance(b, np.ndarray)
        and a.shape == b.shape
        and a.dtype == b.dtype
        and bool(np.array_equal(a, b, equal_nan=a.dtype.kind == "f"))
    )


def _equal(a: Any, b: Any) -> bool:
    """Bit-for-bit equality of frames, arrays, mappings, dataclasses and scalars (NaN == NaN)."""
    if isinstance(a, pd.DataFrame):
        return _equal_frames(a, b)
    if isinstance(a, np.ndarray):
        return _equal_arrays(a, b)
    if isinstance(a, Mapping):
        return isinstance(b, Mapping) and list(a) == list(b) and all(_equal(a[k], b[k]) for k in a)
    if dataclasses.is_dataclass(a) and not isinstance(a, type):
        return type(a) is type(b) and all(
            _equal(getattr(a, f.name), getattr(b, f.name)) for f in dataclasses.fields(a)
        )
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (math.isnan(a) and math.isnan(b))
    return bool(a == b)


@dataclass(frozen=True)
class Snap:
    """What a target's projections are a function of (``parts``, compared whole), its truth (the
    scoring side) and the players it scores."""

    parts: dict[str, Any]
    truth: pd.DataFrame
    persons: tuple[str, ...]


def _snapshot(
    inputs: M6Inputs, competition: str, season: int, checkpoint: float, spec: M6Backtest = SPEC
) -> Snap:
    world = m6_backtest.prepare(inputs)
    ti: TargetInputs = target_inputs(
        inputs, competition, season, checkpoint, spec=spec, rounds=synthetic_rounds, world=world
    )
    parts: dict[str, Any] = {
        "history": ti.history,
        "ages": ti.ages,
        "impact": ti.impact,
        "curve": ti.curve,
        "drift": ti.drift,
        "translation": ti.translation,
        "targets": ti.targets,
        "projections": {m6_backtest.cell_key(v, h): project_cell(ti, v, h) for v, h in CELLS},
        "baselines": baseline_predictions(ti),
        "loss_sds": loss_sds(world, spec, competition),
    }
    if competition == m6_backtest.IMPACT_COMPETITION:
        parts["spm_noise_unit"] = spm_noise_unit(world, season)
    return Snap(parts, ti.truth, tuple(ti.targets["person_id"]))


@cache
def _base(case: tuple[str, int, float], spec: M6Backtest = SPEC) -> Snap:
    return _snapshot(INPUTS, *case, spec)


def _changed(base: Snap, edited: Snap) -> list[str]:
    """The names of the parts that differ between two snapshots."""
    a, b = base.parts, edited.parts
    if ("brapm" in base.truth) != ("brapm" in edited.truth):
        # the baselines predict the stats the truth has (``TargetInputs.stats``): without the
        # target season's BRAPM snapshot there is no truth for it, hence no baseline row either
        a, b = (
            {**p, "baselines": {k: v[v["stat"] != "brapm"] for k, v in p["baselines"].items()}}
            for p in (a, b)
        )
    return [name for name in a if not _equal(a[name], b[name])]


# --- Edits of the inputs -----------------------------------------------------------------------


def _cut(case: tuple[str, int, float]) -> pd.Timestamp:
    """The cutoff of a checkpoint target, or the first tip-off of the season (next-season)."""
    competition, season, checkpoint = case
    cutoff = target_inputs(
        INPUTS, competition, season, checkpoint, spec=SPEC, rounds=synthetic_rounds
    ).cutoff
    return cutoff if cutoff is not None else _start(competition, season)


def _start(competition: str, season: int) -> pd.Timestamp:
    games = INPUTS.games[competition]
    return pd.Timestamp(games.loc[games["season"] == season, "tipoff_utc"].min())


def _future(case: tuple[str, int, float], *, delete: bool) -> Select:
    """Rows a target may not read: later seasons, the other league's target season, and the
    target league's target season from the cutoff (a deletion removes only the season's last
    rounds, two rounds before its end, so that the scored window stays populated)."""
    competition, season, _ = case
    cut = _cut(case)
    if delete:  # the last round and the playoffs
        cut = _start(competition, season) + pd.Timedelta(
            days=7 * (synthetic_rounds(competition, season) - 1)
        )

    def select(comp: str, s: "pd.Series[Any]", tip: "pd.Series[Any]") -> "pd.Series[bool]":
        if comp != competition:
            return s >= season
        return (s > season) | ((s == season) & (tip >= cut))

    return select


def _earlier(case: tuple[str, int, float], *, in_season: bool) -> Select:
    """Rows a target may read: the previous season (both leagues), or the target season's games
    before the cutoff."""
    competition, season, _ = case
    cut = _cut(case)

    def select(comp: str, s: "pd.Series[Any]", tip: "pd.Series[Any]") -> "pd.Series[bool]":
        if in_season:
            return (s == season) & (tip < cut) & (comp == competition)
        return s == season - 1

    return select


def _edit_lines(inputs: M6Inputs, select: Select, *, delete: bool) -> M6Inputs:
    """The player lines picked by ``select`` get doubled counts (and two more minutes), or are
    deleted together with their games."""
    games, lines = dict(inputs.games), dict(inputs.player_games)
    for competition, frame in inputs.player_games.items():
        tip = frame["game_id"].map(inputs.games[competition].set_index("game_id")["tipoff_utc"])
        picked = select(competition, frame["season"], tip)
        if delete:
            lines[competition] = frame[~picked].reset_index(drop=True)
            g = inputs.games[competition]
            gone = select(competition, g["season"], g["tipoff_utc"])
            games[competition] = g[~gone].reset_index(drop=True)
        else:
            new = frame.copy()
            new.loc[picked, list(COUNT_COLUMNS)] = new.loc[picked, list(COUNT_COLUMNS)] * 2
            new.loc[picked, "sec"] = new.loc[picked, "sec"] + 120
            lines[competition] = new
    return replace(inputs, games=games, player_games=lines)


def _edit_brapm(inputs: M6Inputs, *, since: int, until: int | None, delete: bool) -> M6Inputs:
    """The BRAPM snapshots of seasons ``since`` ... ``until`` (open end: None) moved or deleted."""
    snaps = inputs.brapm
    hit = snaps["season"] >= since
    if until is not None:
        hit = hit & (snaps["season"] <= until)
    if delete:
        return replace(inputs, brapm=snaps[~hit].reset_index(drop=True))
    new = snaps.copy()
    new.loc[hit, "value"] = new.loc[hit, "value"] + 7.0
    new.loc[hit, "sd"] = new.loc[hit, "sd"] * 1.5
    return replace(inputs, brapm=new)


def _edit_meta(inputs: M6Inputs, season: int, *, delete: bool) -> M6Inputs:
    """The ages and the M4 translations of the seasons after ``season`` (the ages of ``season``
    itself are an input of a checkpoint's partial row, known before the season starts)."""
    ages = inputs.ages
    late = ages["season"] > season
    translations = dict(inputs.translations)
    if delete:
        ages = ages[~late].reset_index(drop=True)
        translations = {s: t for s, t in translations.items() if s <= season}
    else:
        ages = ages.copy()
        ages.loc[late, "age"] = ages.loc[late, "age"] + 1.0
        translations = {
            s: replace(t, delta={k: v * 2.0 + 0.01 for k, v in t.delta.items()})
            if s > season
            else t
            for s, t in translations.items()
        }
    return replace(inputs, ages=ages, translations=translations)


def _lines(inputs: M6Inputs, case: tuple[str, int, float], *, delete: bool) -> M6Inputs:
    return _edit_lines(inputs, _future(case, delete=delete), delete=delete)


def _everything(inputs: M6Inputs, case: tuple[str, int, float], *, delete: bool) -> M6Inputs:
    """Every kind of future edit at once: the lines, the BRAPM snapshots of the target season and
    later, the later ages and translations."""
    edited = _lines(inputs, case, delete=delete)
    edited = _edit_brapm(edited, since=case[1], until=None, delete=delete)
    return _edit_meta(edited, case[1], delete=delete)


FUTURE: dict[str, Callable[[M6Inputs, tuple[str, int, float]], M6Inputs]] = {
    "scores": lambda i, c: _everything(i, c, delete=False),
    "deleted": lambda i, c: _everything(i, c, delete=True),
}


# --- Real code: future edits change nothing a projection reads ----------------------------------


@pytest.mark.parametrize("edit", FUTURE)
@pytest.mark.parametrize("case", CASES, ids=lambda c: f"{c[0]}-{c[1]}-{c[2]}")
def test_a_future_edit_leaves_every_input_and_projection_bit_identical(
    case: tuple[str, int, float], edit: str
) -> None:
    base = _base(case)
    edited = _snapshot(FUTURE[edit](INPUTS, case), *case)
    assert _changed(base, edited) == []
    assert edited.persons == base.persons  # the scored set is part of "bit-identical"
    assert not _equal(base.truth, edited.truth)  # the edit did take effect
    if case[0] == "euroleague" and case[2] == 0.0:  # the season's BRAPM snapshot is a truth
        assert "brapm" in base.truth and ("brapm" in edited.truth) == (edit == "scores")


def test_the_snapshot_covers_every_cell_and_part() -> None:
    snap = _base(("euroleague", 2016, 0.5))
    assert len(snap.parts["projections"]) == len(CELLS) == 12
    assert set(snap.parts["baselines"]) == set(m6_backtest.BASELINES)
    assert {"history", "ages", "impact", "curve", "drift", "translation", "targets"} <= set(
        snap.parts
    )
    assert {"loss_sds", "spm_noise_unit"} <= set(snap.parts)
    assert all(len(f) for f in snap.parts["projections"].values())
    assert snap.parts["curve"].delta and snap.parts["translation"] is not None


# --- Real code: an earlier edit does move them ---------------------------------------------------


@pytest.mark.parametrize("case", SENSITIVE, ids=lambda c: f"{c[0]}-{c[1]}-{c[2]}")
def test_the_previous_season_moves_history_curve_baselines_and_projections(
    case: tuple[str, int, float],
) -> None:
    edited = _snapshot(_edit_lines(INPUTS, _earlier(case, in_season=False), delete=False), *case)
    changed = _changed(_base(case), edited)
    assert {"history", "curve", "projections", "baselines"} <= set(changed)


@pytest.mark.parametrize(
    "case", [c for c in SENSITIVE if c[2] > 0.0], ids=lambda c: f"{c[0]}-{c[1]}-{c[2]}"
)
def test_a_game_before_the_cutoff_moves_the_partial_row_and_the_projections(
    case: tuple[str, int, float],
) -> None:
    edited = _snapshot(_edit_lines(INPUTS, _earlier(case, in_season=True), delete=False), *case)
    changed = _changed(_base(case), edited)
    assert {"history", "projections"} <= set(changed)


@pytest.mark.parametrize(
    "case", [c for c in SENSITIVE if c[0] == "euroleague"], ids=lambda c: f"{c[0]}-{c[1]}-{c[2]}"
)
def test_an_earlier_brapm_snapshot_moves_the_impact_rows_and_the_projections(
    case: tuple[str, int, float],
) -> None:
    edited = _snapshot(
        _edit_brapm(INPUTS, since=case[1] - 1, until=case[1] - 1, delete=False), *case
    )
    changed = _changed(_base(case), edited)
    assert {"impact", "projections"} <= set(changed)


# --- D11 standardisation and the SPM noise unit --------------------------------------------------


def _sds(inputs: M6Inputs, competition: str) -> dict[str, float]:
    return loss_sds(m6_backtest.prepare(inputs), SPEC, competition)


@pytest.mark.parametrize("competition", ["euroleague", "gbl"])
def test_the_loss_sds_ignore_every_season_from_the_first_tuning_season(competition: str) -> None:
    first = min(SPEC.tuning)
    base = _sds(INPUTS, competition)
    after = lambda comp, s, tip: s >= first  # noqa: E731
    for delete in (False, True):
        edited = _edit_brapm(
            _edit_lines(INPUTS, after, delete=delete), since=first, until=None, delete=delete
        )
        assert _equal(_sds(edited, competition), base)
    before = lambda comp, s, tip: s == first - 1  # noqa: E731
    assert not _equal(_sds(_edit_lines(INPUTS, before, delete=False), competition), base)
    if competition == "euroleague":
        moved = _edit_brapm(INPUTS, since=first - 1, until=first - 1, delete=False)
        assert not _equal(_sds(moved, competition), base)


@pytest.mark.parametrize("before", [2014, 2016, 2018])
def test_the_spm_noise_unit_ignores_the_seasons_from_the_target(before: int) -> None:
    base = spm_noise_unit(m6_backtest.prepare(INPUTS), before)
    for delete in (False, True):
        edited = _edit_lines(INPUTS, lambda comp, s, tip: s >= before, delete=delete)
        assert spm_noise_unit(m6_backtest.prepare(edited), before) == base
    earlier = _edit_lines(INPUTS, lambda comp, s, tip: s == before - 1, delete=False)
    assert spm_noise_unit(m6_backtest.prepare(earlier), before) != base


# --- The similarity pool -------------------------------------------------------------------------

QUERY = 2016  # the pool of a query in season 2016 is the seasons before it, as for the targets


def _walk_forward_pool(seasons: pd.DataFrame, query: int) -> list[int]:
    return sorted(int(s) for s in seasons["season"].unique() if s < query)


def _leaky_pool(seasons: pd.DataFrame, query: int) -> list[int]:
    """The planted leak: a pool standardised with a season after the query's cutoff."""
    return sorted(int(s) for s in seasons["season"].unique() if s <= query + 1)


def _pool_state(inputs: M6Inputs, pool: Callable[[pd.DataFrame, int], list[int]]) -> dict[str, Any]:
    seasons = m6_backtest.prepare(inputs).complete
    emb = embed(seasons, None, pool_seasons=pool(seasons, QUERY))
    query = raw_features(seasons[seasons["season"] == QUERY - 1].head(25), None)
    return {
        "keys": emb.keys,
        "features": emb.features,
        "mean": emb.mean,
        "sd": emb.sd,
        "matrix": emb.matrix,
        "neighbours": neighbours(query, emb),
    }


def _pool_changed(
    edit: Callable[[M6Inputs], M6Inputs], pool: Callable[[pd.DataFrame, int], list[int]]
) -> list[str]:
    base, edited = _pool_state(INPUTS, pool), _pool_state(edit(INPUTS), pool)
    return [k for k in base if not _equal(base[k], edited[k])]


def _from_the_query(inputs: M6Inputs) -> M6Inputs:
    """Seasons from the query's on: other counts."""
    return _edit_lines(inputs, lambda comp, s, tip: s >= QUERY, delete=False)


def _from_the_query_deleted(inputs: M6Inputs) -> M6Inputs:
    return _edit_lines(inputs, lambda comp, s, tip: s >= QUERY, delete=True)


def test_the_walk_forward_similarity_pool_ignores_the_query_season_and_later() -> None:
    for edit in (_from_the_query, _from_the_query_deleted):
        assert _pool_changed(edit, _walk_forward_pool) == []


def test_a_season_before_the_query_moves_the_similarity_pool() -> None:
    previous = lambda i: _edit_lines(i, lambda comp, s, tip: s == QUERY - 1, delete=False)  # noqa: E731
    changed = _pool_changed(previous, _walk_forward_pool)
    assert {"mean", "sd", "matrix", "neighbours"} <= set(changed)


def test_the_leaky_pool_has_another_mean_and_sd_than_the_walk_forward_pool() -> None:
    seasons = m6_backtest.prepare(INPUTS).complete
    good = embed(seasons, None, pool_seasons=_walk_forward_pool(seasons, QUERY))
    leaky = embed(seasons, None, pool_seasons=_leaky_pool(seasons, QUERY))
    assert not np.array_equal(good.mean, leaky.mean)
    assert not np.array_equal(good.sd, leaky.sd)


# --- Planted leaks -------------------------------------------------------------------------------

LEAK_CASES = [("euroleague", 2018, 0.0), ("euroleague", 2016, 0.5)]
CONTROL_CASE = LEAK_CASES[:1]


def _spy(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Make ``prepare`` remember the complete player-season frame (every season, the future
    included) and the ages of the run in progress, for a leak to read."""
    real = m6_backtest.prepare
    seen: dict[str, Any] = {}

    def spying(inputs: M6Inputs) -> Any:
        world = real(inputs)
        seen["complete"], seen["ages"] = world.complete, inputs.ages
        return world

    monkeypatch.setattr(m6_backtest, "prepare", spying)
    return seen


def _leak_check(case: tuple[str, int, float], *, cached: bool = False) -> list[str]:
    """The very same check as the real tests, run uncached (the patches are in force): the
    parts that change when the games from the cutoff on get other counts."""
    edited = _snapshot(_lines(INPUTS, case, delete=False), *case)
    return _changed(_base(case) if cached else _snapshot(INPUTS, *case), edited)


@pytest.mark.parametrize("case", CONTROL_CASE, ids=lambda c: f"{c[0]}-{c[1]}-{c[2]}")
def test_the_check_is_clean_on_the_real_code(case: tuple[str, int, float]) -> None:
    assert _leak_check(case, cached=True) == []


@pytest.mark.parametrize("case", LEAK_CASES, ids=lambda c: f"{c[0]}-{c[1]}-{c[2]}")
def test_planted_leak_an_aging_curve_fitted_through_the_target_season(
    case: tuple[str, int, float], monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _spy(monkeypatch)
    real = m6_backtest.aging_curve

    def through_the_target(
        history: pd.DataFrame, ages: pd.DataFrame, cutoff_season: int, **kw: Any
    ) -> Any:
        full = seen["complete"]
        full = full[full["season"] <= cutoff_season].reset_index(drop=True)  # target season too
        return real(full, seen["ages"], cutoff_season + 1, **kw)

    monkeypatch.setattr(m6_backtest, "aging_curve", through_the_target)
    changed = _leak_check(case)
    assert "curve" in changed
    assert "projections" in changed
    assert "history" not in changed  # only the curve leaks: the check names what moved


@pytest.mark.parametrize("case", LEAK_CASES, ids=lambda c: f"{c[0]}-{c[1]}-{c[2]}")
def test_planted_leak_a_prior_that_includes_the_target_season(
    case: tuple[str, int, float], monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _spy(monkeypatch)
    real = projection._prior_window

    def with_the_target_season(rates: pd.DataFrame, competition: str, season: int) -> Any:
        full = rate_table(seen["complete"])
        full["poss"] = full["pts_n"]
        kept = pd.concat([rates[rates["season"] < season], full[full["season"] == season]])
        return real(kept.reset_index(drop=True), competition, season + 1)

    monkeypatch.setattr(projection, "_prior_window", with_the_target_season)
    changed = _leak_check(case)
    assert "projections" in changed
    assert "history" not in changed and "curve" not in changed


def test_planted_leak_a_similarity_pool_standardised_with_future_seasons() -> None:
    assert _pool_changed(_from_the_query, _walk_forward_pool) == []  # the real pool: clean
    for edit in (_from_the_query, _from_the_query_deleted):
        changed = _pool_changed(edit, _leaky_pool)
        assert {"mean", "sd", "matrix"} <= set(changed)  # the check flags the leaky pool
