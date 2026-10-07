"""M6 backtest (weeks 16-18 L6): replay past seasons walk-forward, project every scored player with
each variant and baseline, score the projections against what the player then did.

Targets (L-a). *Next-season*: season ``t`` projected from everything before its first tip-off.
*Rest-of-season*: season ``t`` at a checkpoint (25 / 50 / 75% of the regular season, M7's rule:
k = floor(f * R) completed rounds, cutoff = the first tip-off of round k + 1, D12), from every game
before the cutoff, the current season entering as a ``partial`` row. R is the season's last played
regular-season round (``rounds`` overrides it): M7's format R except in a season cut short, and
``season_format`` does not cover every season M6 scores. Both use the regular-season
games only (history, truth and exposure alike). A target is scored when the player has at least
``min_poss`` possessions in the target window (``min_poss * (1 - f)`` at a checkpoint: the floor
scaled to the window) and a senior season before ``t`` in either league.

Truth (L-b, D13). Per-100 rates of ``COUNT_STATS``, the proportions of ``PCT_STATS`` (NaN without
an attempt) over the target window. EuroLeague only: ``brapm`` = the target season's season-end
snapshot (next-season targets only, no snapshot exists at a cutoff) and ``spm`` = the injected
``M6Inputs.spm`` (``spm(rates_frame, before_season) -> Series``, a model fitted on data before
``before_season``) applied to the target window's box rates. ``rates_frame`` is ``spm_frame`` of
player-season rows: ``rate_table`` plus ``poss`` and the per-100 rates ``fg2a`` and ``pf``. The GBL
has no impact stats (D13). A stat whose truth is NaN for a player (no attempt) is left out of that
player's loss and the stat weights are renormalised over the stats present: with K stats scored
in the block and n present, the player's loss is ``K / n * sum of the present squared errors``.
A stat any compared model cannot project for a player (e.g. no league impact mean) is left out
the same way, for all models of the block, so the losses stay comparable (on the tuning seasons'
next-season targets the models of a block are every cell and baseline, elsewhere the chosen cell
and the baselines).

Impact inputs of a EuroLeague target. BRAPM rows: the committed snapshots of seasons before ``t``
(never ``t``'s own: that is the truth). SPM rows: ``spm`` with ``before_season = t`` applied to
every EuroLeague history row (the partial row included at a checkpoint), one measurement model
for all rows of a block. Its sd: ``sqrt(u / poss)`` with ``u`` the split-half noise unit of the
seasons before ``t`` (the games of a season alternate between two halves by tip-off order; the
SPM of the two halves of a player differ by ``u * (1/poss_a + 1/poss_b)`` in variance; halves
with fewer than ``SPM_MIN_HALF_POSS`` possessions are skipped).

Metrics (L-g). Per target player-season, possession-weighted by the target window's possessions:
the *projection loss* = mean over players of ``K / n * sum_k ((truth_k - mean_k) / SD_k)^2`` with
``SD_k`` the D11 SD of the stat (possession-weighted SD of the stat over player-seasons with at
least ``min_poss`` possessions in the seasons before the first tuning target of the competition;
BRAPM from its snapshots of those seasons, SPM from ``spm(., first tuning season)`` on the same
rows; computed here from the inputs, those seasons only). Per stat: weighted MAE, the share of
truths inside the 80% interval (unweighted; variants only) and, for the impact stats, the
weighted Gaussian CRPS of the predictive. Movers (the player's newest usable season is in the
other league) are a separate slice (n and loss).

Variants and baselines (L-f). ``VARIANTS`` x ``spec.half_lives`` are the choosable cells; the
baselines are never chosen: ``same_as_last`` (the newest usable season's rate; the next older one
when it is NaN), ``league_mean`` (the possession-weighted mean of the target competition's most
recent complete season) and ``marcel`` (weights 5/4/3 on the three most recent usable seasons of
the stat, by weight x possessions, or x attempts for a percentage; usable = any league, rates
not translated; a gap in the seasons shifts the weights, they go to the three most recent
usable seasons; regressed with 1,200 possessions - 1,200 x the league's attempts per possession
for a percentage - of that league mean; then multiplied by ``1 + 0.006 * (29 - age)`` below 29
and ``1 + 0.003 * (29 - age)`` from 29 on, no age no adjustment; impact stats the same blend
of impact values with possessions, regressed by 1,200 possessions, no age adjustment). Baselines
have no interval. League means for the impact stats are the possession-weighted means of the
EuroLeague impact rows of that season.

Walk-forward. ``target_inputs`` is the one place that touches the data of a target: the history
(rows of seasons before ``t``, plus the partial season-``t`` row at a checkpoint), the ages of
the history rows, the impact rows, the aging curve (``aging_curve(history, ages, t)``), the drift
(D6: ``fit_drift(history, t)`` for a tuning target, frozen at ``fit_drift(history, first
validation season)`` for a validation or test target), the translation (D5: M4's fit for ``t``;
a target without a fit may have no mover, asserted) and the truth. Nothing else reads a table.

Choice (L-i). On the tuning seasons' next-season targets only: the lowest loss over every
(variant, half-life) cell; a relative difference below ``tie_tolerance`` goes to the simpler
variant (``VARIANTS`` order) and within a variant to the smaller half-life. ``on_edge`` says
whether the chosen half-life is the grid's smallest or largest. ``fixed`` skips the choice (the
GBL scores the EuroLeague verdict, ``gated: false``). Rest-of-season and the splits after tuning
score only the chosen cell and the baselines.

Gate (L-h), EuroLeague validation next-season targets: the chosen cell's loss is below
``marcel``'s and ``same_as_last``'s (point rule) and its pooled tuning + validation coverage is
inside ``spec.coverage_band`` for every stat; player-cluster bootstrap 95% CIs of the loss
differences are reported.

``tuning_only`` scores the tuning seasons alone, so no validation or test number exists anywhere
(the verdict must be committed before validation is scored); ``score_test`` adds the test
seasons (only after the validation commit). Two runs on the same inputs are byte-identical.
"""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import numpy as np
import pandas as pd
import pandera.pandas as pa

from eurohoops.config import M6Backtest
from eurohoops.eval.m3_backtest import _data_sha256
from eurohoops.eval.m7_backtest import checkpoint_cutoff
from eurohoops.eval.metrics import crps_normal
from eurohoops.models.aging import AgingCurve, aging_curve, apply
from eurohoops.models.elo import FloatArray
from eurohoops.models.player_seasons import (
    COUNT_STATS,
    IMPACT_STATS,
    PCT_STATS,
    build_player_seasons,
    rate_table,
)
from eurohoops.models.projection import (
    IMPACT_SCHEMA,
    TARGET_SCHEMA,
    VARIANTS,
    Translation,
    fit_drift,
    project,
    variant_params,
)
from eurohoops.models.team_eff import IntArray
from eurohoops.parse.schemas import validated

SPLITS = ("tuning", "validation", "test")
BASELINES = ("same_as_last", "league_mean", "marcel")
GATE_BASELINES = ("marcel", "same_as_last")
BOX_STATS = COUNT_STATS + PCT_STATS
IMPACT_COMPETITION = "euroleague"  # the only competition with impact stats (D13)
MARCEL_WEIGHTS = (5, 4, 3)
MARCEL_REGRESSION = 1200.0  # possessions of the league mean added to the weighted sums
MARCEL_PEAK_AGE = 29.0
MARCEL_YOUNG_SLOPE = 0.006
MARCEL_OLD_SLOPE = 0.003
SPM_MIN_HALF_POSS = 100.0
SD_FLOOR = 1e-9
GATE_RULE = (
    "EuroLeague validation next-season projection loss: the chosen variant below marcel's and "
    "same_as_last's (point rule), and its pooled tuning + validation 80% interval coverage inside "
    "the band for every stat; player-cluster bootstrap 95% CIs of the loss differences reported"
)
PLAYER_COLUMNS = (
    "split",
    "season",
    "checkpoint",
    "model",
    "person_id",
    "stat",
    "truth",
    "mean",
    "lo80",
    "hi80",
    "sq_err_std",
    "weight",
)
PLAYERS_SCHEMA = pa.DataFrameSchema(
    {
        "split": pa.Column(str, pa.Check.isin(SPLITS)),
        "season": pa.Column("int64"),
        "checkpoint": pa.Column("float64", pa.Check.in_range(0.0, 1.0, include_max=False)),
        "model": pa.Column(str),
        "person_id": pa.Column(str),
        "stat": pa.Column(str, pa.Check.isin(BOX_STATS + IMPACT_STATS)),
        "truth": pa.Column("float64"),
        "mean": pa.Column("float64"),
        "lo80": pa.Column("float64", nullable=True),
        "hi80": pa.Column("float64", nullable=True),
        "sq_err_std": pa.Column("float64", pa.Check.ge(0.0)),
        "weight": pa.Column("float64", pa.Check.gt(0.0)),
    },
    unique=["split", "season", "checkpoint", "model", "person_id", "stat"],
    strict=True,
    ordered=True,
)


def _round(value: float | None) -> float | None:
    return None if value is None or not math.isfinite(value) else round(float(value), 6)


# --- Inputs, cells and the choice -----------------------------------------------------------------

SpmFn = Callable[[pd.DataFrame, int], "pd.Series[float]"]


@dataclass(frozen=True)
class M6Inputs:
    """Everything a run reads. ``games`` / ``player_games`` map a competition to its games mart
    frame (season, round, phase, tipoff_utc, ...) and its player-game frame (``stats.box`` /
    ``parse.gbl_box_lines``: season, game_id, team, player_id, sec, poss and the counts);
    ``xwalk`` is ``player_xwalk``; ``ages`` (``AGES_SCHEMA``) lives in memory only; ``brapm`` holds
    the season-end snapshots as ``IMPACT_SCHEMA`` rows (``stat == "brapm"``, ``sd`` = the
    snapshot's ``sd_total``; may be empty); ``spm`` is the injected SPM (module docstring);
    ``translations`` maps a target season to M4's fit for it."""

    games: Mapping[str, pd.DataFrame]
    player_games: Mapping[str, pd.DataFrame]
    xwalk: pd.DataFrame
    ages: pd.DataFrame
    brapm: pd.DataFrame
    spm: SpmFn
    translations: Mapping[int, Translation]


def cell_key(variant: str, half_life: float) -> str:
    return f"{variant}@{half_life:g}"


def cells(spec: M6Backtest) -> list[tuple[str, float]]:
    """Every (variant, half-life) cell in simplicity order: the variants in ``VARIANTS`` order,
    each with its half-lives ascending (the tie-break order)."""
    return [(v, h) for v in VARIANTS for h in sorted(spec.half_lives)]


def model_version(variant: str, half_life: float) -> str:
    return f"m6-{variant}-h{half_life:g}-v1"


def scored_splits(spec: M6Backtest, tuning_only: bool, score_test: bool) -> dict[int, str]:
    """Season -> split name for the seasons a run scores, ordered by season: tuning always,
    validation unless ``tuning_only``, test when ``score_test`` as well."""
    names = ["tuning"] if tuning_only else ["tuning", "validation"]
    if score_test and not tuning_only:
        names.append("test")
    return dict(sorted({season: name for name in names for season in getattr(spec, name)}.items()))


def choose_cell(table: Sequence[tuple[str, float, float]], tolerance: float) -> int:
    """Index of the winner of a table of (variant, half-life, tuning loss) in simplicity order:
    the lowest loss, or among every cell whose loss is within ``tolerance`` (relative, strictly
    below) of it the first, i.e. the simplest variant and then the smallest half-life."""
    finite = [(i, loss) for i, (_, _, loss) in enumerate(table) if math.isfinite(loss)]
    if not finite:
        raise ValueError("no cell has a finite tuning loss")
    best = min(loss for _, loss in finite)
    return min(i for i, loss in finite if loss == best or loss - best < tolerance * best)


def gate_block(
    variant: str,
    half_life: float,
    *,
    loss_chosen: float | None,
    loss_marcel: float | None,
    loss_same_as_last: float | None,
    coverage: Mapping[str, float | None],
    band: tuple[float, float],
    differences: Mapping[str, Any],
    gated: bool,
) -> dict[str, Any]:
    """The gate (L-h, point rule) from unrounded values; ``None`` anywhere leaves it undecided."""
    passed = None
    loss_ok = coverage_ok = None
    if None not in (loss_chosen, loss_marcel, loss_same_as_last):
        assert loss_chosen is not None and loss_marcel is not None
        assert loss_same_as_last is not None
        loss_ok = bool(loss_chosen < loss_marcel and loss_chosen < loss_same_as_last)
    if coverage and all(v is not None for v in coverage.values()):
        coverage_ok = all(band[0] <= v <= band[1] for v in coverage.values() if v is not None)
    if loss_ok is not None and coverage_ok is not None:
        passed = bool(loss_ok and coverage_ok)
    return {
        "rule": GATE_RULE,
        "gated": gated,
        "variant": variant,
        "half_life": half_life,
        "loss_chosen": _round(loss_chosen),
        "loss_marcel": _round(loss_marcel),
        "loss_same_as_last": _round(loss_same_as_last),
        "loss_ok": loss_ok,
        "coverage_pooled": {k: _round(v) for k, v in coverage.items()},
        "coverage_band": list(band),
        "coverage_ok": coverage_ok,
        "passed": passed,
        **differences,
    }


# --- The world: the tables a run reads, filtered once ---------------------------------------------


@dataclass
class World:
    """The regular-season player lines of each competition with their tip-offs, the player-season
    frame over every season (``complete``: a target filters it by season, never reads past ``t``)
    and memos of what depends on (competition, season) alone."""

    inputs: M6Inputs
    lines: dict[str, pd.DataFrame]
    complete: pd.DataFrame
    halves: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = field(default_factory=dict)
    memo: dict[tuple[Any, ...], Any] = field(default_factory=dict)


def prepare(inputs: M6Inputs) -> World:
    lines = {}
    for competition, player_games in inputs.player_games.items():
        games = inputs.games[competition]
        keep = games.loc[games["phase"] == "RS", ["game_id", "tipoff_utc"]]
        lines[competition] = player_games.merge(keep, on="game_id", how="inner")
    return World(inputs, lines, build_player_seasons(lines, inputs.xwalk))


def spm_frame(seasons: pd.DataFrame) -> pd.DataFrame:
    """The rates frame ``M6Inputs.spm`` receives: ``rate_table`` of the player-season rows, plus
    ``poss`` and the per-100 rates ``fg2a`` and ``pf`` (SPM's 11 stats are ``COUNT_STATS`` with
    these two); the index is 0..n-1 in the rows' order."""
    rows = seasons.reset_index(drop=True)
    frame = rate_table(rows)
    poss = rows["poss"].to_numpy(dtype=np.float64)
    frame["poss"] = poss
    for stat in ("fg2a", "pf"):
        frame[stat] = 100.0 * rows[stat].to_numpy(dtype=np.float64) / np.maximum(poss, 1e-12)
    return frame


def _spm_values(spm: SpmFn, frame: pd.DataFrame, before: int) -> FloatArray:
    values = np.asarray(spm(frame, before), dtype=np.float64)
    if values.shape != (len(frame),):
        raise ValueError("the injected spm must return one value per row of the rates frame")
    return values


def _halves(world: World, season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The two half-seasons (alternate games by tip-off order) of a EuroLeague season as
    ``spm_frame`` rows, with their ``person_id``."""
    if season not in world.halves:
        lines = world.lines[IMPACT_COMPETITION]
        lines = lines[lines["season"] == season]
        order = (
            lines[["game_id", "tipoff_utc"]]
            .drop_duplicates()
            .sort_values(["tipoff_utc", "game_id"])
        )
        parity = dict(zip(order["game_id"], np.arange(len(order)) % 2, strict=True))
        side = lines["game_id"].map(parity)
        halves = []
        for half in (0, 1):
            rows = build_player_seasons(
                {IMPACT_COMPETITION: lines[side == half]}, world.inputs.xwalk
            )
            frame = spm_frame(rows)
            frame["person_id"] = rows["person_id"].to_numpy()
            halves.append(frame)
        world.halves[season] = (halves[0], halves[1])
    return world.halves[season]


def spm_noise_unit(world: World, before: int) -> float:
    """``u`` in sd(SPM of a row with ``poss`` possessions) = sqrt(u / poss): the split-half
    estimate over the EuroLeague seasons before ``before`` (module docstring)."""
    key = ("spm_noise", before)
    if key not in world.memo:
        el = world.complete[world.complete["competition"] == IMPACT_COMPETITION]
        squares = weights = 0.0
        for season in sorted(int(s) for s in el["season"].unique() if s < before):
            sides = []
            for frame in _halves(world, season):
                values = _spm_values(world.inputs.spm, frame, before)
                sides.append(
                    pd.DataFrame(
                        {"poss": frame["poss"].to_numpy(), "spm": values},
                        index=pd.Index(frame["person_id"]),
                    )
                )
            both = sides[0].join(sides[1], how="inner", lsuffix="_a", rsuffix="_b")
            both = both[
                (both["poss_a"] >= SPM_MIN_HALF_POSS) & (both["poss_b"] >= SPM_MIN_HALF_POSS)
            ]
            squares += float(((both["spm_a"] - both["spm_b"]) ** 2).sum())
            weights += float((1.0 / both["poss_a"] + 1.0 / both["poss_b"]).sum())
        if weights <= 0.0:
            raise ValueError(f"no split-half SPM pairs before season {before}")
        world.memo[key] = squares / weights
    return float(world.memo[key])


def _weighted_sd(x: FloatArray, w: FloatArray, stat: str) -> float:
    ok = np.isfinite(x) & (w > 0)
    if ok.sum() < 2:
        raise ValueError(f"no loss SD for {stat}: fewer than two player-seasons before tuning")
    x, w = x[ok], w[ok]
    mean = float((w * x).sum() / w.sum())
    sd = math.sqrt(float((w * (x - mean) ** 2).sum() / w.sum()))
    if sd <= 0.0:
        raise ValueError(f"the loss SD of {stat} is 0")
    return sd


def loss_sds(world: World, spec: M6Backtest, competition: str) -> dict[str, float]:
    """The D11 SDs: possession-weighted SD of each stat over the competition's player-seasons
    with at least ``spec.min_poss`` possessions in the seasons before the first tuning target
    (impact stats: EuroLeague only; BRAPM from its snapshots of those seasons, SPM from the
    injected model for the first tuning season on the same rows). Nothing at or after the first
    tuning season is read."""
    first = min(spec.tuning)
    seasons = world.complete
    rows = seasons[
        (seasons["competition"] == competition)
        & (seasons["season"] < first)
        & (seasons["poss"] >= spec.min_poss)
    ].reset_index(drop=True)
    if rows.empty:
        raise ValueError(f"no {competition} player-seasons before season {first}")
    poss = rows["poss"].to_numpy(dtype=np.float64)
    rates = rate_table(rows)
    out = {s: _weighted_sd(rates[s].to_numpy(dtype=np.float64), poss, s) for s in BOX_STATS}
    if competition == IMPACT_COMPETITION:
        spm = _spm_values(world.inputs.spm, spm_frame(rows), first)
        out["spm"] = _weighted_sd(spm, poss, "spm")
        snaps = world.inputs.brapm
        snaps = snaps[
            (snaps["competition"] == competition)
            & (snaps["stat"] == "brapm")
            & (snaps["season"] < first)
        ]
        joined = snaps.merge(
            rows[["person_id", "competition", "season", "poss"]],
            on=["person_id", "competition", "season"],
        )
        if len(joined) >= 2:
            out["brapm"] = _weighted_sd(
                joined["value"].to_numpy(dtype=np.float64),
                joined["poss"].to_numpy(dtype=np.float64),
                "brapm",
            )
    return out


# --- The inputs of one target block ---------------------------------------------------------------


@dataclass(frozen=True)
class TargetInputs:
    """Everything the projections of one (competition, season, checkpoint) are a function of: the
    walk-forward rule says none of it may depend on a game at or after the cutoff (a season
    before ``season`` for a next-season target). ``truth`` is the scoring side: one row per
    scored player with ``poss`` (the target window's possessions), ``mover`` and the truth of
    each stat in ``stats``."""

    competition: str
    season: int
    checkpoint: float
    cutoff: pd.Timestamp | None
    stats: tuple[str, ...]
    history: pd.DataFrame
    ages: pd.DataFrame
    impact: pd.DataFrame | None
    curve: AgingCurve
    drift: dict[str, float]
    translation: Translation | None
    targets: pd.DataFrame
    truth: pd.DataFrame


def _history(
    world: World, competition: str, season: int, cutoff: pd.Timestamp | None
) -> pd.DataFrame:
    """Every player-season row of a season before ``season`` and, at a checkpoint, the partial
    ``season`` row of ``competition`` built from the games before the cutoff."""
    prior = world.complete[world.complete["season"] < season]
    if cutoff is None:
        return prior.reset_index(drop=True)
    lines = world.lines[competition]
    before = lines[(lines["season"] == season) & (lines["tipoff_utc"] < cutoff)]
    part = build_player_seasons(
        {competition: before}, world.inputs.xwalk, partial={competition: season}
    )
    debut = prior.groupby("person_id")["debut_season"].min()
    part["debut_season"] = (
        part["person_id"].map(debut).fillna(season).clip(upper=season).astype("int64")
    )
    both = pd.concat([prior, part], ignore_index=True)
    return both.sort_values(["competition", "season", "person_id"]).reset_index(drop=True)


def _target_rows(
    world: World, competition: str, season: int, cutoff: pd.Timestamp | None
) -> pd.DataFrame:
    """The player-season rows of the target window: the season, or its games from the cutoff."""
    if cutoff is None:
        complete = world.complete
        rows = complete[(complete["competition"] == competition) & (complete["season"] == season)]
        return rows.reset_index(drop=True)
    lines = world.lines[competition]
    after = lines[(lines["season"] == season) & (lines["tipoff_utc"] >= cutoff)]
    return build_player_seasons({competition: after}, world.inputs.xwalk)


def _newest(history: pd.DataFrame, competition: str) -> pd.DataFrame:
    """Per person the newest usable row (the partial season-``t`` one included; a tie goes to
    ``competition``'s)."""
    usable = history.assign(mine=history["competition"] == competition)
    usable = usable.sort_values(["person_id", "season", "mine"], kind="stable")
    return usable.drop_duplicates("person_id", keep="last").set_index("person_id")


def _impact_rows(world: World, history: pd.DataFrame, season: int) -> pd.DataFrame:
    """IMPACT_SCHEMA rows of a EuroLeague target (module docstring)."""
    inputs = world.inputs
    el = history[(history["competition"] == IMPACT_COMPETITION) & (history["poss"] > 0)]
    unit = spm_noise_unit(world, season)
    spm = pd.DataFrame(
        {
            "person_id": el["person_id"].to_numpy(),
            "competition": IMPACT_COMPETITION,
            "season": el["season"].to_numpy(),
            "stat": "spm",
            "value": _spm_values(inputs.spm, spm_frame(el), season),
            "sd": np.sqrt(unit / el["poss"].to_numpy(dtype=np.float64)),
        }
    )
    snaps = inputs.brapm
    snaps = snaps[
        (snaps["competition"] == IMPACT_COMPETITION)
        & (snaps["stat"] == "brapm")
        & (snaps["season"] < season)
    ]
    rows = pd.concat([snaps[list(IMPACT_SCHEMA.columns)], spm], ignore_index=True)
    rows = rows.sort_values(["stat", "season", "person_id"]).reset_index(drop=True)
    return validated(rows, IMPACT_SCHEMA)


def _exposure(
    history: pd.DataFrame, people: pd.Series, season: int, checkpoint: float
) -> FloatArray:
    """The reference possessions of the predictive interval: the player's possessions in his
    newest season before ``season``; at a checkpoint his partial-season possessions scaled by
    (1 - f) / f (D12), or, without any, that newest season's scaled by 1 - f."""
    prior = history[history["season"] < season]
    newest = prior[prior["season"] == prior.groupby("person_id")["season"].transform("max")]
    last = people.map(newest.groupby("person_id")["poss"].sum()).to_numpy(dtype=np.float64)
    if checkpoint == 0.0:
        floored: FloatArray = np.maximum(last, 1.0)
        return floored
    part = history[history["partial"]].groupby("person_id")["poss"].sum()
    own = people.map(part).to_numpy(dtype=np.float64)
    scaled = np.where(own > 0, own * (1.0 - checkpoint) / checkpoint, last * (1.0 - checkpoint))
    return np.maximum(scaled, 1.0)


def _truth(
    inputs: M6Inputs, scored: pd.DataFrame, competition: str, season: int, checkpoint: float
) -> pd.DataFrame:
    """The truth of each scored player (``scored``: the target window's rows): possessions and
    the stats (module docstring); BRAPM only for a next-season EuroLeague target with snapshots."""
    rates = rate_table(scored)
    truth = pd.DataFrame({"person_id": scored["person_id"], "poss": scored["poss"]})
    for stat in BOX_STATS:
        truth[stat] = rates[stat].to_numpy()
    if competition == IMPACT_COMPETITION:
        truth["spm"] = _spm_values(inputs.spm, spm_frame(scored), season)
        snaps = inputs.brapm
        snaps = snaps[
            (snaps["competition"] == competition)
            & (snaps["stat"] == "brapm")
            & (snaps["season"] == season)
        ]
        if checkpoint == 0.0 and not snaps.empty:
            truth["brapm"] = (
                truth["person_id"].map(snaps.set_index("person_id")["value"]).to_numpy()
            )
    return truth


def _played_rounds(regular: pd.DataFrame) -> int:
    """R: the season's last played regular-season round. It is the format's R of M7 except in a
    season cut short (EuroLeague 2019-20); ``season_format`` does not cover every season M6
    scores (it excludes 2019, 2021 and the GBL's 2023-24)."""
    return int(regular.loc[regular["played"], "round"].max())


def target_inputs(
    inputs: M6Inputs,
    competition: str,
    season: int,
    checkpoint: float,
    *,
    spec: M6Backtest,
    rounds: Callable[[str, int], int] | None = None,
    world: World | None = None,
) -> TargetInputs:
    """The inputs of one target block (``checkpoint`` 0.0: next-season; else the fraction of the
    season's regular-season rounds played at the cutoff). ``world`` is ``prepare(inputs)``, passed
    to share it between calls."""
    if not 0.0 <= checkpoint < 1.0:
        raise ValueError(f"checkpoint must be in [0, 1), not {checkpoint}")
    world = prepare(inputs) if world is None else world
    cutoff = None
    if checkpoint > 0.0:
        games = inputs.games[competition]
        regular = games[(games["season"] == season) & (games["phase"] == "RS")]
        total = rounds(competition, season) if rounds else _played_rounds(regular)
        done = math.floor(checkpoint * total)
        cutoff = checkpoint_cutoff(regular, done)
    history = _history(world, competition, season, cutoff)
    window = _target_rows(world, competition, season, cutoff)
    prior_people = set(history.loc[history["season"] < season, "person_id"])
    floor = spec.min_poss * (1.0 - checkpoint)
    scored = window[(window["poss"] >= floor) & window["person_id"].isin(prior_people)]
    scored = scored.sort_values("person_id").reset_index(drop=True)
    if scored.empty:
        raise ValueError(f"no scored players in {competition} {season} at checkpoint {checkpoint}")

    people = scored["person_id"]
    movers = (_newest(history, competition).loc[people, "competition"] != competition).to_numpy()
    translation = inputs.translations.get(season)
    if translation is None and movers.any():
        raise ValueError(f"a mover in {competition} {season} but no M4 translation for it (D5)")
    targets = pd.DataFrame(
        {
            "person_id": people.to_numpy(),
            "competition": competition,
            "season": season,
            "checkpoint": checkpoint,
            "exposure": _exposure(history, people, season, checkpoint),
        }
    )
    targets = validated(targets.astype({"season": "int64", "checkpoint": "float64"}), TARGET_SCHEMA)
    impact = _impact_rows(world, history, season) if competition == IMPACT_COMPETITION else None
    truth = _truth(inputs, scored, competition, season, checkpoint)
    truth.insert(2, "mover", movers)
    ages = inputs.ages.merge(
        history[["person_id", "season"]].drop_duplicates(), on=["person_id", "season"]
    )
    drift_before = season if season in spec.tuning else spec.validation[0]
    memo = world.memo
    if ("curve", competition, season) not in memo:
        memo["curve", competition, season] = aging_curve(history, ages, season, impact=impact)
    if ("drift", competition, season, drift_before) not in memo:
        memo["drift", competition, season, drift_before] = fit_drift(history, drift_before, impact)
    return TargetInputs(
        competition=competition,
        season=season,
        checkpoint=checkpoint,
        cutoff=cutoff,
        stats=tuple(c for c in truth.columns if c in BOX_STATS + IMPACT_STATS),
        history=history,
        ages=ages,
        impact=impact,
        curve=memo["curve", competition, season],
        drift=memo["drift", competition, season, drift_before],
        translation=translation,
        targets=targets,
        truth=truth,
    )


# --- Projections and baselines --------------------------------------------------------------------


def project_cell(ti: TargetInputs, variant: str, half_life: float) -> pd.DataFrame:
    """The long ``PROJECTIONS_SCHEMA`` frame of one cell for a block's targets."""
    params = variant_params(variant, half_life)
    return project(
        ti.history,
        ti.targets,
        params,
        drift=ti.drift,
        impact=ti.impact,
        ages=ti.ages,
        aging=partial(apply, ti.curve) if params.aging else None,
        translation=ti.translation,
    )


def _stat_rows(ti: TargetInputs, rates: pd.DataFrame, stat: str) -> pd.DataFrame:
    """The usable (person, season, x, n) rows of one stat, ``mine`` flagging the target
    competition's (ties in a season go to it). Impact rows carry the possessions of their season."""
    if stat in IMPACT_STATS:
        if ti.impact is None:
            return pd.DataFrame(columns=["person_id", "season", "mine", "x", "n"])
        own = ti.impact[ti.impact["stat"] == stat].merge(
            ti.history[["person_id", "competition", "season", "poss"]],
            on=["person_id", "competition", "season"],
        )
        x, n, frame = own["value"], own["poss"], own
    else:
        x, n, frame = rates[stat], rates[f"{stat}_n"], rates
    rows = pd.DataFrame(
        {
            "person_id": frame["person_id"].to_numpy(),
            "season": frame["season"].to_numpy(),
            "mine": (frame["competition"] == ti.competition).to_numpy(),
            "x": x.to_numpy(dtype=np.float64),
            "n": n.to_numpy(dtype=np.float64),
        }
    )
    usable: pd.DataFrame = rows[np.isfinite(rows["x"]) & (rows["n"] > 0)]
    return usable


def _recent(rows: pd.DataFrame, limit: int) -> pd.DataFrame:
    """Each person's ``limit`` most recent rows with their rank 0, 1, ... (newest first)."""
    rows = rows.sort_values(
        ["person_id", "season", "mine"], ascending=[True, False, False], kind="stable"
    )
    rows = rows.assign(rank=rows.groupby("person_id").cumcount())
    return rows[rows["rank"] < limit]


def _league_level(ti: TargetInputs, rates: pd.DataFrame, stat: str) -> tuple[float, float]:
    """The league mean of a stat and its attempts per possession, from the target competition's
    most recent complete season (NaN when there is none, e.g. no impact rows)."""
    complete = ti.history[(ti.history["competition"] == ti.competition) & ~ti.history["partial"]]
    if complete.empty:
        raise ValueError(f"no complete {ti.competition} season before {ti.season}")
    last = complete["season"].max()
    if stat in IMPACT_STATS:
        if ti.impact is None:
            return math.nan, 1.0
        own = ti.impact[(ti.impact["stat"] == stat) & (ti.impact["season"] == last)]
        own = own.merge(
            complete[["person_id", "competition", "season", "poss"]],
            on=["person_id", "competition", "season"],
        )
        if own["poss"].sum() <= 0:
            return math.nan, 1.0
        return float((own["poss"] * own["value"]).sum() / own["poss"].sum()), 1.0
    sub = rates[
        (rates["competition"] == ti.competition) & ~rates["partial"] & (rates["season"] == last)
    ]
    ok = np.isfinite(sub[stat]) & (sub[f"{stat}_n"] > 0)
    n = sub.loc[ok, f"{stat}_n"]
    mean = float((n * sub.loc[ok, stat]).sum() / n.sum())
    return mean, float(sub[f"{stat}_n"].sum() / sub["pts_n"].sum())


def marcel_age_factor(age: FloatArray) -> FloatArray:
    """Marcel's fixed age adjustment (1 for an unknown age)."""
    slope = np.where(age < MARCEL_PEAK_AGE, MARCEL_YOUNG_SLOPE, MARCEL_OLD_SLOPE)
    factor: FloatArray = np.where(np.isnan(age), 1.0, 1.0 + slope * (MARCEL_PEAK_AGE - age))
    return factor


def marcel_blend(
    recent: pd.DataFrame, people: pd.Series, mean: float, per_poss: float
) -> FloatArray:
    """Marcel's regressed blend per person (module docstring) from each person's three most
    recent rows (``_recent``): weights 5/4/3 times their exposure, plus ``MARCEL_REGRESSION``
    possessions (x ``per_poss`` attempts per possession) of the league ``mean``; a person without
    rows gets ``mean``."""
    weight = recent["rank"].map(dict(enumerate(MARCEL_WEIGHTS))).to_numpy(dtype=np.float64)
    exposure = weight * recent["n"].to_numpy()
    sums = (
        pd.DataFrame(
            {
                "num": exposure * recent["x"].to_numpy(),
                "den": exposure,
                "person_id": recent["person_id"].to_numpy(),
            }
        )
        .groupby("person_id")
        .sum()
    )
    regression = MARCEL_REGRESSION * per_poss
    num = people.map(sums["num"]).to_numpy(dtype=np.float64)
    den = people.map(sums["den"]).to_numpy(dtype=np.float64)
    blend: FloatArray = (np.nan_to_num(num) + regression * mean) / (np.nan_to_num(den) + regression)
    return blend


def baseline_predictions(ti: TargetInputs) -> dict[str, pd.DataFrame]:
    """Every baseline's point predictions (person_id, stat, mean; the interval columns NaN) for
    a block's targets, keyed by name (module docstring)."""
    rates = rate_table(ti.history)
    people = ti.targets["person_id"]
    offsets = (ti.ages["age"] - ti.ages["season"]).groupby(ti.ages["person_id"]).mean()
    factor = marcel_age_factor(people.map(offsets).to_numpy(dtype=np.float64) + ti.season)
    values: dict[str, list[pd.DataFrame]] = {name: [] for name in BASELINES}
    for stat in ti.stats:
        mean, per_poss = _league_level(ti, rates, stat)
        rows = _stat_rows(ti, rates, stat)
        recent = _recent(rows, len(MARCEL_WEIGHTS))
        newest = recent[recent["rank"] == 0].set_index("person_id")["x"]
        last = people.map(newest).to_numpy(dtype=np.float64)
        blend = marcel_blend(recent, people, mean, per_poss)
        if stat not in IMPACT_STATS:
            blend = np.clip(blend * factor, 0.0, 1.0 if stat in PCT_STATS else None)
        by_name = {
            "same_as_last": np.where(np.isnan(last), mean, last),
            "league_mean": np.full(len(people), mean),
            "marcel": blend,
        }
        for name, value in by_name.items():
            values[name].append(
                pd.DataFrame({"person_id": people.to_numpy(), "stat": stat, "mean": value})
            )
    out = {}
    for name, frames in values.items():
        out[name] = pd.concat(frames, ignore_index=True)
        out[name]["lo80"] = out[name]["hi80"] = out[name]["sd"] = np.nan
    return out


# --- Scoring --------------------------------------------------------------------------------------


def score_block(
    ti: TargetInputs, predictions: Mapping[str, pd.DataFrame], sds: Mapping[str, float]
) -> dict[str, pd.DataFrame]:
    """One long frame per model: person_id, stat, truth, mean, lo80, hi80, sd, weight, mover, k,
    sq_err_std, abs_err, covered, crps, ordered by person then stat. The stats of a player are
    those with a finite truth and a finite prediction from every model (module docstring)."""
    stats = [s for s in ti.stats if s in sds]
    long = ti.truth.melt(
        id_vars=["person_id", "poss", "mover"],
        value_vars=stats,
        var_name="stat",
        value_name="truth",
    )
    long["order"] = long["stat"].map({s: i for i, s in enumerate(stats)})
    long = long[np.isfinite(long["truth"])]
    key = ["person_id", "stat"]
    common = long[key]
    for frame in predictions.values():
        common = common.merge(frame.loc[np.isfinite(frame["mean"]), key], on=key)
    base = long.merge(common, on=key).sort_values(["person_id", "order"], kind="stable")
    base = base.rename(columns={"poss": "weight"}).drop(columns="order")
    out = {}
    for model, frame in predictions.items():
        rows = base.merge(frame[[*key, "mean", "lo80", "hi80", "sd"]], on=key, how="left")
        scale = rows["stat"].map(sds).to_numpy(dtype=np.float64)
        error = rows["truth"].to_numpy() - rows["mean"].to_numpy()
        rows["k"] = len(stats)
        rows["sq_err_std"] = (error / scale) ** 2
        rows["abs_err"] = np.abs(error)
        inside = (rows["truth"] >= rows["lo80"]) & (rows["truth"] <= rows["hi80"])
        rows["covered"] = np.where(rows["lo80"].notna(), inside.astype(float), np.nan)
        crps = np.full(len(rows), np.nan)
        impact = rows["stat"].isin(IMPACT_STATS).to_numpy() & rows["sd"].notna().to_numpy()
        if impact.any():
            crps[impact] = crps_normal(
                rows.loc[impact, "mean"].to_numpy(dtype=np.float64),
                np.maximum(rows.loc[impact, "sd"].to_numpy(dtype=np.float64), SD_FLOOR),
                rows.loc[impact, "truth"].to_numpy(dtype=np.float64),
            )
        rows["crps"] = crps
        out[model] = rows.reset_index(drop=True)
    return out


def player_losses(rows: pd.DataFrame) -> pd.DataFrame:
    """One model's rows -> one row per (season, checkpoint, person): the projection loss
    ``K / n * sum of the present squared errors``, the weight and the mover flag."""
    grouped = rows.groupby(["season", "checkpoint", "person_id"], sort=True)
    out = grouped.agg(
        total=("sq_err_std", "sum"),
        n=("sq_err_std", "size"),
        k=("k", "first"),
        weight=("weight", "first"),
        mover=("mover", "first"),
    )
    out["loss"] = out["total"] * out["k"] / out["n"]
    return out


def _weighted_mean(x: pd.Series, w: pd.Series) -> float | None:
    return float((x * w).sum() / w.sum()) if len(x) and w.sum() > 0 else None


def model_scores(rows: pd.DataFrame, stats: Sequence[str]) -> dict[str, Any]:
    """Unrounded metrics of one model's pooled rows: n, loss, per-stat MAE and coverage (None for
    a point-only model) and, for the impact stats, CRPS."""
    losses = player_losses(rows)
    mae: dict[str, float | None] = {}
    coverage: dict[str, float | None] = {}
    crps: dict[str, float | None] = {}
    for stat in stats:
        sub = rows[rows["stat"] == stat]
        mae[stat] = _weighted_mean(sub["abs_err"], sub["weight"])
        seen = sub["covered"].dropna()
        coverage[stat] = float(seen.mean()) if len(seen) else None
        if stat in IMPACT_STATS:
            has = sub[sub["crps"].notna()]
            crps[stat] = _weighted_mean(has["crps"], has["weight"])
    return {
        "n": len(losses),
        "loss": _weighted_mean(losses["loss"], losses["weight"]),
        "mae": mae,
        "coverage": coverage if any(v is not None for v in coverage.values()) else None,
        "crps": crps,
    }


def _rounded(scores: dict[str, Any]) -> dict[str, Any]:
    def each(values: dict[str, float | None] | None) -> dict[str, float | None] | None:
        return None if values is None else {k: _round(v) for k, v in values.items()}

    return {
        "n": scores["n"],
        "loss": _round(scores["loss"]),
        "mae": each(scores["mae"]),
        "coverage": each(scores["coverage"]),
        "crps": each(scores["crps"]),
    }


def cluster_bootstrap_ci(
    numerator: FloatArray, denominator: FloatArray, cluster: IntArray, resamples: int, seed: int
) -> tuple[float, float, float]:
    """The weighted mean ``sum(numerator) / sum(denominator)`` and the percentile 95% CI of it
    over resamples of whole clusters (ids 0..C-1, drawn with replacement); NaN when empty."""
    if not len(numerator):
        return math.nan, math.nan, math.nan
    nums = np.bincount(cluster, weights=numerator)
    dens = np.bincount(cluster, weights=denominator)
    draw = np.random.default_rng(seed).integers(0, len(nums), size=(resamples, len(nums)))
    means = nums[draw].sum(axis=1) / dens[draw].sum(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(numerator.sum() / denominator.sum()), float(low), float(high)


def loss_difference(
    mine: pd.DataFrame, other: pd.DataFrame, spec: M6Backtest
) -> dict[str, Any] | None:
    """Mean weighted loss of ``mine`` minus ``other`` (their rows, same block and players) with
    its person-cluster bootstrap CI; None without rows."""
    a, b = player_losses(mine), player_losses(other)
    if not a.index.equals(b.index):
        raise ValueError("the models' rows are not aligned")
    if a.empty:
        return None
    weight = a["weight"].to_numpy(dtype=np.float64)
    diff = (a["loss"] - b["loss"]).to_numpy(dtype=np.float64)
    people = a.index.get_level_values("person_id")
    cluster = pd.factorize(people)[0].astype(np.int64)
    mean, low, high = cluster_bootstrap_ci(
        weight * diff, weight, cluster, spec.bootstrap_resamples, spec.bootstrap_seed
    )
    return {
        "mean": _round(mean),
        "ci95": [_round(low), _round(high)],
        "clusters": int(cluster.max()) + 1,
        "resamples": spec.bootstrap_resamples,
        "seed": spec.bootstrap_seed,
    }


# --- The run --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Scored:
    """A block's identity and its rows per model."""

    split: str
    season: int
    checkpoint: float
    rows: dict[str, pd.DataFrame]


def _stacked(scored: Sequence[_Scored], models: Sequence[str]) -> pd.DataFrame:
    parts = [
        s.rows[m].assign(split=s.split, season=s.season, checkpoint=s.checkpoint, model=m)
        for s in scored
        for m in models
        if m in s.rows
    ]
    return pd.concat(parts, ignore_index=True)


def _block_metrics(
    rows: pd.DataFrame, split: str, checkpoint: float, models: Sequence[str], stats: Sequence[str]
) -> dict[str, Any]:
    pool = rows[(rows["split"] == split) & (rows["checkpoint"] == checkpoint)]
    return {m: _rounded(model_scores(pool[pool["model"] == m], stats)) for m in models}


def _data_hash(inputs: M6Inputs) -> str:
    translations = pd.DataFrame(
        [
            (season, stat, t.delta[stat], t.c[stat])
            for season, t in sorted(inputs.translations.items())
            for stat in sorted(t.delta)
        ],
        columns=["season", "stat", "delta", "c"],
    )
    frames = [
        f for comp in sorted(inputs.games) for f in (inputs.games[comp], inputs.player_games[comp])
    ]
    return _data_sha256(*frames, inputs.xwalk, inputs.ages, inputs.brapm, translations)


def run_m6_backtest(
    inputs: M6Inputs,
    *,
    spec: M6Backtest,
    competition: str,
    tuning_only: bool = False,
    score_test: bool = False,
    fixed: Mapping[str, Any] | None = None,
    rounds: Callable[[str, int], int] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """The M6 report and the per-player rows (module docstring). ``tuning_only``: only the tuning
    seasons are scored, so no validation or test number exists anywhere (the verdict must be
    committed before validation is scored). ``score_test``: also the test seasons (only after
    the validation commit). ``fixed``: ``{"variant", "half_life"}``, the cell to score instead of
    choosing one (the GBL scores the EuroLeague verdict; the gate is then not binding)."""
    split_of = scored_splits(spec, tuning_only, score_test)
    grid = cells(spec)
    if fixed is not None and (fixed["variant"], float(fixed["half_life"])) not in grid:
        raise ValueError(f"the fixed cell {dict(fixed)!r} is not on the grid {grid}")
    world = prepare(inputs)
    sds = loss_sds(world, spec, competition)
    tuning_seasons = [s for s, split in split_of.items() if split == "tuning"]

    # Stage A: every cell and baseline on the tuning seasons' next-season targets.
    grid_scored: list[_Scored] = []
    for season in tuning_seasons:
        ti = target_inputs(inputs, competition, season, 0.0, spec=spec, rounds=rounds, world=world)
        predictions = {cell_key(v, h): project_cell(ti, v, h) for v, h in grid}
        predictions |= baseline_predictions(ti)
        grid_scored.append(_Scored("tuning", season, 0.0, score_block(ti, predictions, sds)))
    stats = tuple(s for s in (*BOX_STATS, *IMPACT_STATS) if s in sds)
    grid_rows = _stacked(grid_scored, [cell_key(v, h) for v, h in grid] + list(BASELINES))
    grid_metrics = {
        m: model_scores(grid_rows[grid_rows["model"] == m], stats)
        for m in [cell_key(v, h) for v, h in grid] + list(BASELINES)
    }
    table = [(v, h, grid_metrics[cell_key(v, h)]["loss"]) for v, h in grid]
    if fixed is None:
        variant, half_life, _ = table[choose_cell(table, spec.tie_tolerance)]
    else:
        variant, half_life = fixed["variant"], float(fixed["half_life"])
    chosen = cell_key(variant, half_life)
    models = [chosen, *BASELINES]

    # Stage B: the chosen cell and the baselines on everything else.
    scored = [
        _Scored(s.split, s.season, s.checkpoint, {m: s.rows[m] for m in models})
        for s in grid_scored
    ]
    for season, split in split_of.items():
        for checkpoint in (*((0.0,) if split != "tuning" else ()), *spec.checkpoints):
            ti = target_inputs(
                inputs, competition, season, checkpoint, spec=spec, rounds=rounds, world=world
            )
            predictions = {chosen: project_cell(ti, variant, half_life)}
            predictions |= baseline_predictions(ti)
            scored.append(_Scored(split, season, checkpoint, score_block(ti, predictions, sds)))
    rows = _stacked(scored, models)
    report = _report(
        inputs=inputs,
        spec=spec,
        competition=competition,
        split_of=split_of,
        sds=sds,
        stats=stats,
        grid=grid,
        grid_metrics=grid_metrics,
        rows=rows,
        variant=variant,
        half_life=half_life,
        tuning_only=tuning_only,
        score_test=score_test,
        fixed=fixed,
    )
    return report, _players_frame(rows)


def _players_frame(rows: pd.DataFrame) -> pd.DataFrame:
    order = {s: i for i, s in enumerate(SPLITS)}
    frame = rows.assign(split_order=rows["split"].map(order)).sort_values(
        ["split_order", "season", "checkpoint", "model", "person_id", "stat"], kind="stable"
    )
    frame = frame[list(PLAYER_COLUMNS)].round(6).reset_index(drop=True)
    return validated(frame, PLAYERS_SCHEMA)


def _split_block(
    *,
    rows: pd.DataFrame,
    split: str,
    seasons: Sequence[int],
    models: Sequence[str],
    stats: Sequence[str],
    checkpoints: Sequence[float],
) -> dict[str, Any]:
    return {
        "seasons": list(seasons),
        "next_season": _block_metrics(rows, split, 0.0, models, stats),
        "rest_of_season": {
            f"{f:g}": _block_metrics(rows, split, f, models, stats) for f in sorted(checkpoints)
        },
    }


def _report(
    *,
    inputs: M6Inputs,
    spec: M6Backtest,
    competition: str,
    split_of: Mapping[int, str],
    sds: Mapping[str, float],
    stats: Sequence[str],
    grid: Sequence[tuple[str, float]],
    grid_metrics: Mapping[str, dict[str, Any]],
    rows: pd.DataFrame,
    variant: str,
    half_life: float,
    tuning_only: bool,
    score_test: bool,
    fixed: Mapping[str, Any] | None,
) -> dict[str, Any]:
    chosen = cell_key(variant, half_life)
    models = [chosen, *BASELINES]
    splits = {name: [s for s, n in split_of.items() if n == name] for name in SPLITS}
    splits = {name: seasons for name, seasons in splits.items() if seasons}
    half_lives = sorted(spec.half_lives)
    nxt = rows[rows["checkpoint"] == 0.0]
    losses = {m: _rounded(v) for m, v in grid_metrics.items()}
    report: dict[str, Any] = {
        "model": "m6",
        "model_version": model_version(variant, half_life),
        "competition": competition,
        "splits": splits,
        "stats": list(stats),
        "checkpoints": list(spec.checkpoints),
        "min_poss": spec.min_poss,
        "interval": spec.interval,
        "sd": {s: _round(v) for s, v in sds.items()},
        "grid": {
            "objective": "next-season projection loss on the tuning seasons",
            "variants": list(VARIANTS),
            "half_lives": half_lives,
            "tie_tolerance": spec.tie_tolerance,
        },
        "tuning": {
            "seasons": splits["tuning"],
            "next_season": losses,
            "rest_of_season": {
                f"{f:g}": _block_metrics(rows, "tuning", f, models, stats)
                for f in sorted(spec.checkpoints)
            },
        },
        "chosen": {
            "variant": variant,
            "half_life": half_life,
            "on_edge": half_life in (half_lives[0], half_lives[-1]),
            **({"fixed": "the EuroLeague verdict (L-a), not chosen here"} if fixed else {}),
            "tuning_loss": losses[chosen]["loss"],
        },
        "data_sha256": _data_hash(inputs),
        "tuning_only": tuning_only,
        "validation_scored": not tuning_only,
        "test_scored": score_test and not tuning_only,
    }
    for name in ("validation", "test"):
        if name in splits:
            report[name] = _split_block(
                rows=rows,
                split=name,
                seasons=splits[name],
                models=models,
                stats=stats,
                checkpoints=spec.checkpoints,
            )
    report["movers"] = {
        name: {m: _mover_slice(nxt[(nxt["split"] == name) & (nxt["model"] == m)]) for m in models}
        for name in splits
    }
    report["differences"] = {
        name: {
            f"vs_{b}": loss_difference(
                nxt[(nxt["split"] == name) & (nxt["model"] == chosen)],
                nxt[(nxt["split"] == name) & (nxt["model"] == b)],
                spec,
            )
            for b in BASELINES
        }
        for name in splits
    }
    if not tuning_only:
        report["gate"] = _gate(report, nxt, chosen, spec, stats, gated=fixed is None)
    return report


def _mover_slice(rows: pd.DataFrame) -> dict[str, Any]:
    movers = player_losses(rows)
    movers = movers[movers["mover"].astype(bool)]
    return {"n": len(movers), "loss": _round(_weighted_mean(movers["loss"], movers["weight"]))}


def _gate(
    report: Mapping[str, Any],
    nxt: pd.DataFrame,
    chosen: str,
    spec: M6Backtest,
    stats: Sequence[str],
    *,
    gated: bool,
) -> dict[str, Any]:
    """The gate from the validation rows and the pooled tuning + validation coverage of the
    chosen cell."""
    validation = nxt[nxt["split"] == "validation"]

    def loss_of(model: str) -> float | None:
        loss: float | None = model_scores(validation[validation["model"] == model], stats)["loss"]
        return loss

    pooled = nxt[nxt["split"].isin(["tuning", "validation"]) & (nxt["model"] == chosen)]
    coverage = model_scores(pooled, stats)["coverage"] or {}
    return gate_block(
        report["chosen"]["variant"],
        report["chosen"]["half_life"],
        loss_chosen=loss_of(chosen),
        loss_marcel=loss_of("marcel"),
        loss_same_as_last=loss_of("same_as_last"),
        coverage=coverage,
        band=spec.coverage_band,
        differences={
            f"loss_vs_{b}": report["differences"]["validation"][f"vs_{b}"] for b in GATE_BASELINES
        },
        gated=gated,
    )


def format_m6_table(report: dict[str, Any]) -> str:
    def cell(value: float | None, width: int, digits: int) -> str:
        return f"{'-' if value is None else f'{value:.{digits}f}':>{width}}"

    chosen = report["chosen"]
    lines = [
        f"chosen: {chosen['variant']} half-life {chosen['half_life']:g} "
        f"(tuning loss {chosen['tuning_loss']}, on edge: {chosen['on_edge']})"
    ]
    header = f"{'split':<11}{'target':<9}{'model':<18}{'n':>6}{'loss':>9}"
    lines += [header, "-" * len(header)]

    def block(split: str, label: str, models: dict[str, Any]) -> None:
        for name, m in models.items():
            lines.append(f"{split:<11}{label:<9}{name:<18}{m['n']:>6}{cell(m['loss'], 9, 4)}")

    block("tuning", "next", report["tuning"]["next_season"])
    for split in ("validation", "test"):
        if split in report:
            block(split, "next", report[split]["next_season"])
    gate = report.get("gate")
    if gate is not None:
        verdict = {True: "PASS", False: "FAIL", None: "n/a"}[gate["passed"]]
        lines.append(
            f"gate (validation loss, {gate['variant']} vs marcel {gate['loss_marcel']} and "
            f"same_as_last {gate['loss_same_as_last']}; coverage in band: "
            f"{gate['coverage_ok']}) -> {verdict}"
        )
    return "\n".join(lines)
