"""M7 backtest (week 14-16 K4): replay past seasons at checkpoints, simulate the rest of each,
score the probabilities against the real final tables.

At each checkpoint (K-e) of a scored season the harness takes M1's rating posterior (K-b,
``team_eff.rating_posteriors``) and pace point on every game before the cutoff, plays the
season's remaining regular-season fixtures ``n_sims`` times with ``sim.season.simulate`` and
reads the final-rank distribution of every team. Everything is a pure function of the inputs and
the seeds: ``seed`` + the checkpoint's index, the same for every model at a checkpoint.

Definitions (orchestrator decisions on top of reports/m7_progress.md):

- **Checkpoint (D1)**: k = floor(f * R) completed rounds, R = the format's
  ``regular_season_rounds``; the cutoff is the first tip-off of regular-season round k + 1 of
  that season. The state is every regular-season game of the season that tipped off before the
  cutoff; the remaining fixtures are the rest of the season's regular-season rows (columns
  home, away, neutral, tipoff_utc; sorted by tip-off then home then away), a repeated ordered
  pair keeping its first row (D2). The checkpoint index, which fixes the seed, is the position in
  the list of every (season, fraction) pair of the run sorted by season then fraction, so a
  tuning-only run and a full run draw the same numbers for the tuning seasons.
- **Strengths**: a draw is theta ~ N(m, c * S) with m and S the posterior restricted to
  [mu, h, off(teams), def(teams)] (teams = the season's regular-season teams, sorted; each must
  have both ratings). One ``rng.standard_normal((n_sims, k)) @ L.T`` per checkpoint, L the
  Cholesky factor of c * S, then off = mu + off_t, def = def_t, home = h. With c = 0 (or a zero
  covariance) the normals are still drawn but every simulation gets exactly m.
- **Pace**: M1's pace point [mu_p, pace * n] as ``PaceModel(mu_p, pace of the season's teams)``.
- **Noise**: M1's committed ``tuned.margin`` (Student-t df, or Normal). A ``*_pace`` margin
  variant (the GBL's) uses its constant scale * sqrt(mean pace of the remaining fixtures /
  ref_pace) at that checkpoint (D8).
- **Variants**: ``sim_full`` (sampled strengths, M1's noise), ``sim_net`` (noise scale shrunk
  so that strength variance + noise variance = M1's predictive variance V, floored at
  ``noise_floor`` of M1's scale; the margin's strength variance per remaining game is
  (pace/100)^2 * a'Sa with a the coefficients of ORtg_H - ORtg_A on [mu, h, off, def], averaged
  over the remaining games), ``sim_inflate_c`` (``sim_net`` with the strength covariance times c,
  its noise scale computed from the uninflated S, D9). Baselines, never chosen: ``point_sim``
  (c = 0, M1's full noise), ``elo_sim`` (comparison Elo at the cutoff, D4) and ``standings_now``
  (the current table ranked by win percentage, ties by ``standings.rank``; 0/1 probabilities).
- **Targets**: the final table is ``standings.rank`` on every regular-season game of the season
  (regulation scores where given) with the format's deductions; the cut lines are
  ``formats.cut_lines``; the champion is the winner of the season's last played ``FF`` game
  (EuroLeague) or ``PO`` game (GBL), or unknown.
- **Metrics** (pooled per split and per checkpoint fraction, from the rounded CSV rows): Brier of
  P(direct cut), RPS of the final rank = sum_{k<n} (P(place <= k) - 1[place <= k])^2 / (n - 1),
  log loss of the cut probability clipped to [1e-4, 1 - 1e-4], Brier of P(top 10) in seasons with a
  play-in, ten-bin reliability and the Spiegelhalter z of the cut probability, title Brier per
  season = sum over teams (P(title) - 1[champion])^2 averaged over the season's checkpoints.
  Brier differences carry a team-cluster bootstrap (clusters = (season, team), all of a
  team-season's checkpoints together) percentile 95% CI.
- **Choice (K-i)**: on the tuning seasons only, the lowest mean RPS among the variants; a difference
  below ``tie_tolerance`` goes to the simpler one (``sim_full`` < ``sim_net`` < ``sim_inflate_c``
  by increasing c). ``fixed`` skips the choice (the GBL scores the EuroLeague verdict).
- **Gate (K-h)**: on the validation seasons the chosen variant's mean Brier is below
  ``point_sim``'s and ``standings_now``'s, and the Spiegelhalter |z| pooled over tuning and
  validation is below 1.96.
"""

import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pandera.pandas as pa

from eurohoops.config import M7Backtest
from eurohoops.eval.m3_backtest import _data_sha256
from eurohoops.eval.metrics import reliability
from eurohoops.logs import TIME_FORMAT
from eurohoops.models.elo import EloParams, FloatArray, prepare, season_ratings
from eurohoops.models.team_eff import (
    DecayParams,
    IntArray,
    MarginModel,
    RatingPosterior,
    pace_points,
    prepare_history,
    rating_posteriors,
)
from eurohoops.parse.schemas import validated
from eurohoops.sim.formats import cut_lines, season_format
from eurohoops.sim.played import season_results
from eurohoops.sim.season import NoiseModel, PaceModel, SimOutput, StrengthSampler, simulate
from eurohoops.standings import Format, Result, rank

SPLITS = ("tuning", "validation", "test")
BASELINES = ("point_sim", "elo_sim", "standings_now")
CHAMPION_PHASE = {"euroleague": "FF", "gbl": "PO"}
LOG_CLIP = 1e-4
Z_LIMIT = 1.96
BINS = 10
GATE_RULE = (
    "validation mean Brier of the direct-playoff cut: the chosen variant below point_sim's and "
    "standings_now's (point rule), and the Spiegelhalter |z| pooled over tuning and validation "
    "below 1.96; team-cluster bootstrap 95% CIs reported"
)
ROW_COLUMNS = (
    "season",
    "split",
    "checkpoint",
    "round",
    "team",
    "model",
    "p_direct",
    "p_play_in",
    "p_top10",
    "p_title",
    "exp_rank",
    "rps",
    "final_place",
    "made_direct",
    "made_top10",
    "champion",
)
TEAMS_SCHEMA = pa.DataFrameSchema(
    {
        "season": pa.Column("int64"),
        "split": pa.Column(str, pa.Check.isin(SPLITS)),
        "checkpoint": pa.Column("float64", pa.Check.in_range(0.0, 1.0, include_min=False)),
        "round": pa.Column("int64", pa.Check.gt(0)),
        "team": pa.Column(str),
        "model": pa.Column(str),
        "p_direct": pa.Column("float64", pa.Check.in_range(0.0, 1.0)),
        "p_play_in": pa.Column("float64", pa.Check.in_range(0.0, 1.0), nullable=True),
        "p_top10": pa.Column("float64", pa.Check.in_range(0.0, 1.0), nullable=True),
        "p_title": pa.Column("float64", pa.Check.in_range(0.0, 1.0)),
        "exp_rank": pa.Column("float64", pa.Check.ge(1.0)),
        "rps": pa.Column("float64", pa.Check.in_range(0.0, 1.0)),
        "final_place": pa.Column("int64", pa.Check.ge(1)),
        "made_direct": pa.Column("int64", pa.Check.isin([0, 1])),
        "made_top10": pa.Column("Int64", pa.Check.isin([0, 1]), nullable=True),
        "champion": pa.Column("Int64", pa.Check.isin([0, 1]), nullable=True),
    },
    unique=["season", "checkpoint", "model", "team"],
    strict=True,
    ordered=True,
)


def _round(value: float | None) -> float | None:
    return None if value is None or not math.isfinite(value) else round(float(value), 6)


# --- Inputs, variants and the choice ------------------------------------------------------------


@dataclass(frozen=True)
class M7Inputs:
    """Everything a run reads. ``games`` and ``team_games`` are the competition's marts over every
    season up to the last scored one; ``regulation`` maps ``game_id`` to the regulation score of
    an overtime game (empty for the GBL); ``tuned_m1`` is ``reports/backtest_m1*.json``'s
    ``tuned`` block and ``elo`` its ``comparison_elo`` block."""

    competition: str
    games: pd.DataFrame
    team_games: pd.DataFrame
    regulation: Mapping[str, tuple[int, int]]
    tuned_m1: dict[str, Any]
    elo: dict[str, Any]


@dataclass(frozen=True)
class Variant:
    """A simulated model: the strength covariance is multiplied by ``spread`` (0: M1's point
    strengths); ``net`` shrinks the noise scale by the strength variance."""

    key: str
    spread: float
    net: bool


def variants(spec: M7Backtest) -> tuple[Variant, ...]:
    """The choosable variants, simplest first (the tie-break order)."""
    return (
        Variant("sim_full", 1.0, False),
        Variant("sim_net", 1.0, True),
        *(Variant(f"sim_inflate_{c:g}", c, True) for c in spec.inflate),
    )


def model_keys(spec: M7Backtest) -> tuple[str, ...]:
    """Every model of a run: the variants, then the baselines."""
    return (*(v.key for v in variants(spec)), *BASELINES)


def model_version(key: str) -> str:
    return f"m7-{key}-v1"


def choose_variant(table: Sequence[tuple[str, float]], tolerance: float) -> int:
    """Index of the winner of a table of (variant key, tuning mean RPS) in simplicity order: the
    lowest RPS, or, among every variant within ``tolerance`` of it (difference < tolerance), the
    simplest."""
    finite = [(i, rps) for i, (_, rps) in enumerate(table) if math.isfinite(rps)]
    if not finite:
        raise ValueError("no variant has a finite tuning RPS")
    best = min(rps for _, rps in finite)
    return min(i for i, rps in finite if rps == best or rps - best < tolerance)


def gate_block(
    variant: str,
    *,
    brier_chosen: float | None,
    brier_point_sim: float | None,
    brier_standings_now: float | None,
    z_pooled: float | None,
    differences: dict[str, Any],
) -> dict[str, Any]:
    """The gate (K-h, point rule) from unrounded values; ``None`` leaves it undecided."""
    passed = None
    if (
        brier_chosen is not None
        and brier_point_sim is not None
        and brier_standings_now is not None
        and z_pooled is not None
    ):
        passed = bool(
            brier_chosen < brier_point_sim
            and brier_chosen < brier_standings_now
            and abs(z_pooled) < Z_LIMIT
        )
    return {
        "rule": GATE_RULE,
        "variant": variant,
        "brier_chosen": _round(brier_chosen),
        "brier_point_sim": _round(brier_point_sim),
        "brier_standings_now": _round(brier_standings_now),
        "spiegelhalter_z_pooled": _round(z_pooled),
        "passed": passed,
        **differences,
    }


# --- Checkpoints --------------------------------------------------------------------------------


@dataclass(frozen=True)
class Checkpoint:
    index: int
    season: int
    split: str
    fraction: float
    round: int  # completed regular-season rounds
    cutoff: pd.Timestamp
    seed: int


def scored_splits(spec: M7Backtest, tuning_only: bool, score_test: bool) -> dict[int, str]:
    """Season -> split name for the seasons a run simulates, ordered by season: tuning always,
    validation unless ``tuning_only``, test when ``score_test`` as well."""
    names = ["tuning"] if tuning_only else ["tuning", "validation"]
    if score_test and not tuning_only:
        names.append("test")
    return dict(sorted({season: name for name in names for season in getattr(spec, name)}.items()))


def checkpoint_round(fmt: Format, fraction: float) -> int:
    """k = floor(f * R) completed regular-season rounds (D1)."""
    return math.floor(fraction * fmt.regular_season_rounds)


def checkpoint_cutoff(season_games: pd.DataFrame, rounds_done: int) -> pd.Timestamp:
    """The first tip-off of regular-season round ``rounds_done`` + 1 (D1)."""
    following = season_games[season_games["round"] == rounds_done + 1]
    if following.empty:
        raise ValueError(f"no regular-season round {rounds_done + 1} to cut the season at")
    cutoff: pd.Timestamp = following["tipoff_utc"].min()
    return cutoff


def remaining_fixtures(season_games: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """The season's rows tipping off at or after ``cutoff``, sorted by tip-off, home, away; a
    repeated ordered pair keeps its first row (D2)."""
    later = season_games[season_games["tipoff_utc"] >= cutoff]
    later = later.sort_values(["tipoff_utc", "home", "away"], kind="stable")
    later = later.drop_duplicates(["home", "away"], keep="first")
    return later[["home", "away", "neutral", "tipoff_utc"]].reset_index(drop=True)


@dataclass(frozen=True)
class _Season:
    """What a season's checkpoints share: its format, teams, regular-season rows and the truth."""

    fmt: Format
    teams: tuple[str, ...]
    regular: pd.DataFrame
    place: dict[str, int]
    champion: str | None


def champion_of(season_games: pd.DataFrame, phase: str) -> str | None:
    """Winner of the season's last played game of ``phase``; None without one."""
    final = season_games[(season_games["phase"] == phase) & season_games["played"]]
    if final.empty:
        return None
    last = final.sort_values(["tipoff_utc", "game_id"]).iloc[-1]
    return str(last["home"] if last["home_score"] > last["away_score"] else last["away"])


def _season(inputs: M7Inputs, fmt: Format, season: int) -> _Season:
    games = inputs.games[inputs.games["season"] == season]
    regular = games[games["phase"] == "RS"]
    if regular.empty:
        raise ValueError(f"no regular-season games in season {season}")
    final = rank(season_results(regular, inputs.regulation), fmt.deducted_wins())
    return _Season(
        fmt=fmt,
        teams=tuple(sorted({*regular["home"], *regular["away"]})),
        regular=regular,
        place={team: place for place, team in enumerate(final, start=1)},
        champion=champion_of(games, CHAMPION_PHASE[inputs.competition]),
    )


def plan_checkpoints(
    seasons: Mapping[int, _Season], split_of: Mapping[int, str], spec: M7Backtest
) -> list[Checkpoint]:
    """Every (season, fraction) of the run sorted by season then fraction; the index fixes the
    seed (``spec.seed`` + index)."""
    plan: list[Checkpoint] = []
    for season in sorted(split_of):
        info = seasons[season]
        for fraction in sorted(spec.checkpoints):
            done = checkpoint_round(info.fmt, fraction)
            plan.append(
                Checkpoint(
                    index=len(plan),
                    season=season,
                    split=split_of[season],
                    fraction=fraction,
                    round=done,
                    cutoff=checkpoint_cutoff(info.regular, done),
                    seed=spec.seed + len(plan),
                )
            )
    return plan


# --- The strength, pace and noise models ----------------------------------------------------------


@dataclass(frozen=True)
class Strengths:
    """M1's posterior restricted to [mu, h, off(teams), def(teams)]."""

    mean: FloatArray
    cov: FloatArray


def restrict_posterior(posterior: RatingPosterior, teams: Sequence[str]) -> Strengths:
    """The posterior over [mu, h, off(teams), def(teams)]; a team without ratings raises."""
    position = {label: i for i, label in enumerate(posterior.labels)}
    labels = ["mu", "h", *(f"off:{t}" for t in teams), *(f"def:{t}" for t in teams)]
    missing = [label for label in labels if label not in position]
    if missing:
        raise ValueError(f"no rating at the cutoff for {missing}")
    if not math.isfinite(posterior.sigma2):
        raise ValueError("the posterior variance is undefined at the cutoff")
    pick = np.array([position[label] for label in labels], dtype=np.int64)
    return Strengths(posterior.mean[pick], posterior.cov[np.ix_(pick, pick)])


def gaussian_sampler(strengths: Strengths, spread: float) -> StrengthSampler:
    """theta ~ N(m, spread * S), one draw per simulation. The normals are always drawn, so the
    random stream does not depend on ``spread``; without spread (0, or a zero covariance) every
    simulation gets exactly m."""
    mean, k = strengths.mean, len(strengths.mean)
    n = (k - 2) // 2
    factor = (
        np.linalg.cholesky(spread * strengths.cov)
        if spread != 0.0 and strengths.cov.any()
        else None
    )

    def sample(n_sims: int, rng: np.random.Generator) -> tuple[FloatArray, FloatArray, FloatArray]:
        z = rng.standard_normal((n_sims, k))
        theta = np.broadcast_to(mean, (n_sims, k)) if factor is None else mean + z @ factor.T
        return theta[:, 2 : 2 + n] + theta[:, :1], theta[:, 2 + n :], theta[:, 1]

    return sample


def fixed_sampler(off: FloatArray, def_: FloatArray, home: float) -> StrengthSampler:
    """The same strengths in every simulation (no draw)."""

    def sample(n_sims: int, rng: np.random.Generator) -> tuple[FloatArray, FloatArray, FloatArray]:
        return (
            np.broadcast_to(off, (n_sims, len(off))),
            np.broadcast_to(def_, (n_sims, len(def_))),
            np.full(n_sims, home),
        )

    return sample


def margin_noise(model: MarginModel, mean_pace: float) -> NoiseModel:
    """M1's margin noise; a ``*_pace`` variant's scale is taken at ``mean_pace`` (D8)."""
    return NoiseModel(float(model.scales(np.array([mean_pace]))[0]), model.df)


def noise_variance(noise: NoiseModel) -> float:
    """Variance of ``scale * e`` for e standard Normal, or Student-t(df) with df > 2."""
    if noise.df is None:
        return noise.scale**2
    if noise.df <= 2.0:
        raise ValueError("a Student-t noise needs df > 2 for a finite variance")
    return noise.scale**2 * noise.df / (noise.df - 2.0)


def net_noise(noise: NoiseModel, strength_var: float, floor: float) -> NoiseModel:
    """The noise whose variance plus ``strength_var`` is M1's predictive variance, at least
    ``floor`` of M1's scale (K-f)."""
    spare = max(noise_variance(noise) - strength_var, 0.0)
    shape = 1.0 if noise.df is None else (noise.df - 2.0) / noise.df
    return NoiseModel(max(math.sqrt(spare * shape), floor * noise.scale), noise.df)


def strength_variance(
    cov: FloatArray,
    home_idx: IntArray,
    away_idx: IntArray,
    neutral: np.ndarray[Any, np.dtype[np.bool_]],
    game_pace: FloatArray,
) -> float:
    """Mean over the games of the variance of the margin's mean, (pace/100)^2 * a'Sa, with a the
    coefficients of ORtg_H - ORtg_A on [mu, h, off, def]: 2*hf on h, +1 off_H, -1 def_A,
    -1 off_A, +1 def_H (``cov`` over [mu, h, off * n, def * n])."""
    n = (len(cov) - 2) // 2
    games = np.arange(len(home_idx))
    a = np.zeros((len(home_idx), len(cov)))
    a[:, 1] = 2.0 * (~neutral)
    a[games, 2 + home_idx] += 1.0
    a[games, 2 + n + away_idx] -= 1.0
    a[games, 2 + away_idx] -= 1.0
    a[games, 2 + n + home_idx] += 1.0
    variance = (game_pace / 100.0) ** 2 * np.einsum("gk,kl,gl->g", a, cov, a)
    return float(variance.mean())


def elo_strength_scale(noise: NoiseModel) -> float:
    """c = scale * ln10 / (1600 * f(0)), f the density of the standard noise at 0: with a margin
    of c * (rating difference), the engine's P(home win) has Elo's logistic slope at an even game
    (D4)."""
    if noise.df is None:
        density = 1.0 / math.sqrt(2.0 * math.pi)
    else:
        density = math.exp(
            math.lgamma((noise.df + 1.0) / 2.0) - math.lgamma(noise.df / 2.0)
        ) / math.sqrt(noise.df * math.pi)
    return noise.scale * math.log(10.0) / (1600.0 * density)


# --- Forecasts of one checkpoint ------------------------------------------------------------------


@dataclass(frozen=True)
class Forecast:
    """A model's probabilities for the teams of a season (``teams`` order)."""

    rank_probs: FloatArray  # (teams, places): [i, j] = P(team i finishes j + 1)
    p_direct: FloatArray
    p_play_in: FloatArray
    p_top10: FloatArray
    p_title: FloatArray


def sim_forecast(out: SimOutput) -> Forecast:
    return Forecast(
        out.rank_counts / out.n_sims, out.p_direct, out.p_play_in, out.p_top10, out.p_title
    )


def standings_order(
    teams: Sequence[str], played: Sequence[Result], deducted: Mapping[str, int]
) -> list[str]:
    """Teams by current win percentage (deducted wins off the numerator); a tie keeps
    ``standings.rank``'s order on the played results."""
    ranked = rank(played, deducted)
    base = [*ranked, *sorted(set(teams) - set(ranked))]
    wins = Counter(r.winner for r in played)
    games = Counter(t for r in played for t in (r.home, r.away))

    def percentage(team: str) -> float:
        return (wins[team] - deducted.get(team, 0)) / games[team] if games[team] else 0.0

    return sorted(base, key=lambda team: -percentage(team))


def order_forecast(order: Sequence[str], teams: Sequence[str], fmt: Format) -> Forecast:
    """The point-mass forecast of a final order: probabilities 0 or 1."""
    place = {team: i for i, team in enumerate(order)}
    rank_probs = np.zeros((len(teams), len(teams)))
    for i, team in enumerate(teams):
        rank_probs[i, place[team]] = 1.0
    lines = cut_lines(fmt)

    def cut(places: tuple[int, ...]) -> FloatArray:
        share: FloatArray = rank_probs[:, [p - 1 for p in places]].sum(axis=1)
        return share

    return Forecast(
        rank_probs,
        cut(lines["direct"]),
        cut(lines.get("play_in", ())),
        cut(tuple(range(1, min(10, len(teams)) + 1))),
        rank_probs[:, 0],
    )


def _forecasts(
    *,
    spec: M7Backtest,
    teams: tuple[str, ...],
    fmt: Format,
    played: Sequence[Result],
    remaining: pd.DataFrame,
    strengths: Strengths,
    pace: PaceModel,
    margin: MarginModel,
    elo_ratings: Mapping[str, float],
    elo_hca: float,
    seed: int,
) -> dict[str, Forecast]:
    """Every model's forecast at one checkpoint, in ``model_keys`` order."""
    index = {team: i for i, team in enumerate(teams)}
    home_idx = np.asarray(remaining["home"].map(index), dtype=np.int64)
    away_idx = np.asarray(remaining["away"].map(index), dtype=np.int64)
    game_pace = pace.mu + pace.team[home_idx] + pace.team[away_idx]
    noise = margin_noise(margin, float(game_pace.mean()))
    net = net_noise(
        noise,
        strength_variance(
            strengths.cov,
            home_idx,
            away_idx,
            remaining["neutral"].to_numpy(dtype=bool),
            game_pace,
        ),
        spec.noise_floor,
    )

    def play(sampler: StrengthSampler, pace_: PaceModel, noise_: NoiseModel) -> Forecast:
        return sim_forecast(
            simulate(
                teams=teams,
                played=played,
                remaining=remaining,
                sampler=sampler,
                pace=pace_,
                noise=noise_,
                fmt=fmt,
                n_sims=spec.n_sims,
                seed=seed,
            )
        )

    out = {
        v.key: play(gaussian_sampler(strengths, v.spread), pace, net if v.net else noise)
        for v in variants(spec)
    }
    out["point_sim"] = play(gaussian_sampler(strengths, 0.0), pace, noise)

    n = len(teams)
    mu, off_t, def_t = strengths.mean[0], strengths.mean[2 : 2 + n], strengths.mean[2 + n :]
    total = float(
        np.mean(
            game_pace
            * (2.0 * mu + off_t[home_idx] - def_t[away_idx] + off_t[away_idx] - def_t[home_idx])
            / 100.0
        )
    )
    slope = elo_strength_scale(noise)
    rating = np.array([elo_ratings[team] for team in teams])
    elo_sampler = fixed_sampler(
        total / 2.0 + slope * rating / 2.0, slope * rating / 2.0, slope * elo_hca / 2.0
    )
    out["elo_sim"] = play(elo_sampler, PaceModel(100.0, np.zeros(n)), noise)
    order = standings_order(teams, played, fmt.deducted_wins())
    out["standings_now"] = order_forecast(order, teams, fmt)
    return out


def _model_rows(cp: Checkpoint, model: str, forecast: Forecast, info: _Season) -> pd.DataFrame:
    """The CSV rows of one model at one checkpoint, one per team."""
    teams, fmt = info.teams, info.fmt
    n = len(teams)
    lines = cut_lines(fmt)
    place = np.array([info.place[team] for team in teams], dtype=np.int64)
    has_top10 = "top10" in lines
    nan = np.full(n, np.nan)
    unknown = pd.array([None] * n, dtype="Int64")
    return pd.DataFrame(
        {
            "season": cp.season,
            "split": cp.split,
            "checkpoint": cp.fraction,
            "round": cp.round,
            "team": list(teams),
            "model": model,
            "p_direct": forecast.p_direct,
            "p_play_in": forecast.p_play_in if has_top10 else nan,
            "p_top10": forecast.p_top10 if has_top10 else nan,
            "p_title": forecast.p_title,
            "exp_rank": forecast.rank_probs @ np.arange(1, n + 1),
            "rps": [rps_of(forecast.rank_probs[i], int(place[i])) for i in range(n)],
            "final_place": place,
            "made_direct": np.isin(place, lines["direct"]).astype(np.int64),
            "made_top10": (
                pd.array((place <= max(lines["top10"])).astype(np.int64), dtype="Int64")
                if has_top10
                else unknown
            ),
            "champion": (
                unknown
                if info.champion is None
                else pd.array([int(t == info.champion) for t in teams], dtype="Int64")
            ),
        },
        columns=list(ROW_COLUMNS),
    )


# --- Metrics ---------------------------------------------------------------------------------


def rps_of(rank_probs: FloatArray, place: int) -> float:
    """Ranked probability score of one team's final-rank distribution: the mean over the
    n - 1 cumulative cut lines of (P(place <= k) - 1[final place <= k])^2."""
    n = len(rank_probs)
    cumulative = np.cumsum(rank_probs)[:-1]
    reached = place <= np.arange(1, n)
    return float(((cumulative - reached) ** 2).sum() / (n - 1))


def spiegelhalter_z(p: FloatArray, y: FloatArray) -> float | None:
    """Spiegelhalter's z of probabilities ``p`` for outcomes ``y``:
    sum (y - p)(1 - 2p) / sqrt(sum (1 - 2p)^2 p (1 - p)); None when the variance is 0."""
    variance = float(np.sum((1.0 - 2.0 * p) ** 2 * p * (1.0 - p)))
    if variance <= 0.0:
        return None
    return float(np.sum((y - p) * (1.0 - 2.0 * p)) / math.sqrt(variance))


def _scores(rows: pd.DataFrame) -> dict[str, Any]:
    """Unrounded metrics of rows (one model): n, brier, rps, log_loss, brier_top10, z."""
    n = len(rows)
    if not n:
        return dict.fromkeys(("brier", "rps", "log_loss", "brier_top10", "z"), None) | {"n": 0}
    p = rows["p_direct"].to_numpy(dtype=np.float64)
    y = rows["made_direct"].to_numpy(dtype=np.float64)
    clipped = np.clip(p, LOG_CLIP, 1.0 - LOG_CLIP)
    top = rows.dropna(subset=["p_top10"])
    return {
        "n": n,
        "brier": float(np.mean((p - y) ** 2)),
        "rps": float(rows["rps"].mean()),
        "log_loss": float(-np.mean(y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped))),
        "brier_top10": (
            float(np.mean((top["p_top10"] - top["made_top10"].astype(float)) ** 2))
            if len(top)
            else None
        ),
        "z": spiegelhalter_z(p, y),
    }


def _metrics(rows: pd.DataFrame) -> dict[str, Any]:
    s = _scores(rows)
    return {
        "n": s["n"],
        "brier": _round(s["brier"]),
        "rps": _round(s["rps"]),
        "log_loss": _round(s["log_loss"]),
        "brier_top10": _round(s["brier_top10"]),
        "spiegelhalter_z": _round(s["z"]),
    }


def _title_brier(rows: pd.DataFrame) -> float | None:
    """Per season the mean over its checkpoints of sum over teams (P(title) - 1[champion])^2, then
    the mean over the seasons whose champion is known."""
    known = rows.dropna(subset=["champion"])
    if known.empty:
        return None
    square = (known["p_title"] - known["champion"].astype(float)) ** 2
    per_checkpoint = square.groupby([known["season"], known["checkpoint"]]).sum()
    return float(per_checkpoint.groupby(level="season").mean().mean())


def cluster_bootstrap_ci(
    diff: FloatArray, cluster: IntArray, resamples: int, seed: int
) -> tuple[float, float, float]:
    """Mean of ``diff`` and the percentile 95% CI of its mean over resamples of whole clusters
    (cluster ids 0..C-1, drawn with replacement); NaN for an empty ``diff``."""
    if not len(diff):
        return math.nan, math.nan, math.nan
    sums = np.bincount(cluster, weights=diff)
    counts = np.bincount(cluster).astype(np.float64)
    draw = np.random.default_rng(seed).integers(0, len(sums), size=(resamples, len(sums)))
    means = sums[draw].sum(axis=1) / counts[draw].sum(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(diff.mean()), float(low), float(high)


def _brier_differences(
    data: pd.DataFrame, chosen: str, spec: M7Backtest
) -> dict[str, dict[str, Any] | None]:
    """Chosen minus each of point_sim, standings_now and elo_sim: mean per-row Brier difference
    of the direct cut with its team-cluster bootstrap CI."""
    mine = data[data["model"] == chosen].reset_index(drop=True)
    out: dict[str, dict[str, Any] | None] = {}
    for baseline in ("point_sim", "standings_now", "elo_sim"):
        other = data[data["model"] == baseline].reset_index(drop=True)
        key = ["season", "checkpoint", "team"]
        if not mine[key].equals(other[key]):
            raise ValueError("the models' rows are not aligned")
        if mine.empty:
            out[f"brier_vs_{baseline}"] = None
            continue
        y = mine["made_direct"].to_numpy(dtype=np.float64)
        diff = (mine["p_direct"].to_numpy() - y) ** 2 - (other["p_direct"].to_numpy() - y) ** 2
        cluster = pd.MultiIndex.from_frame(mine[["season", "team"]]).factorize()[0]
        mean, low, high = cluster_bootstrap_ci(
            diff, cluster.astype(np.int64), spec.bootstrap_resamples, spec.bootstrap_seed
        )
        out[f"brier_vs_{baseline}"] = {
            "mean": _round(mean),
            "ci95": [_round(low), _round(high)],
            "clusters": int(cluster.max()) + 1,
            "resamples": spec.bootstrap_resamples,
            "seed": spec.bootstrap_seed,
        }
    return out


# --- The run -------------------------------------------------------------------------------------


def _checkpoint_rows(
    cp: Checkpoint, info: _Season, forecasts: Mapping[str, Forecast]
) -> pd.DataFrame:
    return pd.concat(
        [_model_rows(cp, model, f, info) for model, f in forecasts.items()], ignore_index=True
    )


def run_m7_backtest(
    inputs: M7Inputs,
    *,
    spec: M7Backtest,
    formats: Callable[[str, int], Format] = season_format,
    tuning_only: bool = False,
    score_test: bool = False,
    fixed: str | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """The M7 report and the per-team-checkpoint rows (module docstring). ``tuning_only``: only
    the tuning seasons are simulated, so no validation or test number exists anywhere (the verdict
    must be committed before validation is scored). ``score_test``: also the test seasons (only
    after the validation commit). ``fixed``: score this variant key instead of choosing one (the
    GBL scores the EuroLeague verdict). ``formats`` supplies each season's ``Format``."""
    keys = [v.key for v in variants(spec)]
    if fixed is not None and fixed not in keys:
        raise ValueError(f"the fixed variant {fixed!r} is not one of {keys}")
    split_of = scored_splits(spec, tuning_only, score_test)
    seasons = {s: _season(inputs, formats(inputs.competition, s), s) for s in split_of}
    plan = plan_checkpoints(seasons, split_of, spec)

    games = inputs.games.sort_values("tipoff_utc", kind="stable").reset_index(drop=True)
    history = prepare_history(games, inputs.team_games)
    tuned = inputs.tuned_m1
    when = [(float(np.float64(cp.cutoff.as_unit("ns").value) / 1e9), cp.season) for cp in plan]
    posteriors = rating_posteriors(history, DecayParams(**tuned["rating"]), when)
    paces = pace_points(history, DecayParams(**tuned["pace"]), when)
    margin = MarginModel(**tuned["margin"])
    elo = EloParams(k=inputs.elo["k"], hca=inputs.elo["hca"], reversion=inputs.elo["reversion"])
    column = {team: i for i, team in enumerate(history.teams)}

    frames = []
    for cp, posterior, pace_point in zip(plan, posteriors, paces, strict=True):
        info = seasons[cp.season]
        regular = info.regular
        played = season_results(regular[regular["tipoff_utc"] < cp.cutoff], inputs.regulation)
        before = prepare(games[games["tipoff_utc"] < cp.cutoff])
        ratings = season_ratings(before, elo, cp.season)
        unrated = [team for team in info.teams if team not in ratings]
        if unrated:
            raise ValueError(f"no Elo rating at the cutoff for {unrated}")
        forecasts = _forecasts(
            spec=spec,
            teams=info.teams,
            fmt=info.fmt,
            played=played,
            remaining=remaining_fixtures(regular, cp.cutoff),
            strengths=restrict_posterior(posterior, info.teams),
            pace=PaceModel(
                float(pace_point[0]),
                np.array([pace_point[1 + column[team]] for team in info.teams]),
            ),
            margin=margin,
            elo_ratings={team: ratings[team][0] for team in info.teams},
            elo_hca=elo.hca,
            seed=cp.seed,
        )
        frames.append(_checkpoint_rows(cp, info, forecasts))
    rows = validated(pd.concat(frames, ignore_index=True).round(6), TEAMS_SCHEMA)
    report = _report(
        inputs=inputs,
        spec=spec,
        rows=rows,
        plan=plan,
        seasons=seasons,
        tuning_only=tuning_only,
        score_test=score_test,
        fixed=fixed,
    )
    return report, rows


def _report(
    inputs: M7Inputs,
    *,
    spec: M7Backtest,
    rows: pd.DataFrame,
    plan: Sequence[Checkpoint],
    seasons: Mapping[int, _Season],
    tuning_only: bool,
    score_test: bool,
    fixed: str | None,
) -> dict[str, Any]:
    keys = model_keys(spec)
    candidates = [v.key for v in variants(spec)]
    splits = [s for s in SPLITS if (rows["split"] == s).any()]
    by_split = {s: rows[rows["split"] == s] for s in splits}

    def of(frame: pd.DataFrame, model: str) -> pd.DataFrame:
        return frame[frame["model"] == model]

    tuning_scores = {m: _scores(of(by_split["tuning"], m)) for m in keys}
    table = [(m, tuning_scores[m]["rps"]) for m in candidates]
    index = choose_variant(table, spec.tie_tolerance) if fixed is None else candidates.index(fixed)
    chosen = candidates[index]
    chosen_variant = variants(spec)[index]
    report: dict[str, Any] = {
        "model": "m7",
        "model_version": model_version(chosen),
        "competition": inputs.competition,
        "seasons": {s: [int(x) for x in sorted(set(by_split[s]["season"]))] for s in splits},
        "checkpoints": list(spec.checkpoints),
        "n_sims": spec.n_sims,
        "seeds": [
            {
                "season": cp.season,
                "fraction": cp.fraction,
                "round": cp.round,
                "cutoff": cp.cutoff.strftime(TIME_FORMAT),
                "seed": cp.seed,
            }
            for cp in plan
        ],
        "formats": {
            str(season): {
                "label": info.fmt.season,
                "teams": info.fmt.teams,
                "rounds": info.fmt.regular_season_rounds,
                "sources": list(info.fmt.sources),
                "unverified": list(info.fmt.unverified),
            }
            for season, info in seasons.items()
        },
        "data_sha256": _data_sha256(inputs.games, inputs.team_games),
        "tuning_only": tuning_only,
        "validation_scored": not tuning_only,
        "test_scored": score_test and not tuning_only,
        "grid": {
            "objective": "mean RPS of the final rank on the tuning seasons",
            "tie_tolerance": spec.tie_tolerance,
            "candidates": {m: _metrics(of(by_split["tuning"], m)) for m in candidates},
            "baselines": {m: _metrics(of(by_split["tuning"], m)) for m in BASELINES},
            "best_on_edge": bool(spec.inflate) and chosen == candidates[-1],
        },
        "chosen": {
            "key": chosen,
            "spread": chosen_variant.spread,
            "net": chosen_variant.net,
            **({"fixed": "the EuroLeague verdict (K-a), not chosen here"} if fixed else {}),
            "tuning_rps": _round(tuning_scores[chosen]["rps"]),
            "tuning_brier": _round(tuning_scores[chosen]["brier"]),
        },
        "metrics": {s: {m: _metrics(of(by_split[s], m)) for m in keys} for s in splits},
        "per_checkpoint": {
            s: {
                f"{f:g}": {
                    m: _metrics(of(by_split[s][by_split[s]["checkpoint"] == round(f, 6)], m))
                    for m in keys
                }
                for f in sorted(spec.checkpoints)
            }
            for s in splits
        },
        "reliability": {
            s: {
                m: reliability(
                    of(by_split[s], m)["p_direct"].to_numpy(dtype=np.float64),
                    of(by_split[s], m)["made_direct"].to_numpy(dtype=np.float64),
                    BINS,
                )
                for m in keys
            }
            for s in splits
        },
        "title_brier": {
            s: {m: _round(_title_brier(of(by_split[s], m))) for m in keys} for s in splits
        },
        "differences": {s: _brier_differences(by_split[s], chosen, spec) for s in splits},
    }
    if not tuning_only:
        report["gate"] = _gate(rows, chosen, report["differences"], gated=fixed is None)
    return report


def _gate(
    rows: pd.DataFrame, chosen: str, differences: dict[str, Any], *, gated: bool
) -> dict[str, Any]:
    """The gate from the validation rows (undecided without any) and the pooled tuning +
    validation z of the chosen variant."""
    validation = rows[rows["split"] == "validation"]
    pooled = rows[rows["split"].isin(["tuning", "validation"])]
    gate = gate_block(
        chosen,
        brier_chosen=_scores(validation[validation["model"] == chosen])["brier"],
        brier_point_sim=_scores(validation[validation["model"] == "point_sim"])["brier"],
        brier_standings_now=_scores(validation[validation["model"] == "standings_now"])["brier"],
        z_pooled=_scores(pooled[pooled["model"] == chosen])["z"],
        differences=dict(differences.get("validation", {})),
    )
    gate["gated"] = gated
    return gate


def format_m7_table(report: dict[str, Any]) -> str:
    def cell(value: float | None, width: int, digits: int) -> str:
        return f"{'-' if value is None else f'{value:.{digits}f}':>{width}}"

    chosen = report["chosen"]
    lines = [f"chosen: {chosen['key']}; tuning RPS {chosen['tuning_rps']}"]
    header = f"{'split':<11}{'model':<17}{'n':>5}{'brier':>8}{'rps':>8}{'logloss':>9}{'z':>8}"
    lines += [header, "-" * len(header)]
    for split, by_model in report["metrics"].items():
        for name, m in by_model.items():
            lines.append(
                f"{split:<11}{name:<17}{m['n']:>5}{cell(m['brier'], 8, 4)}{cell(m['rps'], 8, 4)}"
                f"{cell(m['log_loss'], 9, 4)}{cell(m['spiegelhalter_z'], 8, 2)}"
            )
    gate = report.get("gate")
    if gate is not None:
        verdict = {True: "PASS", False: "FAIL", None: "n/a"}[gate["passed"]]
        lines.append(
            f"gate (validation Brier, {gate['variant']} vs point_sim "
            f"{gate['brier_point_sim']} and standings_now {gate['brier_standings_now']}; "
            f"pooled z {gate['spiegelhalter_z_pooled']}) -> {verdict}"
        )
    return "\n".join(lines)
