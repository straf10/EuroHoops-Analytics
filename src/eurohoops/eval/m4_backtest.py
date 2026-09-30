"""M4 league translation backtest (weeks 12-14 I8): does translating a mover's GBL season predict
his next EuroLeague season better than "same stats"?

Walk forward by EuroLeague target season t. A GBL->EL mover of t (``models/translation_pairs``:
>= 300 GBL minutes in t-1, < 100 EL minutes in t-1, >= 300 EL minutes in t) is predicted from his
GBL t-1 per-100 rates by five models, each fitted only on what was known before t:

- ``same_stats``: the GBL rates as they are (the naive baseline the gate is against);
- ``team_offset``: each stat scaled by the PAO/OLY team-level EL/GBL rate ratio, averaged over
  their club-seasons before t (I-k);
- ``shrunk_same_stats``: the GBL rate shrunk toward the EL mean rate (players >= 300 minutes,
  seasons before t) with weight minutes / (minutes + ``shrink_minutes``);
- ``translate`` and ``translate_team``: ``models/translation`` fitted on the pairs whose later
  season is before t (``translate_team`` adds the prior-season team-strength gap).

Loss per mover: the mean over the 11 stats of ((pred - actual) / sd_k)^2, sd_k the EL
between-player sd of stat k (player-seasons >= 300 minutes) over the tuning seasons; a split's
loss is the EL-minutes-weighted mean over its movers. The variant (``translate`` or
``translate_team``) is chosen on tuning; the gate (I-h) needs the upper bound of the person-level
bootstrap 95% CI of loss(chosen) - loss(same_stats) below 0 on validation. With fewer than
``min_validation_movers`` validation movers, the gate pools 2019-2023 and ``translate`` is the
declared variant (decided before any M4 number existed). EL->GBL movers are scored the same way
in the other direction and reported, not gated.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from eurohoops.config import M4Backtest
from eurohoops.models.box_impact import STAT_COLUMNS
from eurohoops.models.translation import StatFit, fit_translation, fits_report, to_el, to_gbl

FloatArray = npt.NDArray[np.float64]
MODELS = ("same_stats", "team_offset", "shrunk_same_stats", "translate", "translate_team")
VARIANTS = ("translate", "translate_team")
REFERENCE = "same_stats"
GATE_RULE = (
    "chosen translation variant beats same_stats iff the person-level bootstrap 95% CI "
    "(1,000 resamples, seed 20261015) of loss(chosen) - loss(same_stats) on validation has "
    "upper bound < 0; loss = EL-minutes-weighted mean of the per-mover mean over 11 stats of "
    "the squared error standardised by the stat's EL between-player sd (tuning seasons)"
)


def _round(value: float) -> float | None:
    return None if not np.isfinite(value) else round(float(value), 6)


def _gbl_cols() -> list[str]:
    return [f"rate_gbl_{k}" for k in STAT_COLUMNS]


def _el_cols() -> list[str]:
    return [f"rate_el_{k}" for k in STAT_COLUMNS]


def stat_sds(
    rates: pd.DataFrame, seasons: Sequence[int], min_minutes: float, competition: str
) -> dict[str, float]:
    """A league's between-player sd of each stat over ``seasons`` (player-seasons >=
    ``min_minutes``)."""
    rows = rates[
        (rates["competition"] == competition)
        & rates["season"].isin(seasons)
        & (rates["minutes"] >= min_minutes)
    ]
    return {k: float(rows[f"rate_{k}"].std(ddof=1)) for k in STAT_COLUMNS}


def league_means(
    rates: pd.DataFrame, competition: str, before: int, min_minutes: float
) -> FloatArray:
    """Possession-weighted mean rate per stat of a league's player-seasons >= ``min_minutes``
    in seasons before ``before``."""
    rows = rates[
        (rates["competition"] == competition)
        & (rates["season"] < before)
        & (rates["minutes"] >= min_minutes)
    ]
    weights = rows["poss"].to_numpy(dtype=np.float64)
    values = rows[[f"rate_{k}" for k in STAT_COLUMNS]].to_numpy(dtype=np.float64)
    return np.asarray((values * weights[:, None]).sum(axis=0) / weights.sum(), dtype=np.float64)


def team_ratios(
    player_games: Mapping[str, pd.DataFrame], clubs: Mapping[str, str], before: int
) -> FloatArray:
    """Per stat, the mean over PAO/OLY club-seasons before ``before`` (present in both leagues)
    of the club's EL per-possession rate over its GBL one (I-k)."""
    ratios = []
    for gbl_team, el_team in sorted(clubs.items()):
        sums = {}
        for competition, team in (("gbl", gbl_team), ("euroleague", el_team)):
            games = player_games[competition]
            own = games[(games["team"] == team) & (games["season"] < before)]
            sums[competition] = own.groupby("season")[[*STAT_COLUMNS, "poss"]].sum()
        both = sums["gbl"].index.intersection(sums["euroleague"].index)
        for season in both:
            el, gbl = sums["euroleague"].loc[season], sums["gbl"].loc[season]
            el_rate = el[list(STAT_COLUMNS)].to_numpy(dtype=np.float64) / float(el["poss"])
            gbl_rate = gbl[list(STAT_COLUMNS)].to_numpy(dtype=np.float64) / float(gbl["poss"])
            ratios.append(el_rate / np.where(gbl_rate > 0, gbl_rate, np.nan))
    if not ratios:
        return np.ones(len(STAT_COLUMNS))
    return np.asarray(np.nanmean(np.vstack(ratios), axis=0), dtype=np.float64)


def net_offset(
    net: pd.DataFrame, clubs: Mapping[str, str], resamples: int, seed: int
) -> dict[str, Any]:
    """I-k: over PAO/OLY club-seasons in both leagues, the club's net rating per 100 relative to
    the GBL average minus the same relative to the EuroLeague average, i.e. how many points per
    100 the average GBL team is worse than the average EuroLeague team; bootstrap CI over
    club-seasons. Reported, not a model input."""
    rows = []
    for gbl_team, el_team in sorted(clubs.items()):
        gbl = net[(net["competition"] == "gbl") & (net["team"] == gbl_team)].set_index("season")
        el = net[(net["competition"] == "euroleague") & (net["team"] == el_team)].set_index(
            "season"
        )
        for season in sorted(gbl.index.intersection(el.index)):
            gap = float(gbl.loc[season, "net"]) - float(el.loc[season, "net"])
            rows.append({"club": el_team, "season": int(season), "gap": _round(gap)})
    gaps = np.array([r["gap"] for r in rows], dtype=np.float64)
    if not len(gaps):
        return {"club_seasons": [], "mean": None, "ci95": [None, None]}
    rng = np.random.default_rng(seed)
    means = gaps[rng.integers(0, len(gaps), size=(resamples, len(gaps)))].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {
        "label": "points per 100 possessions the average GBL team is below the average "
        "EuroLeague team, from PAO/OLY in both leagues (reported, not gated)",
        "club_seasons": rows,
        "mean": _round(float(gaps.mean())),
        "ci95": [_round(float(low)), _round(float(high))],
        "resamples": resamples,
        "seed": seed,
    }


@dataclass(frozen=True)
class SeasonScores:
    season: int
    movers: pd.DataFrame  # person_id, minutes_el, one loss column per model
    fits: dict[str, dict[str, StatFit]]  # variant -> stat -> fit


def _losses(pred: FloatArray, actual: FloatArray, sd: FloatArray) -> FloatArray:
    return np.asarray((((pred - actual) / sd) ** 2).mean(axis=1), dtype=np.float64)


def score_season(
    pairs: pd.DataFrame,
    rates: pd.DataFrame,
    player_games: Mapping[str, pd.DataFrame],
    *,
    season: int,
    spec: M4Backtest,
    sds: Mapping[str, float],
    clubs: Mapping[str, str],
) -> SeasonScores:
    """Every model's loss for each GBL->EL mover of EuroLeague ``season``."""
    movers = pairs[(pairs["pair_type"] == "gbl_to_el") & (pairs["season_el"] == season)]
    movers = movers.sort_values("person_id").reset_index(drop=True)
    gbl = movers[_gbl_cols()].to_numpy(dtype=np.float64)
    actual = movers[_el_cols()].to_numpy(dtype=np.float64)
    sd = np.array([sds[k] for k in STAT_COLUMNS])
    fits = {
        "translate": fit_translation(pairs, before=season),
        "translate_team": fit_translation(pairs, before=season, team=True),
    }
    minutes = movers["minutes_gbl"].to_numpy(dtype=np.float64)[:, None]
    shrink = minutes / (minutes + spec.shrink_minutes)
    el_mean = league_means(rates, "euroleague", season, spec.sd_min_minutes)
    preds = {
        "same_stats": gbl,
        "team_offset": gbl * team_ratios(player_games, clubs, season),
        "shrunk_same_stats": shrink * gbl + (1.0 - shrink) * el_mean,
        "translate": to_el(movers[_gbl_cols()], fits["translate"]).to_numpy(dtype=np.float64),
        "translate_team": to_el(
            movers[_gbl_cols()], fits["translate_team"], movers["team_gap"]
        ).to_numpy(dtype=np.float64),
    }
    table = movers[["person_id", "minutes_el"]].copy()
    for model, pred in preds.items():
        table[model] = _losses(pred, actual, sd) if len(movers) else []
    return SeasonScores(season, table, fits)


def reverse_season(
    pairs: pd.DataFrame, *, season: int, sds_gbl: Mapping[str, float]
) -> pd.DataFrame:
    """EL->GBL movers of GBL ``season``: same_stats vs translate (reported, not gated)."""
    movers = pairs[(pairs["pair_type"] == "el_to_gbl") & (pairs["season_gbl"] == season)]
    movers = movers.sort_values("person_id").reset_index(drop=True)
    table = movers[["person_id", "minutes_gbl"]].copy()
    if movers.empty:
        return table.assign(same_stats=[], translate=[])
    el = movers[_el_cols()].to_numpy(dtype=np.float64)
    actual = movers[_gbl_cols()].to_numpy(dtype=np.float64)
    sd = np.array([sds_gbl[k] for k in STAT_COLUMNS])
    fits = fit_translation(pairs, before=season)
    translated = to_gbl(movers[_el_cols()], fits).to_numpy(dtype=np.float64)
    table["same_stats"] = _losses(el, actual, sd)
    table["translate"] = _losses(translated, actual, sd)
    return table


def weighted_loss(table: pd.DataFrame, model: str, weight: str = "minutes_el") -> float:
    w = table[weight].to_numpy(dtype=np.float64)
    if not len(w):
        return float("nan")
    return float((table[model].to_numpy(dtype=np.float64) * w).sum() / w.sum())


def person_bootstrap(
    table: pd.DataFrame, model: str, reference: str, resamples: int, seed: int
) -> dict[str, Any]:
    """Mean and 95% CI of loss(model) - loss(reference), resampling persons (a person who moves
    in two seasons of a pooled split is one cluster)."""
    if table.empty:
        return {"mean": None, "ci95": [None, None], "resamples": resamples, "seed": seed}
    codes, inverse = np.unique(table["person_id"].to_numpy(dtype=str), return_inverse=True)
    w = table["minutes_el"].to_numpy(dtype=np.float64)
    diff = (table[model] - table[reference]).to_numpy(dtype=np.float64)
    sums = np.bincount(inverse, weights=w * diff, minlength=len(codes))
    wsum = np.bincount(inverse, weights=w, minlength=len(codes))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(codes), size=(resamples, len(codes)))
    stats = sums[draws].sum(axis=1) / wsum[draws].sum(axis=1)
    low, high = np.percentile(stats, [2.5, 97.5])
    return {
        "mean": _round(float(sums.sum() / wsum.sum())),
        "ci95": [_round(float(low)), _round(float(high))],
        "resamples": resamples,
        "seed": seed,
    }


def _split_block(scores: Sequence[SeasonScores]) -> dict[str, Any]:
    table = pd.concat([s.movers for s in scores], ignore_index=True)
    return {
        "n": len(table),
        "seasons": {str(s.season): len(s.movers) for s in scores},
        "loss": {m: _round(weighted_loss(table, m)) for m in MODELS},
        "per_season": {
            str(s.season): {m: _round(weighted_loss(s.movers, m)) for m in MODELS} for s in scores
        },
    }


def run_m4_backtest(
    pairs: pd.DataFrame,
    rates: pd.DataFrame,
    player_games: Mapping[str, pd.DataFrame],
    *,
    spec: M4Backtest,
    clubs: Mapping[str, str],
    tuning_only: bool = False,
    score_test: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """(backtest report, translation report). ``tuning_only`` scores tuning and states the
    choice (the verdict commit); otherwise validation too, and test with ``score_test``."""
    if tuning_only and score_test:
        raise ValueError("--tuning-only and --score-test exclude each other")
    sds = stat_sds(rates, spec.tuning, spec.sd_min_minutes, "euroleague")
    sds_gbl = stat_sds(rates, spec.tuning, spec.sd_min_minutes, "gbl")

    def season_scores(seasons: Sequence[int]) -> list[SeasonScores]:
        return [
            score_season(pairs, rates, player_games, season=s, spec=spec, sds=sds, clubs=clubs)
            for s in seasons
        ]

    movers = pairs[pairs["pair_type"] == "gbl_to_el"]["season_el"].value_counts()
    n_validation = int(sum(movers.get(s, 0) for s in spec.validation))
    pooled = n_validation < spec.min_validation_movers
    tuning = season_scores(spec.tuning)
    tuning_block = _split_block(tuning)
    if pooled:
        chosen = "translate"
        choice_rule = (
            f"declared: {n_validation} validation movers < {spec.min_validation_movers}, so the "
            "gate pools tuning and validation with translate"
        )
    else:
        chosen = min(VARIANTS, key=lambda v: (tuning_block["loss"][v], v))
        choice_rule = "lowest tuning loss of translate and translate_team"
    report: dict[str, Any] = {
        "model": "m4",
        "seasons": {
            "tuning": list(spec.tuning),
            "validation": list(spec.validation),
            "test": list(spec.test),
        },
        "movers_gbl_to_el": {str(s): int(n) for s, n in sorted(movers.items())},
        "sd": {k: _round(v) for k, v in sds.items()},
        "chosen": {"variant": chosen, "rule": choice_rule, "tuning_loss": tuning_block["loss"]},
        "metrics": {"tuning": tuning_block},
        "validation_scored": False,
        "test_scored": False,
        "gate": None,
    }
    fits_by_season = {str(s.season): {v: fits_report(f) for v, f in s.fits.items()} for s in tuning}
    if not tuning_only:
        validation = season_scores(spec.validation)
        report["metrics"]["validation"] = _split_block(validation)
        report["validation_scored"] = True
        gate_scores = [*tuning, *validation] if pooled else validation
        gate_table = pd.concat([s.movers for s in gate_scores], ignore_index=True)
        diff = person_bootstrap(
            gate_table, chosen, REFERENCE, spec.bootstrap_resamples, spec.bootstrap_seed
        )
        high = diff["ci95"][1]
        report["gate"] = {
            "rule": GATE_RULE,
            "metric": "standardised squared error, minutes-weighted",
            "variant": chosen,
            "reference": REFERENCE,
            "pooled": pooled,
            "seasons": [s.season for s in gate_scores],
            "n_movers": len(gate_table),
            "loss_diff": diff,
            "passed": high is not None and high < 0,
        }
        fits_by_season |= {
            str(s.season): {v: fits_report(f) for v, f in s.fits.items()} for s in validation
        }
        reverse_seasons = [*spec.tuning, *spec.validation]
        if score_test:
            test = season_scores(spec.test)
            report["metrics"]["test"] = _split_block(test)
            report["test_scored"] = True
            fits_by_season |= {
                str(s.season): {v: fits_report(f) for v, f in s.fits.items()} for s in test
            }
            reverse_seasons += list(spec.test)
        reverse = pd.concat(
            [reverse_season(pairs, season=s, sds_gbl=sds_gbl) for s in reverse_seasons],
            ignore_index=True,
        )
        report["el_to_gbl"] = {
            "label": "reported, not gated",
            "seasons": reverse_seasons,
            "n": len(reverse),
            "loss": {
                m: _round(weighted_loss(reverse, m, "minutes_gbl"))
                for m in ("same_stats", "translate")
            },
        }
    report["pairs"] = {k: int(v) for k, v in pairs["pair_type"].value_counts().sort_index().items()}
    translation = {"model": "m4", "fits_by_target_season": fits_by_season}
    return report, translation
