"""Free-throw generation (F-e): FT trips and FT points per team-game from EuroLeague play-by-play.

The shot feed has no free-throw attempts on missed shots (it logs made free throws only), so
xPTS alone undervalues shots that draw fouls. The play-by-play has every free throw (``FTM``
made, ``FTA`` missed), but its fouls are untyped (``CM`` is any personal foul): a trip can be
tied to the shot that drew it only when it is an **and-one**, i.e. a one-shot trip whose last
field-goal row before it is a made field goal by the same team at the same clock. That shot's
``NUMBEROFPLAY`` is the feed's ``NUM_ANOT``, which gives the shot's distance band. Every other
trip (shooting fouls on misses, bonus free throws, technicals) stays at team level.

A trip is a run of free-throw rows of one team at one clock (period + ``MARKERTIME``), other
rows in between (substitutions, time-outs) do not split it. Expected FT points of a team-game:

    sum over bands b of FGA_b * and-one trips per FGA_b * points per and-one trip
    + FGA * other trips per FGA * points per other trip

with every rate a ratio of totals over the seasons it is fitted on (leave-one-season-out for
development seasons, all development seasons for validation and test).

**The F2 check (re-specified by the user, weeks 7-10b decision 2)** tests what the model can
know: the within-season *shares*, which a league-wide level shift leaves alone. Per held-out
season: (i) each distance band's share of the and-one FT points, expected vs actual; (ii) each
team's share of the league's FT points, expected vs actual, summarised by the mean absolute
share gap and the Pearson r across the season's teams. A team's share is of per-game rates
(its FT points per game over the sum of every team's), so a team that played more playoff games
does not get a bigger share. Each tolerance is ``SHARE_SE_MULTIPLE`` x the statistic's
game-level bootstrap standard error on the same season (``SHARE_RESAMPLES`` resamples of the
season's games, seed ``SHARE_SEED``):

- band b: |actual share - expected share| <= 2 SE of that gap;
- teams: mean |gap| <= 2 x the root-mean-square of the teams' gap SEs (the noise scale of one
  team's gap; with a right model the mean |gap| sits near 0.8 of it);
- teams: r >= 2 SE of r (expected shares track the actual ones beyond noise).

The per-season mean gap of FT points per team-game (the first F2 check, ±0.1) stays in the
report as a limitation, labelled "season level, not gated": out-of-season rates cannot know a
season's level.
"""

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from eurohoops.ingest.cache import read_cached
from eurohoops.models.elo import FloatArray
from eurohoops.parse.shot_table import BANDS, MADE_CODES, SHOT_VALUE, period_of
from eurohoops.parse.stints import QUARTERS

FT_CODES = frozenset({"FTM", "FTA"})
COUNT_COLUMNS = (
    "fga",
    *(f"fga_{b}" for b in BANDS),
    "fta",
    "ftm",
    "trips",
    "and_one_trips",
    *(f"and_one_trips_{b}" for b in BANDS),
    "and_one_ftm",
    *(f"and_one_ftm_{b}" for b in BANDS),
    "other_trips",
    "other_ftm",
)


@dataclass(frozen=True)
class Trip:
    team: str
    period: int
    clock: str
    fta: int
    ftm: int
    kind: str  # "and_one" or "other"
    shot_event: int | None  # NUMBEROFPLAY of the made field goal an and-one belongs to


def pbp_rows(pbp: dict[str, Any]) -> list[dict[str, Any]]:
    """Every PBP row in log order."""
    return [row for key in (*QUARTERS, "ExtraTime") for row in pbp.get(key) or []]


def trips(rows: Sequence[dict[str, Any]]) -> list[Trip]:
    """Free-throw trips of one game in log order, and-ones tied to their made field goal."""
    out: list[dict[str, Any]] = []
    last_fg: dict[str, Any] | None = None
    for row in rows:
        kind = str(row.get("PLAYTYPE") or "").strip()
        team = str(row.get("CODETEAM") or "").strip()
        if kind in SHOT_VALUE:
            last_fg = row
            continue
        if kind not in FT_CODES:
            continue
        key = (team, period_of(int(row["MINUTE"])), str(row.get("MARKERTIME") or ""))
        if out and out[-1]["key"] == key:
            out[-1]["fta"] += 1
            out[-1]["ftm"] += kind == "FTM"
            continue
        tied = (
            last_fg is not None
            and str(last_fg.get("PLAYTYPE") or "").strip() in MADE_CODES
            and str(last_fg.get("CODETEAM") or "").strip() == team
            and (period_of(int(last_fg["MINUTE"])), str(last_fg.get("MARKERTIME") or "")) == key[1:]
        )
        out.append(
            {
                "key": key,
                "fta": 1,
                "ftm": int(kind == "FTM"),
                "event": int(last_fg["NUMBEROFPLAY"]) if tied and last_fg is not None else None,
            }
        )
    return [
        Trip(
            t["key"][0],
            t["key"][1],
            t["key"][2],
            t["fta"],
            t["ftm"],
            "and_one" if t["event"] is not None and t["fta"] == 1 else "other",
            t["event"] if t["event"] is not None and t["fta"] == 1 else None,
        )
        for t in out
    ]


def build_ft_team_games(raw_dir: Path, shots: pd.DataFrame, excluded: pd.DataFrame) -> pd.DataFrame:
    """One row per validated-season team-game of the shot table: FGA (all, and per band of the
    kept shots), FT trips and FT points by kind. Games without cached PBP are left out."""
    band_of = {
        (str(g), int(e)): str(b)
        for g, e, b in zip(shots["game_id"], shots["event"], shots["band"], strict=True)
    }
    kept = shots[shots["validated_season"]]
    fga_band = kept.pivot_table(
        index=["season", "game_id", "team"],
        columns="band",
        values="event",
        aggfunc="size",
        fill_value=0,
    )
    fga_band = fga_band.reindex(columns=list(BANDS), fill_value=0).add_prefix("fga_")
    dropped = excluded[excluded["game_id"].isin(set(kept["game_id"]))]
    fga_all = pd.concat([kept[["game_id", "team"]], dropped[["game_id", "team"]]])
    fga = fga_all.groupby(["game_id", "team"]).size().rename("fga")
    frame = fga_band.reset_index()
    frame["fga"] = [
        fga.get((g, t), 0) for g, t in zip(frame["game_id"], frame["team"], strict=True)
    ]
    ft: dict[tuple[str, str], dict[str, int]] = {}
    has_pbp: set[str] = set()
    for game_id, season in sorted(set(zip(frame["game_id"], frame["season"], strict=True))):
        code = str(game_id).split("_")[1]
        path = raw_dir / "playbyplay" / f"E{season}" / f"{code}.json.gz"
        if not path.exists():
            continue
        has_pbp.add(str(game_id))
        for trip in trips(pbp_rows(json.loads(read_cached(path)))):
            tally = ft.setdefault((str(game_id), trip.team), {})
            band = band_of.get((str(game_id), trip.shot_event or -1))
            # an and-one tied to an excluded shot has no band: it counts as an other trip
            kind = "and_one" if trip.kind == "and_one" and band is not None else "other"
            names = ["fta", "ftm", "trips", f"{kind}_trips", f"{kind}_ftm"]
            if kind == "and_one":
                names += [f"and_one_trips_{band}", f"and_one_ftm_{band}"]
            for name in names:
                step = trip.fta if name == "fta" else trip.ftm if "ftm" in name.split("_") else 1
                tally[name] = tally.get(name, 0) + step
    frame = frame[frame["game_id"].isin(has_pbp)].reset_index(drop=True)
    for column in COUNT_COLUMNS:
        if column.startswith("fga"):
            continue
        frame[column] = [
            ft.get((g, t), {}).get(column, 0)
            for g, t in zip(frame["game_id"], frame["team"], strict=True)
        ]
    frame = frame[["season", "game_id", "team", *COUNT_COLUMNS]]
    return frame.astype({c: "int64" for c in ("season", *COUNT_COLUMNS)}).sort_values(
        ["game_id", "team"], ignore_index=True
    )


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def fit_rates(team_games: pd.DataFrame) -> dict[str, Any]:
    """Ratio-of-totals rates on the given team-games (0 where a denominator is empty)."""
    total = team_games.sum(numeric_only=True)
    return {
        "and_one_trips_per_fga": {
            b: _ratio(total[f"and_one_trips_{b}"], total[f"fga_{b}"]) for b in BANDS
        },
        "points_per_and_one_trip": _ratio(total["and_one_ftm"], total["and_one_trips"]),
        "other_trips_per_fga": _ratio(total["other_trips"], total["fga"]),
        "points_per_other_trip": _ratio(total["other_ftm"], total["other_trips"]),
    }


def expected_ft_points(team_games: pd.DataFrame, rates: dict[str, Any]) -> pd.Series:
    and_one = (
        sum(team_games[f"fga_{b}"] * rates["and_one_trips_per_fga"][b] for b in BANDS)
        * rates["points_per_and_one_trip"]
    )
    other = team_games["fga"] * rates["other_trips_per_fga"] * rates["points_per_other_trip"]
    return pd.Series(np.asarray(and_one + other, dtype=np.float64), index=team_games.index)


def expected_and_one_by_band(team_games: pd.DataFrame, rates: dict[str, Any]) -> pd.DataFrame:
    """Expected and-one FT points per team-game in each distance band (one column per band)."""
    return pd.DataFrame(
        {
            b: team_games[f"fga_{b}"].to_numpy(dtype=np.float64)
            * rates["and_one_trips_per_fga"][b]
            * rates["points_per_and_one_trip"]
            for b in BANDS
        },
        index=team_games.index,
    )


# The first F2 check, kept as a limitation (season level, not gated): mean |actual - expected|
# FT points per team-game.
DEVELOPMENT_TOLERANCE = 0.1
VALIDATION_TOLERANCE = 0.2
LEVEL_CHECK = "season level, not gated"
SHARE_RESAMPLES = 1000
SHARE_SEED = 20261001
SHARE_SE_MULTIPLE = 2.0
# Count rule (the user's decision, 2026-09-26): ~13 x 6 band checks at 2 SE flag a few by chance
# even when the model is right, so the gate allows up to the 95th percentile of that count.
BAND_FLAG_RATE = math.erfc(SHARE_SE_MULTIPLE / math.sqrt(2.0))  # P(|Z| > 2) = 0.0455
BAND_FLAG_QUANTILE = 0.95


def allowed_flags(checks: int, rate: float = BAND_FLAG_RATE, q: float = BAND_FLAG_QUANTILE) -> int:
    """The q-quantile of Binomial(checks, rate): flags a right model reaches at most with
    probability q."""
    cdf = 0.0
    for k in range(checks + 1):
        cdf += math.comb(checks, k) * rate**k * (1.0 - rate) ** (checks - k)
        if cdf >= q:
            return k
    return checks


def _game_weights(game_ids: npt.NDArray[Any], resamples: int, seed: int) -> FloatArray:
    """(resamples x team-games) weights: how often each row's game is drawn when the season's
    games are resampled with replacement."""
    _, inverse = np.unique(game_ids, return_inverse=True)
    n_games = int(inverse.max()) + 1
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_games, size=(resamples, n_games))
    counts = np.stack([np.bincount(row, minlength=n_games) for row in draws]).astype(np.float64)
    out: FloatArray = counts[:, inverse]
    return out


def _band_gaps(w: FloatArray, actual: FloatArray, expected: FloatArray) -> FloatArray:
    """(rows of w) x bands: actual minus expected share of the and-one FT points."""
    a, e = w @ actual, w @ expected
    out: FloatArray = a / a.sum(axis=1, keepdims=True) - e / e.sum(axis=1, keepdims=True)
    return out


def _team_stats(
    w: FloatArray, teams: FloatArray, actual: FloatArray, expected: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Per row of w: the teams' share gaps (actual - expected, shares of per-game FT points)
    and the Pearson r of expected vs actual shares across teams."""
    games = w @ teams
    with np.errstate(invalid="ignore", divide="ignore"):
        rate_a, rate_e = (w * actual) @ teams / games, (w * expected) @ teams / games
    share_a = rate_a / np.nansum(rate_a, axis=1, keepdims=True)
    share_e = rate_e / np.nansum(rate_e, axis=1, keepdims=True)
    da = share_a - np.nanmean(share_a, axis=1, keepdims=True)
    de = share_e - np.nanmean(share_e, axis=1, keepdims=True)
    r = np.nansum(da * de, axis=1) / np.sqrt(np.nansum(da**2, axis=1) * np.nansum(de**2, axis=1))
    return share_a - share_e, r


def share_checks(
    rows: pd.DataFrame,
    expected_bands: pd.DataFrame,
    expected_total: pd.Series,
    resamples: int = SHARE_RESAMPLES,
    seed: int = SHARE_SEED,
) -> dict[str, Any]:
    """The F2 share checks of one held-out season (see the module doc); JSON-ready."""
    actual_bands = rows[[f"and_one_ftm_{b}" for b in BANDS]].to_numpy(dtype=np.float64)
    exp_bands = expected_bands.loc[rows.index, list(BANDS)].to_numpy(dtype=np.float64)
    names = sorted(set(rows["team"]))
    teams = (rows["team"].to_numpy()[:, None] == np.array(names)[None, :]).astype(np.float64)
    ftm = rows["ftm"].to_numpy(dtype=np.float64)
    exp_ft = expected_total[rows.index].to_numpy(dtype=np.float64)
    one = np.ones((1, len(rows)))
    boot = _game_weights(rows["game_id"].to_numpy(), resamples, seed)

    band_gap = _band_gaps(one, actual_bands, exp_bands)[0]
    band_se = _band_gaps(boot, actual_bands, exp_bands).std(axis=0, ddof=1)
    a_share = actual_bands.sum(axis=0) / actual_bands.sum()
    e_share = exp_bands.sum(axis=0) / exp_bands.sum()
    bands = {
        b: {
            "actual_share": round(float(a_share[i]), 6),
            "expected_share": round(float(e_share[i]), 6),
            "gap": round(float(band_gap[i]), 6),
            "bootstrap_se": round(float(band_se[i]), 6),
            "tolerance": round(float(SHARE_SE_MULTIPLE * band_se[i]), 6),
            "within_tolerance": bool(abs(band_gap[i]) <= SHARE_SE_MULTIPLE * band_se[i]),
        }
        for i, b in enumerate(BANDS)
    }

    gaps, r = _team_stats(one, teams, ftm, exp_ft)
    boot_gaps, boot_r = _team_stats(boot, teams, ftm, exp_ft)
    gap_se = np.nanstd(boot_gaps, axis=0, ddof=1)
    noise = float(np.sqrt(np.mean(gap_se**2)))
    mean_abs = float(np.mean(np.abs(gaps[0])))
    r_se = float(np.nanstd(boot_r, ddof=1))
    team_block = {
        "teams": len(names),
        "mean_abs_share_gap": round(mean_abs, 6),
        "gap_se_rms": round(noise, 6),
        "mean_abs_tolerance": round(SHARE_SE_MULTIPLE * noise, 6),
        "mean_abs_within_tolerance": bool(mean_abs <= SHARE_SE_MULTIPLE * noise),
        "pearson_r": round(float(r[0]), 6),
        "r_bootstrap_se": round(r_se, 6),
        "r_tolerance": round(SHARE_SE_MULTIPLE * r_se, 6),
        "r_within_tolerance": bool(r[0] >= SHARE_SE_MULTIPLE * r_se),
    }
    return {
        "bands": bands,
        "band_flags": sum(not b["within_tolerance"] for b in bands.values()),
        "teams": team_block,
        "teams_pass": bool(
            team_block["mean_abs_within_tolerance"] and team_block["r_within_tolerance"]
        ),
    }


def share_gate(seasons: dict[str, Any]) -> dict[str, Any]:
    """The F2 gate over every held-out season: band flags within the count rule, and both team
    checks in every season."""
    checks = sum(len(s["shares"]["bands"]) for s in seasons.values())
    flags = sum(s["shares"]["band_flags"] for s in seasons.values())
    allowed = allowed_flags(checks)
    teams_ok = all(s["shares"]["teams_pass"] for s in seasons.values())
    return {
        "band_checks": checks,
        "band_flags": flags,
        "band_flag_rate_if_right": round(BAND_FLAG_RATE, 6),
        "band_flags_allowed": allowed,
        "band_flags_within": flags <= allowed,
        "team_checks_pass_every_season": teams_ok,
        "passed": bool(flags <= allowed and teams_ok),
    }


def _rounded(rates: dict[str, Any]) -> dict[str, Any]:
    return {
        k: {b: round(v, 6) for b, v in value.items()}
        if isinstance(value, dict)
        else round(value, 6)
        for k, value in rates.items()
    }


def ft_report(
    team_games: pd.DataFrame, development: Sequence[int], validation: Sequence[int]
) -> dict[str, Any]:
    """Rates on development; per season the share checks (the F2 gate) and the mean actual vs
    expected FT points per team-game (season level, not gated); LOSO for development seasons.
    JSON-ready."""
    scored = team_games[team_games["season"].isin([*development, *validation])]
    expected = out_of_fold_expected(scored, development, validation)
    by_band = out_of_fold(scored, development, validation, expected_and_one_by_band)
    seasons: dict[str, Any] = {}
    for season, rows in scored.groupby("season"):
        gap = float((rows["ftm"] - expected[rows.index]).mean())
        tolerance = DEVELOPMENT_TOLERANCE if season in development else VALIDATION_TOLERANCE
        total = rows.sum(numeric_only=True)
        seasons[str(season)] = {
            "split": "development" if season in development else "validation",
            "team_games": len(rows),
            "ft_points_per_team_game": round(float(rows["ftm"].mean()), 4),
            "expected_per_team_game": round(float(expected[rows.index].mean()), 4),
            "mean_gap": round(gap, 4),
            "tolerance": tolerance,
            "within_tolerance": abs(gap) <= tolerance,
            "trips_per_fga": round(float(total["trips"] / total["fga"]), 6),
            "points_per_trip": round(float(total["ftm"] / total["trips"]), 6),
            "and_one_share_of_trips": round(float(total["and_one_trips"] / total["trips"]), 6),
            "level_check": LEVEL_CHECK,
            "shares": share_checks(rows, by_band, expected),
        }
    dev = scored["season"].isin(development)
    return {
        "method": "ratio-of-totals rates; development seasons leave-one-season-out",
        "fouls_tied_to_shots": "and-ones only (a one-shot trip after the same team's made field "
        "goal at the same clock); other trips are team-level",
        "development": list(development),
        "validation": list(validation),
        "rates_development": _rounded(fit_rates(scored[dev])),
        "season_ft_points_sd_development": round(
            float(scored[dev].groupby("season")["ftm"].mean().std(ddof=1)), 4
        ),
        "seasons": seasons,
        "level_check": LEVEL_CHECK,
        "all_within_tolerance": all(s["within_tolerance"] for s in seasons.values()),
        "share_check": {
            "gate": "F2 (re-specified 2026-09-26): band and team shares within "
            f"{SHARE_SE_MULTIPLE:g} x their game-level bootstrap SE, every development season "
            "and validation; band flags by the count rule (95th percentile of a right model)",
            "resamples": SHARE_RESAMPLES,
            "seed": SHARE_SEED,
            "se_multiple": SHARE_SE_MULTIPLE,
            **share_gate(seasons),
        },
        "share_checks_pass": share_gate(seasons)["passed"],
    }


Predict = Callable[[pd.DataFrame, dict[str, Any]], pd.DataFrame]


def out_of_fold(
    team_games: pd.DataFrame, development: Sequence[int], later: Sequence[int], predict: Predict
) -> pd.DataFrame:
    """``predict(rows, rates)`` per team-game: development seasons with rates fitted on the other
    development seasons (LOSO), later seasons with rates fitted on all development seasons;
    NaN elsewhere."""
    dev = team_games["season"].isin(development)
    parts = []
    for season in development:
        mask = team_games["season"] == season
        parts.append(predict(team_games[mask], fit_rates(team_games[dev & ~mask])))
    held = team_games["season"].isin(later)
    parts.append(predict(team_games[held], fit_rates(team_games[dev])))
    return pd.concat(parts).reindex(team_games.index)


def out_of_fold_expected(
    team_games: pd.DataFrame, development: Sequence[int], later: Sequence[int]
) -> pd.Series:
    """Expected FT points per team-game, out of fold (see ``out_of_fold``)."""
    frame = out_of_fold(
        team_games, development, later, lambda tg, r: expected_ft_points(tg, r).to_frame("x")
    )
    return frame["x"]
