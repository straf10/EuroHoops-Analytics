"""Live M6 (weeks 16-18 L9): the 2026-27 player projections, the GBL undervalued list, the
over/under board and the "plays like" lists.

Projections. The verdict's cell (``reports/backtest_m6.json`` ``chosen``) projects every player
with a group-stage line in the live season, per competition, through the harness's own inputs
(``m6_backtest.live_inputs``: the same history, impact rows, drift and noise as a test target).
The checkpoint is the largest declared one whose rounds are complete (D22): 0 (next-season,
from every season before the live one) until k = floor(0.25 R) rounds are played, then 0.25,
0.5, 0.75, so a projection changes four times a season and a day without a newly completed
checkpoint leaves it unchanged. The intervals carry the D19 variance scale frozen at the verdict
for that checkpoint (each competition's own report: the GBL's from its fixed-cell run). M4's
newest fit translates (D5). Ages are not read: the chosen variant does not age (a variant that
did would fit the curve from ``ages``, in memory only).

Undervalued (GBL, L-e, D24): a GBL player of the live season with at most
``UNDERVALUED_MAX_SEASONS`` seasons since his first senior appearance and fewer than
``UNDERVALUED_MAX_MPG`` minutes per game in his newest complete GBL season, whose projected
rates translated to the EuroLeague (M4's ``(rate + c) exp(delta) - c``, floored at 0; fg2a and pf,
which M6 does not project, from that newest season) give a EuroLeague SPM in the top quarter of
the GBL's rotation players (``min_poss`` possessions in their newest complete GBL season) by the
same rule. Sorted by that SPM.

Board (D23) and similarity: the newest complete season (``board_season``: the last season with a
BRAPM snapshot and M2 xPTS), stabilities from the M6 tuning seasons; the "plays like" pool is
every complete season with ``min_poss`` possessions, and a live player's query is his newest
such season. Neither moves during the live season.

No output carries a birth date or an age (``bios.py`` rule).
"""

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from eurohoops.eval.m6_backtest import (
    M6Inputs,
    TargetInputs,
    World,
    live_inputs,
    project_cell,
    scale_intervals,
)
from eurohoops.live_sim import completed_rounds
from eurohoops.models.board import DIMENSIONS, LABELS, UNLABELLED, Stability
from eurohoops.models.player_seasons import COUNT_STATS, PROJECTED_STATS, person_ids
from eurohoops.models.projection import Translation
from eurohoops.models.similarity import BOX_FEATURES, SHOT_FEATURES
from eurohoops.stats.export import display_name

AGE_KEYS = re.compile(r"(^age$|_age$|^age_|birth|born|dob)", re.IGNORECASE)
ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
UNDERVALUED_MAX_SEASONS = 3  # L-e: seasons since the first senior appearance (no age bucket)
UNDERVALUED_MAX_MPG = 20.0  # "low minutes" (declared)
UNDERVALUED_QUANTILE = 0.75  # top quarter of the GBL rotation by translated EuroLeague SPM
DIGITS = 4


def person_names(
    el_players: pd.DataFrame, gbl_names: pd.DataFrame, xwalk: pd.DataFrame
) -> dict[str, str]:
    """Person id -> display name: the EuroLeague box-score spelling (``stats.box`` lines, the
    last line of each id), else the GBL name table's (``parse.player_names`` rows, the newest
    season of each id)."""
    names: dict[str, str] = {}
    if not gbl_names.empty:
        last = gbl_names.sort_values("season").drop_duplicates("source_id", keep="last")
        ids = person_ids(last["source_id"], "gbl", xwalk)["person_id"]
        for pid, surname, first in zip(ids, last["surname_raw"], last["first_raw"], strict=True):
            names[pid] = display_name(f"{surname}, {first}")
    if not el_players.empty:
        last_line = el_players.drop_duplicates("player_id", keep="last")
        ids = person_ids(last_line["player_id"], "euroleague", xwalk)["person_id"]
        names.update(
            {pid: display_name(str(n)) for pid, n in zip(ids, last_line["player"], strict=True)}
        )
    return names


def personal_fields(node: Any, path: str = "$") -> list[str]:
    """Every key of ``node`` (a JSON payload) that names an age or a birth date, and every string
    value that is an ISO date alone (the shape of a birth date; a time stamp with a time is a
    tip-off or a cutoff): a report must have none (``bios.py`` rule, L-e)."""
    found = []
    if isinstance(node, dict):
        for key, value in node.items():
            if AGE_KEYS.search(str(key)):
                found.append(f"{path}.{key}")
            found += personal_fields(value, f"{path}.{key}")
    elif isinstance(node, list):
        for i, value in enumerate(node):
            found += personal_fields(value, f"{path}[{i}]")
    elif isinstance(node, str) and ISO_DATE.match(node):
        found.append(f"{path} = {node}")
    return found


def live_checkpoint(regular: pd.DataFrame, rounds: int, checkpoints: Sequence[float]) -> float:
    """The largest of 0 and ``checkpoints`` whose k = floor(f R) regular-season rounds are all
    played (D22). ``regular``: the live season's regular-season games with ``round`` and
    ``played``."""
    done = completed_rounds(regular)
    return max(f for f in (0.0, *checkpoints) if math.floor(f * rounds) <= done)


def live_people(complete: pd.DataFrame, competition: str, season: int) -> list[str]:
    """Persons with a group-stage line with seconds in ``competition``'s live season."""
    rows = complete[
        (complete["competition"] == competition)
        & (complete["season"] == season)
        & (complete["games"] > 0)
    ]
    return sorted(rows["person_id"].unique())


def live_projections(
    inputs: M6Inputs,
    competition: str,
    season: int,
    checkpoint: float,
    people: Sequence[str],
    *,
    chosen: Mapping[str, Any],
    scale: Mapping[str, float],
    spec: Any,
    translation: Translation | None,
    rounds: int,
    world: World,
) -> tuple[pd.DataFrame, TargetInputs]:
    """The chosen cell's projections (long, ``PROJECTIONS_SCHEMA``) of ``people`` with the
    frozen D19 ``scale`` applied to their intervals, and the block's inputs."""
    ti = live_inputs(
        inputs,
        competition,
        season,
        checkpoint,
        people,
        spec=spec,
        translation=translation,
        rounds=lambda _c, _s: rounds,
        world=world,
        with_curve=False,
    )
    rows = project_cell(ti, str(chosen["variant"]), float(chosen["half_life"]))
    return scale_intervals(rows, scale, spec.interval), ti


def inputs_digest(blocks: Sequence[TargetInputs]) -> str:
    """sha256 of what the projections are a function of: each block's history, targets, impact
    rows, drift and translation (not the schedule: a newly scheduled game moves nothing)."""
    h = hashlib.sha256()
    for ti in blocks:
        for frame in (ti.history, ti.targets, ti.impact):
            if frame is not None:
                h.update(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes())
        extra = {
            "drift": sorted(ti.drift.items()),
            "translation": None
            if ti.translation is None
            else [sorted(ti.translation.delta.items()), sorted(ti.translation.c.items())],
        }
        h.update(json.dumps(extra, sort_keys=True).encode())
    return h.hexdigest()


def _r(value: float) -> float | None:
    return None if not math.isfinite(value) else round(float(value), DIGITS)


def player_rows(
    projections: pd.DataFrame,
    seasons: pd.DataFrame,
    names: Mapping[str, str],
) -> list[dict[str, Any]]:
    """One report row per projected person: team and debut from the live season's I1 rows
    (``seasons``), each stat's mean and 80% interval (a stat without a mean is left out),
    the decayed possessions and seasons behind the points projection, the flags."""
    meta = {(str(r["person_id"]), str(r["competition"])): r for r in seasons.to_dict("records")}
    grouped: dict[tuple[str, str], list[dict[Any, Any]]] = {}
    for r in projections.to_dict("records"):
        grouped.setdefault((str(r["person_id"]), str(r["competition"])), []).append(r)
    out = []
    for key in sorted(grouped):
        rows, row = grouped[key], meta[key]
        stats = {
            str(r["stat"]): {"mean": _r(r["mean"]), "lo80": _r(r["lo80"]), "hi80": _r(r["hi80"])}
            for r in rows
            if math.isfinite(r["mean"])
        }
        pts = next(r for r in rows if r["stat"] == "pts")
        flags = sorted({f for r in rows for f in str(r["flags"]).split("|") if f})
        out.append(
            {
                "person_id": key[0],
                "competition": key[1],
                "team": str(row["team"]),
                "name": names.get(key[0], key[0]),
                "debut_season": int(row["debut_season"]),
                "stats": stats,
                "exposure": _r(float(pts["exposure"])),
                "n_seasons": int(pts["n_seasons"]),
                "flags": flags,
            }
        )
    return out


def translate_el(rates: pd.DataFrame, translation: Translation) -> pd.DataFrame:
    """GBL per-100 rates as EuroLeague rates: ``(rate + c) exp(delta) - c`` per column M4
    translates, floored at 0 (a rate is never negative)."""
    out = rates.copy()
    for stat in rates.columns:
        delta, c = translation.delta[stat], translation.c[stat]
        out[stat] = np.maximum((rates[stat] + c) * math.exp(delta) - c, 0.0)
    return out


def undervalued(
    projections: pd.DataFrame,
    complete: pd.DataFrame,
    season: int,
    *,
    translation: Translation,
    spm: Any,
    names: Mapping[str, str],
    teams: Mapping[str, str],
    min_poss: float,
) -> list[dict[str, Any]]:
    """The GBL undervalued list (module docstring, D24). ``projections``: the GBL live block;
    ``complete``: the I1 rows of every season; ``translation`` carries every SPM stat."""
    gbl = complete[(complete["competition"] == "gbl") & (complete["season"] < season)]
    newest = gbl.sort_values("season").drop_duplicates("person_id", keep="last")
    newest = newest.set_index("person_id")
    means = projections.pivot_table(index="person_id", columns="stat", values="mean")
    people = means.index.intersection(newest.index)
    if people.empty:
        return []
    last = newest.loc[people]
    poss = last["poss"].to_numpy(dtype=np.float64)
    rates = pd.DataFrame(
        {stat: means.loc[people, stat].to_numpy(dtype=np.float64) for stat in COUNT_STATS},
        index=people,
    )
    for stat in ("fg2a", "pf"):
        rates[stat] = 100.0 * last[stat].to_numpy(dtype=np.float64) / np.maximum(poss, 1e-12)
    el = translate_el(rates, translation)
    frame = el.reset_index(drop=True).assign(competition="euroleague", season=season)
    value = np.asarray(spm(frame, season), dtype=np.float64)
    mpg = last["minutes"].to_numpy(dtype=np.float64) / np.maximum(
        last["games"].to_numpy(dtype=np.float64), 1.0
    )
    since = season - last["debut_season"].to_numpy(dtype=np.int64)
    rotation = poss >= min_poss
    if not rotation.any():
        return []
    cut = float(np.quantile(value[rotation], UNDERVALUED_QUANTILE))
    pick = (since <= UNDERVALUED_MAX_SEASONS) & (mpg < UNDERVALUED_MAX_MPG) & (value >= cut)
    ids = [str(p) for p in people]
    last_season = last["season"].to_numpy(dtype=np.int64)
    last_team = [str(t) for t in last["team"]]
    translated = el.to_numpy(dtype=np.float64)
    columns = list(el.columns)
    order = sorted(np.flatnonzero(pick), key=lambda i: (-value[i], ids[i]))
    return [
        {
            "person_id": ids[i],
            "team": teams.get(ids[i], last_team[i]),
            "name": names.get(ids[i], ids[i]),
            "seasons_since_debut": int(since[i]),
            "minutes": _r(float(mpg[i])),
            "projected_brapm": None,
            "projected_spm": _r(float(value[i])),
            "translated_el": {s: _r(float(translated[i, columns.index(s)])) for s in COUNT_STATS},
            "reason": (
                f"translated EuroLeague SPM {value[i]:+.1f} (GBL rotation top quarter: "
                f">= {cut:+.1f}) at {mpg[i]:.1f} minutes per game in {last_season[i]}, "
                f"{since[i]} season(s) since his debut"
            ),
        }
        for i in order
    ]


def projections_report(
    *,
    chosen: Mapping[str, Any],
    gate_passed: bool,
    season: int,
    checkpoints: Mapping[str, float],
    digest: str,
    players: list[dict[str, Any]],
    undervalued_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """``reports/m6_projections.json`` (the shape the read model serves): ``checkpoint`` is the
    EuroLeague's, ``checkpoints`` each competition's (a player's is his competition's)."""
    return {
        "model": "m6",
        "variant": str(chosen["variant"]),
        "half_life": float(chosen["half_life"]),
        "gated": True,
        "gate_passed": gate_passed,
        "season": season,
        "checkpoint": checkpoints["euroleague"],
        "checkpoints": dict(checkpoints),
        "data_sha256": digest,
        "stats": list(PROJECTED_STATS),
        "players": players,
        "undervalued": undervalued_rows,
    }


def board_report(
    rows: pd.DataFrame,
    stabilities: Mapping[str, Stability],
    season: int,
    names: Mapping[str, str],
    teams: Mapping[tuple[str, str], str],
) -> dict[str, Any]:
    """``reports/m6_board.json``: each dimension's stability, ``n_min`` and k, and the
    ``BOARD_SCHEMA`` rows with name and team, and the number of rows per label (spaces in a
    label become underscores; on/off has none)."""
    dims = {}
    for dim in DIMENSIONS:
        if dim not in stabilities:
            continue
        stab = stabilities[dim]
        k = None if not 0.0 < stab.r < 1.0 else stab.mean_n * (1.0 - stab.r) / stab.r
        dims[dim] = {
            "stability": _r(stab.r),
            "pairs": stab.pairs,
            "n_min": stab.n_min,
            "labelled": dim not in UNLABELLED,
            "k": None if k is None else _r(k),
        }
    out = []
    for r in rows.to_dict("records"):
        key = (str(r["person_id"]), str(r["competition"]))
        out.append(
            {
                **{k: (_r(v) if isinstance(v, float) else v) for k, v in r.items()},
                "season": int(r["season"]),
                "name": names.get(key[0], key[0]),
                "team": teams.get(key, ""),
            }
        )
    counts = {
        dim: {
            label.replace(" ", "_"): sum(r["dimension"] == dim and r["label"] == label for r in out)
            for label in LABELS
        }
        for dim in dims
        if dim not in UNLABELLED
    }
    return {
        "model": "m6",
        "season": season,
        "dimensions": dims,
        "label_counts": counts,
        "rows": out,
    }


def similarity_report(
    similar: pd.DataFrame,
    season: int,
    pool_seasons: Sequence[int],
    shot_features: bool,
    names: Mapping[str, str],
) -> dict[str, Any]:
    """``reports/m6_similarity.json``: per live person his ``neighbours`` rows (``SIMILAR_SCHEMA``)
    as the match's person, competition, season, name and score, with his query season."""
    players: dict[str, list[dict[str, Any]]] = {}
    for r in similar.sort_values(["person_id", "rank"]).to_dict("records"):
        match = str(r["match_person_id"])
        players.setdefault(str(r["person_id"]), []).append(
            {
                "rank": int(r["rank"]),
                "person_id": match,
                "competition": str(r["match_competition"]),
                "season": int(r["match_season"]),
                "name": names.get(match, match),
                "score": _r(float(r["score"])),
                "query_season": int(r["season"]),
            }
        )
    el = [*BOX_FEATURES, *(SHOT_FEATURES if shot_features else ())]
    return {
        "model": "m6",
        "season": season,
        "features": {"euroleague": el, "gbl": list(BOX_FEATURES)},
        "pool_seasons": sorted(int(s) for s in pool_seasons),
        "players": players,
    }


def query_rows(
    complete: pd.DataFrame, people: Sequence[str], before: int, min_poss: float
) -> pd.DataFrame:
    """Each live person's newest complete season before ``before`` with ``min_poss``
    possessions (his "plays like" query); a person without one has none."""
    rows = complete[
        complete["person_id"].isin(set(people))
        & (complete["season"] < before)
        & (complete["poss"] >= min_poss)
        & ~complete["partial"]
    ]
    rows = rows.sort_values(["person_id", "season", "poss"], kind="stable")
    return rows.drop_duplicates("person_id", keep="last").reset_index(drop=True)
