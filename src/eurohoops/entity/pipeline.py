"""Entity resolution run (weeks 12-14 I3-I5): names -> matcher -> crosswalk -> report.

The matcher (``entity.match``) reads names through the injected transliteration
(``entity.translit``) and similarity (``entity.similarity``). The silver set (D2 of
reports/week12-14_progress.md) is built without names: a GBL id and a EuroLeague id on the same
Greek club in the same season with equal or near birth dates (D9) are positives, every other
pair of that club-season with two known, different birth dates a negative. The birth-date
thresholds ``t_dob`` and ``t_near`` were set by D9 from the name scores of all equal- and
near-date candidate pairs (label-free); the missing-date branch (``w_surname``, ``t_nodob``,
``b_club``, ``b_jersey``) is chosen on the silver set with birth dates hidden by ``tune``, before
any label existed.
"""

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast

import pandas as pd

from eurohoops.entity.match import (
    MatchParams,
    assign,
    candidate_pairs,
    careers,
    dob_status,
    score_pairs,
)
from eurohoops.entity.similarity import jaro_winkler
from eurohoops.entity.translit import latin_key, variants
from eurohoops.entity.xwalk import build_xwalk, entity_report, pair_metrics, wilson
from eurohoops.parse.player_names import pbp_links

# `tune` on the silver set (reports/entity_tuning.json); t_dob and t_near from D9
ENTITY_PARAMS = MatchParams(w_surname=0.5, t_nodob=0.82, b_club=0.08, b_jersey=0.0)
TUNING_BASE = MatchParams()
PBP_NAME_THRESHOLD = 0.9
LABEL_MIN_MINUTES = 100.0
LABEL_PER_STRATUM = 50
LABEL_CANDIDATES = 3
LABEL_SEED = 20261015
LABEL_OVERRIDE_PREFIX = "label:"
OVERRIDE_COLUMNS = [
    "competition_a",
    "source_a",
    "competition_b",
    "source_b",
    "decision",
    "reason",
    "date",
]
TUNING_GRID: dict[str, tuple[float, ...]] = {
    "w_surname": (0.5, 0.6, 0.7, 0.8),
    # widened once (2026-09-30, D9) after the first run chose the 0.86 / 0.08 edges
    "t_nodob": (0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.96),
    "b_club": (0.0, 0.03, 0.05, 0.08, 0.10),
    "b_jersey": (0.0, 0.03),
}


def read_overrides(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=OVERRIDE_COLUMNS, dtype=str)
    return pd.read_csv(path, dtype=str, keep_default_na=False)[OVERRIDE_COLUMNS]


def same_name(a: str, b: str) -> bool:
    """Whether two full names (either script) are the same by the matcher's transliteration."""
    target = latin_key(b) if not any("\u0370" <= ch <= "\u03ff" for ch in b) else None
    left = variants(a)
    right = (target,) if target is not None else variants(b)
    return max(jaro_winkler(x, y) for x in left for y in right) >= PBP_NAME_THRESHOLD


@dataclass(frozen=True)
class EntityRun:
    scored: pd.DataFrame
    matches: pd.DataFrame
    links: pd.DataFrame
    xwalk: pd.DataFrame
    report: dict[str, Any]


def run_entity(
    names: pd.DataFrame,
    bios: pd.DataFrame,
    overrides: pd.DataFrame,
    clubs: Mapping[str, str],
    params: MatchParams = ENTITY_PARAMS,
) -> EntityRun:
    pairs = candidate_pairs(names, bios, clubs, variants, latin_key, params)
    scored = score_pairs(pairs, names, variants, latin_key, jaro_winkler, params)
    matches = assign(scored, overrides)
    links = pbp_links(names, same_name)
    linked = links[links["linked_source_id"].notna()]
    xwalk = build_xwalk(names, matches, overrides, linked)
    report = entity_report(names, scored, xwalk, params)
    unlinked = links[links["linked_source_id"].isna()]
    report["pbp_links"] = {
        "keys": len(links),
        "linked": len(linked),
        "unlinked": dict(zip(unlinked["source_id"], unlinked["reason"], strict=True)),
    }
    return EntityRun(scored, matches, links, xwalk, report)


def silver_set(
    names: pd.DataFrame, bios: pd.DataFrame, clubs: Mapping[str, str]
) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    """(positives, negatives) as (GBL id, EL id), per D2 and D9: equal or near birth dates are
    positives, different ones negatives."""
    born = {
        (c, s): d
        for c, s, d in bios[["competition", "source_id", "birth_date"]].itertuples(index=False)
        if pd.notna(d)
    }
    near_days = ENTITY_PARAMS.near_days
    pos: set[tuple[str, str]] = set()
    neg: set[tuple[str, str]] = set()
    for gbl_team, el_team in clubs.items():
        gbl = names[(names["competition"] == "gbl") & (names["team"] == gbl_team)]
        el = names[(names["competition"] == "euroleague") & (names["team"] == el_team)]
        for season in sorted(set(gbl["season"]) & set(el["season"])):
            g_ids = sorted(set(gbl.loc[gbl["season"] == season, "source_id"]))
            e_ids = sorted(set(el.loc[el["season"] == season, "source_id"]))
            for g, e in itertools.product(g_ids, e_ids):
                dg, de = born.get(("gbl", g)), born.get(("euroleague", e))
                if dg is None or de is None:
                    continue
                same = dob_status(dg, de, near_days) in {"equal", "near"}
                (pos if same else neg).add((g, e))
    return pos, neg - pos


def _predicted(matches: pd.DataFrame) -> set[tuple[str, str]]:
    return set(zip(matches["gbl_id"].astype(str), matches["el_id"].astype(str), strict=True))


def silver_metrics(
    run_matches: pd.DataFrame, pos: set[tuple[str, str]], neg: set[tuple[str, str]]
) -> dict[str, Any]:
    metrics: dict[str, Any] = dict(pair_metrics(_predicted(run_matches), pos, neg))
    metrics["positives"], metrics["negatives"] = len(pos), len(neg)
    tp, fn = int(metrics["tp"]), int(metrics["fn"])
    metrics["recall_ci95"] = list(wilson(tp, tp + fn))
    return metrics


def _is_greek(text: str) -> bool:
    return any("\u0370" <= ch <= "\u03ff" for ch in text)


def _frozen_overrides(overrides: pd.DataFrame) -> pd.DataFrame:
    """The overrides that existed before the labels: everything not marked ``label:``."""
    return overrides[~overrides["reason"].str.startswith(LABEL_OVERRIDE_PREFIX)]


def _careers_text(
    names: pd.DataFrame, competition: str, team_names: Mapping[str, str]
) -> dict[str, str]:
    rows = names[names["competition"] == competition].sort_values(["source_id", "season", "team"])
    out: dict[str, list[str]] = {}
    for r in rows.itertuples(index=False):
        jersey = "" if pd.isna(r.jersey) else f" #{r.jersey!s}"
        minutes = float(cast(Any, r.minutes))
        team = team_names.get(str(r.team), str(r.team))
        out.setdefault(str(r.source_id), []).append(f"{r.season!s} {team}{jersey} {minutes:.0f}m")
    return {k: "; ".join(v) for k, v in out.items()}


def draft_labels(
    names: pd.DataFrame,
    bios: pd.DataFrame,
    overrides: pd.DataFrame,
    clubs: Mapping[str, str],
    *,
    params: MatchParams = ENTITY_PARAMS,
    team_names: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """The owner's label sheet (I-g, D8): GBL ids with >= ``LABEL_MIN_MINUTES`` minutes in 4
    strata {Greek-script, Latin-only name} x {matched, not matched by the frozen matcher},
    ``LABEL_PER_STRATUM`` each (seeded), with the evidence and the top-3 EuroLeague candidates by
    name score (birth dates shown; the sheet stays in data/, only the labels are committed)."""
    frozen = _frozen_overrides(overrides)
    pairs = candidate_pairs(names, bios, clubs, variants, latin_key, params)
    scored = score_pairs(
        pairs, names, variants, latin_key, jaro_winkler, params, skip_different=False
    )
    matched = set(assign(scored, frozen)["gbl_id"])
    gbl = names[(names["competition"] == "gbl") & ~names["source_id"].str.startswith("pbp:")]
    totals = gbl.groupby("source_id").agg(
        minutes=("minutes", "sum"), surname=("surname_raw", "first"), first=("first_raw", "first")
    )
    pool = totals[totals["minutes"] >= LABEL_MIN_MINUTES].reset_index()
    greek = [_is_greek(f"{s} {f}") for s, f in zip(pool["surname"], pool["first"], strict=True)]
    pool["script"] = ["greek" if g else "latin" for g in greek]
    pool["frozen"] = ["matched" if s in matched else "unmatched" for s in pool["source_id"]]
    pool["stratum"] = pool["script"] + "/" + pool["frozen"]
    picks = [
        group.sample(min(LABEL_PER_STRATUM, len(group)), random_state=LABEL_SEED)
        for _, group in pool.sort_values("source_id").groupby("stratum", sort=True)
    ]
    sample = pd.concat(picks).sort_values(["stratum", "source_id"])
    born = {
        (c, s): d
        for c, s, d in bios[["competition", "source_id", "birth_date"]].itertuples(index=False)
    }
    el_names = careers(names[names["competition"] == "euroleague"]).set_index("source_id")
    shown = team_names or {}
    gbl_text = _careers_text(names, "gbl", shown)
    el_text = _careers_text(names, "euroleague", shown)
    rows = []
    for g in sample.itertuples(index=False):
        own = scored[scored["gbl_id"] == g.source_id].sort_values(
            ["name_score", "el_id"], ascending=[False, True]
        )
        row: dict[str, Any] = {
            "gbl_id": g.source_id,
            "stratum": g.stratum,
            "stratum_size": int((pool["stratum"] == g.stratum).sum()),
            "gbl_name": f"{g.surname} {g.first}",
            "gbl_birth_date": born.get(("gbl", g.source_id)),
            "gbl_career": gbl_text.get(str(g.source_id), ""),
            "esake_url": f"https://www.esake.gr/el/action/EsakeplayerView?idplayer={g.source_id}&mode=1",
        }
        for i, cand in enumerate(own.head(LABEL_CANDIDATES).itertuples(index=False), start=1):
            el = el_names.loc[cand.el_id]
            row |= {
                f"cand{i}_el_id": cand.el_id,
                f"cand{i}_name": f"{el['surname']}, {el['first']}",
                f"cand{i}_birth_date": born.get(("euroleague", cand.el_id)),
                f"cand{i}_career": el_text.get(str(cand.el_id), ""),
                f"cand{i}_name_score": round(float(cast(Any, cand.name_score)), 3),
            }
        row["el_id_true"] = ""
        rows.append(row)
    return pd.DataFrame(rows)


def label_metrics(run: EntityRun, overrides: pd.DataFrame, labels: pd.DataFrame) -> dict[str, Any]:
    """Precision and recall on the owner's labels (I-g), per stratum and weighted by stratum
    sizes, for the matcher as frozen (overrides from the labels excluded) and after overrides.

    ``labels``: ``gbl_id, stratum, stratum_size, el_id_true`` (an EL id or ``none``)."""
    frozen = _predicted(assign(run.scored, _frozen_overrides(overrides)))
    x = run.xwalk
    el_of = x[x["competition"] == "euroleague"].groupby("person_id")["source_id"].min()
    gbl = x[(x["competition"] == "gbl") & x["person_id"].isin(el_of.index)]
    after = set(zip(gbl["source_id"], el_of.reindex(gbl["person_id"]), strict=True))
    return {
        "as_frozen": _label_block(frozen, labels),
        "after_overrides": {
            **_label_block(after, labels),
            "label": "fitted on the labels (overrides added from them)",
        },
    }


def _label_block(predicted: set[tuple[str, str]], labels: pd.DataFrame) -> dict[str, Any]:
    pick = {g: e for g, e in predicted}
    strata: dict[str, dict[str, Any]] = {}
    weighted = {"correct": 0.0, "predicted": 0.0, "true": 0.0}
    for stratum, group in labels.groupby("stratum", sort=True):
        truth = dict(zip(group["gbl_id"], group["el_id_true"].str.strip(), strict=True))
        guess = {g: pick.get(g) for g in truth}
        predicted_n = sum(v is not None for v in guess.values())
        true_n = sum(t != "none" for t in truth.values())
        correct = sum(guess[g] is not None and guess[g] == truth[g] for g in truth)
        weight = float(group["stratum_size"].astype(int).iloc[0]) / len(group)
        weighted["correct"] += weight * correct
        weighted["predicted"] += weight * predicted_n
        weighted["true"] += weight * true_n
        strata[str(stratum)] = {
            "n": len(group),
            "predicted": predicted_n,
            "true_matches": true_n,
            "correct": correct,
            "precision": round(correct / predicted_n, 6) if predicted_n else None,
            "recall": round(correct / true_n, 6) if true_n else None,
            "precision_ci95": list(wilson(correct, predicted_n)),
            "recall_ci95": list(wilson(correct, true_n)),
        }
    totals = {
        k: sum(s[k] for s in strata.values()) for k in ("predicted", "true_matches", "correct")
    }
    return {
        "strata": strata,
        "precision": round(totals["correct"] / totals["predicted"], 6)
        if totals["predicted"]
        else None,
        "recall": round(totals["correct"] / totals["true_matches"], 6)
        if totals["true_matches"]
        else None,
        "precision_ci95": list(wilson(totals["correct"], totals["predicted"])),
        "recall_ci95": list(wilson(totals["correct"], totals["true_matches"])),
        "precision_weighted": round(weighted["correct"] / weighted["predicted"], 6)
        if weighted["predicted"]
        else None,
        "recall_weighted": round(weighted["correct"] / weighted["true"], 6)
        if weighted["true"]
        else None,
    }


def tune(
    names: pd.DataFrame,
    bios: pd.DataFrame,
    clubs: Mapping[str, str],
    grid: Mapping[str, Sequence[float]] = TUNING_GRID,
) -> dict[str, Any]:
    """The missing-date branch (D3, D9) on the silver set with birth dates hidden: maximise F1
    subject to silver precision >= 0.99 (if no grid point meets it: the best F1; ties: the
    earlier grid point). ``t_dob``/``t_near`` stay as set by D9; the chosen parameters are then
    also scored on the silver set with birth dates visible."""
    pos, neg = silver_set(names, bios, clubs)
    none = pd.DataFrame(columns=OVERRIDE_COLUMNS, dtype=str)
    base = TUNING_BASE
    silver_ids = {g for g, _ in pos | neg} | {e for _, e in pos | neg}
    sub = names[names["source_id"].isin(silver_ids)]
    hidden = bios.iloc[0:0]
    rows = []
    keys = list(grid)
    rest = [k for k in keys if k != "w_surname"]
    for w in grid["w_surname"]:
        # blocking and name scores depend on w only; thresholds and bonuses are re-applied
        params = replace(base, w_surname=float(w))
        pairs = candidate_pairs(sub, hidden, clubs, variants, latin_key, params)
        scored = score_pairs(pairs, sub, variants, latin_key, jaro_winkler, params)
        for values in itertools.product(*(grid[k] for k in rest)):
            point = {"w_surname": float(w), **dict(zip(rest, map(float, values), strict=True))}
            bonus = point["b_club"] * scored["same_club_season"].astype(float) + point[
                "b_jersey"
            ] * scored["same_jersey"].astype(float)
            trial = scored.assign(score=scored["name_score"] + bonus)
            trial["accepted"] = trial["score"] >= point["t_nodob"]
            m = silver_metrics(assign(trial, none), pos, neg)
            rows.append({**point, **{k: m[k] for k in ("precision", "recall", "f1")}})
    table = pd.DataFrame(rows)
    ok = table[table["precision"] >= 0.99]
    best = (ok if not ok.empty else table).sort_values("f1", ascending=False, kind="stable").iloc[0]
    chosen = MatchParams(**{**asdict(base), **{k: float(best[k]) for k in keys}})
    pairs = candidate_pairs(sub, bios, clubs, variants, latin_key, chosen)
    scored = score_pairs(pairs, sub, variants, latin_key, jaro_winkler, chosen)
    visible = silver_metrics(assign(scored, none), pos, neg)
    return {
        "silver": {"positives": len(pos), "negatives": len(neg)},
        "grid": {k: list(v) for k, v in grid.items()},
        "chosen": asdict(chosen),
        "chosen_silver_hidden_dates": {k: float(best[k]) for k in ("precision", "recall", "f1")},
        "chosen_silver_visible_dates": {k: visible[k] for k in ("precision", "recall", "f1")},
        "precision_rule_met": not ok.empty,
        "table": table.round(6).to_dict("records"),
    }
