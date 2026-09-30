"""Cross-league player matcher (D3): block, score, one-to-one assign.

Rule (``reports/week12-14_progress.md`` D3). For a GBL id ``g`` and EuroLeague id ``e``:

- ``name_score`` = max(w·surname_sim + (1−w)·first_sim, full_sim), each sim the best
  Jaro-Winkler over transliteration variants (injected). Empty first → surname only; a
  one-letter first that equals the other's first letter scores 1.0.
- Birth dates equal → accept iff ``name_score ≥ t_dob``; near (D9: ESAKE and the EuroLeague
  disagree by days, a month, a swap or a year) → iff ``name_score ≥ t_near``; different →
  never; unknown → iff ``name_score + b_club·same_club_season + b_jersey·same_jersey ≥ t_nodob``.
- Assignment score adds +1.0 for equal dates and +0.5 for near ones, so they outrank unknowns.
- Blocking (cheap, recall-safe): career windows ``[first−window, last+window]`` overlap,
  and at least one of same club-season, equal birth date, or matching surname first letter
  (latin_key of EL vs some variant of GBL). Only GBL×EL pairs are scored.
"""

from __future__ import annotations

import functools
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

Variants = Callable[[str], tuple[str, ...]]
LatinKey = Callable[[str], str]
Similarity = Callable[[str, str], float]


@dataclass(frozen=True)
class MatchParams:
    w_surname: float = 0.6
    t_dob: float = 0.78
    t_near: float = 0.95
    t_nodob: float = 0.92
    b_club: float = 0.05
    b_jersey: float = 0.03
    window: int = 2
    near_days: int = 31


def careers(names: pd.DataFrame) -> pd.DataFrame:
    """One row per (competition, source_id): career span and dominant spelling."""
    rows: list[dict[str, object]] = []
    for (comp, sid), group in names.groupby(["competition", "source_id"], sort=True):
        spelling = (
            group.groupby(["surname_raw", "first_raw"], sort=True)
            .agg(games=("games", "sum"))
            .reset_index()
            .sort_values(["games", "surname_raw", "first_raw"], ascending=[False, True, True])
        )
        top = spelling.iloc[0]
        rows.append(
            {
                "competition": comp,
                "source_id": sid,
                "first_season": int(group["season"].min()),
                "last_season": int(group["season"].max()),
                "surname": str(top["surname_raw"]),
                "first": str(top["first_raw"]),
                "minutes": float(group["minutes"].sum()),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "competition",
            "source_id",
            "first_season",
            "last_season",
            "surname",
            "first",
            "minutes",
        ],
    )


def _bio_map(bios: pd.DataFrame) -> dict[tuple[str, str], date | None]:
    if bios.empty:
        return {}
    out: dict[tuple[str, str], date | None] = {}
    for row in bios.itertuples(index=False):
        birth = cast(date | None, row.birth_date)
        if birth is not None and not isinstance(birth, date):
            birth = None
        out[(str(row.competition), str(row.source_id))] = birth
    return out


def _same_club_season(names: pd.DataFrame, clubs: Mapping[str, str]) -> set[tuple[str, str]]:
    """GBL×EL id pairs that shared a mapped club in the same season."""
    gbl = names[names["competition"] == "gbl"]
    el = names[names["competition"] == "euroleague"]
    pairs: set[tuple[str, str]] = set()
    for season, g_season in gbl.groupby("season", sort=False):
        e_season = el[el["season"] == season]
        if e_season.empty:
            continue
        el_by_team: dict[str, set[str]] = defaultdict(set)
        for row in e_season.itertuples(index=False):
            el_by_team[str(row.team)].add(str(row.source_id))
        for row in g_season.itertuples(index=False):
            el_team = clubs.get(str(row.team))
            if el_team is None:
                continue
            for el_id in el_by_team.get(el_team, ()):
                pairs.add((str(row.source_id), el_id))
    return pairs


def _same_jersey(names: pd.DataFrame) -> set[tuple[str, str]]:
    """GBL×EL id pairs that wore the same non-null jersey in some shared season."""
    gbl = names[(names["competition"] == "gbl") & names["jersey"].notna()]
    el = names[(names["competition"] == "euroleague") & names["jersey"].notna()]
    pairs: set[tuple[str, str]] = set()
    for season, g_season in gbl.groupby("season", sort=False):
        e_season = el[el["season"] == season]
        if e_season.empty:
            continue
        el_by_jersey: dict[str, set[str]] = defaultdict(set)
        for row in e_season.itertuples(index=False):
            el_by_jersey[str(row.jersey)].add(str(row.source_id))
        for row in g_season.itertuples(index=False):
            for el_id in el_by_jersey.get(str(row.jersey), ()):
                pairs.add((str(row.source_id), el_id))
    return pairs


def dob_status(g_dob: date | None, e_dob: date | None, near_days: int) -> str:
    """``equal``, ``near`` (the sources disagree slightly: within ``near_days``, day and month
    swapped, or the year off by one), ``different`` or ``unknown`` (D9)."""
    if g_dob is None or e_dob is None:
        return "unknown"
    if g_dob == e_dob:
        return "equal"
    swapped = (g_dob.day, g_dob.month, g_dob.year) == (e_dob.month, e_dob.day, e_dob.year)
    year_off = (g_dob.day, g_dob.month) == (e_dob.day, e_dob.month) and abs(
        g_dob.year - e_dob.year
    ) == 1
    if abs((g_dob - e_dob).days) <= near_days or swapped or year_off:
        return "near"
    return "different"


def candidate_pairs(  # noqa: PLR0917 -- fixed matcher API (names, bios, clubs, fns, params)
    names: pd.DataFrame,
    bios: pd.DataFrame,
    clubs: Mapping[str, str],
    variants: Variants,
    latin_key: LatinKey,
    params: MatchParams,
) -> pd.DataFrame:
    """Block GBL×EL pairs; columns ``gbl_id, el_id, same_club_season, same_jersey, dob``."""
    cards = careers(names)
    gbl = cards[cards["competition"] == "gbl"].reset_index(drop=True)
    el = cards[cards["competition"] == "euroleague"].reset_index(drop=True)
    bios_map = _bio_map(bios)
    club_pairs = _same_club_season(names, clubs)
    jersey_pairs = _same_jersey(names)

    el_ids = [str(s) for s in el["source_id"]]
    el_letters = [(latin_key(str(s)) or " ")[0] for s in el["surname"]]
    el_dobs = [bios_map.get(("euroleague", e_id)) for e_id in el_ids]
    el_first = el["first_season"].to_numpy(dtype=np.int64) - params.window
    el_last = el["last_season"].to_numpy(dtype=np.int64) + params.window

    rows: list[dict[str, object]] = []
    for g in gbl.itertuples(index=False):
        g_id = str(g.source_id)
        g_letters = {v[0] for v in variants(str(g.surname)) if v}
        g_dob = bios_map.get(("gbl", g_id))
        g_lo = int(cast(Any, g.first_season)) - params.window
        g_hi = int(cast(Any, g.last_season)) + params.window
        for j in np.flatnonzero((g_lo <= el_last) & (el_first <= g_hi)):
            e_id = el_ids[j]
            same_club = (g_id, e_id) in club_pairs
            dob = dob_status(g_dob, el_dobs[j], params.near_days)
            letter_hit = el_letters[j] in g_letters
            if not (same_club or dob in {"equal", "near"} or letter_hit):
                continue
            rows.append(
                {
                    "gbl_id": g_id,
                    "el_id": e_id,
                    "same_club_season": same_club,
                    "same_jersey": (g_id, e_id) in jersey_pairs,
                    "dob": dob,
                }
            )
    out = pd.DataFrame(
        rows,
        columns=["gbl_id", "el_id", "same_club_season", "same_jersey", "dob"],
    )
    if out.empty:
        return out
    return out.sort_values(["gbl_id", "el_id"]).reset_index(drop=True)


def _first_sim(
    g_first: str,
    e_first: str,
    variants: Variants,
    latin_key: LatinKey,
    similarity: Similarity,
) -> float | None:
    """Return first-name similarity, or ``None`` when either side is empty."""
    if not g_first or not e_first:
        return None
    e_key = latin_key(e_first)
    best = 0.0
    for v in variants(g_first):
        best = max(best, similarity(v, e_key))
        if len(v) == 1 and e_key and v[0] == e_key[0]:
            best = 1.0
        if len(e_key) == 1 and v and v[0] == e_key[0]:
            best = 1.0
    return best


def _name_features(  # noqa: PLR0917 -- surname/first/full sim need both sides and injectables
    g_surname: str,
    g_first: str,
    e_surname: str,
    e_first: str,
    variants: Variants,
    latin_key: LatinKey,
    similarity: Similarity,
    params: MatchParams,
) -> tuple[float, float, float, float]:
    e_sur_key = latin_key(e_surname)
    surname_sim = max((similarity(v, e_sur_key) for v in variants(g_surname)), default=0.0)
    first = _first_sim(g_first, e_first, variants, latin_key, similarity)
    e_full_key = latin_key(f"{e_surname} {e_first}".strip())
    full_sim = max(
        (similarity(v, e_full_key) for v in variants(f"{g_surname} {g_first}".strip())),
        default=0.0,
    )
    if first is None:
        name_score = surname_sim
        first_sim = 0.0
    else:
        first_sim = first
        weighted = params.w_surname * surname_sim + (1.0 - params.w_surname) * first_sim
        name_score = max(weighted, full_sim)
    return surname_sim, first_sim, full_sim, name_score


def score_pairs(  # noqa: PLR0917 -- fixed matcher API (pairs, names, injectables, params)
    pairs: pd.DataFrame,
    names: pd.DataFrame,
    variants: Variants,
    latin_key: LatinKey,
    similarity: Similarity,
    params: MatchParams,
    *,
    skip_different: bool = True,
) -> pd.DataFrame:
    """Add name sims, ``name_score``, ``accepted``, and assignment ``score``.

    Pairs whose birth dates differ can never be accepted; with ``skip_different`` (the matcher
    run) their name features are left NaN, which saves most of the scoring time (≈ 64% of the
    real candidate pairs, 2026-09-30). The label sheet scores them with ``skip_different=False``.
    """
    cards = careers(names)
    by_id = {(str(r.competition), str(r.source_id)): r for r in cards.itertuples(index=False)}
    latin_key = functools.lru_cache(maxsize=None)(latin_key)
    variant_cache: dict[str, tuple[str, ...]] = {}

    def v_cached(sid: str, text: str) -> tuple[str, ...]:
        key = f"{sid}:{text}"
        hit = variant_cache.get(key)
        if hit is None:
            hit = variants(text)
            variant_cache[key] = hit
        return hit

    records: list[dict[str, object]] = []
    for row in pairs.itertuples(index=False):
        g = by_id[("gbl", str(row.gbl_id))]
        e = by_id[("euroleague", str(row.el_id))]
        dob = str(row.dob)
        club = bool(row.same_club_season)
        jersey = bool(row.same_jersey)
        if skip_different and dob == "different":
            surname_sim = first_sim = full_sim = name_score = float("nan")
        else:

            def g_variants(text: str, sid: str = str(row.gbl_id)) -> tuple[str, ...]:
                return v_cached(sid, text)

            surname_sim, first_sim, full_sim, name_score = _name_features(
                str(g.surname),
                str(g.first),
                str(e.surname),
                str(e.first),
                g_variants,
                latin_key,
                similarity,
                params,
            )
        if dob == "equal":
            accepted = name_score >= params.t_dob
            score = name_score + 1.0
        elif dob == "near":
            accepted = name_score >= params.t_near
            score = name_score + 0.5
        elif dob == "different":
            accepted = False
            score = name_score
        else:
            score = name_score + params.b_club * float(club) + params.b_jersey * float(jersey)
            accepted = score >= params.t_nodob
        records.append(
            {
                "gbl_id": str(row.gbl_id),
                "el_id": str(row.el_id),
                "same_club_season": club,
                "same_jersey": jersey,
                "dob": dob,
                "surname_sim": surname_sim,
                "first_sim": first_sim,
                "full_sim": full_sim,
                "name_score": name_score,
                "accepted": accepted,
                "score": score,
            }
        )
    cols = [
        "gbl_id",
        "el_id",
        "same_club_season",
        "same_jersey",
        "dob",
        "surname_sim",
        "first_sim",
        "full_sim",
        "name_score",
        "accepted",
        "score",
    ]
    out = pd.DataFrame(records, columns=cols)
    if out.empty:
        return out
    return out.sort_values(["gbl_id", "el_id"]).reset_index(drop=True)


def _no_match_pairs(overrides: pd.DataFrame) -> set[tuple[str, str]]:
    """GBL×EL pairs forbidden by ``no_match`` overrides (either column order)."""
    blocked: set[tuple[str, str]] = set()
    if overrides.empty:
        return blocked
    for row in overrides.itertuples(index=False):
        if str(row.decision) != "no_match":
            continue
        ca, cb = str(row.competition_a), str(row.competition_b)
        sa, sb = str(row.source_a), str(row.source_b)
        if ca == "gbl" and cb == "euroleague":
            blocked.add((sa, sb))
        elif ca == "euroleague" and cb == "gbl":
            blocked.add((sb, sa))
    return blocked


def _components(
    edges: list[tuple[str, str, float]],
) -> list[tuple[list[str], list[str], list[tuple[str, str, float]]]]:
    """Connected components of a bipartite GBL–EL graph."""
    adj: dict[str, list[str]] = defaultdict(list)
    edge_map: dict[tuple[str, str], float] = {}
    for g, e, score in edges:
        adj[f"g:{g}"].append(f"e:{e}")
        adj[f"e:{e}"].append(f"g:{g}")
        edge_map[(g, e)] = score
    seen: set[str] = set()
    comps: list[tuple[list[str], list[str], list[tuple[str, str, float]]]] = []
    for start in sorted(adj):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        g_ids: list[str] = []
        e_ids: list[str] = []
        while stack:
            node = stack.pop()
            kind, sid = node.split(":", 1)
            if kind == "g":
                g_ids.append(sid)
            else:
                e_ids.append(sid)
            for nbr in adj[node]:
                if nbr not in seen:
                    seen.add(nbr)
                    stack.append(nbr)
        g_ids = sorted(g_ids)
        e_ids = sorted(e_ids)
        comp_edges = [(g, e, edge_map[(g, e)]) for g in g_ids for e in e_ids if (g, e) in edge_map]
        comps.append((g_ids, e_ids, comp_edges))
    return comps


def _assign_component(
    g_ids: list[str], e_ids: list[str], edges: list[tuple[str, str, float]]
) -> list[tuple[str, str, float]]:
    """Maximise total score with ``linear_sum_assignment`` (cost = −score)."""
    if not g_ids or not e_ids or not edges:
        return []
    score_map = {(g, e): s for g, e, s in edges}
    n_g, n_e = len(g_ids), len(e_ids)
    # Pad to square; unmatched cells get a large positive cost (low score).
    n = max(n_g, n_e)
    fill = 1.0e9
    cost = [[fill] * n for _ in range(n)]
    for i, g in enumerate(g_ids):
        for j, e in enumerate(e_ids):
            if (g, e) in score_map:
                cost[i][j] = -score_map[(g, e)]
    row_ind, col_ind = linear_sum_assignment(cost)
    chosen: list[tuple[str, str, float]] = []
    for i_raw, j_raw in zip(row_ind, col_ind, strict=True):
        i, j = int(i_raw), int(j_raw)
        if i >= n_g or j >= n_e:
            continue
        g, e = g_ids[i], e_ids[j]
        if (g, e) not in score_map:
            continue
        chosen.append((g, e, score_map[(g, e)]))
    return chosen


def assign(scored: pd.DataFrame, overrides: pd.DataFrame) -> pd.DataFrame:
    """One-to-one matches among accepted pairs; ``no_match`` overrides win first."""
    blocked = _no_match_pairs(overrides)
    if scored.empty:
        return pd.DataFrame(columns=["gbl_id", "el_id", "score"])
    work = scored[scored["accepted"]].copy()
    if blocked:
        keep = [
            (str(g), str(e)) not in blocked
            for g, e in zip(work["gbl_id"], work["el_id"], strict=True)
        ]
        work = work.loc[keep]
    edges = [
        (str(r.gbl_id), str(r.el_id), float(cast(Any, r.score)))
        for r in work.sort_values(["gbl_id", "el_id"]).itertuples(index=False)
    ]
    chosen: list[tuple[str, str, float]] = []
    for g_ids, e_ids, comp_edges in _components(edges):
        chosen.extend(_assign_component(g_ids, e_ids, comp_edges))
    chosen.sort(key=lambda t: (t[0], t[1]))
    return pd.DataFrame(chosen, columns=["gbl_id", "el_id", "score"])
