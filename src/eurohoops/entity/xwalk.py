"""Player crosswalk: one person_id per source id (I-b / D3).

Union-find over rule matches, ``match`` overrides and within-league links. ``person_id`` is
``P:<smallest EuroLeague id>`` when the component has one, else ``G:<smallest GBL id>``.
Every ``(competition, source_id)`` in ``names`` appears once; singletons use method ``none``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from math import sqrt
from typing import Any, cast

import pandas as pd
import pandera.pandas as pa

from eurohoops.entity.match import MatchParams
from eurohoops.parse.schemas import validated

METHODS = ("rule", "override", "pbp", "none")

XWALK_SCHEMA = pa.DataFrameSchema(
    {
        "person_id": pa.Column(str),
        "competition": pa.Column(str, pa.Check.isin(["euroleague", "gbl"])),
        "source_id": pa.Column(str),
        "method": pa.Column(str, pa.Check.isin(list(METHODS))),
        "score": pa.Column("float64", nullable=True),
    },
    unique=["competition", "source_id"],
    strict=True,
)


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[tuple[str, str], tuple[str, str]] = {}
        self.rank: dict[tuple[str, str], int] = {}

    def add(self, node: tuple[str, str]) -> None:
        if node not in self.parent:
            self.parent[node] = node
            self.rank[node] = 0

    def find(self, node: tuple[str, str]) -> tuple[str, str]:
        self.add(node)
        while self.parent[node] != node:
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node

    def union(self, a: tuple[str, str], b: tuple[str, str]) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def _nodes_from_names(names: pd.DataFrame) -> list[tuple[str, str]]:
    keys = names[["competition", "source_id"]].drop_duplicates()
    return sorted((str(r.competition), str(r.source_id)) for r in keys.itertuples(index=False))


def _override_match_edges(
    overrides: pd.DataFrame,
) -> list[tuple[tuple[str, str], tuple[str, str]]]:
    edges: list[tuple[tuple[str, str], tuple[str, str]]] = []
    if overrides.empty:
        return edges
    for row in overrides.itertuples(index=False):
        if str(row.decision) != "match":
            continue
        a = (str(row.competition_a), str(row.source_a))
        b = (str(row.competition_b), str(row.source_b))
        edges.append((a, b))
    return edges


def _link_edges(links: pd.DataFrame) -> list[tuple[tuple[str, str], tuple[str, str]]]:
    edges: list[tuple[tuple[str, str], tuple[str, str]]] = []
    if links.empty:
        return edges
    for row in links.itertuples(index=False):
        comp = str(row.competition)
        a = (comp, str(row.source_id))
        b = (comp, str(row.linked_source_id))
        edges.append((a, b))
    return edges


def _rule_edges(matches: pd.DataFrame) -> list[tuple[tuple[str, str], tuple[str, str]]]:
    edges: list[tuple[tuple[str, str], tuple[str, str]]] = []
    if matches.empty:
        return edges
    for row in matches.itertuples(index=False):
        edges.append((("gbl", str(row.gbl_id)), ("euroleague", str(row.el_id))))
    return edges


def _person_id(members: list[tuple[str, str]]) -> str:
    el = sorted(sid for comp, sid in members if comp == "euroleague")
    if el:
        return f"P:{el[0]}"
    gbl = sorted(sid for comp, sid in members if comp == "gbl")
    return f"G:{gbl[0]}"


def _connected(
    nodes: set[tuple[str, str]], edges: list[tuple[tuple[str, str], tuple[str, str]]]
) -> bool:
    """Whether ``nodes`` are in one component using only ``edges`` (paths may leave the set)."""
    if len(nodes) <= 1:
        return True
    adj: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    start = next(iter(nodes))
    seen: set[tuple[str, str]] = {start}
    stack = [start]
    while stack:
        cur = stack.pop()
        for nbr in adj[cur]:
            if nbr not in seen:
                seen.add(nbr)
                stack.append(nbr)
    return nodes <= seen


def _check_same_competition(
    groups: dict[tuple[str, str], list[tuple[str, str]]],
    soft_edges: list[tuple[tuple[str, str], tuple[str, str]]],
) -> None:
    """Raise if a person holds two same-competition ids without an override/link path."""
    for members in groups.values():
        by_comp: dict[str, list[str]] = defaultdict(list)
        for comp, sid in members:
            by_comp[comp].append(sid)
        for comp, ids in by_comp.items():
            if len(ids) < 2:
                continue
            node_set = {(comp, sid) for sid in ids}
            if not _connected(node_set, soft_edges):
                raise ValueError(
                    f"person holds multiple {comp} ids without override/link: {sorted(ids)}"
                )


def _method_sets(
    matches: pd.DataFrame,
    overrides: pd.DataFrame,
    links: pd.DataFrame,
) -> tuple[
    set[tuple[str, str]], set[tuple[str, str]], set[tuple[str, str]], dict[tuple[str, str], float]
]:
    override_nodes: set[tuple[str, str]] = set()
    for a, b in _override_match_edges(overrides):
        override_nodes.add(a)
        override_nodes.add(b)
    link_nodes: set[tuple[str, str]] = set()
    for a, b in _link_edges(links):
        link_nodes.add(a)
        link_nodes.add(b)
    rule_nodes: set[tuple[str, str]] = set()
    rule_score: dict[tuple[str, str], float] = {}
    if not matches.empty:
        for row in matches.itertuples(index=False):
            g = ("gbl", str(row.gbl_id))
            e = ("euroleague", str(row.el_id))
            rule_nodes.add(g)
            rule_nodes.add(e)
            score = float(cast(Any, row.score))
            rule_score[g] = score
            rule_score[e] = score
    return override_nodes, link_nodes, rule_nodes, rule_score


def _row_method(
    node: tuple[str, str],
    override_nodes: set[tuple[str, str]],
    link_nodes: set[tuple[str, str]],
    rule_nodes: set[tuple[str, str]],
) -> str:
    if node in override_nodes:
        return "override"
    if node in link_nodes:
        return "pbp"
    if node in rule_nodes:
        return "rule"
    return "none"


def build_xwalk(
    names: pd.DataFrame,
    matches: pd.DataFrame,
    overrides: pd.DataFrame,
    links: pd.DataFrame,
) -> pd.DataFrame:
    """Map every name-table source id to one person; validated against ``XWALK_SCHEMA``."""
    nodes = _nodes_from_names(names)
    uf = _UnionFind()
    for node in nodes:
        uf.add(node)

    rule = _rule_edges(matches)
    soft = _override_match_edges(overrides) + _link_edges(links)
    for a, b in rule + soft:
        uf.union(a, b)

    groups: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    for node in nodes:
        groups[uf.find(node)].append(node)
    _check_same_competition(groups, soft)

    override_nodes, link_nodes, rule_nodes, rule_score = _method_sets(matches, overrides, links)
    rows: list[dict[str, object]] = []
    for members in groups.values():
        pid = _person_id(members)
        for comp, sid in sorted(members):
            node = (comp, sid)
            rows.append(
                {
                    "person_id": pid,
                    "competition": comp,
                    "source_id": sid,
                    "method": _row_method(node, override_nodes, link_nodes, rule_nodes),
                    "score": rule_score.get(node, float("nan")),
                }
            )
    frame = pd.DataFrame(rows, columns=list(XWALK_SCHEMA.columns))
    frame = frame.sort_values(["person_id", "competition", "source_id"]).reset_index(drop=True)
    return validated(frame, XWALK_SCHEMA)


def entity_report(
    names: pd.DataFrame,
    scored: pd.DataFrame,
    xwalk: pd.DataFrame,
    params: MatchParams,
) -> dict[str, Any]:
    """JSON-ready summary counts for ``reports/entity_resolution.json``."""
    by_method: dict[str, int] = {m: int((xwalk["method"] == m).sum()) for m in METHODS}
    by_comp: dict[str, int] = {
        c: int((xwalk["competition"] == c).sum()) for c in ("euroleague", "gbl")
    }
    persons_with_el = set(xwalk.loc[xwalk["competition"] == "euroleague", "person_id"])
    matched_gbl = int(
        xwalk.loc[
            (xwalk["competition"] == "gbl") & xwalk["person_id"].isin(persons_with_el),
            "source_id",
        ].nunique()
    )
    return {
        "by_method": by_method,
        "by_competition": by_comp,
        "candidate_pairs": len(scored),
        "accepted_pairs": int(scored["accepted"].sum()) if not scored.empty else 0,
        "matched_gbl_ids": matched_gbl,
        "persons": int(xwalk["person_id"].nunique()),
        "name_rows": len(names),
        "params": asdict(params),
    }


def pair_metrics(
    predicted: set[tuple[str, str]],
    truth_pos: set[tuple[str, str]],
    truth_neg: set[tuple[str, str]],
) -> dict[str, float | int]:
    """Precision/recall/F1 on labelled pairs; predictions outside the truth set are ignored."""
    evaluated = truth_pos | truth_neg
    pred = predicted & evaluated
    tp = len(pred & truth_pos)
    fp = len(pred & truth_neg)
    fn = len(truth_pos - predicted)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion ``k/n``."""
    if n <= 0:
        return (0.0, 1.0)
    phat = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = phat + z2 / (2.0 * n)
    margin = z * sqrt((phat * (1.0 - phat) + z2 / (4.0 * n)) / n)
    return ((centre - margin) / denom, (centre + margin) / denom)
