"""Jaro-Winkler string similarity for entity name matching (weeks 12-14 I-d / I2).

Pure Python, no third-party deps. Prefix scale ``p = 0.1``, max common prefix 4,
applied always (no Jaro boost threshold). Empty vs empty is 1.0; empty vs non-empty is 0.0.
"""

from __future__ import annotations

_PREFIX_SCALE = 0.1
_MAX_PREFIX = 4


def jaro_winkler(a: str, b: str) -> float:
    """Return the Jaro-Winkler similarity of ``a`` and ``b`` in ``[0, 1]``."""
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0

    jaro = _jaro(a, b)
    prefix = 0
    limit = min(_MAX_PREFIX, len(a), len(b))
    while prefix < limit and a[prefix] == b[prefix]:
        prefix += 1
    return jaro + prefix * _PREFIX_SCALE * (1.0 - jaro)


def _jaro(a: str, b: str) -> float:
    len_a = len(a)
    len_b = len(b)
    match_window = max(max(len_a, len_b) // 2 - 1, 0)

    a_flags = [False] * len_a
    b_flags = [False] * len_b
    matches = 0

    for i in range(len_a):
        start = i - match_window if i > match_window else 0
        end = min(i + match_window + 1, len_b)
        for j in range(start, end):
            if b_flags[j] or a[i] != b[j]:
                continue
            a_flags[i] = True
            b_flags[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    k = 0
    transpositions = 0
    for i in range(len_a):
        if not a_flags[i]:
            continue
        while not b_flags[k]:
            k += 1
        if a[i] != b[k]:
            transpositions += 1
        k += 1

    m = float(matches)
    return (m / len_a + m / len_b + (m - transpositions / 2.0) / m) / 3.0
