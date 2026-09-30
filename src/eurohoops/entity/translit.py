"""Greek↔Latin name folding and reverse-phonetic variants (weeks 12-14 I-d / I2).

``fold`` uppercases, strips accents/diaeresis/tonos, maps punctuation to spaces, and per token
folds minority-script look-alike letters into the majority script (ties → Greek).

``variants`` yields up to 64 distinct Latin spellings of a folded name: ELOT 743 first, then
reverse-phonetic alternatives for foreign names written in Greek (ΜΠ/ΝΤ/ΓΚ, ΟΥ→W, Ι→J before
a vowel, …). Each spelling is the Latin tokens sorted and joined by one space.

``latin_key`` is the EuroLeague-side normaliser: fold, keep A–Z and spaces, sort tokens.
"""

from __future__ import annotations

import unicodedata
from functools import lru_cache

_MAX_VARIANTS = 64
_BEAM_WIDTH = 256

# Latin ↔ Greek look-alikes (uppercase only; used after case fold).
_LATIN_TO_GREEK = str.maketrans(
    {
        "A": "Α",
        "B": "Β",
        "E": "Ε",
        "Z": "Ζ",
        "H": "Η",
        "I": "Ι",
        "K": "Κ",
        "M": "Μ",
        "N": "Ν",
        "O": "Ο",
        "P": "Ρ",
        "T": "Τ",
        "Y": "Υ",
        "X": "Χ",
    }
)
_GREEK_TO_LATIN = str.maketrans(
    {
        "Α": "A",
        "Β": "B",
        "Ε": "E",
        "Ζ": "Z",
        "Η": "H",
        "Ι": "I",
        "Κ": "K",
        "Μ": "M",
        "Ν": "N",
        "Ο": "O",
        "Ρ": "P",
        "Τ": "T",
        "Υ": "Y",
        "Χ": "X",
    }
)

_GREEK_LETTERS = frozenset("ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ")
_LATIN_LETTERS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
_VOWELS_GREEK = frozenset("ΑΕΗΙΟΥΩ")
_SOFT_AFTER_GAMMA = frozenset("ΙΕΗΥ")

_PUNCT_TO_SPACE = str.maketrans({c: " " for c in "-'’.,"})

# Digraph / trigraph units: (greek, start_only | None, options with ELOT first).
# start_only True → only at word start; False → only inside; None → anywhere.
_MULTI_UNITS: list[tuple[str, bool | None, list[tuple[str, float]]]] = [
    ("ΟΥΑ", True, [("WA", 1.0), ("OUA", 0.6), ("UWA", 0.4)]),
    ("ΟΥΕ", True, [("WE", 1.0), ("OUE", 0.6), ("UWE", 0.4)]),
    ("ΟΥΙ", True, [("WI", 1.0), ("OUI", 0.6), ("UWI", 0.4)]),
    ("ΑΟΥ", None, [("OW", 1.0), ("AOU", 0.7), ("AU", 0.5), ("AW", 0.5), ("OU", 0.4)]),
    ("ΜΠ", True, [("B", 1.0), ("MP", 0.4)]),
    ("ΜΠ", False, [("B", 1.0), ("MB", 0.8), ("MP", 0.4)]),
    ("ΝΤ", True, [("D", 1.0), ("NT", 0.7), ("ND", 0.5)]),
    ("ΝΤ", False, [("NT", 1.0), ("D", 0.8), ("ND", 0.7)]),
    ("ΓΚ", True, [("G", 1.0), ("GK", 0.4)]),
    ("ΓΚ", False, [("G", 1.0), ("NG", 0.8), ("GK", 0.4)]),
    ("ΓΓ", None, [("NG", 1.0), ("G", 0.6)]),
    ("ΤΖ", None, [("TZ", 1.0), ("J", 0.9), ("DZ", 0.6)]),
    ("ΤΣ", None, [("TS", 1.0), ("CH", 0.7), ("C", 0.5), ("TZ", 0.4)]),
    ("ΓΙ", None, [("GI", 1.0), ("Y", 0.9), ("J", 0.7)]),
    ("ΟΥ", None, [("OU", 1.0), ("U", 0.8), ("W", 0.7), ("OO", 0.5)]),
    ("ΑΙ", None, [("AI", 1.0), ("E", 0.8), ("AY", 0.6)]),
    # ΕΙ→A covers foreign spellings like ΓΚΕΪΜΠΡΙΕΛ → GABRIEL after Ϊ→Ι fold.
    ("ΕΙ", None, [("EI", 1.0), ("I", 0.9), ("A", 0.85), ("EY", 0.6), ("AY", 0.5), ("EE", 0.5)]),
    ("ΟΙ", None, [("OI", 1.0), ("I", 0.8), ("OY", 0.6)]),
    ("ΑΥ", None, [("AV", 1.0), ("AF", 0.8), ("AU", 0.6), ("AW", 0.5)]),
    ("ΕΥ", None, [("EV", 1.0), ("EF", 0.8), ("EU", 0.6)]),
]

_DOUBLES = {
    "ΛΛ": "L",
    "ΝΝ": "N",
    "ΠΠ": "P",
    "ΣΣ": "S",
    "ΤΤ": "T",
    "ΚΚ": "K",
    "ΜΜ": "M",
    "ΡΡ": "R",
}

# Context-free single-letter ELOT-first options.
_SINGLE: dict[str, list[tuple[str, float]]] = {
    "Α": [("A", 1.0)],
    "Β": [("V", 1.0), ("B", 0.7), ("W", 0.4)],
    "Δ": [("D", 1.0), ("TH", 0.5)],
    "Ε": [("E", 1.0), ("A", 0.7)],
    "Ζ": [("Z", 1.0)],
    "Η": [("I", 1.0), ("E", 0.7), ("EE", 0.4)],
    "Θ": [("TH", 1.0), ("T", 0.5)],
    "Κ": [("K", 1.0), ("C", 0.7), ("CK", 0.4), ("Q", 0.3)],
    "Λ": [("L", 1.0)],
    "Μ": [("M", 1.0)],
    "Ξ": [("X", 1.0), ("KS", 0.5)],
    "Ο": [("O", 1.0), ("OO", 0.4)],
    "Π": [("P", 1.0)],
    "Ρ": [("R", 1.0)],
    "Σ": [("S", 1.0), ("SS", 0.5), ("Z", 0.4), ("C", 0.35)],
    "Τ": [("T", 1.0)],
    "Υ": [("Y", 1.0), ("I", 0.7)],
    "Φ": [("F", 1.0), ("PH", 0.7)],
    "Χ": [("CH", 1.0), ("H", 0.85), ("KH", 0.5)],
    "Ψ": [("PS", 1.0)],
    "Ω": [("O", 1.0)],
}


def fold(text: str) -> str:
    """Uppercase, strip accents, fold look-alikes per-token, normalise punctuation/spaces."""
    if not text:
        return ""
    upper = text.upper().replace("ς", "Σ").replace("Ϲ", "Σ")
    decomposed = unicodedata.normalize("NFD", upper)
    stripped = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    spaced = stripped.translate(_PUNCT_TO_SPACE)
    tokens = spaced.split()
    return " ".join(_fold_token(tok) for tok in tokens)


def latin_key(text: str) -> str:
    """Fold ``text``, keep A–Z and spaces, sort tokens, join with one space."""
    folded = fold(text)
    cleaned = "".join(ch if ch in _LATIN_LETTERS or ch == " " else "" for ch in folded)
    tokens = [t for t in cleaned.split() if t]
    return " ".join(sorted(tokens))


def variants(name: str) -> tuple[str, ...]:
    """Up to 64 distinct Latin spellings of ``name`` (sorted tokens, best weight first)."""
    return _variants_cached(fold(name))


@lru_cache(maxsize=4096)
def _variants_cached(folded: str) -> tuple[str, ...]:
    if not folded:
        return ()
    tokens = folded.split()
    if all(_is_latin_token(t) for t in tokens):
        key = " ".join(sorted(t for t in tokens if t))
        return (key,) if key else ()

    token_beams: list[list[tuple[float, str]]] = []
    for tok in tokens:
        if _is_latin_token(tok):
            token_beams.append([(1.0, tok)])
        elif _has_greek(tok):
            token_beams.append(_beam_token(tok))
        else:
            latin = "".join(ch for ch in tok if ch in _LATIN_LETTERS)
            if latin:
                token_beams.append([(0.5, latin)])

    if not token_beams:
        return ()

    combined = _combine_token_beams(token_beams)
    best: dict[str, float] = {}
    for weight, parts in combined:
        spelling = " ".join(sorted(parts))
        prev = best.get(spelling)
        if prev is None or weight > prev:
            best[spelling] = weight

    ranked = sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))
    return tuple(spelling for spelling, _ in ranked[:_MAX_VARIANTS])


def _is_latin_token(token: str) -> bool:
    letters = [ch for ch in token if ch.isalpha()]
    return bool(letters) and all(ch in _LATIN_LETTERS for ch in letters)


def _has_greek(token: str) -> bool:
    return any(ch in _GREEK_LETTERS for ch in token)


def _fold_token(token: str) -> str:
    # Count by Unicode block so a Latin look-alike inside a Greek name stays Latin until mapped.
    greek_n = sum(1 for ch in token if "\u0370" <= ch <= "\u03ff" or "\u1f00" <= ch <= "\u1fff")
    latin_n = sum(1 for ch in token if "A" <= ch <= "Z")
    if greek_n == 0 and latin_n == 0:
        return token
    if latin_n > greek_n:
        return token.translate(_GREEK_TO_LATIN)
    return token.translate(_LATIN_TO_GREEK)


def _beam_token(token: str) -> list[tuple[float, str]]:
    """Beam search over romanisation units; return up to 64 (weight, latin) pairs."""
    beam: list[tuple[float, str, int]] = [(1.0, "", 0)]
    n = len(token)
    while True:
        advanced = False
        nxt: list[tuple[float, str, int]] = []
        for weight, prefix, i in beam:
            if i >= n:
                nxt.append((weight, prefix, i))
                continue
            advanced = True
            for consumed, options in _units_at(token, i):
                for latin, opt_w in options:
                    nxt.append((weight * opt_w, prefix + latin, i + consumed))
        if not advanced:
            break
        nxt.sort(key=lambda t: (-t[0], t[1], t[2]))
        seen: dict[tuple[int, str], float] = {}
        deduped: list[tuple[float, str, int]] = []
        for weight, prefix, i in nxt:
            key = (i, prefix)
            prev = seen.get(key)
            if prev is not None and prev >= weight:
                continue
            seen[key] = weight
            deduped.append((weight, prefix, i))
        deduped.sort(key=lambda t: (-t[0], t[1], t[2]))
        beam = deduped[:_BEAM_WIDTH]
        if all(i >= n for _, _, i in beam):
            break

    finished = [(w, s) for w, s, i in beam if i >= n and s]
    finished.sort(key=lambda t: (-t[0], t[1]))
    out: dict[str, float] = {}
    for w, s in finished:
        if s not in out or w > out[s]:
            out[s] = w
    ranked = sorted(out.items(), key=lambda kv: (-kv[1], kv[0]))
    return [(w, s) for s, w in ranked[:_MAX_VARIANTS]]


def _combine_token_beams(
    token_beams: list[list[tuple[float, str]]],
) -> list[tuple[float, list[str]]]:
    """Cartesian product of per-token beams, capped at 64 by joint weight."""
    combos: list[tuple[float, list[str]]] = [(1.0, [])]
    for beam in token_beams:
        nxt: list[tuple[float, list[str]]] = []
        for cw, parts in combos:
            for tw, latin in beam:
                nxt.append((cw * tw, [*parts, latin]))
        nxt.sort(key=lambda t: (-t[0], " ".join(sorted(t[1]))))
        combos = nxt[:_MAX_VARIANTS]
    return combos


def _units_at(token: str, i: int) -> list[tuple[int, list[tuple[str, float]]]]:
    """Possible (consumed_len, [(latin, weight), …]) starting at ``i``."""
    start = i == 0
    rest = token[i:]
    out: list[tuple[int, list[tuple[str, float]]]] = []

    for greek, start_only, options in _MULTI_UNITS:
        if not rest.startswith(greek):
            continue
        if start_only is True and not start:
            continue
        if start_only is False and start:
            continue
        out.append((len(greek), options))

    for dig, latin in _DOUBLES.items():
        if rest.startswith(dig):
            out.append((2, [(latin * 2, 1.0), (latin, 0.7)]))

    following = token[i + 1] if i + 1 < len(token) else ""
    out.append((1, _single_options(token[i], following, start=start)))
    return out


def _single_options(ch: str, following: str, *, start: bool) -> list[tuple[str, float]]:
    """ELOT-first romanisation options for one Greek letter."""
    if ch == "Γ":
        soft = following in _SOFT_AFTER_GAMMA
        return [("G", 1.0), ("Y", 0.85 if soft else 0.5)]
    if ch == "Ι":
        before_vowel = following in _VOWELS_GREEK
        # Ι→E covers reverse-phonetic foreign spellings (ΠΙΤΕΡΣ → PETERS).
        return (
            [("I", 1.0), ("J", 0.9), ("Y", 0.7), ("E", 0.65)]
            if before_vowel
            else [("I", 1.0), ("Y", 0.7), ("E", 0.65), ("J", 0.4)]
        )
    if ch == "Ν":
        return [("N", 1.0), ("NT", 0.55)] if start else [("N", 1.0)]
    if ch in _SINGLE:
        return _SINGLE[ch]
    if ch in _LATIN_LETTERS:
        return [(ch, 1.0)]
    return [("", 0.1)]
