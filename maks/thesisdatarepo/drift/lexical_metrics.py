"""MTLD and hapax ratio from token lists (no spaCy)."""

from __future__ import annotations

from collections import Counter


def hapax_ratio(tokens: list[str]) -> float:
    """Share of token *positions* whose type occurs exactly once in the document."""
    if not tokens:
        return 0.0
    c = Counter(tokens)
    once_positions = sum(1 for t in tokens if c[t] == 1)
    return once_positions / len(tokens)


def mtld_forward(tokens: list[str], *, ttr_threshold: float = 0.72, min_seg: int = 10) -> float:
    """
    Forward MTLD-style length (McCarthy & Jarvis): accumulate factors when running
    TTR in the segment drops at or below ``ttr_threshold``. Returns 0 for very short texts.
    """
    n = len(tokens)
    if n < min_seg:
        return 0.0
    factors = 0.0
    idx = 0
    while idx < n:
        seen: set[str] = set()
        for j in range(idx, n):
            seen.add(tokens[j])
            size = j - idx + 1
            if size >= min_seg and len(seen) / size <= ttr_threshold:
                factors += 1.0
                idx = j + 1
                break
        else:
            tail = tokens[idx:]
            if tail:
                tl = len(tail)
                uniq = len(set(tail))
                factors += max(uniq / (ttr_threshold * max(tl, 1)), 1e-6)
            break
    return n / factors if factors else float(n)


def lexical_for_tokens(tokens: list[str]) -> tuple[float, float, float]:
    """Returns (mtld, hapax_ratio, ttr)."""
    if not tokens:
        return 0.0, 0.0, 0.0
    ttr = len(set(tokens)) / len(tokens)
    mt = mtld_forward(tokens) if len(tokens) > 500 else ttr
    hx = hapax_ratio(tokens)
    return float(mt), float(hx), float(ttr)
