"""Lightweight text normalization (references strip; simple sentence split)."""

from __future__ import annotations

import re

_REFERENCES_BLOCK = re.compile(r"(?is)^\s*References\s*\n.*$")


def strip_references_section(text: str) -> str:
    """Remove bibliography / references block when clearly delimited."""
    if not text:
        return ""
    return _REFERENCES_BLOCK.sub("", text).strip()


def simple_sentence_split(text: str) -> list[str]:
    """Rough sentence boundaries (spaCy can replace this later)."""
    if not text.strip():
        return []
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


def word_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower(), flags=re.IGNORECASE)
