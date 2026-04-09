"""Heuristic trimming of thesis PDFs: keep pages before references / appendix (pypdf)."""

from __future__ import annotations

import csv
import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Literal

from pypdf import PdfReader, PdfWriter

logger = logging.getLogger(__name__)

# Lines at the top of a page scanned for section-heading-style back matter (EN + DA).
_MAX_HEADING_SCAN_LINES = 25

# Compiled pattern: line looks like a standalone section title (optional chapter number).
_BACK_MATTER_LINE = re.compile(
    r"""
    ^\s*
    (?:\d+[\.\)]\s+)?                    # "7 " or "7."
    (
        references?|bibliography|literatur|litteratur|bibliografi|referencer|
        works\s+cited|kilder|kildeliste|
        appendix(?:\s+[a-z0-9]+)?|appendiks|appendices|
        bilag(?:\s+[a-zæøåA-ZÆØÅ0-9.\-]+)?|
        curriculum\s+vitae
    )
    \s*\.?\s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)


class FallbackPolicy(str, Enum):
    """When no back-matter heading is found."""

    KEEP_ALL = "keep_all"
    KEEP_FIRST_FRACTION = "keep_first_fraction"


@dataclass
class ProcessResult:
    """One input PDF outcome."""

    source_path: Path
    pages_total: int
    pages_kept: int
    back_matter_start_0based: int | None
    reason: str
    notes: str = ""
    output_path: Path | None = None


def _first_lines(text: str, max_lines: int) -> list[str]:
    lines: list[str] = []
    for raw in text.splitlines():
        s = raw.strip()
        if not s:
            continue
        lines.append(s)
        if len(lines) >= max_lines:
            break
    return lines


def page_looks_like_back_matter_start(
    page_text: str,
    *,
    max_heading_lines: int = _MAX_HEADING_SCAN_LINES,
) -> bool:
    """True if the top of the page looks like a references/appendix section heading."""
    for line in _first_lines(page_text, max_heading_lines):
        if _BACK_MATTER_LINE.match(line):
            return True
        # Short standalone "Bilag" / "Appendix" lines (some theses omit letters).
        if len(line) <= 40 and re.match(
            r"^(bilag|appendix|appendiks)\s*\.?\s*$", line, re.IGNORECASE
        ):
            return True
    return False


def find_back_matter_start_page(
    page_texts: list[str],
    *,
    min_page_fraction: float = 0.45,
) -> int | None:
    """
    Return the 0-based index of the first page that appears to start back matter,
    or None if no candidate is found.

    Scan begins at ``max(0, int(n * min_page_fraction))`` to reduce false positives
    in the introduction.
    """
    n = len(page_texts)
    if n == 0:
        return None
    start = max(0, int(n * min_page_fraction))
    for i in range(start, n):
        if page_looks_like_back_matter_start(page_texts[i]):
            return i
    return None


def resolve_end_exclusive(
    total_pages: int,
    back_matter_start: int | None,
    *,
    policy: FallbackPolicy = FallbackPolicy.KEEP_ALL,
    fallback_fraction: float = 0.95,
) -> tuple[int, str]:
    """
    Return ``(end_exclusive, reason)`` where pages kept are ``0 .. end_exclusive - 1``.

    If ``back_matter_start`` is set, end is that index (drop from references onward).
    If None, apply ``policy`` (default: keep all pages).
    """
    if total_pages <= 0:
        return 0, "empty_pdf"
    if back_matter_start is not None:
        end = max(0, min(back_matter_start, total_pages))
        return end, "matched_heading"
    if policy is FallbackPolicy.KEEP_ALL:
        return total_pages, "no_match_keep_all"
    frac = max(0.0, min(1.0, fallback_fraction))
    end = max(1, int(total_pages * frac))
    return min(end, total_pages), "no_match_fallback_fraction"


def build_context_writer(reader: PdfReader, end_exclusive: int) -> PdfWriter:
    """Build a writer containing pages ``[0, end_exclusive)``."""
    writer = PdfWriter()
    n = min(end_exclusive, len(reader.pages))
    for i in range(n):
        writer.add_page(reader.pages[i])
    return writer


def extract_page_texts(reader: PdfReader) -> list[str]:
    """Extract raw text per page (empty string if none)."""
    return [page.extract_text() or "" for page in reader.pages]


def process_one_pdf(
    pdf_path: Path,
    output_path: Path | None,
    *,
    min_page_fraction: float = 0.45,
    policy: FallbackPolicy = FallbackPolicy.KEEP_ALL,
    fallback_fraction: float = 0.95,
    write_pdf: bool = True,
) -> ProcessResult:
    """
    Read one PDF, trim back matter heuristically, optionally write the result.

    If ``output_path`` is None and ``write_pdf`` is True, no file is written but
    counts are still computed.
    """
    reader = PdfReader(str(pdf_path))
    texts = extract_page_texts(reader)
    total = len(reader.pages)
    cut = find_back_matter_start_page(texts, min_page_fraction=min_page_fraction)
    end_exc, reason = resolve_end_exclusive(
        total,
        cut,
        policy=policy,
        fallback_fraction=fallback_fraction,
    )
    notes = ""
    if cut is None and not texts:
        notes = "no_text_extracted_maybe_scanned"
    elif cut is None and total > 0:
        sample_len = sum(len(t) for t in texts)
        if sample_len < 200:
            notes = "very_little_text_extracted_check_layout_or_ocr"

    out: Path | None = None
    if write_pdf and output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = build_context_writer(reader, end_exc)
        with output_path.open("wb") as f:
            writer.write(f)
        out = output_path

    return ProcessResult(
        source_path=pdf_path,
        pages_total=total,
        pages_kept=end_exc,
        back_matter_start_0based=cut,
        reason=reason,
        notes=notes,
        output_path=out,
    )


def process_folder(
    input_dir: Path,
    output_dir: Path,
    manifest_path: Path,
    *,
    pattern: str = "*.pdf",
    recursive: bool = False,
    manifest_format: Literal["csv", "json"] = "csv",
    min_page_fraction: float = 0.45,
    policy: FallbackPolicy = FallbackPolicy.KEEP_ALL,
    fallback_fraction: float = 0.95,
) -> list[ProcessResult]:
    """
    Process all PDFs under ``input_dir``, write trimmed copies to ``output_dir``,
    and write a manifest (CSV or JSON array) to ``manifest_path``.
    """
    if recursive:
        paths = sorted(input_dir.rglob(pattern))
    else:
        paths = sorted(input_dir.glob(pattern))
    results: list[ProcessResult] = []
    for src in paths:
        if not src.is_file():
            continue
        rel = src.relative_to(input_dir)
        dst = output_dir / rel
        r = process_one_pdf(
            src,
            dst,
            min_page_fraction=min_page_fraction,
            policy=policy,
            fallback_fraction=fallback_fraction,
            write_pdf=True,
        )
        results.append(r)
        logger.info(
            "%s: kept %s/%s pages (%s)",
            src.name,
            r.pages_kept,
            r.pages_total,
            r.reason,
        )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_format == "csv":
        _write_manifest_csv(manifest_path, results)
    else:
        _write_manifest_json(manifest_path, results)
    return results


def _result_to_row(r: ProcessResult) -> dict[str, str | int | None]:
    return {
        "filename": str(r.source_path),
        "pages_total": r.pages_total,
        "pages_kept": r.pages_kept,
        "back_matter_start_0based": r.back_matter_start_0based,
        "reason": r.reason,
        "notes": r.notes,
        "output_path": str(r.output_path) if r.output_path else "",
    }


def _write_manifest_csv(path: Path, results: list[ProcessResult]) -> None:
    fieldnames = [
        "filename",
        "pages_total",
        "pages_kept",
        "back_matter_start_0based",
        "reason",
        "notes",
        "output_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            row = _result_to_row(r)
            w.writerow({k: "" if v is None else v for k, v in row.items()})


def _write_manifest_json(path: Path, results: list[ProcessResult]) -> None:
    payload = [_result_to_row(r) for r in results]
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")

