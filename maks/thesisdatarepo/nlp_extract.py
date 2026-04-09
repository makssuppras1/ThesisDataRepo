"""Extract UTF-8 thesis text from Abstract through content before references (NLP-friendly)."""

from __future__ import annotations

import csv
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pypdf import PdfReader

from thesisdatarepo.pdf_context import (
    FallbackPolicy,
    extract_page_texts,
    find_back_matter_start_page,
    resolve_end_exclusive,
)

logger = logging.getLogger(__name__)

_HEADING_LINES = 25


def configure_batch_logging(level: int = logging.INFO) -> None:
    """If the root logger has no handlers, configure stderr (so INFO shows in notebooks/CLI)."""
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=level, format="%(levelname)s %(message)s")

# Section title at top of page: Abstract (EN + DA + common variants).
_ABSTRACT_LINE = re.compile(
    r"""
    ^\s*
    (?:\d+[\.\)]\s+|\d+\s+)?
    (
        abstract|resum[ée]|sammenfatning|executive\s+summary|synopsis|
        summary
    )
    \s*\.?\s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _first_non_empty_lines(text: str, max_lines: int) -> list[str]:
    lines: list[str] = []
    for raw in text.splitlines():
        s = raw.strip()
        if not s:
            continue
        lines.append(s)
        if len(lines) >= max_lines:
            break
    return lines


def page_looks_like_abstract_start(
    page_text: str,
    *,
    max_heading_lines: int = _HEADING_LINES,
) -> bool:
    """True if a line in the page header looks like an Abstract section title."""
    for line in _first_non_empty_lines(page_text, max_heading_lines):
        if _ABSTRACT_LINE.match(line):
            return True
    return False


def find_abstract_start_page(page_texts: list[str]) -> int | None:
    """First page index (from the start of the document) that looks like Abstract, or None."""
    for i, t in enumerate(page_texts):
        if page_looks_like_abstract_start(t):
            return i
    return None


def normalize_nlp_text(page_chunks: list[str]) -> str:
    """Join non-empty page texts with blank lines; trim and collapse excessive newlines."""
    parts: list[str] = []
    for raw in page_chunks:
        t = (raw or "").strip()
        if t:
            parts.append(t)
    body = "\n\n".join(parts)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body.strip()


@dataclass
class NlpExtractResult:
    """One PDF text extraction outcome."""

    source: str
    output_txt_path: Path | None
    pages_total: int
    page_start_0based: int
    page_end_exclusive: int
    abstract_found: bool
    back_matter_found: bool
    end_reason: str
    notes: str
    char_count: int
    skipped_empty: bool = False


def extract_nlp_page_range(
    page_texts: list[str],
    *,
    min_page_fraction_back_matter: float = 0.45,
    policy: FallbackPolicy = FallbackPolicy.KEEP_ALL,
    fallback_fraction: float = 0.95,
) -> tuple[int, int, bool, bool, str, str]:
    """
    Return ``(start, end_exclusive, abstract_found, back_matter_found, end_reason, notes)``.

    Start defaults to first Abstract page, or 0 if not found.
    End is before first references/appendix-style page when detected; else policy.
    """
    n = len(page_texts)
    if n == 0:
        return 0, 0, False, False, "empty_pdf", ""

    abs_idx = find_abstract_start_page(page_texts)
    start = abs_idx if abs_idx is not None else 0
    abstract_found = abs_idx is not None

    cut = find_back_matter_start_page(
        page_texts, min_page_fraction=min_page_fraction_back_matter
    )
    back_matter_found = cut is not None

    end_exc, end_reason = resolve_end_exclusive(
        n,
        cut,
        policy=policy,
        fallback_fraction=fallback_fraction,
    )

    notes_list: list[str] = []
    if not abstract_found:
        notes_list.append("abstract_not_found_start_at_page_zero")

    if start >= end_exc:
        notes_list.append("empty_range_start_ge_end")
        # Still return a valid empty slice [start, start)
        return start, start, abstract_found, back_matter_found, end_reason, "; ".join(
            notes_list
        )

    return start, end_exc, abstract_found, back_matter_found, end_reason, "; ".join(
        notes_list
    )


def process_pdf_nlp_reader(
    reader: PdfReader,
    source: str,
    output_txt_path: Path | None,
    *,
    min_page_fraction_back_matter: float = 0.45,
    policy: FallbackPolicy = FallbackPolicy.KEEP_ALL,
    fallback_fraction: float = 0.95,
    write_txt: bool = True,
    skip_if_empty: bool = False,
) -> NlpExtractResult:
    """
    Extract normalized text from a ``PdfReader`` (local path, bytes stream, GCS, etc.).

    ``source`` is only used for logging and manifests (e.g. path or ``gs://...`` URI).

    Writes UTF-8 ``.txt`` when ``write_txt`` and ``output_txt_path`` are set.
    If ``skip_if_empty`` is True and the normalized body has length 0, no file is written
    and ``skipped_empty`` is set on the result.
    """
    texts = extract_page_texts(reader)
    total = len(reader.pages)

    start, end_exc, abstract_found, bm_found, end_reason, note_extra = (
        extract_nlp_page_range(
            texts,
            min_page_fraction_back_matter=min_page_fraction_back_matter,
            policy=policy,
            fallback_fraction=fallback_fraction,
        )
    )

    chunks = texts[start:end_exc]
    body = normalize_nlp_text(chunks)
    char_count = len(body)

    notes = note_extra
    if not texts:
        notes = "no_text_extracted_maybe_scanned"
    elif total > 0 and sum(len(t) for t in texts) < 200:
        notes = (
            notes + "; very_little_text_extracted_check_layout_or_ocr"
            if notes
            else "very_little_text_extracted_check_layout_or_ocr"
        )

    skipped_empty = bool(skip_if_empty and char_count == 0)
    if skipped_empty:
        notes = (
            (notes + "; skipped_empty_no_output_file")
            if notes
            else "skipped_empty_no_output_file"
        )

    out: Path | None = None
    if (
        write_txt
        and output_txt_path is not None
        and not skipped_empty
    ):
        output_txt_path.parent.mkdir(parents=True, exist_ok=True)
        output_txt_path.write_text(body, encoding="utf-8")
        out = output_txt_path

    return NlpExtractResult(
        source=source,
        output_txt_path=out,
        pages_total=total,
        page_start_0based=start,
        page_end_exclusive=end_exc,
        abstract_found=abstract_found,
        back_matter_found=bm_found,
        end_reason=end_reason,
        notes=notes,
        char_count=char_count,
        skipped_empty=skipped_empty,
    )


def process_one_pdf_nlp(
    pdf_path: Path,
    output_txt_path: Path | None,
    *,
    min_page_fraction_back_matter: float = 0.45,
    policy: FallbackPolicy = FallbackPolicy.KEEP_ALL,
    fallback_fraction: float = 0.95,
    write_txt: bool = True,
    skip_if_empty: bool = False,
) -> NlpExtractResult:
    """Load a PDF from disk and run :func:`process_pdf_nlp_reader`."""
    reader = PdfReader(str(pdf_path))
    return process_pdf_nlp_reader(
        reader,
        str(pdf_path.resolve()),
        output_txt_path,
        min_page_fraction_back_matter=min_page_fraction_back_matter,
        policy=policy,
        fallback_fraction=fallback_fraction,
        write_txt=write_txt,
        skip_if_empty=skip_if_empty,
    )


def _nlp_result_to_row(r: NlpExtractResult) -> dict[str, str | int | bool]:
    return {
        "source": r.source,
        "pages_total": r.pages_total,
        "page_start_0based": r.page_start_0based,
        "page_end_exclusive": r.page_end_exclusive,
        "abstract_found": r.abstract_found,
        "back_matter_found": r.back_matter_found,
        "end_reason": r.end_reason,
        "char_count": r.char_count,
        "skipped_empty": r.skipped_empty,
        "notes": r.notes,
        "output_txt_path": str(r.output_txt_path) if r.output_txt_path else "",
    }


def _write_nlp_manifest_csv(path: Path, results: list[NlpExtractResult]) -> None:
    fieldnames = [
        "source",
        "pages_total",
        "page_start_0based",
        "page_end_exclusive",
        "abstract_found",
        "back_matter_found",
        "end_reason",
        "char_count",
        "skipped_empty",
        "notes",
        "output_txt_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            row = _nlp_result_to_row(r)
            w.writerow(row)


def _write_nlp_manifest_json(path: Path, results: list[NlpExtractResult]) -> None:
    payload = [_nlp_result_to_row(r) for r in results]
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _append_jsonl(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def process_folder_nlp(
    input_dir: Path,
    output_txt_dir: Path,
    manifest_path: Path,
    *,
    pattern: str = "*.pdf",
    recursive: bool = False,
    manifest_format: Literal["csv", "json"] = "csv",
    jsonl_path: Path | None = None,
    min_page_fraction_back_matter: float = 0.45,
    policy: FallbackPolicy = FallbackPolicy.KEEP_ALL,
    fallback_fraction: float = 0.95,
    clear_jsonl: bool = True,
    skip_if_empty: bool = True,
) -> list[NlpExtractResult]:
    """
    For each PDF under ``input_dir``, write ``<stem>.txt`` under ``output_txt_dir``,
    a manifest to ``manifest_path``, and optionally one JSON object per line to ``jsonl_path``.

    Logs progress as ``[i/N]`` and a final summary. Empty extractions (0 characters) are
    skipped (no ``.txt`` / JSONL line) when ``skip_if_empty`` is True.
    """
    if recursive:
        paths = sorted(p for p in input_dir.rglob(pattern) if p.is_file())
    else:
        paths = sorted(p for p in input_dir.glob(pattern) if p.is_file())

    configure_batch_logging()
    total = len(paths)
    logger.info("NLP folder: %s PDF(s) to process under %s", total, input_dir)

    if jsonl_path is not None and clear_jsonl and jsonl_path.is_file():
        jsonl_path.unlink()

    results: list[NlpExtractResult] = []
    converted = 0
    skipped = 0
    for i, src in enumerate(paths, start=1):
        rel_parent = src.parent.relative_to(input_dir) if src.parent != input_dir else Path()
        dst_dir = output_txt_dir / rel_parent
        dst = dst_dir / f"{src.stem}.txt"
        r = process_one_pdf_nlp(
            src,
            dst,
            min_page_fraction_back_matter=min_page_fraction_back_matter,
            policy=policy,
            fallback_fraction=fallback_fraction,
            write_txt=True,
            skip_if_empty=skip_if_empty,
        )
        results.append(r)
        if r.skipped_empty:
            skipped += 1
            logger.info(
                "[%s/%s] skipped (empty text): %s",
                i,
                total,
                src.name,
            )
        else:
            converted += 1
            logger.info(
                "[%s/%s] converted %s -> %s chars (pages [%s,%s))",
                i,
                total,
                src.name,
                r.char_count,
                r.page_start_0based,
                r.page_end_exclusive,
            )

        if jsonl_path is not None and r.output_txt_path is not None:
            text_body = r.output_txt_path.read_text(encoding="utf-8")
            meta = {
                "source": r.source,
                "pages_total": r.pages_total,
                "page_start_0based": r.page_start_0based,
                "page_end_exclusive": r.page_end_exclusive,
                "abstract_found": r.abstract_found,
                "back_matter_found": r.back_matter_found,
                "end_reason": r.end_reason,
                "char_count": r.char_count,
                "skipped_empty": r.skipped_empty,
                "notes": r.notes,
            }
            _append_jsonl(
                jsonl_path,
                {"id": src.stem, "text": text_body, "meta": meta},
            )

    logger.info(
        "NLP folder done: %s converted, %s skipped (empty), %s total",
        converted,
        skipped,
        total,
    )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_format == "csv":
        _write_nlp_manifest_csv(manifest_path, results)
    else:
        _write_nlp_manifest_json(manifest_path, results)
    return results
