"""Tests for back-matter detection and trimming (no real PDF files required)."""

from pathlib import Path

import fitz

from thesisdatarepo.pdf_context import (
    FallbackPolicy,
    find_back_matter_start_page,
    page_looks_like_back_matter_start,
    process_folder,
    resolve_end_exclusive,
    write_trimmed_pdf,
)


def _blank_pdf(path: Path, pages: int) -> None:
    doc = fitz.open()
    try:
        for _ in range(pages):
            doc.new_page(width=612, height=792)
        doc.save(str(path))
    finally:
        doc.close()


def test_find_back_matter_references_after_min_fraction():
    # 20 pages: min 0.45 -> scan from index 9; "References" on page index 12
    texts = [f"body page {i}\n" * 5 for i in range(20)]
    texts[12] = "Some intro\n\nReferences\n\n[1] Foo"
    assert find_back_matter_start_page(texts, min_page_fraction=0.45) == 12


def test_no_false_positive_in_intro():
    texts = ["intro\n"] * 30
    texts[5] = "See References for more."  # before min fraction
    texts[20] = "Chapter 3\n\nWe discuss methods.\n"
    assert find_back_matter_start_page(texts, min_page_fraction=0.45) is None


def test_danish_litteratur_heading():
    texts = ["x\n"] * 25
    texts[22] = "Litteratur\n\n[1] Bar"
    assert find_back_matter_start_page(texts, min_page_fraction=0.45) == 22


def test_resolve_end_exclusive_matched():
    assert resolve_end_exclusive(50, 40) == (40, "matched_heading")


def test_resolve_end_exclusive_keep_all():
    assert resolve_end_exclusive(50, None) == (50, "no_match_keep_all")


def test_resolve_end_exclusive_fallback_fraction():
    assert resolve_end_exclusive(
        100,
        None,
        policy=FallbackPolicy.KEEP_FIRST_FRACTION,
        fallback_fraction=0.9,
    ) == (90, "no_match_fallback_fraction")


def test_page_looks_like_back_matter_start_bilag():
    assert page_looks_like_back_matter_start("Bilag A\n\nContent")


def test_process_folder_writes_manifest(tmp_path: Path):
    # Minimal PDF with two pages (blank pages have little text; may trigger notes)
    src = tmp_path / "in" / "a.pdf"
    src.parent.mkdir(parents=True)
    _blank_pdf(src, 2)
    out_dir = tmp_path / "out"
    man = tmp_path / "manifest.csv"

    results = process_folder(
        src.parent,
        out_dir,
        man,
        min_page_fraction=0.45,
    )
    assert len(results) == 1
    assert man.read_text(encoding="utf-8").startswith("filename,")
    assert (out_dir / "a.pdf").is_file()


def test_write_trimmed_pdf_slice(tmp_path: Path):
    src = tmp_path / "three.pdf"
    _blank_pdf(src, 3)
    doc = fitz.open(str(src))
    try:
        dst = tmp_path / "two.pdf"
        write_trimmed_pdf(doc, 2, dst)
    finally:
        doc.close()
    out = fitz.open(str(dst))
    try:
        assert len(out) == 2
    finally:
        out.close()
