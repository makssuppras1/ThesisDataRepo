"""Tests for Abstract→back-matter NLP text extraction."""

from pathlib import Path

from thesisdatarepo.nlp_extract import (
    extract_nlp_page_range,
    find_abstract_start_page,
    normalize_nlp_text,
    page_looks_like_abstract_start,
    process_folder_nlp,
    process_one_pdf_nlp,
)
from pypdf import PdfWriter


def test_find_abstract_first_page():
    texts = [""] * 5
    texts[2] = "Abstract\n\nThis is the abstract text."
    assert find_abstract_start_page(texts) == 2


def test_find_abstract_danish():
    texts = ["Title\n", "Sammenfatning\n\nFoo bar\n"]
    assert find_abstract_start_page(texts) == 1


def test_find_abstract_summary_heading():
    texts = ["Title\n", "Intro\n", "Summary\n\nBody of summary.\n"]
    assert find_abstract_start_page(texts) == 2


def test_page_looks_like_abstract_numbered():
    assert page_looks_like_abstract_start("1 Abstract\n\nBody")


def test_extract_nlp_range_abstract_and_references():
    # 20 pages; abstract page 1; references-like heading at 12 (>= 45% of 20 = 9)
    texts = [f"body {i}\n" * 3 for i in range(20)]
    texts[1] = "Abstract\n\nHello"
    texts[12] = "References\n\n[1] A"
    start, end, af, bf, reason, _ = extract_nlp_page_range(texts)
    assert start == 1
    assert end == 12
    assert af is True
    assert bf is True
    assert reason == "matched_heading"


def test_extract_nlp_no_abstract():
    texts = ["x\n"] * 30
    texts[20] = "References\n\n[1] A"
    start, end, af, bf, reason, notes = extract_nlp_page_range(texts)
    assert start == 0
    assert af is False
    assert "abstract_not_found" in notes
    assert end == 20


def test_normalize_nlp_text():
    assert normalize_nlp_text(["  a  ", "", "b\n"]) == "a\n\nb"


def test_process_folder_nlp_writes_txt_and_manifest(tmp_path: Path):
    w = PdfWriter()
    w.add_blank_page(width=612, height=792)
    src = tmp_path / "in" / "doc.pdf"
    src.parent.mkdir(parents=True)
    with src.open("wb") as f:
        w.write(f)
    out_txt = tmp_path / "txt"
    man = tmp_path / "nlp_manifest.csv"
    jl = tmp_path / "corpus.jsonl"
    r = process_folder_nlp(src.parent, out_txt, man, jsonl_path=jl, skip_if_empty=False)
    assert len(r) == 1
    assert (out_txt / "doc.txt").is_file()
    assert man.read_text(encoding="utf-8").startswith("source,")
    line = jl.read_text(encoding="utf-8").strip()
    assert '"id": "doc"' in line


def test_process_folder_nlp_skips_empty_by_default(tmp_path: Path):
    w = PdfWriter()
    w.add_blank_page(width=612, height=792)
    src = tmp_path / "in" / "blank.pdf"
    src.parent.mkdir(parents=True)
    with src.open("wb") as f:
        w.write(f)
    out_txt = tmp_path / "txt"
    man = tmp_path / "manifest.csv"
    r = process_folder_nlp(src.parent, out_txt, man, skip_if_empty=True)
    assert len(r) == 1
    assert r[0].skipped_empty is True
    assert r[0].char_count == 0
    assert not (out_txt / "blank.txt").is_file()


def test_process_one_pdf_nlp_minimal(tmp_path: Path):
    w = PdfWriter()
    w.add_blank_page(width=612, height=792)
    p = tmp_path / "a.pdf"
    with p.open("wb") as f:
        w.write(f)
    t = tmp_path / "a.txt"
    r = process_one_pdf_nlp(p, t)
    assert r.char_count == 0
    assert t.read_text(encoding="utf-8") == ""
