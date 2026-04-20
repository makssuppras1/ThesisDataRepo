"""Unit tests for GCS URI parsing (no network)."""

from pathlib import Path

import pytest

from thesisdatarepo.gcs_nlp import (
    _load_checkpoints,
    _nlp_result_from_jsonable,
    _nlp_result_to_jsonable,
    parse_gs_uri,
)
from thesisdatarepo.nlp_extract import NlpExtractResult


def test_parse_gs_uri():
    b, o = parse_gs_uri("gs://my-bucket/dtu_findit/master_thesis/foo.pdf")
    assert b == "my-bucket"
    assert o == "dtu_findit/master_thesis/foo.pdf"


def test_parse_gs_uri_rejects_non_gs():
    with pytest.raises(ValueError):
        parse_gs_uri("https://example.com/x.pdf")


def test_nlp_result_json_roundtrip(tmp_path: Path):
    r = NlpExtractResult(
        source="gs://bucket/path/a.pdf",
        output_txt_path=tmp_path / "out.txt",
        pages_total=3,
        page_start_0based=0,
        page_end_exclusive=3,
        abstract_found=True,
        back_matter_found=False,
        end_reason="matched_heading",
        notes="",
        char_count=100,
        skipped_empty=False,
    )
    d = _nlp_result_to_jsonable(r)
    r2 = _nlp_result_from_jsonable(d)
    assert r2 == r


def test_load_checkpoints_last_line_wins(tmp_path: Path):
    p = tmp_path / ".nlp_checkpoints.jsonl"
    r1 = _nlp_result_to_jsonable(
        NlpExtractResult(
            source="gs://b/x",
            output_txt_path=None,
            pages_total=0,
            page_start_0based=0,
            page_end_exclusive=0,
            abstract_found=False,
            back_matter_found=False,
            end_reason="empty_blob",
            notes="",
            char_count=0,
            skipped_empty=False,
        )
    )
    r2 = _nlp_result_to_jsonable(
        NlpExtractResult(
            source="gs://b/x",
            output_txt_path=None,
            pages_total=1,
            page_start_0based=0,
            page_end_exclusive=1,
            abstract_found=False,
            back_matter_found=False,
            end_reason="matched_heading",
            notes="",
            char_count=10,
            skipped_empty=False,
        )
    )
    import json

    p.write_text(
        json.dumps({"blob": "path/x.pdf", "result": r1}, ensure_ascii=False)
        + "\n"
        + json.dumps({"blob": "path/x.pdf", "result": r2}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    m = _load_checkpoints(p)
    assert len(m) == 1
    assert m["path/x.pdf"].char_count == 10
