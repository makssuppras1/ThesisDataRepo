"""Unit tests for GCS URI parsing (no network)."""

import pytest

from thesisdatarepo.gcs_nlp import parse_gs_uri


def test_parse_gs_uri():
    b, o = parse_gs_uri("gs://my-bucket/dtu_findit/master_thesis/foo.pdf")
    assert b == "my-bucket"
    assert o == "dtu_findit/master_thesis/foo.pdf"


def test_parse_gs_uri_rejects_non_gs():
    with pytest.raises(ValueError):
        parse_gs_uri("https://example.com/x.pdf")
