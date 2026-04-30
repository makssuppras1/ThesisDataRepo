"""Drift feature helpers."""

from __future__ import annotations

from thesisdatarepo.drift.features import compute_doc_features, title_body_jaccard


def test_title_body_jaccard_overlap() -> None:
    j = title_body_jaccard("Machine Learning Thesis", "This thesis covers machine learning methods.")
    assert 0 < j <= 1


def test_compute_doc_features_non_empty() -> None:
    f = compute_doc_features("Short Title", "First sentence here. Second sentence there.")
    assert f.n_words >= 4
    assert f.n_sentences >= 1
    assert f.mean_sentence_length_words > 0
