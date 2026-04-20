"""Embedding checkpoint metadata (no sentence-transformers)."""

from pathlib import Path

import numpy as np

from thesisdatarepo.analysis.config_loader import load_config
from thesisdatarepo.analysis.embedding import (
    _embedding_fingerprint,
    _try_load_embedding_checkpoint,
    _write_embedding_checkpoint,
    checkpoint_paths,
    clear_embedding_checkpoint,
    embedding_kind,
)
from thesisdatarepo.analysis.paths_util import repo_root


def test_fingerprint_stable_for_same_ids():
    path = repo_root() / "maks" / "analysis_config.example.toml"
    cfg = load_config(path)
    ids = ["a", "b", "c"]
    kind = embedding_kind(cfg)
    fp1 = _embedding_fingerprint(cfg, ids, kind)
    fp2 = _embedding_fingerprint(cfg, ids, kind)
    assert fp1 == fp2


def test_fingerprint_changes_when_doc_order_changes():
    path = repo_root() / "maks" / "analysis_config.example.toml"
    cfg = load_config(path)
    kind = embedding_kind(cfg)
    assert _embedding_fingerprint(cfg, ["a", "b"], kind) != _embedding_fingerprint(
        cfg, ["b", "a"], kind
    )


def test_checkpoint_roundtrip(tmp_path: Path):
    path = repo_root() / "maks" / "analysis_config.example.toml"
    cfg = load_config(path)
    cfg.output_dir = tmp_path
    kind = "fulltext"
    fp = _embedding_fingerprint(cfg, ["x", "y"], kind)
    partial = np.random.randn(1, 8).astype(np.float32)
    _write_embedding_checkpoint(cfg, kind, fp, 2, partial)

    loaded = _try_load_embedding_checkpoint(cfg, kind, fp, 2)
    assert loaded is not None
    arr, start = loaded
    assert start == 1
    np.testing.assert_allclose(arr, partial)

    clear_embedding_checkpoint(cfg, kind)
    npy, js = checkpoint_paths(cfg, kind)
    assert not npy.exists()
    assert not js.exists()
