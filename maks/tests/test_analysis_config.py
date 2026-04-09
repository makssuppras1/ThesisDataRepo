"""Smoke test: analysis config loads (requires no ML run)."""

from thesisdatarepo.analysis.config_loader import load_config
from thesisdatarepo.analysis.paths_util import repo_root


def test_example_toml_loads():
    path = repo_root() / "maks" / "analysis_config.example.toml"
    cfg = load_config(path)
    assert cfg.embedding_model
    assert cfg.corpus_jsonl.name == "corpus_gcs_full.jsonl"
    assert cfg.cluster_method == "hdbscan"
