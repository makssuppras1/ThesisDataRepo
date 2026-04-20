"""Load ``analysis_config.toml`` and resolve paths relative to repo root."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
import tomllib

from thesisdatarepo.analysis.paths_util import repo_root, resolve_path


@dataclass
class AnalysisConfig:
    corpus_source: str  # jsonl | txt_dir
    corpus_jsonl: Path | None
    metadata_csv: Path
    metadata_sep: str
    output_dir: Path
    nlp_txt_dir: Path | None
    corpus_txt_id_mode: str  # full_stem | member_prefix (txt_dir: id = stem before first '_', matches member_id_ss)
    inner_join_metadata: bool
    faculty_csv: Path | None
    columns_id: str
    columns_corpus_join: str  # e.g. pdf_file → stem matches .txt id; empty → join on columns.id
    columns_abstract: str
    columns_publisher: str
    columns_year: str
    columns_faculty: str
    columns_title: str
    embedding_source: str  # corpus | abstract
    embedding_model: str
    embedding_chunked: bool
    # Chunking: "tokens" = model tokenizer windows (recommended); "chars" = legacy character slices.
    chunk_unit: str
    chunk_max_tokens: int
    chunk_overlap_ratio: float
    chunk_pooling: str  # mean | weighted
    prefer_single_embedding: bool
    chunk_size: int
    chunk_overlap: int
    first_chunk_weight: float
    last_chunk_weight: float
    middle_chunk_weight: float
    min_text_chars: int
    sample_size: int
    sample_seed: int
    embedding_use_cache: bool
    # Save partial embeddings every N documents so a later run can resume (0 = disabled).
    embedding_checkpoint_every: int
    umap_n_neighbors: int
    umap_min_dist: float
    umap_metric: str
    umap_random_state: int
    cluster_method: str  # hdbscan | kmeans | agglomerative
    hdbscan_min_cluster_size: int
    hdbscan_min_samples: int
    kmeans_n_clusters: int
    agglomerative_n_clusters: int
    tfidf_max_features: int
    tfidf_min_df: int
    tfidf_top_n: int
    tsne_perplexities: list[float]
    tsne_random_state: int
    evolution_labels_csv: Path | None
    evolution_min_year: int | None


def _defaults() -> dict:
    return {
        "paths": {
            # jsonl: load ids/text from corpus_jsonl. txt_dir: read *.txt under nlp_txt_dir (id = filename stem).
            "corpus_source": "jsonl",
            "corpus_jsonl": "maks/data/corpus_gcs_full.jsonl",
            "metadata_csv": "maks/data/metadata.csv",
            "metadata_sep": ";",
            "output_dir": "maks/data/analysis_out",
            # If set, corpus rows are kept only when ``nlp_txt_dir / f"{id}.txt"`` exists
            # (same naming as GCS NLP output; cross-check JSONL vs on-disk exports).
            "nlp_txt_dir": "",
            # txt_dir only: full_stem = filename stem (match pdf_file); member_prefix = text before first '_' (match member_id_ss)
            "corpus_txt_id_mode": "full_stem",
            # If true, drop corpus rows with no metadata match (empty ``columns.id``).
            "inner_join_metadata": False,
            "faculty_csv": "",
        },
        "columns": {
            "id": "member_id_ss",
            # Metadata column whose filesystem stem matches NLP .txt names (e.g. pdf_file → stem).
            "corpus_join": "",
            "abstract": "abstract_ts",
            "publisher": "Publisher",
            "year": "year",
            "faculty": "Department_new",
            "title": "",
        },
        "embedding": {
            "source": "corpus",
            "model": "sentence-transformers/all-mpnet-base-v2",
            "chunked": True,
            # Evidence-style full-text: tokenizer windows, 10–20% overlap, mean pool (see embedding.py).
            "chunk_unit": "tokens",
            "chunk_max_tokens": 512,
            "chunk_overlap_ratio": 0.15,
            "chunk_pooling": "mean",
            "prefer_single_embedding": True,
            # Legacy character chunking if chunk_unit = "chars"
            "chunk_size": 1600,
            "chunk_overlap": 200,
            "first_chunk_weight": 2.0,
            "last_chunk_weight": 1.5,
            "middle_chunk_weight": 1.0,
            "min_text_chars": 50,
            "sample_size": 0,
            "sample_seed": 42,
            "use_cache": False,
            # 1 = save after each document (safest). 0 = no resume checkpoints.
            "checkpoint_every": 1,
        },
        "umap": {
            "n_neighbors": 15,
            "min_dist": 0.1,
            "metric": "cosine",
            "random_state": 42,
        },
        "cluster": {
            "method": "hdbscan",
            "hdbscan_min_cluster_size": 15,
            "hdbscan_min_samples": 5,
            "kmeans_n_clusters": 15,
            "agglomerative_n_clusters": 15,
        },
        "tfidf": {
            "max_features": 5000,
            "min_df": 2,
            "top_n": 15,
        },
        "tsne": {
            "perplexities": [30.0, 50.0],
            "random_state": 42,
        },
        "evolution": {
            "labels_csv": "",
        },
    }


def load_config(path: Path) -> AnalysisConfig:
    root = repo_root()
    raw: dict = _defaults()
    with path.open("rb") as f:
        user = tomllib.load(f)
    _deep_merge(raw, user)

    p = raw["paths"]
    ev = raw["evolution"]
    labels_ev = (ev.get("labels_csv") or "").strip()
    fac = (p.get("faculty_csv") or "").strip()
    nlp_txt = (p.get("nlp_txt_dir") or "").strip()
    inner_meta = bool(p.get("inner_join_metadata", False))
    txt_id_mode = str(p.get("corpus_txt_id_mode", "full_stem")).lower().strip()
    cj = (p.get("corpus_jsonl") or "").strip()
    my = ev.get("min_year")
    ev_min_year = int(my) if my is not None and str(my).strip() != "" else None
    corpus_src = str(p.get("corpus_source", "jsonl")).lower().strip()

    return AnalysisConfig(
        corpus_source=corpus_src,
        corpus_jsonl=resolve_path(cj, root) if cj else None,
        metadata_csv=resolve_path(p["metadata_csv"], root),
        metadata_sep=str(p.get("metadata_sep", ";")),
        output_dir=resolve_path(p["output_dir"], root),
        nlp_txt_dir=resolve_path(nlp_txt, root) if nlp_txt else None,
        corpus_txt_id_mode=txt_id_mode,
        inner_join_metadata=inner_meta,
        faculty_csv=resolve_path(fac, root) if fac else None,
        columns_id=raw["columns"]["id"],
        columns_corpus_join=(raw["columns"].get("corpus_join") or "").strip(),
        columns_abstract=raw["columns"]["abstract"],
        columns_publisher=raw["columns"]["publisher"],
        columns_year=raw["columns"]["year"],
        columns_faculty=raw["columns"]["faculty"],
        columns_title=(raw["columns"].get("title") or "").strip(),
        embedding_source=raw["embedding"]["source"],
        embedding_model=raw["embedding"]["model"],
        embedding_chunked=bool(raw["embedding"]["chunked"]),
        chunk_unit=str(raw["embedding"].get("chunk_unit", "tokens")).lower().strip(),
        chunk_max_tokens=int(raw["embedding"].get("chunk_max_tokens", 512)),
        chunk_overlap_ratio=float(raw["embedding"].get("chunk_overlap_ratio", 0.15)),
        chunk_pooling=str(raw["embedding"].get("chunk_pooling", "mean")).lower().strip(),
        prefer_single_embedding=bool(
            raw["embedding"].get("prefer_single_embedding", True)
        ),
        chunk_size=int(raw["embedding"]["chunk_size"]),
        chunk_overlap=int(raw["embedding"]["chunk_overlap"]),
        first_chunk_weight=float(raw["embedding"]["first_chunk_weight"]),
        last_chunk_weight=float(raw["embedding"]["last_chunk_weight"]),
        middle_chunk_weight=float(raw["embedding"]["middle_chunk_weight"]),
        min_text_chars=int(raw["embedding"]["min_text_chars"]),
        sample_size=int(raw["embedding"]["sample_size"]),
        sample_seed=int(raw["embedding"]["sample_seed"]),
        embedding_use_cache=bool(raw["embedding"].get("use_cache", False)),
        embedding_checkpoint_every=int(raw["embedding"].get("checkpoint_every", 1)),
        umap_n_neighbors=int(raw["umap"]["n_neighbors"]),
        umap_min_dist=float(raw["umap"]["min_dist"]),
        umap_metric=str(raw["umap"]["metric"]),
        umap_random_state=int(raw["umap"]["random_state"]),
        cluster_method=raw["cluster"]["method"],
        hdbscan_min_cluster_size=int(raw["cluster"]["hdbscan_min_cluster_size"]),
        hdbscan_min_samples=int(raw["cluster"]["hdbscan_min_samples"]),
        kmeans_n_clusters=int(raw["cluster"]["kmeans_n_clusters"]),
        agglomerative_n_clusters=int(raw["cluster"]["agglomerative_n_clusters"]),
        tfidf_max_features=int(raw["tfidf"]["max_features"]),
        tfidf_min_df=int(raw["tfidf"]["min_df"]),
        tfidf_top_n=int(raw["tfidf"]["top_n"]),
        tsne_perplexities=[float(x) for x in raw["tsne"]["perplexities"]],
        tsne_random_state=int(raw["tsne"]["random_state"]),
        evolution_labels_csv=resolve_path(labels_ev, root) if labels_ev else None,
        evolution_min_year=ev_min_year,
    )


def _deep_merge(base: dict, override: dict) -> None:
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
