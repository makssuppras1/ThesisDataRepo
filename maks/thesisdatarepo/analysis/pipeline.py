"""Orchestrate embedding → UMAP → cluster → exports and figures."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

from thesisdatarepo.analysis.clustering import cluster_labels
from thesisdatarepo.analysis.config_loader import AnalysisConfig
from thesisdatarepo.analysis.dim_reduction import (
    run_pca_2d,
    run_tsne_2d,
    run_umap_10d,
    run_umap_2d,
)
from thesisdatarepo.analysis.embedding import (
    cache_path,
    compute_embeddings,
    load_embeddings,
    save_embeddings,
)
from thesisdatarepo.analysis.interdisciplinary import interdisciplinary_table
from thesisdatarepo.analysis.io_data import build_embedding_texts, load_merged_frame
from thesisdatarepo.analysis.keywords_topics import cluster_top_terms
from thesisdatarepo.analysis.plotting import (
    cluster_bar,
    publisher_topic_heatmap,
    scatter_2d,
)

logger = logging.getLogger(__name__)


def run_pipeline(cfg: AnalysisConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    merged = load_merged_frame(cfg)
    df, texts = build_embedding_texts(cfg, merged)

    if cfg.sample_size and cfg.sample_size > 0 and len(df) > cfg.sample_size:
        rng = np.random.default_rng(cfg.sample_seed)
        ix = rng.choice(len(df), size=cfg.sample_size, replace=False)
        df = df.iloc[ix].reset_index(drop=True)
        texts = [texts[int(i)] for i in ix]

    kind = "fulltext" if cfg.embedding_source == "corpus" else "abstract"
    npy_path = cache_path(cfg, kind)

    if cfg.embedding_use_cache and npy_path.is_file():
        logger.info("Loading cached embeddings from %s", npy_path)
        emb = load_embeddings(npy_path)
    else:
        emb = compute_embeddings(cfg, texts, df)
        save_embeddings(npy_path, emb)
        logger.info("Saved embeddings to %s", npy_path)

    emb = normalize(emb, norm="l2", axis=1)

    fig_dir = cfg.output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    logger.info("UMAP 2D / 10D")
    u2 = run_umap_2d(
        emb,
        n_neighbors=cfg.umap_n_neighbors,
        min_dist=cfg.umap_min_dist,
        metric=cfg.umap_metric,
        random_state=cfg.umap_random_state,
    )
    u10 = run_umap_10d(
        emb,
        n_neighbors=cfg.umap_n_neighbors,
        min_dist=cfg.umap_min_dist,
        metric=cfg.umap_metric,
        random_state=cfg.umap_random_state,
    )

    try:
        labels = cluster_labels(
            u10,
            cfg.cluster_method,
            hdbscan_min_cluster_size=cfg.hdbscan_min_cluster_size,
            hdbscan_min_samples=cfg.hdbscan_min_samples,
            kmeans_n_clusters=cfg.kmeans_n_clusters,
            agglomerative_n_clusters=cfg.agglomerative_n_clusters,
        )
    except Exception as e:
        logger.warning("Clustering on UMAP-10 failed (%s); fallback to normalized embeddings", e)
        labels = cluster_labels(
            emb,
            cfg.cluster_method,
            hdbscan_min_cluster_size=cfg.hdbscan_min_cluster_size,
            hdbscan_min_samples=cfg.hdbscan_min_samples,
            kmeans_n_clusters=cfg.kmeans_n_clusters,
            agglomerative_n_clusters=cfg.agglomerative_n_clusters,
        )

    id_col = cfg.columns_id
    pub_col = cfg.columns_publisher
    year_col = cfg.columns_year
    fac_col = cfg.columns_faculty

    out_labels = pd.DataFrame(
        {
            id_col: df[id_col].astype(str),
            "cluster_id": labels,
        }
    )
    if pub_col in df.columns:
        out_labels[pub_col] = df[pub_col]
    if year_col in df.columns:
        out_labels["year"] = pd.to_numeric(df[year_col], errors="coerce")
    if fac_col in df.columns:
        out_labels[fac_col] = df[fac_col]

    labels_path = cfg.output_dir / "clustering_labels.csv"
    out_labels.to_csv(labels_path, index=False)
    logger.info("Wrote %s", labels_path)

    try:
        kw = cluster_top_terms(
            texts,
            labels,
            max_features=cfg.tfidf_max_features,
            min_df=min(cfg.tfidf_min_df, max(1, len(texts) // 5)),
            top_n=cfg.tfidf_top_n,
        )
    except ValueError as e:
        logger.warning("TF-IDF keywords skipped: %s", e)
        kw = pd.DataFrame(columns=["cluster_id", "top_terms", "n_docs"])
    kw_path = cfg.output_dir / "cluster_keywords.csv"
    kw.to_csv(kw_path, index=False)

    if pub_col in df.columns:
        df_plot = df.copy()
        df_plot["cluster_id"] = labels
        inter = interdisciplinary_table(df_plot, "cluster_id", pub_col)
        inter.to_csv(cfg.output_dir / "interdisciplinary.csv", index=False)
        ct = pd.crosstab(df_plot[pub_col].fillna("unknown"), df_plot["cluster_id"])
        ct.to_csv(cfg.output_dir / "publisher_cluster_crosstab.csv")
        publisher_topic_heatmap(ct, fig_dir / "publisher_cluster_heatmap.png")

    ser = pd.Series(labels).value_counts().sort_index()
    cluster_bar(ser, fig_dir / "cluster_sizes.png")

    hue_cluster = pd.Series(labels).astype(str)
    scatter_2d(u2, hue_cluster, "UMAP 2D (clusters)", fig_dir / "umap2d_clusters.png")
    if pub_col in df.columns:
        scatter_2d(u2, df[pub_col].fillna("unknown"), "UMAP 2D (publisher)", fig_dir / "umap2d_publisher.png")
    if fac_col in df.columns:
        scatter_2d(u2, df[fac_col].fillna("unknown"), "UMAP 2D (faculty)", fig_dir / "umap2d_faculty.png")

    del emb
    del u10
    logger.info("Reloading embeddings from disk for t-SNE / PCA")
    emb_disk = load_embeddings(npy_path)
    emb_disk = normalize(emb_disk, norm="l2", axis=1)

    pca2 = run_pca_2d(emb_disk, cfg.tsne_random_state)
    scatter_2d(pca2, hue_cluster, "PCA 2D (clusters)", fig_dir / "pca2d_clusters.png")

    if len(emb_disk) >= 5:
        for perp in cfg.tsne_perplexities:
            max_p = max(2, len(emb_disk) // 4)
            use_p = min(float(perp), float(max_p))
            t2 = run_tsne_2d(emb_disk, perplexity=use_p, random_state=cfg.tsne_random_state)
            scatter_2d(
                t2,
                hue_cluster,
                f"t-SNE 2D (perplexity={use_p})",
                fig_dir / f"tsne2d_p{int(perp)}_clusters.png",
            )
    else:
        logger.warning("Skipping t-SNE (need at least 5 documents)")

    logger.info("Stage-2 pipeline finished. Outputs under %s", cfg.output_dir)
