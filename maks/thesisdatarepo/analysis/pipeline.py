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


def _prepare_df_and_texts(cfg: AnalysisConfig):
    merged = load_merged_frame(cfg)
    df, texts = build_embedding_texts(cfg, merged)
    if cfg.sample_size and cfg.sample_size > 0 and len(df) > cfg.sample_size:
        rng = np.random.default_rng(cfg.sample_seed)
        ix = rng.choice(len(df), size=cfg.sample_size, replace=False)
        df = df.iloc[ix].reset_index(drop=True)
        texts = [texts[int(i)] for i in ix]
    return df, texts


def plot_pca_tsne_figures(
    cfg: AnalysisConfig,
    df: pd.DataFrame,
    emb_disk: np.ndarray,
    labels: np.ndarray,
) -> None:
    """PCA + t-SNE scatter PNGs (clusters, publisher, faculty) and ``tsne*_coords.npy``."""
    pub_col = cfg.columns_publisher
    fac_col = cfg.columns_faculty
    fig_dir = cfg.output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    emb_disk = normalize(np.asarray(emb_disk, dtype=np.float64), norm="l2", axis=1)

    hue_cluster = pd.Series(labels).astype(str)
    pca2 = run_pca_2d(emb_disk, cfg.tsne_random_state)
    scatter_2d(pca2, hue_cluster, "PCA 2D (clusters)", fig_dir / "pca2d_clusters.png")

    n = len(emb_disk)
    if n < 5:
        logger.warning("Skipping t-SNE (need at least 5 documents)")
        return

    for perp in cfg.tsne_perplexities:
        use_p = float(
            min(float(perp), max(2.0, (n - 1) / 3.0), float(n - 1))
        )
        logger.info("t-SNE fit (perplexity=%.1f, n=%s)", use_p, n)
        t2 = run_tsne_2d(emb_disk, perplexity=use_p, random_state=cfg.tsne_random_state)
        tag = f"tsne2d_perp{int(round(use_p))}"
        np.save(fig_dir / f"{tag}_coords.npy", t2)
        scatter_2d(
            t2,
            hue_cluster,
            f"t-SNE 2D (perplexity≈{use_p:.0f}, clusters)",
            fig_dir / f"{tag}_clusters.png",
        )
        if pub_col in df.columns:
            scatter_2d(
                t2,
                df[pub_col].fillna("unknown"),
                f"t-SNE 2D (perplexity≈{use_p:.0f}, publisher)",
                fig_dir / f"{tag}_publisher.png",
            )
        if fac_col in df.columns:
            scatter_2d(
                t2,
                df[fac_col].fillna("unknown"),
                f"t-SNE 2D (perplexity≈{use_p:.0f}, faculty)",
                fig_dir / f"{tag}_faculty.png",
            )


def run_tsne_plots_only(cfg: AnalysisConfig) -> None:
    """
    Regenerate PCA + t-SNE figures from cached ``*_embeddings_*.npy`` and
    ``clustering_labels.csv`` (no re-embedding). Same ``df`` filters as ``run`` must apply.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    df, _ = _prepare_df_and_texts(cfg)
    kind = "fulltext" if cfg.embedding_source == "corpus" else "abstract"
    npy_path = cache_path(cfg, kind)
    if not npy_path.is_file():
        raise FileNotFoundError(
            f"No embedding cache at {npy_path}. Run: thesis-cluster run --config …"
        )
    emb = load_embeddings(npy_path)
    if len(emb) != len(df):
        raise ValueError(
            f"Cache has {len(emb)} rows but filtered dataframe has {len(df)}. "
            "Use the same config (paths, filters, sample_size) as the run that wrote the cache."
        )
    lab_path = cfg.output_dir / "clustering_labels.csv"
    if not lab_path.is_file():
        raise FileNotFoundError(
            f"Missing {lab_path}. Run the full pipeline once: thesis-cluster run --config …"
        )
    lab = pd.read_csv(lab_path)
    id_col = cfg.columns_id
    if id_col not in lab.columns or "cluster_id" not in lab.columns:
        raise KeyError(f"{lab_path} must contain {id_col!r} and cluster_id")
    m = df.merge(lab[[id_col, "cluster_id"]], on=id_col, how="left")
    labels = pd.Series(m["cluster_id"]).fillna(-1).astype(int).to_numpy()
    if (m["cluster_id"].isna()).any():
        logger.warning(
            "cluster_id missing for %s ids (showing as -1 in plots)",
            int(m["cluster_id"].isna().sum()),
        )
    plot_pca_tsne_figures(cfg, df, emb, labels)
    logger.info("t-SNE / PCA figures written under %s", cfg.output_dir / "figures")


def run_pipeline(cfg: AnalysisConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    df, texts = _prepare_df_and_texts(cfg)

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
    plot_pca_tsne_figures(cfg, df, emb_disk, labels)

    logger.info("Stage-2 pipeline finished. Outputs under %s", cfg.output_dir)
