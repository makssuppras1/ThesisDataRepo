"""Cluster in UMAP-10 space or fallback on normalized embeddings."""

from __future__ import annotations

import logging

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans

logger = logging.getLogger(__name__)


def cluster_labels(
    X: np.ndarray,
    method: str,
    *,
    hdbscan_min_cluster_size: int,
    hdbscan_min_samples: int,
    kmeans_n_clusters: int,
    agglomerative_n_clusters: int,
) -> np.ndarray:
    method = method.lower().strip()
    if method == "hdbscan":
        import hdbscan

        cl = hdbscan.HDBSCAN(
            min_cluster_size=hdbscan_min_cluster_size,
            min_samples=hdbscan_min_samples,
            metric="euclidean",
            cluster_selection_method="leaf",
        )
        return cl.fit_predict(X)

    if method == "kmeans":
        km = KMeans(
            n_clusters=min(kmeans_n_clusters, len(X)),
            random_state=42,
            n_init=10,
        )
        return km.fit_predict(X)

    if method == "agglomerative":
        n = min(agglomerative_n_clusters, max(1, len(X) - 1))
        agg = AgglomerativeClustering(n_clusters=n)
        return agg.fit_predict(X)

    raise ValueError(f"Unknown cluster method: {method}")
